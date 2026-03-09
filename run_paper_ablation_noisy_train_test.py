from __future__ import annotations

import argparse
import copy
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from openpyxl import Workbook

from run_paper_ablation import (
    PAPER_CASES,
    REPORT_METRIC_KEYS,
    _case_result_prefixes,
    _list_case_result_files,
    _move_file_safe,
    _resolve_run_id,
    build_case_configs,
)
from test_3D import test
from train_multitask import train
from utils.noise import build_noise_label


DEFAULT_TRAIN_SNRS = (0.0, 5.0, 15.0, 25.0, 35.0)
DEFAULT_TEST_SNRS = (0.0, 5.0, 15.0, 25.0, 35.0)
DEFAULT_TRAIN_NOISE_PROB = 1.0
DEFAULT_NOISE_SEED = 2026
DELTA_KEYS = ("r2", "pcc", "ssim", "mse", "mae", "medae", "psnr")
REPEAT_SUMMARY_KEYS = ("noise_snr_db_measured",) + DELTA_KEYS


def _format_tag_number(value: float | int) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return str(numeric).replace(".", "p").replace("-", "m")


def _collect_headers(rows: list[dict]) -> list[str]:
    headers: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in headers:
                headers.append(key)
    return headers


def _save_rows_excel(xlsx_path: Path, rows: list[dict], sheet_name: str) -> None:
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name

    if not rows:
        ws.append(["info"])
        ws.append(["no rows"])
        wb.save(xlsx_path)
        return

    headers = _collect_headers(rows)
    ws.append(headers)
    for row in rows:
        ws.append([row.get(h, "") for h in headers])
    wb.save(xlsx_path)


def _save_rows_csv(csv_path: Path, rows: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["info"])
            writer.writerow(["no rows"])
        return

    headers = _collect_headers(rows)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h, "") for h in headers})


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _to_numeric(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return None


def _make_noise_cases(snrs: Iterable[float], seed: int) -> list[dict[str, Any]]:
    noise_cases: list[dict[str, Any]] = [
        {
            "noise_case": "clean",
            "test_noise_kind": "none",
            "test_noise_snr_db": None,
            "test_noise_seed": int(seed),
            "test_noise_save_inputs": True,
        }
    ]
    for snr in snrs:
        noise_cases.append(
            {
                "noise_case": build_noise_label("awgn", snr_db=snr, seed=seed),
                "test_noise_kind": "awgn",
                "test_noise_snr_db": float(snr),
                "test_noise_seed": int(seed),
                "test_noise_save_inputs": True,
            }
        )
    return noise_cases


def _augment_run_id_suffix(
    original_suffix: str,
    train_snrs: Iterable[float],
    train_noise_prob: float,
) -> str:
    snr_tag = "_".join(_format_tag_number(snr) for snr in train_snrs)
    prob_tag = _format_tag_number(float(train_noise_prob) * 100.0)
    new_tail = f"_noisytrain_p{prob_tag}_s{snr_tag}"
    suffix = str(original_suffix or "")
    if suffix.endswith(new_tail):
        return suffix
    return f"{suffix}{new_tail}"


def build_noisy_train_case_configs(
    case_name: str,
    epochs_override: int | None,
    train_snrs: Iterable[float],
    train_noise_prob: float,
) -> tuple[dict, dict]:
    train_cfg, test_cfg = build_case_configs(case_name=case_name, epochs_override=epochs_override)
    train_snrs_tuple = tuple(float(snr) for snr in train_snrs)
    noisy_suffix = _augment_run_id_suffix(
        original_suffix=str(train_cfg.get("run_id_suffix", "") or ""),
        train_snrs=train_snrs_tuple,
        train_noise_prob=float(train_noise_prob),
    )

    train_cfg["run_id_suffix"] = noisy_suffix
    train_cfg["train_noise_kind"] = "awgn"
    train_cfg["train_noise_prob"] = float(train_noise_prob)
    train_cfg["train_noise_snr_db_choices"] = train_snrs_tuple

    test_cfg["run_id_suffix"] = noisy_suffix
    test_cfg["test_noise_kind"] = "none"
    test_cfg["test_noise_snr_db"] = None
    test_cfg["test_noise_save_inputs"] = True
    return train_cfg, test_cfg


def _required_paths(train_cfg: dict, test_cfg: dict) -> list[Path]:
    run_id = _resolve_run_id(train_cfg)
    data_flag = str(train_cfg["data_flag"])
    inverse_ckpt = Path("save_train_model") / f"{run_id}_s_uns_{data_flag}.pth"
    full_ckpt = Path("save_train_model") / f"{run_id}_full_ckpt_{data_flag}.pth"
    dedicated_stats = Path("save_train_model") / f"norm_stats_{run_id}_{data_flag}.npy"
    legacy_stats = Path("save_train_model") / f"norm_stats_{data_flag}.npy"

    paths = [inverse_ckpt]
    requires_full_ckpt = bool(test_cfg.get("use_aniso_conditioning", False)) and (
        str(test_cfg.get("aniso_test_protocol", "closed_loop")).strip().lower() == "closed_loop"
    )
    if requires_full_ckpt:
        paths.append(full_ckpt)
    elif not (full_ckpt.exists() or dedicated_stats.exists() or legacy_stats.exists()):
        paths.append(dedicated_stats)
    return paths


def _preflight_cases(
    cases: list[str],
    train_snrs: Iterable[float],
    train_noise_prob: float,
) -> dict[str, tuple[dict, dict]]:
    case_cfgs: dict[str, tuple[dict, dict]] = {}
    missing_lines: list[str] = []

    for case in cases:
        train_cfg, test_cfg = build_noisy_train_case_configs(
            case_name=case,
            epochs_override=None,
            train_snrs=train_snrs,
            train_noise_prob=train_noise_prob,
        )
        case_cfgs[case] = (train_cfg, test_cfg)
        missing = [str(path) for path in _required_paths(train_cfg, test_cfg) if not path.exists()]
        if missing:
            missing_lines.append(f"{case}:")
            missing_lines.extend([f"  - {item}" for item in missing])

    if missing_lines:
        raise FileNotFoundError(
            "Missing required checkpoints/stats for noisy-train paper ablation evaluation:\n"
            + "\n".join(missing_lines)
        )
    return case_cfgs


def _archive_existing_artifacts(results_root: Path, prefixes: tuple[str, str], run_dir: Path) -> None:
    old_files = _list_case_result_files(results_root, prefixes)
    if not old_files:
        return
    archive_dir = run_dir / "_preexisting"
    for path in old_files:
        _move_file_safe(path, archive_dir)
    print(f"[ARCHIVE] moved {len(old_files)} preexisting files -> {archive_dir.as_posix()}")


def _build_detail_row(
    case_dir: str,
    case: str,
    noise_case: str,
    repeat_idx: int,
    train_cfg: dict,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_dir": case_dir,
        "case": case,
        "noise_case": noise_case,
        "repeat_idx": int(repeat_idx),
        "train_noise_kind": train_cfg.get("train_noise_kind", ""),
        "train_noise_prob": train_cfg.get("train_noise_prob", ""),
        "train_noise_snr_db_choices": str(train_cfg.get("train_noise_snr_db_choices", ())),
        "noise_kind": metrics.get("noise_kind", ""),
        "noise_snr_db_target": metrics.get("noise_snr_db_target", ""),
        "noise_snr_db_measured": metrics.get("noise_snr_db_measured", ""),
        "noise_seed": metrics.get("noise_seed", ""),
    }
    for key in REPORT_METRIC_KEYS:
        row[key] = metrics.get(key, "")
    return row


def _build_delta_rows(detail_rows: list[dict]) -> list[dict]:
    baseline_by_case_repeat: dict[tuple[str, int], dict] = {}
    for row in detail_rows:
        if str(row.get("noise_case", "")) == "clean":
            baseline_by_case_repeat[(str(row["case"]), int(row["repeat_idx"]))] = row

    delta_rows: list[dict] = []
    for row in detail_rows:
        key = (str(row["case"]), int(row["repeat_idx"]))
        baseline = baseline_by_case_repeat.get(key)
        if baseline is None:
            raise RuntimeError(f"Missing clean baseline for case={row['case']} repeat={row['repeat_idx']}")

        delta_row: dict[str, Any] = {
            "case_dir": row["case_dir"],
            "case": row["case"],
            "noise_case": row["noise_case"],
            "repeat_idx": row["repeat_idx"],
            "noise_kind": row.get("noise_kind", ""),
            "noise_snr_db_target": row.get("noise_snr_db_target", ""),
            "noise_snr_db_measured": row.get("noise_snr_db_measured", ""),
        }
        for key_name in DELTA_KEYS:
            current = _to_numeric(row.get(key_name))
            base = _to_numeric(baseline.get(key_name))
            delta_row[f"delta_{key_name}"] = None if (current is None or base is None) else float(current - base)
        delta_rows.append(delta_row)
    return delta_rows


def _build_repeatability_rows(detail_rows: list[dict], repeats: int) -> list[dict]:
    if repeats <= 1:
        return []

    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in detail_rows:
        key = (str(row["case"]), str(row["noise_case"]))
        grouped.setdefault(key, []).append(row)

    rows: list[dict] = []
    for (case, noise_case), group in sorted(grouped.items()):
        case_dir = str(group[0].get("case_dir", ""))
        row: dict[str, Any] = {
            "case_dir": case_dir,
            "case": case,
            "noise_case": noise_case,
            "repeat_count": len(group),
            "noise_kind": group[0].get("noise_kind", ""),
            "noise_snr_db_target": group[0].get("noise_snr_db_target", ""),
            "noise_seed": group[0].get("noise_seed", ""),
        }
        is_deterministic = True
        for key_name in REPEAT_SUMMARY_KEYS:
            values = [_to_numeric(item.get(key_name)) for item in group]
            values = [value for value in values if value is not None]
            if not values:
                continue
            values_arr = np.asarray(values, dtype=np.float64)
            row[f"{key_name}_mean"] = float(values_arr.mean())
            row[f"{key_name}_std"] = float(values_arr.std())
            row[f"{key_name}_max_delta"] = float(values_arr.max() - values_arr.min())
            if values_arr.max() - values_arr.min() > 1e-12:
                is_deterministic = False
        row["is_deterministic"] = bool(is_deterministic)
        rows.append(row)
    return rows


def run_paper_ablation_noisy_train_test(
    cases: list[str],
    train_snrs: list[float],
    test_snrs: list[float],
    train_noise_prob: float,
    repeats: int,
    seed: int,
    mode: str,
    epochs_override: int | None,
) -> None:
    results_root = Path("results")
    results_root.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_root = results_root / f"paper_ablation_noisy_train_test_{run_stamp}"
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[RUN] noisy-train paper ablation results root: {run_root.as_posix()}")

    if mode == "test":
        case_cfgs = _preflight_cases(cases, train_snrs=train_snrs, train_noise_prob=train_noise_prob)
    else:
        case_cfgs = {
            case: build_noisy_train_case_configs(
                case_name=case,
                epochs_override=epochs_override,
                train_snrs=train_snrs,
                train_noise_prob=train_noise_prob,
            )
            for case in cases
        }

    noise_cases = _make_noise_cases(snrs=test_snrs, seed=seed)
    detail_rows: list[dict] = []

    for case_idx, case in enumerate(cases, start=1):
        train_cfg, base_test_cfg = case_cfgs[case]
        case_dir_name = f"{case_idx:02d}_{case}"
        case_root = run_root / case_dir_name
        case_root.mkdir(parents=True, exist_ok=True)
        prefixes = _case_result_prefixes(train_cfg)

        with open(case_root / "train_config.json", "w", encoding="utf-8") as f:
            json.dump(_jsonable(train_cfg), f, indent=2, ensure_ascii=True)
        with open(case_root / "base_test_config.json", "w", encoding="utf-8") as f:
            json.dump(_jsonable(base_test_cfg), f, indent=2, ensure_ascii=True)

        print(f"\n[{case_idx}/{len(cases)}] case={case}")

        if mode in ("train", "both"):
            train_dir = case_root / "train_artifacts"
            train_dir.mkdir(parents=True, exist_ok=True)
            _archive_existing_artifacts(results_root, prefixes, train_dir)
            print(
                f"[RUN] train case={case} noise_kind={train_cfg.get('train_noise_kind', 'none')} "
                f"prob={float(train_cfg.get('train_noise_prob', 0.0)):.3f} "
                f"snrs={train_cfg.get('train_noise_snr_db_choices', ())}"
            )
            train(train_cfg)
            new_train_files = _list_case_result_files(results_root, prefixes)
            if new_train_files:
                for path in new_train_files:
                    _move_file_safe(path, train_dir)
                print(f"[SAVE] training artifacts saved to: {train_dir.as_posix()}")
            else:
                print(f"[SAVE][WARN] no training artifacts found for prefixes={prefixes}")

        if mode in ("test", "both"):
            for noise_cfg in noise_cases:
                noise_case = str(noise_cfg["noise_case"])
                for repeat_idx in range(int(repeats)):
                    run_dir = case_root / noise_case / f"run_{repeat_idx}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    _archive_existing_artifacts(results_root, prefixes, run_dir)

                    test_cfg = copy.deepcopy(base_test_cfg)
                    test_cfg.update(noise_cfg)
                    with open(run_dir / "test_config.json", "w", encoding="utf-8") as f:
                        json.dump(_jsonable(test_cfg), f, indent=2, ensure_ascii=True)

                    print(
                        f"[RUN] test case={case} noise_case={noise_case} repeat={repeat_idx} "
                        f"noise_kind={test_cfg['test_noise_kind']}"
                    )
                    metrics = test(test_cfg)
                    if not isinstance(metrics, dict):
                        raise RuntimeError(
                            f"test() did not return metrics dict for case={case}, noise_case={noise_case}"
                        )

                    row = _build_detail_row(
                        case_dir=case_dir_name,
                        case=case,
                        noise_case=str(metrics.get("noise_case", noise_case)),
                        repeat_idx=repeat_idx,
                        train_cfg=train_cfg,
                        metrics=metrics,
                    )
                    detail_rows.append(row)

                    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
                        json.dump(_jsonable(row), f, indent=2, ensure_ascii=True)
                    _save_rows_excel(run_dir / "test_metrics.xlsx", [row], sheet_name="test_metrics")

                    new_test_files = _list_case_result_files(results_root, prefixes)
                    if new_test_files:
                        for path in new_test_files:
                            _move_file_safe(path, run_dir)
                        print(f"[SAVE] case artifacts saved to: {run_dir.as_posix()}")
                    else:
                        print(f"[SAVE][WARN] no case artifacts found for prefixes={prefixes}")

    if not detail_rows:
        print("[SAVE][WARN] no test metrics collected; summary files not generated.")
        return

    detail_csv = run_root / "paper_ablation_noisy_train_test_metrics_summary.csv"
    detail_xlsx = run_root / "paper_ablation_noisy_train_test_metrics_summary.xlsx"
    _save_rows_csv(detail_csv, detail_rows)
    _save_rows_excel(detail_xlsx, detail_rows, sheet_name="summary")
    print(f"[SAVE] detail summary -> {detail_csv.as_posix()}")
    print(f"[SAVE] detail summary -> {detail_xlsx.as_posix()}")

    delta_rows = _build_delta_rows(detail_rows)
    delta_csv = run_root / "paper_ablation_noisy_train_test_delta_summary.csv"
    delta_xlsx = run_root / "paper_ablation_noisy_train_test_delta_summary.xlsx"
    _save_rows_csv(delta_csv, delta_rows)
    _save_rows_excel(delta_xlsx, delta_rows, sheet_name="delta_summary")
    print(f"[SAVE] delta summary -> {delta_csv.as_posix()}")
    print(f"[SAVE] delta summary -> {delta_xlsx.as_posix()}")

    repeat_rows = _build_repeatability_rows(detail_rows, repeats=int(repeats))
    if repeat_rows:
        repeat_csv = run_root / "paper_ablation_noisy_train_test_repeatability_summary.csv"
        repeat_xlsx = run_root / "paper_ablation_noisy_train_test_repeatability_summary.xlsx"
        _save_rows_csv(repeat_csv, repeat_rows)
        _save_rows_excel(repeat_xlsx, repeat_rows, sheet_name="repeatability")
        print(f"[SAVE] repeatability summary -> {repeat_csv.as_posix()}")
        print(f"[SAVE] repeatability summary -> {repeat_xlsx.as_posix()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the five paper ablation cases with AWGN augmentation, then evaluate them under a clean + multi-SNR test matrix."
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=list(PAPER_CASES.keys()),
        choices=sorted(PAPER_CASES.keys()),
        help="Paper ablation cases to run.",
    )
    parser.add_argument(
        "--mode",
        default="both",
        choices=["train", "test", "both"],
        help="Run training only, testing only, or both.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for training epochs.",
    )
    parser.add_argument(
        "--train-snrs",
        nargs="+",
        type=float,
        default=list(DEFAULT_TRAIN_SNRS),
        help="AWGN SNR levels used during training augmentation (default: 0 5 15 25 35).",
    )
    parser.add_argument(
        "--train-noise-prob",
        type=float,
        default=DEFAULT_TRAIN_NOISE_PROB,
        help="Probability of applying AWGN augmentation to each training batch (default: 1.0).",
    )
    parser.add_argument(
        "--test-snrs",
        nargs="+",
        type=float,
        default=list(DEFAULT_TEST_SNRS),
        help="AWGN SNR levels used during testing (default: 0 5 15 25 35).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeat count per case/noise setting during testing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_NOISE_SEED,
        help="Fixed AWGN seed for testing (default: 2026).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_paper_ablation_noisy_train_test(
        cases=args.cases,
        train_snrs=[float(snr) for snr in args.train_snrs],
        test_snrs=[float(snr) for snr in args.test_snrs],
        train_noise_prob=float(args.train_noise_prob),
        repeats=int(args.repeats),
        seed=int(args.seed),
        mode=str(args.mode),
        epochs_override=args.epochs,
    )
