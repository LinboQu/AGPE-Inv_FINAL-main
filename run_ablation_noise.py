import argparse
import copy
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from openpyxl import Workbook

from run_ablation import (
    CASE_PRESETS,
    _case_result_prefixes,
    _list_case_result_files,
    _move_file_safe,
    _resolve_run_id,
    build_case_configs,
)
from test_3D import test
from utils.noise import build_noise_label


DEFAULT_SNRS = (30.0, 20.0, 10.0, 5.0)
DEFAULT_NOISE_SEED = 2026
DELTA_KEYS = (
    "r2",
    "ssim",
    "psnr",
    "shadow_area_mean",
    "shadow_area_max",
    "shadow_area_final_minus_bootstrap",
)
REPEAT_SUMMARY_KEYS = (
    "noise_snr_db_measured",
    "r2",
    "pcc",
    "ssim",
    "mse",
    "mae",
    "medae",
    "psnr",
    "shadow_area_d40",
    "shadow_area_d100",
    "shadow_area_d160",
    "shadow_area_mean",
    "shadow_area_max",
    "bootstrap_r2",
    "final_r2",
    "r2_final_minus_bootstrap",
    "bootstrap_shadow_area_mean",
    "final_shadow_area_mean",
    "shadow_area_final_minus_bootstrap",
)


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


def _to_numeric(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return None


def _make_noise_cases(snrs: Iterable[float], seed: int) -> list[dict]:
    noise_cases = [
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


def _preflight_cases(cases: list[str]) -> dict[str, tuple[dict, dict]]:
    case_cfgs: dict[str, tuple[dict, dict]] = {}
    missing_lines: list[str] = []

    for case in cases:
        train_cfg, test_cfg = build_case_configs(case, epochs_override=None)
        case_cfgs[case] = (train_cfg, test_cfg)

        missing = [str(path) for path in _required_paths(train_cfg, test_cfg) if not path.exists()]
        if missing:
            missing_lines.append(f"{case}:")
            missing_lines.extend([f"  - {item}" for item in missing])

    if missing_lines:
        raise FileNotFoundError(
            "Missing required checkpoints/stats for ablation noise evaluation:\n"
            + "\n".join(missing_lines)
        )
    return case_cfgs


def _build_case_metrics_row(
    case: str,
    train_cfg: dict,
    test_cfg: dict,
    metrics: dict,
    noise_case: str,
    repeat_idx: int,
    artifact_dir: Path,
) -> dict:
    return {
        "case": case,
        "artifact_dir": artifact_dir.as_posix(),
        "noise_case": noise_case,
        "repeat_idx": int(repeat_idx),
        "backend": str(train_cfg.get("aniso_backend", "grid")),
        "use_tensor_strength": bool(train_cfg.get("aniso_use_tensor_strength", False)),
        "aniso_train_protocol": str(train_cfg.get("aniso_train_protocol", "closed_loop")),
        "aniso_test_protocol": str(test_cfg.get("aniso_test_protocol", "closed_loop")),
        "use_aniso_conditioning": bool(train_cfg.get("use_aniso_conditioning", False)),
        "aniso_kappa": float(train_cfg.get("aniso_kappa", 4.0)),
        "lambda_recon": float(train_cfg.get("lambda_recon", 0.0)),
        "lambda_facies": float(train_cfg.get("lambda_facies", 0.0)),
        "agpe_long_edges": bool(train_cfg.get("agpe_long_edges", True)),
        "agpe_refine_graph": bool(train_cfg.get("agpe_refine_graph", True)),
        "agpe_cache_graph": bool(train_cfg.get("agpe_cache_graph", True)),
        "agpe_rebuild_every": int(train_cfg.get("agpe_rebuild_every", 50)),
        "agpe_skel_p_thresh": float(train_cfg.get("agpe_skel_p_thresh", 0.55)),
        "agpe_lift_sigma": float(train_cfg.get("agpe_lift_sigma", 2.2)),
        "agpe_long_weight": float(train_cfg.get("agpe_long_weight", 0.35)),
        "agpe_well_seed_mode": str(train_cfg.get("agpe_well_seed_mode", "depth_gate")),
        "agpe_well_seed_min": float(train_cfg.get("agpe_well_seed_min", 0.0)),
        "agpe_well_soft_alpha": float(train_cfg.get("agpe_well_soft_alpha", 1.0)),
        "use_detail_branch": bool(train_cfg.get("use_detail_branch", False)),
        "detail_gain": float(train_cfg.get("detail_gain", 0.0)),
        "detail_hp_kernel": int(train_cfg.get("detail_hp_kernel", 0)),
        "use_boundary_weight": bool(train_cfg.get("use_boundary_weight", False)),
        "lambda_depth_grad": float(train_cfg.get("lambda_depth_grad", 0.0)),
        "lambda_depth_hf": float(train_cfg.get("lambda_depth_hf", 0.0)),
        "iterative_R": bool(train_cfg.get("iterative_R", False)),
        **metrics,
    }


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

        delta_row = {
            "case": row["case"],
            "noise_case": row["noise_case"],
            "repeat_idx": row["repeat_idx"],
            "noise_kind": row.get("noise_kind", None),
            "noise_snr_db_target": row.get("noise_snr_db_target", None),
            "noise_snr_db_measured": row.get("noise_snr_db_measured", None),
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
        row = {
            "case": case,
            "noise_case": noise_case,
            "repeat_count": len(group),
            "noise_kind": group[0].get("noise_kind", None),
            "noise_snr_db_target": group[0].get("noise_snr_db_target", None),
            "noise_seed": group[0].get("noise_seed", None),
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


def _archive_existing_artifacts(results_root: Path, prefixes: tuple[str, str], run_dir: Path) -> None:
    old_files = _list_case_result_files(results_root, prefixes)
    if not old_files:
        return
    archive_dir = run_dir / "_preexisting"
    for path in old_files:
        _move_file_safe(path, archive_dir)
    print(f"[ARCHIVE] moved {len(old_files)} preexisting files -> {archive_dir.as_posix()}")


def run_ablation_noise(cases: list[str], snrs: list[float], repeats: int, seed: int) -> None:
    results_root = Path("results")
    results_root.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_root = results_root / f"ablation_noise_{run_stamp}"
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[RUN] ablation noise results root: {run_root.as_posix()}")

    case_cfgs = _preflight_cases(cases)
    noise_cases = _make_noise_cases(snrs=snrs, seed=seed)
    detail_rows: list[dict] = []

    for case_idx, case in enumerate(cases, start=1):
        train_cfg, base_test_cfg = case_cfgs[case]
        prefixes = _case_result_prefixes(train_cfg)
        print(f"\n[{case_idx}/{len(cases)}] case={case}")

        for noise_cfg in noise_cases:
            noise_case = str(noise_cfg["noise_case"])
            for repeat_idx in range(int(repeats)):
                run_dir = run_root / case / noise_case / f"run_{repeat_idx}"
                run_dir.mkdir(parents=True, exist_ok=True)
                _archive_existing_artifacts(results_root, prefixes, run_dir)

                test_cfg = copy.deepcopy(base_test_cfg)
                test_cfg.update(noise_cfg)

                print(
                    f"[RUN] test case={case} noise_case={noise_case} repeat={repeat_idx} "
                    f"noise_kind={test_cfg['test_noise_kind']}"
                )
                metrics = test(test_cfg)
                if not isinstance(metrics, dict):
                    raise RuntimeError(f"test() did not return metrics dict for case={case}, noise_case={noise_case}")

                case_metrics = _build_case_metrics_row(
                    case=case,
                    train_cfg=train_cfg,
                    test_cfg=test_cfg,
                    metrics=metrics,
                    noise_case=noise_case,
                    repeat_idx=repeat_idx,
                    artifact_dir=run_dir,
                )
                detail_rows.append(case_metrics)

                with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
                    json.dump(case_metrics, f, indent=2, ensure_ascii=True)
                _save_rows_excel(run_dir / "test_metrics.xlsx", [case_metrics], sheet_name="test_metrics")

                new_files = _list_case_result_files(results_root, prefixes)
                if new_files:
                    for path in new_files:
                        _move_file_safe(path, run_dir)
                    print(f"[SAVE] case artifacts saved to: {run_dir.as_posix()}")
                else:
                    print(f"[SAVE][WARN] no case artifacts found for prefixes={prefixes}")

    detail_csv = run_root / "ablation_noise_metrics_summary.csv"
    detail_xlsx = run_root / "ablation_noise_metrics_summary.xlsx"
    _save_rows_csv(detail_csv, detail_rows)
    _save_rows_excel(detail_xlsx, detail_rows, sheet_name="summary")
    print(f"[SAVE] detail summary -> {detail_csv.as_posix()}")
    print(f"[SAVE] detail summary -> {detail_xlsx.as_posix()}")

    delta_rows = _build_delta_rows(detail_rows)
    delta_csv = run_root / "ablation_noise_delta_summary.csv"
    delta_xlsx = run_root / "ablation_noise_delta_summary.xlsx"
    _save_rows_csv(delta_csv, delta_rows)
    _save_rows_excel(delta_xlsx, delta_rows, sheet_name="delta_summary")
    print(f"[SAVE] delta summary -> {delta_csv.as_posix()}")
    print(f"[SAVE] delta summary -> {delta_xlsx.as_posix()}")

    repeat_rows = _build_repeatability_rows(detail_rows, repeats=int(repeats))
    if repeat_rows:
        repeat_csv = run_root / "ablation_noise_repeatability_summary.csv"
        repeat_xlsx = run_root / "ablation_noise_repeatability_summary.xlsx"
        _save_rows_csv(repeat_csv, repeat_rows)
        _save_rows_excel(repeat_xlsx, repeat_rows, sheet_name="repeatability")
        print(f"[SAVE] repeatability summary -> {repeat_csv.as_posix()}")
        print(f"[SAVE] repeatability summary -> {repeat_xlsx.as_posix()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AWGN noise evaluation for all or selected ablation cases."
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=sorted(CASE_PRESETS.keys()),
        choices=sorted(CASE_PRESETS.keys()),
        help="Ablation cases to evaluate (default: all cases).",
    )
    parser.add_argument(
        "--snrs",
        nargs="+",
        type=float,
        default=list(DEFAULT_SNRS),
        help="AWGN SNR levels in dB (default: 30 20 10 5).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeat count per case/noise setting. Use 3 to check non-determinism.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_NOISE_SEED,
        help="Fixed AWGN seed (default: 2026).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_ablation_noise(
        cases=args.cases,
        snrs=[float(snr) for snr in args.snrs],
        repeats=int(args.repeats),
        seed=int(args.seed),
    )
