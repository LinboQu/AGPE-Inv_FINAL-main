from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from openpyxl import Workbook

from setting import (
    TEST_EXPERT_OVERRIDES,
    TEST_PROFILE,
    TEST_USER_P,
    TRAIN_EXPERT_OVERRIDES,
    TRAIN_PROFILE,
    TRAIN_USER_P,
)
from test_3D import test
from train_multitask import train
from utils.config_resolver import build_test_config, build_train_config


# Paper ablation runner with a fixed five-case chain.
PAPER_ABLATION_COMMON: Dict[str, object] = {
    "use_aniso_conditioning": True,
    "aniso_train_protocol": "closed_loop",
    "aniso_test_protocol": "closed_loop",
    "aniso_backend": "skeleton_graph",
    "aniso_use_tensor_strength": True,
    "aniso_kappa": 4.0,
    "agpe_long_edges": True,
    "iterative_R": True,
    "agpe_cache_graph": True,
    "agpe_refine_graph": True,
    "agpe_rebuild_every": 100,
    "R_update_every": 50,
    "aniso_closed_loop_conf_thresh": 0.60,
    "agpe_lift_sigma": 2.3,
    "agpe_long_weight": 0.35,
    "agpe_well_soft_alpha": 0.20,
    "agpe_well_seed_mode": "depth_gate",
    "agpe_well_seed_min": 0.02,
    "use_boundary_weight": False,
    "lambda_depth_grad": 0.005,
    "lambda_depth_hf": 0.001,
    "use_depth_warm_schedule": True,
    "depth_warm_start_epoch": 500,
    "depth_warm_ramp_epochs": 300,
    "use_boundary_warm_schedule": False,
    "ws_every": 4,
    "ws_max_batches": 80,
    "ws_max_batches_stageA": 30,
    "ws_every_late": 12,
    "ws_max_batches_late": 8,
    "use_detail_branch": True,
    "detail_gain": 0.15,
    "lambda_amp_anchor": 0.05,
    "lambda_recon": 1.0,
    "lambda_facies": 0.2,
    "train_noise_kind": "none",
    "train_noise_prob": 0.0,
    "train_noise_snr_db_choices": (),
    "r_channel_dropout_prob": 0.0,
}

PAPER_TEST_COMMON: Dict[str, object] = {
    "test_noise_kind": "none",
    "test_noise_snr_db": None,
    "test_noise_save_inputs": True,
}


def _paper_case(**overrides: object) -> Dict[str, object]:
    out = copy.deepcopy(PAPER_ABLATION_COMMON)
    out.update(overrides)
    return out


PAPER_CASES: Dict[str, Dict[str, object]] = {
    "baseline_1d": _paper_case(
        use_aniso_conditioning=False,
        iterative_R=False,
        lambda_recon=0.0,
        lambda_facies=0.0,
        stageA_lambda_recon_mult=0.0,
        stageA_lambda_facies_mult=0.0,
        use_detail_branch=False,
        use_boundary_weight=False,
        lambda_depth_grad=0.0,
        lambda_depth_hf=0.0,
        lambda_amp_anchor=0.0,
        run_id_suffix="_fair_baseline_1d",
    ),
    "physics_only": _paper_case(
        use_aniso_conditioning=False,
        iterative_R=False,
        lambda_recon=1.0,
        lambda_facies=0.0,
        stageA_lambda_facies_mult=0.0,
        use_detail_branch=False,
        use_boundary_weight=False,
        lambda_depth_grad=0.0,
        lambda_depth_hf=0.0,
        lambda_amp_anchor=0.0,
        run_id_suffix="_fair_physics_only",
    ),
    "farp_isotropic": _paper_case(
        aniso_backend="grid",
        aniso_use_tensor_strength=False,
        aniso_kappa=0.0,
        iterative_R=True,
        run_id_suffix="_fair_farp_isotropic",
    ),
    "farp_noniter": _paper_case(
        iterative_R=False,
        run_id_suffix="_fair_farp_noniter",
    ),
    "proposed": _paper_case(
        agpe_rebuild_every=50,
        use_boundary_weight=False,
        lambda_depth_grad=0.0,
        lambda_depth_hf=0.0,
        train_noise_kind="awgn",
        train_noise_prob=0.5,
        train_noise_snr_db_choices=(30.0, 20.0),
        r_channel_dropout_prob=0.3,
        run_id_suffix="_skel_ref_nobddep_awgntrain_rdrop",
    ),
}


PAPER_CASE_NOTES: Dict[str, str] = {
    "baseline_1d": "inverse only",
    "physics_only": "inverse with physics reconstruction",
    "farp_isotropic": "grid-based isotropic reliability conditioning",
    "farp_noniter": "skeleton-based non-iterative reliability conditioning",
    "proposed": "full proposed configuration",
}

def _resolve_run_id(train_cfg: dict) -> str:
    run_id_base = f"{train_cfg['model_name']}_{train_cfg['Forward_model']}_{train_cfg['Facies_model']}"
    suffix = str(train_cfg.get("run_id_suffix", "") or "")
    return run_id_base if (suffix == "" or run_id_base.endswith(suffix)) else f"{run_id_base}{suffix}"


def _case_result_prefixes(train_cfg: dict) -> tuple[str, str]:
    run_id = _resolve_run_id(train_cfg)
    data_flag = str(train_cfg["data_flag"])
    pref_test = f"{run_id}_s_uns_{data_flag}"
    pref_train = f"{run_id}_{data_flag}"
    return pref_test, pref_train


def _list_case_result_files(results_root: Path, prefixes: tuple[str, str]) -> list[Path]:
    if not results_root.exists():
        return []
    out: list[Path] = []
    for path in results_root.iterdir():
        if not path.is_file():
            continue
        if path.name.startswith(prefixes[0]) or path.name.startswith(prefixes[1]):
            out.append(path)
    return sorted(out)


def _move_file_safe(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if not dst.exists():
        src.replace(dst)
        return dst

    idx = 1
    while True:
        candidate = dst_dir / f"{src.stem}__dup{idx}{src.suffix}"
        if not candidate.exists():
            src.replace(candidate)
            return candidate
        idx += 1


def _save_metrics_excel(xlsx_path: Path, rows: list[dict], sheet_name: str = "metrics") -> None:
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name

    if not rows:
        ws.append(["info"])
        ws.append(["no rows"])
        wb.save(xlsx_path)
        return

    headers = list(rows[0].keys())
    ws.append(headers)
    for row in rows:
        ws.append([row.get(h, "") for h in headers])
    wb.save(xlsx_path)


REPORT_METRIC_KEYS: List[str] = [
    "r2",
    "pcc",
    "p_value",
    "ssim",
    "mse",
    "mae",
    "medae",
    "psnr",
]


def _build_report_row(case_dir: str, metrics: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"case_dir": case_dir}
    for key in REPORT_METRIC_KEYS:
        row[key] = metrics.get(key, "")
    return row


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def build_case_configs(case_name: str, epochs_override: int | None) -> tuple[dict, dict]:
    if case_name not in PAPER_CASES:
        raise ValueError(f"Unknown paper case: {case_name}")

    preset = PAPER_CASES[case_name]
    train_expert = copy.deepcopy(TRAIN_EXPERT_OVERRIDES)
    test_expert = copy.deepcopy(TEST_EXPERT_OVERRIDES)

    train_expert.update(PAPER_ABLATION_COMMON)
    train_expert.update(preset)

    test_expert.update(PAPER_ABLATION_COMMON)
    test_expert.update(PAPER_TEST_COMMON)
    test_expert.update(preset)

    train_cfg = build_train_config(
        profile=TRAIN_PROFILE,
        user_cfg=TRAIN_USER_P,
        expert_overrides=train_expert,
    )
    if epochs_override is not None:
        train_cfg["epochs"] = int(epochs_override)

    test_cfg = build_test_config(
        profile=TEST_PROFILE,
        user_cfg=TEST_USER_P,
        expert_overrides=test_expert,
        train_cfg=train_cfg,
    )
    return train_cfg, test_cfg


def run_cases(cases: List[str], mode: str, epochs_override: int | None) -> None:
    results_root = Path("results")
    results_root.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_root = results_root / f"paper_ablation_{run_stamp}"
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[RUN] paper ablation results root: {run_root.as_posix()}")
    summary_rows: list[dict] = []

    for idx, case in enumerate(cases, start=1):
        train_cfg, test_cfg = build_case_configs(case_name=case, epochs_override=epochs_override)
        prefixes = _case_result_prefixes(train_cfg)
        case_dir = run_root / f"{idx:02d}_{case}"
        case_dir.mkdir(parents=True, exist_ok=True)

        with open(case_dir / "train_config.json", "w", encoding="utf-8") as f:
            json.dump(_jsonable(train_cfg), f, indent=2, ensure_ascii=True)
        with open(case_dir / "test_config.json", "w", encoding="utf-8") as f:
            json.dump(_jsonable(test_cfg), f, indent=2, ensure_ascii=True)

        old_files = _list_case_result_files(results_root, prefixes)
        if old_files:
            old_dir = case_dir / "_preexisting"
            for path in old_files:
                _move_file_safe(path, old_dir)
            print(f"[ARCHIVE] moved {len(old_files)} preexisting files -> {old_dir.as_posix()}")

        print(
            f"\n[{idx}/{len(cases)}] case={case} "
            f"backend={train_cfg.get('aniso_backend', 'grid')} "
            f"iterative_R={int(bool(train_cfg.get('iterative_R', False)))}"
        )
        print(f"[CASE] {PAPER_CASE_NOTES.get(case, '')}")

        if mode in ("train", "both"):
            print("[RUN] train_multitask.train(...)")
            train(train_cfg)

        if mode in ("test", "both"):
            print("[RUN] test_3D.test(...)")
            metrics = test(test_cfg)
            if isinstance(metrics, dict):
                case_metrics = _build_report_row(case_dir.name, metrics)
                _save_metrics_excel(case_dir / "test_metrics.xlsx", [case_metrics], sheet_name="test_metrics")
                summary_rows.append(case_metrics)
                print(f"[SAVE] case metrics -> {(case_dir / 'test_metrics.xlsx').as_posix()}")

        new_files = _list_case_result_files(results_root, prefixes)
        if new_files:
            for path in new_files:
                _move_file_safe(path, case_dir)
            print(f"[SAVE] case artifacts saved to: {case_dir.as_posix()}")
        else:
            print(f"[SAVE][WARN] no case artifacts found for prefixes={prefixes}")

    if summary_rows:
        summary_xlsx = run_root / "ablation_metrics_summary.xlsx"
        _save_metrics_excel(summary_xlsx, summary_rows, sheet_name="summary")
        print(f"[SAVE] paper ablation summary -> {summary_xlsx.as_posix()}")
    else:
        print("[SAVE][WARN] no test metrics collected; summary excel not generated.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the paper ablation case chain."
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=list(PAPER_CASES.keys()),
        choices=sorted(PAPER_CASES.keys()),
        help="Paper storyline cases to run.",
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_cases(cases=args.cases, mode=args.mode, epochs_override=args.epochs)
