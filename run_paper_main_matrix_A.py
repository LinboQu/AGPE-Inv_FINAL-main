from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from utils.config_resolver import (
    DEFAULT_SEED,
    DEFAULT_SELECTED_WELLS_CSV,
    build_test_config,
    build_train_config,
)


PAPER_TRAIN_PROFILE = "stable_r51"
PAPER_TEST_PROFILE = PAPER_TRAIN_PROFILE

# Freeze the paper runner to a stable train/test budget instead of inheriting
# mutable day-to-day defaults from setting.py.
PAPER_TRAIN_USER: Dict[str, Any] = {
    "batch_size": 4,
    "no_wells": 20,
    "lambda_recon": 1.0,
    "lambda_facies": 0.2,
    "epochs": 1000,
    "grad_clip": 1.0,
    "lr": 0.0001,
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "seed": DEFAULT_SEED,
    "selected_wells_csv": DEFAULT_SELECTED_WELLS_CSV,
    "use_aniso_conditioning": True,
    "R_update_every": 50,
    "use_detail_branch": True,
    "detail_gain": 0.15,
    "channel_id": 2,
    "model_name": "VishalNet",
    "Forward_model": "cov_para",
    "Facies_model": "Facies",
    "run_id_suffix": "",
    "data_flag": "Stanford_VI",
}

PAPER_TEST_USER: Dict[str, Any] = {
    "no_wells": PAPER_TRAIN_USER["no_wells"],
    "seed": PAPER_TRAIN_USER["seed"],
    "selected_wells_csv": PAPER_TRAIN_USER["selected_wells_csv"],
    "data_flag": PAPER_TRAIN_USER["data_flag"],
    "use_aniso_conditioning": PAPER_TRAIN_USER["use_aniso_conditioning"],
    "channel_id": PAPER_TRAIN_USER["channel_id"],
    "test_init_bs": 32,
    "test_facies_bs": 64,
    "test_noise_kind": "none",
    "test_noise_snr_db": None,
    "test_noise_seed": DEFAULT_SEED,
    "test_noise_save_inputs": True,
}

# The shared paper baseline is the final deployment recipe approved by the user.
# Individual cases only remove or alter the specific mechanism under study.
PAPER_MAIN_COMMON: Dict[str, Any] = {
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
    "agpe_rebuild_every": 50,
    "R_update_every": 50,
    "aniso_closed_loop_conf_thresh": 0.60,
    "agpe_lift_sigma": 2.3,
    "agpe_long_weight": 0.35,
    "agpe_well_soft_alpha": 0.20,
    "agpe_well_seed_mode": "depth_gate",
    "agpe_well_seed_min": 0.02,
    "use_boundary_weight": False,
    "lambda_depth_grad": 0.0,
    "lambda_depth_hf": 0.0,
    "train_noise_kind": "awgn",
    "train_noise_prob": 0.5,
    "train_noise_snr_db_choices": (30.0, 20.0),
    "r_channel_dropout_prob": 0.3,
    "lambda_recon": 1.0,
    "lambda_facies": 0.2,
}


def _paper_case(**overrides: Any) -> Dict[str, Any]:
    out = copy.deepcopy(PAPER_MAIN_COMMON)
    out.update(overrides)
    return out


PAPER_CASES: Dict[str, Dict[str, Any]] = {
    # 1. Pure inverse baseline.
    "paper_inv_only": _paper_case(
        use_aniso_conditioning=False,
        iterative_R=False,
        lambda_recon=0.0,
        lambda_facies=0.0,
        run_id_suffix="_paper_inv_only",
    ),
    # 2. Add physics reconstruction only.
    "paper_inv_phys": _paper_case(
        use_aniso_conditioning=False,
        iterative_R=False,
        lambda_recon=1.0,
        lambda_facies=0.0,
        run_id_suffix="_paper_inv_phys",
    ),
    # 3. Add facies auxiliary task, still no R-conditioning.
    "paper_inv_phys_facies": _paper_case(
        use_aniso_conditioning=False,
        iterative_R=False,
        lambda_recon=1.0,
        lambda_facies=0.2,
        run_id_suffix="_paper_inv_phys_facies",
    ),
    # 4. Enable isotropic/static reliability conditioning.
    "paper_inv_phys_facies_R_iso_static": _paper_case(
        use_aniso_conditioning=True,
        aniso_backend="grid",
        aniso_use_tensor_strength=False,
        aniso_kappa=0.0,
        iterative_R=False,
        lambda_recon=1.0,
        lambda_facies=0.2,
        run_id_suffix="_paper_inv_phys_facies_R_iso_static",
    ),
    # 5. Change only the R module to anisotropic skeleton/static.
    "paper_inv_phys_facies_R_aniso_static": _paper_case(
        use_aniso_conditioning=True,
        aniso_backend="skeleton_graph",
        aniso_use_tensor_strength=True,
        aniso_kappa=4.0,
        iterative_R=False,
        lambda_recon=1.0,
        lambda_facies=0.2,
        run_id_suffix="_paper_inv_phys_facies_R_aniso_static",
    ),
    # 6. Final full method. This is the paper main model and corresponds to the
    # old experimental line skel_ref_nobddep_awgntrain_rdrop, renamed to avoid confusion.
    "paper_main_full": _paper_case(
        use_aniso_conditioning=True,
        aniso_backend="skeleton_graph",
        aniso_use_tensor_strength=True,
        aniso_kappa=4.0,
        iterative_R=True,
        lambda_recon=1.0,
        lambda_facies=0.2,
        run_id_suffix="_paper_main_full",
    ),
}


PAPER_CASE_NOTES: Dict[str, str] = {
    "paper_inv_only": "inverse only; recon/facies/R disabled",
    "paper_inv_phys": "inverse + physics reconstruction only",
    "paper_inv_phys_facies": "inverse + physics + facies auxiliary task",
    "paper_inv_phys_facies_R_iso_static": "isotropic static reliability conditioning",
    "paper_inv_phys_facies_R_aniso_static": "anisotropic skeleton reliability conditioning without iterative updates",
    "paper_main_full": "final paper method: anisotropic iterative closed-loop under the fixed deployment recipe",
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
    from openpyxl import Workbook

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
    train_expert = copy.deepcopy(PAPER_MAIN_COMMON)
    test_expert = copy.deepcopy(PAPER_MAIN_COMMON)
    train_expert.update(preset)
    test_expert.update(preset)

    train_cfg = build_train_config(
        profile=PAPER_TRAIN_PROFILE,
        user_cfg=PAPER_TRAIN_USER,
        expert_overrides=train_expert,
    )
    if epochs_override is not None:
        train_cfg["epochs"] = int(epochs_override)

    test_cfg = build_test_config(
        profile=PAPER_TEST_PROFILE,
        user_cfg=PAPER_TEST_USER,
        expert_overrides=test_expert,
        train_cfg=train_cfg,
    )
    return train_cfg, test_cfg


def run_cases(cases: List[str], mode: str, epochs_override: int | None) -> None:
    from test_3D import test
    from train_multitask import train

    results_root = Path("results")
    results_root.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_root = results_root / f"paper_main_matrix_A_{run_stamp}"
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[RUN] paper matrix A results root: {run_root.as_posix()}")
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
            f"iterative_R={int(bool(train_cfg.get('iterative_R', False)))} "
            f"use_R={int(bool(train_cfg.get('use_aniso_conditioning', False)))}"
        )
        print(f"[CASE] {PAPER_CASE_NOTES.get(case, '')}")

        if mode in ("train", "both"):
            print("[RUN] train_multitask.train(...)")
            train(train_cfg)

        if mode in ("test", "both"):
            print("[RUN] test_3D.test(...)")
            metrics = test(test_cfg)
            if isinstance(metrics, dict):
                case_metrics = {
                    "case_dir": case_dir.name,
                    "case": case,
                    "case_note": PAPER_CASE_NOTES.get(case, ""),
                    "paper_group": "A_main_mechanism_clean",
                    "backend": str(train_cfg.get("aniso_backend", "grid")),
                    "use_aniso_conditioning": bool(train_cfg.get("use_aniso_conditioning", False)),
                    "iterative_R": bool(train_cfg.get("iterative_R", False)),
                    "use_tensor_strength": bool(train_cfg.get("aniso_use_tensor_strength", False)),
                    "aniso_kappa": float(train_cfg.get("aniso_kappa", 4.0)),
                    "agpe_rebuild_every": int(train_cfg.get("agpe_rebuild_every", 50)),
                    "lambda_recon": float(train_cfg.get("lambda_recon", 0.0)),
                    "lambda_facies": float(train_cfg.get("lambda_facies", 0.0)),
                    "use_boundary_weight": bool(train_cfg.get("use_boundary_weight", False)),
                    "lambda_depth_grad": float(train_cfg.get("lambda_depth_grad", 0.0)),
                    "lambda_depth_hf": float(train_cfg.get("lambda_depth_hf", 0.0)),
                    "train_noise_kind": str(train_cfg.get("train_noise_kind", "none")),
                    "train_noise_prob": float(train_cfg.get("train_noise_prob", 0.0)),
                    "train_noise_snr_db_choices": str(train_cfg.get("train_noise_snr_db_choices", ())),
                    "r_channel_dropout_prob": float(train_cfg.get("r_channel_dropout_prob", 0.0)),
                    "run_id": _resolve_run_id(train_cfg),
                    "model_name": f"{_resolve_run_id(train_cfg)}_s_uns",
                    "data_flag": str(train_cfg.get("data_flag", "")),
                    **metrics,
                }
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
        summary_xlsx = run_root / "paper_main_matrix_A_summary.xlsx"
        _save_metrics_excel(summary_xlsx, summary_rows, sheet_name="summary")
        print(f"[SAVE] paper matrix A summary -> {summary_xlsx.as_posix()}")
    else:
        print("[SAVE][WARN] no test metrics collected; summary excel not generated.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paper main-mechanism ablation matrix A under a frozen deployment recipe."
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=list(PAPER_CASES.keys()),
        choices=sorted(PAPER_CASES.keys()),
        help="Paper cases to run.",
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
