import argparse
import copy
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from openpyxl import Workbook

from setting import (
    TEST_EXPERT_OVERRIDES,
    TEST_PROFILE,
    TEST_USER_P,
    TRAIN_EXPERT_OVERRIDES,
    TRAIN_PROFILE,
    TRAIN_USER_P,
)
from train_multitask import train
from test_3D import test
from utils.config_resolver import build_test_config, build_train_config


# Fair ablation common configuration (aligned with current best closed-loop line).
FAIR_ABLATION_COMMON: Dict[str, object] = {
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
}


def _fair_case(**overrides: object) -> Dict[str, object]:
    out = copy.deepcopy(FAIR_ABLATION_COMMON)
    out.update(overrides)
    return out


CASE_PRESETS: Dict[str, Dict[str, object]] = {
    # Control: legacy grid backend
    "grid": {
        "aniso_backend": "grid",
        "aniso_use_tensor_strength": False,
        "run_id_suffix": "_grid",
    },
    # Grid backend with iterative_R explicitly disabled
    "grid_static": {
        "aniso_backend": "grid",
        "aniso_use_tensor_strength": False,
        "iterative_R": False,
        "run_id_suffix": "_grid_static",
    },
    # Graph backend without tensor-strength modulation (parity with grid)
    "glat": {
        "aniso_backend": "graph_lattice",
        "aniso_use_tensor_strength": False,
        "run_id_suffix": "_glat",
    },
    # Graph-lattice backend with iterative_R explicitly disabled
    "glat_static": {
        "aniso_backend": "graph_lattice",
        "aniso_use_tensor_strength": False,
        "iterative_R": False,
        "run_id_suffix": "_glat_static",
    },
    # Graph backend with tensor-strength modulation enabled
    "glat_ts": {
        "aniso_backend": "graph_lattice",
        "aniso_use_tensor_strength": True,
        "run_id_suffix": "_glat_ts",
    },
    # Skeleton graph without long-range edges
    "skel_nolong": {
        "aniso_backend": "skeleton_graph",
        "aniso_use_tensor_strength": True,
        "agpe_long_edges": False,
        "iterative_R": False,
        "run_id_suffix": "_skel_nolong",
    },
    # Skeleton graph without long-range edges, iterative-R enabled
    "skel_nolong_iter": {
        "aniso_backend": "skeleton_graph",
        "aniso_use_tensor_strength": True,
        "agpe_long_edges": False,
        "iterative_R": True,
        "agpe_cache_graph": True,
        "agpe_refine_graph": False,
        "run_id_suffix": "_skel_nolong_iter",
    },
    # Skeleton graph with long-range edges
    "skel_long": {
        "aniso_backend": "skeleton_graph",
        "aniso_use_tensor_strength": True,
        "agpe_long_edges": True,
        "iterative_R": False,
        "run_id_suffix": "_skel_long",
    },
    # Skeleton graph iterative-R, no topology rebuild (edge-weight update only)
    "skel_noref": {
        "aniso_backend": "skeleton_graph",
        "aniso_use_tensor_strength": True,
        "agpe_long_edges": True,
        "iterative_R": True,
        "agpe_cache_graph": True,
        "agpe_refine_graph": False,
        "run_id_suffix": "_skel_noref",
    },
    # Skeleton graph iterative-R, periodic/triggered topology rebuild enabled
    "skel_ref": {
        "aniso_backend": "skeleton_graph",
        "aniso_use_tensor_strength": True,
        "agpe_long_edges": True,
        "iterative_R": True,
        "agpe_cache_graph": True,
        "agpe_refine_graph": True,
        "agpe_rebuild_every": 50,
        "run_id_suffix": "_skel_ref",
    },
    # Best current line (locked): r51_nobw_rb100_cf060_ls23
    "skel_ref_best_r51": {
        "aniso_backend": "skeleton_graph",
        "aniso_use_tensor_strength": True,
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
        "use_boundary_weight": False,
        "lambda_depth_grad": 0.005,
        "lambda_depth_hf": 0.001,
        "use_depth_warm_schedule": True,
        "depth_warm_start_epoch": 500,
        "depth_warm_ramp_epochs": 300,
        "ws_every": 4,
        "ws_max_batches": 80,
        "ws_max_batches_stageA": 30,
        "ws_every_late": 12,
        "ws_max_batches_late": 8,
        "run_id_suffix": "_skel_ref_best_r51",
    },
    # Isolation-A: Step-4 off (disable detail branch), keep everything else as skel_ref
    "skel_ref_nodetail": {
        "aniso_backend": "skeleton_graph",
        "aniso_use_tensor_strength": True,
        "agpe_long_edges": True,
        "iterative_R": True,
        "agpe_cache_graph": True,
        "agpe_refine_graph": True,
        "agpe_rebuild_every": 50,
        "use_detail_branch": False,
        "run_id_suffix": "_skel_ref_nodetail",
    },
    # Isolation-B: Step-3 boundary/depth constraints off, keep detail branch on
    "skel_ref_nobddep": {
        "aniso_backend": "skeleton_graph",
        "aniso_use_tensor_strength": True,
        "agpe_long_edges": True,
        "iterative_R": True,
        "agpe_cache_graph": True,
        "agpe_refine_graph": True,
        "agpe_rebuild_every": 50,
        "use_boundary_weight": False,
        "lambda_depth_grad": 0.0,
        "lambda_depth_hf": 0.0,
        "run_id_suffix": "_skel_ref_nobddep",
    },
    # Isolation-C: Step-3 boundary/depth constraints off + Step-4 detail branch off
    "skel_ref_nodetail_nobddep": {
        "aniso_backend": "skeleton_graph",
        "aniso_use_tensor_strength": True,
        "agpe_long_edges": True,
        "iterative_R": True,
        "agpe_cache_graph": True,
        "agpe_refine_graph": True,
        "agpe_rebuild_every": 50,
        "use_detail_branch": False,
        "use_boundary_weight": False,
        "lambda_depth_grad": 0.0,
        "lambda_depth_hf": 0.0,
        "run_id_suffix": "_skel_ref_nodetail_nobddep",
    },

    # ---------------------------
    # Fair comparison block (paper-ready)
    # ---------------------------
    # Baseline: no physics, no R (pure inversion only)
    "fair_baseline_1d": _fair_case(
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
    # Physics: forward reconstruction only (still no R/FARP)
    "fair_physics_only": _fair_case(
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
    # FARP-Isotropic: isotropic reliability (phi=0 / no anisotropy)
    "fair_farp_isotropic": _fair_case(
        aniso_backend="grid",
        aniso_use_tensor_strength=False,
        aniso_kappa=0.0,
        iterative_R=True,
        run_id_suffix="_fair_farp_isotropic",
    ),
    # FARP-Non-iter: disable closed-loop iterative R update
    "fair_farp_noniter": _fair_case(
        iterative_R=False,
        run_id_suffix="_fair_farp_noniter",
    ),
    # Proposed: full latest setting (current best line)
    "fair_proposed": _fair_case(
        run_id_suffix="_fair_proposed",
    ),
    # Deconfound-A: only rebuild frequency (100 -> 50), keep others as fair_proposed
    "fair_proposed_rb50": _fair_case(
        agpe_rebuild_every=50,
        run_id_suffix="_fair_proposed_rb50",
    ),
    # Deconfound-B: only disable depth losses, keep rb100 and other settings unchanged
    "fair_proposed_nobddep_rb100": _fair_case(
        lambda_depth_grad=0.0,
        lambda_depth_hf=0.0,
        run_id_suffix="_fair_proposed_nobddep_rb100",
    ),
}


def _resolve_run_id(train_cfg: dict) -> str:
    run_id_base = f"{train_cfg['model_name']}_{train_cfg['Forward_model']}_{train_cfg['Facies_model']}"
    suffix = str(train_cfg.get("run_id_suffix", "") or "")
    return run_id_base if (suffix == "" or run_id_base.endswith(suffix)) else f"{run_id_base}{suffix}"


def _case_result_prefixes(train_cfg: dict) -> tuple[str, str]:
    run_id = _resolve_run_id(train_cfg)
    data_flag = str(train_cfg["data_flag"])
    # test artifacts / train curve
    pref_test = f"{run_id}_s_uns_{data_flag}"
    # training side artifacts (well indices, iterative R snapshots)
    pref_train = f"{run_id}_{data_flag}"
    return pref_test, pref_train


def _list_case_result_files(results_root: Path, prefixes: tuple[str, str]) -> list[Path]:
    if not results_root.exists():
        return []
    out: list[Path] = []
    for p in results_root.iterdir():
        if not p.is_file():
            continue
        if p.name.startswith(prefixes[0]) or p.name.startswith(prefixes[1]):
            out.append(p)
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
    """Save rows to a simple Excel table."""
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name

    if not rows:
        ws.append(["info"])
        ws.append(["no metrics rows"])
        wb.save(xlsx_path)
        return

    headers = list(rows[0].keys())
    ws.append(headers)
    for row in rows:
        ws.append([row.get(h, "") for h in headers])
    wb.save(xlsx_path)


def build_case_configs(case_name: str, epochs_override: int | None) -> tuple[dict, dict]:
    if case_name not in CASE_PRESETS:
        raise ValueError(f"Unknown case: {case_name}")

    preset = CASE_PRESETS[case_name]
    train_expert = copy.deepcopy(TRAIN_EXPERT_OVERRIDES)
    test_expert = copy.deepcopy(TEST_EXPERT_OVERRIDES)

    # Fairness guarantee: all ablation cases start from the same common baseline,
    # then apply case-specific differences only.
    train_expert.update(FAIR_ABLATION_COMMON)
    train_expert.update(preset)

    test_expert.update(FAIR_ABLATION_COMMON)
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
    run_root = results_root / f"ablation_{run_stamp}"
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[RUN] ablation results root: {run_root.as_posix()}")
    summary_rows: list[dict] = []

    for idx, case in enumerate(cases, start=1):
        train_cfg, test_cfg = build_case_configs(case, epochs_override=epochs_override)
        suffix = train_cfg.get("run_id_suffix", "")
        backend = train_cfg.get("aniso_backend", "grid")
        use_ts = bool(train_cfg.get("aniso_use_tensor_strength", False))
        prefixes = _case_result_prefixes(train_cfg)
        case_dir = run_root / f"{idx:02d}_{case}"
        case_dir.mkdir(parents=True, exist_ok=True)

        # Avoid overwrite: move any old same-prefix outputs out of results/ before this case runs.
        old_files = _list_case_result_files(results_root, prefixes)
        if old_files:
            old_dir = case_dir / "_preexisting"
            for p in old_files:
                _move_file_safe(p, old_dir)
            print(f"[ARCHIVE] moved {len(old_files)} preexisting files -> {old_dir.as_posix()}")

        print(
            f"\n[{idx}/{len(cases)}] case={case} "
            f"backend={backend} use_tensor_strength={use_ts} suffix={suffix}"
        )

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
                    "backend": str(backend),
                    "use_tensor_strength": bool(use_ts),
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
                    "agpe_well_seed_mode": str(train_cfg.get("agpe_well_seed_mode", "hard")),
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
                _save_metrics_excel(case_dir / "test_metrics.xlsx", [case_metrics], sheet_name="test_metrics")
                summary_rows.append(case_metrics)
                print(f"[SAVE] case metrics -> {(case_dir / 'test_metrics.xlsx').as_posix()}")

        new_files = _list_case_result_files(results_root, prefixes)
        if new_files:
            for p in new_files:
                _move_file_safe(p, case_dir)
            print(f"[SAVE] case artifacts saved to: {case_dir.as_posix()}")
        else:
            print(f"[SAVE][WARN] no case artifacts found for prefixes={prefixes}")

    if summary_rows:
        summary_xlsx = run_root / "ablation_metrics_summary.xlsx"
        _save_metrics_excel(summary_xlsx, summary_rows, sheet_name="summary")
        print(f"[SAVE] ablation summary metrics -> {summary_xlsx.as_posix()}")
    else:
        print("[SAVE][WARN] no test metrics collected; summary excel not generated.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AGPE ablations for grid / graph_lattice / skeleton_graph backends."
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "fair_baseline_1d",
            "fair_physics_only",
            "fair_farp_isotropic",
            "fair_farp_noniter",
            "fair_proposed",
            "fair_proposed_rb50",
            "fair_proposed_nobddep_rb100",
        ],
        choices=sorted(CASE_PRESETS.keys()),
        help="Ablation cases to run (default: all).",
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
        help="Optional override for training epochs (useful for quick smoke runs).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_cases(cases=args.cases, mode=args.mode, epochs_override=args.epochs)
