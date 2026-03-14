from __future__ import annotations

import argparse
import copy
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from openpyxl import Workbook
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from skimage.metrics import structural_similarity as ssim

from utils.config_resolver import (
    DEFAULT_SEED,
    DEFAULT_SELECTED_WELLS_CSV,
    build_test_config,
    build_train_config,
)
from utils.metrics import compute_vif_mscale


TRAIN_PROFILE = "stable_r51"
TEST_PROFILE = TRAIN_PROFILE

REPRESENTATIVE_DEPTH = 100
REPRESENTATIVE_INLINE = 100
REPRESENTATIVE_XLINE = 50

REPORT_METRIC_KEYS: List[str] = [
    "r2",
    "pcc",
    "p_value",
    "ssim",
    "vif",
    "mse",
    "mae",
    "medae",
]


TUNE_TRAIN_USER: Dict[str, Any] = {
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

TUNE_TEST_USER: Dict[str, Any] = {
    "no_wells": TUNE_TRAIN_USER["no_wells"],
    "seed": TUNE_TRAIN_USER["seed"],
    "selected_wells_csv": TUNE_TRAIN_USER["selected_wells_csv"],
    "data_flag": TUNE_TRAIN_USER["data_flag"],
    "use_aniso_conditioning": TUNE_TRAIN_USER["use_aniso_conditioning"],
    "channel_id": TUNE_TRAIN_USER["channel_id"],
    "test_init_bs": 32,
    "test_facies_bs": 64,
    "test_noise_kind": "none",
    "test_noise_snr_db": None,
    "test_noise_seed": DEFAULT_SEED,
    "test_noise_save_inputs": True,
}


CURRENT_PROPOSED_BASE: Dict[str, Any] = {
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
    "lambda_recon": 1.0,
    "lambda_facies": 0.2,
    "use_detail_branch": True,
    "detail_gain": 0.15,
    "lambda_amp_anchor": 0.05,
    "train_noise_kind": "awgn",
    "train_noise_prob": 0.5,
    "train_noise_snr_db_choices": (30.0, 20.0),
    "r_channel_dropout_prob": 0.3,
}

CLEAN_FULL_BASE: Dict[str, Any] = {
    **CURRENT_PROPOSED_BASE,
    "train_noise_kind": "none",
    "train_noise_prob": 0.0,
    "train_noise_snr_db_choices": (),
    "r_channel_dropout_prob": 0.0,
}


@dataclass(frozen=True)
class TrialSpec:
    stage: str
    group: str
    name: str
    suffix: str
    base: str
    note: str
    overrides: Dict[str, Any]


def _trial(
    *,
    stage: str,
    group: str,
    name: str,
    suffix: str,
    base: str,
    note: str,
    **overrides: Any,
) -> TrialSpec:
    return TrialSpec(
        stage=stage,
        group=group,
        name=name,
        suffix=suffix,
        base=base,
        note=note,
        overrides=dict(overrides),
    )


TRIAL_SPECS: Dict[str, TrialSpec] = {
    "s0_ref_current_proposed": _trial(
        stage="stage0",
        group="reference",
        name="s0_ref_current_proposed",
        suffix="_ptune_s0_curr",
        base="current_proposed",
        note="Exact current Proposed configuration with a dedicated tuning run_id.",
    ),
    "s0_ref_clean_full": _trial(
        stage="stage0",
        group="reference",
        name="s0_ref_clean_full",
        suffix="_ptune_s0_clean",
        base="clean_full",
        note="Clean full-mechanism reference without robustness-specific training tricks.",
    ),
    "s1_d01_detail_soft": _trial(
        stage="stage1",
        group="detail",
        name="s1_d01_detail_soft",
        suffix="_ptune_s1_d01",
        base="current_proposed",
        note="Add a soft amount of depth-gradient and high-frequency regularization.",
        lambda_depth_grad=0.002,
        lambda_depth_hf=0.0004,
    ),
    "s1_d02_detail_mid": _trial(
        stage="stage1",
        group="detail",
        name="s1_d02_detail_mid",
        suffix="_ptune_s1_d02",
        base="current_proposed",
        note="Restore the stable_r51 detail loss level while keeping current robustness tricks.",
        lambda_depth_grad=0.005,
        lambda_depth_hf=0.001,
    ),
    "s1_d03_detail_mid_gain020": _trial(
        stage="stage1",
        group="detail",
        name="s1_d03_detail_mid_gain020",
        suffix="_ptune_s1_d03",
        base="current_proposed",
        note="Increase detail-branch contribution together with mid-strength detail losses.",
        lambda_depth_grad=0.005,
        lambda_depth_hf=0.001,
        detail_gain=0.20,
    ),
    "s1_d04_detail_soft_boundary": _trial(
        stage="stage1",
        group="detail",
        name="s1_d04_detail_soft_boundary",
        suffix="_ptune_s1_d04",
        base="current_proposed",
        note="Soft detail losses plus a weak late AI-only boundary weighting term.",
        lambda_depth_grad=0.002,
        lambda_depth_hf=0.0004,
        use_boundary_weight=True,
        boundary_weight_beta=0.08,
        boundary_weight_width=1,
        use_boundary_warm_schedule=True,
        boundary_beta_start=0.0,
        boundary_beta_end=0.08,
        boundary_warm_start_epoch=500,
        boundary_warm_ramp_epochs=300,
        boundary_weight_apply_ai=True,
        boundary_weight_apply_detail=False,
        boundary_weight_apply_facies=False,
    ),
    "s1_r01_robust_light": _trial(
        stage="stage1",
        group="robustness",
        name="s1_r01_robust_light",
        suffix="_ptune_s1_r01",
        base="current_proposed",
        note="Reduce AWGN probability and channel-dropout strength while keeping the same SNR band.",
        train_noise_kind="awgn",
        train_noise_prob=0.30,
        train_noise_snr_db_choices=(30.0, 20.0),
        r_channel_dropout_prob=0.15,
    ),
    "s1_r02_robust_gentle": _trial(
        stage="stage1",
        group="robustness",
        name="s1_r02_robust_gentle",
        suffix="_ptune_s1_r02",
        base="current_proposed",
        note="Use a gentler noise protocol and lighter channel dropout to recover fine contrast.",
        train_noise_kind="awgn",
        train_noise_prob=0.20,
        train_noise_snr_db_choices=(35.0, 25.0),
        r_channel_dropout_prob=0.10,
    ),
    "s1_r03_clean_detail_mid": _trial(
        stage="stage1",
        group="robustness",
        name="s1_r03_clean_detail_mid",
        suffix="_ptune_s1_r03",
        base="current_proposed",
        note="Disable robustness tricks entirely and add back mid-strength detail losses.",
        train_noise_kind="none",
        train_noise_prob=0.0,
        train_noise_snr_db_choices=(),
        r_channel_dropout_prob=0.0,
        lambda_depth_grad=0.005,
        lambda_depth_hf=0.001,
    ),
    "s1_a01_lift_sigma_20": _trial(
        stage="stage1",
        group="agpe",
        name="s1_a01_lift_sigma_20",
        suffix="_ptune_s1_a01",
        base="current_proposed",
        note="Tighten slice lift-back decay to reduce off-skeleton spreading.",
        agpe_lift_sigma=2.0,
    ),
    "s1_a02_conf_065": _trial(
        stage="stage1",
        group="agpe",
        name="s1_a02_conf_065",
        suffix="_ptune_s1_a02",
        base="current_proposed",
        note="Increase closed-loop channel confidence threshold to constrain graph support.",
        aniso_closed_loop_conf_thresh=0.65,
    ),
    "s1_a03_long_weight_025": _trial(
        stage="stage1",
        group="agpe",
        name="s1_a03_long_weight_025",
        suffix="_ptune_s1_a03",
        base="current_proposed",
        note="Reduce long-edge propagation weight to limit over-smoothing along graph shortcuts.",
        agpe_long_weight=0.25,
    ),
    "s1_a04_soft_alpha_010": _trial(
        stage="stage1",
        group="agpe",
        name="s1_a04_soft_alpha_010",
        suffix="_ptune_s1_a04",
        base="current_proposed",
        note="Reduce soft well blending so the propagated field stays more localized.",
        agpe_well_soft_alpha=0.10,
    ),
    "s1_c01_detail_soft_robust_light_tight": _trial(
        stage="stage1",
        group="combined",
        name="s1_c01_detail_soft_robust_light_tight",
        suffix="_ptune_s1_c01",
        base="current_proposed",
        note="Combined candidate: soft detail losses + lighter robustness + tighter AGPE support.",
        lambda_depth_grad=0.002,
        lambda_depth_hf=0.0004,
        train_noise_kind="awgn",
        train_noise_prob=0.30,
        train_noise_snr_db_choices=(30.0, 20.0),
        r_channel_dropout_prob=0.15,
        agpe_lift_sigma=2.0,
        agpe_long_weight=0.25,
        agpe_well_soft_alpha=0.10,
        aniso_closed_loop_conf_thresh=0.65,
    ),
}

STAGE_ORDERS: Dict[str, List[str]] = {
    "stage0": [
        "s0_ref_current_proposed",
        "s0_ref_clean_full",
    ],
    "stage1": [
        "s1_d01_detail_soft",
        "s1_d02_detail_mid",
        "s1_d03_detail_mid_gain020",
        "s1_d04_detail_soft_boundary",
        "s1_r01_robust_light",
        "s1_r02_robust_gentle",
        "s1_r03_clean_detail_mid",
        "s1_a01_lift_sigma_20",
        "s1_a02_conf_065",
        "s1_a03_long_weight_025",
        "s1_a04_soft_alpha_010",
        "s1_c01_detail_soft_robust_light_tight",
    ],
}
STAGE_ORDERS["all"] = STAGE_ORDERS["stage0"] + STAGE_ORDERS["stage1"]


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _tabular_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_jsonable(value), ensure_ascii=True)
    return value


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
        ws.append([_tabular_value(row.get(h, "")) for h in headers])
    wb.save(xlsx_path)


def _save_metrics_csv(csv_path: Path, rows: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        csv_path.write_text("info\nno rows\n", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({h: _tabular_value(row.get(h, "")) for h in headers})


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


def _base_overrides_for_trial(spec: TrialSpec) -> dict[str, Any]:
    if spec.base == "current_proposed":
        base = CURRENT_PROPOSED_BASE
    elif spec.base == "clean_full":
        base = CLEAN_FULL_BASE
    else:
        raise ValueError(f"Unknown trial base: {spec.base}")
    out = copy.deepcopy(base)
    out.update(copy.deepcopy(spec.overrides))
    out["run_id_suffix"] = spec.suffix
    return out


def _scaled_int(value: int, ratio: float, *, min_value: int = 1) -> int:
    if int(value) <= 0:
        return int(value)
    return max(min_value, int(round(float(value) * float(ratio))))


def _apply_epoch_budget(train_cfg: dict[str, Any], epochs_override: int | None) -> None:
    if epochs_override is None:
        return
    old_epochs = int(train_cfg.get("epochs", epochs_override))
    new_epochs = int(epochs_override)
    if new_epochs <= 0:
        raise ValueError(f"epochs must be positive, got {new_epochs}")
    ratio = float(new_epochs) / max(float(old_epochs), 1.0)
    train_cfg["epochs"] = new_epochs

    scaled_fields = (
        "stageA_epochs",
        "stageB_ramp_epochs",
        "depth_warm_start_epoch",
        "depth_warm_ramp_epochs",
        "late_multitask_start_epoch",
        "alpha_prior_decay_epochs",
        "boundary_warm_start_epoch",
        "boundary_warm_ramp_epochs",
        "save_R_every",
        "R_update_every",
        "agpe_rebuild_every",
        "ws_max_batches",
        "ws_max_batches_stageA",
        "ws_max_batches_late",
    )
    for key in scaled_fields:
        if key in train_cfg:
            train_cfg[key] = _scaled_int(int(train_cfg[key]), ratio, min_value=1)

    train_cfg["stageA_epochs"] = min(int(train_cfg.get("stageA_epochs", 0)), new_epochs)
    train_cfg["stageB_ramp_epochs"] = min(int(train_cfg.get("stageB_ramp_epochs", 0)), new_epochs)
    train_cfg["depth_warm_start_epoch"] = min(int(train_cfg.get("depth_warm_start_epoch", 0)), new_epochs)
    train_cfg["late_multitask_start_epoch"] = min(int(train_cfg.get("late_multitask_start_epoch", 0)), new_epochs)
    train_cfg["alpha_prior_decay_epochs"] = min(int(train_cfg.get("alpha_prior_decay_epochs", new_epochs)), new_epochs)


def build_trial_configs(trial_name: str, epochs_override: int | None) -> tuple[dict, dict, TrialSpec]:
    if trial_name not in TRIAL_SPECS:
        raise ValueError(f"Unknown tuning trial: {trial_name}")

    spec = TRIAL_SPECS[trial_name]
    train_expert = _base_overrides_for_trial(spec)
    test_expert = copy.deepcopy(train_expert)

    train_cfg = build_train_config(
        profile=TRAIN_PROFILE,
        user_cfg=TUNE_TRAIN_USER,
        expert_overrides=train_expert,
    )
    _apply_epoch_budget(train_cfg, epochs_override)

    test_cfg = build_test_config(
        profile=TEST_PROFILE,
        user_cfg=TUNE_TEST_USER,
        expert_overrides=test_expert,
        train_cfg=train_cfg,
    )
    return train_cfg, test_cfg, spec


def _find_unique_file(case_dir: Path, pattern: str) -> Path:
    matches = sorted(case_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching '{pattern}' found in {case_dir}")
    if len(matches) > 1:
        raise RuntimeError(f"Expected a unique match for '{pattern}' in {case_dir}, found {[m.name for m in matches]}")
    return matches[0]


def _load_stats_for_pred(pred_path: Path, data_flag: str) -> dict:
    run_id = pred_path.name
    suffix = f"_s_uns_{data_flag}_pred_AI.npy"
    if run_id.endswith(suffix):
        run_id = run_id[: -len(suffix)]
    else:
        suffix = f"_{data_flag}_pred_AI.npy"
        if run_id.endswith(suffix):
            run_id = run_id[: -len(suffix)]
        else:
            raise ValueError(f"Cannot infer run_id from prediction filename: {pred_path.name}")

    dedicated = Path("save_train_model") / f"norm_stats_{run_id}_{data_flag}.npy"
    legacy = Path("save_train_model") / f"norm_stats_{data_flag}.npy"
    if dedicated.is_file():
        return np.load(dedicated, allow_pickle=True).item()
    if legacy.is_file():
        return np.load(legacy, allow_pickle=True).item()
    raise FileNotFoundError(f"Cannot find stats for prediction file: {pred_path}")


def _load_gt_cube(data_flag: str) -> np.ndarray:
    if data_flag == "Stanford_VI":
        arr = np.load(Path("data") / data_flag / "AI.npy").astype(np.float32, copy=False)
        return np.transpose(arr, (1, 2, 0))
    if data_flag == "Fanny":
        arr = np.load(Path("data") / data_flag / "impedance.npy").astype(np.float32, copy=False)
        if arr.ndim == 3:
            return np.transpose(arr, (1, 2, 0))
        raise ValueError(f"Unsupported Fanny impedance shape: {arr.shape}")
    raise ValueError(f"Unsupported data_flag for visual analysis: {data_flag}")


def _flatten_hilxl_to_nh(arr_hilxl: np.ndarray) -> np.ndarray:
    h, il, xl = arr_hilxl.shape
    return np.transpose(arr_hilxl.reshape(h, il * xl, order="C"), (1, 0)).astype(np.float32, copy=False)


def _denormalize_model(flat: np.ndarray, stats: dict) -> np.ndarray:
    return flat.astype(np.float32, copy=False) * float(stats["model_std"]) + float(stats["model_mean"])


def _maybe_denormalize_pred(flat: np.ndarray, stats: dict, raw_gt_ilxlh: np.ndarray) -> np.ndarray:
    raw_gt_hilxl = np.transpose(raw_gt_ilxlh, (2, 0, 1))
    raw_flat = _flatten_hilxl_to_nh(raw_gt_hilxl)
    cand_phys = flat.astype(np.float32, copy=False)
    cand_std = _denormalize_model(flat, stats)

    def _range_score(arr: np.ndarray) -> float:
        return float(
            abs(np.percentile(arr, 1) - np.percentile(raw_flat, 1))
            + abs(np.percentile(arr, 99) - np.percentile(raw_flat, 99))
        )

    return cand_std if _range_score(cand_std) < _range_score(cand_phys) else cand_phys


def _grad_energy(cube: np.ndarray) -> dict[str, float]:
    gx = np.diff(cube, axis=0)
    gy = np.diff(cube, axis=1)
    gz = np.diff(cube, axis=2)
    lap = (
        cube[:-2, 1:-1, 1:-1]
        + cube[2:, 1:-1, 1:-1]
        + cube[1:-1, :-2, 1:-1]
        + cube[1:-1, 2:, 1:-1]
        + cube[1:-1, 1:-1, :-2]
        + cube[1:-1, 1:-1, 2:]
        - 6.0 * cube[1:-1, 1:-1, 1:-1]
    )
    return {
        "gx_std": float(gx.std()),
        "gy_std": float(gy.std()),
        "gz_std": float(gz.std()),
        "grad_l1_mean": float((np.abs(gx).mean() + np.abs(gy).mean() + np.abs(gz).mean()) / 3.0),
        "lap_std": float(lap.std()),
    }


def _extract_view(cube_ilxlh: np.ndarray, view: str, index: int) -> np.ndarray:
    if view == "depth":
        return cube_ilxlh[:, :, int(index)]
    if view == "inline":
        return cube_ilxlh[int(index), :, :].T
    if view == "xline":
        return cube_ilxlh[:, int(index), :].T
    raise ValueError(f"Unsupported view: {view}")


def _view_metrics(gt2: np.ndarray, pred2: np.ndarray) -> dict[str, float]:
    res = pred2 - gt2
    dr = float(gt2.max() - gt2.min())
    return {
        "mae": float(np.mean(np.abs(res))),
        "medae": float(np.median(np.abs(res))),
        "rmse": float(np.sqrt(np.mean(res ** 2))),
        "bias_mean": float(res.mean()),
        "r2": float(r2_score(gt2.ravel(), pred2.ravel())),
        "pcc": float(pearsonr(gt2.ravel(), pred2.ravel())[0]),
        "ssim": float(ssim(gt2, pred2, data_range=dr)) if dr > 0 else float("nan"),
        "vif": float(compute_vif_mscale(gt2.astype(np.float32), pred2.astype(np.float32))),
        "res_abs_q95": float(np.percentile(np.abs(res), 95)),
    }


def analyze_case_artifacts(case_dir: Path, data_flag: str) -> dict[str, Any]:
    pred_path = _find_unique_file(case_dir, "*_pred_AI.npy")
    pred_flat = np.load(pred_path).astype(np.float32, copy=False)
    if pred_flat.ndim == 3 and pred_flat.shape[1] == 1:
        pred_flat = pred_flat[:, 0, :]
    if pred_flat.ndim != 2:
        raise ValueError(f"Expected prediction array with shape [N,H], got {pred_flat.shape}")

    gt_cube = _load_gt_cube(data_flag)
    il, xl, h = gt_cube.shape
    stats = _load_stats_for_pred(pred_path, data_flag=data_flag)
    pred_flat = _maybe_denormalize_pred(pred_flat, stats, gt_cube)
    pred_cube = pred_flat.reshape(il, xl, h, order="C").astype(np.float32, copy=False)
    res_cube = pred_cube - gt_cube

    pred_grad = _grad_energy(pred_cube)
    gt_grad = _grad_energy(gt_cube)

    view_specs = {
        "depth_100": ("depth", REPRESENTATIVE_DEPTH),
        "inline_100": ("inline", REPRESENTATIVE_INLINE),
        "xline_50": ("xline", REPRESENTATIVE_XLINE),
    }
    view_report: dict[str, Any] = {}
    for key, (view_name, index) in view_specs.items():
        gt2 = _extract_view(gt_cube, view_name, index)
        pred2 = _extract_view(pred_cube, view_name, index)
        view_report[key] = _view_metrics(gt2, pred2)

    depth_mae = np.mean(np.abs(res_cube), axis=(0, 1))
    worst_depth = int(np.argmax(depth_mae))
    best_depth = int(np.argmin(depth_mae))

    return {
        "pred_std_ratio": float(pred_cube.std() / max(gt_cube.std(), 1e-8)),
        "pred_mean_bias": float((pred_cube - gt_cube).mean()),
        "pred_q01": float(np.percentile(pred_cube, 1)),
        "pred_q99": float(np.percentile(pred_cube, 99)),
        "gt_q01": float(np.percentile(gt_cube, 1)),
        "gt_q99": float(np.percentile(gt_cube, 99)),
        "q01_gap": float(np.percentile(pred_cube, 1) - np.percentile(gt_cube, 1)),
        "q99_gap": float(np.percentile(pred_cube, 99) - np.percentile(gt_cube, 99)),
        "grad_l1_ratio": float(pred_grad["grad_l1_mean"] / max(gt_grad["grad_l1_mean"], 1e-8)),
        "lap_ratio": float(pred_grad["lap_std"] / max(gt_grad["lap_std"], 1e-8)),
        "gx_ratio": float(pred_grad["gx_std"] / max(gt_grad["gx_std"], 1e-8)),
        "gy_ratio": float(pred_grad["gy_std"] / max(gt_grad["gy_std"], 1e-8)),
        "gz_ratio": float(pred_grad["gz_std"] / max(gt_grad["gz_std"], 1e-8)),
        "depth100_mae": float(view_report["depth_100"]["mae"]),
        "depth100_ssim": float(view_report["depth_100"]["ssim"]),
        "depth100_vif": float(view_report["depth_100"]["vif"]),
        "inline100_mae": float(view_report["inline_100"]["mae"]),
        "inline100_ssim": float(view_report["inline_100"]["ssim"]),
        "inline100_vif": float(view_report["inline_100"]["vif"]),
        "xline50_mae": float(view_report["xline_50"]["mae"]),
        "xline50_ssim": float(view_report["xline_50"]["ssim"]),
        "xline50_vif": float(view_report["xline_50"]["vif"]),
        "depthwise_mae_mean": float(depth_mae.mean()),
        "depthwise_mae_std": float(depth_mae.std()),
        "worst_depth": worst_depth,
        "worst_depth_mae": float(depth_mae[worst_depth]),
        "best_depth": best_depth,
        "best_depth_mae": float(depth_mae[best_depth]),
        "view_report": view_report,
        "pred_file": str(pred_path),
    }


def _build_summary_row(
    spec: TrialSpec,
    train_cfg: dict[str, Any],
    test_cfg: dict[str, Any],
    metrics: dict[str, Any] | None,
    visual_metrics: dict[str, Any] | None,
    status: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "trial_name": spec.name,
        "stage": spec.stage,
        "group": spec.group,
        "status": status,
        "note": spec.note,
        "base": spec.base,
        "run_id_suffix": train_cfg.get("run_id_suffix", ""),
        "epochs": train_cfg.get("epochs", ""),
        "backend": train_cfg.get("aniso_backend", ""),
        "iterative_R": int(bool(train_cfg.get("iterative_R", False))),
        "lambda_depth_grad": train_cfg.get("lambda_depth_grad", ""),
        "lambda_depth_hf": train_cfg.get("lambda_depth_hf", ""),
        "detail_gain": train_cfg.get("detail_gain", ""),
        "use_boundary_weight": int(bool(train_cfg.get("use_boundary_weight", False))),
        "boundary_weight_beta": train_cfg.get("boundary_weight_beta", ""),
        "train_noise_kind": train_cfg.get("train_noise_kind", ""),
        "train_noise_prob": train_cfg.get("train_noise_prob", ""),
        "train_noise_snr_db_choices": json.dumps(list(train_cfg.get("train_noise_snr_db_choices", []))),
        "r_channel_dropout_prob": train_cfg.get("r_channel_dropout_prob", ""),
        "agpe_lift_sigma": train_cfg.get("agpe_lift_sigma", ""),
        "agpe_long_weight": train_cfg.get("agpe_long_weight", ""),
        "agpe_well_soft_alpha": train_cfg.get("agpe_well_soft_alpha", ""),
        "aniso_closed_loop_conf_thresh": train_cfg.get("aniso_closed_loop_conf_thresh", ""),
        "agpe_rebuild_every": train_cfg.get("agpe_rebuild_every", ""),
        "R_update_every": train_cfg.get("R_update_every", ""),
        "test_noise_kind": test_cfg.get("test_noise_kind", ""),
    }
    if metrics:
        for key in REPORT_METRIC_KEYS:
            row[key] = metrics.get(key, "")
    if visual_metrics:
        for key in (
            "pred_std_ratio",
            "pred_mean_bias",
            "q01_gap",
            "q99_gap",
            "grad_l1_ratio",
            "lap_ratio",
            "gx_ratio",
            "gy_ratio",
            "gz_ratio",
            "depth100_mae",
            "depth100_ssim",
            "depth100_vif",
            "inline100_mae",
            "inline100_ssim",
            "inline100_vif",
            "xline50_mae",
            "xline50_ssim",
            "xline50_vif",
            "depthwise_mae_mean",
            "depthwise_mae_std",
            "worst_depth",
            "worst_depth_mae",
            "best_depth",
            "best_depth_mae",
        ):
            row[key] = visual_metrics.get(key, "")
    return row


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _selection_sort_key(row: dict[str, Any]) -> tuple:
    mae_gate = int(row.get("mae_gate_ok", 0))
    ssim_score = _safe_float(row.get("ssim", float("-inf")), float("-inf"))
    vif_score = _safe_float(row.get("vif", float("-inf")), float("-inf"))
    std_err = abs(_safe_float(row.get("pred_std_ratio", float("inf")), float("inf")) - 1.0)
    lap_err = abs(_safe_float(row.get("lap_ratio", float("inf")), float("inf")) - 1.0)
    mae = _safe_float(row.get("mae", float("inf")), float("inf"))
    medae = _safe_float(row.get("medae", float("inf")), float("inf"))
    return (-mae_gate, -ssim_score, -vif_score, std_err, lap_err, mae, medae)


def _rank_summary_rows(rows: list[dict[str, Any]], mae_tolerance: float, shortlist_k: int) -> list[dict[str, Any]]:
    ref_row = next((r for r in rows if r.get("trial_name") == "s0_ref_current_proposed" and "mae" in r), None)
    if ref_row is None:
        return rows
    ref_mae = _safe_float(ref_row.get("mae"), float("inf"))
    ref_ssim = _safe_float(ref_row.get("ssim"), float("nan"))
    ref_vif = _safe_float(ref_row.get("vif"), float("nan"))

    metric_rows = [r for r in rows if "mae" in r]
    for row in metric_rows:
        mae = _safe_float(row.get("mae"), float("inf"))
        row["mae_gate_ok"] = int(mae <= ref_mae * (1.0 + float(mae_tolerance)))
        row["delta_mae_vs_ref"] = mae - ref_mae
        row["delta_ssim_vs_ref"] = _safe_float(row.get("ssim"), float("nan")) - ref_ssim
        row["delta_vif_vs_ref"] = _safe_float(row.get("vif"), float("nan")) - ref_vif
        row["pred_std_ratio_err"] = abs(_safe_float(row.get("pred_std_ratio"), float("inf")) - 1.0)
        row["lap_ratio_err"] = abs(_safe_float(row.get("lap_ratio"), float("inf")) - 1.0)

    ranked_all = sorted(metric_rows, key=_selection_sort_key)
    for idx, row in enumerate(ranked_all, start=1):
        row["selection_rank_all"] = idx

    stage1_rows = [r for r in ranked_all if r.get("stage") == "stage1"]
    shortlisted = set()
    for idx, row in enumerate(stage1_rows, start=1):
        row["selection_rank_stage1"] = idx
    for row in stage1_rows[: max(int(shortlist_k), 0)]:
        row["shortlisted"] = 1
        shortlisted.add(str(row.get("trial_name")))
    for row in rows:
        if str(row.get("trial_name")) not in shortlisted and "shortlisted" not in row:
            row["shortlisted"] = 0
    return rows


def _save_case_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=True), encoding="utf-8")


def _resolve_trial_order(stage: str, explicit_trials: list[str] | None) -> list[str]:
    if explicit_trials:
        unknown = [name for name in explicit_trials if name not in TRIAL_SPECS]
        if unknown:
            raise ValueError(f"Unknown trial names: {unknown}")
        return explicit_trials
    if stage not in STAGE_ORDERS:
        raise ValueError(f"Unknown stage selection: {stage}")
    return list(STAGE_ORDERS[stage])


def _export_ranked_run_outputs(
    *,
    run_root: Path,
    summary_rows: list[dict[str, Any]],
    trial_registry: list[dict[str, Any]],
) -> None:
    registry_map = {str(item["trial_name"]): Path(str(item["case_dir"])) for item in trial_registry}

    for row in summary_rows:
        trial_name = str(row.get("trial_name"))
        case_dir = registry_map.get(trial_name)
        if case_dir is not None:
            _save_case_json(case_dir / "summary_row.json", row)

    summary_json = run_root / "trial_summary.json"
    summary_csv = run_root / "trial_summary.csv"
    summary_xlsx = run_root / "trial_summary.xlsx"
    shortlist_json = run_root / "stage1_shortlist.json"

    _save_case_json(summary_json, {"rows": summary_rows})
    _save_metrics_csv(summary_csv, summary_rows)
    _save_metrics_excel(summary_xlsx, summary_rows, sheet_name="trial_summary")

    stage1_shortlist = [
        {
            "trial_name": row.get("trial_name"),
            "selection_rank_stage1": row.get("selection_rank_stage1"),
            "mae": row.get("mae"),
            "ssim": row.get("ssim"),
            "vif": row.get("vif"),
            "pred_std_ratio": row.get("pred_std_ratio"),
            "lap_ratio": row.get("lap_ratio"),
            "note": row.get("note"),
        }
        for row in summary_rows
        if int(row.get("shortlisted", 0)) == 1
    ]
    _save_case_json(shortlist_json, {"shortlist": stage1_shortlist})

    required = [summary_json, summary_csv, summary_xlsx, shortlist_json]
    missing = [p.as_posix() for p in required if not p.is_file()]
    if missing:
        raise RuntimeError(f"Missing expected run-summary exports: {missing}")


def run_trials(
    *,
    stage: str,
    trial_names: list[str],
    mode: str,
    epochs_override: int | None,
    dry_run: bool,
    shortlist_k: int,
    mae_tolerance: float,
) -> None:
    train = None
    test = None
    if not dry_run:
        from train_multitask import train as _train
        from test_3D import test as _test

        train = _train
        test = _test

    results_root = Path("results")
    results_root.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_root = results_root / f"proposed_tuning_{run_stamp}"
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[RUN] proposed tuning results root: {run_root.as_posix()}")

    run_meta = {
        "stage": stage,
        "mode": mode,
        "epochs_override": epochs_override,
        "dry_run": dry_run,
        "shortlist_k": shortlist_k,
        "mae_tolerance": mae_tolerance,
        "trial_names": trial_names,
    }
    _save_case_json(run_root / "run_meta.json", run_meta)

    trial_registry = []
    summary_rows: list[dict[str, Any]] = []

    for idx, trial_name in enumerate(trial_names, start=1):
        train_cfg, test_cfg, spec = build_trial_configs(trial_name=trial_name, epochs_override=epochs_override)
        prefixes = _case_result_prefixes(train_cfg)
        case_dir = run_root / f"{idx:02d}_{trial_name}"
        case_dir.mkdir(parents=True, exist_ok=True)

        _save_case_json(case_dir / "train_config.json", train_cfg)
        _save_case_json(case_dir / "test_config.json", test_cfg)
        _save_case_json(case_dir / "trial_meta.json", {"trial": _jsonable(spec.__dict__)})

        trial_registry.append(
            {
                "order": idx,
                "trial_name": trial_name,
                "stage": spec.stage,
                "group": spec.group,
                "case_dir": str(case_dir),
                "suffix": spec.suffix,
                "note": spec.note,
            }
        )

        old_files = _list_case_result_files(results_root, prefixes)
        if old_files:
            old_dir = case_dir / "_preexisting"
            for path in old_files:
                _move_file_safe(path, old_dir)
            print(f"[ARCHIVE] moved {len(old_files)} preexisting files -> {old_dir.as_posix()}")

        print(
            f"\n[{idx}/{len(trial_names)}] trial={trial_name} "
            f"stage={spec.stage} group={spec.group} "
            f"backend={train_cfg.get('aniso_backend', 'grid')} "
            f"iterative_R={int(bool(train_cfg.get('iterative_R', False)))} "
            f"epochs={train_cfg.get('epochs')}"
        )
        print(f"[NOTE] {spec.note}")

        metrics: dict[str, Any] | None = None
        visual_metrics: dict[str, Any] | None = None
        status = "dry_run"

        if dry_run:
            print("[DRY-RUN] skipping train/test execution")
        else:
            if mode in ("train", "both"):
                print("[RUN] train_multitask.train(...)")
                assert train is not None
                train(train_cfg)

            if mode in ("test", "both"):
                print("[RUN] test_3D.test(...)")
                assert test is not None
                metrics = test(test_cfg)
                if isinstance(metrics, dict):
                    _save_case_json(case_dir / "test_metrics.json", metrics)
                    _save_metrics_excel(case_dir / "test_metrics.xlsx", [metrics], sheet_name="test_metrics")
                else:
                    metrics = None

            new_files = _list_case_result_files(results_root, prefixes)
            if new_files:
                for path in new_files:
                    _move_file_safe(path, case_dir)
                print(f"[SAVE] case artifacts saved to: {case_dir.as_posix()}")
            else:
                print(f"[SAVE][WARN] no case artifacts found for prefixes={prefixes}")

            try:
                visual_metrics = analyze_case_artifacts(case_dir=case_dir, data_flag=str(train_cfg["data_flag"]))
                _save_case_json(case_dir / "visual_metrics.json", visual_metrics)
            except Exception as exc:
                print(f"[ANALYZE][WARN] visual analysis failed for {trial_name}: {exc}")
                visual_metrics = None

            status = "tested" if mode in ("test", "both") else "trained"

        summary_row = _build_summary_row(
            spec=spec,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            metrics=metrics,
            visual_metrics=visual_metrics,
            status=status,
        )
        _save_case_json(case_dir / "summary_row.json", summary_row)
        summary_rows.append(summary_row)

    _save_case_json(run_root / "trial_registry.json", {"trials": trial_registry})

    summary_rows = _rank_summary_rows(summary_rows, mae_tolerance=mae_tolerance, shortlist_k=shortlist_k)
    _export_ranked_run_outputs(
        run_root=run_root,
        summary_rows=summary_rows,
        trial_registry=trial_registry,
    )
    print(f"[SAVE] summary -> {(run_root / 'trial_summary.xlsx').as_posix()}")
    print(f"[SAVE] shortlist -> {(run_root / 'stage1_shortlist.json').as_posix()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage 0 + Stage 1 Proposed tuning experiments with metric and visual summaries."
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["stage0", "stage1", "all"],
        help="Which stage set to run.",
    )
    parser.add_argument(
        "--trials",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit trial names. Overrides --stage selection.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["train", "test", "both"],
        help="Whether to train, test, or do both.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epoch override. When set, major schedules are scaled proportionally.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write configs and summary scaffolding without calling train/test.",
    )
    parser.add_argument(
        "--shortlist-k",
        type=int,
        default=4,
        help="How many Stage 1 trials to mark as shortlist candidates.",
    )
    parser.add_argument(
        "--mae-tolerance",
        type=float,
        default=0.02,
        help="MAE tolerance over the Stage 0 current Proposed reference for shortlist gating.",
    )
    parser.add_argument(
        "--list-trials",
        action="store_true",
        help="Print all available trial names and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_trials:
        for name in STAGE_ORDERS["all"]:
            spec = TRIAL_SPECS[name]
            print(
                f"{name:36s} "
                f"stage={spec.stage:6s} "
                f"group={spec.group:10s} "
                f"suffix={spec.suffix:16s} "
                f"note={spec.note}"
            )
        return

    trial_names = _resolve_trial_order(stage=str(args.stage), explicit_trials=args.trials)
    run_trials(
        stage=str(args.stage),
        trial_names=trial_names,
        mode=str(args.mode),
        epochs_override=args.epochs,
        dry_run=bool(args.dry_run),
        shortlist_k=int(args.shortlist_k),
        mae_tolerance=float(args.mae_tolerance),
    )


if __name__ == "__main__":
    main()
