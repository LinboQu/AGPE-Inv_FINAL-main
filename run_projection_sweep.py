import argparse
import copy
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from openpyxl import Workbook
from sklearn.metrics import r2_score

from setting import TCN1D_test_p, TCN1D_train_p
from test_3D import test
from train_multitask import train


CASE_PRESETS: Dict[str, Dict[str, object]] = {
    # Baseline: keep detail branch, disable boundary/depth constraints.
    "base_nobddep_dg015": {
        "aniso_backend": "skeleton_graph",
        "aniso_use_tensor_strength": True,
        "agpe_long_edges": True,
        "iterative_R": True,
        "agpe_cache_graph": True,
        "agpe_refine_graph": True,
        "agpe_rebuild_every": 50,
        "use_detail_branch": True,
        "detail_gain": 0.15,
        "use_boundary_weight": False,
        "lambda_depth_grad": 0.0,
        "lambda_depth_hf": 0.0,
        "use_depth_warm_schedule": False,
        "use_boundary_warm_schedule": False,
        "run_id_suffix": "_prj_base_nobddep_dg015",
    },
    # Step-1: lower detail gain first (anti-drift), still no boundary/depth.
    "base_nobddep_dg005": {
        "aniso_backend": "skeleton_graph",
        "aniso_use_tensor_strength": True,
        "agpe_long_edges": True,
        "iterative_R": True,
        "agpe_cache_graph": True,
        "agpe_refine_graph": True,
        "agpe_rebuild_every": 50,
        "use_detail_branch": True,
        "detail_gain": 0.05,
        "use_boundary_weight": False,
        "lambda_depth_grad": 0.0,
        "lambda_depth_hf": 0.0,
        "use_depth_warm_schedule": False,
        "use_boundary_warm_schedule": False,
        "run_id_suffix": "_prj_base_nobddep_dg005",
    },
    "base_nobddep_dg008": {
        "aniso_backend": "skeleton_graph",
        "aniso_use_tensor_strength": True,
        "agpe_long_edges": True,
        "iterative_R": True,
        "agpe_cache_graph": True,
        "agpe_refine_graph": True,
        "agpe_rebuild_every": 50,
        "use_detail_branch": True,
        "detail_gain": 0.08,
        "use_boundary_weight": False,
        "lambda_depth_grad": 0.0,
        "lambda_depth_hf": 0.0,
        "use_depth_warm_schedule": False,
        "use_boundary_warm_schedule": False,
        "run_id_suffix": "_prj_base_nobddep_dg008",
    },
    # Step-2: recover depth losses with late warm-up (epoch >= 300).
    "depth_l03_h01_late_dg006": {
        "aniso_backend": "skeleton_graph",
        "aniso_use_tensor_strength": True,
        "agpe_long_edges": True,
        "iterative_R": True,
        "agpe_cache_graph": True,
        "agpe_refine_graph": True,
        "agpe_rebuild_every": 50,
        "use_detail_branch": True,
        "detail_gain": 0.06,
        "use_boundary_weight": False,
        "lambda_depth_grad": 0.03,
        "lambda_depth_hf": 0.01,
        "use_depth_warm_schedule": True,
        "depth_warm_start_epoch": 300,
        "depth_warm_ramp_epochs": 200,
        "use_boundary_warm_schedule": False,
        "run_id_suffix": "_prj_depth_l03_h01_late_dg006",
    },
    "depth_l06_h03_late_dg006": {
        "aniso_backend": "skeleton_graph",
        "aniso_use_tensor_strength": True,
        "agpe_long_edges": True,
        "iterative_R": True,
        "agpe_cache_graph": True,
        "agpe_refine_graph": True,
        "agpe_rebuild_every": 50,
        "use_detail_branch": True,
        "detail_gain": 0.06,
        "use_boundary_weight": False,
        "lambda_depth_grad": 0.06,
        "lambda_depth_hf": 0.03,
        "use_depth_warm_schedule": True,
        "depth_warm_start_epoch": 300,
        "depth_warm_ramp_epochs": 200,
        "use_boundary_warm_schedule": False,
        "run_id_suffix": "_prj_depth_l06_h03_late_dg006",
    },
    # Step-3: recover boundary weighting with late and weak ramp.
    "depth_l03_h01_beta06_late_dg006": {
        "aniso_backend": "skeleton_graph",
        "aniso_use_tensor_strength": True,
        "agpe_long_edges": True,
        "iterative_R": True,
        "agpe_cache_graph": True,
        "agpe_refine_graph": True,
        "agpe_rebuild_every": 50,
        "use_detail_branch": True,
        "detail_gain": 0.06,
        "use_boundary_weight": True,
        "boundary_weight_width": 1,
        "boundary_weight_beta": 0.6,
        "use_boundary_warm_schedule": True,
        "boundary_beta_start": 0.0,
        "boundary_beta_end": 0.6,
        "boundary_warm_start_epoch": 300,
        "boundary_warm_ramp_epochs": 200,
        "lambda_depth_grad": 0.03,
        "lambda_depth_hf": 0.01,
        "use_depth_warm_schedule": True,
        "depth_warm_start_epoch": 300,
        "depth_warm_ramp_epochs": 200,
        "run_id_suffix": "_prj_depth_l03_h01_beta06_late_dg006",
    },
    "depth_l03_h01_beta10_late_dg006": {
        "aniso_backend": "skeleton_graph",
        "aniso_use_tensor_strength": True,
        "agpe_long_edges": True,
        "iterative_R": True,
        "agpe_cache_graph": True,
        "agpe_refine_graph": True,
        "agpe_rebuild_every": 50,
        "use_detail_branch": True,
        "detail_gain": 0.06,
        "use_boundary_weight": True,
        "boundary_weight_width": 1,
        "boundary_weight_beta": 1.0,
        "use_boundary_warm_schedule": True,
        "boundary_beta_start": 0.0,
        "boundary_beta_end": 1.0,
        "boundary_warm_start_epoch": 300,
        "boundary_warm_ramp_epochs": 200,
        "lambda_depth_grad": 0.03,
        "lambda_depth_hf": 0.01,
        "use_depth_warm_schedule": True,
        "depth_warm_start_epoch": 300,
        "depth_warm_ramp_epochs": 200,
        "run_id_suffix": "_prj_depth_l03_h01_beta10_late_dg006",
    },
    # -----------------------------
    # Round-2 matrix (focus on detail_gain=0.15)
    # -----------------------------
    "r2_dg015_base_nobddep": {
        "aniso_backend": "skeleton_graph",
        "aniso_use_tensor_strength": True,
        "agpe_long_edges": True,
        "iterative_R": True,
        "agpe_cache_graph": True,
        "agpe_refine_graph": True,
        "agpe_rebuild_every": 50,
        "use_detail_branch": True,
        "detail_gain": 0.15,
        "lambda_amp_anchor": 0.05,
        "use_boundary_weight": False,
        "lambda_depth_grad": 0.0,
        "lambda_depth_hf": 0.0,
        "use_depth_warm_schedule": False,
        "use_boundary_warm_schedule": False,
        "run_id_suffix": "_prj2_dg015_base_nobddep",
    },
    "r2_dg015_amp010_nobddep": {
        "aniso_backend": "skeleton_graph",
        "aniso_use_tensor_strength": True,
        "agpe_long_edges": True,
        "iterative_R": True,
        "agpe_cache_graph": True,
        "agpe_refine_graph": True,
        "agpe_rebuild_every": 50,
        "use_detail_branch": True,
        "detail_gain": 0.15,
        "lambda_amp_anchor": 0.10,
        "use_boundary_weight": False,
        "lambda_depth_grad": 0.0,
        "lambda_depth_hf": 0.0,
        "use_depth_warm_schedule": False,
        "use_boundary_warm_schedule": False,
        "run_id_suffix": "_prj2_dg015_amp010_nobddep",
    },
    "r2_dg015_amp020_nobddep": {
        "aniso_backend": "skeleton_graph",
        "aniso_use_tensor_strength": True,
        "agpe_long_edges": True,
        "iterative_R": True,
        "agpe_cache_graph": True,
        "agpe_refine_graph": True,
        "agpe_rebuild_every": 50,
        "use_detail_branch": True,
        "detail_gain": 0.15,
        "lambda_amp_anchor": 0.20,
        "use_boundary_weight": False,
        "lambda_depth_grad": 0.0,
        "lambda_depth_hf": 0.0,
        "use_depth_warm_schedule": False,
        "use_boundary_warm_schedule": False,
        "run_id_suffix": "_prj2_dg015_amp020_nobddep",
    },
    "r2_dg015_amp010_depth01003_late": {
        "aniso_backend": "skeleton_graph",
        "aniso_use_tensor_strength": True,
        "agpe_long_edges": True,
        "iterative_R": True,
        "agpe_cache_graph": True,
        "agpe_refine_graph": True,
        "agpe_rebuild_every": 50,
        "use_detail_branch": True,
        "detail_gain": 0.15,
        "lambda_amp_anchor": 0.10,
        "use_boundary_weight": False,
        "lambda_depth_grad": 0.01,
        "lambda_depth_hf": 0.003,
        "use_depth_warm_schedule": True,
        "depth_warm_start_epoch": 300,
        "depth_warm_ramp_epochs": 200,
        "use_boundary_warm_schedule": False,
        "run_id_suffix": "_prj2_dg015_amp010_depth01003_late",
    },
    "r2_dg015_amp010_depth01003_beta03_late": {
        "aniso_backend": "skeleton_graph",
        "aniso_use_tensor_strength": True,
        "agpe_long_edges": True,
        "iterative_R": True,
        "agpe_cache_graph": True,
        "agpe_refine_graph": True,
        "agpe_rebuild_every": 50,
        "use_detail_branch": True,
        "detail_gain": 0.15,
        "lambda_amp_anchor": 0.10,
        "use_boundary_weight": True,
        "boundary_weight_width": 1,
        "boundary_weight_beta": 0.3,
        "use_boundary_warm_schedule": True,
        "boundary_beta_start": 0.0,
        "boundary_beta_end": 0.3,
        "boundary_warm_start_epoch": 300,
        "boundary_warm_ramp_epochs": 200,
        "lambda_depth_grad": 0.01,
        "lambda_depth_hf": 0.003,
        "use_depth_warm_schedule": True,
        "depth_warm_start_epoch": 300,
        "depth_warm_ramp_epochs": 200,
        "run_id_suffix": "_prj2_dg015_amp010_depth01003_beta03_late",
    },
    "r2_dg015_amp010_depth01003_beta06_late": {
        "aniso_backend": "skeleton_graph",
        "aniso_use_tensor_strength": True,
        "agpe_long_edges": True,
        "iterative_R": True,
        "agpe_cache_graph": True,
        "agpe_refine_graph": True,
        "agpe_rebuild_every": 50,
        "use_detail_branch": True,
        "detail_gain": 0.15,
        "lambda_amp_anchor": 0.10,
        "use_boundary_weight": True,
        "boundary_weight_width": 1,
        "boundary_weight_beta": 0.6,
        "use_boundary_warm_schedule": True,
        "boundary_beta_start": 0.0,
        "boundary_beta_end": 0.6,
        "boundary_warm_start_epoch": 300,
        "boundary_warm_ramp_epochs": 200,
        "lambda_depth_grad": 0.01,
        "lambda_depth_hf": 0.003,
        "use_depth_warm_schedule": True,
        "depth_warm_start_epoch": 300,
        "depth_warm_ramp_epochs": 200,
        "run_id_suffix": "_prj2_dg015_amp010_depth01003_beta06_late",
    },
}

ROUND2_DEFAULT_CASES: List[str] = [
    "r2_dg015_base_nobddep",
    "r2_dg015_amp010_nobddep",
    "r2_dg015_amp020_nobddep",
    "r2_dg015_amp010_depth01003_late",
    "r2_dg015_amp010_depth01003_beta03_late",
    "r2_dg015_amp010_depth01003_beta06_late",
]

# -----------------------------
# Round-3 matrix (conservative recovery around the only stable baseline)
# -----------------------------
CASE_PRESETS.update(
    {
        "r3_base_dg015_amp005_nobddep": {
            "aniso_backend": "skeleton_graph",
            "aniso_use_tensor_strength": True,
            "agpe_long_edges": True,
            "iterative_R": True,
            "agpe_cache_graph": True,
            "agpe_refine_graph": True,
            "agpe_rebuild_every": 50,
            "use_detail_branch": True,
            "detail_gain": 0.15,
            "lambda_amp_anchor": 0.05,
            "use_boundary_weight": False,
            "lambda_depth_grad": 0.0,
            "lambda_depth_hf": 0.0,
            "use_depth_warm_schedule": False,
            "use_boundary_warm_schedule": False,
            "run_id_suffix": "_prj3_base_dg015_amp005_nobddep",
        },
        "r3_dg015_amp007_nobddep": {
            "aniso_backend": "skeleton_graph",
            "aniso_use_tensor_strength": True,
            "agpe_long_edges": True,
            "iterative_R": True,
            "agpe_cache_graph": True,
            "agpe_refine_graph": True,
            "agpe_rebuild_every": 50,
            "use_detail_branch": True,
            "detail_gain": 0.15,
            "lambda_amp_anchor": 0.07,
            "use_boundary_weight": False,
            "lambda_depth_grad": 0.0,
            "lambda_depth_hf": 0.0,
            "use_depth_warm_schedule": False,
            "use_boundary_warm_schedule": False,
            "run_id_suffix": "_prj3_dg015_amp007_nobddep",
        },
        "r3_depth005001_late_amp005": {
            "aniso_backend": "skeleton_graph",
            "aniso_use_tensor_strength": True,
            "agpe_long_edges": True,
            "iterative_R": True,
            "agpe_cache_graph": True,
            "agpe_refine_graph": True,
            "agpe_rebuild_every": 50,
            "use_detail_branch": True,
            "detail_gain": 0.15,
            "lambda_amp_anchor": 0.05,
            "use_boundary_weight": False,
            "lambda_depth_grad": 0.005,
            "lambda_depth_hf": 0.001,
            "use_depth_warm_schedule": True,
            "depth_warm_start_epoch": 500,
            "depth_warm_ramp_epochs": 300,
            "use_boundary_warm_schedule": False,
            "run_id_suffix": "_prj3_depth005001_late_amp005",
        },
        "r3_depth005001_beta02_late_amp005": {
            "aniso_backend": "skeleton_graph",
            "aniso_use_tensor_strength": True,
            "agpe_long_edges": True,
            "iterative_R": True,
            "agpe_cache_graph": True,
            "agpe_refine_graph": True,
            "agpe_rebuild_every": 50,
            "use_detail_branch": True,
            "detail_gain": 0.15,
            "lambda_amp_anchor": 0.05,
            "use_boundary_weight": True,
            "boundary_weight_width": 1,
            "boundary_weight_beta": 0.2,
            "use_boundary_warm_schedule": True,
            "boundary_beta_start": 0.0,
            "boundary_beta_end": 0.2,
            "boundary_warm_start_epoch": 500,
            "boundary_warm_ramp_epochs": 300,
            "lambda_depth_grad": 0.005,
            "lambda_depth_hf": 0.001,
            "use_depth_warm_schedule": True,
            "depth_warm_start_epoch": 500,
            "depth_warm_ramp_epochs": 300,
            "run_id_suffix": "_prj3_depth005001_beta02_late_amp005",
        },
        "r3_depth005001_beta02_late_amp005_ws_tight": {
            "aniso_backend": "skeleton_graph",
            "aniso_use_tensor_strength": True,
            "agpe_long_edges": True,
            "iterative_R": True,
            "agpe_cache_graph": True,
            "agpe_refine_graph": True,
            "agpe_rebuild_every": 50,
            "use_detail_branch": True,
            "detail_gain": 0.15,
            "lambda_amp_anchor": 0.05,
            "use_boundary_weight": True,
            "boundary_weight_width": 1,
            "boundary_weight_beta": 0.2,
            "use_boundary_warm_schedule": True,
            "boundary_beta_start": 0.0,
            "boundary_beta_end": 0.2,
            "boundary_warm_start_epoch": 500,
            "boundary_warm_ramp_epochs": 300,
            "lambda_depth_grad": 0.005,
            "lambda_depth_hf": 0.001,
            "use_depth_warm_schedule": True,
            "depth_warm_start_epoch": 500,
            "depth_warm_ramp_epochs": 300,
            # Weak-supervision tightening
            "ws_every": 4,
            "ws_max_batches": 80,
            "ws_max_batches_stageA": 30,
            "ws_every_late": 12,
            "ws_max_batches_late": 8,
            "run_id_suffix": "_prj3_depth005001_beta02_late_amp005_ws_tight",
        },
    }
)

ROUND3_DEFAULT_CASES: List[str] = [
    "r3_base_dg015_amp005_nobddep",
    "r3_dg015_amp007_nobddep",
    "r3_depth005001_late_amp005",
    "r3_depth005001_beta02_late_amp005",
    "r3_depth005001_beta02_late_amp005_ws_tight",
]


# -----------------------------
# Round-4 matrix (centered on the current best case from round-3)
# -----------------------------
CASE_PRESETS.update(
    {
        # Anchor: best from round-3.
        "r4_anchor_depth005001_late_amp005": {
            "aniso_backend": "skeleton_graph",
            "aniso_use_tensor_strength": True,
            "agpe_long_edges": True,
            "iterative_R": True,
            "agpe_cache_graph": True,
            "agpe_refine_graph": True,
            "agpe_rebuild_every": 50,
            "use_detail_branch": True,
            "detail_gain": 0.15,
            "lambda_amp_anchor": 0.05,
            "use_boundary_weight": False,
            "lambda_depth_grad": 0.005,
            "lambda_depth_hf": 0.001,
            "use_depth_warm_schedule": True,
            "depth_warm_start_epoch": 500,
            "depth_warm_ramp_epochs": 300,
            "use_boundary_warm_schedule": False,
            "run_id_suffix": "_prj4_anchor_depth005001_late_amp005",
        },
        # Test WS tightening effect alone on top of anchor.
        "r4_anchor_ws_tight_nobw": {
            "aniso_backend": "skeleton_graph",
            "aniso_use_tensor_strength": True,
            "agpe_long_edges": True,
            "iterative_R": True,
            "agpe_cache_graph": True,
            "agpe_refine_graph": True,
            "agpe_rebuild_every": 50,
            "use_detail_branch": True,
            "detail_gain": 0.15,
            "lambda_amp_anchor": 0.05,
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
            "run_id_suffix": "_prj4_anchor_ws_tight_nobw",
        },
        # Recover boundary softly under WS-tight condition.
        "r4_anchor_ws_tight_beta010": {
            "aniso_backend": "skeleton_graph",
            "aniso_use_tensor_strength": True,
            "agpe_long_edges": True,
            "iterative_R": True,
            "agpe_cache_graph": True,
            "agpe_refine_graph": True,
            "agpe_rebuild_every": 50,
            "use_detail_branch": True,
            "detail_gain": 0.15,
            "lambda_amp_anchor": 0.05,
            "use_boundary_weight": True,
            "boundary_weight_width": 1,
            "boundary_weight_beta": 0.10,
            "use_boundary_warm_schedule": True,
            "boundary_beta_start": 0.0,
            "boundary_beta_end": 0.10,
            "boundary_warm_start_epoch": 500,
            "boundary_warm_ramp_epochs": 300,
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
            "run_id_suffix": "_prj4_anchor_ws_tight_beta010",
        },
        "r4_anchor_ws_tight_beta015": {
            "aniso_backend": "skeleton_graph",
            "aniso_use_tensor_strength": True,
            "agpe_long_edges": True,
            "iterative_R": True,
            "agpe_cache_graph": True,
            "agpe_refine_graph": True,
            "agpe_rebuild_every": 50,
            "use_detail_branch": True,
            "detail_gain": 0.15,
            "lambda_amp_anchor": 0.05,
            "use_boundary_weight": True,
            "boundary_weight_width": 1,
            "boundary_weight_beta": 0.15,
            "use_boundary_warm_schedule": True,
            "boundary_beta_start": 0.0,
            "boundary_beta_end": 0.15,
            "boundary_warm_start_epoch": 500,
            "boundary_warm_ramp_epochs": 300,
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
            "run_id_suffix": "_prj4_anchor_ws_tight_beta015",
        },
        "r4_anchor_ws_tight_beta020": {
            "aniso_backend": "skeleton_graph",
            "aniso_use_tensor_strength": True,
            "agpe_long_edges": True,
            "iterative_R": True,
            "agpe_cache_graph": True,
            "agpe_refine_graph": True,
            "agpe_rebuild_every": 50,
            "use_detail_branch": True,
            "detail_gain": 0.15,
            "lambda_amp_anchor": 0.05,
            "use_boundary_weight": True,
            "boundary_weight_width": 1,
            "boundary_weight_beta": 0.20,
            "use_boundary_warm_schedule": True,
            "boundary_beta_start": 0.0,
            "boundary_beta_end": 0.20,
            "boundary_warm_start_epoch": 500,
            "boundary_warm_ramp_epochs": 300,
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
            "run_id_suffix": "_prj4_anchor_ws_tight_beta020",
        },
    }
)

ROUND4_REFERENCE_CASE: str = "r4_anchor_depth005001_late_amp005"

# Formal comparison matrix: keep WS-tight as mandatory baseline and variants.
ROUND4_DEFAULT_CASES: List[str] = [
    "r4_anchor_ws_tight_nobw",
    "r4_anchor_ws_tight_beta010",
    "r4_anchor_ws_tight_beta015",
    "r4_anchor_ws_tight_beta020",
]


def _save_metrics_excel(xlsx_path: Path, rows: list[dict], sheet_name: str = "metrics") -> None:
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


def _resolve_run_id(train_cfg: dict) -> str:
    run_id_base = f"{train_cfg['model_name']}_{train_cfg['Forward_model']}_{train_cfg['Facies_model']}"
    suffix = str(train_cfg.get("run_id_suffix", "") or "")
    return run_id_base if (suffix == "" or run_id_base.endswith(suffix)) else f"{run_id_base}{suffix}"


def _case_result_prefixes(train_cfg: dict) -> tuple[str, str]:
    run_id = _resolve_run_id(train_cfg)
    data_flag = str(train_cfg["data_flag"])
    return f"{run_id}_s_uns_{data_flag}", f"{run_id}_{data_flag}"


def _list_case_result_files(results_root: Path, prefixes: tuple[str, str]) -> list[Path]:
    out: list[Path] = []
    for p in results_root.iterdir():
        if p.is_file() and (p.name.startswith(prefixes[0]) or p.name.startswith(prefixes[1])):
            out.append(p)
    return sorted(out)


def _move_file_safe(src: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if not dst.exists():
        src.replace(dst)
        return
    idx = 1
    while True:
        cand = dst_dir / f"{src.stem}__dup{idx}{src.suffix}"
        if not cand.exists():
            src.replace(cand)
            return
        idx += 1


def _collect_health_metrics(case_dir: Path) -> dict:
    pred_files = sorted(case_dir.glob("*_pred_AI.npy"))
    true_files = sorted(case_dir.glob("*_true_AI.npy"))
    boot_files = sorted(case_dir.glob("*_pred_AI_bootstrap.npy"))
    if not pred_files or not true_files:
        return {}
    pred = np.load(pred_files[0]).reshape(-1).astype(np.float64)
    true = np.load(true_files[0]).reshape(-1).astype(np.float64)
    out = {
        "pred_mean": float(pred.mean()),
        "pred_std": float(pred.std()),
        "pred_min": float(pred.min()),
        "pred_max": float(pred.max()),
    }
    if boot_files:
        boot = np.load(boot_files[0]).reshape(-1).astype(np.float64)
        r2_boot = float(r2_score(true, boot))
        pcc_boot = float(np.corrcoef(true, boot)[0, 1])
        r2_final = float(r2_score(true, pred))
        pcc_final = float(np.corrcoef(true, pred)[0, 1])
        out.update(
            {
                "r2_bootstrap": r2_boot,
                "pcc_bootstrap": pcc_boot,
                "r2_final_from_npy": r2_final,
                "pcc_final_from_npy": pcc_final,
                "r2_final_minus_bootstrap": float(r2_final - r2_boot),
            }
        )
    return out


def _health_gate(row: dict) -> dict:
    pred_mean = abs(float(row.get("pred_mean", 0.0)))
    pred_std = float(row.get("pred_std", 0.0))
    r2_gap = float(row.get("r2_final_minus_bootstrap", 0.0))
    gate_mean_ok = pred_mean <= 0.35
    gate_std_ok = 0.80 <= pred_std <= 1.15
    gate_gap_ok = r2_gap >= -0.01
    return {
        "gate_pred_mean_ok": bool(gate_mean_ok),
        "gate_pred_std_ok": bool(gate_std_ok),
        "gate_bootstrap_gap_ok": bool(gate_gap_ok),
        "gate_overall": bool(gate_mean_ok and gate_std_ok and gate_gap_ok),
    }


def build_case_configs(case_name: str, epochs_override: int | None) -> tuple[dict, dict]:
    if case_name not in CASE_PRESETS:
        raise ValueError(f"Unknown case: {case_name}")
    preset = CASE_PRESETS[case_name]
    train_cfg = copy.deepcopy(TCN1D_train_p)
    test_cfg = copy.deepcopy(TCN1D_test_p)
    train_cfg.update(preset)
    test_cfg.update(preset)
    run_id_base = f"{train_cfg['model_name']}_{train_cfg['Forward_model']}_{train_cfg['Facies_model']}"
    test_cfg["run_id"] = run_id_base
    test_cfg["model_name"] = f"{run_id_base}_s_uns"
    if epochs_override is not None:
        train_cfg["epochs"] = int(epochs_override)
    return train_cfg, test_cfg


def run_cases(cases: List[str], mode: str, epochs_override: int | None) -> None:
    results_root = Path("results")
    results_root.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_root = results_root / f"projection_sweep_{run_stamp}"
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[RUN] projection sweep root: {run_root.as_posix()}")
    summary_rows: list[dict] = []

    for idx, case in enumerate(cases, start=1):
        train_cfg, test_cfg = build_case_configs(case, epochs_override=epochs_override)
        prefixes = _case_result_prefixes(train_cfg)
        case_dir = run_root / f"{idx:02d}_{case}"
        case_dir.mkdir(parents=True, exist_ok=True)

        old_files = _list_case_result_files(results_root, prefixes)
        if old_files:
            old_dir = case_dir / "_preexisting"
            for p in old_files:
                _move_file_safe(p, old_dir)

        print(
            f"\n[{idx}/{len(cases)}] case={case} "
            f"detail_gain={float(train_cfg.get('detail_gain', 0.0)):.3f} "
            f"lambda_amp_anchor={float(train_cfg.get('lambda_amp_anchor', 0.0)):.3f} "
            f"depth=({float(train_cfg.get('lambda_depth_grad', 0.0)):.3f},{float(train_cfg.get('lambda_depth_hf', 0.0)):.3f}) "
            f"boundary={int(bool(train_cfg.get('use_boundary_weight', False)))}"
        )

        if mode in ("train", "both"):
            train(train_cfg)
        if mode in ("test", "both"):
            metrics = test(test_cfg)
            if isinstance(metrics, dict):
                row = {
                    "case_dir": case_dir.name,
                    "case": case,
                    "run_id": _resolve_run_id(train_cfg),
                    "iterative_R": bool(train_cfg.get("iterative_R", True)),
                    "detail_gain": float(train_cfg.get("detail_gain", 0.0)),
                    "use_detail_branch": bool(train_cfg.get("use_detail_branch", True)),
                    "lambda_amp_anchor": float(train_cfg.get("lambda_amp_anchor", 0.0)),
                    "use_boundary_weight": bool(train_cfg.get("use_boundary_weight", False)),
                    "boundary_beta_start": float(train_cfg.get("boundary_beta_start", train_cfg.get("boundary_weight_beta", 0.0))),
                    "boundary_beta_end": float(train_cfg.get("boundary_beta_end", train_cfg.get("boundary_weight_beta", 0.0))),
                    "boundary_warm_start_epoch": int(train_cfg.get("boundary_warm_start_epoch", 0)),
                    "boundary_warm_ramp_epochs": int(train_cfg.get("boundary_warm_ramp_epochs", 1)),
                    "lambda_depth_grad": float(train_cfg.get("lambda_depth_grad", 0.0)),
                    "lambda_depth_hf": float(train_cfg.get("lambda_depth_hf", 0.0)),
                    "depth_warm_start_epoch": int(train_cfg.get("depth_warm_start_epoch", 0)),
                    "depth_warm_ramp_epochs": int(train_cfg.get("depth_warm_ramp_epochs", 1)),
                    **metrics,
                }
                summary_rows.append(row)

        new_files = _list_case_result_files(results_root, prefixes)
        for p in new_files:
            _move_file_safe(p, case_dir)

        if mode in ("test", "both") and summary_rows:
            health = _collect_health_metrics(case_dir)
            if health:
                summary_rows[-1].update(health)
                summary_rows[-1].update(_health_gate(summary_rows[-1]))
                _save_metrics_excel(case_dir / "test_metrics.xlsx", [summary_rows[-1]], sheet_name="test_metrics")

    if summary_rows:
        _save_metrics_excel(run_root / "projection_sweep_summary.xlsx", summary_rows, sheet_name="summary")
        print(f"[SAVE] summary -> {(run_root / 'projection_sweep_summary.xlsx').as_posix()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Projection-control sweep for skeleton_graph anti-drift settings.")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=ROUND4_DEFAULT_CASES,
        choices=sorted(CASE_PRESETS.keys()),
        help="Sweep cases to run (default: round-4 formal ws_tight matrix).",
    )
    parser.add_argument("--mode", default="both", choices=["train", "test", "both"])
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_cases(cases=args.cases, mode=args.mode, epochs_override=args.epochs)
