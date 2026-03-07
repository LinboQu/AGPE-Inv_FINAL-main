from __future__ import annotations

import argparse
import copy
import glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import run_projection_sweep as base


RESULTS_ROOT = Path("results")

# 3-case deep-loss matrix:
# - locked formal baseline
# - trial A: deep-oriented depth loss (late + weak)
# - trial B: trial A + AI-only boundary weighting (late + weak)
CASE_BASE = "r51_nobw_rb100_cf060_ls23"
CASE_TRIAL_DEPTH = "r51_nobw_rb100_cf060_ls23_deepdep"
CASE_TRIAL_AIBW = "r51_nobw_rb100_cf060_ls23_deepdep_aibw"

# Optional diagnostic branch (kept for backward compatibility).
CASE_AIONLY = "r51_beta020_aionly_rb100"


def _list_projection_dirs() -> List[Path]:
    paths = [Path(p) for p in glob.glob(str(RESULTS_ROOT / "projection_sweep_*")) if Path(p).is_dir()]
    paths.sort(key=lambda p: p.stat().st_mtime)
    return paths


def _latest_new_run_root(before: List[Path]) -> Path:
    before_set = {str(p.resolve()) for p in before}
    after = _list_projection_dirs()
    new_dirs = [p for p in after if str(p.resolve()) not in before_set]
    if new_dirs:
        return new_dirs[-1]
    if after:
        return after[-1]
    raise RuntimeError("No projection_sweep_* result directory found under results/.")


def _build_r51_cases() -> Dict[str, dict]:
    """Register R51 cases on top of run_projection_sweep presets."""
    cases: Dict[str, dict] = {}

    # Locked formal baseline from current best stable line.
    base_cfg = copy.deepcopy(base.CASE_PRESETS["r5_anchor_ws_tight_nobw_cache200"])
    base_cfg["agpe_rebuild_every"] = 100
    base_cfg["R_update_every"] = 50
    base_cfg["aniso_closed_loop_conf_thresh"] = 0.60
    base_cfg["agpe_lift_sigma"] = 2.3
    base_cfg["run_id_suffix"] = f"{base_cfg['run_id_suffix']}_r51_rb100_cf060_ls23"
    base.CASE_PRESETS[CASE_BASE] = base_cfg
    cases[CASE_BASE] = base_cfg

    # Trial A: deep-oriented depth regularization (late + weak), no boundary reweighting.
    trial_depth = copy.deepcopy(base_cfg)
    trial_depth["lambda_depth_grad"] = 0.008
    trial_depth["lambda_depth_hf"] = 0.002
    trial_depth["use_depth_warm_schedule"] = True
    trial_depth["depth_warm_start_epoch"] = 650
    trial_depth["depth_warm_ramp_epochs"] = 250
    trial_depth["use_boundary_weight"] = False
    trial_depth["run_id_suffix"] = f"{trial_depth['run_id_suffix']}_deepdep"
    base.CASE_PRESETS[CASE_TRIAL_DEPTH] = trial_depth
    cases[CASE_TRIAL_DEPTH] = trial_depth

    # Trial B: Trial A + AI-only boundary weighting (late + weak), keep detail unweighted.
    trial_aibw = copy.deepcopy(trial_depth)
    trial_aibw["use_boundary_weight"] = True
    trial_aibw["boundary_weight_width"] = 1
    trial_aibw["boundary_weight_apply_ai"] = True
    trial_aibw["boundary_weight_apply_detail"] = False
    trial_aibw["boundary_weight_apply_facies"] = False
    trial_aibw["use_boundary_warm_schedule"] = True
    trial_aibw["boundary_beta_start"] = 0.0
    trial_aibw["boundary_beta_end"] = 0.15
    trial_aibw["boundary_warm_start_epoch"] = 650
    trial_aibw["boundary_warm_ramp_epochs"] = 250
    trial_aibw["run_id_suffix"] = f"{trial_aibw['run_id_suffix']}_aibw"
    base.CASE_PRESETS[CASE_TRIAL_AIBW] = trial_aibw
    cases[CASE_TRIAL_AIBW] = trial_aibw

    # Diagnostic branch (not part of formal default).
    aionly_cfg = copy.deepcopy(base.CASE_PRESETS["r5_anchor_ws_tight_beta020_aionly_cache200"])
    aionly_cfg["agpe_rebuild_every"] = 100
    aionly_cfg["R_update_every"] = 50
    aionly_cfg["run_id_suffix"] = f"{aionly_cfg['run_id_suffix']}_r51_rb100"
    aionly_cfg["boundary_weight_apply_ai"] = True
    aionly_cfg["boundary_weight_apply_detail"] = False
    aionly_cfg["boundary_weight_apply_facies"] = False
    base.CASE_PRESETS[CASE_AIONLY] = aionly_cfg
    cases[CASE_AIONLY] = aionly_cfg

    return cases


def _parse_args(choices: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "R51 runner: deep-loss 3-case matrix (baseline + deepdep + deepdep_aibw), "
            "auto-select by R2 + shadow_area_mean + shadow_area_d160 and run repeats."
        )
    )
    parser.add_argument("--mode", default="both", choices=["train", "test", "both"])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument(
        "--suite",
        default="full",
        choices=[
            "full",
            "confirm3",
            "repeat_base",
            "repeat_depth",
            "repeat_aibw",
            "repeat_longw",
            "repeat_well",
            "repeat_candidate",
            "compare_aionly",
        ],
        help=(
            "full=confirm3 then auto-select and repeat; "
            "confirm3=single 3-case confirmation only; "
            "repeat_base/repeat_depth/repeat_aibw=forced repeats; "
            "repeat_longw/repeat_well/repeat_candidate kept as backward aliases; "
            "compare_aionly=baseline vs aionly diagnostic"
        ),
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        choices=sorted(choices),
        help="Optional manual case list override (single run root).",
    )
    return parser.parse_args()


def _run_once(cases: List[str], mode: str, epochs: int) -> Path:
    before = _list_projection_dirs()
    base.run_cases(cases=cases, mode=mode, epochs_override=epochs)
    return _latest_new_run_root(before)


def _read_summary(run_root: Path) -> pd.DataFrame:
    xlsx = run_root / "projection_sweep_summary.xlsx"
    if not xlsx.is_file():
        raise FileNotFoundError(f"Missing summary file: {xlsx}")
    return pd.read_excel(xlsx)


def _as_float(v) -> float:
    if pd.isna(v):
        return float("nan")
    return float(v)


def _row(df: pd.DataFrame, case_name: str) -> pd.Series:
    sub = df[df["case"] == case_name]
    if len(sub) == 0:
        raise RuntimeError(f"Missing case row in summary: {case_name}")
    return sub.iloc[0]


def _attach_baseline_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Attach/overwrite baseline-referenced deltas for robust gating and reporting."""
    out = df.copy()
    if "case" not in out.columns:
        return out
    if CASE_BASE not in set(out["case"].astype(str).tolist()):
        return out

    base_r = _row(out, CASE_BASE)
    base_r2 = _as_float(base_r.get("r2", np.nan))
    base_shadow = _as_float(base_r.get("shadow_area_mean", np.nan))
    base_d160 = _as_float(base_r.get("shadow_area_d160", np.nan))

    if "r2" in out.columns:
        out["delta_r2_vs_nobw"] = out["r2"].astype(float) - base_r2
    if "shadow_area_mean" in out.columns:
        out["delta_shadow_area_vs_nobw"] = out["shadow_area_mean"].astype(float) - base_shadow
    if "shadow_area_d160" in out.columns:
        out["delta_shadow_area_d160_vs_nobw"] = out["shadow_area_d160"].astype(float) - base_d160

    return out


def _print_brief_table(df: pd.DataFrame) -> None:
    cols = [
        c
        for c in [
            "case",
            "r2",
            "pcc",
            "shadow_area_d160",
            "shadow_area_mean",
            "delta_r2_vs_nobw",
            "delta_shadow_area_vs_nobw",
            "delta_shadow_area_d160_vs_nobw",
            "gate_r2_shadow_ok",
            "keep_for_formal",
        ]
        if c in df.columns
    ]
    if cols:
        print(df[cols].to_string(index=False))


def _confirm_three_cases(df: pd.DataFrame) -> Tuple[str, dict]:
    base_r = _row(df, CASE_BASE)
    base_r2 = _as_float(base_r.get("r2"))
    base_shadow = _as_float(base_r.get("shadow_area_mean"))
    base_d160 = _as_float(base_r.get("shadow_area_d160", np.nan))

    candidates = [CASE_TRIAL_DEPTH, CASE_TRIAL_AIBW]
    trial_details = {}
    passed = []

    for case_name in candidates:
        r = _row(df, case_name)
        r2 = _as_float(r.get("r2"))
        shadow = _as_float(r.get("shadow_area_mean"))
        d160 = _as_float(r.get("shadow_area_d160", np.nan))
        delta_r2 = r2 - base_r2
        delta_shadow = shadow - base_shadow
        delta_d160 = d160 - base_d160
        pass_gate = bool((delta_r2 >= 0.0) and (delta_shadow <= 0.0) and (delta_d160 <= 0.0))
        trial_details[case_name] = {
            "r2": r2,
            "shadow_area_mean": shadow,
            "shadow_area_d160": d160,
            "delta_r2_vs_base": delta_r2,
            "delta_shadow_vs_base": delta_shadow,
            "delta_d160_vs_base": delta_d160,
            "pass_gate": pass_gate,
        }
        if pass_gate:
            passed.append((case_name, r2, shadow, d160))

    # tie-break: higher r2 first, then lower shadow mean, then lower d160
    if passed:
        passed.sort(key=lambda x: (-x[1], x[2], x[3]))
        selected = passed[0][0]
    else:
        selected = CASE_BASE

    detail = {
        "base": {
            "r2": base_r2,
            "shadow_area_mean": base_shadow,
            "shadow_area_d160": base_d160,
        },
        "trials": trial_details,
        "selected": selected,
        "any_pass_gate": bool(len(passed) > 0),
    }
    return selected, detail


def _run_confirm3(mode: str, epochs: int) -> Tuple[Path, str, dict]:
    cases = [CASE_BASE, CASE_TRIAL_DEPTH, CASE_TRIAL_AIBW]
    print(f"[R51] confirm3 cases={cases} mode={mode} epochs={epochs}")
    run_root = _run_once(cases=cases, mode=mode, epochs=epochs)
    df = _attach_baseline_deltas(_read_summary(run_root))
    _print_brief_table(df)
    selected, detail = _confirm_three_cases(df)
    print(f"[R51] confirm3 run root: {run_root.as_posix()}")
    print(f"[R51] confirm3 detail: {detail}")
    print(f"[R51] selected formal case: {selected}")
    return run_root, selected, detail


def _run_repeats(case_name: str, mode: str, epochs: int, repeat_runs: int) -> List[Path]:
    roots: List[Path] = []
    for i in range(int(repeat_runs)):
        print(f"[R51] repeat {i + 1}/{repeat_runs} case={case_name}")
        root = _run_once(cases=[case_name], mode=mode, epochs=epochs)
        roots.append(root)
    return roots


def _print_repeat_summary(roots: List[Path], case_name: str) -> None:
    rows = []
    for root in roots:
        try:
            df = _attach_baseline_deltas(_read_summary(root))
        except Exception:
            continue
        sub = df[df["case"] == case_name]
        if len(sub) == 0:
            continue
        r = sub.iloc[0]
        rows.append(
            {
                "run_root": root.name,
                "r2": _as_float(r.get("r2", np.nan)),
                "pcc": _as_float(r.get("pcc", np.nan)),
                "shadow_area_d160": _as_float(r.get("shadow_area_d160", np.nan)),
                "shadow_area_mean": _as_float(r.get("shadow_area_mean", np.nan)),
                "pred_mean": _as_float(r.get("pred_mean", np.nan)),
                "pred_std": _as_float(r.get("pred_std", np.nan)),
                "delta_r2_vs_nobw": _as_float(r.get("delta_r2_vs_nobw", np.nan)),
                "delta_shadow_area_vs_nobw": _as_float(r.get("delta_shadow_area_vs_nobw", np.nan)),
                "delta_shadow_area_d160_vs_nobw": _as_float(r.get("delta_shadow_area_d160_vs_nobw", np.nan)),
            }
        )

    if not rows:
        print("[R51] repeat summary unavailable.")
        return

    df = pd.DataFrame(rows)
    print("[R51] repeat metrics:")
    print(df.to_string(index=False))
    stats = df[
        [
            "r2",
            "pcc",
            "shadow_area_d160",
            "shadow_area_mean",
            "pred_mean",
            "pred_std",
            "delta_r2_vs_nobw",
            "delta_shadow_area_vs_nobw",
            "delta_shadow_area_d160_vs_nobw",
        ]
    ].agg(["mean", "std"])
    print("[R51] repeat mean/std:")
    print(stats.to_string())


def _run_compare_aionly(mode: str, epochs: int) -> Path:
    cases = [CASE_BASE, CASE_AIONLY]
    print(f"[R51] compare_aionly cases={cases} mode={mode} epochs={epochs}")
    run_root = _run_once(cases=cases, mode=mode, epochs=epochs)
    df = _attach_baseline_deltas(_read_summary(run_root))
    _print_brief_table(df)
    return run_root


def main() -> None:
    r51_cases = _build_r51_cases()
    args = _parse_args(sorted(r51_cases.keys()))

    if args.cases:
        print(f"[R51] manual cases: {args.cases}")
        run_root = _run_once(cases=list(args.cases), mode=args.mode, epochs=int(args.epochs))
        print(f"[R51] manual run root: {run_root.as_posix()}")
        return

    if args.suite == "confirm3":
        _run_confirm3(mode=args.mode, epochs=int(args.epochs))
        return

    if args.suite == "repeat_base":
        roots = _run_repeats(CASE_BASE, mode=args.mode, epochs=int(args.epochs), repeat_runs=int(args.repeat_runs))
        _print_repeat_summary(roots, CASE_BASE)
        return

    if args.suite in ("repeat_depth", "repeat_candidate"):
        roots = _run_repeats(CASE_TRIAL_DEPTH, mode=args.mode, epochs=int(args.epochs), repeat_runs=int(args.repeat_runs))
        _print_repeat_summary(roots, CASE_TRIAL_DEPTH)
        return

    if args.suite in ("repeat_aibw", "repeat_longw", "repeat_well"):
        roots = _run_repeats(CASE_TRIAL_AIBW, mode=args.mode, epochs=int(args.epochs), repeat_runs=int(args.repeat_runs))
        _print_repeat_summary(roots, CASE_TRIAL_AIBW)
        return

    if args.suite == "compare_aionly":
        run_root = _run_compare_aionly(mode=args.mode, epochs=int(args.epochs))
        print(f"[R51] compare_aionly run root: {run_root.as_posix()}")
        return

    # full: do 3-case confirmation first, then repeat selected formal line.
    _, selected, detail = _run_confirm3(mode=args.mode, epochs=int(args.epochs))
    if selected != CASE_BASE:
        print(f"[R51] deep-loss trial confirmed. Formal baseline upgraded to {selected}.")
    else:
        print(f"[R51] no deep-loss trial passed hard gate. Keep baseline {CASE_BASE}.")
    print(f"[R51] decision detail: {detail}")

    roots = _run_repeats(selected, mode=args.mode, epochs=int(args.epochs), repeat_runs=int(args.repeat_runs))
    _print_repeat_summary(roots, selected)


if __name__ == "__main__":
    main()
