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

# Three-case confirmation set.
CASE_BASE = "r51_nobw_rb100"
CASE_BACKUP = "r51_nobw_rb100_cf058_ls22"      # conservative backup
CASE_CANDIDATE = "r51_nobw_rb100_cf060_ls23"   # promoted formal candidate

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

    # Baseline config (rb100) from current best stable line.
    base_cfg = copy.deepcopy(base.CASE_PRESETS["r5_anchor_ws_tight_nobw_cache200"])
    base_cfg["agpe_rebuild_every"] = 100
    base_cfg["R_update_every"] = 50
    base_cfg["run_id_suffix"] = f"{base_cfg['run_id_suffix']}_r51_rb100"
    base.CASE_PRESETS[CASE_BASE] = base_cfg
    cases[CASE_BASE] = base_cfg

    # Backup: single-knob conf tweak only (lift fixed).
    backup_cfg = copy.deepcopy(base_cfg)
    backup_cfg["aniso_closed_loop_conf_thresh"] = 0.58
    backup_cfg["agpe_lift_sigma"] = 2.2
    backup_cfg["run_id_suffix"] = f"{backup_cfg['run_id_suffix']}_cf058_ls22"
    base.CASE_PRESETS[CASE_BACKUP] = backup_cfg
    cases[CASE_BACKUP] = backup_cfg

    # Candidate: single-knob lift tweak only (conf fixed).
    cand_cfg = copy.deepcopy(base_cfg)
    cand_cfg["aniso_closed_loop_conf_thresh"] = 0.60
    cand_cfg["agpe_lift_sigma"] = 2.3
    cand_cfg["run_id_suffix"] = f"{cand_cfg['run_id_suffix']}_cf060_ls23"
    base.CASE_PRESETS[CASE_CANDIDATE] = cand_cfg
    cases[CASE_CANDIDATE] = cand_cfg

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
            "R51 runner: 3-case confirmation (baseline + cf058 + cf060_ls23), "
            "then choose formal line by R2 and shadow_area_mean and run repeats."
        )
    )
    parser.add_argument("--mode", default="both", choices=["train", "test", "both"])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument(
        "--suite",
        default="full",
        choices=["full", "confirm3", "repeat_base", "repeat_candidate", "compare_aionly"],
        help=(
            "full=confirm3 then auto-select and repeat; "
            "confirm3=single 3-case confirmation only; "
            "repeat_base/repeat_candidate=forced repeats; "
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


def _row(df: pd.DataFrame, case_name: str) -> pd.Series:
    sub = df[df["case"] == case_name]
    if len(sub) == 0:
        raise RuntimeError(f"Missing case row in summary: {case_name}")
    return sub.iloc[0]


def _as_float(v) -> float:
    if pd.isna(v):
        return float("nan")
    return float(v)


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
            "gate_r2_shadow_ok",
            "keep_for_formal",
        ]
        if c in df.columns
    ]
    if cols:
        print(df[cols].to_string(index=False))


def _confirm_three_cases(df: pd.DataFrame) -> Tuple[str, dict]:
    base_r = _row(df, CASE_BASE)
    backup_r = _row(df, CASE_BACKUP)
    cand_r = _row(df, CASE_CANDIDATE)

    base_r2 = _as_float(base_r.get("r2"))
    backup_r2 = _as_float(backup_r.get("r2"))
    cand_r2 = _as_float(cand_r.get("r2"))

    base_shadow = _as_float(base_r.get("shadow_area_mean"))
    backup_shadow = _as_float(backup_r.get("shadow_area_mean"))
    cand_shadow = _as_float(cand_r.get("shadow_area_mean"))

    # Formal replacement rule from user: candidate must keep leading in R2 and shadow_area_mean.
    lead_r2 = cand_r2 >= max(base_r2, backup_r2)
    lead_shadow = cand_shadow <= min(base_shadow, backup_shadow)
    candidate_wins = bool(lead_r2 and lead_shadow)

    selected = CASE_CANDIDATE if candidate_wins else CASE_BASE

    detail = {
        "base_r2": base_r2,
        "backup_r2": backup_r2,
        "candidate_r2": cand_r2,
        "base_shadow_mean": base_shadow,
        "backup_shadow_mean": backup_shadow,
        "candidate_shadow_mean": cand_shadow,
        "candidate_delta_r2_vs_base": cand_r2 - base_r2,
        "candidate_delta_shadow_vs_base": cand_shadow - base_shadow,
        "candidate_delta_r2_vs_backup": cand_r2 - backup_r2,
        "candidate_delta_shadow_vs_backup": cand_shadow - backup_shadow,
        # Focused d160 check for reporting (not hard gate).
        "base_shadow_d160": _as_float(base_r.get("shadow_area_d160", np.nan)),
        "backup_shadow_d160": _as_float(backup_r.get("shadow_area_d160", np.nan)),
        "candidate_shadow_d160": _as_float(cand_r.get("shadow_area_d160", np.nan)),
        "candidate_wins": candidate_wins,
    }
    return selected, detail


def _run_confirm3(mode: str, epochs: int) -> Tuple[Path, str, dict]:
    cases = [CASE_BASE, CASE_BACKUP, CASE_CANDIDATE]
    print(f"[R51] confirm3 cases={cases} mode={mode} epochs={epochs}")
    run_root = _run_once(cases=cases, mode=mode, epochs=epochs)
    df = _read_summary(run_root)
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
            df = _read_summary(root)
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
            }
        )

    if not rows:
        print("[R51] repeat summary unavailable.")
        return

    df = pd.DataFrame(rows)
    print("[R51] repeat metrics:")
    print(df.to_string(index=False))
    stats = df[["r2", "pcc", "shadow_area_d160", "shadow_area_mean", "pred_mean", "pred_std"]].agg(["mean", "std"])
    print("[R51] repeat mean/std:")
    print(stats.to_string())


def _run_compare_aionly(mode: str, epochs: int) -> Path:
    cases = [CASE_BASE, CASE_AIONLY]
    print(f"[R51] compare_aionly cases={cases} mode={mode} epochs={epochs}")
    run_root = _run_once(cases=cases, mode=mode, epochs=epochs)
    df = _read_summary(run_root)
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

    if args.suite == "repeat_candidate":
        roots = _run_repeats(CASE_CANDIDATE, mode=args.mode, epochs=int(args.epochs), repeat_runs=int(args.repeat_runs))
        _print_repeat_summary(roots, CASE_CANDIDATE)
        return

    if args.suite == "compare_aionly":
        run_root = _run_compare_aionly(mode=args.mode, epochs=int(args.epochs))
        print(f"[R51] compare_aionly run root: {run_root.as_posix()}")
        return

    # full: do 3-case confirmation first, then repeat selected formal line.
    _, selected, detail = _run_confirm3(mode=args.mode, epochs=int(args.epochs))
    if selected == CASE_CANDIDATE:
        print("[R51] candidate confirmed. Formal baseline replaced by r51_nobw_rb100_cf060_ls23.")
    else:
        print("[R51] candidate not consistently leading. Keep baseline r51_nobw_rb100.")
    print(f"[R51] decision detail: {detail}")

    roots = _run_repeats(selected, mode=args.mode, epochs=int(args.epochs), repeat_runs=int(args.repeat_runs))
    _print_repeat_summary(roots, selected)


if __name__ == "__main__":
    main()
