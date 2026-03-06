from __future__ import annotations

import argparse
import copy
import glob
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd

import run_projection_sweep as base


RESULTS_ROOT = Path("results")
CASE_NOBW = "r51_nobw_rb100"
CASE_AIONLY = "r51_beta020_aionly_rb100"

# Small-range ghost-suppression tuning on top of nobw_rb100.
# Only two knobs are changed: aniso_closed_loop_conf_thresh, agpe_lift_sigma.
MICRO_VARIANTS = [
    # (case_name, conf_thresh, lift_sigma)
    ("r51_nobw_rb100_cf070_ls22", 0.70, 2.2),
    ("r51_nobw_rb100_cf060_ls20", 0.60, 2.0),
    ("r51_nobw_rb100_cf070_ls20", 0.70, 2.0),
]


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
    """Register Round5.1 cases on top of run_projection_sweep presets."""
    r51_cases: Dict[str, dict] = {}

    def make_case(base_name: str, new_name: str, rebuild_every: int, suffix: str, *, aionly: bool = False) -> None:
        cfg = copy.deepcopy(base.CASE_PRESETS[base_name])
        cfg["agpe_rebuild_every"] = int(rebuild_every)
        cfg["R_update_every"] = 50
        cfg["run_id_suffix"] = f"{cfg['run_id_suffix']}{suffix}"
        if aionly:
            cfg["boundary_weight_apply_ai"] = True
            cfg["boundary_weight_apply_detail"] = False
            cfg["boundary_weight_apply_facies"] = False
        base.CASE_PRESETS[new_name] = cfg
        r51_cases[new_name] = cfg

    # Formal baseline fixed as requested.
    make_case(
        "r5_anchor_ws_tight_nobw_cache200",
        CASE_NOBW,
        rebuild_every=100,
        suffix="_r51_rb100",
    )

    # Single diagnostic candidate, evaluated in the same run root with baseline.
    make_case(
        "r5_anchor_ws_tight_beta020_aionly_cache200",
        CASE_AIONLY,
        rebuild_every=100,
        suffix="_r51_rb100",
        aionly=True,
    )

    # Micro tuning variants from nobw baseline: only change conf_thresh + lift_sigma.
    for name, conf_t, lift_s in MICRO_VARIANTS:
        cfg = copy.deepcopy(base.CASE_PRESETS[CASE_NOBW])
        cfg["aniso_closed_loop_conf_thresh"] = float(conf_t)
        cfg["agpe_lift_sigma"] = float(lift_s)
        cfg["run_id_suffix"] = f"{cfg['run_id_suffix']}_cf{int(conf_t * 100):03d}_ls{int(lift_s * 10):02d}"
        base.CASE_PRESETS[name] = cfg
        r51_cases[name] = cfg

    return r51_cases


def _parse_args(choices: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Round5.1 runner: compare nobw_rb100 vs beta020_aionly_rb100 in same run root, "
            "then repeat selected winner 3 times. Also supports a micro sweep on conf_thresh/lift_sigma."
        )
    )
    parser.add_argument("--mode", default="both", choices=["train", "test", "both"])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument(
        "--suite",
        default="full",
        choices=["full", "compare", "repeat_nobw", "repeat_aionly", "micro"],
        help=(
            "full=compare+auto-pick+repeats; compare=only same-root comparison; "
            "repeat_nobw/repeat_aionly=force selected repeats; micro=nobw small-range conf/lift tuning"
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


def _read_summary(run_root: Path) -> pd.DataFrame:
    xlsx = run_root / "projection_sweep_summary.xlsx"
    if not xlsx.is_file():
        raise FileNotFoundError(f"Missing summary file: {xlsx}")
    return pd.read_excel(xlsx)


def _pick_winner_from_compare(df: pd.DataFrame) -> tuple[str, dict]:
    req = {CASE_NOBW, CASE_AIONLY}
    has = set(df.get("case", pd.Series([], dtype=str)).astype(str).tolist())
    if not req.issubset(has):
        missing = sorted(list(req - has))
        raise RuntimeError(f"Comparison summary missing case rows: {missing}")

    nobw = df[df["case"] == CASE_NOBW].iloc[0]
    aionly = df[df["case"] == CASE_AIONLY].iloc[0]

    delta_r2 = aionly.get("delta_r2_vs_nobw", None)
    delta_shadow = aionly.get("delta_shadow_area_vs_nobw", None)

    if delta_r2 is None or pd.isna(delta_r2):
        delta_r2 = float(aionly["r2"]) - float(nobw["r2"])
    else:
        delta_r2 = float(delta_r2)

    if delta_shadow is None or pd.isna(delta_shadow):
        if ("shadow_area_mean" in df.columns) and pd.notna(aionly.get("shadow_area_mean", None)) and pd.notna(nobw.get("shadow_area_mean", None)):
            delta_shadow = float(aionly["shadow_area_mean"]) - float(nobw["shadow_area_mean"])
        else:
            delta_shadow = float("nan")
    else:
        delta_shadow = float(delta_shadow)

    shadow_ok = (not pd.isna(delta_shadow)) and (delta_shadow <= 0.0)
    pass_rule = (delta_r2 >= 0.0) and shadow_ok
    winner = CASE_AIONLY if pass_rule else CASE_NOBW

    detail = {
        "nobw_r2": float(nobw["r2"]),
        "aionly_r2": float(aionly["r2"]),
        "delta_r2_vs_nobw": delta_r2,
        "nobw_shadow": float(nobw.get("shadow_area_mean", float("nan"))),
        "aionly_shadow": float(aionly.get("shadow_area_mean", float("nan"))),
        "delta_shadow_area_vs_nobw": delta_shadow,
        "rule_pass": bool(pass_rule),
    }
    return winner, detail


def _run_once(cases: List[str], mode: str, epochs: int) -> Path:
    before = _list_projection_dirs()
    base.run_cases(cases=cases, mode=mode, epochs_override=epochs)
    return _latest_new_run_root(before)


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
        if "case" not in df.columns:
            continue
        sub = df[df["case"] == case_name]
        if len(sub) == 0:
            continue
        r = sub.iloc[0]
        rows.append(
            {
                "run_root": root.name,
                "r2": float(r.get("r2", float("nan"))),
                "pcc": float(r.get("pcc", float("nan"))),
                "shadow_area_mean": float(r.get("shadow_area_mean", float("nan"))),
                "pred_mean": float(r.get("pred_mean", float("nan"))),
                "pred_std": float(r.get("pred_std", float("nan"))),
            }
        )

    if not rows:
        print("[R51] repeat summary unavailable (no readable summary rows).")
        return

    df = pd.DataFrame(rows)
    print("[R51] repeat metrics:")
    print(df.to_string(index=False))
    stats = df[["r2", "pcc", "shadow_area_mean", "pred_mean", "pred_std"]].agg(["mean", "std"])
    print("[R51] repeat mean/std:")
    print(stats.to_string())


def _run_micro_tuning(mode: str, epochs: int) -> Path:
    cases = [CASE_NOBW] + [name for name, _, _ in MICRO_VARIANTS]
    print(f"[R51][MICRO] running cases={cases} mode={mode} epochs={epochs}")
    run_root = _run_once(cases=cases, mode=mode, epochs=epochs)
    df = _read_summary(run_root)
    cols = [
        c
        for c in [
            "case",
            "r2",
            "pcc",
            "shadow_area_mean",
            "pred_mean",
            "pred_std",
            "delta_r2_vs_nobw",
            "delta_shadow_area_vs_nobw",
            "gate_r2_shadow_ok",
            "keep_for_formal",
        ]
        if c in df.columns
    ]
    if cols:
        print("[R51][MICRO] summary:")
        print(df[cols].to_string(index=False))
    return run_root


def main() -> None:
    r51_cases = _build_r51_cases()
    all_choices = sorted(r51_cases.keys())
    args = _parse_args(all_choices)

    if args.cases:
        print(f"[R51] manual cases: {args.cases}")
        run_root = _run_once(cases=list(args.cases), mode=args.mode, epochs=int(args.epochs))
        print(f"[R51] manual run root: {run_root.as_posix()}")
        return

    if args.suite == "micro":
        run_root = _run_micro_tuning(mode=args.mode, epochs=int(args.epochs))
        print(f"[R51][MICRO] run root: {run_root.as_posix()}")
        return

    if args.suite == "compare":
        run_root = _run_once(cases=[CASE_NOBW, CASE_AIONLY], mode=args.mode, epochs=int(args.epochs))
        df = _read_summary(run_root)
        winner, detail = _pick_winner_from_compare(df)
        print(f"[R51] compare run root: {run_root.as_posix()}")
        print(f"[R51] compare detail: {detail}")
        print(f"[R51] selected winner: {winner}")
        return

    if args.suite in ("repeat_nobw", "repeat_aionly"):
        winner = CASE_NOBW if args.suite == "repeat_nobw" else CASE_AIONLY
        roots = _run_repeats(case_name=winner, mode=args.mode, epochs=int(args.epochs), repeat_runs=int(args.repeat_runs))
        _print_repeat_summary(roots, winner)
        return

    # full: compare first, then auto-select, then repeat 3 runs.
    compare_root = _run_once(cases=[CASE_NOBW, CASE_AIONLY], mode=args.mode, epochs=int(args.epochs))
    compare_df = _read_summary(compare_root)
    winner, detail = _pick_winner_from_compare(compare_df)

    print(f"[R51] compare run root: {compare_root.as_posix()}")
    print(f"[R51] compare detail: {detail}")
    print(f"[R51] selected winner: {winner}")
    if winner == CASE_NOBW:
        print("[R51] aionly fails rule (delta_r2>=0 and delta_shadow<=0), archived as diagnostic.")
    else:
        print("[R51] aionly passes rule, upgrade to formal candidate.")

    repeat_roots = _run_repeats(
        case_name=winner,
        mode=args.mode,
        epochs=int(args.epochs),
        repeat_runs=int(args.repeat_runs),
    )
    _print_repeat_summary(repeat_roots, winner)


if __name__ == "__main__":
    main()
