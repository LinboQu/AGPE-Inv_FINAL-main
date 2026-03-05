from __future__ import annotations

import argparse
import copy
from typing import Dict, List

import run_projection_sweep as base


def _build_r51_cases() -> Dict[str, dict]:
    """Register Round5.1 cases on top of run_projection_sweep presets."""
    r51_cases: Dict[str, dict] = {}

    def make_case(base_name: str, new_name: str, rebuild_every: int, suffix: str, *, detail_weighted: bool | None = None) -> None:
        cfg = copy.deepcopy(base.CASE_PRESETS[base_name])
        cfg["agpe_rebuild_every"] = int(rebuild_every)
        cfg["R_update_every"] = 50
        cfg["run_id_suffix"] = f"{cfg['run_id_suffix']}{suffix}"
        if detail_weighted is not None:
            cfg["boundary_weight_apply_ai"] = True
            cfg["boundary_weight_apply_detail"] = bool(detail_weighted)
            cfg["boundary_weight_apply_facies"] = False
        base.CASE_PRESETS[new_name] = cfg
        r51_cases[new_name] = cfg

    # Formal lines: nobw vs beta020_aidetail, each at rebuild=100/150.
    make_case(
        "r5_anchor_ws_tight_nobw_cache200",
        "r51_nobw_rb100",
        rebuild_every=100,
        suffix="_r51_rb100",
    )
    make_case(
        "r5_anchor_ws_tight_beta020_aidetail_cache200",
        "r51_beta020_aidetail_rb100",
        rebuild_every=100,
        suffix="_r51_rb100",
    )
    make_case(
        "r5_anchor_ws_tight_nobw_cache200",
        "r51_nobw_rb150",
        rebuild_every=150,
        suffix="_r51_rb150",
    )
    make_case(
        "r5_anchor_ws_tight_beta020_aidetail_cache200",
        "r51_beta020_aidetail_rb150",
        rebuild_every=150,
        suffix="_r51_rb150",
    )

    # Diagnostic only: aionly at rebuild=150 (not for formal comparison).
    make_case(
        "r5_anchor_ws_tight_beta020_aionly_cache200",
        "r51_beta020_aionly_rb150_diag",
        rebuild_every=150,
        suffix="_r51_rb150_diag",
        detail_weighted=False,
    )

    return r51_cases


def _parse_args(choices: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Round5.1 projection sweep runner: two formal lines + one diagnostic line."
    )
    parser.add_argument("--mode", default="both", choices=["train", "test", "both"])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument(
        "--suite",
        default="all",
        choices=["formal", "diag", "all"],
        help="formal=two main lines only; diag=aionly only; all=formal+diag.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        choices=sorted(choices),
        help="Optional explicit case list override.",
    )
    return parser.parse_args()


def main() -> None:
    _build_r51_cases()

    groups = {
        "formal_rb100": ["r51_nobw_rb100", "r51_beta020_aidetail_rb100"],
        "formal_rb150": ["r51_nobw_rb150", "r51_beta020_aidetail_rb150"],
        "diag": ["r51_beta020_aionly_rb150_diag"],
    }
    all_choices = [c for group in groups.values() for c in group]

    args = _parse_args(all_choices)

    if args.cases:
        # Manual override: single sweep with user-specified cases.
        print(f"[R51] manual cases: {args.cases}")
        base.run_cases(cases=list(args.cases), mode=args.mode, epochs_override=args.epochs)
        return

    if args.suite == "formal":
        plan = ["formal_rb100", "formal_rb150"]
    elif args.suite == "diag":
        plan = ["diag"]
    else:
        plan = ["formal_rb100", "formal_rb150", "diag"]

    for name in plan:
        cases = groups[name]
        print(f"[R51] running group={name} cases={cases} mode={args.mode} epochs={args.epochs}")
        base.run_cases(cases=cases, mode=args.mode, epochs_override=args.epochs)


if __name__ == "__main__":
    main()
