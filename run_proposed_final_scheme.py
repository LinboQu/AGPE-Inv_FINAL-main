from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from utils.config_resolver import build_test_config

from run_proposed_tuning import (
    REPORT_METRIC_KEYS,
    TEST_PROFILE,
    TUNE_TEST_USER,
    _apply_epoch_budget,
    _augment_selection_fields,
    _build_summary_row as _shared_build_summary_row,
    _case_result_prefixes,
    _jsonable,
    _list_case_result_files,
    _move_file_safe,
    _safe_float,
    _save_case_json,
    _save_metrics_csv,
    _save_metrics_excel,
    _selection_sort_key,
    analyze_case_artifacts,
)


@dataclass(frozen=True)
class FinalTrialSpec:
    name: str
    suffix: str
    base_kind: str
    note: str
    overrides: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def discover_latest_best_baseline(results_root: Path) -> dict[str, Any]:
    run_dirs = sorted(
        [p for p in results_root.iterdir() if p.is_dir() and p.name.startswith("proposed_tuning_")],
        reverse=True,
    )
    for run_root in run_dirs:
        summary_payload = _load_json(run_root / "trial_summary.json")
        if summary_payload is None:
            continue
        rows = [row for row in summary_payload.get("rows", []) if "mae" in row]
        if not rows:
            continue
        rows = sorted(rows, key=lambda row: _safe_float(row.get("selection_rank_all"), float("inf")))
        best_row = rows[0]

        case_dir = None
        top_payload = _load_json(run_root / "top_trials.json")
        if top_payload is not None and top_payload.get("top_trials"):
            top_entry = top_payload["top_trials"][0]
            if str(top_entry.get("trial_name")) == str(best_row.get("trial_name")):
                case_dir = Path(str(top_entry.get("case_dir")))

        if case_dir is None:
            matches = [
                p
                for p in sorted(run_root.iterdir())
                if p.is_dir() and p.name.endswith(str(best_row.get("trial_name")))
            ]
            if matches:
                case_dir = matches[0]

        if case_dir is None:
            continue

        train_cfg = _load_json(case_dir / "train_config.json")
        test_cfg = _load_json(case_dir / "test_config.json")
        if train_cfg is None or test_cfg is None:
            continue

        return {
            "run_root": run_root,
            "trial_name": str(best_row.get("trial_name")),
            "case_dir": case_dir,
            "summary_row": best_row,
            "train_cfg": train_cfg,
            "test_cfg": test_cfg,
        }

    raise FileNotFoundError("No tested proposed_tuning run with usable trial_summary.json was found.")


def _final_scheme_defaults() -> dict[str, Any]:
    return {
        "lambda_region_residual": 0.0,
        "lambda_region_bias": 0.0,
        "use_facies_class_weighted_ce": True,
        "facies_ce_gamma": 0.50,
        "facies_ce_min_weight": 0.50,
        "facies_ce_max_weight": 3.00,
        "lambda_facies_residual": 0.30,
        "facies_residual_mode": "charbonnier",
        "facies_residual_eps": 1e-3,
        "facies_residual_huber_delta": 0.05,
        "facies_residual_background_weight": 0.0,
        "facies_residual_channel_weight": 1.50,
        "facies_residual_pointbar_weight": 2.20,
        "lambda_boundary_band_residual": 0.45,
        "lambda_boundary_band_grad": 0.0,
        "boundary_band_mode": "charbonnier",
        "boundary_band_eps": 1e-3,
        "boundary_band_huber_delta": 0.05,
        "boundary_band_width": 2,
        "lambda_masked_stat": 0.08,
        "masked_stat_channel_weight": 1.00,
        "masked_stat_pointbar_weight": 1.50,
        "masked_stat_eps": 1e-6,
        "use_structural_warm_schedule": True,
        "structural_warm_start_epoch": 550,
        "structural_warm_ramp_epochs": 250,
    }


def build_trial_specs() -> dict[str, FinalTrialSpec]:
    return {
        "final_ref_best_stage4_region_res": FinalTrialSpec(
            name="final_ref_best_stage4_region_res",
            suffix="_pfinal_ref_s4best",
            base_kind="baseline",
            note="Exact reference of the latest best baseline discovered from proposed_tuning results.",
            overrides={},
        ),
        "final_charb_ce_soft": FinalTrialSpec(
            name="final_charb_ce_soft",
            suffix="_pfinal_charb_ce",
            base_kind="final_scheme",
            note="Final scheme warm-up with facies-aware Charbonnier residual and class-weighted CE.",
            overrides={
                "lambda_boundary_band_residual": 0.0,
                "lambda_boundary_band_grad": 0.0,
                "lambda_masked_stat": 0.0,
            },
        ),
        "final_charb_bdres_ce": FinalTrialSpec(
            name="final_charb_bdres_ce",
            suffix="_pfinal_charb_bdres",
            base_kind="final_scheme",
            note="Add channel/point-bar-aware 1D boundary-band residual control on top of Charbonnier+CE.",
            overrides={
                "lambda_masked_stat": 0.0,
            },
        ),
        "final_charb_bdres_stat_ce": FinalTrialSpec(
            name="final_charb_bdres_stat_ce",
            suffix="_pfinal_charb_bdres_stat",
            base_kind="final_scheme",
            note="Recommended full scheme: Charbonnier facies residual + boundary-band residual + masked mean/std + weighted CE.",
            overrides={},
        ),
        "final_charb_bdres_bdgrad_stat_ce": FinalTrialSpec(
            name="final_charb_bdres_bdgrad_stat_ce",
            suffix="_pfinal_charb_bdres_bdgrad_stat",
            base_kind="final_scheme",
            note="Full scheme plus a weak boundary-band depth-gradient term.",
            overrides={
                "lambda_boundary_band_grad": 0.12,
            },
        ),
        "final_charb_bdres_stat_ce_pbstrong": FinalTrialSpec(
            name="final_charb_bdres_stat_ce_pbstrong",
            suffix="_pfinal_charb_bdres_stat_pb2",
            base_kind="final_scheme",
            note="Full scheme with stronger point-bar emphasis in residual/stat terms.",
            overrides={
                "facies_residual_pointbar_weight": 2.60,
                "masked_stat_pointbar_weight": 1.80,
            },
        ),
    }


def build_trial_train_cfg(
    *,
    baseline_train_cfg: dict[str, Any],
    spec: FinalTrialSpec,
    epochs_override: int | None,
) -> dict[str, Any]:
    train_cfg = copy.deepcopy(baseline_train_cfg)
    if spec.base_kind == "final_scheme":
        train_cfg.update(copy.deepcopy(_final_scheme_defaults()))
    elif spec.base_kind != "baseline":
        raise ValueError(f"Unknown base_kind: {spec.base_kind}")
    train_cfg.update(copy.deepcopy(spec.overrides))
    train_cfg["run_id_suffix"] = spec.suffix
    _apply_epoch_budget(train_cfg, epochs_override)
    return train_cfg


def build_trial_test_cfg(train_cfg: dict[str, Any]) -> dict[str, Any]:
    return build_test_config(
        profile=TEST_PROFILE,
        user_cfg=TUNE_TEST_USER,
        expert_overrides={"run_id_suffix": train_cfg.get("run_id_suffix", "")},
        train_cfg=train_cfg,
    )


def _build_final_summary_row(
    *,
    spec: FinalTrialSpec,
    train_cfg: dict[str, Any],
    test_cfg: dict[str, Any],
    metrics: dict[str, Any] | None,
    visual_metrics: dict[str, Any] | None,
    status: str,
    baseline_info: dict[str, Any],
) -> dict[str, Any]:
    proxy_spec = type(
        "ProxySpec",
        (),
        {
            "name": spec.name,
            "stage": "final",
            "group": spec.base_kind,
            "note": spec.note,
            "base": spec.base_kind,
        },
    )()
    row = _shared_build_summary_row(
        spec=proxy_spec,
        train_cfg=train_cfg,
        test_cfg=test_cfg,
        metrics=metrics,
        visual_metrics=visual_metrics,
        status=status,
    )
    row.update(
        {
            "baseline_source_run": baseline_info["run_root"].name,
            "baseline_source_trial": baseline_info["trial_name"],
            "baseline_source_case": baseline_info["case_dir"].name,
            "lambda_facies_residual": train_cfg.get("lambda_facies_residual", ""),
            "facies_residual_mode": train_cfg.get("facies_residual_mode", ""),
            "facies_residual_channel_weight": train_cfg.get("facies_residual_channel_weight", ""),
            "facies_residual_pointbar_weight": train_cfg.get("facies_residual_pointbar_weight", ""),
            "lambda_boundary_band_residual": train_cfg.get("lambda_boundary_band_residual", ""),
            "lambda_boundary_band_grad": train_cfg.get("lambda_boundary_band_grad", ""),
            "boundary_band_mode": train_cfg.get("boundary_band_mode", ""),
            "boundary_band_width": train_cfg.get("boundary_band_width", ""),
            "lambda_masked_stat": train_cfg.get("lambda_masked_stat", ""),
            "masked_stat_channel_weight": train_cfg.get("masked_stat_channel_weight", ""),
            "masked_stat_pointbar_weight": train_cfg.get("masked_stat_pointbar_weight", ""),
            "use_structural_warm_schedule": int(bool(train_cfg.get("use_structural_warm_schedule", False))),
            "structural_warm_start_epoch": train_cfg.get("structural_warm_start_epoch", ""),
            "structural_warm_ramp_epochs": train_cfg.get("structural_warm_ramp_epochs", ""),
        }
    )
    return row


def rank_final_rows(
    rows: list[dict[str, Any]],
    *,
    ref_trial_name: str,
    mae_tolerance: float,
) -> list[dict[str, Any]]:
    ref_row = next((row for row in rows if row.get("trial_name") == ref_trial_name and "mae" in row), None)
    ref_mae = _safe_float(ref_row.get("mae"), float("inf")) if ref_row is not None else float("inf")
    ref_ssim = _safe_float(ref_row.get("ssim"), float("nan")) if ref_row is not None else float("nan")
    ref_vif = _safe_float(ref_row.get("vif"), float("nan")) if ref_row is not None else float("nan")

    metric_rows = [row for row in rows if "mae" in row]
    for row in metric_rows:
        mae = _safe_float(row.get("mae"), float("inf"))
        if ref_row is not None:
            row["mae_gate_ok"] = int(mae <= ref_mae * (1.0 + float(mae_tolerance)))
            row["delta_mae_vs_ref"] = mae - ref_mae
            row["delta_ssim_vs_ref"] = _safe_float(row.get("ssim"), float("nan")) - ref_ssim
            row["delta_vif_vs_ref"] = _safe_float(row.get("vif"), float("nan")) - ref_vif
        else:
            row["mae_gate_ok"] = 1
            row["delta_mae_vs_ref"] = ""
            row["delta_ssim_vs_ref"] = ""
            row["delta_vif_vs_ref"] = ""
        _augment_selection_fields(row)

    ranked = sorted(metric_rows, key=_selection_sort_key)
    for idx, row in enumerate(ranked, start=1):
        row["selection_rank_all"] = idx
        row["selection_rank_final"] = idx
    for row in rows:
        if "selection_rank_all" not in row:
            row["selection_rank_all"] = ""
            row["selection_rank_final"] = ""
    return rows


def export_final_outputs(
    *,
    run_root: Path,
    summary_rows: list[dict[str, Any]],
    trial_registry: list[dict[str, Any]],
    shortlist_k: int,
    visual_top_k: int,
    baseline_info: dict[str, Any],
) -> None:
    registry_map = {str(item["trial_name"]): Path(str(item["case_dir"])) for item in trial_registry}
    for row in summary_rows:
        case_dir = registry_map.get(str(row.get("trial_name")))
        if case_dir is not None:
            _save_case_json(case_dir / "summary_row.json", row)

    summary_json = run_root / "trial_summary.json"
    summary_csv = run_root / "trial_summary.csv"
    summary_xlsx = run_root / "trial_summary.xlsx"
    shortlist_json = run_root / "final_shortlist.json"
    top_trials_json = run_root / "top_trials.json"
    baseline_json = run_root / "baseline_selection.json"

    _save_case_json(summary_json, {"rows": summary_rows})
    _save_metrics_csv(summary_csv, summary_rows)
    _save_metrics_excel(summary_xlsx, summary_rows, sheet_name="final_scheme")
    _save_case_json(
        baseline_json,
        {
            "run_root": str(baseline_info["run_root"]),
            "trial_name": baseline_info["trial_name"],
            "case_dir": str(baseline_info["case_dir"]),
            "summary_row": baseline_info["summary_row"],
        },
    )

    ranked = sorted([row for row in summary_rows if "mae" in row], key=_selection_sort_key)
    shortlist = []
    for row in ranked[: max(int(shortlist_k), 0)]:
        shortlist.append(
            {
                "trial_name": row.get("trial_name"),
                "selection_rank_final": row.get("selection_rank_final"),
                "selection_rank_all": row.get("selection_rank_all"),
                "mae": row.get("mae"),
                "ssim": row.get("ssim"),
                "vif": row.get("vif"),
                "pred_std_ratio": row.get("pred_std_ratio"),
                "lap_ratio": row.get("lap_ratio"),
                "q01_gap_abs": row.get("q01_gap_abs"),
                "q99_gap_abs": row.get("q99_gap_abs"),
                "note": row.get("note"),
            }
        )
    _save_case_json(
        shortlist_json,
        {
            "preferred_stage": "final",
            "shortlist_k": int(shortlist_k),
            "shortlist": shortlist,
        },
    )

    top_trials = []
    for row in ranked[: max(int(visual_top_k), 0)]:
        trial_name = str(row.get("trial_name"))
        case_dir = registry_map.get(trial_name)
        top_trials.append(
            {
                "trial_name": trial_name,
                "case_dir": str(case_dir.resolve()) if case_dir is not None else "",
                "case_name": case_dir.name if case_dir is not None else trial_name,
                "stage": "final",
                "group": row.get("group"),
                "selection_rank_stage": row.get("selection_rank_final"),
                "selection_rank_all": row.get("selection_rank_all"),
                "display_label": f"Rank {row.get('selection_rank_final')}: {trial_name}",
                "mae": row.get("mae"),
                "ssim": row.get("ssim"),
                "vif": row.get("vif"),
                "pred_std_ratio": row.get("pred_std_ratio"),
                "lap_ratio": row.get("lap_ratio"),
                "q01_gap_abs": row.get("q01_gap_abs"),
                "q99_gap_abs": row.get("q99_gap_abs"),
            }
        )
    _save_case_json(
        top_trials_json,
        {
            "preferred_stage": "final",
            "visual_top_k": int(visual_top_k),
            "source_shortlist_k": int(shortlist_k),
            "top_trials": top_trials,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the final structure-aware Proposed tuning scheme starting from the latest best proposed_tuning baseline."
    )
    parser.add_argument("--trials", type=str, nargs="*", default=None, help="Optional explicit final trial names.")
    parser.add_argument("--mode", type=str, default="both", choices=["train", "test", "both"])
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--visual-top-k", type=int, default=5)
    parser.add_argument("--shortlist-k", type=int, default=4)
    parser.add_argument("--mae-tolerance", type=float, default=0.02)
    parser.add_argument("--list-trials", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_specs = build_trial_specs()
    if args.list_trials:
        for name, spec in all_specs.items():
            print(f"{name:40s} base={spec.base_kind:12s} suffix={spec.suffix:24s} note={spec.note}")
        return

    trial_names = list(all_specs.keys()) if not args.trials else list(args.trials)
    unknown = [name for name in trial_names if name not in all_specs]
    if unknown:
        raise ValueError(f"Unknown final trial names: {unknown}")

    results_root = Path("results")
    baseline_info = discover_latest_best_baseline(results_root)
    print(
        f"[BASELINE] latest best run={baseline_info['run_root'].name} "
        f"trial={baseline_info['trial_name']} mae={baseline_info['summary_row'].get('mae')} "
        f"ssim={baseline_info['summary_row'].get('ssim')} vif={baseline_info['summary_row'].get('vif')}"
    )

    train = None
    test = None
    if not args.dry_run:
        from train_multitask import train as _train
        from test_3D import test as _test

        train = _train
        test = _test

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_root = results_root / f"proposed_final_scheme_{run_stamp}"
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[RUN] final scheme results root: {run_root.as_posix()}")

    _save_case_json(
        run_root / "run_meta.json",
        {
            "mode": args.mode,
            "epochs_override": args.epochs,
            "dry_run": bool(args.dry_run),
            "visual_top_k": int(args.visual_top_k),
            "shortlist_k": int(args.shortlist_k),
            "mae_tolerance": float(args.mae_tolerance),
            "trial_names": trial_names,
            "baseline_run_root": str(baseline_info["run_root"]),
            "baseline_trial_name": baseline_info["trial_name"],
        },
    )

    summary_rows: list[dict[str, Any]] = []
    trial_registry: list[dict[str, Any]] = []

    baseline_train_cfg = copy.deepcopy(baseline_info["train_cfg"])
    for idx, trial_name in enumerate(trial_names, start=1):
        spec = all_specs[trial_name]
        train_cfg = build_trial_train_cfg(
            baseline_train_cfg=baseline_train_cfg,
            spec=spec,
            epochs_override=args.epochs,
        )
        test_cfg = build_trial_test_cfg(train_cfg)
        case_dir = run_root / f"{idx:02d}_{trial_name}"
        case_dir.mkdir(parents=True, exist_ok=True)
        prefixes = _case_result_prefixes(train_cfg)

        _save_case_json(case_dir / "train_config.json", train_cfg)
        _save_case_json(case_dir / "test_config.json", test_cfg)
        _save_case_json(case_dir / "trial_meta.json", {"trial": _jsonable(spec.__dict__)})
        trial_registry.append(
            {
                "order": idx,
                "trial_name": trial_name,
                "stage": "final",
                "group": spec.base_kind,
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

        print(f"\n[{idx}/{len(trial_names)}] trial={trial_name} base={spec.base_kind} epochs={train_cfg.get('epochs')}")
        print(f"[NOTE] {spec.note}")

        metrics: dict[str, Any] | None = None
        visual_metrics: dict[str, Any] | None = None
        status = "dry_run"

        if args.dry_run:
            print("[DRY-RUN] skipping train/test execution")
        else:
            if args.mode in ("train", "both"):
                print("[RUN] train_multitask.train(...)")
                assert train is not None
                train(train_cfg)

            if args.mode in ("test", "both"):
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

            status = "tested" if args.mode in ("test", "both") else "trained"

        row = _build_final_summary_row(
            spec=spec,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            metrics=metrics,
            visual_metrics=visual_metrics,
            status=status,
            baseline_info=baseline_info,
        )
        _save_case_json(case_dir / "summary_row.json", row)
        summary_rows.append(row)

    _save_case_json(run_root / "trial_registry.json", {"trials": trial_registry})
    summary_rows = rank_final_rows(
        summary_rows,
        ref_trial_name="final_ref_best_stage4_region_res",
        mae_tolerance=float(args.mae_tolerance),
    )
    export_final_outputs(
        run_root=run_root,
        summary_rows=summary_rows,
        trial_registry=trial_registry,
        shortlist_k=int(args.shortlist_k),
        visual_top_k=int(args.visual_top_k),
        baseline_info=baseline_info,
    )
    print(f"[SAVE] summary -> {(run_root / 'trial_summary.xlsx').as_posix()}")
    print(f"[SAVE] shortlist -> {(run_root / 'final_shortlist.json').as_posix()}")
    print(f"[SAVE] top trials -> {(run_root / 'top_trials.json').as_posix()}")


if __name__ == "__main__":
    main()
