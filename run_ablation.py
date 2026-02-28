import argparse
import copy
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from setting import TCN1D_test_p, TCN1D_train_p
from train_multitask import train
from test_3D import test


CASE_PRESETS: Dict[str, Dict[str, object]] = {
    # Control: legacy grid backend
    "grid": {
        "aniso_backend": "grid",
        "aniso_use_tensor_strength": False,
        "run_id_suffix": "_grid",
    },
    # Graph backend without tensor-strength modulation (parity with grid)
    "glat": {
        "aniso_backend": "graph_lattice",
        "aniso_use_tensor_strength": False,
        "run_id_suffix": "_glat",
    },
    # Graph backend with tensor-strength modulation enabled
    "glat_ts": {
        "aniso_backend": "graph_lattice",
        "aniso_use_tensor_strength": True,
        "run_id_suffix": "_glat_ts",
    },
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


def build_case_configs(case_name: str, epochs_override: int | None) -> tuple[dict, dict]:
    if case_name not in CASE_PRESETS:
        raise ValueError(f"Unknown case: {case_name}")

    preset = CASE_PRESETS[case_name]
    train_cfg = copy.deepcopy(TCN1D_train_p)
    test_cfg = copy.deepcopy(TCN1D_test_p)

    train_cfg.update(preset)
    test_cfg.update(preset)

    # Keep train/test run-id chain strictly aligned with current model combo.
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
    run_root = results_root / f"ablation_{run_stamp}"
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[RUN] ablation results root: {run_root.as_posix()}")

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
            test(test_cfg)

        new_files = _list_case_result_files(results_root, prefixes)
        if new_files:
            for p in new_files:
                _move_file_safe(p, case_dir)
            print(f"[SAVE] case artifacts saved to: {case_dir.as_posix()}")
        else:
            print(f"[SAVE][WARN] no case artifacts found for prefixes={prefixes}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AGPE ablations for grid / graph_lattice backends."
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["grid", "glat", "glat_ts"],
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
