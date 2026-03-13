from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def resolve_repo_root(start: Path | None = None) -> Path:
    root = Path.cwd() if start is None else Path(start)
    if (root / "utils").exists():
        return root
    if (root.parent / "utils").exists():
        return root.parent
    raise FileNotFoundError("Cannot locate repo root from current working directory.")


REPO_ROOT = resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.geomorphology_classification import Facies_model_class
from utils.noise import apply_test_noise
from utils.utils import standardize


DEFAULT_RESULT_DIR = REPO_ROOT / "results" / "paper_ablation_20260308_191828_636300" / "05_proposed"


def infer_run_id_for_full_ckpt(model_name: str) -> str:
    run_id = model_name
    for suf in ["_s_uns", "_uns", "_s", "_best", "_final"]:
        if run_id.endswith(suf):
            run_id = run_id[: -len(suf)]
    return run_id


def resolve_run_id_and_model_name(cfg: dict) -> tuple[str, str]:
    suffix = str(cfg.get("run_id_suffix", "") or "")
    if "run_id" in cfg and cfg["run_id"] is not None and str(cfg["run_id"]).strip() != "":
        run_id_base = str(cfg["run_id"]).strip()
    else:
        run_id_base = infer_run_id_for_full_ckpt(str(cfg.get("model_name", "")))
    run_id = run_id_base if (suffix == "" or run_id_base.endswith(suffix)) else f"{run_id_base}{suffix}"
    model_name = f"{run_id}_s_uns"
    return run_id, model_name


def load_stats_strict(run_id: str, data_flag: str) -> dict:
    full_ckpt_path = REPO_ROOT / "save_train_model" / f"{run_id}_full_ckpt_{data_flag}.pth"
    if full_ckpt_path.is_file():
        ckpt = torch.load(full_ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and ("stats" in ckpt) and (ckpt["stats"] is not None):
            return ckpt["stats"]
        raise RuntimeError(f"full_ckpt exists but stats missing: {full_ckpt_path}")

    dedicated = REPO_ROOT / "save_train_model" / f"norm_stats_{run_id}_{data_flag}.npy"
    if dedicated.is_file():
        return np.load(dedicated, allow_pickle=True).item()

    legacy = REPO_ROOT / "save_train_model" / f"norm_stats_{data_flag}.npy"
    if legacy.is_file():
        return np.load(legacy, allow_pickle=True).item()

    raise FileNotFoundError(
        f"Cannot find stats for run_id={run_id}, data_flag={data_flag}"
    )


def load_full_ckpt_strict(run_id: str, data_flag: str, device: torch.device) -> dict:
    full_ckpt_path = REPO_ROOT / "save_train_model" / f"{run_id}_full_ckpt_{data_flag}.pth"
    if not full_ckpt_path.is_file():
        raise FileNotFoundError(full_ckpt_path)
    ckpt = torch.load(full_ckpt_path, map_location=device)
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"full_ckpt is not a dict: {full_ckpt_path}")
    if "facies_state_dict" not in ckpt:
        raise RuntimeError(f"facies_state_dict missing in full_ckpt: {full_ckpt_path}")
    return ckpt


def get_data_raw(data_flag: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if data_flag != "Stanford_VI":
        raise ValueError(f"Only Stanford_VI is supported by this exporter, got {data_flag}")

    data_dir = REPO_ROOT / "data" / data_flag
    seismic3d = np.load(data_dir / "synth_40HZ.npy")
    model3d = np.load(data_dir / "AI.npy")
    facies3d = np.load(data_dir / "Facies.npy")

    h, il, xl = seismic3d.shape
    meta = {
        "H": h,
        "inline": il,
        "xline": xl,
        "seismic3d": seismic3d,
        "model3d": model3d,
        "facies3d": facies3d,
    }
    seismic_raw = np.transpose(seismic3d.reshape(h, il * xl), (1, 0))
    model_raw = np.transpose(model3d.reshape(h, il * xl), (1, 0))
    facies_raw = np.transpose(facies3d.reshape(h, il * xl), (1, 0))
    return seismic_raw, model_raw, facies_raw, meta


@torch.no_grad()
def infer_inverse_all(
    inverse_model: torch.nn.Module,
    seismic_np: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    x = torch.from_numpy(seismic_np.astype(np.float32))
    out_list: list[np.ndarray] = []
    inverse_model.eval()
    for i in range(0, x.shape[0], int(batch_size)):
        xb = x[i:i + int(batch_size)].to(device)
        yb = inverse_model(xb)
        yb = yb.squeeze(1) if yb.ndim == 3 else yb
        out_list.append(yb.detach().cpu().numpy())
    return np.concatenate(out_list, axis=0).astype(np.float32)


@torch.no_grad()
def infer_facies_prob_all(
    facies_model: torch.nn.Module,
    ai_np: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    x = torch.from_numpy(ai_np[:, np.newaxis, :].astype(np.float32))
    out_list: list[np.ndarray] = []
    facies_model.eval()
    for i in range(0, x.shape[0], int(batch_size)):
        xb = x[i:i + int(batch_size)].to(device)
        logits = facies_model(xb)
        probs = torch.softmax(logits, dim=1)
        out_list.append(probs.detach().cpu().numpy())
    return np.concatenate(out_list, axis=0).astype(np.float32)


def merge_configs(result_dir: Path) -> dict:
    train_cfg = json.loads((result_dir / "train_config.json").read_text(encoding="utf-8"))
    test_cfg = json.loads((result_dir / "test_config.json").read_text(encoding="utf-8"))
    cfg = dict(train_cfg)
    cfg.update(test_cfg)
    return cfg


def find_existing_bootstrap_ai(result_dir: Path) -> tuple[Path | None, str | None]:
    matches = sorted(result_dir.glob("*_pred_AI_bootstrap.npy"))
    if len(matches) == 0:
        return None, None
    if len(matches) > 1:
        raise RuntimeError(f"Multiple bootstrap AI files found in {result_dir}: {[p.name for p in matches]}")
    path = matches[0]
    prefix = path.name[: -len("_pred_AI_bootstrap.npy")]
    return path, prefix


def sanitize_output_folder(result_dir: Path) -> str:
    return f"{result_dir.parent.name}__{result_dir.name}"


def to_hilxl(prob_nkh: np.ndarray, il: int, xl: int, h: int) -> np.ndarray:
    n, k, hs = prob_nkh.shape
    if n != il * xl or hs != h:
        raise ValueError(f"Unexpected facies prob shape {prob_nkh.shape}; expected N={il * xl}, H={h}")
    return np.transpose(prob_nkh.reshape(il, xl, k, h), (3, 0, 1, 2)).astype(np.float32, copy=False)


def to_hilxl_scalar(arr_nh: np.ndarray, il: int, xl: int, h: int) -> np.ndarray:
    n, hs = arr_nh.shape
    if n != il * xl or hs != h:
        raise ValueError(f"Unexpected scalar field shape {arr_nh.shape}; expected N={il * xl}, H={h}")
    return np.transpose(arr_nh.reshape(il, xl, h), (2, 0, 1)).astype(np.float32, copy=False)


def compute_channel_fields(
    facies_prob_hilxlk: np.ndarray,
    *,
    channel_id: int,
    conf_thresh: float,
    neutral_p: float,
) -> dict[str, np.ndarray]:
    p_pred = facies_prob_hilxlk[..., int(channel_id)].astype(np.float32, copy=False)
    conf = facies_prob_hilxlk.max(axis=-1).astype(np.float32, copy=False)
    p_mix = p_pred.astype(np.float32, copy=False)
    p_fallback = np.full_like(p_pred, float(neutral_p), dtype=np.float32)
    p_channel = np.where(conf >= float(conf_thresh), p_mix, p_fallback).clip(0.0, 1.0).astype(np.float32)
    facies_pred = np.argmax(facies_prob_hilxlk, axis=-1).astype(np.int64)
    conf_gate = (conf >= float(conf_thresh)).astype(np.float32)
    return {
        "p_pred_hilxl": p_pred,
        "conf_hilxl": conf,
        "p_mix_hilxl": p_mix,
        "p_fallback_hilxl": p_fallback,
        "p_channel_hilxl": p_channel,
        "facies_pred_hilxl": facies_pred,
        "conf_gate_hilxl": conf_gate,
    }


def prepare_bootstrap_ai(cfg: dict, result_dir: Path, device: torch.device) -> tuple[np.ndarray, str]:
    bootstrap_path, prefix = find_existing_bootstrap_ai(result_dir)
    if bootstrap_path is not None and prefix is not None:
        ai_boot = np.load(bootstrap_path).astype(np.float32, copy=False)
        return ai_boot, prefix

    run_id, model_name = resolve_run_id_and_model_name(cfg)
    data_flag = str(cfg["data_flag"])
    seismic_raw, model_raw, _, meta = get_data_raw(data_flag=data_flag)
    noise_kind = str(cfg.get("test_noise_kind", "none")).strip().lower()
    noise_snr_db = cfg.get("test_noise_snr_db", None)
    noise_seed = int(cfg.get("test_noise_seed", 2026))
    seismic_raw, _, _ = apply_test_noise(
        seismic_raw=seismic_raw,
        meta=meta,
        noise_kind=noise_kind,
        snr_db=noise_snr_db,
        seed=noise_seed,
    )
    stats = load_stats_strict(run_id=run_id, data_flag=data_flag)
    seismic_std, _, _ = standardize(seismic_raw, model_raw, stats=stats)
    s_len = seismic_std.shape[-1]
    n = int((s_len // 8) * 8)
    seismic_std = seismic_std[:, :n]
    seismic = seismic_std[:, np.newaxis, :].astype(np.float32)
    z = np.zeros_like(seismic[:, :1, :], dtype=np.float32)
    seismic_boot = np.concatenate([seismic, z], axis=1)

    ckpt_path = REPO_ROOT / "save_train_model" / f"{model_name}_{data_flag}.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    inverse_model = torch.load(ckpt_path, map_location=device).to(device)
    ai_boot = infer_inverse_all(
        inverse_model=inverse_model,
        seismic_np=seismic_boot,
        device=device,
        batch_size=int(cfg.get("test_init_bs", 32)),
    )
    prefix = f"{model_name}_{data_flag}"
    return ai_boot, prefix


def export_channel_arrays(result_dir: Path, output_dir: Path | None = None) -> Path:
    cfg = merge_configs(result_dir)
    run_id, _ = resolve_run_id_and_model_name(cfg)
    data_flag = str(cfg["data_flag"])
    channel_id = int(cfg.get("channel_id", 2))
    conf_thresh = float(cfg.get("aniso_closed_loop_conf_thresh", cfg.get("conf_thresh", 0.60)))
    neutral_p = float(cfg.get("agpe_neutral_p", cfg.get("neutral_p", 0.5)))

    result_dir = result_dir.resolve()
    if output_dir is None:
        output_dir = REPO_ROOT / "visualize_3d" / "_outputs" / "farp_channel_arrays" / sanitize_output_folder(result_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seismic_raw, model_raw, _, meta = get_data_raw(data_flag=data_flag)
    stats = load_stats_strict(run_id=run_id, data_flag=data_flag)
    _, model_std, _ = standardize(seismic_raw, model_raw, stats=stats)
    h = int((model_std.shape[-1] // 8) * 8)
    il = int(meta["inline"])
    xl = int(meta["xline"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ai_boot, prefix = prepare_bootstrap_ai(cfg=cfg, result_dir=result_dir, device=device)
    ai_boot = ai_boot[:, :h].astype(np.float32, copy=False)

    full_ckpt = load_full_ckpt_strict(run_id=run_id, data_flag=data_flag, device=device)
    facies_model = Facies_model_class(facies_n=int(cfg.get("facies_n", 4))).to(device)
    facies_model.load_state_dict(full_ckpt["facies_state_dict"], strict=True)
    facies_model.eval()

    facies_prob_nkh = infer_facies_prob_all(
        facies_model=facies_model,
        ai_np=ai_boot,
        device=device,
        batch_size=int(cfg.get("test_facies_bs", 64)),
    )
    facies_prob_hilxlk = to_hilxl(facies_prob_nkh, il=il, xl=xl, h=h)
    fields = compute_channel_fields(
        facies_prob_hilxlk=facies_prob_hilxlk,
        channel_id=channel_id,
        conf_thresh=conf_thresh,
        neutral_p=neutral_p,
    )
    ai_boot_hilxl = to_hilxl_scalar(ai_boot, il=il, xl=xl, h=h)

    np.save(output_dir / f"{prefix}_facies_prob_HILXLK.npy", facies_prob_hilxlk.astype(np.float32))
    np.save(output_dir / f"{prefix}_facies_pred_HILXL.npy", fields["facies_pred_hilxl"].astype(np.int64))
    np.save(output_dir / f"{prefix}_p_pred_HILXL.npy", fields["p_pred_hilxl"].astype(np.float32))
    np.save(output_dir / f"{prefix}_conf_HILXL.npy", fields["conf_hilxl"].astype(np.float32))
    np.save(output_dir / f"{prefix}_conf_gate_HILXL.npy", fields["conf_gate_hilxl"].astype(np.float32))
    np.save(output_dir / f"{prefix}_p_mix_HILXL.npy", fields["p_mix_hilxl"].astype(np.float32))
    np.save(output_dir / f"{prefix}_p_fallback_HILXL.npy", fields["p_fallback_hilxl"].astype(np.float32))
    np.save(output_dir / f"{prefix}_p_channel_HILXL.npy", fields["p_channel_hilxl"].astype(np.float32))
    np.save(output_dir / f"{prefix}_pred_AI_bootstrap_HILXL.npy", ai_boot_hilxl.astype(np.float32))

    meta_out = {
        "result_dir": str(result_dir),
        "output_dir": str(output_dir),
        "run_id": run_id,
        "data_flag": data_flag,
        "channel_id": channel_id,
        "conf_thresh": conf_thresh,
        "neutral_p": neutral_p,
        "protocol_note": "Matches test_3D closed-loop path: facies head softmax on bootstrap AI, then p_channel = where(conf >= conf_thresh, p_pred, neutral_p).",
        "shapes": {
            "facies_prob_HILXLK": list(facies_prob_hilxlk.shape),
            "p_channel_HILXL": list(fields["p_channel_hilxl"].shape),
            "pred_AI_bootstrap_HILXL": list(ai_boot_hilxl.shape),
        },
        "source_files": {
            "test_config": str(result_dir / "test_config.json"),
            "train_config": str(result_dir / "train_config.json"),
            "full_ckpt": str(REPO_ROOT / "save_train_model" / f"{run_id}_full_ckpt_{data_flag}.pth"),
        },
        "suggested_notebook_cfg_for_chain": {
            "facies_prob_path": str(output_dir / f"{prefix}_facies_prob_HILXLK.npy"),
            "p_channel_path": None,
            "conf_path": None,
            "alpha_prior": float(cfg.get("aniso_closed_loop_alpha_prior", 0.0)),
            "conf_thresh": conf_thresh,
            "neutral_p": neutral_p,
            "note": "Use facies_prob_path only if you want visualize_farp_similarity.ipynb to reconstruct p_pred/conf/p_mix/p_channel."
        },
        "optional_direct_arrays": {
            "p_pred_path": str(output_dir / f"{prefix}_p_pred_HILXL.npy"),
            "conf_path": str(output_dir / f"{prefix}_conf_HILXL.npy"),
            "p_channel_path": str(output_dir / f"{prefix}_p_channel_HILXL.npy"),
        },
    }
    (output_dir / f"{prefix}_channel_meta.json").write_text(json.dumps(meta_out, indent=2, ensure_ascii=True), encoding="utf-8")
    return output_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export facies probability / p_channel / conf arrays for FARP visualization.")
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="Result directory containing train_config.json and test_config.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional custom output directory. Default: visualize_3d/_outputs/farp_channel_arrays/<result>",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    out = export_channel_arrays(result_dir=args.result_dir, output_dir=args.output_dir)
    print(f"[EXPORT] channel arrays written to: {out}")


if __name__ == "__main__":
    main()
