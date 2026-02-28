"""
test_3D.py (FINAL, drop-in replacement)

Capabilities:
1) Load Stanford VI / Fanny datasets.
2) Load trained checkpoints and run full-volume inference.
3) Report R2 / PCC / SSIM / PSNR / MSE / MAE / MedAE.
4) Save visualizations to results/:
   - Pred/True xline=50, inline=100, depth slices 40/100/160
   - Seismic xline=50 and single-trace comparison
5) If use_aniso_conditioning=True, save R(x) slices + R3D.npy.

Usage (VSCode / F5):
- Configure TCN1D_test_p in setting.py:
  - data_flag='Stanford_VI'
  - model_name='VishalNet_cov_para_Facies_s_uns'
  - no_wells=20
  - use_aniso_conditioning=True/False
"""

import os
import errno
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import ndimage

import torch
from os.path import join
from torch.utils.data import DataLoader

from setting import *
from utils.utils import standardize
from utils.datasets import SeismicDataset1D

from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
import cv2  # Keep cv2 import style to avoid from-import issues across environments.
# anisotropic reliability (FARP)
from utils.reliability_aniso import build_R_and_prior_from_cube

def _infer_run_id_for_full_ckpt(model_name: str) -> str:
    """
    Your repo saves full_ckpt as:
      {run_id}_full_ckpt_{data_flag}.pth
    while inference ckpt may be:
      {run_id}_s_uns_{data_flag}.pth
    We therefore try to map 'model_name' -> 'run_id'.
    """
    run_id = model_name
    # common suffixes in this repo
    for suf in ["_s_uns", "_uns", "_s", "_best", "_final"]:
        if run_id.endswith(suf):
            run_id = run_id[: -len(suf)]
    return run_id


def _resolve_run_id_and_model_name(cfg: dict) -> tuple[str, str]:
    """Resolve a stable run_id / model_name chain with optional run_id_suffix."""
    suffix = str(cfg.get("run_id_suffix", "") or "")
    if "run_id" in cfg and cfg["run_id"] is not None and str(cfg["run_id"]).strip() != "":
        run_id_base = str(cfg["run_id"]).strip()
    else:
        run_id_base = _infer_run_id_for_full_ckpt(str(cfg.get("model_name", "")))
    run_id = run_id_base if (suffix == "" or run_id_base.endswith(suffix)) else f"{run_id_base}{suffix}"
    model_name = f"{run_id}_s_uns"
    return run_id, model_name


def load_stats_strict(run_id: str, data_flag: str) -> dict:
    """
    Strict stats loading priority:
      1) full_ckpt: save_train_model/{run_id}_full_ckpt_{data_flag}.pth  (BEST)
      2) dedicated: save_train_model/norm_stats_{run_id}_{data_flag}.npy (if you enable in training)
      3) legacy   : save_train_model/norm_stats_{data_flag}.npy          (FALLBACK ONLY)

    This fully eliminates 'norm_stats_{data_flag}.npy' being overwritten by other runs.
    """
    full_ckpt_path = join("save_train_model", f"{run_id}_full_ckpt_{data_flag}.pth")

    # 1) from full_ckpt
    if os.path.isfile(full_ckpt_path):
        ckpt = torch.load(full_ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and ("stats" in ckpt) and (ckpt["stats"] is not None):
            print(f"[NORM] stats loaded from full_ckpt: {full_ckpt_path}")
            return ckpt["stats"]
        raise RuntimeError(f"[NORM][ERROR] full_ckpt exists but stats missing: {full_ckpt_path}")

    # 2) dedicated stats file (optional, if you save it during training)
    p2 = join("save_train_model", f"norm_stats_{run_id}_{data_flag}.npy")
    if os.path.isfile(p2):
        print(f"[NORM] stats loaded from dedicated file: {p2}")
        return np.load(p2, allow_pickle=True).item()

    # 3) legacy fallback (may mismatch)
    p3 = join("save_train_model", f"norm_stats_{data_flag}.npy")
    if os.path.isfile(p3):
        print(f"[NORM][WARN] using legacy stats (may mismatch): {p3}")
        return np.load(p3, allow_pickle=True).item()

    raise FileNotFoundError(
        f"Cannot find stats for run_id={run_id}, data_flag={data_flag}\n"
        f"Tried:\n  {full_ckpt_path}\n  {p2}\n  {p3}"
    )

# -----------------------------
# Utility functions
# -----------------------------
def load_selected_wells_trace_indices(
    csv_path: str,
    IL: int,
    XL: int,
    no_wells: int = 20,
    seed: int = 2026,
):
    """
    Read selected wells CSV (expects columns INLINE, XLINE) and convert to trace indices.

    Trace index convention MUST match your flatten order:
      idx = inline * XL + xline
    which is consistent with your get_data_raw flatten:
      seismic3d (H,IL,XL) -> reshape(H, IL*XL) with XL fastest.

    Returns:
      traces_well: np.ndarray shape (no_wells,) dtype int64
    """
    import csv

    if (csv_path is None) or (not os.path.isfile(csv_path)):
        raise FileNotFoundError(f"selected_wells_csv not found: {csv_path}")

    ils = []
    xls = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if ("INLINE" not in reader.fieldnames) or ("XLINE" not in reader.fieldnames):
            raise ValueError(
                f"CSV must contain INLINE and XLINE columns. Got columns: {reader.fieldnames}"
            )
        for row in reader:
            try:
                ils.append(int(float(row["INLINE"])))
                xls.append(int(float(row["XLINE"])))
            except Exception:
                continue

    if len(ils) == 0:
        raise ValueError(f"No valid wells parsed from: {csv_path}")

    il = np.clip(np.asarray(ils, dtype=np.int64), 0, IL - 1)
    xl = np.clip(np.asarray(xls, dtype=np.int64), 0, XL - 1)

    traces = (il * XL + xl).astype(np.int64)
    traces = np.unique(traces)

    if no_wells is not None and len(traces) != int(no_wells):
        rng = np.random.default_rng(int(seed))
        if len(traces) > int(no_wells):
            traces = rng.choice(traces, size=int(no_wells), replace=False).astype(np.int64)
        else:
            print(f"[WELLS][WARN] CSV has {len(traces)} wells < requested {no_wells}. Using {len(traces)}.")

    return traces

def _ensure_dir(path: str) -> None:
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


def _percentile_vminmax(arr: np.ndarray, p_low=1, p_high=99):
    """Compute robust vmin/vmax using percentiles."""
    vmin = np.percentile(arr, p_low)
    vmax = np.percentile(arr, p_high)
    return float(vmin), float(vmax)


def get_data_raw(data_flag='Stanford_VI'):
    """
    Load raw data only (no standardization or cropping).
    Returns:
      seismic_raw: (N, H) raw seismic
      model_raw  : (N, H) raw target model
      facies_raw : (N, H) raw facies
      meta       : dict(H, inline, xline, seismic3d, model3d, facies3d)
    """
    meta = {}

    if data_flag == 'Stanford_VI':
        # Load raw 3D cubes with shape (H, IL, XL).
        seismic3d = np.load(join('data', data_flag, 'synth_40HZ.npy'))
        model3d = np.load(join('data', data_flag, 'AI.npy'))
        facies3d = np.load(join('data', data_flag, 'Facies.npy'))

        H, IL, XL = seismic3d.shape
        meta = {
            'H': H, 'inline': IL, 'xline': XL,
            'seismic3d': seismic3d, 'model3d': model3d, 'facies3d': facies3d
        }

        # Flatten to trace dimension: (H, IL*XL) -> (IL*XL, H)
        seismic_raw = np.transpose(seismic3d.reshape(H, IL * XL), (1, 0))
        model_raw = np.transpose(model3d.reshape(H, IL * XL), (1, 0))
        facies_raw = np.transpose(facies3d.reshape(H, IL * XL), (1, 0))

        print(f"[{data_flag}] raw shapes: model={model_raw.shape}, seismic={seismic_raw.shape}, facies={facies_raw.shape}")
        print(f"[{data_flag}] raw means: model={float(model_raw.mean()):.4f}, seismic={float(seismic_raw.mean()):.4f}")

    elif data_flag == 'Fanny':
        # Fanny raw-data loading branch (kept consistent with training logic).
        seismic_raw = np.load(join('data', data_flag, 'seismic.npy'))
        GR_raw = np.load(join('data', data_flag, 'GR.npy'))
        model_raw = np.load(join('data', data_flag, 'Impedance.npy'))
        facies_raw = np.load(join('data', data_flag, 'facies.npy'))
        # Use GR as inversion target (consistent with training).
        model_raw = GR_raw
        # Infer Fanny inline/xline assuming a square grid of traces.
        n_traces = model_raw.shape[0]
        IL = XL = int(np.sqrt(n_traces))
        meta = {
            'H': model_raw.shape[-1], 'inline': IL, 'xline': XL,
            'seismic3d': seismic_raw.reshape(IL, XL, -1).transpose(2,0,1),  # Adapt to (H, IL, XL)
            'model3d': model_raw.reshape(IL, XL, -1).transpose(2,0,1)
        }
        print(f"[{data_flag}] raw shapes: model={model_raw.shape}, seismic={seismic_raw.shape}")

    else:
        raise ValueError(f"Unsupported dataset: {data_flag}")

    return seismic_raw, model_raw, facies_raw, meta


def show_stanford_vi(
    AI_act_flat: np.ndarray,
    AI_pred_flat: np.ndarray,
    seismic_flat: np.ndarray,
    meta: dict,
    out_prefix: str,
    xline_pick: int = 50,
    inline_pick: int = 100,
    depth_slices=(40, 100, 160),
):
    """
    Visualize Stanford VI predictions/ground truth slices and sections.
    Adapt to the verified reshape order to avoid stripe artifacts.
    """
    _ensure_dir("results")

    H = int(meta["H"])
    IL = int(meta["inline"])
    XL = int(meta["xline"])
    # Use reshape order inferred by the sanity-check; default to C-order.
    reshape_order = meta.get("reshape_order", "C")

    # Reshape back to 3D using the validated reshape order.
    AI_act = AI_act_flat.reshape(IL, XL, H, order=reshape_order)
    AI_pred = AI_pred_flat.reshape(IL, XL, H, order=reshape_order)

    # Reshape seismic input consistently.
    if seismic_flat.ndim == 3:
        seis_amp = seismic_flat[:, 0, :].reshape(IL, XL, H, order=reshape_order)
    else:
        seis_amp = seismic_flat.reshape(IL, XL, H, order=reshape_order)

    # Add light texture noise for clearer seismic visualization.
    blurred = ndimage.gaussian_filter(seis_amp, sigma=1.1)
    seis_plot = blurred + 0.5 * blurred.std() * np.random.random(blurred.shape)

    # Convert index axes to distance (25 m per inline/xline step).
    il_dist = IL * 25
    xl_dist = XL * 25

    # Use robust color limits from ground truth percentiles.
    vmin, vmax = _percentile_vminmax(AI_act, 1, 99)

    # -------- Pred xline section --------
    xline_pick = int(np.clip(xline_pick, 0, XL - 1))
    fig, ax = plt.subplots(figsize=(16, 8), dpi=400)
    ax.imshow(AI_pred[:, xline_pick, :].T, vmin=vmin, vmax=vmax, extent=(0, il_dist, H, 0))
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.tick_params(axis="both", labelsize=18)
    ax.set_aspect(80 / 10)
    ax.set_xlabel("Distance (m)", fontsize=20)
    ax.set_ylabel("Depth (m)", fontsize=20)
    ax.set_title(f"Inversion Profile (Pred) | xline={xline_pick}", fontsize=20)
    plt.savefig(f"results/{out_prefix}_Pred_xline_{xline_pick}.png", bbox_inches='tight')
    plt.close()

    # -------- Pred inline section --------
    inline_pick = int(np.clip(inline_pick, 0, IL - 1))
    fig, ax = plt.subplots(figsize=(16, 8), dpi=400)
    ax.imshow(AI_pred[inline_pick, :, :].T, vmin=vmin, vmax=vmax, extent=(0, xl_dist, H, 0))
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.tick_params(axis="both", labelsize=18)
    ax.set_aspect(45 / 10)
    ax.set_xlabel("Distance (m)", fontsize=20)
    ax.set_ylabel("Depth (m)", fontsize=20)
    ax.set_title(f"Inversion Profile (Pred) | inline={inline_pick}", fontsize=20)
    plt.savefig(f"results/{out_prefix}_Pred_inline_{inline_pick}.png", bbox_inches='tight')
    plt.close()

    # -------- Pred depth slice --------
    for z in depth_slices:
        z = int(np.clip(z, 0, H - 1))
        fig, ax = plt.subplots(figsize=(16, 8), dpi=400)
        ax.imshow(AI_pred[:, :, z], vmin=vmin, vmax=vmax, extent=(0, xl_dist, 0, il_dist))
        ax.xaxis.set_major_locator(MultipleLocator(1000))
        ax.yaxis.set_major_locator(MultipleLocator(1000))
        ax.tick_params(axis="both", labelsize=18)
        ax.set_xlabel("x(m)", fontsize=20)
        ax.set_ylabel("y(m)", fontsize=20)
        ax.set_title(f"Inversion Slice (Pred) | depth={z}", fontsize=20)
        plt.savefig(f"results/{out_prefix}_Pred_depth_{z}.png", bbox_inches='tight')
        plt.close()

    # -------- True xline section --------
    fig, ax = plt.subplots(figsize=(16, 8), dpi=400)
    ax.imshow(AI_act[:, xline_pick, :].T, vmin=vmin, vmax=vmax, extent=(0, il_dist, H, 0))
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.tick_params(axis="both", labelsize=18)
    ax.set_aspect(80 / 10)
    ax.set_xlabel("Distance (m)", fontsize=20)
    ax.set_ylabel("Depth (m)", fontsize=20)
    ax.set_title(f"Ground Truth | xline={xline_pick}", fontsize=20)
    plt.savefig(f"results/{out_prefix}_True_xline_{xline_pick}.png", bbox_inches='tight')
    plt.close()

    # -------- True inline section --------
    fig, ax = plt.subplots(figsize=(16, 8), dpi=400)
    ax.imshow(AI_act[inline_pick, :, :].T, vmin=vmin, vmax=vmax, extent=(0, xl_dist, H, 0))
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.tick_params(axis="both", labelsize=18)
    ax.set_aspect(45 / 10)
    ax.set_xlabel("Distance (m)", fontsize=20)
    ax.set_ylabel("Depth (m)", fontsize=20)
    ax.set_title(f"Ground Truth | inline={inline_pick}", fontsize=20)
    plt.savefig(f"results/{out_prefix}_True_inline_{inline_pick}.png", bbox_inches='tight')
    plt.close()

    # -------- True depth slice --------
    for z in depth_slices:
        z = int(np.clip(z, 0, H - 1))
        fig, ax = plt.subplots(figsize=(16, 8), dpi=400)
        ax.imshow(AI_act[:, :, z], vmin=vmin, vmax=vmax, extent=(0, xl_dist, 0, il_dist))
        ax.xaxis.set_major_locator(MultipleLocator(1000))
        ax.yaxis.set_major_locator(MultipleLocator(1000))
        ax.tick_params(axis="both", labelsize=18)
        ax.set_xlabel("x(m)", fontsize=20)
        ax.set_ylabel("y(m)", fontsize=20)
        ax.set_title(f"Ground Truth | depth={z}", fontsize=20)
        plt.savefig(f"results/{out_prefix}_True_depth_{z}.png", bbox_inches='tight')
    plt.close()

    # -------- Seismic xline section --------
    fig, ax = plt.subplots(figsize=(16, 8), dpi=400)
    ax.imshow(seis_plot[:, xline_pick, :].T, cmap="seismic", extent=(0, il_dist, H, 0))
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.tick_params(axis="both", labelsize=18)
    ax.set_aspect(80 / 10)
    ax.set_xlabel("Distance (m)", fontsize=20)
    ax.set_ylabel("Depth (m)", fontsize=20)
    ax.set_title(f"Seismic Profile | xline={xline_pick}", fontsize=20)
    plt.savefig(f"results/{out_prefix}_Seismic_xline_{xline_pick}.png", bbox_inches='tight')
    plt.close()

    # -------- Single-trace comparison --------
    B_x, B_y = inline_pick, xline_pick
    depth_index = np.arange(H) * 1.0

    fig, ax = plt.subplots(figsize=(16, 6), dpi=400)
    ax.plot(depth_index, AI_pred[B_x, B_y, :], linestyle="--", label="Pred", linewidth=3.0, color='red')
    ax.plot(depth_index, AI_act[B_x, B_y, :], linestyle="-", label="True", linewidth=3.0, color='blue')
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.tick_params(axis="both", labelsize=18)
    ax.set_xlabel("Depth(m)", fontsize=20)
    ax.set_ylabel("Impedance (standardized)", fontsize=20)
    ax.set_title(f"Trace inline={B_x}, xline={B_y}", fontsize=20)
    plt.legend(loc="upper left", fontsize=16)
    plt.savefig(f"results/{out_prefix}_Trace_inline_{B_x}_xline_{B_y}.png", bbox_inches='tight')
    plt.close()


def save_R_visualization(R_flat_torch: torch.Tensor, meta: dict, out_prefix: str, depth_slices=(40, 100, 160)):
    """
    Visualize and save R(x) 3D data and depth slices with consistent reshape order.
    """
    _ensure_dir("results")
    H = int(meta["H"])
    IL = int(meta["inline"])
    XL = int(meta["xline"])
    reshape_order = meta.get("reshape_order", "C")

    # Reshape R data using the validated order.
    R3d = R_flat_torch.detach().cpu().numpy().reshape(IL, XL, H, order=reshape_order)
    # Save both IL-XL-H and H-IL-XL layouts.
    np.save(f"results/{out_prefix}_R3d_ILXLH.npy", R3d)
    np.save(f"results/{out_prefix}_R3d_HILXL.npy", np.transpose(R3d, (2, 0, 1)))

    # Save depth slices of R(x).
    for z in depth_slices:
        z = int(np.clip(z, 0, H - 1))
        plt.figure(figsize=(6, 5), dpi=250)
        plt.imshow(R3d[:, :, z], cmap="hot")
        plt.colorbar(label="Reliability R(x)")
        plt.title(f"Anisotropic reliability R | depth={z}")
        plt.tight_layout()
        plt.savefig(f"results/{out_prefix}_R_depth_{z}.png")
        plt.close()


# -----------------------------
# Main test function
def test(test_p: dict) -> dict:
    _ensure_dir("results")
    _ensure_dir("save_train_model")

    # Read config.
    cfg = {**TCN1D_train_p, **test_p}
    run_id, model_name = _resolve_run_id_and_model_name(cfg)
    data_flag = cfg["data_flag"]
    no_wells = int(cfg.get("no_wells", 20))
    print(f"[TEST] resolved run_id={run_id} model_name={model_name}")
    print(f"[AGPE] backend={cfg.get('aniso_backend', 'grid')} use_tensor_strength={bool(cfg.get('aniso_use_tensor_strength', False))}")

    # Merge train/test config (test fields override training defaults).
    # cfg already merged above

    ### 1. Load raw data (no standardization/cropping yet)
    seismic_raw, model_raw, facies_raw, meta = get_data_raw(data_flag=data_flag)

    ### 2. Trace-order sanity check
    IL, XL, H = meta["inline"], meta["xline"], meta["H"]
    # Verify flatten/reshape reversibility under current order.
    model_3d = model_raw.reshape(IL, XL, H)
    model_back = model_3d.reshape(IL * XL, H)
    reshape_err = np.abs(model_back - model_raw).max()
    print(f"[SANITY CHECK] flatten/reshape max error: {reshape_err:.6f}")
    
    # If needed, auto-switch between C/F reshape orders.
    if reshape_err > 1e-10:
        print("[WARNING] flatten/reshape is not reversible; trying Fortran order (order='F').")
        model_3d_F = model_raw.reshape(IL, XL, H, order='F')
        model_back_F = model_3d_F.reshape(IL * XL, H, order='F')
        reshape_err_F = np.abs(model_back_F - model_raw).max()
        print(f"[SANITY CHECK] max error after Fortran-order check: {reshape_err_F:.6f}")
        
        if reshape_err_F < 1e-10:
            meta["reshape_order"] = "F"
            print("[SANITY CHECK] fixed: using Fortran-order reshape.")
        else:
            raise RuntimeError(
                f"flatten/reshape remains non-reversible: raw_err={reshape_err:.6f}, fixed_err={reshape_err_F:.6f}\n"
                f"Please verify dims: IL={IL}, XL={XL}, H={H}, model_raw.shape={model_raw.shape}"
            )
    else:
        meta["reshape_order"] = "C"
        print("[SANITY CHECK] flatten/reshape reversible; using C-order.")

    ### 3. Load training normalization stats (prefer full_ckpt to avoid mismatches)
    stats = load_stats_strict(run_id=run_id, data_flag=data_flag)
    print(f"[NORM] stats loaded | mode={stats.get('mode')} | keys={list(stats.keys())}")

    ### 4. Standardize using training stats
    seismic, model, _ = standardize(seismic_raw, model_raw, stats=stats)

    ### 5. Crop to multiples of 8 (UNet/TCN downsampling compatibility)
    s_L = seismic.shape[-1]
    n = int((s_L // 8) * 8)
    seismic = seismic[:, :n]
    model = model[:, :n]
    facies = facies_raw[:, :n]

    ### 6. Add channel dimension (same as training)
    seismic = seismic[:, np.newaxis, :].astype(np.float32)  # (N,1,H)
    model = model[:, np.newaxis, :].astype(np.float32)
    facies = facies[:, np.newaxis, :]

    ### 7. Build anisotropic R channel (if enabled)
    R_flat = None
    if cfg.get("use_aniso_conditioning", False) and data_flag == "Stanford_VI":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use selected real well locations from CSV as AGPE seeds.
        seed = int(cfg.get("seed", 2026))
        csv_path = cfg.get("selected_wells_csv", None)
        
        traces_well = load_selected_wells_trace_indices(
            csv_path=csv_path,
            IL=int(meta["inline"]),
            XL=int(meta["xline"]),
            no_wells=int(no_wells),
            seed=seed,
        )
        print(f"[WELLS] using selected wells from CSV: {csv_path} | count={len(traces_well)}")
        np.save(f"results/{model_name}_{data_flag}_well_trace_indices.npy", traces_well)

        well_idx = torch.from_numpy(traces_well).to(device)
        # Move 3D cubes to device.
        seis3d = torch.from_numpy(meta["seismic3d"]).to(device=device, dtype=torch.float32)
        fac3d = torch.from_numpy(meta["facies3d"]).to(device=device, dtype=torch.long)
        ai3d = torch.from_numpy(meta["model3d"]).to(device=device, dtype=torch.float32)

        # Build R(x).
        R_flat, _ = build_R_and_prior_from_cube(
            seismic_3d=seis3d,
            facies_3d=fac3d,
            ai_3d=ai3d,
            well_trace_indices=well_idx,
            channel_id=int(cfg.get("channel_id", 2)),
            steps_R=int(cfg.get("aniso_steps_R", 25)),
            eta=float(cfg.get("aniso_eta", 0.6)),
            gamma=float(cfg.get("aniso_gamma", 8.0)),
            tau=float(cfg.get("aniso_tau", 0.6)),
            kappa=float(cfg.get("aniso_kappa", 4.0)),
            sigma_st=float(cfg.get("aniso_sigma_st", 1.2)),
            backend=str(cfg.get("aniso_backend", "grid")),
            aniso_use_tensor_strength=bool(cfg.get("aniso_use_tensor_strength", False)),
            aniso_tensor_strength_power=float(cfg.get("aniso_tensor_strength_power", 1.0)),
            agpe_skel_p_thresh=float(cfg.get("agpe_skel_p_thresh", 0.60)),
            agpe_skel_min_nodes=int(cfg.get("agpe_skel_min_nodes", 30)),
            agpe_skel_snap_radius=int(cfg.get("agpe_skel_snap_radius", 5)),
            agpe_long_edges=bool(cfg.get("agpe_long_edges", True)),
            agpe_long_max_step=int(cfg.get("agpe_long_max_step", 6)),
            agpe_long_step=int(cfg.get("agpe_long_step", 2)),
            agpe_long_cos_thresh=float(cfg.get("agpe_long_cos_thresh", 0.70)),
            agpe_long_weight=float(cfg.get("agpe_long_weight", 0.50)),
            agpe_edge_tau_p=float(cfg.get("agpe_edge_tau_p", 0.25)),
            agpe_lift_sigma=float(cfg.get("agpe_lift_sigma", 3.0)),
            use_soft_prior=False,
        )

        # Append anisotropic reliability channel R(x) to seismic input.
        R_np = R_flat.detach().cpu().numpy()[:, np.newaxis, :].astype(np.float32)
        seismic = np.concatenate([seismic, R_np], axis=1)  # (N,2,H)

        # Save R(x) visualization.
        out_prefix = f"{model_name}_{data_flag}"
        save_R_visualization(R_flat, meta, out_prefix, depth_slices=(40, 100, 160))

    print(f"[TEST] final input shapes: seismic={seismic.shape}, model={model.shape}")

    ### 8. Build DataLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    traces_test = np.arange(len(model), dtype=int)
    test_dataset = SeismicDataset1D(seismic, model, traces_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    ### 9. Load model
    ckpt_path = f"save_train_model/{model_name}_{data_flag}.pth"
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), ckpt_path)
    print(f"[TEST] loading model: {ckpt_path}")
    inver_model = torch.load(ckpt_path, map_location=device).to(device)

    ### 10. Full-volume inference
    print("[TEST] inferencing...")
    x0, y0 = test_dataset[0]
    H = y0.shape[-1]

    AI_pred = torch.zeros((len(test_dataset), H), dtype=torch.float32, device=device)
    AI_act = torch.zeros((len(test_dataset), H), dtype=torch.float32, device=device)

    mem = 0
    with torch.no_grad():
        inver_model.eval()
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = inver_model(x)

            bs = x.shape[0]
            # Handle both (B,1,H) and (B,H) model outputs.
            AI_pred[mem:mem + bs] = y_pred.squeeze(1) if y_pred.ndim == 3 else y_pred
            AI_act[mem:mem + bs] = y.squeeze(1) if y.ndim == 3 else y
            mem += bs

    # Convert to numpy.
    AI_pred_np = AI_pred.detach().cpu().numpy()
    AI_act_np = AI_act.detach().cpu().numpy()

    # Save predicted/ground-truth arrays.
    out_prefix = f"{model_name}_{data_flag}"
    np.save(f"results/{out_prefix}_pred_AI.npy", AI_pred_np)
    np.save(f"results/{out_prefix}_true_AI.npy", AI_act_np)

    ### 11. Compute quantitative metrics
    print("\n[TEST] quantitative metrics:")
    # R2 score
    r2 = r2_score(AI_act_np.ravel(), AI_pred_np.ravel())
    print(f"  R2        : {r2:.4f}")
    
    # Pearson correlation coefficient (PCC).
    pcc, p_value = pearsonr(AI_act_np.ravel(), AI_pred_np.ravel())
    print(f"  PCC       : {pcc:.4f} (p-value: {p_value:.2e})")
    
    # SSIM (structural similarity)
    dr = AI_act_np.max() - AI_act_np.min() + 1e-12
    ssim_score = ssim(AI_act_np.T, AI_pred_np.T, data_range=dr)
    print(f"  SSIM      : {ssim_score:.4f}")
    
    # MSE/MAE/MedAE
    mse = np.mean((AI_pred_np - AI_act_np) ** 2)
    mae = np.mean(np.abs(AI_pred_np - AI_act_np))
    medae = np.median(np.abs(AI_pred_np - AI_act_np))
    print(f"  MSE       : {mse:.4f}")
    print(f"  MAE       : {mae:.4f}")
    print(f"  MedAE     : {medae:.4f}")
    
    # PSNR (peak signal-to-noise ratio).
    dr_full = max(AI_act_np.max(), AI_pred_np.max()) - min(AI_act_np.min(), AI_pred_np.min()) + 1e-12
    psnr = 20 * np.log10(dr_full) - 10 * np.log10(mse + 1e-12)
    print(f"  PSNR      : {psnr:.4f} dB")

    metrics = {
        "run_id": run_id,
        "model_name": model_name,
        "data_flag": data_flag,
        "r2": float(r2),
        "pcc": float(pcc),
        "p_value": float(p_value),
        "ssim": float(ssim_score),
        "mse": float(mse),
        "mae": float(mae),
        "medae": float(medae),
        "psnr": float(psnr),
    }

    # 12. Visualization
    if data_flag == "Stanford_VI":
        show_stanford_vi(
            AI_act_flat=AI_act_np,
            AI_pred_flat=AI_pred_np,
            seismic_flat=seismic,
            meta=meta,
            out_prefix=out_prefix,
            xline_pick=50,
            inline_pick=100,
            depth_slices=(40, 100, 160),
        )
        print(f"\n[TEST] visualizations saved to results/ with prefix: {out_prefix}")
    else:
        print("\n[TEST] Fanny visualization is not expanded in this script.")


    return metrics

if __name__ == "__main__":
    test(TCN1D_test_p)

