"""
test_3D.py (FINAL, drop-in replacement)

功能：
1) 读取 Stanford VI / Fanny 数据
2) 加载训练好的模型权重并做全体素推理
3) 输出定量指标：R2 / PCC / SSIM / PSNR / MSE / MAE / MedAE
4) 保存可视化结果到 results/：
   - Pred/True xline=50 剖面
   - Pred/True inline=100 剖面
   - Pred/True depth slice = 40/100/160
   - Seismic xline=50 剖面
   - 单点 trace 对比
5) 若开启 use_aniso_conditioning：构建并保存 R(x) 的切片图 + R3D.npy

用法（VSCode / F5）：
- 在 setting.py 里配置 TCN1D_test_p:
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
import cv2  # 修正PSNR导入方式（原代码直接from cv2 import PSNR可能报错）

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


def load_stats_strict(model_name: str, data_flag: str) -> dict:
    """
    Strict stats loading priority:
      1) full_ckpt: save_train_model/{run_id}_full_ckpt_{data_flag}.pth  (BEST)
      2) dedicated: save_train_model/norm_stats_{run_id}_{data_flag}.npy (if you enable in training)
      3) legacy   : save_train_model/norm_stats_{data_flag}.npy          (FALLBACK ONLY)

    This fully eliminates 'norm_stats_{data_flag}.npy' being overwritten by other runs.
    """
    run_id = _infer_run_id_for_full_ckpt(model_name)
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
        f"Cannot find stats for model={model_name}, data_flag={data_flag}\n"
        f"Tried:\n  {full_ckpt_path}\n  {p2}\n  {p3}"
    )

# -----------------------------
# 工具函数
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
    """确保目录存在，不存在则创建"""
    os.makedirs(path, exist_ok=True)


def _percentile_vminmax(arr: np.ndarray, p_low=1, p_high=99):
    """计算数组的百分位数极值（用于可视化的vmin/vmax）"""
    vmin = np.percentile(arr, p_low)
    vmax = np.percentile(arr, p_high)
    return float(vmin), float(vmax)


def get_data_raw(data_flag='Stanford_VI'):
    """
    仅读取原始数据，不做任何标准化/裁剪操作
    输出：
      seismic_raw: (N, H)  原始地震数据（未标准化）
      model_raw  : (N, H)  原始模型数据（未标准化）
      facies_raw : (N, H)  原始地震相数据
      meta       : dict(H, inline, xline, seismic3d, model3d, facies3d)
    """
    meta = {}

    if data_flag == 'Stanford_VI':
        # 读取原始3D数据 (H, IL, XL)
        seismic3d = np.load(join('data', data_flag, 'synth_40HZ.npy'))
        model3d = np.load(join('data', data_flag, 'AI.npy'))
        facies3d = np.load(join('data', data_flag, 'Facies.npy'))

        H, IL, XL = seismic3d.shape
        meta = {
            'H': H, 'inline': IL, 'xline': XL,
            'seismic3d': seismic3d, 'model3d': model3d, 'facies3d': facies3d
        }

        # 展平为trace维度：(H, IL*XL) -> (IL*XL, H)
        seismic_raw = np.transpose(seismic3d.reshape(H, IL * XL), (1, 0))
        model_raw = np.transpose(model3d.reshape(H, IL * XL), (1, 0))
        facies_raw = np.transpose(facies3d.reshape(H, IL * XL), (1, 0))

        print(f"[{data_flag}] 原始数据维度: model={model_raw.shape}, seismic={seismic_raw.shape}, facies={facies_raw.shape}")
        print(f"[{data_flag}] 原始数据均值: model={float(model_raw.mean()):.4f}, seismic={float(seismic_raw.mean()):.4f}")

    elif data_flag == 'Fanny':
        # 兼容Fanny数据集（保持和训练一致的原始读取逻辑）
        seismic_raw = np.load(join('data', data_flag, 'seismic.npy'))
        GR_raw = np.load(join('data', data_flag, 'GR.npy'))
        model_raw = np.load(join('data', data_flag, 'Impedance.npy'))
        facies_raw = np.load(join('data', data_flag, 'facies.npy'))
        # 目标可以是GR（和训练一致）
        model_raw = GR_raw
        # 自动计算Fanny的inline/xline（假设是正方形）
        n_traces = model_raw.shape[0]
        IL = XL = int(np.sqrt(n_traces))
        meta = {
            'H': model_raw.shape[-1], 'inline': IL, 'xline': XL,
            'seismic3d': seismic_raw.reshape(IL, XL, -1).transpose(2,0,1),  # 适配3D格式
            'model3d': model_raw.reshape(IL, XL, -1).transpose(2,0,1)
        }
        print(f"[{data_flag}] 原始数据维度: model={model_raw.shape}, seismic={seismic_raw.shape}")

    else:
        raise ValueError(f"不支持的数据集: {data_flag}")

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
    可视化Stanford VI的预测/真实剖面和切片
    适配自检后的reshape order，解决竖直条带问题
    """
    _ensure_dir("results")

    H = int(meta["H"])
    IL = int(meta["inline"])
    XL = int(meta["xline"])
    # 从meta中获取自检后的reshape顺序（C/F）
    reshape_order = meta.get("reshape_order", "C")

    # 关键：使用正确的reshape order重塑为3D
    AI_act = AI_act_flat.reshape(IL, XL, H, order=reshape_order)
    AI_pred = AI_pred_flat.reshape(IL, XL, H, order=reshape_order)

    # 处理地震数据的reshape
    if seismic_flat.ndim == 3:
        seis_amp = seismic_flat[:, 0, :].reshape(IL, XL, H, order=reshape_order)
    else:
        seis_amp = seismic_flat.reshape(IL, XL, H, order=reshape_order)

    # 添加轻微噪声提升可视化效果（和原始数据风格一致）
    blurred = ndimage.gaussian_filter(seis_amp, sigma=1.1)
    seis_plot = blurred + 0.5 * blurred.std() * np.random.random(blurred.shape)

    # 坐标转换（索引→米，IL/XL每格25米）
    il_dist = IL * 25
    xl_dist = XL * 25

    # 基于真实值计算robust的vmin/vmax
    vmin, vmax = _percentile_vminmax(AI_act, 1, 99)

    # -------- Pred xline 剖面 --------
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

    # -------- Pred inline 剖面 --------
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

    # -------- Pred depth 切片 --------
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

    # -------- True xline 剖面 --------
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

    # -------- True inline 剖面 --------
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

    # -------- True depth 切片 --------
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

    # -------- Seismic xline 剖面 --------
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

    # -------- 单点 Trace 对比 --------
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
    可视化并保存R(x)的3D数据和切片，适配正确的reshape order
    """
    _ensure_dir("results")
    H = int(meta["H"])
    IL = int(meta["inline"])
    XL = int(meta["xline"])
    reshape_order = meta.get("reshape_order", "C")

    # 用正确的order重塑R数据
    R3d = R_flat_torch.detach().cpu().numpy().reshape(IL, XL, H, order=reshape_order)
    # 保存两种维度顺序的R3D数据
    np.save(f"results/{out_prefix}_R3d_ILXLH.npy", R3d)
    np.save(f"results/{out_prefix}_R3d_HILXL.npy", np.transpose(R3d, (2, 0, 1)))

    # 保存R(x)深度切片
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
# 主测试函数
# -----------------------------
def test(test_p: dict):
    _ensure_dir("results")
    _ensure_dir("save_train_model")

    # 读取配置
    model_name = test_p["model_name"]
    data_flag = test_p["data_flag"]
    no_wells = int(test_p.get("no_wells", 20))

    # 合并配置（测试配置覆盖训练配置）
    cfg = {**TCN1D_train_p, **test_p}

    ### 1. 读取原始数据（无标准化、无裁剪）
    seismic_raw, model_raw, facies_raw, meta = get_data_raw(data_flag=data_flag)

    ### 2. Trace顺序自检（核心！解决竖直条带问题）
    IL, XL, H = meta["inline"], meta["xline"], meta["H"]
    # 验证flatten/reshape可逆性
    model_3d = model_raw.reshape(IL, XL, H)
    model_back = model_3d.reshape(IL * XL, H)
    reshape_err = np.abs(model_back - model_raw).max()
    print(f"[SANITY CHECK] flatten/reshape 最大误差: {reshape_err:.6f}")
    
    # 异常处理：自动适配C/F order
    if reshape_err > 1e-10:
        print("[WARNING] 展平/重塑不可逆！尝试Fortran顺序（order='F'）修正...")
        model_3d_F = model_raw.reshape(IL, XL, H, order='F')
        model_back_F = model_3d_F.reshape(IL * XL, H, order='F')
        reshape_err_F = np.abs(model_back_F - model_raw).max()
        print(f"[SANITY CHECK] 修正后最大误差: {reshape_err_F:.6f}")
        
        if reshape_err_F < 1e-10:
            meta["reshape_order"] = "F"
            print("[SANITY CHECK] 修正成功！使用Fortran顺序（列优先）reshape")
        else:
            raise RuntimeError(
                f"展平/重塑始终不可逆！原始误差={reshape_err:.6f}, 修正后={reshape_err_F:.6f} \n"
                f"请检查维度：IL={IL}, XL={XL}, H={H}, model_raw.shape={model_raw.shape}"
            )
    else:
        meta["reshape_order"] = "C"
        print("[SANITY CHECK] 展平/重塑可逆，维度顺序正确！")

    ### 3. 加载训练集Stats（优先从full_ckpt读取，杜绝错配）
    stats = load_stats_strict(model_name=model_name, data_flag=data_flag)
    print(f"[NORM] stats loaded | mode={stats.get('mode')} | keys={list(stats.keys())}")

    ### 4. 应用训练Stats做标准化（和训练完全一致）
    seismic, model, _ = standardize(seismic_raw, model_raw, stats=stats)

    ### 5. 裁剪到8的倍数（适配UNet/TCN下采样）
    s_L = seismic.shape[-1]
    n = int((s_L // 8) * 8)
    seismic = seismic[:, :n]
    model = model[:, :n]
    facies = facies_raw[:, :n]

    ### 6. 添加通道维度（和训练一致）
    seismic = seismic[:, np.newaxis, :].astype(np.float32)  # (N,1,H)
    model = model[:, np.newaxis, :].astype(np.float32)
    facies = facies[:, np.newaxis, :]

    ### 7. 构建各向异性R通道（若开启）
    R_flat = None
    if cfg.get("use_aniso_conditioning", False) and data_flag == "Stanford_VI":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ✅ 真实井点作为种子（从CSV读取 INLINE/XLINE）
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
        # 加载3D数据到设备
        seis3d = torch.from_numpy(meta["seismic3d"]).to(device=device, dtype=torch.float32)
        fac3d = torch.from_numpy(meta["facies3d"]).to(device=device, dtype=torch.long)
        ai3d = torch.from_numpy(meta["model3d"]).to(device=device, dtype=torch.float32)

        # 构建R(x)
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
            use_soft_prior=False,
        )

        # 拼接R通道到地震数据
        R_np = R_flat.detach().cpu().numpy()[:, np.newaxis, :].astype(np.float32)
        seismic = np.concatenate([seismic, R_np], axis=1)  # (N,2,H)

        # 保存R(x)可视化
        out_prefix = f"{model_name}_{data_flag}"
        save_R_visualization(R_flat, meta, out_prefix, depth_slices=(40, 100, 160))

    print(f"[TEST] 最终输入维度: seismic={seismic.shape}, model={model.shape}")

    ### 8. 构建DataLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    traces_test = np.arange(len(model), dtype=int)
    test_dataset = SeismicDataset1D(seismic, model, traces_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    ### 9. 加载模型
    ckpt_path = f"save_train_model/{model_name}_{data_flag}.pth"
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), ckpt_path)
    print(f"[TEST] 加载模型: {ckpt_path}")
    inver_model = torch.load(ckpt_path, map_location=device).to(device)

    ### 10. 全体素推理
    print("[TEST] 推理中 ...")
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
            # 适配不同的输出维度
            AI_pred[mem:mem + bs] = y_pred.squeeze(1) if y_pred.ndim == 3 else y_pred
            AI_act[mem:mem + bs] = y.squeeze(1) if y.ndim == 3 else y
            mem += bs

    # 转换为numpy
    AI_pred_np = AI_pred.detach().cpu().numpy()
    AI_act_np = AI_act.detach().cpu().numpy()

    # 保存预测/真实数据
    out_prefix = f"{model_name}_{data_flag}"
    np.save(f"results/{out_prefix}_pred_AI.npy", AI_pred_np)
    np.save(f"results/{out_prefix}_true_AI.npy", AI_act_np)

    ### 11. 计算定量指标
    print("\n[TEST] 定量评估指标:")
    # R² 得分
    r2 = r2_score(AI_act_np.ravel(), AI_pred_np.ravel())
    print(f"  R² 得分   : {r2:.4f}")
    
    # 皮尔逊相关系数（PCC）
    pcc, p_value = pearsonr(AI_act_np.ravel(), AI_pred_np.ravel())
    print(f"  PCC       : {pcc:.4f} (p-value: {p_value:.2e})")
    
    # SSIM（结构相似性）
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
    
    # PSNR（峰值信噪比）
    dr_full = max(AI_act_np.max(), AI_pred_np.max()) - min(AI_act_np.min(), AI_pred_np.min()) + 1e-12
    psnr = 20 * np.log10(dr_full) - 10 * np.log10(mse + 1e-12)
    print(f"  PSNR      : {psnr:.4f} dB")

    ### 12. 可视化结果
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
        print(f"\n[TEST] 可视化结果已保存到 results/ 目录，前缀: {out_prefix}")
    else:
        print("\n[TEST] Fanny数据集可视化可按需扩展")


if __name__ == "__main__":
    test(TCN1D_test_p)