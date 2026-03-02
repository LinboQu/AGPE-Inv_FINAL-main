"""
train_multitask.py (UPDATED)

Key upgrades (no network changes)
--------------------------------
1) Robust stats/ckpt binding:
   - Save FULL checkpoint (state_dict + stats + train_p) every run.
2) Use real well locations as seeds:
   - Read selected_wells_20_seed2026.csv (INLINE/XLINE) and convert to trace indices.
3) Facies-adaptive anisotropic conditioning R(x) that can be UPDATED iteratively:
   - Initial R uses facies prior (for Stanford VI-E we use Facies.npy as prior).
   - Optionally refresh R every R_update_every epochs using predicted facies probabilities
     + physics residual damping, and EMA update to stabilize.

Notes:
- This file assumes utils/datasets.py returns CPU tensors. We move tensors to GPU in the loop.
- Default behavior remains compatible if you keep iterative_R=False.
"""

import os
import csv
import errno
import random
from os.path import join

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.CNN2Layer import VishalNet
from model.tcn import TCN_IV_1D_C
from model.M2M_LSTM import GRU_MM
from model.Unet_1D import Unet_1D
from model.Transformer import TransformerModel
from model.Forward import forward_model_0, forward_model_1, forward_model_2
from model.geomorphology_classification import Facies_model_class

from setting import *
from utils.utils import standardize
from utils.datasets import SeismicDataset1D, SeismicDataset1D_SPF, SeismicDataset1D_SPF_WS
from utils.reliability_aniso import build_R_and_prior_from_cube


# -----------------------------
# helpers
# -----------------------------
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_selected_wells_trace_indices(
    csv_path: str | None,
    IL: int,
    XL: int,
    no_wells: int,
    seed: int = 2026,
) -> np.ndarray:
    """
    Convert (INLINE, XLINE) in CSV to flattened trace indices (inline * XL + xline).
    If csv_path is missing, fallback to uniform linspace sampling (legacy).
    """
    if csv_path is None or (not os.path.isfile(csv_path)):
        # fallback: keep legacy behavior
        return np.linspace(0, IL * XL - 1, int(no_wells), dtype=np.int64)

    ils: list[int] = []
    xls: list[int] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or ("INLINE" not in reader.fieldnames) or ("XLINE" not in reader.fieldnames):
            raise ValueError(f"CSV must contain INLINE,XLINE columns. Got: {reader.fieldnames}")
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
    traces = np.unique((il * XL + xl).astype(np.int64))

    # enforce count
    if len(traces) != int(no_wells):
        rng = np.random.default_rng(int(seed))
        if len(traces) > int(no_wells):
            traces = rng.choice(traces, size=int(no_wells), replace=False).astype(np.int64)
        else:
            print(f"[WELLS][WARN] CSV has {len(traces)} wells < requested {no_wells}. Using {len(traces)}.")

    return traces


def get_data_SPF(no_wells=10, data_flag="Stanford_VI", get_F=0):
    """
    Read Stanford VI / Fanny raw cubes and standardize using global_model stats.
    Returns:
      seismic: (N,1,H) float32
      model  : (N,1,H) float32
      facies : (N,1,H) int64
      meta   : dict(H, inline, xline, seismic3d, model3d, facies3d)
      stats  : dict(mean/std etc) used for standardize
    """
    meta = {}

    if data_flag == "Stanford_VI":
        seismic3d = np.load(join("data", data_flag, "synth_40HZ.npy"))  # (H,IL,XL)
        model3d = np.load(join("data", data_flag, "AI.npy"))
        facies3d = np.load(join("data", data_flag, "Facies.npy"))

        H, IL, XL = seismic3d.shape
        meta = {"H": H, "inline": IL, "xline": XL, "seismic3d": seismic3d, "model3d": model3d, "facies3d": facies3d}

        seismic = np.transpose(seismic3d.reshape(H, IL * XL), (1, 0))
        model = np.transpose(model3d.reshape(H, IL * XL), (1, 0))
        facies = np.transpose(facies3d.reshape(H, IL * XL), (1, 0))

        print(f"[{data_flag}] raw shapes: model={model.shape}, seismic={seismic.shape}, facies={facies.shape}")
        print(f"[{data_flag}] raw means : model={float(model.mean()):.4f}, seismic={float(seismic.mean()):.4f}")

    elif data_flag == "Fanny":
        seismic = np.load(join("data", data_flag, "seismic.npy"))
        model = np.load(join("data", data_flag, "impedance.npy"))
        facies = np.load(join("data", data_flag, "facies.npy"))
        H = model.shape[-1]
        n_traces = model.shape[0]
        IL = XL = int(np.sqrt(n_traces))
        meta = {"H": H, "inline": IL, "xline": XL}

    else:
        raise ValueError(f"Unsupported data_flag: {data_flag}")

    # standardize (global_model) and return stats
    seismic, model, stats = standardize(seismic, model, no_wells=no_wells, mode="global_model")

    # crop to multiple of 8 (for UNet-like downsampling)
    s_L = seismic.shape[-1]
    n = int((s_L // 8) * 8)
    seismic = seismic[:, :n]
    model = model[:, :n]
    facies = facies[:, :n]

    return (
        seismic[:, np.newaxis, :].astype(np.float32),
        model[:, np.newaxis, :].astype(np.float32),
        facies[:, np.newaxis, :].astype(np.int64),
        meta,
        stats,
    )


# -----------------------------
# main train
# -----------------------------
def train(train_p: dict):
    # pick model classes
    model_name = train_p["model_name"]
    Forward_model = train_p["Forward_model"]
    Facies_model_C = train_p["Facies_model"]

    if model_name == "tcnc":
        choice_model = TCN_IV_1D_C
    elif model_name == "VishalNet":
        choice_model = VishalNet
    elif model_name == "GRU_MM":
        choice_model = GRU_MM
    elif model_name == "Unet_1D":
        choice_model = Unet_1D
    elif model_name == "Transformer":
        choice_model = TransformerModel
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    if Forward_model == "cnn":
        forward = forward_model_0
    elif Forward_model == "convolution":
        forward = forward_model_1
    elif Forward_model == "cov_para":
        forward = forward_model_2
    else:
        raise ValueError(f"Unknown Forward_model: {Forward_model}")

    if Facies_model_C != "Facies":
        raise ValueError(f"Unknown Facies model: {Facies_model_C}")
    Facies_class = Facies_model_class

    data_flag = train_p["data_flag"]
    no_wells = int(train_p.get("no_wells", 20))
    seed = int(train_p.get("seed", 2026))
    selected_wells_csv = train_p.get("selected_wells_csv", None)
    deterministic_mode = bool(train_p.get("deterministic_mode", True))
    deterministic_warn_only = bool(train_p.get("deterministic_warn_only", True))

    # Reproducibility baseline: fix all random sources and enable deterministic kernels.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_mode:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=deterministic_warn_only)
        except TypeError:
            # Older torch versions do not support warn_only.
            torch.use_deterministic_algorithms(True)
        print(
            f"[DET] enabled seed={seed} cudnn.deterministic=True cudnn.benchmark=False "
            f"warn_only={deterministic_warn_only}"
        )
    else:
        print(f"[DET] disabled seed={seed}")

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ensure_dir("save_train_model")
    _ensure_dir("results")

    # data
    seismic, model, facies, meta, stats = get_data_SPF(no_wells=no_wells, data_flag=data_flag, get_F=train_p.get("get_F", 0))

    # save stats (both legacy + run-specific)
    run_id_base = f"{model_name}_{Forward_model}_{Facies_model_C}"
    run_id_suffix = str(train_p.get("run_id_suffix", "") or "")
    run_id = run_id_base if (run_id_suffix == "" or run_id_base.endswith(run_id_suffix)) else f"{run_id_base}{run_id_suffix}"
    np.save(join("save_train_model", f"norm_stats_{data_flag}.npy"), stats)  # legacy (may be overwritten)
    np.save(join("save_train_model", f"norm_stats_{run_id}_{data_flag}.npy"), stats)  # strong binding
    print(f"[NORM] saved stats: norm_stats_{run_id}_{data_flag}.npy (and legacy norm_stats_{data_flag}.npy)")
    aniso_train_protocol = str(train_p.get("aniso_train_protocol", "closed_loop")).strip().lower()
    if aniso_train_protocol not in ("closed_loop", "oracle"):
        raise ValueError(
            f"Unknown aniso_train_protocol='{aniso_train_protocol}', expected 'closed_loop' or 'oracle'."
        )
    closed_loop_alpha_prior = float(train_p.get("aniso_closed_loop_alpha_prior", 0.0))
    closed_loop_conf_thresh = float(train_p.get("aniso_closed_loop_conf_thresh", train_p.get("conf_thresh", 0.60)))
    agpe_neutral_p = float(train_p.get("agpe_neutral_p", 0.5))
    print(
        f"[AGPE] backend={train_p.get('aniso_backend', 'grid')} "
        f"use_tensor_strength={bool(train_p.get('aniso_use_tensor_strength', False))} "
        f"train_protocol={aniso_train_protocol}"
    )

    # -----------------------------
    # Build / maintain anisotropic R(x)
    # -----------------------------
    R_prev_flat = None
    prior_np = None
    agpe_cache = None
    aniso_backend = str(train_p.get("aniso_backend", "grid"))
    want_graph_cache = aniso_backend == "skeleton_graph"
    graph_log_path = join("results", f"{run_id}_{data_flag}_agpe_graph_log.csv")
    r_update_every_cfg = int(train_p.get("R_update_every", 50))
    use_tensor_strength_cfg = bool(train_p.get("aniso_use_tensor_strength", False))
    lambda_phys_damp_cfg = float(train_p.get("lambda_phys_damp", 0.0))

    def _append_aniso_health_log(
        epoch_idx: int,
        r_now_flat: torch.Tensor,
        alpha_prior_value: float,
        conf_thresh_value: float,
        iterative_flag: bool,
    ) -> None:
        if r_now_flat is None:
            return
        if not bool(train_p.get("use_aniso_conditioning", False)):
            return
        if data_flag != "Stanford_VI":
            return

        stats = getattr(agpe_cache, "last_stats", {}) if (want_graph_cache and agpe_cache is not None) else {}
        r_now = r_now_flat.detach().cpu().numpy().reshape(-1)
        if r_now.size == 0:
            return

        row = {
            "epoch": int(epoch_idx),
            "backend": str(train_p.get("aniso_backend", "grid")),
            "n_nodes_mean": float(stats.get("n_nodes_mean", 0.0)),
            "n_edges_mean": float(stats.get("n_edges_mean", 0.0)),
            "long_edge_ratio": float(stats.get("long_edge_ratio", 0.0)),
            "snap_ok_ratio": float(stats.get("snap_ok_ratio", 0.0)),
            "fallback_slices": int(stats.get("fallback_slices", 0)),
            "rebuild_slices": int(stats.get("rebuild_slices", 0)),
            "cache_hits": int(stats.get("cache_hits", 0)),
            "R_mean": float(r_now.mean()),
            "R_ratio_gt_0p5": float((r_now > 0.5).mean()),
            "iterative_R": int(bool(iterative_flag)),
            "R_update_every": int(r_update_every_cfg),
            "use_tensor_strength": int(bool(use_tensor_strength_cfg)),
            "alpha_prior": float(alpha_prior_value),
            "conf_thresh": float(conf_thresh_value),
            "lambda_phys_damp": float(lambda_phys_damp_cfg),
            "R_std": float(r_now.std()),
            "R_min": float(r_now.min()),
            "R_max": float(r_now.max()),
            "R_ratio_lt_0p1": float((r_now < 0.1).mean()),
        }

        write_header = not os.path.isfile(graph_log_path)
        with open(graph_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "backend",
                    "n_nodes_mean",
                    "n_edges_mean",
                    "long_edge_ratio",
                    "snap_ok_ratio",
                    "fallback_slices",
                    "rebuild_slices",
                    "cache_hits",
                    "R_mean",
                    "R_ratio_gt_0p5",
                    "iterative_R",
                    "R_update_every",
                    "use_tensor_strength",
                    "alpha_prior",
                    "conf_thresh",
                    "lambda_phys_damp",
                    "R_std",
                    "R_min",
                    "R_max",
                    "R_ratio_lt_0p1",
                ],
            )
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    if train_p.get("use_aniso_conditioning", False) and data_flag == "Stanford_VI":
        H, IL, XL = int(meta["H"]), int(meta["inline"]), int(meta["xline"])
        traces_train = load_selected_wells_trace_indices(selected_wells_csv, IL, XL, no_wells=no_wells, seed=seed)
        print(f"[WELLS] using wells: {selected_wells_csv} | count={len(traces_train)}")
        np.save(join("results", f"{run_id}_{data_flag}_well_trace_indices.npy"), traces_train)

        well_idx = torch.from_numpy(traces_train.astype(np.int64)).to(device)
        seis3d = torch.from_numpy(meta["seismic3d"]).to(device=device, dtype=torch.float32)
        ai3d = torch.from_numpy(meta["model3d"]).to(device=device, dtype=torch.float32)

        ch_id = int(train_p.get("channel_id", 2))
        if aniso_train_protocol == "oracle":
            fac_prior3d = torch.from_numpy(meta["facies3d"]).to(device=device, dtype=torch.long)
            p0_3d = (fac_prior3d == ch_id).float()
            conf0_3d = torch.ones_like(p0_3d)
            r0_alpha_prior = 1.0
            r0_conf_thresh = 0.0
            print("[AGPE][TRAIN][WARN] protocol=oracle (uses facies prior volume).")
        else:
            fac_prior3d = None
            p0_3d = torch.full_like(ai3d, float(agpe_neutral_p))
            conf0_3d = torch.ones_like(p0_3d)
            r0_alpha_prior = float(closed_loop_alpha_prior)
            r0_conf_thresh = float(closed_loop_conf_thresh)
            print("[AGPE][TRAIN] protocol=closed_loop (R0 without facies oracle).")
        
        r0_out = build_R_and_prior_from_cube(
            seismic_3d=seis3d,
            ai_3d=ai3d,
            well_trace_indices=well_idx,
            
            # ✅ key: provide one of the required "channel-likeness" sources
            p_channel_3d=p0_3d,
            conf_3d=conf0_3d,
            
            # anchor prior (still used as your geological prior)
            facies_prior_3d=fac_prior3d,
            
            channel_id=ch_id,
            alpha_prior=float(r0_alpha_prior),
            conf_thresh=float(r0_conf_thresh),
            neutral_p=float(agpe_neutral_p),
            steps_R=int(train_p.get("aniso_steps_R", 25)),
            eta=float(train_p.get("aniso_eta", 0.6)),
            gamma=float(train_p.get("aniso_gamma", 8.0)),
            tau=float(train_p.get("aniso_tau", 0.6)),
            kappa=float(train_p.get("aniso_kappa", 4.0)),
            sigma_st=float(train_p.get("aniso_sigma_st", 1.2)),
            backend=str(train_p.get("aniso_backend", "grid")),
            aniso_use_tensor_strength=bool(train_p.get("aniso_use_tensor_strength", False)),
            aniso_tensor_strength_power=float(train_p.get("aniso_tensor_strength_power", 1.0)),
            agpe_skel_p_thresh=float(train_p.get("agpe_skel_p_thresh", 0.55)),
            agpe_skel_min_nodes=int(train_p.get("agpe_skel_min_nodes", 30)),
            agpe_skel_snap_radius=int(train_p.get("agpe_skel_snap_radius", 5)),
            agpe_long_edges=bool(train_p.get("agpe_long_edges", True)),
            agpe_long_max_step=int(train_p.get("agpe_long_max_step", 6)),
            agpe_long_step=int(train_p.get("agpe_long_step", 2)),
            agpe_long_cos_thresh=float(train_p.get("agpe_long_cos_thresh", 0.70)),
            agpe_long_weight=float(train_p.get("agpe_long_weight", 0.35)),
            agpe_edge_tau_p=float(train_p.get("agpe_edge_tau_p", 0.25)),
            agpe_lift_sigma=float(train_p.get("agpe_lift_sigma", 2.2)),
            agpe_well_seed_mode=str(train_p.get("agpe_well_seed_mode", "depth_gate")),
            agpe_well_seed_power=float(train_p.get("agpe_well_seed_power", 1.0)),
            agpe_well_seed_min=float(train_p.get("agpe_well_seed_min", 0.02)),
            agpe_well_seed_use_conf=bool(train_p.get("agpe_well_seed_use_conf", True)),
            agpe_well_soft_alpha=float(train_p.get("agpe_well_soft_alpha", 0.20)),
            agpe_cache_graph=bool(train_p.get("agpe_cache_graph", True)),
            agpe_refine_graph=bool(train_p.get("agpe_refine_graph", True)),
            agpe_rebuild_every=int(train_p.get("agpe_rebuild_every", 50)),
            agpe_topo_change_pch_l1=float(train_p.get("agpe_topo_change_pch_l1", 0.05)),
            epoch=0,
            graph_cache=agpe_cache,
            return_graph_cache=bool(want_graph_cache),
            use_soft_prior=bool(train_p.get("use_soft_prior", False)),
            steps_prior=int(train_p.get("aniso_steps_prior", 35)),
        )
        if want_graph_cache:
            R_prev_flat, prior_flat, agpe_cache = r0_out
        else:
            R_prev_flat, prior_flat = r0_out

        # append R as extra channel
        R_np = R_prev_flat.detach().cpu().numpy()[:, np.newaxis, :].astype(np.float32)  # (N,1,H)
        seismic = np.concatenate([seismic, R_np], axis=1)  # (N,2,H)
        prior_np = prior_flat.detach().cpu().numpy()[:, np.newaxis, :].astype(np.float32) if prior_flat is not None else None

        # log stats
        r_np = R_prev_flat.detach().cpu().numpy()
        print(f"[R0] mean={r_np.mean():.4f} max={r_np.max():.4f} ratio(R>0.5)={(r_np>0.5).mean():.4f}")
        _append_aniso_health_log(
            epoch_idx=0,
            r_now_flat=R_prev_flat,
            alpha_prior_value=float(r0_alpha_prior),
            conf_thresh_value=float(r0_conf_thresh),
            iterative_flag=bool(train_p.get("iterative_R", False) and (R_prev_flat is not None)),
        )
    else:
        # legacy: linearly sampled wells (no R channel)
        traces_train = np.linspace(0, len(model) - 1, no_wells, dtype=int)

    # datasets / loaders
    train_dataset = SeismicDataset1D_SPF(seismic, model, facies, traces_train)

    configured_workers = int(train_p.get("num_workers", 0))
    if deterministic_mode and configured_workers != 0:
        print(f"[DET] override num_workers {configured_workers} -> 0 for deterministic baseline.")
    num_workers = 0 if deterministic_mode else configured_workers
    pin_memory = bool(train_p.get("pin_memory", torch.cuda.is_available()))
    persistent_workers = bool(train_p.get("persistent_workers", True)) and (num_workers > 0)
    train_loader_generator = torch.Generator()
    train_loader_generator.manual_seed(seed + 11)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_p.get("batch_size", 4)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
        generator=train_loader_generator,
    )

    # Weak supervision loader (scheme A scheduling)
    u = train_p.get("unsupervised_seismic", None)
    if u is None or int(u) <= 0:
        ws_traces = np.arange(len(model), dtype=int)
    else:
        ws_traces = np.linspace(0, len(model) - 1, int(u), dtype=int)

    Wsupervised_dataset = SeismicDataset1D_SPF_WS(seismic, facies, ws_traces, prior=prior_np)
    Wsupervised_loader = DataLoader(
        Wsupervised_dataset,
        batch_size=int(train_p.get("batch_size", 4)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed + 29),
    )

    # validation (small subset)
    traces_validation = np.linspace(0, len(model) - 1, 3, dtype=int)
    val_dataset = SeismicDataset1D(seismic, model, traces_validation)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

    # models
    in_ch = int(seismic.shape[1])
    try:
        inverse_model = choice_model(input_dim=in_ch).to(device)
    except TypeError:
        try:
            inverse_model = choice_model(in_ch).to(device)
        except TypeError:
            inverse_model = choice_model().to(device)

    forward_model = forward().to(device)
    Facies_model = Facies_class(facies_n=int(train_p.get("facies_n", 4))).to(device)

    # losses
    criterion_ai = torch.nn.MSELoss()
    criterion_rec = torch.nn.MSELoss()
    criterion_facies = nn.CrossEntropyLoss()

    lam_ai = float(train_p.get("lambda_ai", 5.0))
    lam_fac = float(train_p.get("lambda_facies", 0.2))
    lam_rec = float(train_p.get("lambda_recon", 1.0))
    use_two_stage_loss = bool(train_p.get("use_two_stage_loss", True))
    stageA_epochs = int(train_p.get("stageA_epochs", 200))
    stageB_ramp_epochs = int(train_p.get("stageB_ramp_epochs", 200))
    stageA_lambda_ai_mult = float(train_p.get("stageA_lambda_ai_mult", 1.8))
    stageA_lambda_facies_mult = float(train_p.get("stageA_lambda_facies_mult", 0.30))
    stageA_lambda_recon_mult = float(train_p.get("stageA_lambda_recon_mult", 0.30))
    lambda_amp_anchor = float(train_p.get("lambda_amp_anchor", 0.05))
    facies_detach_y = bool(train_p.get("facies_detach_y", True))
    grad_clip = float(train_p.get("grad_clip", 0.0))

    optimizer = torch.optim.Adam(
        list(inverse_model.parameters()) + list(forward_model.parameters()) + list(Facies_model.parameters()),
        weight_decay=float(train_p.get("weight_decay", 1e-4)),
        lr=float(train_p.get("lr", 1e-4)),
    )

    # weak supervision scheduling
    ws_every = int(train_p.get("ws_every", 5))
    ws_max_batches = int(train_p.get("ws_max_batches", 50))
    ws_max_batches_stageA = int(train_p.get("ws_max_batches_stageA", max(10, ws_max_batches // 3)))

    # iterative R scheduling (recommended)
    iterative_R = bool(train_p.get("iterative_R", False)) and (R_prev_flat is not None)
    R_update_every = int(train_p.get("R_update_every", 50))
    R_ema_beta = float(train_p.get("R_ema_beta", 0.85))
    alpha_start = float(train_p.get("alpha_prior_start", 1.0))
    alpha_end = float(train_p.get("alpha_prior_end", 0.3))
    alpha_decay_epochs = int(train_p.get("alpha_prior_decay_epochs", max(1, train_p.get("epochs", 1000))))
    conf_thresh = float(train_p.get("conf_thresh", 0.75))
    lambda_phys_damp = float(train_p.get("lambda_phys_damp", 0.0))
    agpe_cache_graph = bool(train_p.get("agpe_cache_graph", True))
    agpe_refine_graph = bool(train_p.get("agpe_refine_graph", True))
    agpe_rebuild_every = int(train_p.get("agpe_rebuild_every", 50))
    agpe_topo_change_pch_l1 = float(train_p.get("agpe_topo_change_pch_l1", 0.05))

    def _alpha_prior(epoch: int) -> float:
        t = min(1.0, max(0.0, epoch / float(alpha_decay_epochs)))
        return alpha_start * (1 - t) + alpha_end * t

    def _loss_weights(epoch: int) -> tuple[float, float, float]:
        """Two-stage schedule to avoid early weak-supervision drift."""
        if (not use_two_stage_loss) or stageA_epochs <= 0:
            return lam_ai, lam_fac, lam_rec

        lam_ai_A = lam_ai * stageA_lambda_ai_mult
        lam_fac_A = lam_fac * stageA_lambda_facies_mult
        lam_rec_A = lam_rec * stageA_lambda_recon_mult

        if epoch < stageA_epochs:
            return lam_ai_A, lam_fac_A, lam_rec_A

        if stageB_ramp_epochs <= 0:
            return lam_ai, lam_fac, lam_rec

        t = min(1.0, max(0.0, (epoch - stageA_epochs) / float(stageB_ramp_epochs)))
        ai_t = lam_ai_A * (1.0 - t) + lam_ai * t
        fac_t = lam_fac_A * (1.0 - t) + lam_fac * t
        rec_t = lam_rec_A * (1.0 - t) + lam_rec * t
        return ai_t, fac_t, rec_t

    @torch.no_grad()
    def update_R(epoch: int) -> None:
        """Recompute p_channel/conf/residual from current models, then rebuild R and EMA-update."""
        nonlocal seismic, R_prev_flat, agpe_cache

        if not iterative_R:
            return
        if R_update_every <= 0:
            return
        if epoch == 0 or (epoch % R_update_every) != 0:
            return

        print(f"[R-UPDATE] epoch={epoch} ...")

        inverse_model.eval()
        forward_model.eval()
        Facies_model.eval()

        # build predicted facies probabilities + physics residual on ALL traces
        N = seismic.shape[0]
        Hs = seismic.shape[-1]
        bs = int(train_p.get("R_update_bs", 16))
        # sequential loader over all traces (shuffle=False)
        all_traces = np.arange(N, dtype=int)
        all_ds = SeismicDataset1D_SPF(seismic, model, facies, all_traces)
        all_ld = DataLoader(all_ds, batch_size=bs, shuffle=False, num_workers=0)

        pch = np.zeros((N, Hs), dtype=np.float32)
        conf = np.zeros((N, Hs), dtype=np.float32)
        pres = np.zeros((N, Hs), dtype=np.float32)

        mem = 0
        for x, y_gt, z_gt in all_ld:
            x = x.to(device, non_blocking=True)
            y_pred = inverse_model(x)
            x_rec = forward_model(y_pred)

            fac_in = y_pred.detach() if facies_detach_y else y_pred
            logits = Facies_model(fac_in)  # [B,K,H]
            probs = torch.softmax(logits, dim=1)
            pch_b = probs[:, int(train_p.get("channel_id", 2)), :].detach().cpu().numpy()
            conf_b = probs.max(dim=1).values.detach().cpu().numpy()

            # physics residual (abs error in seismic reconstruction)
            # x_rec: [B,1,H], x[:,0:1,:] is amplitude channel
            res_b = torch.abs(x_rec - x[:, 0:1, :]).squeeze(1).detach().cpu().numpy()

            bsz = x.shape[0]
            pch[mem:mem+bsz] = pch_b
            conf[mem:mem+bsz] = conf_b
            pres[mem:mem+bsz] = res_b
            mem += bsz

        # reshape to 3D (H,IL,XL)
        IL, XL = int(meta["inline"]), int(meta["xline"])
        H = int(meta["H"])
        assert H == Hs, f"Depth mismatch: meta.H={H}, Hs={Hs}"

        pch_3d = torch.from_numpy(pch.T.reshape(H, IL, XL)).to(device=device, dtype=torch.float32)
        conf_3d = torch.from_numpy(conf.T.reshape(H, IL, XL)).to(device=device, dtype=torch.float32)
        pres_3d = torch.from_numpy(pres.T.reshape(H, IL, XL)).to(device=device, dtype=torch.float32)

        if aniso_train_protocol == "oracle":
            fac_prior3d = torch.from_numpy(meta["facies3d"]).to(device=device, dtype=torch.long)
            alpha = _alpha_prior(epoch)
            conf_use = float(conf_thresh)
        else:
            fac_prior3d = None
            alpha = float(closed_loop_alpha_prior)
            conf_use = float(closed_loop_conf_thresh)

        # rebuild R_new from p_channel/conf under selected protocol
        well_idx = torch.from_numpy(traces_train.astype(np.int64)).to(device=device)

        r_upd_out = build_R_and_prior_from_cube(
            seismic_3d=torch.from_numpy(meta["seismic3d"]).to(device=device, dtype=torch.float32),
            ai_3d=torch.from_numpy(meta["model3d"]).to(device=device, dtype=torch.float32),
            well_trace_indices=well_idx,
            p_channel_3d=pch_3d,
            conf_3d=conf_3d,
            facies_prior_3d=fac_prior3d,
            channel_id=int(train_p.get("channel_id", 2)),
            alpha_prior=float(alpha),
            conf_thresh=float(conf_use),
            neutral_p=float(agpe_neutral_p),
            steps_R=int(train_p.get("aniso_steps_R", 25)),
            eta=float(train_p.get("aniso_eta", 0.6)),
            gamma=float(train_p.get("aniso_gamma", 8.0)),
            tau=float(train_p.get("aniso_tau", 0.6)),
            kappa=float(train_p.get("aniso_kappa", 4.0)),
            sigma_st=float(train_p.get("aniso_sigma_st", 1.2)),
            backend=str(train_p.get("aniso_backend", "grid")),
            aniso_use_tensor_strength=bool(train_p.get("aniso_use_tensor_strength", False)),
            aniso_tensor_strength_power=float(train_p.get("aniso_tensor_strength_power", 1.0)),
            agpe_skel_p_thresh=float(train_p.get("agpe_skel_p_thresh", 0.55)),
            agpe_skel_min_nodes=int(train_p.get("agpe_skel_min_nodes", 30)),
            agpe_skel_snap_radius=int(train_p.get("agpe_skel_snap_radius", 5)),
            agpe_long_edges=bool(train_p.get("agpe_long_edges", True)),
            agpe_long_max_step=int(train_p.get("agpe_long_max_step", 6)),
            agpe_long_step=int(train_p.get("agpe_long_step", 2)),
            agpe_long_cos_thresh=float(train_p.get("agpe_long_cos_thresh", 0.70)),
            agpe_long_weight=float(train_p.get("agpe_long_weight", 0.35)),
            agpe_edge_tau_p=float(train_p.get("agpe_edge_tau_p", 0.25)),
            agpe_lift_sigma=float(train_p.get("agpe_lift_sigma", 2.2)),
            agpe_well_seed_mode=str(train_p.get("agpe_well_seed_mode", "depth_gate")),
            agpe_well_seed_power=float(train_p.get("agpe_well_seed_power", 1.0)),
            agpe_well_seed_min=float(train_p.get("agpe_well_seed_min", 0.02)),
            agpe_well_seed_use_conf=bool(train_p.get("agpe_well_seed_use_conf", True)),
            agpe_well_soft_alpha=float(train_p.get("agpe_well_soft_alpha", 0.20)),
            agpe_cache_graph=agpe_cache_graph,
            agpe_refine_graph=agpe_refine_graph,
            agpe_rebuild_every=agpe_rebuild_every,
            agpe_topo_change_pch_l1=agpe_topo_change_pch_l1,
            epoch=int(epoch),
            graph_cache=agpe_cache,
            return_graph_cache=bool(want_graph_cache),
            phys_residual_3d=pres_3d if lambda_phys_damp > 0 else None,
            lambda_phys=float(lambda_phys_damp),
            use_soft_prior=False,
        )
        if want_graph_cache:
            R_new_flat, _, agpe_cache = r_upd_out
        else:
            R_new_flat, _ = r_upd_out

        # EMA update for stability
        R_upd = (float(R_ema_beta) * R_prev_flat + (1.0 - float(R_ema_beta)) * R_new_flat).clamp(0.0, 1.0)
        R_prev_flat = R_upd

        # write back to seismic second channel in-place
        R_np = R_upd.detach().cpu().numpy().astype(np.float32)[:, np.newaxis, :]
        seismic[:, 1:2, :] = R_np

        # log
        r_np = R_upd.detach().cpu().numpy()
        print(f"[R-UPDATE] alpha_prior={alpha:.3f} mean={r_np.mean():.4f} max={r_np.max():.4f} ratio(R>0.5)={(r_np>0.5).mean():.4f}")
        _append_aniso_health_log(
            epoch_idx=epoch,
            r_now_flat=R_upd,
            alpha_prior_value=float(alpha),
            conf_thresh_value=float(conf_use),
            iterative_flag=bool(iterative_R),
        )

        # optional save
        if int(train_p.get("save_R_every", 0)) > 0 and (epoch % int(train_p["save_R_every"])) == 0:
            np.save(join("results", f"{run_id}_{data_flag}_R_flat_epoch{epoch:04d}.npy"), r_np)

        inverse_model.train()
        forward_model.train()
        Facies_model.train()

    # training loop
    train_loss = []
    val_loss = []
    ws_loss_list = []

    for epoch in range(int(train_p.get("epochs", 1000))):
        update_R(epoch)
        lam_ai_t, lam_fac_t, lam_rec_t = _loss_weights(epoch)
        ws_cap_t = ws_max_batches_stageA if (use_two_stage_loss and epoch < max(0, stageA_epochs)) else ws_max_batches
        if epoch == 0 or epoch == stageA_epochs or epoch == (stageA_epochs + stageB_ramp_epochs):
            print(
                f"[LOSS-SCHED] epoch={epoch} "
                f"lam_ai={lam_ai_t:.3f} lam_fac={lam_fac_t:.3f} lam_rec={lam_rec_t:.3f} "
                f"ws_max_batches={ws_cap_t} lambda_amp_anchor={lambda_amp_anchor:.4f}"
            )

        inverse_model.train()
        forward_model.train()
        Facies_model.train()

        # weak supervision (Scheme A)
        if (ws_every > 0) and ((epoch % ws_every) == 0):
            ws_running = 0.0
            ws_batches = 0
            for bi, batch in enumerate(Wsupervised_loader):
                if ws_cap_t > 0 and bi >= ws_cap_t:
                    break
                optimizer.zero_grad()

                if len(batch) == 2:
                    x, z = batch
                    prior_b = None
                else:
                    x, z, prior_b = batch

                x = x.to(device, non_blocking=True)
                z = z.to(device, non_blocking=True)
                if prior_b is not None:
                    prior_b = prior_b.to(device, non_blocking=True)

                y_pred = inverse_model(x)
                x_rec = forward_model(y_pred)
                fac_in = y_pred.detach() if facies_detach_y else y_pred
                fac_logits = Facies_model(fac_in)

                loss_ws = (lam_fac_t * criterion_facies(fac_logits, z)) + (lam_rec_t * criterion_rec(x_rec, x[:, 0:1, :]))

                if prior_b is not None and bool(train_p.get("use_soft_prior", False)):
                    # weight by current R channel if present
                    Rch = x[:, 1:2, :] if x.shape[1] > 1 else 1.0
                    w = 0.1 + 0.9 * Rch
                    l_prior = ((y_pred - prior_b) ** 2 * w).mean() * float(train_p.get("lambda_prior", 0.20))
                    loss_ws = loss_ws + l_prior

                loss_ws.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(inverse_model.parameters()) + list(forward_model.parameters()) + list(Facies_model.parameters()),
                        max_norm=grad_clip,
                    )
                optimizer.step()
                ws_running += float(loss_ws.item())
                ws_batches += 1

            if ws_batches > 0:
                ws_loss_list.append(ws_running / ws_batches)

        # supervised on wells
        running = 0.0
        nb = 0
        for x, y, z in train_loader:
            optimizer.zero_grad()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            z = z.to(device, non_blocking=True)

            y_pred = inverse_model(x)
            x_rec = forward_model(y_pred)
            fac_in = y_pred.detach() if facies_detach_y else y_pred
            fac_logits = Facies_model(fac_in)

            l_ai = criterion_ai(y_pred, y)
            l_fac = criterion_facies(fac_logits, z)
            l_rec = criterion_rec(x_rec, x[:, 0:1, :])
            # Amplitude anchor: penalize global mean/std drift to reduce polarity/scale branch flips.
            amp_anchor = torch.tensor(0.0, device=device)
            if lambda_amp_anchor > 0:
                pred_mean = y_pred.mean(dim=(1, 2))
                true_mean = y.mean(dim=(1, 2))
                pred_std = y_pred.std(dim=(1, 2), unbiased=False)
                true_std = y.std(dim=(1, 2), unbiased=False)
                amp_anchor = ((pred_mean - true_mean) ** 2 + (pred_std - true_std) ** 2).mean()

            loss = (lam_ai_t * l_ai) + (lam_fac_t * l_fac) + (lam_rec_t * l_rec) + (lambda_amp_anchor * amp_anchor)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(inverse_model.parameters()) + list(forward_model.parameters()) + list(Facies_model.parameters()),
                    max_norm=grad_clip,
                )
            optimizer.step()

            running += float(loss.item())
            nb += 1

        if nb > 0:
            train_loss.append(running / nb)

        # val (AI only)
        inverse_model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                y_pred = inverse_model(x)
                loss_val = criterion_ai(y_pred, y)
                val_loss.append(float(loss_val.item()))

        print(f"Epoch {epoch:04d} | Train {train_loss[-1]:.4f} | Val {val_loss[-1]:.4f}")

        # save best-like checkpoints every some epochs (optional)
        if int(train_p.get("save_every", 0)) > 0 and (epoch % int(train_p["save_every"])) == 0:
            ckpt_path = join("save_train_model", f"{run_id}_{data_flag}_epoch{epoch:04d}.pth")
            torch.save(inverse_model, ckpt_path)

    # -----------------------------
    # save FULL checkpoint with stats (STRICT BINDING)
    # -----------------------------
    full_ckpt_path = join("save_train_model", f"{run_id}_full_ckpt_{data_flag}.pth")
    torch.save(
        {
            "inverse_state_dict": inverse_model.state_dict(),
            "forward_state_dict": forward_model.state_dict(),
            "facies_state_dict": Facies_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(train_p.get("epochs", 1000)) - 1,
            "stats": stats,
            "train_p": train_p,
        },
        full_ckpt_path,
    )
    print(f"[CKPT] full checkpoint saved: {full_ckpt_path}")

    # also keep legacy "model object" save for test_3D.py compatibility
    legacy_path = join("save_train_model", f"{run_id}_s_uns_{data_flag}.pth")
    torch.save(inverse_model, legacy_path)
    print(f"[CKPT] legacy model saved: {legacy_path}")

    # curves
    plt.figure()
    plt.plot(train_loss, "r", label="train")
    plt.plot(val_loss, "k", label="val")
    plt.legend()
    plt.tight_layout()
    plt.savefig(join("results", f"{run_id}_s_uns_{data_flag}.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    train(train_p=TCN1D_train_p)
