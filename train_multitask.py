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
import argparse
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

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ensure_dir("save_train_model")
    _ensure_dir("results")

    # data
    seismic, model, facies, meta, stats = get_data_SPF(no_wells=no_wells, data_flag=data_flag, get_F=train_p.get("get_F", 0))

    # save stats (both legacy + run-specific)
    run_id = f"{model_name}_{Forward_model}_{Facies_model_C}"
    np.save(join("save_train_model", f"norm_stats_{data_flag}.npy"), stats)  # legacy (may be overwritten)
    np.save(join("save_train_model", f"norm_stats_{run_id}_{data_flag}.npy"), stats)  # strong binding
    print(f"[NORM] saved stats: norm_stats_{run_id}_{data_flag}.npy (and legacy norm_stats_{data_flag}.npy)")

    # -----------------------------
    # Build / maintain anisotropic R(x)
    # -----------------------------
    R_prev_flat = None
    prior_np = None

    if train_p.get("use_aniso_conditioning", False) and data_flag == "Stanford_VI":
        H, IL, XL = int(meta["H"]), int(meta["inline"]), int(meta["xline"])
        traces_train = load_selected_wells_trace_indices(selected_wells_csv, IL, XL, no_wells=no_wells, seed=seed)
        print(f"[WELLS] using wells: {selected_wells_csv} | count={len(traces_train)}")
        np.save(join("results", f"{run_id}_{data_flag}_well_trace_indices.npy"), traces_train)

        # build initial R from facies PRIOR (Stanford VI: Facies.npy is available; in real: interpreter prior)
        well_idx = torch.from_numpy(traces_train.astype(np.int64)).to(device)
        seis3d = torch.from_numpy(meta["seismic3d"]).to(device=device, dtype=torch.float32)
        fac_prior3d = torch.from_numpy(meta["facies3d"]).to(device=device, dtype=torch.long)
        ai3d = torch.from_numpy(meta["model3d"]).to(device=device, dtype=torch.float32)

        # ---- NEW: provide p_channel_3d + conf_3d for init (do NOT rely on facies_3d truth) ----
        ch_id = int(train_p.get("channel_id", 2))
        p0_3d = (fac_prior3d == ch_id).float()          # [H,IL,XL] in {0,1}
        conf0_3d = torch.ones_like(p0_3d)               # [H,IL,XL] all confident
        
        R_prev_flat, prior_flat = build_R_and_prior_from_cube(
            seismic_3d=seis3d,
            ai_3d=ai3d,
            well_trace_indices=well_idx,
            
            # ✅ key: provide one of the required "channel-likeness" sources
            p_channel_3d=p0_3d,
            conf_3d=conf0_3d,
            
            # anchor prior (still used as your geological prior)
            facies_prior_3d=fac_prior3d,
            
            channel_id=ch_id,
            alpha_prior=1.0,      # initial: pure prior
            conf_thresh=0.0,
            steps_R=int(train_p.get("aniso_steps_R", 25)),
            eta=float(train_p.get("aniso_eta", 0.6)),
            gamma=float(train_p.get("aniso_gamma", 8.0)),
            tau=float(train_p.get("aniso_tau", 0.6)),
            kappa=float(train_p.get("aniso_kappa", 4.0)),
            sigma_st=float(train_p.get("aniso_sigma_st", 1.2)),
            use_soft_prior=bool(train_p.get("use_soft_prior", False)),
            steps_prior=int(train_p.get("aniso_steps_prior", 35)),
        )

        # append R as extra channel
        R_np = R_prev_flat.detach().cpu().numpy()[:, np.newaxis, :].astype(np.float32)  # (N,1,H)
        seismic = np.concatenate([seismic, R_np], axis=1)  # (N,2,H)
        prior_np = prior_flat.detach().cpu().numpy()[:, np.newaxis, :].astype(np.float32) if prior_flat is not None else None

        # log stats
        r_np = R_prev_flat.detach().cpu().numpy()
        print(f"[R0] mean={r_np.mean():.4f} max={r_np.max():.4f} ratio(R>0.5)={(r_np>0.5).mean():.4f}")
    else:
        # legacy: linearly sampled wells (no R channel)
        traces_train = np.linspace(0, len(model) - 1, no_wells, dtype=int)

    # datasets / loaders
    train_dataset = SeismicDataset1D_SPF(seismic, model, facies, traces_train)

    num_workers = int(train_p.get("num_workers", 0))
    pin_memory = bool(train_p.get("pin_memory", torch.cuda.is_available()))
    persistent_workers = bool(train_p.get("persistent_workers", True)) and (num_workers > 0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_p.get("batch_size", 4)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
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

    # iterative R scheduling (recommended)
    iterative_R = bool(train_p.get("iterative_R", False)) and (R_prev_flat is not None)
    R_update_every = int(train_p.get("R_update_every", 50))
    R_ema_beta = float(train_p.get("R_ema_beta", 0.85))
    alpha_start = float(train_p.get("alpha_prior_start", 1.0))
    alpha_end = float(train_p.get("alpha_prior_end", 0.3))
    alpha_decay_epochs = int(train_p.get("alpha_prior_decay_epochs", max(1, train_p.get("epochs", 1000))))
    conf_thresh = float(train_p.get("conf_thresh", 0.75))
    lambda_phys_damp = float(train_p.get("lambda_phys_damp", 0.0))

    def _alpha_prior(epoch: int) -> float:
        t = min(1.0, max(0.0, epoch / float(alpha_decay_epochs)))
        return alpha_start * (1 - t) + alpha_end * t

    @torch.no_grad()
    def update_R(epoch: int) -> None:
        """Recompute p_channel/conf/residual from current models, then rebuild R and EMA-update."""
        nonlocal seismic, R_prev_flat

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

        # prior facies (anchor). For Stanford VI-E we use Facies.npy; for real data use interpreter prior.
        fac_prior3d = torch.from_numpy(meta["facies3d"]).to(device=device, dtype=torch.long)

        # rebuild R_new from p_channel/conf, with prior mixing + confidence gating + physics damping
        alpha = _alpha_prior(epoch)
        well_idx = torch.from_numpy(traces_train.astype(np.int64)).to(device=device)

        R_new_flat, _ = build_R_and_prior_from_cube(
            seismic_3d=torch.from_numpy(meta["seismic3d"]).to(device=device, dtype=torch.float32),
            ai_3d=torch.from_numpy(meta["model3d"]).to(device=device, dtype=torch.float32),
            well_trace_indices=well_idx,
            p_channel_3d=pch_3d,
            conf_3d=conf_3d,
            facies_prior_3d=fac_prior3d,
            channel_id=int(train_p.get("channel_id", 2)),
            alpha_prior=float(alpha),
            conf_thresh=float(conf_thresh),
            steps_R=int(train_p.get("aniso_steps_R", 25)),
            eta=float(train_p.get("aniso_eta", 0.6)),
            gamma=float(train_p.get("aniso_gamma", 8.0)),
            tau=float(train_p.get("aniso_tau", 0.6)),
            kappa=float(train_p.get("aniso_kappa", 4.0)),
            sigma_st=float(train_p.get("aniso_sigma_st", 1.2)),
            phys_residual_3d=pres_3d if lambda_phys_damp > 0 else None,
            lambda_phys=float(lambda_phys_damp),
            use_soft_prior=False,
        )

        # EMA update for stability
        R_upd = (float(R_ema_beta) * R_prev_flat + (1.0 - float(R_ema_beta)) * R_new_flat).clamp(0.0, 1.0)
        R_prev_flat = R_upd

        # write back to seismic second channel in-place
        R_np = R_upd.detach().cpu().numpy().astype(np.float32)[:, np.newaxis, :]
        seismic[:, 1:2, :] = R_np

        # log
        r_np = R_upd.detach().cpu().numpy()
        print(f"[R-UPDATE] alpha_prior={alpha:.3f} mean={r_np.mean():.4f} max={r_np.max():.4f} ratio(R>0.5)={(r_np>0.5).mean():.4f}")

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

        inverse_model.train()
        forward_model.train()
        Facies_model.train()

        # weak supervision (Scheme A)
        if (ws_every > 0) and ((epoch % ws_every) == 0):
            ws_running = 0.0
            ws_batches = 0
            for bi, batch in enumerate(Wsupervised_loader):
                if ws_max_batches > 0 and bi >= ws_max_batches:
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

                loss_ws = (lam_fac * criterion_facies(fac_logits, z)) + (lam_rec * criterion_rec(x_rec, x[:, 0:1, :]))

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

            loss = (lam_ai * l_ai) + (lam_fac * l_fac) + (lam_rec * l_rec)
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
