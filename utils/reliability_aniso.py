from __future__ import annotations

"""Facies-adaptive anisotropic reliability propagation (FARP).

Core idea
---------
We propagate a reliability field R(x) from sparse well seeds in an anisotropic
metric space. The metric is modulated by:
  1) channel-likeness p_ch(x)  (from facies label OR facies probability)
  2) waveform/attribute similarity (from seismic amplitude + gradient)
  3) local dominant orientation (structure tensor of p_ch)

This module supports BOTH:
  - synthetic / labeled case (Stanford VI-E): facies_3d given (hard labels)
  - iterative / pseudo-label case: provide p_channel_3d and conf_3d (or facies_prob_3d)

In iterative coupling, the recommended pattern is:
  p_ch := mix( prior_facies , predicted_facies , alpha ) with confidence gating,
  then R := EMA(R_prev, R_new).
"""

import numpy as np
import torch
import torch.nn.functional as F
from utils.agpe_graph import (
    AGPEGraphCache,
    build_skeleton_graph,
    build_cache_for_slice,
    compute_lattice_edge_weight,
    get_lattice_edges,
    graph_diffuse,
    lift_nodes_to_grid,
    lift_nodes_to_grid_cached,
    maybe_rebuild_cache,
    update_edge_weight_only,
)


def _sobel_xy(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sobel gradients for 2D tensors.

    x: [B,1,H,W]
    returns gx, gy: [B,1,H,W]
    """
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return gx, gy


def _gauss_blur(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Separable Gaussian blur with a small fixed kernel."""
    if sigma <= 0:
        return x
    radius = int(3 * sigma + 0.5)
    size = 2 * radius + 1
    coords = torch.arange(size, device=x.device, dtype=x.dtype) - radius
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / (g.sum() + 1e-12)
    g1 = g.view(1, 1, 1, size)
    g2 = g.view(1, 1, size, 1)
    x = F.conv2d(x, g1.expand(x.size(1), 1, 1, size), padding=(0, radius), groups=x.size(1))
    x = F.conv2d(x, g2.expand(x.size(1), 1, size, 1), padding=(radius, 0), groups=x.size(1))
    return x


def structure_tensor_orientation_and_strength(
    p_channel: torch.Tensor,
    sigma: float = 1.2,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute local dominant direction and anisotropy strength from structure tensor.

    p_channel: [B,1,H,W] (float, 0..1)
    returns:
      v: [B,2,H,W] unit vector (vx, vy)
      strength: [B,1,H,W] in [0,1], a=(lambda1-lambda2)/(lambda1+lambda2)
    """
    gx, gy = _sobel_xy(p_channel)
    j11 = _gauss_blur(gx * gx, sigma)
    j22 = _gauss_blur(gy * gy, sigma)
    j12 = _gauss_blur(gx * gy, sigma)

    angle = 0.5 * torch.atan2(2 * j12, (j11 - j22 + eps))
    vx = torch.cos(angle)
    vy = torch.sin(angle)
    v = torch.cat([vx, vy], dim=1)
    v = v / (torch.sqrt((v * v).sum(dim=1, keepdim=True)) + eps)

    tr = j11 + j22
    delta = torch.sqrt((j11 - j22) * (j11 - j22) + 4.0 * j12 * j12 + eps)
    lam1 = 0.5 * (tr + delta)
    lam2 = 0.5 * (tr - delta)
    strength = ((lam1 - lam2) / (lam1 + lam2 + eps)).clamp(0.0, 1.0)
    return v, strength


def structure_tensor_orientation(p_channel: torch.Tensor, sigma: float = 1.2, eps: float = 1e-8) -> torch.Tensor:
    """Backward-compatible orientation-only wrapper."""
    v, _ = structure_tensor_orientation_and_strength(p_channel, sigma=sigma, eps=eps)
    return v


def _neighbor_shift_8(x: torch.Tensor) -> list[torch.Tensor]:
    """8-neighborhood shifts for [B,C,H,W]."""
    E = F.pad(x[..., :, 1:], (0, 1, 0, 0))
    W = F.pad(x[..., :, :-1], (1, 0, 0, 0))
    S = F.pad(x[..., 1:, :], (0, 0, 0, 1))
    N = F.pad(x[..., :-1, :], (0, 0, 1, 0))
    SE = F.pad(x[..., 1:, 1:], (0, 1, 0, 1))
    SW = F.pad(x[..., 1:, :-1], (1, 0, 0, 1))
    NE = F.pad(x[..., :-1, 1:], (0, 1, 1, 0))
    NW = F.pad(x[..., :-1, :-1], (1, 0, 1, 0))
    return [E, W, S, N, SE, SW, NE, NW]


def _apply_well_constraint(r: torch.Tensor, well_mask: torch.Tensor, well_soft_alpha: float) -> torch.Tensor:
    """Apply hard/soft well constraint to reliability field."""
    alpha = float(well_soft_alpha)
    if alpha >= 1.0 - 1e-8:
        return torch.maximum(r, well_mask)
    if alpha <= 0.0:
        return r
    return ((1.0 - alpha) * r + alpha * well_mask).clamp(0.0, 1.0)


@torch.no_grad()
def anisotropic_reliability_2d(
    well_mask: torch.Tensor,  # [B,1,H,W]
    p_channel: torch.Tensor,  # [B,1,H,W]
    feat: torch.Tensor | None = None,  # [B,C,H,W]
    steps: int = 25,
    eta: float = 0.6,
    gamma: float = 8.0,
    tau: float = 0.6,
    kappa: float = 4.0,
    sigma_st: float = 1.2,
    well_soft_alpha: float = 1.0,
    damp: torch.Tensor | None = None,  # [B,1,H,W] optional, 0..1, smaller -> harder to propagate
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Propagate reliability on a 2D grid slice."""
    v = structure_tensor_orientation(p_channel, sigma=sigma_st, eps=eps)  # [B,2,H,W]
    R = well_mask.clone()

    # channel gate: [0,1], steeper when gamma is large
    g = torch.sigmoid(gamma * (p_channel - 0.5))
    if damp is not None:
        g = g * damp.clamp(0.0, 1.0)

    # 8 unit directions
    dirs = torch.tensor(
        [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, 1], [1, -1], [-1, -1]],
        device=well_mask.device,
        dtype=well_mask.dtype,
    )
    dirs = dirs / (torch.sqrt((dirs * dirs).sum(dim=1, keepdim=True)) + eps)

    for _ in range(steps):
        Rn = _neighbor_shift_8(R)
        vx, vy = v[:, 0:1], v[:, 1:2]

        # orientation affinity
        a_list = []
        for d in dirs:
            cos = d[0] * vx + d[1] * vy
            a_list.append(torch.exp(kappa * (cos * cos)))

        # similarity affinity
        if feat is not None:
            Fn = _neighbor_shift_8(feat)
            s_list = []
            for fnb in Fn:
                dist = torch.sqrt(((feat - fnb) ** 2).sum(dim=1, keepdim=True) + eps)
                s_list.append(torch.exp(-dist / (tau + eps)))
        else:
            s_list = [None] * 8

        w_sum = torch.zeros_like(g)
        w_list = []
        for i in range(8):
            s = s_list[i] if s_list[i] is not None else torch.ones_like(g)
            w = g * a_list[i] * s
            w_list.append(w)
            w_sum = w_sum + w
        w_sum = w_sum + eps

        R_prop = torch.zeros_like(R)
        for w, rnb in zip(w_list, Rn):
            R_prop = R_prop + (w / w_sum) * rnb

        R = (1 - eta) * R + eta * R_prop
        # Keep seed floor during diffusion; apply soft blending only once after iterations.
        R = torch.maximum(R, well_mask)

    if 0.0 < float(well_soft_alpha) < 1.0 - 1e-8:
        R = _apply_well_constraint(R, well_mask, well_soft_alpha)

    return R, v


@torch.no_grad()
def graph_lattice_reliability_2d(
    well_mask: torch.Tensor,  # [B,1,H,W]
    p_channel: torch.Tensor,  # [B,1,H,W]
    feat: torch.Tensor | None = None,  # [B,C,H,W]
    steps: int = 25,
    eta: float = 0.6,
    gamma: float = 8.0,
    tau: float = 0.6,
    kappa: float = 4.0,
    sigma_st: float = 1.2,
    well_soft_alpha: float = 1.0,
    damp: torch.Tensor | None = None,  # [B,1,H,W]
    use_tensor_strength: bool = True,
    tensor_strength_power: float = 1.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Graph diffusion backend on 8-neighbor lattice with explicit directed edges."""
    v, strength = structure_tensor_orientation_and_strength(p_channel, sigma=sigma_st, eps=eps)

    bsz, _, il, xl = well_mask.shape
    src, dst = get_lattice_edges(il=il, xl=xl, diag=True, device=well_mask.device)

    r_out = torch.zeros_like(well_mask)
    for b in range(bsz):
        r0 = well_mask[b, 0].reshape(-1)
        wm = well_mask[b, 0].reshape(-1)
        pch = p_channel[b, 0].reshape(-1)
        vf = v[b].permute(1, 2, 0).reshape(-1, 2)

        feat_f = None if feat is None else feat[b].permute(1, 2, 0).reshape(-1, feat.shape[1])
        damp_f = None if damp is None else damp[b, 0].reshape(-1)
        str_f = strength[b, 0].reshape(-1) if use_tensor_strength else None

        w = compute_lattice_edge_weight(
            v_flat=vf,
            pch_flat=pch,
            feat_flat=feat_f,
            damp_flat=damp_f,
            src=src,
            dst=dst,
            il=il,
            xl=xl,
            gamma=gamma,
            tau=tau,
            kappa=kappa,
            anis_strength_flat=str_f,
            strength_power=tensor_strength_power,
            eps=eps,
        )
        r = graph_diffuse(
            r0,
            wm,
            src,
            dst,
            w,
            steps=steps,
            eta=eta,
            well_soft_alpha=well_soft_alpha,
            eps=eps,
        )
        r_out[b, 0] = r.reshape(il, xl)

    return r_out, v


@torch.no_grad()
def skeleton_graph_reliability_2d(
    well_mask: torch.Tensor,  # [B,1,H,W]
    p_channel: torch.Tensor,  # [B,1,H,W]
    feat: torch.Tensor | None = None,  # [B,C,H,W]
    conf: torch.Tensor | None = None,  # [B,1,H,W]
    steps: int = 25,
    eta: float = 0.6,
    gamma: float = 8.0,
    tau: float = 0.6,
    kappa: float = 4.0,
    sigma_st: float = 1.2,
    well_soft_alpha: float = 1.0,
    damp: torch.Tensor | None = None,  # [B,1,H,W]
    use_tensor_strength: bool = True,
    tensor_strength_power: float = 1.0,
    agpe_skel_p_thresh: float = 0.55,
    agpe_skel_min_nodes: int = 30,
    agpe_skel_snap_radius: int = 5,
    agpe_long_edges: bool = True,
    agpe_long_max_step: int = 6,
    agpe_long_step: int = 2,
    agpe_long_cos_thresh: float = 0.70,
    agpe_long_weight: float = 0.35,
    agpe_edge_tau_p: float = 0.25,
    agpe_lift_sigma: float = 2.2,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Skeleton-graph diffusion with fallback to graph_lattice."""
    v, strength = structure_tensor_orientation_and_strength(p_channel, sigma=sigma_st, eps=eps)

    bsz, _, il, xl = well_mask.shape
    r_out = torch.zeros_like(well_mask)

    def _fallback_one(bidx: int) -> torch.Tensor:
        fb, _ = graph_lattice_reliability_2d(
            well_mask=well_mask[bidx:bidx + 1],
            p_channel=p_channel[bidx:bidx + 1],
            feat=None if feat is None else feat[bidx:bidx + 1],
            steps=steps,
            eta=eta,
            gamma=gamma,
            tau=tau,
            kappa=kappa,
            sigma_st=sigma_st,
            well_soft_alpha=well_soft_alpha,
            damp=None if damp is None else damp[bidx:bidx + 1],
            use_tensor_strength=use_tensor_strength,
            tensor_strength_power=tensor_strength_power,
            eps=eps,
        )
        return fb[0, 0]

    for b in range(bsz):
        conf_b = conf[b:b + 1] if conf is not None else torch.ones_like(p_channel[b:b + 1])
        graph = build_skeleton_graph(
            pch=p_channel[b, 0],
            conf=conf_b[0, 0],
            v=v[b],
            il=il,
            xl=xl,
            p_thresh=float(agpe_skel_p_thresh),
            min_nodes=int(agpe_skel_min_nodes),
            long_edges=bool(agpe_long_edges),
            long_max_step=int(agpe_long_max_step),
            long_step=int(agpe_long_step),
            long_cos_thresh=float(agpe_long_cos_thresh),
        )
        if graph is None or graph["node_rc"].shape[0] == 0 or graph["src"].size == 0:
            r_out[b, 0] = _fallback_one(b)
            continue

        node_rc = graph["node_rc"]
        node_map = graph["node_map"]
        skel_mask = graph["skel_mask"]
        src_np = graph["src"]
        dst_np = graph["dst"]
        is_long_np = graph["is_long"]

        # map well seeds to skeleton nodes; if none can be snapped, fallback
        wm_np = well_mask[b, 0].detach().cpu().numpy()
        seed_rc = np.argwhere(wm_np)
        seed_nodes: dict[int, float] = {}
        snap_radius = max(int(agpe_skel_snap_radius), 0)
        for r0, c0 in seed_rc:
            wv = float(wm_np[int(r0), int(c0)])
            if wv <= 0.0:
                continue
            idx = int(node_map[r0, c0])
            if idx >= 0:
                seed_nodes[idx] = max(seed_nodes.get(idx, 0.0), wv)
                continue
            if snap_radius <= 0:
                continue
            rr0 = max(0, int(r0) - snap_radius)
            rr1 = min(il, int(r0) + snap_radius + 1)
            cc0 = max(0, int(c0) - snap_radius)
            cc1 = min(xl, int(c0) + snap_radius + 1)
            sub = node_map[rr0:rr1, cc0:cc1]
            cand = np.argwhere(sub >= 0)
            if cand.size == 0:
                continue
            cand[:, 0] += rr0
            cand[:, 1] += cc0
            d2 = (cand[:, 0] - int(r0)) ** 2 + (cand[:, 1] - int(c0)) ** 2
            best = cand[int(np.argmin(d2))]
            best_idx = int(node_map[int(best[0]), int(best[1])])
            if best_idx >= 0:
                seed_nodes[best_idx] = max(seed_nodes.get(best_idx, 0.0), wv)

        if not seed_nodes:
            r_out[b, 0] = _fallback_one(b)
            continue

        device = well_mask.device
        src = torch.from_numpy(src_np).to(device=device, dtype=torch.long)
        dst = torch.from_numpy(dst_np).to(device=device, dtype=torch.long)
        is_long = torch.from_numpy(is_long_np.astype(np.uint8)).to(device=device, dtype=torch.bool)

        rc_t = torch.from_numpy(node_rc).to(device=device, dtype=torch.long)
        rr = rc_t[:, 0]
        cc = rc_t[:, 1]

        pch_nodes = p_channel[b, 0, rr, cc]
        conf_nodes = conf_b[0, 0, rr, cc].clamp(0.0, 1.0)
        damp_nodes = None if damp is None else damp[b, 0, rr, cc].clamp(0.0, 1.0)

        v_nodes = v[b].permute(1, 2, 0)[rr, cc]
        v_norm = torch.sqrt((v_nodes * v_nodes).sum(dim=1, keepdim=True) + eps)
        v_nodes = v_nodes / v_norm

        if use_tensor_strength:
            strength_nodes = strength[b, 0, rr, cc].clamp(0.0, 1.0).pow(float(tensor_strength_power))
        else:
            strength_nodes = torch.ones_like(pch_nodes)

        if feat is None:
            feat_nodes = None
        else:
            feat_nodes = feat[b].permute(1, 2, 0)[rr, cc]  # [M,C]

        src_r = rr[src]
        src_c = cc[src]
        dst_r = rr[dst]
        dst_c = cc[dst]

        dx = (src_c - dst_c).to(pch_nodes.dtype)
        dy = (src_r - dst_r).to(pch_nodes.dtype)
        dnorm = torch.sqrt(dx * dx + dy * dy + eps)
        dx = dx / dnorm
        dy = dy / dnorm

        cos = dx * v_nodes[dst, 0] + dy * v_nodes[dst, 1]
        g = torch.sigmoid(float(gamma) * (pch_nodes[dst] - 0.5))
        if damp_nodes is not None:
            g = g * damp_nodes[dst]
        a = torch.exp(float(kappa) * strength_nodes[dst] * (cos * cos))

        if feat_nodes is None:
            s = torch.ones_like(g)
        else:
            dist = torch.sqrt(((feat_nodes[dst] - feat_nodes[src]) ** 2).sum(dim=1) + eps)
            s = torch.exp(-dist / (float(tau) + eps))

        # edge-level facies-aware penalty + confidence gate
        edge_tau_p = max(float(agpe_edge_tau_p), eps)
        p_edge = torch.exp(-torch.abs(pch_nodes[dst] - pch_nodes[src]) / edge_tau_p)
        p_edge = p_edge * (conf_nodes[dst] * conf_nodes[src])

        w = g * a * s * p_edge
        if bool(agpe_long_edges):
            w = torch.where(is_long, w * float(agpe_long_weight), w)
        w = w.clamp(min=eps)

        m = int(node_rc.shape[0])
        r0 = torch.zeros((m,), dtype=well_mask.dtype, device=device)
        wm_node = torch.zeros_like(r0)
        seed_keys = sorted(seed_nodes.keys())
        seed_idx = torch.tensor(seed_keys, dtype=torch.long, device=device)
        seed_val = torch.tensor([seed_nodes[int(k)] for k in seed_keys], dtype=well_mask.dtype, device=device)
        seed_val = seed_val.clamp(0.0, 1.0)
        r0[seed_idx] = seed_val
        wm_node[seed_idx] = seed_val

        r_nodes = graph_diffuse(
            r0,
            wm_node,
            src,
            dst,
            w,
            steps=steps,
            eta=eta,
            well_soft_alpha=well_soft_alpha,
            eps=eps,
        )
        r_grid_np = lift_nodes_to_grid(
            r_nodes.detach().cpu().numpy().astype(np.float32),
            node_map=node_map,
            skel_mask=skel_mask,
            lift_sigma=float(agpe_lift_sigma),
        )
        r_grid = torch.from_numpy(r_grid_np).to(device=device, dtype=well_mask.dtype)
        r_grid = _apply_well_constraint(r_grid, well_mask[b, 0], well_soft_alpha)
        r_out[b, 0] = r_grid.clamp(0.0, 1.0)

    return r_out, v


def _snap_well_seeds_to_skeleton(
    well_mask_2d: np.ndarray,
    node_map: np.ndarray,
    snap_radius: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Map 2D well seeds to skeleton nodes with optional local snapping."""
    wm_np = np.asarray(well_mask_2d, dtype=np.float32)
    node_map = np.asarray(node_map, dtype=np.int32)
    il, xl = node_map.shape

    seed_rc = np.argwhere(wm_np > 0.0)
    if seed_rc.size == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32), 0, 0

    seed_nodes: dict[int, float] = {}
    snap_ok = 0
    r_snap = max(int(snap_radius), 0)
    for r0, c0 in seed_rc:
        wv = float(wm_np[int(r0), int(c0)])
        if wv <= 0.0:
            continue
        idx = int(node_map[int(r0), int(c0)])
        if idx >= 0:
            seed_nodes[idx] = max(seed_nodes.get(idx, 0.0), wv)
            snap_ok += 1
            continue
        if r_snap <= 0:
            continue

        rr0 = max(0, int(r0) - r_snap)
        rr1 = min(il, int(r0) + r_snap + 1)
        cc0 = max(0, int(c0) - r_snap)
        cc1 = min(xl, int(c0) + r_snap + 1)
        sub = node_map[rr0:rr1, cc0:cc1]
        cand = np.argwhere(sub >= 0)
        if cand.size == 0:
            continue
        cand[:, 0] += rr0
        cand[:, 1] += cc0
        d2 = (cand[:, 0] - int(r0)) ** 2 + (cand[:, 1] - int(c0)) ** 2
        best = cand[int(np.argmin(d2))]
        best_idx = int(node_map[int(best[0]), int(best[1])])
        if best_idx >= 0:
            seed_nodes[best_idx] = max(seed_nodes.get(best_idx, 0.0), wv)
            snap_ok += 1

    if not seed_nodes:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32), int(seed_rc.shape[0]), int(snap_ok)

    seed_idx = np.asarray(sorted(seed_nodes.keys()), dtype=np.int64)
    seed_val = np.asarray([seed_nodes[int(k)] for k in seed_idx], dtype=np.float32)
    seed_val = np.clip(seed_val, 0.0, 1.0)
    return seed_idx, seed_val, int(seed_rc.shape[0]), int(snap_ok)


def _ensure_probs(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """If x looks like logits, convert to probs; else assume already probs."""
    # Heuristic: if values outside [0,1] OR row-sum not ~1 => treat as logits.
    if x.numel() == 0:
        return x
    x_min = float(x.min())
    x_max = float(x.max())
    if (x_min < -1e-3) or (x_max > 1.0 + 1e-3):
        return torch.softmax(x, dim=dim)
    # already probs (best effort)
    return x


@torch.no_grad()
def build_R_and_prior_from_cube(
    seismic_3d: torch.Tensor,  # [H,IL,XL]
    ai_3d: torch.Tensor,       # [H,IL,XL] float
    well_trace_indices: torch.Tensor,  # [Nw] indices in flattened IL*XL order
    *,
    # --- channel-likeness sources (choose one) ---
    facies_3d: torch.Tensor | None = None,        # [H,IL,XL] int (hard labels)
    facies_prob_3d: torch.Tensor | None = None,   # [H,IL,XL,K] or [K,H,IL,XL] (prob OR logits)
    p_channel_3d: torch.Tensor | None = None,     # [H,IL,XL] float in [0,1]
    conf_3d: torch.Tensor | None = None,          # [H,IL,XL] float in [0,1]
    facies_prior_3d: torch.Tensor | None = None,  # [H,IL,XL] int (prior / anchor, e.g. from interpreter)
    # --- mixing / gating ---
    channel_id: int = 2,
    alpha_prior: float = 1.0,     # 1.0 => only prior/hard; 0.0 => only predicted
    conf_thresh: float = 0.75,    # below this, fall back to neutral/prior
    neutral_p: float = 0.5,       # fallback p_channel when no prior
    # --- anisotropic propagation ---
    steps_R: int = 25,
    eta: float = 0.6,
    gamma: float = 8.0,
    tau: float = 0.6,
    kappa: float = 4.0,
    sigma_st: float = 1.2,
    backend: str = "grid",                 # "grid" | "graph_lattice" | "skeleton_graph"
    aniso_use_tensor_strength: bool = True,
    aniso_tensor_strength_power: float = 1.0,
    # --- skeleton graph params ---
    agpe_skel_p_thresh: float = 0.55,
    agpe_skel_min_nodes: int = 30,
    agpe_skel_snap_radius: int = 5,
    agpe_long_edges: bool = True,
    agpe_long_max_step: int = 6,
    agpe_long_step: int = 2,
    agpe_long_cos_thresh: float = 0.70,
    agpe_long_weight: float = 0.35,
    agpe_edge_tau_p: float = 0.25,
    agpe_lift_sigma: float = 2.2,
    agpe_well_seed_mode: str = "hard",     # "hard" | "depth_gate"
    agpe_well_seed_power: float = 1.0,
    agpe_well_seed_min: float = 0.02,
    agpe_well_seed_use_conf: bool = True,
    agpe_well_soft_alpha: float = 0.20,
    agpe_cache_graph: bool = True,
    agpe_refine_graph: bool = True,
    agpe_rebuild_every: int = 50,
    agpe_topo_change_pch_l1: float = 0.05,
    epoch: int | None = None,
    graph_cache: AGPEGraphCache | None = None,
    return_graph_cache: bool = False,
    # --- physics damping (optional) ---
    phys_residual_3d: torch.Tensor | None = None,  # [H,IL,XL] >=0, larger => harder to propagate
    lambda_phys: float = 0.0,                      # 0 disables
    # --- optional soft impedance prior ---
    use_soft_prior: bool = False,
    steps_prior: int = 35,
    eps: float = 1e-8,
) -> (
    tuple[torch.Tensor, torch.Tensor | None]
    | tuple[torch.Tensor, torch.Tensor | None, AGPEGraphCache]
):
    """Build R(x) (and optional impedance prior) for the full cube.

    Returns:
      R_flat: [N, H] where N=IL*XL, values in [0,1]
      prior_flat (optional): [N,H]
      graph_cache (optional): only returned when return_graph_cache=True
    """
    device = seismic_3d.device
    H, IL, XL = seismic_3d.shape
    N = IL * XL

    # well trace locations (depth weighting is applied after p_channel/conf are prepared)
    well_seed_base = torch.zeros((H, IL, XL), device=device, dtype=torch.float32)
    ii = (well_trace_indices // XL).long()
    jj = (well_trace_indices % XL).long()
    well_seed_base[:, ii, jj] = 1.0

    # ---------- build p_channel and conf ----------
    # 1) prior channel probability (hard)
    if facies_prior_3d is not None:
        p_prior = (facies_prior_3d == channel_id).float()
    elif facies_3d is not None:
        p_prior = (facies_3d == channel_id).float()
    else:
        p_prior = None

    # 2) predicted / provided p_channel
    p_pred = None
    conf = None

    if p_channel_3d is not None:
        p_pred = p_channel_3d.float()
        conf = conf_3d.float() if conf_3d is not None else torch.ones_like(p_pred)
    elif facies_prob_3d is not None:
        fp = facies_prob_3d
        if fp.dim() == 4 and fp.shape[0] != H:
            # maybe [K,H,IL,XL] -> [H,IL,XL,K]
            fp = fp.permute(1, 2, 3, 0).contiguous()
        fp = _ensure_probs(fp.float(), dim=-1)
        p_pred = fp[..., channel_id]
        conf = fp.max(dim=-1).values
    elif facies_3d is not None:
        # fallback: hard labels as "pred"
        p_pred = (facies_3d == channel_id).float()
        conf = torch.ones_like(p_pred)

    if p_pred is None:
        raise ValueError("Need one of: p_channel_3d / facies_prob_3d / facies_3d.")

    # 3) mix + confidence gating
    if p_prior is not None:
        p_mix = alpha_prior * p_prior + (1.0 - alpha_prior) * p_pred
        p_fallback = alpha_prior * p_prior + (1.0 - alpha_prior) * float(neutral_p)
    else:
        p_mix = p_pred
        p_fallback = torch.full_like(p_pred, float(neutral_p))

    if conf is None:
        conf = torch.ones_like(p_pred)

    pch = torch.where(conf >= float(conf_thresh), p_mix, p_fallback).clamp(0.0, 1.0)

    # ---------- depth-gated well seed injection ----------
    seed_mode = str(agpe_well_seed_mode).strip().lower()
    seed_power = max(float(agpe_well_seed_power), 0.0)
    seed_min = float(np.clip(float(agpe_well_seed_min), 0.0, 1.0))
    if seed_mode == "depth_gate":
        seed_w = pch.clamp(0.0, 1.0)
        if bool(agpe_well_seed_use_conf):
            seed_w = seed_w * conf.clamp(0.0, 1.0)
        if abs(seed_power - 1.0) > 1e-8:
            seed_w = seed_w.pow(seed_power)
        if seed_min > 0.0:
            seed_w = seed_w.clamp(min=seed_min, max=1.0)
        well_mask_3d = well_seed_base * seed_w
    else:
        well_mask_3d = well_seed_base

    # ---------- waveform/attribute embedding for similarity ----------
    amp = seismic_3d
    gx = F.pad(amp[:, :, 1:] - amp[:, :, :-1], (0, 1, 0, 0))
    gy = F.pad(amp[:, 1:, :] - amp[:, :-1, :], (0, 0, 0, 1))
    gmag = torch.sqrt(gx * gx + gy * gy + eps)
    feat = torch.stack([amp, gmag], dim=1)  # [H,2,IL,XL]

    # ---------- optional physics damping ----------
    if (phys_residual_3d is not None) and (lambda_phys > 0):
        res = phys_residual_3d.float().clamp(min=0.0)
        # robust normalization per depth slice (avoid being dominated by outliers)
        med = torch.median(res.reshape(H, -1), dim=1).values.view(H, 1, 1) + eps
        r = (res / med).clamp(0.0, 10.0)
        damp3d = torch.exp(-float(lambda_phys) * r).clamp(0.0, 1.0)  # 1 good, 0 bad
    else:
        damp3d = None

    # ---------- do 2D propagation per depth slice ----------
    R_out = torch.zeros((H, IL, XL), device=device, dtype=torch.float32)

    if use_soft_prior:
        prior = torch.zeros((H, IL, XL), device=device, dtype=torch.float32)
        prior[:, ii, jj] = ai_3d[:, ii, jj]
    else:
        prior = None

    backend_name = str(backend)
    cache_obj = graph_cache if isinstance(graph_cache, AGPEGraphCache) else AGPEGraphCache()
    if not bool(agpe_cache_graph):
        cache_obj.slices.clear()

    graph_stats = {
        "backend": backend_name,
        "valid_graph_slices": 0.0,
        "n_nodes_sum": 0.0,
        "n_edges_sum": 0.0,
        "n_long_edges_sum": 0.0,
        "snap_total": 0.0,
        "snap_ok": 0.0,
        "fallback_slices": 0.0,
        "rebuild_slices": 0.0,
        "cache_hits": 0.0,
    }

    for k in range(H):
        wm = well_mask_3d[k:k+1].unsqueeze(1)      # [1,1,IL,XL]
        pc = pch[k:k+1].unsqueeze(1)               # [1,1,IL,XL]
        fk = feat[k:k+1]                           # [1,2,IL,XL]
        ck = conf[k:k+1].unsqueeze(1) if conf is not None else None
        dk = damp3d[k:k+1].unsqueeze(1) if damp3d is not None else None

        if backend_name == "grid":
            Rk, _ = anisotropic_reliability_2d(
                wm,
                pc,
                fk,
                steps=steps_R,
                eta=eta,
                gamma=gamma,
                tau=tau,
                kappa=kappa,
                sigma_st=sigma_st,
                well_soft_alpha=float(agpe_well_soft_alpha),
                damp=dk,
            )
        elif backend_name == "graph_lattice":
            Rk, _ = graph_lattice_reliability_2d(
                wm,
                pc,
                fk,
                steps=steps_R,
                eta=eta,
                gamma=gamma,
                tau=tau,
                kappa=kappa,
                sigma_st=sigma_st,
                well_soft_alpha=float(agpe_well_soft_alpha),
                damp=dk,
                use_tensor_strength=bool(aniso_use_tensor_strength),
                tensor_strength_power=float(aniso_tensor_strength_power),
            )
        elif backend_name == "skeleton_graph":
            v_k, str_k = structure_tensor_orientation_and_strength(pc, sigma=sigma_st, eps=eps)
            pch_np = pc[0, 0].detach().cpu().numpy().astype(np.float32)
            conf_np = (
                ck[0, 0].detach().cpu().numpy().astype(np.float32)
                if ck is not None
                else np.ones_like(pch_np, dtype=np.float32)
            )
            mask_np = (pch_np >= float(agpe_skel_p_thresh)) & (conf_np > 0.0)

            cache_slice = cache_obj.slices.get(k) if bool(agpe_cache_graph) else None
            if cache_slice is None:
                pch_l1 = float("inf")
            else:
                prev = cache_slice.pch_prev
                pch_l1 = float(np.mean(np.abs(pch_np - prev))) if prev.shape == pch_np.shape else float("inf")

            if cache_slice is None:
                rebuild = True
            elif bool(agpe_refine_graph):
                periodic_rebuild = (
                    (epoch is not None)
                    and (int(agpe_rebuild_every) > 0)
                    and (int(epoch) % int(agpe_rebuild_every) == 0)
                )
                rebuild = maybe_rebuild_cache(
                    cache_slice=cache_slice,
                    new_mask=mask_np,
                    topo_change_metric=pch_l1,
                    threshold=float(agpe_topo_change_pch_l1),
                    force=bool(periodic_rebuild),
                )
            else:
                rebuild = False

            if rebuild:
                cache_slice = build_cache_for_slice(
                    pch=pc[0, 0],
                    conf=ck[0, 0] if ck is not None else None,
                    v=v_k[0],
                    il=IL,
                    xl=XL,
                    p_thresh=float(agpe_skel_p_thresh),
                    min_nodes=int(agpe_skel_min_nodes),
                    long_edges=bool(agpe_long_edges),
                    long_max_step=int(agpe_long_max_step),
                    long_step=int(agpe_long_step),
                    long_cos_thresh=float(agpe_long_cos_thresh),
                )
                graph_stats["rebuild_slices"] += 1.0
                if bool(agpe_cache_graph):
                    if cache_slice is None:
                        cache_obj.slices.pop(k, None)
                    else:
                        cache_obj.slices[k] = cache_slice
            else:
                graph_stats["cache_hits"] += 1.0

            if cache_slice is None or cache_slice.n_nodes <= 0 or cache_slice.n_edges <= 0:
                Rk, _ = graph_lattice_reliability_2d(
                    wm,
                    pc,
                    fk,
                    steps=steps_R,
                    eta=eta,
                    gamma=gamma,
                    tau=tau,
                    kappa=kappa,
                    sigma_st=sigma_st,
                    well_soft_alpha=float(agpe_well_soft_alpha),
                    damp=dk,
                    use_tensor_strength=bool(aniso_use_tensor_strength),
                    tensor_strength_power=float(aniso_tensor_strength_power),
                    eps=eps,
                )
                graph_stats["fallback_slices"] += 1.0
            else:
                seed_idx_np, seed_val_np, seed_total, snap_ok = _snap_well_seeds_to_skeleton(
                    well_mask_2d=wm[0, 0].detach().cpu().numpy(),
                    node_map=cache_slice.node_map,
                    snap_radius=int(agpe_skel_snap_radius),
                )
                graph_stats["snap_total"] += float(seed_total)
                graph_stats["snap_ok"] += float(snap_ok)
                cache_slice.snap_total = int(seed_total)
                cache_slice.snap_ok = int(snap_ok)

                if seed_idx_np.size == 0:
                    Rk, _ = graph_lattice_reliability_2d(
                        wm,
                        pc,
                        fk,
                        steps=steps_R,
                        eta=eta,
                        gamma=gamma,
                        tau=tau,
                        kappa=kappa,
                        sigma_st=sigma_st,
                        well_soft_alpha=float(agpe_well_soft_alpha),
                        damp=dk,
                        use_tensor_strength=bool(aniso_use_tensor_strength),
                        tensor_strength_power=float(aniso_tensor_strength_power),
                        eps=eps,
                    )
                    cache_slice.fallback = True
                    graph_stats["fallback_slices"] += 1.0
                else:
                    src, dst, w = update_edge_weight_only(
                        cache_slice=cache_slice,
                        pch=pc[0, 0],
                        conf=ck[0, 0] if ck is not None else None,
                        v=v_k[0],
                        strength=str_k[0],
                        feat=fk[0],
                        damp=dk[0, 0] if dk is not None else None,
                        gamma=float(gamma),
                        tau=float(tau),
                        kappa=float(kappa),
                        use_tensor_strength=bool(aniso_use_tensor_strength),
                        tensor_strength_power=float(aniso_tensor_strength_power),
                        agpe_edge_tau_p=float(agpe_edge_tau_p),
                        agpe_long_edges=bool(agpe_long_edges),
                        agpe_long_weight=float(agpe_long_weight),
                        eps=eps,
                    )

                    node_n = int(cache_slice.n_nodes)
                    r0 = torch.zeros((node_n,), device=device, dtype=torch.float32)
                    wm_node = torch.zeros_like(r0)
                    seed_idx = torch.from_numpy(seed_idx_np).to(device=device, dtype=torch.long)
                    seed_val = torch.from_numpy(seed_val_np).to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
                    r0[seed_idx] = seed_val
                    wm_node[seed_idx] = seed_val

                    r_nodes = graph_diffuse(
                        r0,
                        wm_node,
                        src,
                        dst,
                        w,
                        steps=steps_R,
                        eta=eta,
                        well_soft_alpha=float(agpe_well_soft_alpha),
                        eps=eps,
                    )
                    r_grid_np = lift_nodes_to_grid_cached(
                        r_nodes=r_nodes.detach().cpu().numpy().astype(np.float32),
                        lift_nearest_idx=cache_slice.lift_nearest_idx,
                        lift_dist=cache_slice.lift_dist,
                        lift_sigma=float(agpe_lift_sigma),
                    )
                    r_grid = torch.from_numpy(r_grid_np).to(device=device, dtype=torch.float32)
                    r_grid = _apply_well_constraint(r_grid, wm[0, 0], float(agpe_well_soft_alpha))
                    Rk = r_grid.unsqueeze(0).unsqueeze(0).clamp(0.0, 1.0)

                    cache_slice.pch_prev = pch_np.copy()
                    cache_slice.mask = mask_np.copy()
                    cache_slice.fallback = False

                    graph_stats["valid_graph_slices"] += 1.0
                    graph_stats["n_nodes_sum"] += float(cache_slice.n_nodes)
                    graph_stats["n_edges_sum"] += float(cache_slice.n_edges)
                    graph_stats["n_long_edges_sum"] += float(cache_slice.n_long_edges)
                    if bool(agpe_cache_graph):
                        cache_obj.slices[k] = cache_slice
        else:
            raise ValueError(
                f"Unknown backend='{backend_name}', expected 'grid', 'graph_lattice', or 'skeleton_graph'."
            )

        R_out[k] = Rk[0, 0]

        if use_soft_prior and prior is not None:
            pk = prior[k:k+1].unsqueeze(1)  # [1,1,IL,XL]
            cond = Rk.clamp(0, 1)
            for _ in range(steps_prior):
                pn = _neighbor_shift_8(pk)
                condn = _neighbor_shift_8(cond)
                wsum = torch.zeros_like(pk)
                acc = torch.zeros_like(pk)
                for wv, pnb in zip(condn, pn):
                    wsum = wsum + wv
                    acc = acc + wv * pnb
                pk = acc / (wsum + eps)
                pk = pk * (1 - wm) + prior[k:k+1].unsqueeze(1) * wm
            prior[k] = pk[0, 0]

    # flatten to [N,H]
    R_flat = R_out.reshape(H, N).transpose(0, 1).contiguous()
    if prior is not None:
        prior_flat = prior.reshape(H, N).transpose(0, 1).contiguous()
    else:
        prior_flat = None

    if backend_name == "skeleton_graph":
        valid = max(float(graph_stats["valid_graph_slices"]), 1.0)
        edge_sum = float(graph_stats["n_edges_sum"])
        snap_total = float(graph_stats["snap_total"])
        graph_stats_out = {
            "backend": backend_name,
            "n_nodes_mean": float(graph_stats["n_nodes_sum"]) / valid if graph_stats["valid_graph_slices"] > 0 else 0.0,
            "n_edges_mean": float(graph_stats["n_edges_sum"]) / valid if graph_stats["valid_graph_slices"] > 0 else 0.0,
            "long_edge_ratio": float(graph_stats["n_long_edges_sum"]) / max(edge_sum, 1.0),
            "snap_ok_ratio": float(graph_stats["snap_ok"]) / max(snap_total, 1.0),
            "fallback_slices": int(graph_stats["fallback_slices"]),
            "rebuild_slices": int(graph_stats["rebuild_slices"]),
            "cache_hits": int(graph_stats["cache_hits"]),
        }
    else:
        graph_stats_out = {
            "backend": backend_name,
            "n_nodes_mean": 0.0,
            "n_edges_mean": 0.0,
            "long_edge_ratio": 0.0,
            "snap_ok_ratio": 0.0,
            "fallback_slices": 0,
            "rebuild_slices": 0,
            "cache_hits": 0,
        }

    cache_obj.last_stats = graph_stats_out
    if return_graph_cache:
        return R_flat, prior_flat, cache_obj
    return R_flat, prior_flat
