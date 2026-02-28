from __future__ import annotations

import torch

_EDGE_CACHE: dict[tuple[int, int, bool, str], tuple[torch.Tensor, torch.Tensor]] = {}


def get_lattice_edges(
    il: int,
    xl: int,
    diag: bool = True,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return directed lattice edges (src -> dst) for flattened idx=i*XL+j."""
    dev = torch.device("cpu") if device is None else torch.device(device)
    key = (int(il), int(xl), bool(diag), str(dev))
    if key in _EDGE_CACHE:
        return _EDGE_CACHE[key]

    ii = torch.arange(il, device=dev, dtype=torch.long).view(il, 1).expand(il, xl)
    jj = torch.arange(xl, device=dev, dtype=torch.long).view(1, xl).expand(il, xl)

    # Keep direction convention aligned with grid backend:
    # E, W, S, N, SE, SW, NE, NW where dir=(dx,dy).
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    if diag:
        dirs += [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    src_all: list[torch.Tensor] = []
    dst_all: list[torch.Tensor] = []
    for di, dj in dirs:
        ni = ii + di
        nj = jj + dj
        mask = (ni >= 0) & (ni < il) & (nj >= 0) & (nj < xl)
        dst = (ii[mask] * xl + jj[mask]).reshape(-1)
        src = (ni[mask] * xl + nj[mask]).reshape(-1)
        src_all.append(src)
        dst_all.append(dst)

    src_idx = torch.cat(src_all, dim=0).long()
    dst_idx = torch.cat(dst_all, dim=0).long()
    _EDGE_CACHE[key] = (src_idx, dst_idx)
    return src_idx, dst_idx


@torch.no_grad()
def graph_diffuse(
    r0_flat: torch.Tensor,
    well_mask_flat: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    w: torch.Tensor,
    steps: int,
    eta: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Diffusion on directed weighted graph."""
    r = r0_flat.clone()
    for _ in range(int(steps)):
        acc = torch.zeros_like(r)
        den = torch.zeros_like(r)
        acc.index_add_(0, dst, w * r[src])
        den.index_add_(0, dst, w)
        r_prop = acc / (den + eps)
        r = (1.0 - float(eta)) * r + float(eta) * r_prop
        r = torch.maximum(r, well_mask_flat)
    return r


@torch.no_grad()
def compute_lattice_edge_weight(
    v_flat: torch.Tensor,  # [N,2]
    pch_flat: torch.Tensor,  # [N]
    feat_flat: torch.Tensor | None,  # [N,C]
    damp_flat: torch.Tensor | None,  # [N]
    src: torch.Tensor,  # [E]
    dst: torch.Tensor,  # [E]
    il: int,
    xl: int,
    gamma: float,
    tau: float,
    kappa: float,
    anis_strength_flat: torch.Tensor | None = None,  # [N]
    strength_power: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute edge weight w_ij for directed edge (i<-j) where i=dst, j=src."""
    vx = v_flat[:, 0]
    vy = v_flat[:, 1]

    dst_i = torch.div(dst, xl, rounding_mode="floor")
    dst_j = dst - dst_i * xl
    src_i = torch.div(src, xl, rounding_mode="floor")
    src_j = src - src_i * xl

    dx = (src_j - dst_j).to(v_flat.dtype)
    dy = (src_i - dst_i).to(v_flat.dtype)
    dnorm = torch.sqrt(dx * dx + dy * dy + eps)
    dx = dx / dnorm
    dy = dy / dnorm

    cos = dx * vx[dst] + dy * vy[dst]
    g = torch.sigmoid(float(gamma) * (pch_flat[dst] - 0.5))
    if damp_flat is not None:
        g = g * damp_flat[dst].clamp(0.0, 1.0)

    if anis_strength_flat is None:
        strength = torch.ones_like(g)
    else:
        strength = anis_strength_flat[dst].clamp(0.0, 1.0).pow(float(strength_power))
    a = torch.exp(float(kappa) * strength * (cos * cos))

    if feat_flat is None:
        s = torch.ones_like(g)
    else:
        dist = torch.sqrt(((feat_flat[dst] - feat_flat[src]) ** 2).sum(dim=1) + eps)
        s = torch.exp(-dist / (float(tau) + eps))

    return g * a * s
