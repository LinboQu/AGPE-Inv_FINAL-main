from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

_EDGE_CACHE: dict[tuple[int, int, bool, str], tuple[torch.Tensor, torch.Tensor]] = {}


@dataclass
class AGPESliceCache:
    """Per-depth skeleton graph cache entry."""

    node_rc: np.ndarray
    node_map: np.ndarray
    skel_mask: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    is_long: np.ndarray
    lift_nearest_idx: np.ndarray
    lift_dist: np.ndarray
    mask: np.ndarray
    pch_prev: np.ndarray
    n_nodes: int
    n_edges: int
    n_long_edges: int
    snap_ok: int = 0
    snap_total: int = 0
    fallback: bool = False


@dataclass
class AGPEGraphCache:
    """Cube-level graph cache for iterative AGPE updates."""

    slices: dict[int, AGPESliceCache] = field(default_factory=dict)
    last_stats: dict[str, float] = field(default_factory=dict)


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
    well_soft_alpha: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Diffusion on directed weighted graph."""
    alpha = float(well_soft_alpha)
    r = r0_flat.clone()
    for _ in range(int(steps)):
        acc = torch.zeros_like(r)
        den = torch.zeros_like(r)
        acc.index_add_(0, dst, w * r[src])
        den.index_add_(0, dst, w)
        r_prop = acc / (den + eps)
        r = (1.0 - float(eta)) * r + float(eta) * r_prop
        # Keep seed floor during diffusion; apply soft blending only once at the end.
        r = torch.maximum(r, well_mask_flat)
    if 0.0 < alpha < 1.0 - 1e-8:
        r = ((1.0 - alpha) * r + alpha * well_mask_flat).clamp(0.0, 1.0)
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


def zhang_suen_thinning(mask: np.ndarray) -> np.ndarray:
    """Pure-numpy Zhang-Suen thinning for a binary 2D mask."""
    img = (np.asarray(mask) > 0).astype(np.uint8)
    if img.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape={img.shape}")
    if img.shape[0] < 3 or img.shape[1] < 3:
        return img

    def _neighbors(y: int, x: int) -> tuple[int, int, int, int, int, int, int, int]:
        p2 = img[y - 1, x]
        p3 = img[y - 1, x + 1]
        p4 = img[y, x + 1]
        p5 = img[y + 1, x + 1]
        p6 = img[y + 1, x]
        p7 = img[y + 1, x - 1]
        p8 = img[y, x - 1]
        p9 = img[y - 1, x - 1]
        return p2, p3, p4, p5, p6, p7, p8, p9

    def _transitions(nei: tuple[int, int, int, int, int, int, int, int]) -> int:
        seq = list(nei) + [nei[0]]
        t = 0
        for i in range(8):
            if seq[i] == 0 and seq[i + 1] == 1:
                t += 1
        return t

    changed = True
    h, w = img.shape
    while changed:
        changed = False

        to_del_1: list[tuple[int, int]] = []
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if img[y, x] != 1:
                    continue
                n = _neighbors(y, x)
                b = int(sum(n))
                if b < 2 or b > 6:
                    continue
                a = _transitions(n)
                if a != 1:
                    continue
                p2, p3, p4, p5, p6, p7, p8, p9 = n
                if p2 * p4 * p6 != 0:
                    continue
                if p4 * p6 * p8 != 0:
                    continue
                to_del_1.append((y, x))

        if to_del_1:
            changed = True
            for y, x in to_del_1:
                img[y, x] = 0

        to_del_2: list[tuple[int, int]] = []
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if img[y, x] != 1:
                    continue
                n = _neighbors(y, x)
                b = int(sum(n))
                if b < 2 or b > 6:
                    continue
                a = _transitions(n)
                if a != 1:
                    continue
                p2, p3, p4, p5, p6, p7, p8, p9 = n
                if p2 * p4 * p8 != 0:
                    continue
                if p2 * p6 * p8 != 0:
                    continue
                to_del_2.append((y, x))

        if to_del_2:
            changed = True
            for y, x in to_del_2:
                img[y, x] = 0

    return img


def _to_numpy_2d(x: torch.Tensor | np.ndarray, il: int, xl: int, name: str) -> np.ndarray:
    arr = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D after squeeze, got shape={arr.shape}")
    if arr.shape != (il, xl):
        arr = arr.reshape(il, xl)
    return arr


def _to_numpy_v(x: torch.Tensor | np.ndarray, il: int, xl: int) -> np.ndarray:
    arr = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"v must be 3D after squeeze, got shape={arr.shape}")
    if arr.shape[0] == 2 and arr.shape[1] == il and arr.shape[2] == xl:
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.shape[0] == il and arr.shape[1] == xl and arr.shape[2] == 2:
        pass
    else:
        raise ValueError(f"Unsupported v shape={arr.shape}, expected (2,IL,XL) or (IL,XL,2)")
    return arr.astype(np.float32)


def build_skeleton_graph(
    pch: torch.Tensor | np.ndarray,
    conf: torch.Tensor | np.ndarray | None,
    v: torch.Tensor | np.ndarray,
    il: int,
    xl: int,
    p_thresh: float,
    min_nodes: int,
    long_edges: bool,
    long_max_step: int,
    long_step: int,
    long_cos_thresh: float,
) -> dict[str, np.ndarray] | None:
    """Build skeleton graph with local 8-neighbor edges and optional long-range edges."""
    pch_np = _to_numpy_2d(pch, il, xl, "pch").astype(np.float32)
    conf_np = None if conf is None else _to_numpy_2d(conf, il, xl, "conf").astype(np.float32)
    v_np = _to_numpy_v(v, il, xl)

    mask = pch_np >= float(p_thresh)
    if conf_np is not None:
        mask = mask & (conf_np > 0.0)

    if not np.any(mask):
        return None

    skel_mask = zhang_suen_thinning(mask.astype(np.uint8)).astype(bool)
    node_rc = np.argwhere(skel_mask).astype(np.int32)
    if node_rc.shape[0] < int(min_nodes):
        return None

    node_map = np.full((il, xl), -1, dtype=np.int32)
    for idx, (r, c) in enumerate(node_rc):
        node_map[r, c] = int(idx)

    edge_kind: dict[tuple[int, int], bool] = {}

    def _add_edge(s: int, d: int, is_long_edge: bool) -> None:
        key = (int(s), int(d))
        prev = edge_kind.get(key)
        if prev is None:
            edge_kind[key] = bool(is_long_edge)
        elif is_long_edge and (not prev):
            edge_kind[key] = True

    # local 8-neighbor skeleton edges (directed)
    for i, (r, c) in enumerate(node_rc):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            rr = int(r + dr)
            cc = int(c + dc)
            if rr < 0 or rr >= il or cc < 0 or cc >= xl:
                continue
            j = int(node_map[rr, cc])
            if j >= 0 and j != i:
                _add_edge(i, j, False)

    # long-range edges along +/- local orientation
    if bool(long_edges):
        max_step = max(int(long_max_step), 0)
        step_size = max(int(long_step), 1)
        cos_th = float(long_cos_thresh)

        for i, (r, c) in enumerate(node_rc):
            vx = float(v_np[r, c, 0])
            vy = float(v_np[r, c, 1])
            nrm = float(np.hypot(vx, vy))
            if nrm <= 1e-8:
                continue
            vx /= nrm
            vy /= nrm

            for sign in (1.0, -1.0):
                dir_x = sign * vx
                dir_y = sign * vy
                connected = False
                for step in range(step_size, max_step + 1, step_size):
                    rr = int(round(float(r) + dir_y * float(step)))
                    cc = int(round(float(c) + dir_x * float(step)))
                    if rr < 0 or rr >= il or cc < 0 or cc >= xl:
                        continue

                    r0 = max(0, rr - 1)
                    r1 = min(il, rr + 2)
                    c0 = max(0, cc - 1)
                    c1 = min(xl, cc + 2)
                    local_map = node_map[r0:r1, c0:c1]
                    cand_local = np.argwhere(local_map >= 0)
                    if cand_local.size == 0:
                        continue

                    cand_local[:, 0] += r0
                    cand_local[:, 1] += c0
                    d2 = (cand_local[:, 0] - rr) ** 2 + (cand_local[:, 1] - cc) ** 2
                    best = cand_local[int(np.argmin(d2))]
                    br, bc = int(best[0]), int(best[1])
                    j = int(node_map[br, bc])
                    if j < 0 or j == i:
                        continue

                    dx = float(bc - c)
                    dy = float(br - r)
                    dnorm = float(np.hypot(dx, dy))
                    if dnorm <= 1e-8:
                        continue
                    cos_val = abs((dx / dnorm) * vx + (dy / dnorm) * vy)
                    if cos_val >= cos_th:
                        _add_edge(i, j, True)
                        _add_edge(j, i, True)
                        connected = True
                        break
                if connected:
                    continue

    if not edge_kind:
        src = np.empty((0,), dtype=np.int64)
        dst = np.empty((0,), dtype=np.int64)
        is_long = np.empty((0,), dtype=bool)
    else:
        items = sorted(edge_kind.items(), key=lambda kv: (kv[0][0], kv[0][1]))
        src = np.asarray([k[0] for k, _ in items], dtype=np.int64)
        dst = np.asarray([k[1] for k, _ in items], dtype=np.int64)
        is_long = np.asarray([v for _, v in items], dtype=bool)

    return {
        "node_rc": node_rc,
        "node_map": node_map,
        "skel_mask": skel_mask,
        "src": src,
        "dst": dst,
        "is_long": is_long,
    }


def lift_nodes_to_grid(
    r_nodes: np.ndarray,
    node_map: np.ndarray,
    skel_mask: np.ndarray,
    lift_sigma: float,
) -> np.ndarray:
    """Lift node reliabilities to full grid via nearest skeleton node + distance decay."""
    il, xl = node_map.shape
    r_nodes = np.asarray(r_nodes, dtype=np.float32).reshape(-1)
    skel_mask = np.asarray(skel_mask, dtype=bool)

    if r_nodes.size == 0 or (not np.any(skel_mask)):
        return np.zeros((il, xl), dtype=np.float32)

    dist, inds = distance_transform_edt(~skel_mask, return_indices=True)
    nn_r = inds[0]
    nn_c = inds[1]
    nn_idx = node_map[nn_r, nn_c]

    out = np.zeros((il, xl), dtype=np.float32)
    valid = nn_idx >= 0
    if float(lift_sigma) > 0:
        decay = np.exp(-dist / float(lift_sigma)).astype(np.float32)
    else:
        decay = np.ones_like(dist, dtype=np.float32)
    out[valid] = r_nodes[nn_idx[valid]] * decay[valid]
    return out.astype(np.float32)


def _compute_lift_mapping(node_map: np.ndarray, skel_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Precompute nearest skeleton-node index map and distance map for lifting."""
    skel_mask = np.asarray(skel_mask, dtype=bool)
    node_map = np.asarray(node_map, dtype=np.int32)
    il, xl = node_map.shape

    if not np.any(skel_mask):
        return np.full((il, xl), -1, dtype=np.int32), np.full((il, xl), np.inf, dtype=np.float32)

    dist, inds = distance_transform_edt(~skel_mask, return_indices=True)
    nn_idx = node_map[inds[0], inds[1]].astype(np.int32)
    return nn_idx, dist.astype(np.float32)


def lift_nodes_to_grid_cached(
    r_nodes: np.ndarray,
    lift_nearest_idx: np.ndarray,
    lift_dist: np.ndarray,
    lift_sigma: float,
) -> np.ndarray:
    """Lift node reliabilities with precomputed nearest-node and distance maps."""
    r_nodes = np.asarray(r_nodes, dtype=np.float32).reshape(-1)
    nn_idx = np.asarray(lift_nearest_idx, dtype=np.int32)
    dist = np.asarray(lift_dist, dtype=np.float32)
    il, xl = nn_idx.shape

    if r_nodes.size == 0:
        return np.zeros((il, xl), dtype=np.float32)

    out = np.zeros((il, xl), dtype=np.float32)
    valid = nn_idx >= 0
    if float(lift_sigma) > 0:
        decay = np.exp(-dist / float(lift_sigma)).astype(np.float32)
    else:
        decay = np.ones_like(dist, dtype=np.float32)
    out[valid] = r_nodes[nn_idx[valid]] * decay[valid]
    return out.astype(np.float32)


def build_cache_for_slice(
    pch: torch.Tensor | np.ndarray,
    conf: torch.Tensor | np.ndarray | None,
    v: torch.Tensor | np.ndarray,
    il: int,
    xl: int,
    p_thresh: float,
    min_nodes: int,
    long_edges: bool,
    long_max_step: int,
    long_step: int,
    long_cos_thresh: float,
) -> AGPESliceCache | None:
    """Build full skeleton-graph cache for one depth slice."""
    graph = build_skeleton_graph(
        pch=pch,
        conf=conf,
        v=v,
        il=il,
        xl=xl,
        p_thresh=p_thresh,
        min_nodes=min_nodes,
        long_edges=long_edges,
        long_max_step=long_max_step,
        long_step=long_step,
        long_cos_thresh=long_cos_thresh,
    )
    if graph is None:
        return None

    node_rc = graph["node_rc"]
    src = graph["src"]
    dst = graph["dst"]
    is_long = graph["is_long"]
    if node_rc.size == 0 or src.size == 0:
        return None

    node_map = graph["node_map"].astype(np.int32, copy=False)
    skel_mask = graph["skel_mask"].astype(bool, copy=False)
    lift_nearest_idx, lift_dist = _compute_lift_mapping(node_map=node_map, skel_mask=skel_mask)

    pch_np = _to_numpy_2d(pch, il, xl, "pch").astype(np.float32)
    if conf is None:
        mask = pch_np >= float(p_thresh)
    else:
        conf_np = _to_numpy_2d(conf, il, xl, "conf").astype(np.float32)
        mask = (pch_np >= float(p_thresh)) & (conf_np > 0.0)

    return AGPESliceCache(
        node_rc=node_rc.astype(np.int32, copy=False),
        node_map=node_map,
        skel_mask=skel_mask,
        src=src.astype(np.int64, copy=False),
        dst=dst.astype(np.int64, copy=False),
        is_long=is_long.astype(bool, copy=False),
        lift_nearest_idx=lift_nearest_idx,
        lift_dist=lift_dist,
        mask=mask.astype(bool, copy=False),
        pch_prev=pch_np.copy(),
        n_nodes=int(node_rc.shape[0]),
        n_edges=int(src.shape[0]),
        n_long_edges=int(is_long.sum()),
    )


def maybe_rebuild_cache(
    cache_slice: AGPESliceCache | None,
    new_mask: np.ndarray,
    topo_change_metric: float,
    threshold: float,
    force: bool = False,
) -> bool:
    """Decide whether skeleton topology should be rebuilt."""
    if cache_slice is None:
        return True
    if force:
        return True
    if cache_slice.mask.shape != np.asarray(new_mask).shape:
        return True
    return float(topo_change_metric) > float(threshold)


@torch.no_grad()
def update_edge_weight_only(
    cache_slice: AGPESliceCache,
    pch: torch.Tensor,
    conf: torch.Tensor | None,
    v: torch.Tensor,
    strength: torch.Tensor | None,
    feat: torch.Tensor | None,
    damp: torch.Tensor | None,
    gamma: float,
    tau: float,
    kappa: float,
    use_tensor_strength: bool,
    tensor_strength_power: float,
    agpe_edge_tau_p: float,
    agpe_long_edges: bool,
    agpe_long_weight: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update edge weights on a fixed skeleton topology."""
    device = pch.device
    dtype = pch.dtype

    src = torch.from_numpy(cache_slice.src).to(device=device, dtype=torch.long)
    dst = torch.from_numpy(cache_slice.dst).to(device=device, dtype=torch.long)
    is_long = torch.from_numpy(cache_slice.is_long.astype(np.uint8)).to(device=device, dtype=torch.bool)

    rc_t = torch.from_numpy(cache_slice.node_rc).to(device=device, dtype=torch.long)
    rr = rc_t[:, 0]
    cc = rc_t[:, 1]

    pch_nodes = pch[rr, cc]
    conf_nodes = torch.ones_like(pch_nodes) if conf is None else conf[rr, cc].clamp(0.0, 1.0)
    damp_nodes = None if damp is None else damp[rr, cc].clamp(0.0, 1.0)

    if v.dim() != 3:
        raise ValueError(f"v must be 3D (2,IL,XL) or (IL,XL,2), got shape={tuple(v.shape)}")
    if v.shape[0] == 2:
        v_nodes = v.permute(1, 2, 0)[rr, cc]
    elif v.shape[-1] == 2:
        v_nodes = v[rr, cc]
    else:
        raise ValueError(f"Unsupported v shape={tuple(v.shape)}")
    v_norm = torch.sqrt((v_nodes * v_nodes).sum(dim=1, keepdim=True) + eps)
    v_nodes = v_nodes / v_norm

    if use_tensor_strength and (strength is not None):
        if strength.dim() == 2:
            str_nodes = strength[rr, cc]
        elif strength.dim() == 3 and strength.shape[0] == 1:
            str_nodes = strength[0, rr, cc]
        else:
            raise ValueError(f"Unsupported strength shape={tuple(strength.shape)}")
        strength_nodes = str_nodes.clamp(0.0, 1.0).pow(float(tensor_strength_power))
    else:
        strength_nodes = torch.ones_like(pch_nodes)

    feat_nodes = None
    if feat is not None:
        if feat.dim() != 3:
            raise ValueError(f"feat must be 3D, got shape={tuple(feat.shape)}")
        if feat.shape[0] == pch.shape[0] and feat.shape[1] == pch.shape[1]:
            feat_nodes = feat[rr, cc]
        else:
            feat_nodes = feat.permute(1, 2, 0)[rr, cc]

    src_r = rr[src]
    src_c = cc[src]
    dst_r = rr[dst]
    dst_c = cc[dst]

    dx = (src_c - dst_c).to(dtype)
    dy = (src_r - dst_r).to(dtype)
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

    edge_tau_p = max(float(agpe_edge_tau_p), eps)
    p_edge = torch.exp(-torch.abs(pch_nodes[dst] - pch_nodes[src]) / edge_tau_p)
    p_edge = p_edge * (conf_nodes[dst] * conf_nodes[src])

    w = g * a * s * p_edge
    if bool(agpe_long_edges):
        w = torch.where(is_long, w * float(agpe_long_weight), w)
    w = w.clamp(min=eps)

    return src, dst, w
