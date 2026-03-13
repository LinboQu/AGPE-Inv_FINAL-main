from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def build_default_config(repo_root: Path | None = None) -> dict:
    repo_root = Path.cwd() if repo_root is None else Path(repo_root)
    if not (repo_root / "utils").exists():
        if (repo_root.parent / "utils").exists():
            repo_root = repo_root.parent
        else:
            raise FileNotFoundError("Cannot locate repo root from current working directory.")

    data_dir = repo_root / "data" / "Stanford_VI"
    output_dir = repo_root / "visualize_3d" / "_outputs" / "farp_similarity"
    output_dir.mkdir(parents=True, exist_ok=True)

    return {
        "repo_root": repo_root,
        "data_dir": data_dir,
        "output_dir": output_dir,
        "seismic_path": data_dir / "synth_40HZ.npy",
        "facies_path": data_dir / "Facies.npy",
        "ai_path": data_dir / "AI.npy",
        "wells_csv": data_dir / "selected_wells_20_seed2026.csv",
        "facies_prob_path": None,
        "p_channel_path": None,
        "conf_path": None,
        "facies_prior_path": None,
        "r3d_path": None,
        "channel_id": 2,
        "alpha_prior": 1.0,
        "conf_thresh": 0.75,
        "neutral_p": 0.5,
        "focus_depth": 100,
        "focus_well_index": 0,
        "anchor_mode": "well",
        "manual_inline": 57,
        "manual_xline": 68,
        "backend": "skeleton_graph",
        "steps_r": 25,
        "eta": 0.6,
        "gamma": 8.0,
        "tau": 0.6,
        "kappa": 4.0,
        "sigma_st": 1.2,
        "well_soft_alpha": 0.20,
        "use_tensor_strength": False,
        "tensor_strength_power": 1.0,
        "agpe_skel_p_thresh": 0.55,
        "agpe_skel_min_nodes": 30,
        "agpe_skel_snap_radius": 5,
        "agpe_long_edges": True,
        "agpe_long_max_step": 6,
        "agpe_long_step": 2,
        "agpe_long_cos_thresh": 0.70,
        "agpe_long_weight": 0.35,
        "agpe_edge_tau_p": 0.25,
        "agpe_lift_sigma": 2.2,
        "local_radius": 10,
        "quiver_step": 10,
        "sim_clip": (1.0, 99.0),
        "fig_dpi": 180,
        "feature_scatter_max_points": 30000,
        "feature_scatter_seed": 2026,
        "graph_edge_max_plot": 7000,
        "diffusion_capture_steps": (0, 1, 5, 12, 25),
        "stack_depth_offsets": (-2, -1, 0, 1, 2),
        "save_outputs": False,
        "compute_slice_r": True,
        "compute_full_r": False,
        "full_r_backend": "skeleton_graph",
    }


def _ensure_repo_import(repo_root: Path) -> None:
    root_str = str(repo_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for visualize_farp_similarity. Use a Python/Jupyter environment with torch installed."
        ) from exc
    return torch


def _load_cube(path: Path, name: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"{name} must be 3D, got shape={arr.shape}")
    return arr.astype(np.float32, copy=False)


def _load_optional_array(path: Path | None, name: str) -> np.ndarray | None:
    if path is None:
        return None
    arr = np.load(Path(path))
    if arr.ndim not in (3, 4):
        raise ValueError(f"{name} must be 3D or 4D, got shape={arr.shape}")
    return np.asarray(arr)


def _load_selected_wells(csv_path: Path, il_max: int, xl_max: int) -> list[dict]:
    wells = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            inline = int(float(row["INLINE"]))
            xline = int(float(row["XLINE"]))
            if not (0 <= inline < il_max and 0 <= xline < xl_max):
                continue
            wells.append(
                {
                    "wellname": row.get("WELLNAME", f"IL{inline}_XL{xline}"),
                    "inline": inline,
                    "xline": xline,
                    "trace_index": inline * xl_max + xline,
                }
            )
    if not wells:
        raise ValueError(f"No valid wells found in {csv_path}")
    return wells


def _build_feature_cube(seismic_hilxl: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    amp = np.asarray(seismic_hilxl, dtype=np.float32)
    gx = np.pad(amp[:, :, 1:] - amp[:, :, :-1], ((0, 0), (0, 0), (0, 1)), mode="constant")
    gy = np.pad(amp[:, 1:, :] - amp[:, :-1, :], ((0, 0), (0, 1), (0, 0)), mode="constant")
    gmag = np.sqrt(gx * gx + gy * gy).astype(np.float32, copy=False)
    feat = np.stack([amp, gmag], axis=1).astype(np.float32, copy=False)
    return amp, gx, gy, gmag, feat


def _ensure_probs_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    if x.size == 0:
        return x
    x = np.asarray(x, dtype=np.float32)
    if (float(x.min()) < -1e-3) or (float(x.max()) > 1.0 + 1e-3):
        x = x - np.max(x, axis=axis, keepdims=True)
        ex = np.exp(x, dtype=np.float32)
        return ex / (np.sum(ex, axis=axis, keepdims=True) + 1e-8)
    return x


def _resolve_cube_shape(arr: np.ndarray, h: int, il: int, xl: int, name: str) -> np.ndarray:
    if arr.shape == (h, il, xl):
        return np.asarray(arr)
    if arr.shape == (il, xl, h):
        return np.transpose(arr, (2, 0, 1))
    raise ValueError(f"{name} must have shape {(h, il, xl)} or {(il, xl, h)}, got {arr.shape}")


def _resolve_prob_shape(arr: np.ndarray, h: int, il: int, xl: int, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 4:
        raise ValueError(f"{name} must be 4D, got shape={arr.shape}")
    if arr.shape[:3] == (h, il, xl):
        return arr
    if arr.shape[1:] == (h, il, xl):
        return np.transpose(arr, (1, 2, 3, 0))
    if arr.shape[:3] == (il, xl, h):
        return np.transpose(arr, (2, 0, 1, 3))
    raise ValueError(
        f"{name} must look like [H,IL,XL,K] or [K,H,IL,XL], got {arr.shape}"
    )


def _resolve_channel_components(
    *,
    facies_3d: np.ndarray | None,
    facies_prob_3d: np.ndarray | None,
    p_channel_3d: np.ndarray | None,
    conf_3d: np.ndarray | None,
    facies_prior_3d: np.ndarray | None,
    channel_id: int,
    alpha_prior: float,
    conf_thresh: float,
    neutral_p: float,
) -> dict:
    if facies_prior_3d is not None:
        p_prior = (facies_prior_3d == int(channel_id)).astype(np.float32)
        prior_source = "facies_prior_3d"
    elif facies_3d is not None:
        p_prior = (facies_3d == int(channel_id)).astype(np.float32)
        prior_source = "facies_3d"
    else:
        p_prior = None
        prior_source = "none"

    if p_channel_3d is not None:
        p_pred = np.asarray(p_channel_3d, dtype=np.float32)
        conf = np.asarray(conf_3d, dtype=np.float32) if conf_3d is not None else np.ones_like(p_pred, dtype=np.float32)
        pred_source = "p_channel_3d"
    elif facies_prob_3d is not None:
        fp = _ensure_probs_np(np.asarray(facies_prob_3d, dtype=np.float32), axis=-1)
        if int(channel_id) < 0 or int(channel_id) >= fp.shape[-1]:
            raise ValueError(f"channel_id={channel_id} out of range for facies_prob_3d with K={fp.shape[-1]}")
        p_pred = fp[..., int(channel_id)].astype(np.float32, copy=False)
        conf = fp.max(axis=-1).astype(np.float32, copy=False)
        pred_source = "facies_prob_3d"
    elif facies_3d is not None:
        p_pred = (facies_3d == int(channel_id)).astype(np.float32)
        conf = np.ones_like(p_pred, dtype=np.float32)
        pred_source = "facies_3d(hard_fallback)"
    else:
        raise ValueError("Need one of: p_channel_3d / facies_prob_3d / facies_3d.")

    if p_prior is not None:
        p_mix = float(alpha_prior) * p_prior + (1.0 - float(alpha_prior)) * p_pred
        p_fallback = float(alpha_prior) * p_prior + (1.0 - float(alpha_prior)) * float(neutral_p)
    else:
        p_mix = p_pred
        p_fallback = np.full_like(p_pred, float(neutral_p), dtype=np.float32)

    conf = np.asarray(conf, dtype=np.float32)
    conf_gate = conf >= float(conf_thresh)
    p_channel = np.where(conf_gate, p_mix, p_fallback).clip(0.0, 1.0).astype(np.float32)

    return {
        "p_prior_3d": None if p_prior is None else p_prior.astype(np.float32, copy=False),
        "p_pred_3d": p_pred.astype(np.float32, copy=False),
        "conf_3d": conf.astype(np.float32, copy=False),
        "p_mix_3d": p_mix.astype(np.float32, copy=False),
        "p_fallback_3d": np.asarray(p_fallback, dtype=np.float32),
        "p_channel_3d": p_channel,
        "conf_gate_3d": conf_gate.astype(np.float32),
        "prior_source": prior_source,
        "pred_source": pred_source,
    }


def _percentile_limits(arr: np.ndarray, pct=(1.0, 99.0), symmetric: bool = False) -> tuple[float, float]:
    valid = np.asarray(arr)[np.isfinite(arr)]
    if valid.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(valid, pct[0]))
    hi = float(np.percentile(valid, pct[1]))
    if symmetric:
        m = max(abs(lo), abs(hi))
        lo, hi = -m, m
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def _maybe_savefig(fig, cfg: dict, stem: str) -> None:
    if cfg.get("save_outputs", False):
        fig.savefig(Path(cfg["output_dir"]) / f"{stem}.png", dpi=int(cfg["fig_dpi"]), bbox_inches="tight")


def _finalize_figure(fig, *, top: float = 0.92, left: float = 0.06, right: float = 0.98, bottom: float = 0.08, wspace: float = 0.28, hspace: float = 0.28) -> None:
    # Avoid tight_layout() here: this environment can crash inside matplotlib/numpy during automatic layout.
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)


def _build_well_mask(il: int, xl: int, wells: list[dict]) -> np.ndarray:
    mask = np.zeros((il, xl), dtype=np.float32)
    for w in wells:
        mask[w["inline"], w["xline"]] = 1.0
    return mask


def _choose_anchor(cfg: dict, wells: list[dict], gmag_slice: np.ndarray) -> dict:
    mode = str(cfg["anchor_mode"]).strip().lower()
    if mode == "well":
        idx = int(np.clip(cfg["focus_well_index"], 0, len(wells) - 1))
        w = wells[idx]
        return {"mode": "well", "well_index": idx, "wellname": w["wellname"], "inline": w["inline"], "xline": w["xline"]}
    if mode == "max_gmag":
        rr, cc = np.unravel_index(int(np.argmax(gmag_slice)), gmag_slice.shape)
        return {"mode": "max_gmag", "well_index": None, "wellname": "max_gmag", "inline": int(rr), "xline": int(cc)}
    if mode == "manual":
        rr = int(np.clip(cfg["manual_inline"], 0, gmag_slice.shape[0] - 1))
        cc = int(np.clip(cfg["manual_xline"], 0, gmag_slice.shape[1] - 1))
        return {"mode": "manual", "well_index": None, "wellname": "manual", "inline": rr, "xline": cc}
    raise ValueError(f"Unknown anchor_mode={cfg['anchor_mode']}")


def _resolve_capture_steps(cfg: dict) -> list[int]:
    steps_r = int(cfg["steps_r"])
    raw_steps = cfg.get("diffusion_capture_steps", (0, 1, steps_r // 2, steps_r))
    capture = {0, steps_r}
    for step in raw_steps:
        step_i = int(step)
        if 0 <= step_i <= steps_r:
            capture.add(step_i)
    return sorted(capture)


def _node_values_to_grid(
    node_rc: np.ndarray,
    values: np.ndarray,
    shape: tuple[int, int],
    *,
    fill_value: float = np.nan,
) -> np.ndarray:
    out = np.full(shape, fill_value, dtype=np.float32)
    if node_rc.size == 0 or values.size == 0:
        return out
    rows = node_rc[:, 0].astype(np.int64, copy=False)
    cols = node_rc[:, 1].astype(np.int64, copy=False)
    out[rows, cols] = np.asarray(values, dtype=np.float32).reshape(-1)
    return out


def _edge_mean_to_nodes(dst_idx: np.ndarray, values: np.ndarray, n_nodes: int) -> np.ndarray:
    out = np.full((n_nodes,), np.nan, dtype=np.float32)
    if n_nodes <= 0 or dst_idx.size == 0 or values.size == 0:
        return out
    sums = np.zeros((n_nodes,), dtype=np.float64)
    cnt = np.zeros((n_nodes,), dtype=np.float64)
    np.add.at(sums, dst_idx.astype(np.int64, copy=False), np.asarray(values, dtype=np.float64))
    np.add.at(cnt, dst_idx.astype(np.int64, copy=False), 1.0)
    valid = cnt > 0.0
    out[valid] = (sums[valid] / cnt[valid]).astype(np.float32)
    return out


def _edge_sum_to_nodes(dst_idx: np.ndarray, values: np.ndarray, n_nodes: int) -> np.ndarray:
    out = np.zeros((n_nodes,), dtype=np.float32)
    if n_nodes <= 0 or dst_idx.size == 0 or values.size == 0:
        return out
    np.add.at(out, dst_idx.astype(np.int64, copy=False), np.asarray(values, dtype=np.float32))
    return out


def _compute_degree_maps(node_rc: np.ndarray, src_idx: np.ndarray, dst_idx: np.ndarray, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    n_nodes = int(node_rc.shape[0])
    in_deg = np.zeros((n_nodes,), dtype=np.float32)
    out_deg = np.zeros((n_nodes,), dtype=np.float32)
    if dst_idx.size > 0:
        np.add.at(in_deg, dst_idx.astype(np.int64, copy=False), 1.0)
    if src_idx.size > 0:
        np.add.at(out_deg, src_idx.astype(np.int64, copy=False), 1.0)
    total_deg = in_deg + out_deg
    return {
        "in": _node_values_to_grid(node_rc, in_deg, shape, fill_value=np.nan),
        "out": _node_values_to_grid(node_rc, out_deg, shape, fill_value=np.nan),
        "total": _node_values_to_grid(node_rc, total_deg, shape, fill_value=np.nan),
    }


def _snap_well_seeds_with_mapping(
    well_mask_2d: np.ndarray,
    node_map: np.ndarray,
    snap_radius: int,
) -> dict:
    wm_np = np.asarray(well_mask_2d, dtype=np.float32)
    node_map = np.asarray(node_map, dtype=np.int32)
    il, xl = node_map.shape

    raw_seed_rc = np.argwhere(wm_np > 0.0).astype(np.int32)
    raw_seed_val = np.asarray([wm_np[int(r), int(c)] for r, c in raw_seed_rc], dtype=np.float32)
    seed_nodes: dict[int, float] = {}
    snapped_records: list[dict] = []
    failed_records: list[dict] = []
    snap_ok = 0
    r_snap = max(int(snap_radius), 0)

    for r0, c0 in raw_seed_rc:
        r0_i = int(r0)
        c0_i = int(c0)
        wv = float(wm_np[r0_i, c0_i])
        if wv <= 0.0:
            continue
        idx = int(node_map[r0_i, c0_i])
        snap_rc: tuple[int, int] | None = None
        if idx >= 0:
            snap_rc = (r0_i, c0_i)
        elif r_snap > 0:
            rr0 = max(0, r0_i - r_snap)
            rr1 = min(il, r0_i + r_snap + 1)
            cc0 = max(0, c0_i - r_snap)
            cc1 = min(xl, c0_i + r_snap + 1)
            sub = node_map[rr0:rr1, cc0:cc1]
            cand = np.argwhere(sub >= 0)
            if cand.size > 0:
                cand[:, 0] += rr0
                cand[:, 1] += cc0
                d2 = (cand[:, 0] - r0_i) ** 2 + (cand[:, 1] - c0_i) ** 2
                best = cand[int(np.argmin(d2))]
                idx = int(node_map[int(best[0]), int(best[1])])
                if idx >= 0:
                    snap_rc = (int(best[0]), int(best[1]))
        if idx >= 0 and snap_rc is not None:
            seed_nodes[idx] = max(seed_nodes.get(idx, 0.0), wv)
            snapped_records.append(
                {
                    "raw_rc": (r0_i, c0_i),
                    "snap_rc": snap_rc,
                    "node_idx": idx,
                    "value": wv,
                }
            )
            snap_ok += 1
        else:
            failed_records.append({"raw_rc": (r0_i, c0_i), "value": wv})

    if seed_nodes:
        seed_idx = np.asarray(sorted(seed_nodes.keys()), dtype=np.int64)
        seed_val = np.asarray([seed_nodes[int(k)] for k in seed_idx], dtype=np.float32)
        seed_val = np.clip(seed_val, 0.0, 1.0)
    else:
        seed_idx = np.empty((0,), dtype=np.int64)
        seed_val = np.empty((0,), dtype=np.float32)

    return {
        "raw_seed_rc": raw_seed_rc,
        "raw_seed_val": raw_seed_val,
        "snapped_records": snapped_records,
        "failed_records": failed_records,
        "seed_idx": seed_idx,
        "seed_val": seed_val,
        "seed_total": int(raw_seed_rc.shape[0]),
        "snap_ok": int(snap_ok),
    }


def _graph_diffuse_with_history(
    r0_flat,
    well_mask_flat,
    src,
    dst,
    w,
    *,
    steps: int,
    eta: float,
    well_soft_alpha: float,
    eps: float,
    capture_steps: list[int],
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    alpha = float(well_soft_alpha)
    r = r0_flat.clone()
    history: dict[int, np.ndarray] = {}
    capture = set(int(x) for x in capture_steps)
    if 0 in capture:
        history[0] = r.detach().cpu().numpy().astype(np.float32)

    for step in range(1, int(steps) + 1):
        acc = r.new_zeros(r.shape)
        den = r.new_zeros(r.shape)
        acc.index_add_(0, dst, w * r[src])
        den.index_add_(0, dst, w)
        r_prop = acc / (den + float(eps))
        r = (1.0 - float(eta)) * r + float(eta) * r_prop
        r = r.maximum(well_mask_flat)
        if step in capture:
            history[step] = r.detach().cpu().numpy().astype(np.float32)

    if 0.0 < alpha < 1.0 - 1e-8:
        r = ((1.0 - alpha) * r + alpha * well_mask_flat).clamp(0.0, 1.0)
    if int(steps) in capture:
        history[int(steps)] = r.detach().cpu().numpy().astype(np.float32)
    return r.detach().cpu().numpy().astype(np.float32), history


def _collect_skeleton_slice_debug(
    *,
    cfg: dict,
    torch,
    pch_slice: np.ndarray,
    conf_slice: np.ndarray,
    feat_slice: np.ndarray,
    v_slice: np.ndarray,
    strength_slice: np.ndarray,
    well_mask_2d: np.ndarray,
    build_cache_for_slice,
    compute_edge_weight_terms,
    lift_nodes_to_grid_cached,
    apply_well_constraint,
) -> dict:
    pch_t = torch.from_numpy(np.asarray(pch_slice, dtype=np.float32))
    conf_t = torch.from_numpy(np.asarray(conf_slice, dtype=np.float32))
    feat_t = torch.from_numpy(np.asarray(feat_slice, dtype=np.float32))
    v_t = torch.from_numpy(np.asarray(v_slice, dtype=np.float32))
    strength_t = torch.from_numpy(np.asarray(strength_slice, dtype=np.float32))
    wm_t = torch.from_numpy(np.asarray(well_mask_2d, dtype=np.float32))

    mask_np = (np.asarray(pch_slice, dtype=np.float32) >= float(cfg["agpe_skel_p_thresh"])) & (
        np.asarray(conf_slice, dtype=np.float32) > 0.0
    )
    out = {
        "enabled": True,
        "mask": mask_np.astype(bool, copy=False),
        "cache_available": False,
        "fallback_reason": None,
        "capture_steps": _resolve_capture_steps(cfg),
        "seed_total": int(np.count_nonzero(well_mask_2d > 0.0)),
        "snap_ok": 0,
    }

    cache_slice = build_cache_for_slice(
        pch=pch_t,
        conf=conf_t,
        v=v_t,
        il=int(pch_slice.shape[0]),
        xl=int(pch_slice.shape[1]),
        p_thresh=float(cfg["agpe_skel_p_thresh"]),
        min_nodes=int(cfg["agpe_skel_min_nodes"]),
        long_edges=bool(cfg["agpe_long_edges"]),
        long_max_step=int(cfg["agpe_long_max_step"]),
        long_step=int(cfg["agpe_long_step"]),
        long_cos_thresh=float(cfg["agpe_long_cos_thresh"]),
    )

    if cache_slice is None:
        out["fallback_reason"] = "cache_unavailable"
        return out

    out["cache_available"] = True
    out["cache_slice"] = cache_slice
    out["skel_mask"] = cache_slice.skel_mask.astype(bool, copy=False)
    out["node_rc"] = cache_slice.node_rc.astype(np.int32, copy=False)
    out["node_map"] = cache_slice.node_map.astype(np.int32, copy=False)
    out["src"] = cache_slice.src.astype(np.int64, copy=False)
    out["dst"] = cache_slice.dst.astype(np.int64, copy=False)
    out["is_long"] = cache_slice.is_long.astype(bool, copy=False)
    out["n_nodes"] = int(cache_slice.n_nodes)
    out["n_edges"] = int(cache_slice.n_edges)
    out["n_long_edges"] = int(cache_slice.n_long_edges)
    out["degree_maps"] = _compute_degree_maps(cache_slice.node_rc, cache_slice.src, cache_slice.dst, pch_slice.shape)

    snap = _snap_well_seeds_with_mapping(well_mask_2d=well_mask_2d, node_map=cache_slice.node_map, snap_radius=int(cfg["agpe_skel_snap_radius"]))
    out.update(snap)
    out["snap_ok"] = int(snap["snap_ok"])
    out["snapped_seed_rc"] = (
        cache_slice.node_rc[snap["seed_idx"]] if snap["seed_idx"].size > 0 else np.empty((0, 2), dtype=np.int32)
    )

    terms_t = compute_edge_weight_terms(
        cache_slice=cache_slice,
        pch=pch_t,
        conf=conf_t,
        v=v_t,
        strength=strength_t,
        feat=feat_t,
        damp=None,
        gamma=float(cfg["gamma"]),
        tau=float(cfg["tau"]),
        kappa=float(cfg["kappa"]),
        use_tensor_strength=bool(cfg["use_tensor_strength"]),
        tensor_strength_power=float(cfg["tensor_strength_power"]),
        agpe_edge_tau_p=float(cfg["agpe_edge_tau_p"]),
        agpe_long_edges=bool(cfg["agpe_long_edges"]),
        agpe_long_weight=float(cfg["agpe_long_weight"]),
        eps=1e-8,
    )
    edge_terms = {key: value.detach().cpu().numpy().astype(np.float32) for key, value in terms_t.items() if key not in {"src", "dst", "is_long", "rr", "cc"}}
    out["edge_terms"] = edge_terms
    out["src_plot"] = terms_t["src"].detach().cpu().numpy().astype(np.int64)
    out["dst_plot"] = terms_t["dst"].detach().cpu().numpy().astype(np.int64)
    out["is_long_plot"] = terms_t["is_long"].detach().cpu().numpy().astype(bool)
    out["edge_node_means"] = {
        name: _node_values_to_grid(
            cache_slice.node_rc,
            _edge_mean_to_nodes(out["dst_plot"], values, int(cache_slice.n_nodes)),
            pch_slice.shape,
            fill_value=np.nan,
        )
        for name, values in edge_terms.items()
        if values.ndim == 1 and values.shape[0] == out["dst_plot"].shape[0]
    }
    local_count_nodes = _edge_sum_to_nodes(out["dst_plot"][~out["is_long_plot"]], np.ones((np.count_nonzero(~out["is_long_plot"]),), dtype=np.float32), int(cache_slice.n_nodes))
    long_count_nodes = _edge_sum_to_nodes(out["dst_plot"][out["is_long_plot"]], np.ones((np.count_nonzero(out["is_long_plot"]),), dtype=np.float32), int(cache_slice.n_nodes))
    out["edge_count_nodes"] = {
        "local_in": local_count_nodes,
        "long_in": long_count_nodes,
    }
    out["edge_count_maps"] = {
        "local_in": _node_values_to_grid(cache_slice.node_rc, local_count_nodes, pch_slice.shape, fill_value=np.nan),
        "long_in": _node_values_to_grid(cache_slice.node_rc, long_count_nodes, pch_slice.shape, fill_value=np.nan),
    }

    node_n = int(cache_slice.n_nodes)
    r0 = torch.zeros((node_n,), dtype=torch.float32)
    wm_node = torch.zeros_like(r0)
    if snap["seed_idx"].size > 0:
        seed_idx_t = torch.from_numpy(snap["seed_idx"]).to(dtype=torch.long)
        seed_val_t = torch.from_numpy(snap["seed_val"]).to(dtype=torch.float32).clamp(0.0, 1.0)
        r0[seed_idx_t] = seed_val_t
        wm_node[seed_idx_t] = seed_val_t

    out["r0_nodes"] = r0.detach().cpu().numpy().astype(np.float32)
    out["wm_node"] = wm_node.detach().cpu().numpy().astype(np.float32)
    out["r0_grid"] = _node_values_to_grid(cache_slice.node_rc, out["r0_nodes"], pch_slice.shape, fill_value=np.nan)
    out["r_node_history"] = {}
    out["r_node_grids"] = {}
    out["r_nodes_final"] = np.zeros((node_n,), dtype=np.float32)
    out["r_nodes_grid_final"] = _node_values_to_grid(cache_slice.node_rc, out["r_nodes_final"], pch_slice.shape, fill_value=np.nan)

    if snap["seed_idx"].size == 0:
        out["fallback_reason"] = "no_snapped_seed"
        return out

    final_nodes_np, history_np = _graph_diffuse_with_history(
        r0_flat=r0,
        well_mask_flat=wm_node,
        src=terms_t["src"],
        dst=terms_t["dst"],
        w=terms_t["w"],
        steps=int(cfg["steps_r"]),
        eta=float(cfg["eta"]),
        well_soft_alpha=float(cfg["well_soft_alpha"]),
        eps=1e-8,
        capture_steps=out["capture_steps"],
    )
    out["r_nodes_final"] = final_nodes_np
    out["r_node_history"] = history_np
    out["r_node_grids"] = {
        step: _node_values_to_grid(cache_slice.node_rc, values, pch_slice.shape, fill_value=np.nan)
        for step, values in history_np.items()
    }
    out["r_nodes_grid_final"] = _node_values_to_grid(cache_slice.node_rc, final_nodes_np, pch_slice.shape, fill_value=np.nan)

    r_grid_pre = lift_nodes_to_grid_cached(
        r_nodes=final_nodes_np,
        lift_nearest_idx=cache_slice.lift_nearest_idx,
        lift_dist=cache_slice.lift_dist,
        lift_sigma=float(cfg["agpe_lift_sigma"]),
    ).astype(np.float32, copy=False)
    r_grid_post = apply_well_constraint(
        torch.from_numpy(r_grid_pre.copy()),
        wm_t,
        float(cfg["well_soft_alpha"]),
    ).detach().cpu().numpy().astype(np.float32)
    if float(cfg["agpe_lift_sigma"]) > 0.0:
        lift_decay = np.exp(-cache_slice.lift_dist / float(cfg["agpe_lift_sigma"])).astype(np.float32)
    else:
        lift_decay = np.ones_like(cache_slice.lift_dist, dtype=np.float32)

    out["lift_nearest_idx"] = cache_slice.lift_nearest_idx.astype(np.int32, copy=False)
    out["lift_dist"] = cache_slice.lift_dist.astype(np.float32, copy=False)
    out["lift_decay"] = lift_decay
    out["r_grid_pre"] = r_grid_pre
    out["r_grid_post"] = r_grid_post
    out["r_grid_delta"] = (r_grid_post - r_grid_pre).astype(np.float32, copy=False)
    return out


def _crop(arr: np.ndarray, center_il: int, center_xl: int, radius: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    il0 = max(0, center_il - radius)
    il1 = min(arr.shape[0], center_il + radius + 1)
    xl0 = max(0, center_xl - radius)
    xl1 = min(arr.shape[1], center_xl + radius + 1)
    return arr[il0:il1, xl0:xl1], (il0, il1, xl0, xl1)


def prepare_bundle(cfg: dict) -> dict:
    repo_root = Path(cfg["repo_root"])
    _ensure_repo_import(repo_root)
    torch = _require_torch()

    from utils.agpe_graph import (
        build_cache_for_slice,
        compute_edge_weight_terms,
        compute_lattice_edge_weight,
        get_lattice_edges,
        lift_nodes_to_grid_cached,
    )
    from utils.reliability_aniso import (
        _apply_well_constraint,
        anisotropic_reliability_2d,
        graph_lattice_reliability_2d,
        structure_tensor_orientation_and_strength,
    )

    seismic_hilxl = _load_cube(Path(cfg["seismic_path"]), "seismic")
    facies_hilxl = _load_cube(Path(cfg["facies_path"]), "facies").astype(np.int64, copy=False)
    ai_hilxl = _load_cube(Path(cfg["ai_path"]), "AI")

    amp_hilxl, gx_hilxl, gy_hilxl, gmag_hilxl, feat_h2ilxl = _build_feature_cube(seismic_hilxl)
    h, il, xl = seismic_hilxl.shape
    facies_prob_3d = _load_optional_array(cfg.get("facies_prob_path"), "facies_prob_3d")
    p_channel_3d = _load_optional_array(cfg.get("p_channel_path"), "p_channel_3d")
    conf_3d = _load_optional_array(cfg.get("conf_path"), "conf_3d")
    facies_prior_3d = _load_optional_array(cfg.get("facies_prior_path"), "facies_prior_3d")
    if facies_prob_3d is not None:
        facies_prob_3d = _resolve_prob_shape(facies_prob_3d, h=h, il=il, xl=xl, name="facies_prob_3d").astype(np.float32, copy=False)
    if p_channel_3d is not None:
        p_channel_3d = _resolve_cube_shape(p_channel_3d, h=h, il=il, xl=xl, name="p_channel_3d").astype(np.float32, copy=False)
    if conf_3d is not None:
        conf_3d = _resolve_cube_shape(conf_3d, h=h, il=il, xl=xl, name="conf_3d").astype(np.float32, copy=False)
    if facies_prior_3d is not None:
        facies_prior_3d = _resolve_cube_shape(facies_prior_3d, h=h, il=il, xl=xl, name="facies_prior_3d").astype(np.int64, copy=False)

    channel_components = _resolve_channel_components(
        facies_3d=facies_hilxl,
        facies_prob_3d=facies_prob_3d,
        p_channel_3d=p_channel_3d,
        conf_3d=conf_3d,
        facies_prior_3d=facies_prior_3d,
        channel_id=int(cfg["channel_id"]),
        alpha_prior=float(cfg["alpha_prior"]),
        conf_thresh=float(cfg["conf_thresh"]),
        neutral_p=float(cfg["neutral_p"]),
    )

    wells = _load_selected_wells(Path(cfg["wells_csv"]), il_max=il, xl_max=xl)
    well_mask_2d = _build_well_mask(il=il, xl=xl, wells=wells)

    focus_depth = int(np.clip(cfg["focus_depth"], 0, h - 1))
    amp_slice = amp_hilxl[focus_depth]
    gx_slice = gx_hilxl[focus_depth]
    gy_slice = gy_hilxl[focus_depth]
    gmag_slice = gmag_hilxl[focus_depth]
    feat_slice = feat_h2ilxl[focus_depth]
    p_prior_slice = None if channel_components["p_prior_3d"] is None else channel_components["p_prior_3d"][focus_depth]
    p_pred_slice = channel_components["p_pred_3d"][focus_depth]
    conf_slice = channel_components["conf_3d"][focus_depth]
    p_mix_slice = channel_components["p_mix_3d"][focus_depth]
    p_fallback_slice = channel_components["p_fallback_3d"][focus_depth]
    pch_slice = channel_components["p_channel_3d"][focus_depth]
    conf_gate_slice = channel_components["conf_gate_3d"][focus_depth]

    anchor = _choose_anchor(cfg, wells, gmag_slice)
    anchor_il = int(anchor["inline"])
    anchor_xl = int(anchor["xline"])

    dist_map = np.sqrt((amp_slice - amp_slice[anchor_il, anchor_xl]) ** 2 + (gmag_slice - gmag_slice[anchor_il, anchor_xl]) ** 2).astype(np.float32)
    sim_map = np.exp(-dist_map / (float(cfg["tau"]) + 1e-8)).astype(np.float32)

    x = torch.from_numpy(pch_slice[None, None].astype(np.float32))
    v_t, strength_t = structure_tensor_orientation_and_strength(x, sigma=float(cfg["sigma_st"]))
    v_slice = v_t[0].detach().cpu().numpy()
    strength_slice = strength_t[0, 0].detach().cpu().numpy()
    skeleton_debug = _collect_skeleton_slice_debug(
        cfg=cfg,
        torch=torch,
        pch_slice=pch_slice,
        conf_slice=conf_slice,
        feat_slice=feat_slice,
        v_slice=v_slice,
        strength_slice=strength_slice,
        well_mask_2d=well_mask_2d,
        build_cache_for_slice=build_cache_for_slice,
        compute_edge_weight_terms=compute_edge_weight_terms,
        lift_nodes_to_grid_cached=lift_nodes_to_grid_cached,
        apply_well_constraint=_apply_well_constraint,
    )

    src, dst = get_lattice_edges(il=il, xl=xl, diag=True, device=torch.device("cpu"))
    v_flat = torch.from_numpy(np.moveaxis(v_slice, 0, -1).reshape(-1, 2).astype(np.float32))
    pch_flat = torch.from_numpy(pch_slice.reshape(-1).astype(np.float32))
    feat_flat = torch.from_numpy(np.moveaxis(feat_slice, 0, -1).reshape(-1, feat_slice.shape[0]).astype(np.float32))
    strength_flat = None
    if cfg.get("use_tensor_strength", False):
        strength_flat = torch.from_numpy(strength_slice.reshape(-1).astype(np.float32))
    w = compute_lattice_edge_weight(
        v_flat=v_flat,
        pch_flat=pch_flat,
        feat_flat=feat_flat,
        damp_flat=None,
        src=src,
        dst=dst,
        il=il,
        xl=xl,
        gamma=float(cfg["gamma"]),
        tau=float(cfg["tau"]),
        kappa=float(cfg["kappa"]),
        anis_strength_flat=strength_flat,
        strength_power=float(cfg["tensor_strength_power"]),
    )
    acc = torch.zeros(il * xl, dtype=torch.float32)
    deg = torch.zeros(il * xl, dtype=torch.float32)
    acc.index_add_(0, dst, w.detach().cpu())
    deg.index_add_(0, dst, torch.ones_like(w.detach().cpu()))
    incoming_weight_map = (acc / (deg + 1e-8)).reshape(il, xl).numpy()

    local_maps = {name: np.full((3, 3), np.nan, dtype=np.float32) for name in ("g", "a", "s", "w")}
    anchor_feat = feat_slice[:, anchor_il, anchor_xl].astype(np.float32)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = anchor_il + dr
            cc = anchor_xl + dc
            if not (0 <= rr < il and 0 <= cc < xl):
                continue
            g = 1.0 / (1.0 + np.exp(-float(cfg["gamma"]) * (float(pch_slice[rr, cc]) - 0.5)))
            dx = float(anchor_xl - cc)
            dy = float(anchor_il - rr)
            dnorm = np.sqrt(dx * dx + dy * dy + 1e-8)
            dx /= dnorm
            dy /= dnorm
            vx = float(v_slice[0, rr, cc])
            vy = float(v_slice[1, rr, cc])
            cos = dx * vx + dy * vy
            strength_val = 1.0
            if cfg.get("use_tensor_strength", False):
                strength_val = float(np.clip(strength_slice[rr, cc], 0.0, 1.0) ** float(cfg["tensor_strength_power"]))
            a = float(np.exp(float(cfg["kappa"]) * strength_val * (cos * cos)))
            dist = float(np.sqrt(np.sum((feat_slice[:, rr, cc] - anchor_feat) ** 2) + 1e-8))
            s = float(np.exp(-dist / (float(cfg["tau"]) + 1e-8)))
            local_maps["g"][dr + 1, dc + 1] = g
            local_maps["a"][dr + 1, dc + 1] = a
            local_maps["s"][dr + 1, dc + 1] = s
            local_maps["w"][dr + 1, dc + 1] = g * a * s

    r_slice = None
    if cfg.get("compute_slice_r", True):
        wm = torch.from_numpy(well_mask_2d[None, None].astype(np.float32))
        pc = torch.from_numpy(pch_slice[None, None].astype(np.float32))
        fk = torch.from_numpy(feat_slice[None].astype(np.float32))
        if str(cfg["backend"]) == "grid":
            r_out, _ = anisotropic_reliability_2d(
                wm,
                pc,
                fk,
                steps=int(cfg["steps_r"]),
                eta=float(cfg["eta"]),
                gamma=float(cfg["gamma"]),
                tau=float(cfg["tau"]),
                kappa=float(cfg["kappa"]),
                sigma_st=float(cfg["sigma_st"]),
                well_soft_alpha=float(cfg["well_soft_alpha"]),
            )
        elif str(cfg["backend"]) == "graph_lattice":
            r_out, _ = graph_lattice_reliability_2d(
                wm,
                pc,
                fk,
                steps=int(cfg["steps_r"]),
                eta=float(cfg["eta"]),
                gamma=float(cfg["gamma"]),
                tau=float(cfg["tau"]),
                kappa=float(cfg["kappa"]),
                sigma_st=float(cfg["sigma_st"]),
                well_soft_alpha=float(cfg["well_soft_alpha"]),
                use_tensor_strength=bool(cfg["use_tensor_strength"]),
                tensor_strength_power=float(cfg["tensor_strength_power"]),
            )
        elif str(cfg["backend"]) == "skeleton_graph":
            if skeleton_debug.get("cache_available", False) and skeleton_debug.get("fallback_reason") is None:
                r_out = torch.from_numpy(skeleton_debug["r_grid_post"][None, None].astype(np.float32))
            else:
                r_out, _ = graph_lattice_reliability_2d(
                    wm,
                    pc,
                    fk,
                    steps=int(cfg["steps_r"]),
                    eta=float(cfg["eta"]),
                    gamma=float(cfg["gamma"]),
                    tau=float(cfg["tau"]),
                    kappa=float(cfg["kappa"]),
                    sigma_st=float(cfg["sigma_st"]),
                    well_soft_alpha=float(cfg["well_soft_alpha"]),
                    use_tensor_strength=bool(cfg["use_tensor_strength"]),
                    tensor_strength_power=float(cfg["tensor_strength_power"]),
                )
        else:
            raise ValueError(f"Unsupported backend={cfg['backend']} for slice visualization")
        r_slice = r_out[0, 0].detach().cpu().numpy()

    amp_crop, crop_box = _crop(amp_slice, anchor_il, anchor_xl, int(cfg["local_radius"]))
    gmag_crop, _ = _crop(gmag_slice, anchor_il, anchor_xl, int(cfg["local_radius"]))
    sim_crop, _ = _crop(sim_map, anchor_il, anchor_xl, int(cfg["local_radius"]))
    w_crop, _ = _crop(incoming_weight_map, anchor_il, anchor_xl, int(cfg["local_radius"]))
    r_crop = None if r_slice is None else _crop(r_slice, anchor_il, anchor_xl, int(cfg["local_radius"]))[0]

    return {
        "cfg": cfg,
        "seismic_hilxl": seismic_hilxl,
        "facies_hilxl": facies_hilxl,
        "facies_prob_3d": facies_prob_3d,
        "p_channel_input_3d": p_channel_3d,
        "conf_input_3d": conf_3d,
        "facies_prior_3d": facies_prior_3d,
        "channel_components": channel_components,
        "ai_hilxl": ai_hilxl,
        "amp_hilxl": amp_hilxl,
        "gx_hilxl": gx_hilxl,
        "gy_hilxl": gy_hilxl,
        "gmag_hilxl": gmag_hilxl,
        "wells": wells,
        "well_mask_2d": well_mask_2d,
        "shape": (h, il, xl),
        "focus_depth": focus_depth,
        "amp_slice": amp_slice,
        "gx_slice": gx_slice,
        "gy_slice": gy_slice,
        "gmag_slice": gmag_slice,
        "feat_slice": feat_slice,
        "p_prior_slice": p_prior_slice,
        "p_pred_slice": p_pred_slice,
        "conf_slice": conf_slice,
        "p_mix_slice": p_mix_slice,
        "p_fallback_slice": p_fallback_slice,
        "pch_slice": pch_slice,
        "conf_gate_slice": conf_gate_slice,
        "anchor": anchor,
        "anchor_il": anchor_il,
        "anchor_xl": anchor_xl,
        "dist_map": dist_map,
        "sim_map": sim_map,
        "v_slice": v_slice,
        "strength_slice": strength_slice,
        "incoming_weight_map": incoming_weight_map,
        "local_maps": local_maps,
        "skeleton_debug": skeleton_debug,
        "r_slice": r_slice,
        "amp_crop": amp_crop,
        "gmag_crop": gmag_crop,
        "sim_crop": sim_crop,
        "w_crop": w_crop,
        "r_crop": r_crop,
        "crop_box": crop_box,
    }


def print_summary(bundle: dict) -> None:
    h, il, xl = bundle["shape"]
    print("cube shape H,IL,XL:", (h, il, xl))
    print("n_wells:", len(bundle["wells"]))
    print("focus_depth:", bundle["focus_depth"])
    print("anchor:", bundle["anchor"])
    print("p_prior source:", bundle["channel_components"]["prior_source"])
    print("p_pred source:", bundle["channel_components"]["pred_source"])
    ai = bundle["anchor_il"]
    ax = bundle["anchor_xl"]
    print(
        "amp(anchor)=%.4f gmag(anchor)=%.4f p_pred(anchor)=%.4f conf(anchor)=%.4f p_channel(anchor)=%.4f"
        % (
            bundle["amp_slice"][ai, ax],
            bundle["gmag_slice"][ai, ax],
            bundle["p_pred_slice"][ai, ax],
            bundle["conf_slice"][ai, ax],
            bundle["pch_slice"][ai, ax],
        )
    )


def plot_overview(bundle: dict) -> None:
    cfg = bundle["cfg"]
    amp_vmin, amp_vmax = _percentile_limits(bundle["amp_slice"], pct=cfg["sim_clip"], symmetric=True)
    grad_vmin, grad_vmax = _percentile_limits(
        np.concatenate([bundle["gx_slice"].ravel(), bundle["gy_slice"].ravel()]),
        pct=cfg["sim_clip"],
        symmetric=True,
    )
    gmag_vmin, gmag_vmax = _percentile_limits(bundle["gmag_slice"], pct=cfg["sim_clip"], symmetric=False)
    weight_vmin, weight_vmax = _percentile_limits(bundle["incoming_weight_map"], pct=cfg["sim_clip"], symmetric=False)
    bundle["amp_limits"] = (amp_vmin, amp_vmax)
    bundle["grad_limits"] = (grad_vmin, grad_vmax)
    bundle["gmag_limits"] = (gmag_vmin, gmag_vmax)
    bundle["weight_limits"] = (weight_vmin, weight_vmax)
    p_vmin, p_vmax = 0.0, 1.0

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=int(cfg["fig_dpi"]))
    plots = [
        (bundle["amp_slice"], "Seismic amplitude amp", "seismic", amp_vmin, amp_vmax),
        (bundle["gx_slice"], "Forward diff gx", "coolwarm", grad_vmin, grad_vmax),
        (bundle["gy_slice"], "Forward diff gy", "coolwarm", grad_vmin, grad_vmax),
        (bundle["gmag_slice"], "Gradient magnitude gmag", "magma", gmag_vmin, gmag_vmax),
        (bundle["sim_map"], f"Anchor similarity exp(-dist/{cfg['tau']:.2f})", "viridis", 0.0, 1.0),
        (bundle["incoming_weight_map"], "Incoming lattice weight proxy", "turbo", weight_vmin, weight_vmax),
    ]
    for ax, (arr, title, cmap, vmin, vmax) in zip(axes.ravel(), plots):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        ax.scatter([w["xline"] for w in bundle["wells"]], [w["inline"] for w in bundle["wells"]], s=20, c="white", edgecolors="black", linewidths=0.5)
        ax.scatter(bundle["anchor_xl"], bundle["anchor_il"], s=90, c="gold", edgecolors="black", linewidths=1.0, marker="*")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Xline")
        ax.set_ylabel("Inline")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"FARP similarity feature overview | depth={bundle['focus_depth']} | anchor=({bundle['anchor_il']}, {bundle['anchor_xl']})",
        fontsize=14,
    )
    _finalize_figure(fig, top=0.92)
    _maybe_savefig(fig, cfg, f"depth{bundle['focus_depth']:03d}_overview")
    plt.show()

    feat_points = np.column_stack([bundle["amp_slice"].ravel(), bundle["gmag_slice"].ravel()]).astype(np.float32, copy=False)
    sim_flat = bundle["sim_map"].ravel()
    pch_flat = bundle["pch_slice"].ravel()
    max_points = int(cfg.get("feature_scatter_max_points", feat_points.shape[0]))
    if feat_points.shape[0] > max_points:
        rng = np.random.default_rng(int(cfg.get("feature_scatter_seed", 2026)))
        sel = rng.choice(feat_points.shape[0], size=max_points, replace=False)
        feat_points_plot = feat_points[sel]
        sim_plot = sim_flat[sel]
        pch_plot = pch_flat[sel]
    else:
        feat_points_plot = feat_points
        sim_plot = sim_flat
        pch_plot = pch_flat
    well_feat = np.asarray(
        [[bundle["amp_slice"][w["inline"], w["xline"]], bundle["gmag_slice"][w["inline"], w["xline"]]] for w in bundle["wells"]],
        dtype=np.float32,
    )
    anchor_feat = bundle["feat_slice"][:, bundle["anchor_il"], bundle["anchor_xl"]]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), dpi=int(cfg["fig_dpi"]))
    sc0 = axes[0].scatter(feat_points_plot[:, 0], feat_points_plot[:, 1], c=sim_plot, s=8, cmap="viridis", alpha=0.65, linewidths=0.0)
    axes[0].scatter(well_feat[:, 0], well_feat[:, 1], s=35, c="white", edgecolors="black", linewidths=0.6, label="wells")
    axes[0].scatter(anchor_feat[0], anchor_feat[1], s=120, c="gold", edgecolors="black", linewidths=1.0, marker="*", label="anchor")
    axes[0].set_title("Feature space: feat(x) = [amp(x), gmag(x)] | color = similarity")
    axes[0].set_xlabel("amp")
    axes[0].set_ylabel("gmag")
    axes[0].legend(loc="best")
    fig.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.04)

    sc1 = axes[1].scatter(feat_points_plot[:, 0], feat_points_plot[:, 1], c=pch_plot, s=8, cmap="turbo", alpha=0.65, linewidths=0.0, vmin=0.0, vmax=1.0)
    axes[1].scatter(well_feat[:, 0], well_feat[:, 1], s=35, c="white", edgecolors="black", linewidths=0.6, label="wells")
    axes[1].scatter(anchor_feat[0], anchor_feat[1], s=120, c="gold", edgecolors="black", linewidths=1.0, marker="*", label="anchor")
    axes[1].set_title("Same feature space | color = final p_channel")
    axes[1].set_xlabel("amp")
    axes[1].set_ylabel("gmag")
    axes[1].legend(loc="best")
    fig.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.04)
    _finalize_figure(fig, top=0.95)
    _maybe_savefig(fig, cfg, f"depth{bundle['focus_depth']:03d}_feature_space")
    plt.show()

    p_panels = []
    if bundle["p_prior_slice"] is not None:
        p_panels.append((bundle["p_prior_slice"], "p_prior", "gray_r", p_vmin, p_vmax))
    p_panels.extend(
        [
            (bundle["p_pred_slice"], "p_pred", "turbo", p_vmin, p_vmax),
            (bundle["conf_slice"], "conf", "viridis", p_vmin, p_vmax),
            (bundle["conf_gate_slice"], f"conf >= {cfg['conf_thresh']:.2f}", "gray_r", p_vmin, p_vmax),
            (bundle["p_mix_slice"], "p_mix", "turbo", p_vmin, p_vmax),
            (bundle["p_fallback_slice"], "p_fallback", "turbo", p_vmin, p_vmax),
            (bundle["pch_slice"], "final p_channel", "turbo", p_vmin, p_vmax),
        ]
    )
    ncols = 3
    nrows = int(np.ceil(len(p_panels) / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.2 * nrows), dpi=int(cfg["fig_dpi"]))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    for ax, panel in zip(axes.ravel(), p_panels):
        arr, title, cmap, vmin, vmax = panel
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        ax.scatter([w["xline"] for w in bundle["wells"]], [w["inline"] for w in bundle["wells"]], s=20, c="white", edgecolors="black", linewidths=0.5)
        ax.scatter(bundle["anchor_xl"], bundle["anchor_il"], s=90, c="gold", edgecolors="black", linewidths=1.0, marker="*")
        ax.set_title(title)
        ax.set_xlabel("Xline")
        ax.set_ylabel("Inline")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes.ravel()[len(p_panels):]:
        ax.axis("off")
    fig.suptitle(
        f"Channel-probability construction | prior={bundle['channel_components']['prior_source']} | pred={bundle['channel_components']['pred_source']}",
        fontsize=14,
    )
    _finalize_figure(fig, top=0.90)
    _maybe_savefig(fig, cfg, f"depth{bundle['focus_depth']:03d}_pchannel_chain")
    plt.show()

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=int(cfg["fig_dpi"]))
    im = ax.imshow(bundle["pch_slice"], cmap="gray_r", vmin=0.0, vmax=1.0, origin="upper")
    yy, xx = np.mgrid[0:bundle["shape"][1], 0:bundle["shape"][2]]
    step = max(1, int(cfg["quiver_step"]))
    ax.quiver(
        xx[::step, ::step],
        yy[::step, ::step],
        bundle["v_slice"][1, ::step, ::step],
        bundle["v_slice"][0, ::step, ::step],
        bundle["strength_slice"][::step, ::step],
        cmap="plasma",
        pivot="mid",
        scale=35,
        width=0.003,
    )
    ax.scatter([w["xline"] for w in bundle["wells"]], [w["inline"] for w in bundle["wells"]], s=24, c="white", edgecolors="black", linewidths=0.5)
    ax.scatter(bundle["anchor_xl"], bundle["anchor_il"], s=95, c="gold", edgecolors="black", linewidths=1.0, marker="*")
    ax.set_title("Channel gate p_channel with structure-tensor orientation")
    ax.set_xlabel("Xline")
    ax.set_ylabel("Inline")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _finalize_figure(fig, top=0.92)
    _maybe_savefig(fig, cfg, f"depth{bundle['focus_depth']:03d}_orientation")
    plt.show()


def plot_local(bundle: dict) -> None:
    cfg = bundle["cfg"]
    amp_vmin, amp_vmax = bundle["amp_limits"]
    gmag_vmin, gmag_vmax = bundle["gmag_limits"]
    weight_vmin, weight_vmax = bundle["weight_limits"]
    il0, il1, xl0, xl1 = bundle["crop_box"]
    panels = [
        (bundle["amp_crop"], "Local amp", "seismic", amp_vmin, amp_vmax),
        (bundle["gmag_crop"], "Local gmag", "magma", gmag_vmin, gmag_vmax),
        (bundle["sim_crop"], "Local similarity", "viridis", 0.0, 1.0),
        (bundle["w_crop"], "Local incoming weight", "turbo", weight_vmin, weight_vmax),
    ]
    if bundle["r_crop"] is not None:
        panels.append((bundle["r_crop"], f"Slice R(x) | {cfg['backend']}", "turbo", 0.0, 1.0))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 4.8), dpi=int(cfg["fig_dpi"]))
    axes = np.atleast_1d(axes)
    for ax, (arr, title, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        ax.scatter(bundle["anchor_xl"] - xl0, bundle["anchor_il"] - il0, s=90, c="gold", edgecolors="black", linewidths=1.0, marker="*")
        ax.set_title(title)
        ax.set_xlabel(f"xline [{xl0}:{xl1})")
        ax.set_ylabel(f"inline [{il0}:{il1})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Anchor-centered local crop", fontsize=14)
    _finalize_figure(fig, top=0.92)
    _maybe_savefig(fig, cfg, f"depth{bundle['focus_depth']:03d}_local_crop")
    plt.show()

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), dpi=int(cfg["fig_dpi"]))
    titles = {"g": "gate g", "a": "orientation affinity a", "s": "feature similarity s", "w": "combined w = g*a*s"}
    for ax, key in zip(axes, ("g", "a", "s", "w")):
        mat = bundle["local_maps"][key]
        im = ax.imshow(mat, cmap="viridis", origin="upper")
        for rr in range(3):
            for cc in range(3):
                if np.isfinite(mat[rr, cc]):
                    ax.text(cc, rr, f"{mat[rr, cc]:.2f}", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
                elif rr == 1 and cc == 1:
                    ax.text(cc, rr, "anchor", ha="center", va="center", color="black", fontsize=9)
        ax.set_xticks([0, 1, 2], ["-1", "0", "+1"])
        ax.set_yticks([0, 1, 2], ["-1", "0", "+1"])
        ax.set_xlabel("d xline")
        ax.set_ylabel("d inline")
        ax.set_title(titles[key])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Exact 3x3 directed transition terms from anchor to its 8 neighbors", fontsize=14)
    _finalize_figure(fig, top=0.90)
    _maybe_savefig(fig, cfg, f"depth{bundle['focus_depth']:03d}_local_terms")
    plt.show()


def _get_edge_segments(node_rc: np.ndarray, src_idx: np.ndarray, dst_idx: np.ndarray, max_edges: int) -> tuple[np.ndarray, np.ndarray]:
    edge_n = int(src_idx.shape[0])
    if edge_n <= 0:
        return np.empty((0, 2, 2), dtype=np.float32), np.empty((0,), dtype=np.int64)
    if max_edges > 0 and edge_n > max_edges:
        sel = np.linspace(0, edge_n - 1, max_edges, dtype=np.int64)
    else:
        sel = np.arange(edge_n, dtype=np.int64)
    src_sel = src_idx[sel].astype(np.int64, copy=False)
    dst_sel = dst_idx[sel].astype(np.int64, copy=False)
    segments = np.stack(
        [
            np.column_stack([node_rc[src_sel, 1], node_rc[src_sel, 0]]),
            np.column_stack([node_rc[dst_sel, 1], node_rc[dst_sel, 0]]),
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    return segments, sel


def _plot_edge_term_panel(
    *,
    fig,
    ax,
    background: np.ndarray,
    node_rc: np.ndarray,
    src_idx: np.ndarray,
    dst_idx: np.ndarray,
    values: np.ndarray,
    title: str,
    max_edges: int,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    wells: list[dict] | None = None,
) -> None:
    ax.imshow(background, cmap="gray_r", vmin=0.0, vmax=1.0, origin="upper")
    segments, sel = _get_edge_segments(node_rc, src_idx, dst_idx, max_edges=max_edges)
    if segments.shape[0] > 0:
        values_sel = np.asarray(values, dtype=np.float32).reshape(-1)[sel]
        if vmin is None or vmax is None:
            vmin, vmax = _percentile_limits(values_sel, pct=(2.0, 98.0), symmetric=False)
        lc = LineCollection(segments, cmap=cmap, linewidths=1.2, alpha=0.95)
        lc.set_array(values_sel)
        lc.set_clim(vmin, vmax)
        ax.add_collection(lc)
        fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
    ax.scatter(node_rc[:, 1], node_rc[:, 0], s=5, c="white", linewidths=0.0, alpha=0.85)
    if wells is not None:
        ax.scatter([w["xline"] for w in wells], [w["inline"] for w in wells], s=18, c="black", marker="+", linewidths=0.8)
    ax.set_title(title)
    ax.set_xlabel("Xline")
    ax.set_ylabel("Inline")


def plot_skeleton_mask_extraction(bundle: dict) -> None:
    cfg = bundle["cfg"]
    dbg = bundle["skeleton_debug"]
    p_vmin, p_vmax = 0.0, 1.0
    amp_vmin, amp_vmax = bundle["amp_limits"]

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.0), dpi=int(cfg["fig_dpi"]))
    panels = [
        (bundle["pch_slice"], "final p_channel", "turbo", p_vmin, p_vmax),
        (bundle["conf_slice"], "conf", "viridis", p_vmin, p_vmax),
        (dbg["mask"].astype(np.float32), f"channel-like mask\np_channel >= {cfg['agpe_skel_p_thresh']:.2f} and conf > 0", "gray_r", 0.0, 1.0),
        (bundle["amp_slice"], "amp with mask contour", "seismic", amp_vmin, amp_vmax),
        (bundle["pch_slice"], "p_channel with mask contour", "gray_r", p_vmin, p_vmax),
        (dbg["mask"].astype(np.float32), "mask + wells", "gray_r", 0.0, 1.0),
    ]
    for idx, (ax, (arr, title, cmap, vmin, vmax)) in enumerate(zip(axes.ravel(), panels)):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        if idx in (3, 4):
            ax.contour(dbg["mask"].astype(np.float32), levels=[0.5], colors=["gold"], linewidths=1.0)
        ax.scatter([w["xline"] for w in bundle["wells"]], [w["inline"] for w in bundle["wells"]], s=18, c="white", edgecolors="black", linewidths=0.4)
        ax.scatter(bundle["anchor_xl"], bundle["anchor_il"], s=85, c="gold", edgecolors="black", linewidths=1.0, marker="*")
        ax.set_title(title)
        ax.set_xlabel("Xline")
        ax.set_ylabel("Inline")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("1. Channel-like mask extraction from gated p_channel", fontsize=14)
    _finalize_figure(fig, top=0.92)
    _maybe_savefig(fig, cfg, f"depth{bundle['focus_depth']:03d}_skeleton_01_mask")
    plt.show()


def plot_skeleton_extraction(bundle: dict) -> None:
    cfg = bundle["cfg"]
    dbg = bundle["skeleton_debug"]
    if not dbg.get("cache_available", False):
        print("Skeleton extraction skipped: no valid skeleton graph could be built for the focus slice.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 11.0), dpi=int(cfg["fig_dpi"]))
    panels = [
        (dbg["mask"].astype(np.float32), "channel-like mask", "gray_r", 0.0, 1.0),
        (dbg["skel_mask"].astype(np.float32), "skeleton extraction skel_mask", "gray_r", 0.0, 1.0),
        (bundle["pch_slice"], "p_channel + skeleton overlay", "gray_r", 0.0, 1.0),
        (dbg["degree_maps"]["total"], "skeleton node total degree", "viridis", None, None),
    ]
    for idx, (ax, (arr, title, cmap, vmin, vmax)) in enumerate(zip(axes.ravel(), panels)):
        if vmin is None or vmax is None:
            vmin, vmax = _percentile_limits(arr[np.isfinite(arr)], pct=(0.0, 100.0), symmetric=False)
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        if idx == 2:
            ax.contour(dbg["skel_mask"].astype(np.float32), levels=[0.5], colors=["cyan"], linewidths=1.0)
            ax.scatter(dbg["node_rc"][:, 1], dbg["node_rc"][:, 0], s=5, c="gold", linewidths=0.0)
        ax.set_title(title)
        ax.set_xlabel("Xline")
        ax.set_ylabel("Inline")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"2. Skeleton extraction | n_nodes={dbg['n_nodes']} | n_edges={dbg['n_edges']} | n_long_edges={dbg['n_long_edges']}",
        fontsize=14,
    )
    _finalize_figure(fig, top=0.92)
    _maybe_savefig(fig, cfg, f"depth{bundle['focus_depth']:03d}_skeleton_02_extract")
    plt.show()


def plot_skeleton_graph_construction(bundle: dict) -> None:
    cfg = bundle["cfg"]
    dbg = bundle["skeleton_debug"]
    if not dbg.get("cache_available", False):
        print("Skeleton graph construction skipped: no valid skeleton graph could be built for the focus slice.")
        return

    node_rc = dbg["node_rc"]
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.5), dpi=int(cfg["fig_dpi"]))
    panels = [
        (dbg["edge_count_maps"]["local_in"], "local edge incidence on graph nodes", "viridis"),
        (dbg["edge_count_maps"]["long_in"], "long-edge incidence on graph nodes", "magma"),
        (dbg["degree_maps"]["total"], "node degree on graph", "plasma"),
    ]
    for ax, (arr, title, cmap) in zip(axes, panels):
        ax.imshow(bundle["pch_slice"], cmap="gray_r", vmin=0.0, vmax=1.0, origin="upper")
        valid = np.isfinite(arr)
        if np.any(valid):
            rows, cols = np.where(valid)
            vals = arr[valid]
            sc = ax.scatter(cols, rows, c=vals, s=18, cmap=cmap, linewidths=0.0, alpha=0.95)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        ax.contour(dbg["skel_mask"].astype(np.float32), levels=[0.5], colors=["white"], linewidths=0.7, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("Xline")
        ax.set_ylabel("Inline")

    fig.suptitle("3. Skeleton graph construction", fontsize=14)
    _finalize_figure(fig, top=0.90)
    _maybe_savefig(fig, cfg, f"depth{bundle['focus_depth']:03d}_skeleton_03_graph")
    plt.show()


def plot_seed_snapping(bundle: dict) -> None:
    cfg = bundle["cfg"]
    dbg = bundle["skeleton_debug"]
    if not dbg.get("cache_available", False):
        print("Seed snapping skipped: no valid skeleton graph could be built for the focus slice.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.4), dpi=int(cfg["fig_dpi"]))
    base = bundle["pch_slice"]
    for ax in axes:
        ax.imshow(base, cmap="gray_r", vmin=0.0, vmax=1.0, origin="upper")
        ax.contour(dbg["skel_mask"].astype(np.float32), levels=[0.5], colors=["white"], linewidths=0.7, alpha=0.7)
        ax.set_xlabel("Xline")
        ax.set_ylabel("Inline")

    raw_rc = dbg["raw_seed_rc"]
    if raw_rc.size > 0:
        axes[0].scatter(raw_rc[:, 1], raw_rc[:, 0], s=36, c="deepskyblue", edgecolors="black", linewidths=0.5)
    axes[0].set_title("raw well seeds on slice")

    snapped_rc = dbg["snapped_seed_rc"]
    if snapped_rc.size > 0:
        axes[1].scatter(snapped_rc[:, 1], snapped_rc[:, 0], s=46, c="gold", edgecolors="black", linewidths=0.5)
    axes[1].set_title("unique snapped graph nodes")

    if raw_rc.size > 0:
        axes[2].scatter(raw_rc[:, 1], raw_rc[:, 0], s=26, c="deepskyblue", edgecolors="black", linewidths=0.4, label="raw well seed")
    if snapped_rc.size > 0:
        axes[2].scatter(snapped_rc[:, 1], snapped_rc[:, 0], s=42, c="gold", edgecolors="black", linewidths=0.5, label="snapped node")
    for rec in dbg["snapped_records"]:
        raw_r, raw_c = rec["raw_rc"]
        snap_r, snap_c = rec["snap_rc"]
        axes[2].plot([raw_c, snap_c], [raw_r, snap_r], color="orange", linewidth=0.9, alpha=0.8)
    if dbg["failed_records"]:
        fail_rc = np.asarray([rec["raw_rc"] for rec in dbg["failed_records"]], dtype=np.int32)
        axes[2].scatter(fail_rc[:, 1], fail_rc[:, 0], s=38, c="red", marker="x", linewidths=1.0, label="failed snap")
    axes[2].legend(loc="best")
    axes[2].set_title(f"seed snapping to graph | snap_ok={dbg['snap_ok']} / {dbg['seed_total']}")

    fig.suptitle("4. Well seed snapping to graph", fontsize=14)
    _finalize_figure(fig, top=0.90)
    _maybe_savefig(fig, cfg, f"depth{bundle['focus_depth']:03d}_skeleton_04_snapping")
    plt.show()


def plot_anisotropic_graph_diffusion(bundle: dict) -> None:
    cfg = bundle["cfg"]
    dbg = bundle["skeleton_debug"]
    if not dbg.get("cache_available", False):
        print("Anisotropic graph diffusion skipped: no valid skeleton graph could be built for the focus slice.")
        return

    node_rc = dbg["node_rc"]
    term_names = [
        ("g", "gate g"),
        ("a", "orientation affinity a"),
        ("s", "feature similarity s"),
        ("p_edge", "channel/conf continuity p_edge"),
        ("w", "combined weight w"),
    ]

    fig, axes = plt.subplots(1, len(term_names), figsize=(4.1 * len(term_names), 4.8), dpi=int(cfg["fig_dpi"]))
    axes = np.atleast_1d(axes)
    for ax, (name, title) in zip(axes, term_names):
        ax.imshow(bundle["pch_slice"], cmap="gray_r", vmin=0.0, vmax=1.0, origin="upper")
        arr = dbg["edge_node_means"][name]
        valid = np.isfinite(arr)
        if np.any(valid):
            rows, cols = np.where(valid)
            vals = arr[valid]
            sc = ax.scatter(cols, rows, c=vals, s=18, cmap="viridis", linewidths=0.0, alpha=0.95)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        ax.contour(dbg["skel_mask"].astype(np.float32), levels=[0.5], colors=["white"], linewidths=0.6, alpha=0.7)
        ax.scatter(node_rc[:, 1], node_rc[:, 0], s=4, c="white", linewidths=0.0, alpha=0.4)
        ax.set_title(title)
        ax.set_xlabel("Xline")
        ax.set_ylabel("Inline")
    fig.suptitle("5. Anisotropic graph diffusion | edge-weight decomposition", fontsize=14)
    _finalize_figure(fig, top=0.88)
    _maybe_savefig(fig, cfg, f"depth{bundle['focus_depth']:03d}_skeleton_05_edge_terms")
    plt.show()

    capture_steps = dbg["capture_steps"]
    ncols = len(capture_steps)
    fig, axes = plt.subplots(1, ncols, figsize=(4.0 * ncols, 4.8), dpi=int(cfg["fig_dpi"]))
    axes = np.atleast_1d(axes)
    for ax, step in zip(axes, capture_steps):
        ax.imshow(bundle["pch_slice"], cmap="gray_r", vmin=0.0, vmax=1.0, origin="upper")
        grid = dbg["r_node_grids"].get(step)
        if grid is not None:
            node_mask = np.isfinite(grid)
            if np.any(node_mask):
                rows, cols = np.where(node_mask)
                vals = grid[node_mask]
                sc = ax.scatter(cols, rows, c=vals, s=18, cmap="turbo", vmin=0.0, vmax=1.0, linewidths=0.0)
                fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        ax.contour(dbg["skel_mask"].astype(np.float32), levels=[0.5], colors=["white"], linewidths=0.6, alpha=0.7)
        ax.set_title(f"node reliability on graph | step={step}")
        ax.set_xlabel("Xline")
        ax.set_ylabel("Inline")
    fig.suptitle(
        f"5. Anisotropic graph diffusion | node states | fallback={dbg.get('fallback_reason')}",
        fontsize=14,
    )
    _finalize_figure(fig, top=0.88)
    _maybe_savefig(fig, cfg, f"depth{bundle['focus_depth']:03d}_skeleton_05_node_history")
    plt.show()


def plot_lift_back_and_slice_stack(bundle: dict) -> None:
    cfg = bundle["cfg"]
    dbg = bundle["skeleton_debug"]
    if not dbg.get("cache_available", False) or dbg.get("fallback_reason") is not None:
        print("Lift-back visualization skipped: skeleton graph was not usable for slice diffusion on the focus depth.")
        return

    nearest_mod = np.where(dbg["lift_nearest_idx"] >= 0, dbg["lift_nearest_idx"] % 20, np.nan).astype(np.float32)
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 10.5), dpi=int(cfg["fig_dpi"]))
    panels = [
        (nearest_mod, "nearest skeleton node id mod 20", "tab20", None, None),
        (dbg["lift_dist"], "lift distance to skeleton", "magma", None, None),
        (dbg["lift_decay"], "lift decay exp(-dist / lift_sigma)", "viridis", 0.0, 1.0),
        (dbg["r_grid_pre"], "lift-back to slice before well constraint", "turbo", 0.0, 1.0),
        (dbg["r_grid_post"], "slice reliability after well constraint", "turbo", 0.0, 1.0),
        (dbg["r_grid_delta"], "well-constraint delta", "coolwarm", None, None),
    ]
    for ax, (arr, title, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        if vmin is None or vmax is None:
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                vmin, vmax = 0.0, 1.0
            elif title == "well-constraint delta":
                vmin, vmax = _percentile_limits(finite, pct=(1.0, 99.0), symmetric=True)
            else:
                vmin, vmax = _percentile_limits(finite, pct=(1.0, 99.0), symmetric=False)
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        ax.scatter([w["xline"] for w in bundle["wells"]], [w["inline"] for w in bundle["wells"]], s=18, c="white", edgecolors="black", linewidths=0.4)
        ax.set_title(title)
        ax.set_xlabel("Xline")
        ax.set_ylabel("Inline")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("6. Lift-back to slice from graph nodes", fontsize=14)
    _finalize_figure(fig, top=0.92)
    _maybe_savefig(fig, cfg, f"depth{bundle['focus_depth']:03d}_skeleton_06_lift_back")
    plt.show()


def plot_traces(bundle: dict, r_cube: np.ndarray | None = None) -> None:
    cfg = bundle["cfg"]
    ai = bundle["anchor_il"]
    ax = bundle["anchor_xl"]
    depth_axis = np.arange(bundle["shape"][0])
    p_comp = bundle["channel_components"]
    traces = [
        ("amp trace", bundle["amp_hilxl"][:, ai, ax], "#1f77b4"),
        ("gmag trace", bundle["gmag_hilxl"][:, ai, ax], "#d62728"),
        ("p_pred trace", p_comp["p_pred_3d"][:, ai, ax], "#2ca02c"),
        ("conf trace", p_comp["conf_3d"][:, ai, ax], "#17becf"),
        ("p_channel trace", p_comp["p_channel_3d"][:, ai, ax], "#ff7f0e"),
        ("AI trace", bundle["ai_hilxl"][:, ai, ax], "#9467bd"),
    ]
    if p_comp["p_prior_3d"] is not None:
        traces.insert(2, ("p_prior trace", p_comp["p_prior_3d"][:, ai, ax], "#8c564b"))
    if r_cube is not None:
        traces.append(("R(x) trace", r_cube[:, ai, ax], "#7f7f7f"))

    fig, axes = plt.subplots(1, len(traces), figsize=(4.2 * len(traces), 5), dpi=int(cfg["fig_dpi"]), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, (title, values, color) in zip(axes, traces):
        axis.plot(values, depth_axis, color=color, lw=1.5)
        focus_depth = int(bundle["focus_depth"])
        focus_depth = int(np.clip(focus_depth, 0, len(depth_axis) - 1))
        axis.scatter([values[focus_depth]], [focus_depth], s=28, c="black", zorder=3)
        axis.invert_yaxis()
        axis.set_title(title)
        axis.set_xlabel("value")
    axes[0].set_ylabel("Depth index")
    fig.suptitle(
        f"Anchor trace summary | {bundle['anchor']['wellname']} | inline={ai}, xline={ax}",
        fontsize=14,
    )
    _finalize_figure(fig, top=0.92)
    _maybe_savefig(fig, cfg, f"depth{bundle['focus_depth']:03d}_trace_summary")
    plt.show()


def load_or_compute_full_r(bundle: dict) -> tuple[np.ndarray | None, object | None]:
    cfg = bundle["cfg"]
    torch = _require_torch()
    r3d_path = cfg.get("r3d_path", None)
    if r3d_path is not None:
        arr = np.load(Path(r3d_path))
        h, il, xl = bundle["shape"]
        if arr.shape == (h, il, xl):
            return arr.astype(np.float32, copy=False), None
        if arr.shape == (il, xl, h):
            return np.transpose(arr, (2, 0, 1)).astype(np.float32, copy=False), None
        raise ValueError(f"Unsupported R cube shape {arr.shape}; expected {(h, il, xl)} or {(il, xl, h)}")

    if not cfg.get("compute_full_r", False):
        return None, None

    from utils.reliability_aniso import build_R_and_prior_from_cube

    h, il, xl = bundle["shape"]
    ai_dummy = torch.zeros((h, il, xl), dtype=torch.float32)
    well_trace_indices = torch.tensor([w["trace_index"] for w in bundle["wells"]], dtype=torch.long)
    build_kwargs = {
        "seismic_3d": torch.from_numpy(bundle["seismic_hilxl"].astype(np.float32)),
        "ai_3d": ai_dummy,
        "well_trace_indices": well_trace_indices,
        "facies_3d": torch.from_numpy(bundle["facies_hilxl"].astype(np.int64)),
        "facies_prob_3d": None if bundle["facies_prob_3d"] is None else torch.from_numpy(bundle["facies_prob_3d"].astype(np.float32)),
        "p_channel_3d": None if bundle["p_channel_input_3d"] is None else torch.from_numpy(bundle["p_channel_input_3d"].astype(np.float32)),
        "conf_3d": None if bundle["conf_input_3d"] is None else torch.from_numpy(bundle["conf_input_3d"].astype(np.float32)),
        "facies_prior_3d": None if bundle["facies_prior_3d"] is None else torch.from_numpy(bundle["facies_prior_3d"].astype(np.int64)),
        "channel_id": int(cfg["channel_id"]),
        "alpha_prior": float(cfg["alpha_prior"]),
        "conf_thresh": float(cfg["conf_thresh"]),
        "neutral_p": float(cfg["neutral_p"]),
        "steps_R": int(cfg["steps_r"]),
        "eta": float(cfg["eta"]),
        "gamma": float(cfg["gamma"]),
        "tau": float(cfg["tau"]),
        "kappa": float(cfg["kappa"]),
        "sigma_st": float(cfg["sigma_st"]),
        "backend": str(cfg["full_r_backend"]),
        "aniso_use_tensor_strength": bool(cfg["use_tensor_strength"]),
        "aniso_tensor_strength_power": float(cfg["tensor_strength_power"]),
        "agpe_skel_p_thresh": float(cfg["agpe_skel_p_thresh"]),
        "agpe_skel_min_nodes": int(cfg["agpe_skel_min_nodes"]),
        "agpe_skel_snap_radius": int(cfg["agpe_skel_snap_radius"]),
        "agpe_long_edges": bool(cfg["agpe_long_edges"]),
        "agpe_long_max_step": int(cfg["agpe_long_max_step"]),
        "agpe_long_step": int(cfg["agpe_long_step"]),
        "agpe_long_cos_thresh": float(cfg["agpe_long_cos_thresh"]),
        "agpe_long_weight": float(cfg["agpe_long_weight"]),
        "agpe_edge_tau_p": float(cfg["agpe_edge_tau_p"]),
        "agpe_lift_sigma": float(cfg["agpe_lift_sigma"]),
        "agpe_well_soft_alpha": float(cfg["well_soft_alpha"]),
        "use_soft_prior": False,
    }

    if str(cfg["full_r_backend"]) == "skeleton_graph":
        r_flat, _, graph_cache = build_R_and_prior_from_cube(return_graph_cache=True, **build_kwargs)
    else:
        r_flat, _ = build_R_and_prior_from_cube(**build_kwargs)
        graph_cache = None
    r_cube = r_flat.detach().cpu().numpy().T.reshape(h, il, xl).astype(np.float32, copy=False)
    return r_cube, graph_cache


def plot_full_r(bundle: dict, r_cube: np.ndarray | None) -> None:
    if r_cube is None:
        print("Full-cube R view skipped. Set cfg['r3d_path'] or cfg['compute_full_r']=True if you need it.")
        return

    cfg = bundle["cfg"]
    il_pick = bundle["anchor_il"]
    xl_pick = bundle["anchor_xl"]
    z_pick = bundle["focus_depth"]
    depth_offsets = tuple(cfg.get("stack_depth_offsets", (-2, -1, 0, 1, 2)))
    depth_picks = []
    for offset in depth_offsets:
        depth_picks.append(int(np.clip(z_pick + int(offset), 0, bundle["shape"][0] - 1)))
    depth_picks = list(dict.fromkeys(depth_picks))

    fig, axes = plt.subplots(1, len(depth_picks), figsize=(3.8 * len(depth_picks), 4.5), dpi=int(cfg["fig_dpi"]))
    axes = np.atleast_1d(axes)
    for ax, depth_idx in zip(axes, depth_picks):
        im = ax.imshow(r_cube[depth_idx], cmap="turbo", vmin=0.0, vmax=1.0, origin="upper")
        ax.scatter([w["xline"] for w in bundle["wells"]], [w["inline"] for w in bundle["wells"]], s=16, c="white", edgecolors="black", linewidths=0.4)
        ax.scatter(xl_pick, il_pick, s=80, c="gold", edgecolors="black", linewidths=0.8, marker="*")
        ax.set_title(f"R depth slice z={depth_idx}")
        ax.set_xlabel("Xline")
        ax.set_ylabel("Inline")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("7. Stack to 3D reliability field R(x) | neighboring depth slices", fontsize=14)
    _finalize_figure(fig, top=0.88)
    _maybe_savefig(fig, cfg, f"depth{bundle['focus_depth']:03d}_skeleton_07_depth_stack")
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), dpi=int(cfg["fig_dpi"]))
    ims = [
        axes[0].imshow(r_cube[z_pick], cmap="turbo", vmin=0.0, vmax=1.0, origin="upper"),
        axes[1].imshow(r_cube[:, il_pick, :], cmap="turbo", vmin=0.0, vmax=1.0, origin="upper", aspect="auto"),
        axes[2].imshow(r_cube[:, :, xl_pick], cmap="turbo", vmin=0.0, vmax=1.0, origin="upper", aspect="auto"),
    ]
    axes[0].set_title(f"R depth slice z={z_pick}")
    axes[0].scatter([w["xline"] for w in bundle["wells"]], [w["inline"] for w in bundle["wells"]], s=18, c="white", edgecolors="black", linewidths=0.5)
    axes[0].scatter(xl_pick, il_pick, s=95, c="gold", edgecolors="black", linewidths=1.0, marker="*")
    axes[1].set_title(f"R inline section il={il_pick}")
    axes[2].set_title(f"R xline section xl={xl_pick}")
    axes[0].set_xlabel("Xline")
    axes[0].set_ylabel("Inline")
    axes[1].set_xlabel("Xline")
    axes[1].set_ylabel("Depth index")
    axes[2].set_xlabel("Inline")
    axes[2].set_ylabel("Depth index")
    for ax, im in zip(axes, ims):
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("7. Stack to 3D reliability field R(x) | orthogonal sections", fontsize=14)
    _finalize_figure(fig, top=0.92)
    _maybe_savefig(fig, cfg, f"depth{bundle['focus_depth']:03d}_skeleton_07_sections")
    plt.show()

    graph_cache = bundle.get("full_graph_cache")
    if graph_cache is None or not getattr(graph_cache, "slices", None):
        return

    depth_axis = np.arange(bundle["shape"][0], dtype=np.int32)
    n_nodes = np.full_like(depth_axis, np.nan, dtype=np.float32)
    n_edges = np.full_like(depth_axis, np.nan, dtype=np.float32)
    long_ratio = np.full_like(depth_axis, np.nan, dtype=np.float32)
    snap_ok_ratio = np.full_like(depth_axis, np.nan, dtype=np.float32)
    fallback = np.zeros_like(depth_axis, dtype=np.float32)
    for depth_idx, cache_slice in graph_cache.slices.items():
        if 0 <= int(depth_idx) < depth_axis.size:
            n_nodes[int(depth_idx)] = float(cache_slice.n_nodes)
            n_edges[int(depth_idx)] = float(cache_slice.n_edges)
            long_ratio[int(depth_idx)] = float(cache_slice.n_long_edges) / max(float(cache_slice.n_edges), 1.0)
            snap_ok_ratio[int(depth_idx)] = float(cache_slice.snap_ok) / max(float(cache_slice.snap_total), 1.0)
            fallback[int(depth_idx)] = 1.0 if bool(cache_slice.fallback) else 0.0

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8), dpi=int(cfg["fig_dpi"]), sharex=True)
    series = [
        (n_nodes, "n_nodes / slice", "#1f77b4"),
        (n_edges, "n_edges / slice", "#d62728"),
        (long_ratio, "long-edge ratio", "#2ca02c"),
        (snap_ok_ratio, "snap-ok ratio", "#9467bd"),
    ]
    for ax, (values, title, color) in zip(axes.ravel(), series):
        ax.plot(depth_axis, values, color=color, lw=1.3)
        ax.axvline(z_pick, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Depth index")
        ax.set_ylabel("value")
    fig.suptitle(
        f"7. Stack to 3D reliability field R(x) | skeleton-graph slice stats | backend={graph_cache.last_stats.get('backend', 'n/a')}",
        fontsize=14,
    )
    _finalize_figure(fig, top=0.90)
    _maybe_savefig(fig, cfg, f"depth{bundle['focus_depth']:03d}_skeleton_07_stats")
    plt.show()


def run_all(cfg: dict | None = None) -> tuple[dict, np.ndarray | None]:
    cfg = build_default_config() if cfg is None else cfg
    bundle = prepare_bundle(cfg)
    print_summary(bundle)
    plot_overview(bundle)
    plot_local(bundle)
    plot_skeleton_mask_extraction(bundle)
    plot_skeleton_extraction(bundle)
    plot_skeleton_graph_construction(bundle)
    plot_seed_snapping(bundle)
    plot_anisotropic_graph_diffusion(bundle)
    plot_lift_back_and_slice_stack(bundle)
    r_cube, graph_cache = load_or_compute_full_r(bundle)
    bundle["full_graph_cache"] = graph_cache
    plot_traces(bundle, r_cube=r_cube)
    plot_full_r(bundle, r_cube)
    return bundle, r_cube
