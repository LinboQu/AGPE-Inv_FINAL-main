from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


def _format_snr_db(snr_db: float | int) -> str:
    value = float(snr_db)
    if value.is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def build_noise_label(
    noise_kind: str | None,
    snr_db: float | int | None = None,
    seed: int | None = None,
) -> str:
    kind = str(noise_kind or "none").strip().lower()
    if kind in {"", "none", "clean"}:
        return "clean"
    if kind == "awgn":
        if snr_db is None:
            raise ValueError("snr_db is required when noise_kind='awgn'.")
        if seed is None:
            raise ValueError("seed is required when noise_kind='awgn'.")
        return f"awgn_{_format_snr_db(snr_db)}db_seed{int(seed)}"
    raise ValueError(f"Unsupported noise_kind='{noise_kind}'.")


def measure_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    clean64 = np.asarray(clean, dtype=np.float64)
    noisy64 = np.asarray(noisy, dtype=np.float64)
    if clean64.shape != noisy64.shape:
        raise ValueError(
            f"clean/noisy shape mismatch: clean={clean64.shape}, noisy={noisy64.shape}"
        )

    signal_power = float(np.mean(np.square(clean64)))
    noise_power = float(np.mean(np.square(noisy64 - clean64)))
    if signal_power <= 0.0:
        raise ValueError("Signal power must be positive to measure SNR.")
    if noise_power <= 0.0:
        return float("inf")
    return float(10.0 * math.log10(signal_power / noise_power))


def add_awgn_by_snr(
    seismic_raw: np.ndarray,
    snr_db: float | int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    clean = np.asarray(seismic_raw)
    clean64 = np.asarray(clean, dtype=np.float64)

    signal_power = float(np.mean(np.square(clean64)))
    if signal_power <= 0.0:
        raise ValueError("Signal power must be positive for AWGN injection.")

    snr_target = float(snr_db)
    noise_power = float(signal_power / (10.0 ** (snr_target / 10.0)))
    sigma = float(math.sqrt(noise_power))
    rng = np.random.default_rng(int(seed))
    noise = rng.normal(loc=0.0, scale=sigma, size=clean64.shape)
    noisy64 = clean64 + noise

    snr_measured = measure_snr_db(clean64, noisy64)
    snr_error = abs(snr_measured - snr_target)
    if snr_error > 0.2:
        raise RuntimeError(
            f"Measured SNR deviates too much from target: "
            f"target={snr_target:.4f} dB, measured={snr_measured:.4f} dB, error={snr_error:.4f} dB"
        )

    out_dtype = clean.dtype if np.issubdtype(clean.dtype, np.floating) else np.float32
    noisy = noisy64.astype(out_dtype, copy=False)
    meta = {
        "noise_kind": "awgn",
        "noise_label": build_noise_label("awgn", snr_db=snr_target, seed=seed),
        "snr_db_target": snr_target,
        "snr_db_measured": float(snr_measured),
        "snr_db_error": float(snr_error),
        "seed": int(seed),
        "signal_power": float(signal_power),
        "noise_power": float(noise_power),
        "sigma": float(sigma),
    }
    return noisy, meta


def _rebuild_seismic3d_from_flat(
    seismic_flat: np.ndarray,
    meta: Mapping[str, Any],
) -> np.ndarray:
    required = {"H", "inline", "xline"}
    missing = sorted(required - set(meta.keys()))
    if missing:
        raise KeyError(f"meta missing required keys for seismic3d rebuild: {missing}")

    H = int(meta["H"])
    inline = int(meta["inline"])
    xline = int(meta["xline"])
    expected_shape = (inline * xline, H)
    if tuple(seismic_flat.shape) != expected_shape:
        raise ValueError(
            f"Unexpected seismic_flat shape: got={tuple(seismic_flat.shape)}, expected={expected_shape}"
        )
    return np.ascontiguousarray(np.asarray(seismic_flat).T.reshape(H, inline, xline))


def apply_test_noise(
    seismic_raw: np.ndarray,
    meta: Mapping[str, Any],
    noise_kind: str | None,
    snr_db: float | int | None,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    kind = str(noise_kind or "none").strip().lower()
    meta_out = {str(k): v for k, v in meta.items()}

    if kind in {"", "none", "clean"}:
        noise_meta = {
            "noise_kind": "none",
            "noise_label": build_noise_label("none"),
            "snr_db_target": None,
            "snr_db_measured": None,
            "snr_db_error": None,
            "seed": None,
            "signal_power": None,
            "noise_power": None,
            "sigma": None,
        }
        return np.asarray(seismic_raw), meta_out, noise_meta

    if kind != "awgn":
        raise ValueError(f"Unsupported test noise_kind='{noise_kind}'.")
    if snr_db is None:
        raise ValueError("test_noise_snr_db is required when test_noise_kind='awgn'.")

    seismic_noisy, noise_meta = add_awgn_by_snr(seismic_raw, snr_db=snr_db, seed=seed)
    meta_out["seismic3d"] = _rebuild_seismic3d_from_flat(seismic_noisy, meta_out)
    return seismic_noisy, meta_out, noise_meta
