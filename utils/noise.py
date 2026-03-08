from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import torch


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


def normalize_snr_choices(snr_choices: Any) -> tuple[float, ...]:
    if snr_choices is None:
        return ()
    if isinstance(snr_choices, str):
        text = snr_choices.strip()
        if text == "":
            return ()
        for token in "[]()":
            text = text.replace(token, " ")
        text = text.replace(";", ",")
        parts = [p.strip() for chunk in text.split(",") for p in chunk.split() if p.strip()]
        return tuple(float(part) for part in parts)
    if isinstance(snr_choices, np.ndarray):
        return tuple(float(x) for x in snr_choices.reshape(-1).tolist())
    if isinstance(snr_choices, (list, tuple, set)):
        return tuple(float(x) for x in snr_choices)
    return (float(snr_choices),)


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


def apply_train_input_perturbation(
    seismic_batch: torch.Tensor,
    noise_kind: str | None = None,
    noise_prob: float = 0.0,
    noise_snr_db_choices: Any = None,
    r_channel_dropout_prob: float = 0.0,
    seed: int | None = None,
) -> torch.Tensor:
    if seismic_batch.ndim != 3:
        raise ValueError(
            f"Expected seismic_batch with shape [B,C,H], got {tuple(seismic_batch.shape)}"
        )

    out = seismic_batch.clone()
    noise_kind_norm = str(noise_kind or "none").strip().lower()
    noise_prob_value = float(np.clip(float(noise_prob), 0.0, 1.0))
    rdrop_prob_value = float(np.clip(float(r_channel_dropout_prob), 0.0, 1.0))
    snr_choices = normalize_snr_choices(noise_snr_db_choices)

    use_awgn = (noise_kind_norm == "awgn") and (noise_prob_value > 0.0) and (len(snr_choices) > 0)
    use_rdrop = (out.shape[1] > 1) and (rdrop_prob_value > 0.0)
    if not use_awgn and not use_rdrop:
        return out

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(0 if seed is None else seed))
    batch_size = int(out.shape[0])

    if use_awgn:
        amp_cpu = out[:, 0, :].detach().to(device="cpu", dtype=torch.float32)
        apply_mask = torch.rand((batch_size,), generator=gen, device="cpu") < noise_prob_value
        if bool(apply_mask.any()):
            snr_values = torch.tensor(snr_choices, dtype=torch.float32, device="cpu")
            snr_idx = torch.randint(
                low=0,
                high=int(len(snr_choices)),
                size=(batch_size,),
                generator=gen,
                device="cpu",
            )
            chosen_snr = snr_values[snr_idx].unsqueeze(1)
            signal_power = amp_cpu.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-12)
            noise_power = signal_power / torch.pow(torch.tensor(10.0, dtype=torch.float32), chosen_snr / 10.0)
            sigma = noise_power.sqrt()
            noise = torch.randn(amp_cpu.shape, generator=gen, device="cpu") * sigma
            amp_cpu[apply_mask] = amp_cpu[apply_mask] + noise[apply_mask]
            out[:, 0, :] = amp_cpu.to(device=out.device, dtype=out.dtype)

    if use_rdrop:
        drop_mask = torch.rand((batch_size,), generator=gen, device="cpu") < rdrop_prob_value
        if bool(drop_mask.any()):
            out[drop_mask.to(device=out.device), 1:2, :] = 0.0

    return out
