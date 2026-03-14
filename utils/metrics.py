from __future__ import annotations

import numpy as np
from scipy import ndimage


def _validate_metric_image_pair(
    reference: np.ndarray,
    distorted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ref = np.asarray(reference, dtype=np.float32)
    dist = np.asarray(distorted, dtype=np.float32)
    if ref.shape != dist.shape:
        raise ValueError(f"reference/distorted shape mismatch: {ref.shape} vs {dist.shape}")
    if ref.ndim != 2:
        raise ValueError(f"VIF expects 2D inputs, got ndim={ref.ndim}")
    return ref, dist


def compute_vif_mscale(
    reference: np.ndarray,
    distorted: np.ndarray,
    sigma_nsq: float = 2.0,
    eps: float = 1e-10,
) -> float:
    """
    Compute the pixel-domain multi-scale VIF score.

    This follows the standard four-scale implementation commonly used in
    image-quality toolkits. Inputs must be 2D arrays with identical shape.
    """
    ref, dist = _validate_metric_image_pair(reference, distorted)
    ref_scale = ref
    dist_scale = dist
    num = 0.0
    den = 0.0

    for scale in range(1, 5):
        kernel_size = 2 ** (5 - scale) + 1  # 17, 9, 5, 3
        sigma = kernel_size / 5.0

        if scale > 1:
            ref_scale = ndimage.gaussian_filter(ref_scale, sigma=sigma, mode="reflect")[::2, ::2]
            dist_scale = ndimage.gaussian_filter(dist_scale, sigma=sigma, mode="reflect")[::2, ::2]

        mu_ref = ndimage.gaussian_filter(ref_scale, sigma=sigma, mode="reflect")
        mu_dist = ndimage.gaussian_filter(dist_scale, sigma=sigma, mode="reflect")

        mu_ref_sq = mu_ref * mu_ref
        mu_dist_sq = mu_dist * mu_dist
        mu_ref_dist = mu_ref * mu_dist

        sigma_ref_sq = ndimage.gaussian_filter(ref_scale * ref_scale, sigma=sigma, mode="reflect") - mu_ref_sq
        sigma_dist_sq = ndimage.gaussian_filter(dist_scale * dist_scale, sigma=sigma, mode="reflect") - mu_dist_sq
        sigma_ref_dist = ndimage.gaussian_filter(ref_scale * dist_scale, sigma=sigma, mode="reflect") - mu_ref_dist

        np.maximum(sigma_ref_sq, 0.0, out=sigma_ref_sq)
        np.maximum(sigma_dist_sq, 0.0, out=sigma_dist_sq)

        with np.errstate(divide="ignore", invalid="ignore"):
            gain = sigma_ref_dist / (sigma_ref_sq + eps)
        sv_sq = sigma_dist_sq - gain * sigma_ref_dist

        mask_ref_flat = sigma_ref_sq < eps
        gain[mask_ref_flat] = 0.0
        sv_sq[mask_ref_flat] = sigma_dist_sq[mask_ref_flat]
        sigma_ref_sq[mask_ref_flat] = 0.0

        mask_dist_flat = sigma_dist_sq < eps
        gain[mask_dist_flat] = 0.0
        sv_sq[mask_dist_flat] = 0.0

        mask_gain_neg = gain < 0.0
        gain[mask_gain_neg] = 0.0
        sv_sq[mask_gain_neg] = sigma_dist_sq[mask_gain_neg]

        sv_sq[sv_sq <= eps] = eps

        vif_num = np.log1p((gain * gain) * sigma_ref_sq / (sv_sq + sigma_nsq))
        vif_den = np.log1p(sigma_ref_sq / sigma_nsq)
        num += float(np.sum(vif_num, dtype=np.float64))
        den += float(np.sum(vif_den, dtype=np.float64))

    if den <= eps:
        return 1.0 if np.allclose(ref, dist, rtol=1e-5, atol=1e-6) else 0.0
    return float(num / den)

