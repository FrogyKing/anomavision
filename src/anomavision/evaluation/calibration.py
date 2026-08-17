"""Unsupervised threshold calibration fitted strictly on held-out normal calibration images."""

from __future__ import annotations

import numpy as np

from anomavision.schemas import CalibrationParams


def fit_unsupervised_calibration(
    category: str,
    calibration_image_scores: np.ndarray,
    calibration_anomaly_maps: list[np.ndarray],
    reference_centroid_norm: float = 1.0,
    reference_mmd_baseline: float = 0.01,
    quantile_alpha: float = 0.98,
) -> CalibrationParams:
    """Fit image and pixel anomaly thresholds exclusively on normal calibration data.

    Anti-Leakage Design:
      Because the calibration split consists solely of held-out normal images (never stored in the
      coreset memory bank M), its nearest-neighbor distances reflect the genuine out-of-sample normal
      score distribution. No anomalous test images or test labels are used to select this threshold.
      We combine the empirical quantile Q_alpha and a parametric Gaussian tail bound (mu + 2.5 * sigma)
      to control false-alarm rates under finite calibration sample sizes.
    """
    scores = np.asarray(calibration_image_scores, dtype=np.float64)
    if len(scores) == 0:
        raise ValueError("calibration_image_scores cannot be empty")

    mean_val = float(np.mean(scores))
    std_val = float(np.std(scores, ddof=1)) if len(scores) > 1 else max(0.02, mean_val * 0.1)
    min_val = float(np.min(scores))
    max_val = float(np.max(scores))

    q_val = float(np.quantile(scores, quantile_alpha))
    gaussian_tail = mean_val + 2.5 * std_val
    image_threshold = float(max(q_val, gaussian_tail))

    flat_pixels = np.concatenate([m.ravel() for m in calibration_anomaly_maps]).astype(np.float64)
    pix_q = float(np.quantile(flat_pixels, 0.995))
    pix_mean = float(np.mean(flat_pixels))
    pix_std = float(np.std(flat_pixels))
    pixel_threshold = float(max(pix_q, pix_mean + 3.0 * pix_std))

    return CalibrationParams(
        category=category,
        method="unsupervised_quantile",
        is_purely_unsupervised=True,
        quantile_alpha=quantile_alpha,
        image_threshold=image_threshold,
        pixel_threshold=pixel_threshold,
        cal_score_mean=mean_val,
        cal_score_std=std_val,
        cal_score_min=min_val,
        cal_score_max=max_val,
        reference_centroid_norm=reference_centroid_norm,
        reference_mmd_baseline=reference_mmd_baseline,
    )


def normalize_anomaly_score(raw_score: float, cal: CalibrationParams) -> float:
    """Map raw distance score onto an interpretable [0, 1] scale centered at 0.5 = threshold."""
    scale = max(cal.cal_score_std * 2.5, (cal.image_threshold - cal.cal_score_mean), 1e-4)
    z = (raw_score - cal.image_threshold) / scale
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0))))
