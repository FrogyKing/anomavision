"""Distributional embedding drift monitoring via multi-bandwidth Radial Basis Function MMD."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def compute_rbf_mmd_squared(
    x: np.ndarray,
    y: np.ndarray,
    bandwidths: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0),
) -> float:
    """Compute unbiased/empirical Maximum Mean Discrepancy squared (MMD^2) with multi-RBF kernel.

    Given reference sample X = {x_i}_{i=1}^m and query sample Y = {y_j}_{j=1}^n in R^d:
        k(u, v) = (1 / |Sigma|) * sum_{sigma in Sigma} exp(-||u - v||_2^2 / (2 * sigma^2))
        MMD^2(X, Y) = E[k(x, x')] + E[k(y, y')] - 2 E[k(x, y)]
    Returns a non-negative scalar drift statistic.
    """
    x_arr = np.asarray(x, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(1, -1)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(1, -1)

    x_sq = np.sum(x_arr * x_arr, axis=1, keepdims=True)
    y_sq = np.sum(y_arr * y_arr, axis=1, keepdims=True)

    d_xx = np.maximum(0.0, x_sq + x_sq.T - 2.0 * (x_arr @ x_arr.T))
    d_yy = np.maximum(0.0, y_sq + y_sq.T - 2.0 * (y_arr @ y_arr.T))
    d_xy = np.maximum(0.0, x_sq + y_sq.T - 2.0 * (x_arr @ y_arr.T))

    k_xx = np.zeros_like(d_xx, dtype=np.float32)
    k_yy = np.zeros_like(d_yy, dtype=np.float32)
    k_xy = np.zeros_like(d_xy, dtype=np.float32)

    for sigma in bandwidths:
        gamma = 1.0 / (2.0 * sigma * sigma)
        k_xx += np.exp(-gamma * d_xx)
        k_yy += np.exp(-gamma * d_yy)
        k_xy += np.exp(-gamma * d_xy)

    num_bw = float(len(bandwidths))
    k_xx /= num_bw
    k_yy /= num_bw
    k_xy /= num_bw

    mmd2 = float(np.mean(k_xx) + np.mean(k_yy) - 2.0 * np.mean(k_xy))
    return max(0.0, mmd2)


@dataclass
class DriftReferenceStats:
    """Reference normal patch distribution summary for real-time embedding drift telemetry."""

    reference_anchors: np.ndarray  # Subsampled normal reference patches (M_ref, d)
    baseline_mmd_mean: float
    baseline_mmd_std: float
    alert_threshold: float

    @classmethod
    def from_reference_patches(
        cls,
        train_patches: np.ndarray,
        calibration_patches_by_image: np.ndarray,
        max_anchors: int = 512,
        seed: int = 42,
    ) -> DriftReferenceStats:
        """Fit drift baseline using normal train anchors and held-out normal calibration images."""
        rng = np.random.default_rng(seed)
        total = len(train_patches)
        n_anchors = min(max_anchors, total)
        idx = rng.choice(total, size=n_anchors, replace=False)
        anchors = train_patches[idx].astype(np.float32, copy=True)

        cal_mmds: list[float] = []
        for img_patches in calibration_patches_by_image:
            # Subsample 128 patches per image for fast <1ms drift check
            if len(img_patches) > 128:
                sub_idx = rng.choice(len(img_patches), size=128, replace=False)
                sample_p = img_patches[sub_idx]
            else:
                sample_p = img_patches
            cal_mmds.append(compute_rbf_mmd_squared(anchors, sample_p))

        mean_val = float(np.mean(cal_mmds)) if cal_mmds else 0.01
        std_val = float(np.std(cal_mmds)) if len(cal_mmds) > 1 else 0.005
        # Drift alert threshold: mean + 4 * std (separates covariate shift from single-patch defects)
        alert_thresh = mean_val + 4.0 * max(std_val, 1e-3)
        return cls(
            reference_anchors=anchors,
            baseline_mmd_mean=mean_val,
            baseline_mmd_std=std_val,
            alert_threshold=alert_thresh,
        )

    def score_query_patches(self, query_patches: np.ndarray) -> tuple[float, bool]:
        """Compute MMD^2 of a query image's patch cloud against normal reference anchors."""
        step = max(1, len(query_patches) // 128)
        sub_query = query_patches[::step]
        mmd_val = compute_rbf_mmd_squared(self.reference_anchors, sub_query)
        return mmd_val, bool(mmd_val > self.alert_threshold)
