"""Quantitative industrial anomaly detection metrics with bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label as connected_components
from sklearn.metrics import precision_recall_curve, roc_auc_score

from anomavision.schemas import MetricWithCI


def compute_bootstrap_auroc(
    labels: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> MetricWithCI:
    """Compute ROC-AUC and stratified non-parametric 95% bootstrap confidence interval."""
    y_true = np.asarray(labels, dtype=np.int32)
    y_score = np.asarray(scores, dtype=np.float64)

    if len(np.unique(y_true)) < 2:
        raise ValueError("Both normal (0) and anomalous (1) samples are required to compute AUROC")

    point_est = float(roc_auc_score(y_true, y_score))
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    rng = np.random.default_rng(seed)
    boot_vals: list[float] = []
    for _ in range(n_bootstrap):
        b_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        b_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([b_pos, b_neg])
        boot_vals.append(float(roc_auc_score(y_true[idx], y_score[idx])))

    arr = np.asarray(boot_vals, dtype=np.float64)
    ci_low = float(np.percentile(arr, 2.5))
    ci_high = float(np.percentile(arr, 97.5))
    std_err = float(np.std(arr, ddof=1))
    return MetricWithCI(
        value=point_est,
        ci_lower=ci_low,
        ci_upper=ci_high,
        std_error=std_err,
    )


def compute_pixel_auroc_with_ci(
    gt_masks: list[np.ndarray],
    anomaly_maps: list[np.ndarray],
    n_bootstrap: int = 100,
    subsample_pixels_per_image: int = 4096,
    seed: int = 42,
) -> MetricWithCI:
    """Compute pixel-level AUROC with image-cluster bootstrap 95% confidence interval."""
    if len(gt_masks) != len(anomaly_maps):
        raise ValueError("gt_masks and anomaly_maps must have the same number of images")

    rng = np.random.default_rng(seed)
    sub_labels: list[np.ndarray] = []
    sub_scores: list[np.ndarray] = []

    for mask, amap in zip(gt_masks, anomaly_maps, strict=True):
        flat_m = (mask.ravel() > 0).astype(np.int32)
        flat_a = amap.ravel().astype(np.float32)
        if len(flat_m) > subsample_pixels_per_image:
            # Stratified pixel subsampling within each image to preserve exact defect boundaries
            pos_p = np.where(flat_m == 1)[0]
            neg_p = np.where(flat_m == 0)[0]
            if len(pos_p) > 0 and len(neg_p) > 0:
                n_pos = max(1, min(len(pos_p), subsample_pixels_per_image // 2))
                n_neg = subsample_pixels_per_image - n_pos
                sel_pos = rng.choice(pos_p, size=n_pos, replace=len(pos_p) < n_pos)
                sel_neg = rng.choice(neg_p, size=n_neg, replace=len(neg_p) < n_neg)
                sel = np.concatenate([sel_pos, sel_neg])
            else:
                sel = rng.choice(len(flat_m), size=subsample_pixels_per_image, replace=False)
            sub_labels.append(flat_m[sel])
            sub_scores.append(flat_a[sel])
        else:
            sub_labels.append(flat_m)
            sub_scores.append(flat_a)

    all_y = np.concatenate(sub_labels)
    all_s = np.concatenate(sub_scores)
    point_est = float(roc_auc_score(all_y, all_s))

    # Cluster bootstrap at the image level to respect intra-image spatial correlation
    has_defect = np.array([int(np.any(m > 0)) for m in sub_labels])
    pos_imgs = np.where(has_defect == 1)[0]
    neg_imgs = np.where(has_defect == 0)[0]

    boot_vals: list[float] = []
    if len(pos_imgs) > 0 and len(neg_imgs) > 0:
        for _ in range(n_bootstrap):
            b_pos = rng.choice(pos_imgs, size=len(pos_imgs), replace=True)
            b_neg = rng.choice(neg_imgs, size=len(neg_imgs), replace=True)
            img_indices = np.concatenate([b_pos, b_neg])
            by = np.concatenate([sub_labels[i] for i in img_indices])
            bs = np.concatenate([sub_scores[i] for i in img_indices])
            if len(np.unique(by)) == 2:
                boot_vals.append(float(roc_auc_score(by, bs)))

    if not boot_vals:
        return MetricWithCI(value=point_est, ci_lower=point_est, ci_upper=point_est, std_error=0.0)

    arr = np.asarray(boot_vals, dtype=np.float64)
    return MetricWithCI(
        value=point_est,
        ci_lower=float(np.percentile(arr, 2.5)),
        ci_upper=float(np.percentile(arr, 97.5)),
        std_error=float(np.std(arr, ddof=1)),
    )


def compute_aupro(
    gt_masks: list[np.ndarray],
    anomaly_maps: list[np.ndarray],
    max_fpr: float = 0.30,
    num_thresholds: int = 120,
) -> float:
    """Compute Area Under the Per-Region Overlap (AUPRO) curve integrated up to FPR <= 0.30.

    Reference:
        Bergmann et al., "The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset
        for Unsupervised Anomaly Detection", IJCV 2021.
    """
    if len(gt_masks) != len(anomaly_maps):
        raise ValueError("gt_masks and anomaly_maps length mismatch")

    # Identify connected defect components across all ground-truth masks
    component_slices: list[tuple[int, np.ndarray, int]] = []
    bg_scores_list: list[np.ndarray] = []
    all_min = float("inf")
    all_max = float("-inf")

    for img_idx, (mask, amap) in enumerate(zip(gt_masks, anomaly_maps, strict=True)):
        bin_mask = (mask > 0).astype(np.int32)
        all_min = min(all_min, float(np.min(amap)))
        all_max = max(all_max, float(np.max(amap)))

        bg_scores_list.append(amap[bin_mask == 0].ravel())
        if np.any(bin_mask > 0):
            labeled, n_comp = connected_components(bin_mask)
            for comp_id in range(1, n_comp + 1):
                comp_pixels = amap[labeled == comp_id].ravel()
                if len(comp_pixels) > 0:
                    component_slices.append((img_idx, comp_pixels, len(comp_pixels)))

    if not component_slices or all_max <= all_min:
        return 0.0

    bg_scores = np.concatenate(bg_scores_list)
    total_bg = float(len(bg_scores))
    if total_bg == 0:
        return 0.0

    # Compute exact cumulative exceedances across num_thresholds bins in O(N_pixels) single pass
    bin_edges = np.linspace(all_min, all_max + 1e-7, num=num_thresholds + 1)
    bg_hist, _ = np.histogram(bg_scores, bins=bin_edges)
    # Reverse cumulative sum: count of background pixels >= bin_edges[t]
    fprs = np.cumsum(bg_hist[::-1])[::-1].astype(np.float64) / total_bg

    comp_pro_accum = np.zeros(num_thresholds, dtype=np.float64)
    for _, comp_scores, comp_area in component_slices:
        c_hist, _ = np.histogram(comp_scores, bins=bin_edges)
        c_exceed = np.cumsum(c_hist[::-1])[::-1].astype(np.float64) / float(comp_area)
        comp_pro_accum += c_exceed
    pros = comp_pro_accum / float(len(component_slices))

    # Sort by increasing FPR and integrate up to max_fpr
    order = np.argsort(fprs)
    fprs = fprs[order]
    pros = pros[order]

    valid = fprs <= max_fpr
    if np.sum(valid) < 2:
        return float(np.clip(pros[0], 0.0, 1.0))

    fpr_clipped = np.concatenate([[0.0], fprs[valid], [max_fpr]])
    last_pro = float(np.interp(max_fpr, fprs, pros))
    pro_clipped = np.concatenate([[0.0], pros[valid], [last_pro]])

    area = float(np.trapezoid(pro_clipped, fpr_clipped))
    return float(np.clip(area / max_fpr, 0.0, 1.0))


def compute_binary_classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute Precision, Recall, F1 at the calibrated threshold, plus oracle post-hoc F1 upper bound."""
    y_true = np.asarray(labels, dtype=np.int32)
    y_score = np.asarray(scores, dtype=np.float64)
    y_pred = (y_score >= threshold).astype(np.int32)

    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_score)
    denom = prec_curve + rec_curve
    f1_curve = np.where(denom > 0, 2.0 * prec_curve * rec_curve / np.maximum(denom, 1e-12), 0.0)
    f1_oracle = float(np.max(f1_curve))

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f1_oracle_upper_bound": f1_oracle,
    }


def compute_per_defect_auroc(
    defect_types: list[str],
    scores: np.ndarray,
) -> dict[str, float]:
    """Compute image-level AUROC for each individual defect class versus normal ('good') test samples."""
    scores_arr = np.asarray(scores, dtype=np.float64)
    good_mask = np.array([d == "good" for d in defect_types])
    good_scores = scores_arr[good_mask]

    unique_defects = sorted({d for d in defect_types if d != "good"})
    results: dict[str, float] = {}
    for defect in unique_defects:
        def_mask = np.array([d == defect for d in defect_types])
        def_scores = scores_arr[def_mask]
        if len(good_scores) > 0 and len(def_scores) > 0:
            y = np.concatenate([np.zeros(len(good_scores), dtype=np.int32), np.ones(len(def_scores), dtype=np.int32)])
            s = np.concatenate([good_scores, def_scores])
            results[defect] = float(roc_auc_score(y, s))
    return results
