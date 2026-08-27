"""Unit tests for evaluation metrics, AUPRO integration, unsupervised calibration, and visualization."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from anomavision.evaluation.calibration import fit_unsupervised_calibration, normalize_anomaly_score
from anomavision.evaluation.metrics import (
    compute_aupro,
    compute_binary_classification_metrics,
    compute_bootstrap_auroc,
    compute_per_defect_auroc,
    compute_pixel_auroc_with_ci,
)
from anomavision.evaluation.visualization import (
    create_inspection_panel_figure,
    encode_rgb_to_base64_png,
    render_heatmap_rgb,
)


def test_unsupervised_calibration_and_normalization() -> None:
    rng = np.random.default_rng(42)
    cal_scores = rng.normal(loc=0.20, scale=0.02, size=30).astype(np.float32)
    cal_maps = [rng.normal(loc=0.15, scale=0.02, size=(32, 32)).astype(np.float32) for _ in range(5)]

    params = fit_unsupervised_calibration("bottle", cal_scores, cal_maps)
    assert params.is_purely_unsupervised is True
    assert params.image_threshold > float(np.mean(cal_scores))
    assert params.pixel_threshold > 0.15

    # Score equal to threshold maps to exactly 0.5
    assert np.isclose(normalize_anomaly_score(params.image_threshold, params), 0.5, atol=1e-5)
    assert normalize_anomaly_score(params.image_threshold + 0.5, params) > 0.85


def test_metrics_auroc_aupro_and_f1() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
    scores = np.array([0.10, 0.12, 0.15, 0.18, 0.45, 0.52, 0.60, 0.75], dtype=np.float32)

    img_auroc = compute_bootstrap_auroc(labels, scores, n_bootstrap=200, seed=42)
    assert np.isclose(img_auroc.value, 1.0)
    assert img_auroc.ci_lower >= 0.95

    bin_metrics = compute_binary_classification_metrics(labels, scores, threshold=0.25)
    assert np.isclose(bin_metrics["precision"], 1.0)
    assert np.isclose(bin_metrics["recall"], 1.0)
    assert np.isclose(bin_metrics["f1"], 1.0)

    defects = ["good", "good", "good", "good", "broken_large", "broken_large", "broken_small", "broken_small"]
    per_def = compute_per_defect_auroc(defects, scores)
    assert np.isclose(per_def["broken_large"], 1.0)
    assert np.isclose(per_def["broken_small"], 1.0)

    # Pixel AUROC and AUPRO check
    gt_masks = [np.zeros((32, 32), dtype=np.uint8) for _ in range(4)]
    for _ in range(4):
        m = np.zeros((32, 32), dtype=np.uint8)
        m[10:20, 10:20] = 1
        gt_masks.append(m)

    amaps: list[np.ndarray] = []
    for m in gt_masks:
        amap = np.ones((32, 32), dtype=np.float32) * 0.1
        amap[m == 1] = 0.9
        amaps.append(amap)

    pix_auroc = compute_pixel_auroc_with_ci(gt_masks, amaps, n_bootstrap=25, seed=42)
    assert pix_auroc.value > 0.99

    aupro = compute_aupro(gt_masks, amaps)
    assert aupro > 0.90


def test_visualization_panel_rendering(tmp_path: Path) -> None:
    rgb = np.zeros((48, 48, 3), dtype=np.uint8)
    rgb[12:36, 12:36] = 180
    gt = np.zeros((48, 48), dtype=np.uint8)
    gt[20:28, 20:28] = 1
    amap = np.random.default_rng(1).uniform(0.1, 0.3, size=(48, 48)).astype(np.float32)
    amap[20:28, 20:28] = 0.85
    sam_mask = np.zeros((48, 48), dtype=np.uint8)
    sam_mask[10:38, 10:38] = 1

    heatmap = render_heatmap_rgb(amap)
    assert heatmap.shape == (48, 48, 3)
    b64 = encode_rgb_to_base64_png(heatmap)
    assert len(b64) > 50

    out_file = tmp_path / "panel.png"
    create_inspection_panel_figure(
        image_rgb=rgb,
        gt_mask=gt,
        anomaly_map=amap,
        sam_mask=sam_mask,
        pixel_threshold=0.5,
        title="Test Inspection Panel",
        out_path=out_file,
    )
    assert out_file.is_file()
