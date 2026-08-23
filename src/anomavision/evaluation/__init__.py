"""Statistical evaluation metrics, unsupervised calibration, and visualization tools."""

from anomavision.evaluation.calibration import fit_unsupervised_calibration, normalize_anomaly_score
from anomavision.evaluation.metrics import (
    compute_aupro,
    compute_binary_classification_metrics,
    compute_bootstrap_auroc,
    compute_per_defect_auroc,
     compute_pixel_auroc_with_ci,
)
from anomavision.evaluation.visualization import (
    create_ablation_plot,
    create_evaluation_summary_plot,
    create_inspection_panel_figure,
    encode_rgb_to_base64_png,
    render_heatmap_rgb,
    render_overlay_rgb,
)

__all__ = [
    "compute_aupro",
    "compute_binary_classification_metrics",
    "compute_bootstrap_auroc",
    "compute_per_defect_auroc",
    "compute_pixel_auroc_with_ci",
    "create_ablation_plot",
    "create_evaluation_summary_plot",
    "create_inspection_panel_figure",
    "encode_rgb_to_base64_png",
    "fit_unsupervised_calibration",
    "normalize_anomaly_score",
    "render_heatmap_rgb",
    "render_overlay_rgb",
]
