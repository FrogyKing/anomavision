"""Publication-quality visualization and heatmap rendering utilities."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve

from anomavision.schemas import AblationPoint, CategoryEvaluationReport

matplotlib.use("Agg")


def render_heatmap_rgb(
    anomaly_map: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap_name: str = "inferno",
) -> np.ndarray:
    """Convert a 2D float anomaly map into an RGB uint8 heatmap (H, W, 3)."""
    low = float(np.min(anomaly_map)) if vmin is None else float(vmin)
    high = float(np.max(anomaly_map)) if vmax is None else float(vmax)
    if high <= low + 1e-8:
        normed = np.zeros_like(anomaly_map, dtype=np.float32)
    else:
        normed = np.clip((anomaly_map - low) / (high - low), 0.0, 1.0)

    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(normed)
    rgb = (rgba[:, :, :3] * 255.0).astype(np.uint8)
    return rgb


def render_overlay_rgb(
    image_rgb: np.ndarray,
    anomaly_map: np.ndarray,
    pixel_threshold: float,
    alpha: float = 0.45,
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend anomaly heatmap over original RGB image and highlight supra-threshold defect regions."""
    heatmap_rgb = render_heatmap_rgb(anomaly_map, vmin=vmin, vmax=vmax)
    blended = ((1.0 - alpha) * image_rgb.astype(np.float32) + alpha * heatmap_rgb.astype(np.float32)).astype(
        np.uint8
    )

    binary_mask = (anomaly_map >= pixel_threshold).astype(np.uint8)
    # Draw crisp cyan/red boundary contour around predicted defect mask
    if np.any(binary_mask > 0):
        padded = np.pad(binary_mask, 1, mode="constant", constant_values=0)
        eroded = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
        boundary = (binary_mask > 0) & (eroded == 0)
        blended[boundary] = np.array([0, 255, 212], dtype=np.uint8)

    return blended, binary_mask * 255


def encode_rgb_to_base64_png(rgb_or_gray: np.ndarray) -> str:
    """Encode an RGB or grayscale uint8 numpy array to a base64 PNG string."""
    pil_img = Image.fromarray(rgb_or_gray.astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def create_inspection_panel_figure(
    image_rgb: np.ndarray,
    gt_mask: np.ndarray | None,
    anomaly_map: np.ndarray,
    sam_mask: np.ndarray | None,
    pixel_threshold: float,
    title: str,
    out_path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Save a multi-panel diagnostic figure comparing RGB, GT mask, heatmap, overlay, and SAM mask."""
    overlay_rgb, pred_mask_255 = render_overlay_rgb(
        image_rgb=image_rgb,
        anomaly_map=anomaly_map,
        pixel_threshold=pixel_threshold,
        vmin=vmin,
        vmax=vmax,
    )
    heatmap_rgb = render_heatmap_rgb(anomaly_map, vmin=vmin, vmax=vmax)

    n_cols = 5 if sam_mask is not None else 4
    fig, axes = plt.subplots(1, n_cols, figsize=(3.4 * n_cols, 3.8), dpi=160)
    fig.patch.set_facecolor("#0f172a")

    panels = [
        ("Input RGB", image_rgb, None),
        (
            "Ground Truth Mask",
            (gt_mask * 255).astype(np.uint8) if gt_mask is not None else np.zeros(image_rgb.shape[:2], dtype=np.uint8),
            "gray",
        ),
        ("DINOv3 PatchCore Heatmap", heatmap_rgb, None),
        ("Calibrated Defect Overlay", overlay_rgb, None),
    ]
    if sam_mask is not None:
        sam_vis = (image_rgb.astype(np.float32) * (0.25 + 0.75 * (sam_mask[:, :, None] > 0))).astype(np.uint8)
        panels.append(("SAM 3 Foreground Gate", sam_vis, None))

    for ax, (panel_title, arr, cmap) in zip(axes, panels, strict=True):
        if cmap is not None:
            ax.imshow(arr, cmap=cmap, vmin=0, vmax=255)
        else:
            ax.imshow(arr)
        ax.set_title(panel_title, color="#f8fafc", fontsize=10, fontweight="bold", pad=6)
        ax.axis("off")

    fig.suptitle(title, color="#38bdf8", fontsize=11.5, fontweight="bold", y=0.98)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def create_evaluation_summary_plot(
    category: str,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
    baseline_scores: np.ndarray,
    calibration_scores: np.ndarray,
    threshold: float,
    report: CategoryEvaluationReport,
    out_path: Path,
) -> None:
    """Generate a publication-grade 2-panel ROC + Score Distribution figure for a category."""
    fig, (ax_roc, ax_dist) = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=180)
    fig.patch.set_facecolor("#0f172a")

    for ax in (ax_roc, ax_dist):
        ax.set_facecolor("#1e293b")
        for spine in ax.spines.values():
            spine.set_color("#475569")
        ax.tick_params(colors="#cbd5e1")
        ax.xaxis.label.set_color("#e2e8f0")
        ax.yaxis.label.set_color("#e2e8f0")
        ax.title.set_color("#f8fafc")

    fpr_main, tpr_main, _ = roc_curve(test_labels, test_scores)
    fpr_base, tpr_base, _ = roc_curve(test_labels, baseline_scores)

    ax_roc.plot(
        fpr_main,
        tpr_main,
        color="#38bdf8",
        linewidth=2.4,
        label=f"DINOv3 + PatchCore (AUROC = {report.image_auroc.value:.4f})",
    )
    ax_roc.plot(
        fpr_base,
        tpr_base,
        color="#94a3b8",
        linestyle="--",
        linewidth=1.8,
        label=f"Global Centroid Baseline (AUROC = {report.baseline_image_auroc:.4f})",
    )
    ax_roc.plot([0, 1], [0, 1], color="#475569", linestyle=":", linewidth=1.0)
    ax_roc.set_xlabel("False Positive Rate (FPR)")
    ax_roc.set_ylabel("True Positive Rate (TPR)")
    ax_roc.set_title(f"MVTec AD '{category}' — Image-Level ROC Curve", fontweight="bold")
    ax_roc.legend(loc="lower right", facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc")
    ax_roc.grid(True, color="#334155", linestyle="--", alpha=0.5)

    normal_test = test_scores[test_labels == 0]
    anom_test = test_scores[test_labels == 1]

    bins = np.linspace(
        min(float(np.min(calibration_scores)), float(np.min(test_scores))) * 0.9,
        float(np.max(test_scores)) * 1.05,
        28,
    )
    ax_dist.hist(
        calibration_scores,
        bins=bins,
        alpha=0.55,
        color="#22c55e",
        label=f"Normal Calibration (n={len(calibration_scores)})",
        density=True,
    )
    ax_dist.hist(
        normal_test,
        bins=bins,
        alpha=0.60,
        color="#38bdf8",
        label=f"Normal Test (n={len(normal_test)})",
        density=True,
    )
    ax_dist.hist(
        anom_test,
        bins=bins,
        alpha=0.65,
        color="#f43f5e",
        label=f"Anomalous Test (n={len(anom_test)})",
        density=True,
    )
    ax_dist.axvline(
        threshold,
        color="#facc15",
        linestyle="--",
        linewidth=2.2,
        label=f"Unsupervised Threshold = {threshold:.4f}",
    )
    ax_dist.set_xlabel("PatchCore Anomaly Score S(X)")
    ax_dist.set_ylabel("Empirical Density")
    ax_dist.set_title(f"MVTec AD '{category}' — Score Separation & Threshold", fontweight="bold")
    ax_dist.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8.5)
    ax_dist.grid(True, color="#334155", linestyle="--", alpha=0.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def create_ablation_plot(ablation_points: list[AblationPoint], out_path: Path) -> None:
    """Render ablation comparison across coreset ratios, distance metrics, and normalization."""
    if not ablation_points:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5), dpi=180)
    fig.patch.set_facecolor("#0f172a")

    for ax in (ax1, ax2):
        ax.set_facecolor("#1e293b")
        for spine in ax.spines.values():
            spine.set_color("#475569")
        ax.tick_params(colors="#cbd5e1")
        ax.xaxis.label.set_color("#e2e8f0")
        ax.yaxis.label.set_color("#e2e8f0")
        ax.title.set_color("#f8fafc")

    labels = [
        f"{p.metric[:3].upper()}-{'L2' if p.l2_normalize else 'Raw'}\n({int(p.coreset_ratio * 100)}%)"
        for p in ablation_points
    ]
    x = np.arange(len(ablation_points))
    width = 0.26

    img_auroc = [p.image_auroc for p in ablation_points]
    pix_auroc = [p.pixel_auroc for p in ablation_points]
    aupro = [p.aupro for p in ablation_points]

    ax1.bar(x - width, img_auroc, width, label="Image AUROC", color="#38bdf8")
    ax1.bar(x, pix_auroc, width, label="Pixel AUROC", color="#818cf8")
    ax1.bar(x + width, aupro, width, label="AUPRO", color="#34d399")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylim(max(0.5, min(aupro + img_auroc) - 0.08), 1.01)
    ax1.set_ylabel("Score")
    ax1.set_title("Detection & Localization Quality Across Configurations", fontweight="bold")
    ax1.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8.5)
    ax1.grid(True, axis="y", color="#334155", linestyle="--", alpha=0.5)

    mem_mb = [p.memory_bank_size_mb for p in ablation_points]
    lat_ms = [p.retrieval_latency_ms for p in ablation_points]

    ax2.bar(x - width / 2, mem_mb, width, label="Memory Bank (MB)", color="#f59e0b")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x, lat_ms, color="#f43f5e", marker="o", linewidth=2.2, label="k-NN Latency (ms)")
    ax2_twin.tick_params(colors="#f43f5e")
    ax2_twin.yaxis.label.set_color("#f43f5e")
    ax2_twin.set_ylabel("Patch k-NN Search Latency (ms)")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Coreset Memory Footprint (MB)")
    ax2.set_title("Memory Footprint vs Search Latency Tradeoff", fontweight="bold")
    ax2.grid(True, axis="y", color="#334155", linestyle="--", alpha=0.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
