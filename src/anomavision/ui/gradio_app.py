"""Gradio industrial optical inspection console for AnomaVision."""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import numpy as np
from PIL import Image

from anomavision import __version__
from anomavision.pipeline.inference import InferenceEngine

INDUSTRIAL_CSS = """
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.inspection-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 18px 24px;
    margin-bottom: 14px;
    color: #f8fafc;
}
.decision-card-pass {
    background: rgba(16, 185, 129, 0.12);
    border: 2px solid #10b981;
    border-radius: 8px;
    padding: 14px 18px;
    color: #ecfdf5;
}
.decision-card-fail {
    background: rgba(244, 63, 94, 0.14);
    border: 2px solid #f43f5e;
    border-radius: 8px;
    padding: 14px 18px;
    color: #fff1f2;
}
"""


def _decode_b64_to_numpy(b64_str: str | None) -> np.ndarray | None:
    if not b64_str:
        return None
    raw = base64.b64decode(b64_str)
    with Image.open(io.BytesIO(raw)) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _format_decision_banner(
    decision_label: str,
    raw_score: float,
    threshold: float,
    norm_score: float,
    drift_mmd: float,
    drift_alert: bool,
    inf_ms: float,
    ret_ms: float,
) -> str:
    is_anom = decision_label == "ANOMALOUS"
    css_class = "decision-card-fail" if is_anom else "decision-card-pass"
    badge_color = "#f43f5e" if is_anom else "#10b981"
    status_text = "REJECT — ANOMALY DETECTED" if is_anom else "PASS — WITHIN NORMAL SPEC"
    drift_text = "ALERT (Covariate Shift)" if drift_alert else "NOMINAL (In-Distribution)"

    return f"""
    <div class="{css_class}">
      <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
        <div>
          <span style="background:{badge_color}; color:#0f172a; font-weight:800; padding:4px 10px; border-radius:4px; font-size:0.85rem; letter-spacing:0.05em;">
            {status_text}
          </span>
          <span style="margin-left:12px; font-size:1.15rem; font-weight:700;">
            PatchCore Score: {raw_score:.4f}
          </span>
          <span style="margin-left:8px; font-size:0.92rem; opacity:0.85;">
            (Calibrated Threshold: {threshold:.4f} | Normalized: {norm_score:.3f})
          </span>
        </div>
        <div style="font-size:0.85rem; opacity:0.9;">
          Inference: <b>{inf_ms:.1f} ms</b> | FAISS Search: <b>{ret_ms:.2f} ms</b> |
          Drift MMD²: <b>{drift_mmd:.5f}</b> ({drift_text})
        </div>
      </div>
    </div>
    """


def run_ui_inspection(
    engine: InferenceEngine,
    image: np.ndarray | Image.Image | None,
    category: str,
    use_sam_refinement: bool,
    top_k_similar: int = 4,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    str,
    list[tuple[np.ndarray, str]],
    str,
]:
    """Execute inspection for the Gradio UI and return formatted visual and telemetry outputs."""
    if image is None:
        empty_banner = "<div style='padding:12px; border:1px solid #475569; border-radius:8px;'>Upload or select a component image to run inspection.</div>"
        return None, None, None, None, empty_banner, [], "{}"

    if isinstance(image, Image.Image):
        rgb_preview = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        rgb_preview = image[:, :, :3].astype(np.uint8)

    res = engine.predict(
        image=rgb_preview,
        category=category,
        use_sam_refinement=use_sam_refinement,
        top_k_similar=top_k_similar,
        include_visualizations=True,
    )

    heatmap_arr = _decode_b64_to_numpy(res.heatmap_base64_png)
    overlay_arr = _decode_b64_to_numpy(res.overlay_base64_png)
    sam_arr = _decode_b64_to_numpy(res.sam_mask_base64_png)
    if sam_arr is None:
        sam_arr = _decode_b64_to_numpy(res.mask_base64_png)

    banner_html = _format_decision_banner(
        decision_label=res.decision_label,
        raw_score=res.anomaly_score,
        threshold=res.threshold,
        norm_score=res.normalized_score,
        drift_mmd=res.embedding_drift_mmd,
        drift_alert=res.drift_alert,
        inf_ms=res.inference_latency_ms,
        ret_ms=res.retrieval_latency_ms,
    )

    gallery_items: list[tuple[np.ndarray, str]] = []
    runtime = engine.categories[category]
    thumbs_dir = runtime.bundle_dir / "reference_thumbnails"
    for neighbor in res.similar_references:
        fname = Path(neighbor.image_relpath).name
        candidate_path = thumbs_dir / fname
        if candidate_path.is_file():
            with Image.open(candidate_path) as ref_img:
                ref_arr = np.asarray(ref_img.convert("RGB"), dtype=np.uint8)
        else:
            # Fallback to first available reference thumbnail if exact ID wasn't in top-25 subset
            avail = sorted(thumbs_dir.glob("*.png")) if thumbs_dir.is_dir() else []
            if avail:
                with Image.open(avail[(neighbor.rank - 1) % len(avail)]) as ref_img:
                    ref_arr = np.asarray(ref_img.convert("RGB"), dtype=np.uint8)
            else:
                ref_arr = np.zeros((128, 128, 3), dtype=np.uint8)

        caption = f"#{neighbor.rank} {neighbor.sample_id} | Cosine Sim: {neighbor.similarity_score:.4f}"
        gallery_items.append((ref_arr, caption))

    telemetry_dict: dict[str, Any] = {
        "request_id": res.request_id,
        "category": res.category,
        "decision": res.decision_label,
        "anomaly_score": res.anomaly_score,
        "calibrated_threshold": res.threshold,
        "normalized_score": res.normalized_score,
        "confidence_margin": res.confidence_margin,
        "max_patch_score": res.max_patch_score,
        "mean_top_k_patch_score": res.mean_top_k_patch_score,
        "sam_object_coverage": res.sam_object_coverage,
        "embedding_drift_mmd2": res.embedding_drift_mmd,
        "drift_alert": res.drift_alert,
        "inference_latency_ms": res.inference_latency_ms,
        "retrieval_latency_ms": res.retrieval_latency_ms,
        "model_version": res.model_version,
        "backbone_model_id": engine.cfg.backbone.model_id,
        "backbone_revision": res.backbone_revision,
        "config_fingerprint": engine.cfg.fingerprint(),
    }

    return (
        rgb_preview,
        heatmap_arr,
        overlay_arr,
        sam_arr,
        banner_html,
        gallery_items,
        json.dumps(telemetry_dict, indent=2),
    )


def build_gradio_interface(
    engine: InferenceEngine | None = None,
    bundles_root: Path = Path("artifacts/bundles"),
) -> gr.Blocks:
    """Construct the multi-tab Gradio industrial inspection console."""
    if engine is None:
        engine = InferenceEngine(bundles_root=bundles_root, load_sam=False)

    categories = engine.available_categories or ["bottle"]
    default_cat = categories[0]

    # Collect real MVTec AD example images from packaged bundles
    example_rows: list[list[Any]] = []
    for cat in categories:
        ex_dir = bundles_root / cat / "examples"
        if ex_dir.is_dir():
            for png_path in sorted(ex_dir.glob("*.png"))[:4]:
                example_rows.append([str(png_path), cat, False, 4])

    with gr.Blocks(title="AnomaVision — Visual Anomaly Intelligence") as demo:
        gr.HTML(f"<style>{INDUSTRIAL_CSS}</style>")
        gr.HTML(
            f"""
            <div class="inspection-header">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                  <h2 style="margin:0; font-size:1.45rem; color:#38bdf8;">
                    AnomaVision — Industrial Optical Anomaly Intelligence
                  </h2>
                  <p style="margin:4px 0 0 0; font-size:0.92rem; color:#cbd5e1;">
                    DINOv3 ViT-S/16 Dense Patch Representations &bull; Greedy k-Center Coreset Memory &bull;
                    FAISS Reference Retrieval &bull; SAM 3 Foreground Gating
                  </p>
                </div>
                <div style="text-align:right; font-size:0.82rem; color:#94a3b8;">
                  <div>Version: <b>v{__version__}</b> | Fingerprint: <code>{engine.cfg.fingerprint()}</code></div>
                  <div>Backbone Rev: <code>{engine.cfg.backbone.revision[:10]}</code> | Grid: 28&times;28 (448px)</div>
                </div>
              </div>
            </div>
            """
        )

        with gr.Tabs():
            with gr.TabItem("Live Optical Inspection"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(
                            type="numpy",
                            label="Component Input Image (Upload or Select Example)",
                            height=290,
                        )
                        category_dropdown = gr.Dropdown(
                            choices=categories,
                            value=default_cat,
                            label="Inspection Category (MVTec AD Memory Bank)",
                        )
                        with gr.Row():
                            sam_checkbox = gr.Checkbox(
                                value=False,
                                label="Apply SAM 3 Object Mask Gating",
                            )
                            topk_slider = gr.Slider(
                                minimum=1,
                                maximum=8,
                                step=1,
                                value=4,
                                label="Top-K Normal References",
                            )
                        inspect_btn = gr.Button("Run Optical Inspection", variant="primary")

                    with gr.Column(scale=3):
                        decision_banner = gr.HTML(
                            "<div style='padding:12px; border:1px solid #334155; border-radius:8px;'>"
                            "Ready for inspection. Upload a component image or click an MVTec AD example below."
                            "</div>"
                        )
                        with gr.Row():
                            orig_preview = gr.Image(label="Normalized RGB Preview", height=250)
                            heatmap_view = gr.Image(label="DINOv3 PatchCore Anomaly Heatmap", height=250)
                            overlay_view = gr.Image(label="Calibrated Defect Contour Overlay", height=250)
                            mask_view = gr.Image(label="Binary Defect / SAM Gate Mask", height=250)

                with gr.Row():
                    with gr.Column(scale=2):
                        similar_gallery = gr.Gallery(
                            label="FAISS Nearest Normal Reference Components (train_fit Memory)",
                            columns=4,
                            height=230,
                            object_fit="contain",
                        )
                    with gr.Column(scale=1):
                        telemetry_code = gr.Code(
                            label="Structured Inspection & Drift Telemetry (JSON)",
                            language="json",
                            lines=11,
                        )

                if example_rows:
                    gr.Examples(
                        examples=example_rows,
                        inputs=[input_image, category_dropdown, sam_checkbox, topk_slider],
                        label="Authentic MVTec AD Held-Out Test Samples (bottle & metal_nut)",
                    )

                inspect_btn.click(
                    fn=lambda img, cat, sam_flag, topk: run_ui_inspection(
                        engine, img, cat, sam_flag, int(topk)
                    ),
                    inputs=[input_image, category_dropdown, sam_checkbox, topk_slider],
                    outputs=[
                        orig_preview,
                        heatmap_view,
                        overlay_view,
                        mask_view,
                        decision_banner,
                        similar_gallery,
                        telemetry_code,
                    ],
                )

            with gr.TabItem("Quantitative Benchmarks & Ablations"):
                for cat in categories:
                    rep = engine.categories[cat].evaluation_report
                    plots_dir = bundles_root / cat / "plots"
                    gr.Markdown(
                        f"### Category: `{cat}` — Image AUROC: **{rep.image_auroc.value:.4f}** "
                        f"(95% CI `[{rep.image_auroc.ci_lower:.4f}, {rep.image_auroc.ci_upper:.4f}]`) | "
                        f"Pixel AUROC: **{rep.pixel_auroc.value:.4f}** | "
                        f"AUPRO: **{rep.aupro:.4f}** | "
                        f"Unsupervised F1: **{rep.f1_at_threshold:.4f}**"
                    )
                    with gr.Row():
                        roc_plot = plots_dir / f"evaluation_summary_{cat}.png"
                        ab_plot = plots_dir / f"ablation_{cat}.png"
                        if roc_plot.is_file():
                            gr.Image(value=str(roc_plot), label=f"{cat} — ROC & Calibration Distribution")
                        if ab_plot.is_file():
                            gr.Image(value=str(ab_plot), label=f"{cat} — Coreset & Metric Ablation")

    return demo
