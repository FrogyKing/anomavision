"""FastAPI application providing health, readiness, metadata, prediction, retrieval, and metrics endpoints."""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from PIL import Image, UnidentifiedImageError

from anomavision import __version__
from anomavision.monitoring.telemetry import (
    ERRORS_TOTAL,
    REQUESTS_TOTAL,
    MonitoringTracker,
    configure_structured_logging,
    render_prometheus_metrics,
)
from anomavision.pipeline.inference import InferenceEngine
from anomavision.schemas import PredictionResult, SimilarNeighbor

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB limit


def _decode_upload_image(raw_bytes: bytes) -> Image.Image:
    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty")
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded file exceeds 20 MB maximum size")
    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            img.verify()
        with Image.open(io.BytesIO(raw_bytes)) as img:
            return img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted image file: {exc}") from exc


def create_app(
    engine: InferenceEngine | None = None,
    bundles_dir: Path | None = None,
) -> FastAPI:
    """Create and configure the FastAPI production service."""
    configure_structured_logging()

    if engine is None:
        resolved_bundles = bundles_dir or Path(
            os.environ.get("ANOMAVISION_BUNDLES_DIR", "artifacts/bundles")
        )
        load_sam = os.environ.get("ANOMAVISION_LOAD_SAM", "0") == "1"
        engine = InferenceEngine(bundles_root=resolved_bundles, load_sam=load_sam)

    tracker = MonitoringTracker()

    api = FastAPI(
        title="AnomaVision Visual Anomaly Intelligence API",
        version=__version__,
        description=(
            "Production-grade open-world industrial anomaly detection, pixel-level localization, "
            "FAISS visual similarity retrieval, and MMD embedding drift monitoring."
        ),
    )
    api.state.engine = engine
    api.state.tracker = tracker

    @api.get("/health")
    def health() -> dict[str, Any]:
        REQUESTS_TOTAL.labels(endpoint="/health", category="system", status="200").inc()
        return {
            "status": "ok",
            "service": "anomavision-api",
            "version": __version__,
        }

    @api.get("/ready")
    def ready() -> dict[str, Any]:
        eng: InferenceEngine = api.state.engine
        cats = eng.available_categories
        if not cats:
            ERRORS_TOTAL.labels(endpoint="/ready", error_type="no_bundles_loaded").inc()
            raise HTTPException(status_code=503, detail="No category model bundles are loaded")
        REQUESTS_TOTAL.labels(endpoint="/ready", category="system", status="200").inc()
        return {
            "status": "ready",
            "available_categories": cats,
            "backbone_model_id": eng.cfg.backbone.model_id,
            "backbone_revision": eng.cfg.backbone.revision,
        }

    @api.get("/metadata")
    def metadata() -> dict[str, Any]:
        eng: InferenceEngine = api.state.engine
        REQUESTS_TOTAL.labels(endpoint="/metadata", category="system", status="200").inc()
        cat_details: dict[str, Any] = {}
        for cat, runtime in eng.categories.items():
            cat_details[cat] = {
                "calibration": runtime.calibration.model_dump(mode="json"),
                "evaluation_summary": {
                    "image_auroc": runtime.evaluation_report.image_auroc.model_dump(mode="json"),
                    "pixel_auroc": runtime.evaluation_report.pixel_auroc.model_dump(mode="json"),
                    "aupro": runtime.evaluation_report.aupro,
                    "f1_unsupervised": runtime.evaluation_report.f1_at_threshold,
                    "memory_bank_patches": runtime.evaluation_report.memory_bank_patches,
                    "memory_bank_size_mb": runtime.evaluation_report.memory_bank_size_mb,
                },
            }
        return {
            "version": __version__,
            "config_fingerprint": eng.cfg.fingerprint(),
            "backbone": eng.cfg.backbone.model_dump(mode="json"),
            "sam": eng.cfg.sam.model_dump(mode="json"),
            "coreset": eng.cfg.coreset.model_dump(mode="json"),
            "categories": cat_details,
        }

    @api.post("/predict", response_model=PredictionResult)
    async def predict(
        file: UploadFile = File(..., description="Inspected component RGB image (PNG/JPEG)"),
        category: str = Form("bottle", description="MVTec AD category name (e.g., bottle, metal_nut)"),
        use_sam_refinement: bool = Form(False, description="Apply SAM 3 foreground mask gating if loaded"),
        top_k_similar: int = Form(4, ge=1, le=12, description="Number of similar normal references"),
        include_visualizations: bool = Form(True, description="Include base64 PNG heatmap and overlay"),
    ) -> PredictionResult:
        eng: InferenceEngine = api.state.engine
        mon: MonitoringTracker = api.state.tracker

        if category not in eng.categories:
            ERRORS_TOTAL.labels(endpoint="/predict", error_type="invalid_category").inc()
            REQUESTS_TOTAL.labels(endpoint="/predict", category=category, status="404").inc()
            raise HTTPException(
                status_code=404,
                detail=f"Unknown category '{category}'. Available: {eng.available_categories}",
            )

        raw_bytes = await file.read()
        pil_img = _decode_upload_image(raw_bytes)

        try:
            result = eng.predict(
                image=pil_img,
                category=category,
                use_sam_refinement=use_sam_refinement,
                top_k_similar=top_k_similar,
                include_visualizations=include_visualizations,
            )
        except Exception as exc:
            ERRORS_TOTAL.labels(endpoint="/predict", error_type="inference_error").inc()
            REQUESTS_TOTAL.labels(endpoint="/predict", category=category, status="500").inc()
            logger.exception("Prediction failed for category '%s'", category)
            raise HTTPException(status_code=500, detail=f"Inference execution error: {exc}") from exc

        mon.record_prediction(result)
        REQUESTS_TOTAL.labels(endpoint="/predict", category=category, status="200").inc()
        logger.info(
            "Processed inspection request",
            extra={
                "extra_fields": {
                    "request_id": result.request_id,
                    "category": category,
                    "anomaly_score": result.anomaly_score,
                    "decision": result.decision_label,
                    "drift_mmd": result.embedding_drift_mmd,
                    "latency_ms": result.inference_latency_ms,
                }
            },
        )
        return result

    @api.post("/retrieve", response_model=list[SimilarNeighbor])
    async def retrieve(
        file: UploadFile = File(..., description="Query component RGB image"),
        category: str = Form("bottle", description="MVTec AD category name"),
        top_k: int = Form(4, ge=1, le=12, description="Number of normal references to retrieve"),
    ) -> list[SimilarNeighbor]:
        eng: InferenceEngine = api.state.engine
        if category not in eng.categories:
            ERRORS_TOTAL.labels(endpoint="/retrieve", error_type="invalid_category").inc()
            REQUESTS_TOTAL.labels(endpoint="/retrieve", category=category, status="404").inc()
            raise HTTPException(
                status_code=404,
                detail=f"Unknown category '{category}'. Available: {eng.available_categories}",
            )
        raw_bytes = await file.read()
        pil_img = _decode_upload_image(raw_bytes)
        neighbors, _ = eng.retrieve_similar(image=pil_img, category=category, top_k=top_k)
        REQUESTS_TOTAL.labels(endpoint="/retrieve", category=category, status="200").inc()
        return neighbors

    @api.get("/monitoring/drift")
    def monitoring_drift() -> dict[str, Any]:
        mon: MonitoringTracker = api.state.tracker
        REQUESTS_TOTAL.labels(endpoint="/monitoring/drift", category="system", status="200").inc()
        return mon.get_summary()

    @api.get("/metrics")
    def prometheus_metrics() -> Response:
        payload, content_type = render_prometheus_metrics()
        return Response(content=payload, media_type=content_type)

    return api
