"""Production inference engine loading packaged category bundles for FastAPI and Gradio."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from anomavision import __version__
from anomavision.config import PipelineConfig
from anomavision.core.detector import PatchCoreDetector
from anomavision.core.drift import DriftReferenceStats
from anomavision.core.retrieval import ReferenceImageSimilarityIndex
from anomavision.evaluation.calibration import normalize_anomaly_score
from anomavision.evaluation.visualization import (
    encode_rgb_to_base64_png,
    render_heatmap_rgb,
    render_overlay_rgb,
)
from anomavision.models.dinov3 import DINOv3FeatureExtractor
from anomavision.models.sam import SAM3ObjectSegmenter, apply_object_mask_refinement
from anomavision.schemas import (
    CalibrationParams,
    CategoryEvaluationReport,
    PredictionResult,
    SimilarNeighbor,
)

logger = logging.getLogger(__name__)


@dataclass
class CategoryBundleRuntime:
    """Loaded in-memory artifacts for a single MVTec AD category."""

    category: str
    bundle_dir: Path
    detector: PatchCoreDetector
    retrieval_index: ReferenceImageSimilarityIndex
    calibration: CalibrationParams
    drift_stats: DriftReferenceStats
    evaluation_report: CategoryEvaluationReport


class InferenceEngine:
    """Stateful inference engine serving predictions and similarity retrieval across categories."""

    def __init__(
        self,
        bundles_root: Path,
        config_path: Path | None = None,
        load_sam: bool = False,
        device: str | None = None,
        hf_token: str | None = None,
        feature_extractor: Any | None = None,
        sam_segmenter: Any | None = None,
    ) -> None:
        self.bundles_root = bundles_root
        self.categories: dict[str, CategoryBundleRuntime] = {}

        # Load config from first available bundle or explicit path
        if config_path is not None and config_path.is_file():
            self.cfg = PipelineConfig.from_yaml(config_path)
        else:
            bundle_cfgs = sorted(bundles_root.glob("*/pipeline_config.yaml")) if bundles_root.is_dir() else []
            if bundle_cfgs:
                self.cfg = PipelineConfig.from_yaml(bundle_cfgs[0])
            else:
                self.cfg = PipelineConfig()

        self._load_all_bundles()

        if feature_extractor is not None:
            self.extractor = feature_extractor
        else:
            self.extractor = DINOv3FeatureExtractor(
                self.cfg.backbone,
                device=device,
                hf_token=hf_token,
            )

        self.sam_segmenter = sam_segmenter
        if self.sam_segmenter is None and load_sam and self.cfg.sam.enabled:
            self.sam_segmenter = SAM3ObjectSegmenter(
                self.cfg.sam,
                device=device,
                hf_token=hf_token,
            )

    def _load_all_bundles(self) -> None:
        if not self.bundles_root.is_dir():
            return

        for cat_dir in sorted(self.bundles_root.iterdir()):
            if not cat_dir.is_dir():
                continue
            coreset_path = cat_dir / "patchcore_coreset.npz"
            faiss_path = cat_dir / "reference_images.faiss"
            meta_path = cat_dir / "reference_metadata.json"
            cal_path = cat_dir / "calibration.json"
            drift_path = cat_dir / "drift_reference.npz"
            rep_path = cat_dir / "evaluation_report.json"

            if not (coreset_path.is_file() and faiss_path.is_file() and cal_path.is_file()):
                continue

            detector = PatchCoreDetector.load(coreset_path, self.cfg.coreset)
            retrieval_index = ReferenceImageSimilarityIndex.load(faiss_path, meta_path)
            calibration = CalibrationParams.model_validate_json(cal_path.read_text(encoding="utf-8"))
            report = CategoryEvaluationReport.model_validate_json(rep_path.read_text(encoding="utf-8"))

            drift_npz = np.load(drift_path)
            drift_stats = DriftReferenceStats(
                reference_anchors=drift_npz["reference_anchors"].astype(np.float32),
                baseline_mmd_mean=float(drift_npz["baseline_mmd_mean"]),
                baseline_mmd_std=float(drift_npz["baseline_mmd_std"]),
                alert_threshold=float(drift_npz["alert_threshold"]),
            )

            self.categories[cat_dir.name] = CategoryBundleRuntime(
                category=cat_dir.name,
                bundle_dir=cat_dir,
                detector=detector,
                retrieval_index=retrieval_index,
                calibration=calibration,
                drift_stats=drift_stats,
                evaluation_report=report,
            )

    @property
    def available_categories(self) -> list[str]:
        return sorted(self.categories.keys())

    def predict(
        self,
        image: np.ndarray | Image.Image,
        category: str = "bottle",
        use_sam_refinement: bool = False,
        top_k_similar: int = 4,
        include_visualizations: bool = True,
    ) -> PredictionResult:
        """Run end-to-end anomaly detection, localization, similarity search, and drift check."""
        if category not in self.categories:
            raise KeyError(
                f"Category '{category}' is not loaded. Available categories: {self.available_categories}"
            )
        runtime = self.categories[category]

        if isinstance(image, Image.Image):
            image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        else:
            if image.ndim == 2:
                image_rgb = np.stack([image] * 3, axis=-1).astype(np.uint8)
            else:
                image_rgb = image[:, :, :3].astype(np.uint8)

        t0 = time.perf_counter()
        feat_batch = self.extractor.extract_batch([image_rgb])
        pred_batch = runtime.detector.predict_batch(
            patch_embeddings=feat_batch.patch_embeddings,
            grid_size=feat_batch.grid_size,
            original_sizes=feat_batch.original_sizes,
        )

        anomaly_map = pred_batch.anomaly_maps[0]
        sam_coverage: float | None = None
        sam_mask_arr: np.ndarray | None = None

        if use_sam_refinement and self.sam_segmenter is not None:
            sam_mask_arr, sam_coverage = self.sam_segmenter.segment_object(image_rgb, category=category)
            anomaly_map = apply_object_mask_refinement(
                anomaly_map=anomaly_map,
                object_mask=sam_mask_arr,
                background_suppression_factor=self.cfg.sam.background_suppression_factor,
            )

        inf_latency_ms = (time.perf_counter() - t0) * 1000.0

        # Retrieve visually similar normal training references
        neighbors, ret_latency_ms = runtime.retrieval_index.query(
            feat_batch.global_embeddings[0],
            top_k=top_k_similar,
        )

        # Compute patch embedding distribution drift (MMD^2)
        mmd_val, drift_alert = runtime.drift_stats.score_query_patches(feat_batch.patch_embeddings[0])

        raw_score = float(pred_batch.image_scores[0])
        threshold = runtime.calibration.image_threshold
        norm_score = normalize_anomaly_score(raw_score, runtime.calibration)
        is_anom = bool(raw_score >= threshold)
        decision: str = "ANOMALOUS" if is_anom else "NORMAL"
        margin = float(raw_score - threshold)

        heatmap_b64: str | None = None
        overlay_b64: str | None = None
        mask_b64: str | None = None
        sam_b64: str | None = None

        if include_visualizations:
            vmin = runtime.calibration.cal_score_min * 0.7
            vmax = max(float(np.max(anomaly_map)), runtime.calibration.pixel_threshold * 1.25)
            heatmap_rgb = render_heatmap_rgb(anomaly_map, vmin=vmin, vmax=vmax)
            overlay_rgb, binary_mask_255 = render_overlay_rgb(
                image_rgb=image_rgb,
                anomaly_map=anomaly_map,
                pixel_threshold=runtime.calibration.pixel_threshold,
                vmin=vmin,
                vmax=vmax,
            )
            heatmap_b64 = encode_rgb_to_base64_png(heatmap_rgb)
            overlay_b64 = encode_rgb_to_base64_png(overlay_rgb)
            mask_b64 = encode_rgb_to_base64_png(binary_mask_255)
            if sam_mask_arr is not None:
                sam_vis = (
                    image_rgb.astype(np.float32) * (0.25 + 0.75 * (sam_mask_arr[:, :, None] > 0))
                ).astype(np.uint8)
                sam_b64 = encode_rgb_to_base64_png(sam_vis)

        return PredictionResult(
            request_id=str(uuid.uuid4()),
            category=category,
            anomaly_score=round(raw_score, 6),
            normalized_score=round(norm_score, 6),
            threshold=round(threshold, 6),
            is_anomalous=is_anom,
            decision_label=decision,  # type: ignore[arg-type]
            confidence_margin=round(margin, 6),
            max_patch_score=round(float(pred_batch.max_patch_scores[0]), 6),
            mean_top_k_patch_score=round(float(pred_batch.top_k_patch_scores[0]), 6),
            sam_object_coverage=round(sam_coverage, 4) if sam_coverage is not None else None,
            embedding_drift_mmd=round(mmd_val, 6),
            drift_alert=drift_alert,
            inference_latency_ms=round(inf_latency_ms, 3),
            retrieval_latency_ms=round(ret_latency_ms, 3),
            similar_references=neighbors,
            heatmap_base64_png=heatmap_b64,
            overlay_base64_png=overlay_b64,
            mask_base64_png=mask_b64,
            sam_mask_base64_png=sam_b64,
            model_version=__version__,
            backbone_revision=self.cfg.backbone.revision,
        )

    def retrieve_similar(
        self,
        image: np.ndarray | Image.Image,
        category: str = "bottle",
        top_k: int = 4,
    ) -> tuple[list[SimilarNeighbor], float]:
        """Retrieve visually similar normal reference samples for a query image."""
        if category not in self.categories:
            raise KeyError(f"Category '{category}' is not loaded")
        runtime = self.categories[category]
        feat_batch = self.extractor.extract_batch([image])
        return runtime.retrieval_index.query(feat_batch.global_embeddings[0], top_k=top_k)
