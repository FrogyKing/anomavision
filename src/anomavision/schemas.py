"""Pydantic domain schemas for manifests, evaluation reports, API payloads, and monitoring."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class SampleRecord(BaseModel):
    """Metadata record for a single MVTec AD sample."""

    sample_id: str
    category: str
    split: Literal["train_fit", "calibration", "test"]
    defect_type: str
    is_anomalous: bool
    image_relpath: str
    mask_relpath: str | None = None
    image_sha256: str
    mask_sha256: str | None = None
    width: int
    height: int


class CategoryDatasetManifest(BaseModel):
    """Integrity and split manifest for an MVTec AD category."""

    category: str
    dataset_id: str
    dataset_revision: str
    license_id: str
    created_at: datetime
    seed: int
    calibration_fraction: float
    num_train_fit: int
    num_calibration: int
    num_test_good: int
    num_test_anomalous: int
    defect_types: list[str]
    manifest_sha256: str
    samples: list[SampleRecord]


class CalibrationParams(BaseModel):
    """Threshold and score normalization parameters fitted on calibration data."""

    category: str
    method: Literal["unsupervised_quantile", "supervised_f1_optimal"]
    is_purely_unsupervised: bool = True
    quantile_alpha: float = 0.99
    image_threshold: float
    pixel_threshold: float
    cal_score_mean: float
    cal_score_std: float
    cal_score_min: float
    cal_score_max: float
    reference_centroid_norm: float
    reference_mmd_baseline: float

    model_config = {"populate_by_name": True}


class MetricWithCI(BaseModel):
    """Point estimate with bootstrap 95% confidence interval."""

    value: float
    ci_lower: float
    ci_upper: float
    std_error: float


class CategoryEvaluationReport(BaseModel):
    """Comprehensive quantitative evaluation report for a single MVTec AD category."""

    category: str
    config_fingerprint: str
    backbone_model_id: str
    backbone_revision: str
    sam_variant: str
    image_auroc: MetricWithCI
    pixel_auroc: MetricWithCI
    aupro: float
    precision_at_threshold: float
    recall_at_threshold: float
    f1_at_threshold: float
    f1_oracle_upper_bound: float
    threshold_used: float
    threshold_procedure: str
    baseline_image_auroc: float
    baseline_pixel_auroc: float
    baseline_name: str
    memory_bank_patches: int
    total_train_patches: int
    coreset_ratio_actual: float
    memory_bank_size_mb: float
    inference_latency_ms_mean: float
    inference_latency_ms_p95: float
    retrieval_latency_ms_mean: float
    peak_gpu_memory_mb: float
    per_defect_auroc: dict[str, float]


class AblationPoint(BaseModel):
    """Single experimental configuration and measured outcome in an ablation study."""

    experiment_id: str
    category: str
    l2_normalize: bool
    metric: Literal["cosine", "euclidean"]
    coreset_ratio: float
    memory_bank_patches: int
    memory_bank_size_mb: float
    image_auroc: float
    pixel_auroc: float
    aupro: float
    inference_latency_ms: float
    retrieval_latency_ms: float


class SimilarNeighbor(BaseModel):
    """A visually similar normal reference sample retrieved from FAISS."""

    rank: int
    sample_id: str
    category: str
    image_relpath: str
    distance: float
    similarity_score: float


class PredictionResult(BaseModel):
    """Structured inference output for the API and UI."""

    request_id: str
    category: str
    anomaly_score: float
    normalized_score: float
    threshold: float
    is_anomalous: bool
    decision_label: Literal["ANOMALOUS", "NORMAL"]
    confidence_margin: float
    max_patch_score: float
    mean_top_k_patch_score: float
    sam_object_coverage: float | None = None
    embedding_drift_mmd: float
    drift_alert: bool
    inference_latency_ms: float
    retrieval_latency_ms: float
    similar_references: list[SimilarNeighbor] = Field(default_factory=list)
    heatmap_base64_png: str | None = None
    overlay_base64_png: str | None = None
    mask_base64_png: str | None = None
    sam_mask_base64_png: str | None = None
    model_version: str
    backbone_revision: str


class CostEstimateRecord(BaseModel):
    """Explicit Modal compute usage and cost estimate record."""

    job_name: str
    started_at: datetime
    completed_at: datetime
    duration_seconds: float
    hardware_type: str
    hourly_rate_usd: float
    estimated_cost_usd: float
    notes: str
