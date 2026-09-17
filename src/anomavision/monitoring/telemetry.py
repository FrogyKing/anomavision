"""Prometheus metrics exposition, rolling drift telemetry, and structured JSON logging."""

from __future__ import annotations

import json
import logging
from collections import deque
from datetime import UTC, datetime
from typing import Any

import numpy as np
from prometheus_client import (
     CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from anomavision.schemas import PredictionResult

REGISTRY = CollectorRegistry(auto_describe=True)

REQUESTS_TOTAL = Counter(
    "anomavision_requests_total",
    "Total HTTP requests processed by AnomaVision API",
    ["endpoint", "category", "status"],
    registry=REGISTRY,
)

ERRORS_TOTAL = Counter(
    "anomavision_errors_total",
    "Total errors encountered during API request processing",
    ["endpoint", "error_type"],
    registry=REGISTRY,
)

INFERENCE_LATENCY_SECONDS = Histogram(
    "anomavision_inference_latency_seconds",
    "End-to-end anomaly detection inference latency in seconds",
    ["category"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=REGISTRY,
)

RETRIEVAL_LATENCY_SECONDS = Histogram(
    "anomavision_retrieval_latency_seconds",
    "FAISS similarity retrieval latency in seconds",
    ["category"],
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1),
    registry=REGISTRY,
)

ANOMALY_SCORE_HISTOGRAM = Histogram(
    "anomavision_anomaly_score",
    "Distribution of raw PatchCore anomaly scores",
    ["category"],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0),
    registry=REGISTRY,
)

PREDICTIONS_BY_DECISION = Counter(
    "anomavision_predictions_total",
    "Total inspection decisions by category and classification outcome",
    ["category", "decision"],
    registry=REGISTRY,
)

EMBEDDING_DRIFT_MMD_GAUGE = Gauge(
    "anomavision_embedding_drift_mmd",
    "Latest multi-bandwidth RBF Maximum Mean Discrepancy (MMD^2) against normal reference",
    ["category"],
    registry=REGISTRY,
)

DRIFT_ALERTS_TOTAL = Counter(
    "anomavision_drift_alerts_total",
    "Total covariate embedding drift alerts triggered when MMD^2 exceeds 4-sigma threshold",
    ["category"],
    registry=REGISTRY,
)


class JSONLogFormatter(logging.Formatter):
    """Structured JSON log formatter for production observability."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            payload.update(record.extra_fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, separators=(",", ":"))


def configure_structured_logging(level: int = logging.INFO) -> None:
    """Configure root logger with structured JSON formatting."""
    root = logging.getLogger()
    root.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(JSONLogFormatter())
    root.handlers = [handler]


class MonitoringTracker:
    """Rolling in-memory telemetry tracker for score distributions and unlabeled embedding drift.

    Operational Distinction:
      - `embedding_drift_alert` (MMD^2 > threshold) measures unlabeled covariate shift in the
        incoming DINOv3 patch representation space (e.g., camera defocus, lighting shift, or
        novel background clutter). It is available immediately at inference time without ground-truth
        labels.
      - Actual classification accuracy / F1 degradation requires delayed human QA confirmation
        labels on flagged or sampled parts and must not be conflated with unsupervised drift alerts.
    """

    def __init__(self, window_size: int = 200) -> None:
        self.window_size = window_size
        self._scores: dict[str, deque[float]] = {}
        self._mmds: dict[str, deque[float]] = {}
        self._decisions: dict[str, deque[int]] = {}
        self._drift_alerts: dict[str, int] = {}

    def record_prediction(self, res: PredictionResult) -> None:
        cat = res.category
        if cat not in self._scores:
            self._scores[cat] = deque(maxlen=self.window_size)
            self._mmds[cat] = deque(maxlen=self.window_size)
            self._decisions[cat] = deque(maxlen=self.window_size)
            self._drift_alerts[cat] = 0

        self._scores[cat].append(res.anomaly_score)
        self._mmds[cat].append(res.embedding_drift_mmd)
        self._decisions[cat].append(1 if res.is_anomalous else 0)

        INFERENCE_LATENCY_SECONDS.labels(category=cat).observe(res.inference_latency_ms / 1000.0)
        RETRIEVAL_LATENCY_SECONDS.labels(category=cat).observe(res.retrieval_latency_ms / 1000.0)
        ANOMALY_SCORE_HISTOGRAM.labels(category=cat).observe(res.anomaly_score)
        PREDICTIONS_BY_DECISION.labels(category=cat, decision=res.decision_label).inc()
        EMBEDDING_DRIFT_MMD_GAUGE.labels(category=cat).set(res.embedding_drift_mmd)

        if res.drift_alert:
            self._drift_alerts[cat] += 1
            DRIFT_ALERTS_TOTAL.labels(category=cat).inc()

    def get_summary(self) -> dict[str, Any]:
        categories_summary: dict[str, Any] = {}
        for cat, scores_dq in self._scores.items():
            scores = np.asarray(scores_dq, dtype=np.float64)
            mmds = np.asarray(self._mmds[cat], dtype=np.float64)
            decs = np.asarray(self._decisions[cat], dtype=np.float64)
            categories_summary[cat] = {
                "window_samples": len(scores),
                "anomaly_score_mean": round(float(np.mean(scores)), 6) if len(scores) else 0.0,
                "anomaly_score_p95": round(float(np.percentile(scores, 95)), 6) if len(scores) else 0.0,
                "flagged_anomalous_rate": round(float(np.mean(decs)), 4) if len(decs) else 0.0,
                "embedding_mmd_mean": round(float(np.mean(mmds)), 6) if len(mmds) else 0.0,
                "embedding_mmd_max": round(float(np.max(mmds)), 6) if len(mmds) else 0.0,
                "total_drift_alerts": self._drift_alerts.get(cat, 0),
            }
        return {
            "monitoring_Note": (
                "Covariate embedding drift (MMD^2) and score quantile shifts are unsupervised leading "
                "indicators computed without ground-truth labels. Actual precision/recall degradation "
                "requires delayed human QA inspection labels."
            ),
            "categories": categories_summary,
        }


def render_prometheus_metrics() -> tuple[bytes, str]:
    """Serialize Prometheus metrics registry to standard text exposition format."""
    return generate_latest(REGISTRY), "text/plain; version=0.0.4; charset=utf-8"
