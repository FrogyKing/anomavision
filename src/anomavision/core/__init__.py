"""Core mathematical algorithms for coreset selection, anomaly scoring, retrieval, and drift."""

from anomavision.core.coreset import CoresetSelectionResult, select_greedy_kcenter_coreset
from anomavision.core.detector import (
    AnomalyPredictionBatch,
    GlobalCentroidBaseline,
    PatchCoreDetector,
    compute_pairwise_distances,
    upsample_patch_scores,
)
from anomavision.core.drift import DriftReferenceStats, compute_rbf_mmd_squared
from anomavision.core.retrieval import ReferenceImageSimilarityIndex

__all__ = [
    "AnomalyPredictionBatch",
    "CoresetSelectionResult",
    "DriftReferenceStats",
    "GlobalCentroidBaseline",
    "PatchCoreDetector",
    "ReferenceImageSimilarityIndex",
    "compute_pairwise_distances",
    "compute_rbf_mmd_squared",
    "select_greedy_kcenter_coreset",
    "upsample_patch_scores",
]
