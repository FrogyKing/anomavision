"""End-to-end MLOps pipeline stages, experiment tracking, and production inference engine."""

from anomavision.pipeline.inference import InferenceEngine
from anomavision.pipeline.stages import (
    run_category_pipeline,
    stage_build_memory_bank,
    stage_cache_checkpoints,
    stage_calibrate,
    stage_evaluate_and_ablate,
    stage_extract_embeddings,
    stage_package_bundle,
    stage_prepare_dataset,
)
from anomavision.pipeline.tracking import (
    log_experiment_to_mlflow,
    record_modal_cost_estimate,
    write_dvc_and_lineage_manifests,
)

__all__ = [
    "InferenceEngine",
    "log_experiment_to_mlflow",
    "record_modal_cost_estimate",
    "run_category_pipeline",
    "stage_build_memory_bank",
    "stage_cache_checkpoints",
    "stage_calibrate",
    "stage_evaluate_and_ablate",
    "stage_extract_embeddings",
    "stage_package_bundle",
    "stage_prepare_dataset",
    "write_dvc_and_lineage_manifests",
]
