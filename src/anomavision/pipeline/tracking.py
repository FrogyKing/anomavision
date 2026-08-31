"""MLflow experiment tracking, DVC artifact lineage generation, and Modal cost estimation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from anomavision.config import PipelineConfig
from anomavision.schemas import AblationPoint, CategoryEvaluationReport, CostEstimateRecord


def get_git_commit_sha() -> str:
    """Resolve current Git HEAD commit SHA or 'uncommitted'."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def log_experiment_to_mlflow(
    cfg: PipelineConfig,
    reports: list[CategoryEvaluationReport],
    ablations: list[AblationPoint],
    artifacts_root: Path,
    tracking_uri: str | None = None,
) -> str:
    """Record full experiment parameters, metrics, model revisions, and plots in MLflow."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    if tracking_uri is None:
        db_path = (artifacts_root / "mlflow.db").resolve().as_posix()
        tracking_uri = f"sqlite:///{db_path}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("AnomaVision-MVTec-AD")

    git_sha = get_git_commit_sha()
    with mlflow.start_run(run_name=f"dinov3_patchcore_{cfg.fingerprint()[:8]}") as run:
        mlflow.log_params(
            {
                "git_commit_sha": git_sha,
                "config_fingerprint": cfg.fingerprint(),
                "backbone_model_id": cfg.backbone.model_id,
                "backbone_revision": cfg.backbone.revision,
                "image_size": cfg.backbone.image_size,
                "patch_size": cfg.backbone.patch_size,
                "num_register_tokens": cfg.backbone.num_register_tokens,
                "extract_layers": ",".join(str(x) for x in cfg.backbone.extract_layers),
                "neighborhood_kernel": cfg.backbone.neighborhood_kernel,
                "l2_normalize": cfg.backbone.l2_normalize,
                "sam_enabled": cfg.sam.enabled,
                "sam_model_id": cfg.sam.primary_model_id,
                "sam_revision": cfg.sam.primary_revision,
                "coreset_ratio": cfg.coreset.coreset_ratio,
                "coreset_metric": cfg.coreset.metric,
                "coreset_projection_dim": cfg.coreset.projection_dim,
                "n_neighbors": cfg.coreset.n_neighbors,
                "dataset_id": cfg.dataset.hf_dataset_id,
                "dataset_revision": cfg.dataset.hf_revision,
                "calibration_fraction": cfg.dataset.calibration_fraction,
            }
        )

        for rep in reports:
            cat = rep.category
            mlflow.log_metrics(
                {
                    f"{cat}_image_auroc": rep.image_auroc.value,
                    f"{cat}_image_auroc_ci_low": rep.image_auroc.ci_lower,
                    f"{cat}_image_auroc_ci_high": rep.image_auroc.ci_upper,
                    f"{cat}_pixel_auroc": rep.pixel_auroc.value,
                    f"{cat}_aupro": rep.aupro,
                    f"{cat}_f1_unsupervised": rep.f1_at_threshold,
                    f"{cat}_f1_oracle": rep.f1_oracle_upper_bound,
                    f"{cat}_baseline_image_auroc": rep.baseline_image_auroc,
                    f"{cat}_baseline_pixel_auroc": rep.baseline_pixel_auroc,
                    f"{cat}_memory_bank_mb": rep.memory_bank_size_mb,
                    f"{cat}_inference_latency_ms": rep.inference_latency_ms_mean,
                    f"{cat}_retrieval_latency_ms": rep.retrieval_latency_ms_mean,
                    f"{cat}_peak_gpu_memory_mb": rep.peak_gpu_memory_mb,
                }
            )

        if reports:
            mlflow.log_metrics(
                {
                    "mean_image_auroc": float(sum(r.image_auroc.value for r in reports) / len(reports)),
                    "mean_pixel_auroc": float(sum(r.pixel_auroc.value for r in reports) / len(reports)),
                    "mean_aupro": float(sum(r.aupro for r in reports) / len(reports)),
                    "mean_f1_unsupervised": float(sum(r.f1_at_threshold for r in reports) / len(reports)),
                }
            )

        for ab in ablations:
            mlflow.log_metrics(
                {
                    f"ablation_{ab.experiment_id}_img_auroc": ab.image_auroc,
                    f"ablation_{ab.experiment_id}_pix_auroc": ab.pixel_auroc,
                    f"ablation_{ab.experiment_id}_aupro": ab.aupro,
                }
            )

        bundles_local = artifacts_root / "bundles"
        if bundles_local.is_dir():
            mlflow.log_artifacts(str(bundles_local), artifact_path="bundles")

        return str(run.info.run_id)


def record_modal_cost_estimate(
    job_name: str,
    started_at: datetime,
    completed_at: datetime,
    hardware_type: str,
    hourly_rate_usd: float,
    artifacts_root: Path,
    notes: str = "Estimated from wall-clock container execution duration and published Modal GPU pricing.",
) -> CostEstimateRecord:
    """Compute and append a Modal compute cost estimate record."""
    duration_s = max(0.0, (completed_at - started_at).total_seconds())
    cost_usd = (duration_s / 3600.0) * hourly_rate_usd

    rec = CostEstimateRecord(
        job_name=job_name,
        started_at=started_at,
        completed_at=completed_at,
        duration_seconds=round(duration_s, 3),
        hardware_type=hardware_type,
        hourly_rate_usd=hourly_rate_usd,
        estimated_cost_usd=round(cost_usd, 6),
        notes=notes,
    )

    cost_file = artifacts_root / "cost_telemetry.json"
    cost_file.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if cost_file.is_file():
        try:
            existing = json.loads(cost_file.read_text(encoding="utf-8"))
        except Exception:
            existing = []
    existing.append(rec.model_dump(mode="json"))

    total_usd = sum(float(item.get("estimated_cost_usd", 0.0)) for item in existing)
    summary_file = artifacts_root / "cost_summary.json"
    summary_file.write_text(
        json.dumps(
            {
                "total_estimated_cost_usd": round(total_usd, 4),
                "budget_target_usd": 15.0,
                "budget_max_usd": 30.0,
                "within_budget": total_usd < 15.0,
                "jobs": existing,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    cost_file.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return rec


def write_dvc_and_lineage_manifests(
    cfg: PipelineConfig,
    reports: list[CategoryEvaluationReport],
    artifacts_root: Path,
    project_root: Path,
) -> Path:
    """Write DVC pipeline definition (`dvc.yaml`) and cryptographic artifact lineage manifest."""
    dvc_spec = {
        "stages": {
            "run_modal_pipeline": {
                "cmd": "uv run anomavision run-pipeline --config configs/default.yaml",
                "deps": [
                    "configs/default.yaml",
                    "src/anomavision/models/dinov3.py",
                    "src/anomavision/models/sam.py",
                    "src/anomavision/core/coreset.py",
                    "src/anomavision/core/detector.py",
                    "src/anomavision/pipeline/stages.py",
                ],
                "outs": [
                    "artifacts/evaluation_summary.json",
                    "artifacts/artifact_lineage_manifest.json",
                ],
            }
        }
    }
    dvc_path = project_root / "dvc.yaml"
    dvc_path.write_text(yaml.safe_dump(dvc_spec, sort_keys=False), encoding="utf-8")

    # Compute SHA-256 of all local bundle files to bind local artifacts to Modal Volume state
    bundle_hashes: dict[str, str] = {}
    bundles_dir = artifacts_root / "bundles"
    if bundles_dir.is_dir():
        for fpath in sorted(bundles_dir.rglob("*")):
            if fpath.is_file():
                h = hashlib.sha256(fpath.read_bytes()).hexdigest()
                bundle_hashes[fpath.relative_to(artifacts_root).as_posix()] = h

    lineage = {
        "git_commit_sha": get_git_commit_sha(),
        "config_fingerprint": cfg.fingerprint(),
        "modal_volume_name": cfg.modal.volume_name,
        "modal_mount_path": cfg.modal.mount_path,
        "dataset": {
            "hf_dataset_id": cfg.dataset.hf_dataset_id,
            "hf_revision": cfg.dataset.hf_revision,
            "official_source_url": cfg.dataset.official_source_url,
            "license_id": cfg.dataset.license_id,
            "categories": cfg.dataset.categories,
        },
        "models": {
            "dinov3_backbone": {
                "model_id": cfg.backbone.model_id,
                "revision": cfg.backbone.revision,
            },
            "sam3_primary": {
                "model_id": cfg.sam.primary_model_id,
                "revision": cfg.sam.primary_revision,
                "weights_sha256": cfg.sam.primary_weights_sha256,
            },
            "sam3_1_optional": {
                "model_id": cfg.sam.optional_sam31_model_id,
                "revision": cfg.sam.optional_sam31_revision,
            },
        },
        "categories_evaluated": [r.category for r in reports],
        "artifact_sha256": bundle_hashes,
    }

    lineage_path = artifacts_root / "artifact_lineage_manifest.json"
    lineage_path.parent.mkdir(parents=True, exist_ok=True)
    lineage_path.write_text(json.dumps(lineage, indent=2), encoding="utf-8")
    return lineage_path
