"""Modal GPU execution definition mounting persistent Volume `anomavision-data` and Secret `huggingface`."""

from __future__ import annotations

import io
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

VOLUME_NAME = "anomavision-data"
MOUNT_PATH = "/data"
HF_SECRET_NAME = "huggingface"

app = modal.App("anomavision-pipeline")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME)

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "accelerate>=1.0.0",
        "faiss-cpu>=1.9.0",
        "huggingface-hub>=0.26.0",
        "matplotlib>=3.9.0",
        "numpy>=1.26.0",
        "pillow>=10.4.0",
        "pydantic>=2.9.0",
        "pydantic-settings>=2.5.0",
        "pyyaml>=6.0.2",
        "safetensors>=0.4.5",
        "scikit-learn>=1.5.0",
        "scipy>=1.13.0",
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "transformers>=5.0.0",
    )
    .add_local_python_source("anomavision")
)


@app.function(
    image=gpu_image,
    gpu="L4",
    volumes={MOUNT_PATH: volume},
    secrets=[hf_secret],
    timeout=1800,
)
def execute_remote_gpu_pipeline(
    config_dict: dict[str, Any],
    force: bool = False,
    run_ablations: bool = True,
    categories: list[str] | None = None,
) -> dict[str, Any]:
    """Run the full AnomaVision GPU pipeline on Modal L4 and return serialized bundles + reports."""
    import logging

    from anomavision.config import PipelineConfig
    from anomavision.modal_storage import (
        VolumeLayout,
        configure_hf_cache_env,
        resolve_hf_token,
    )
    from anomavision.pipeline.stages import run_category_pipeline, stage_cache_checkpoints

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    started_at = datetime.now(UTC)

    cfg = PipelineConfig.model_validate(config_dict)
    layout = VolumeLayout(Path(MOUNT_PATH))
    layout.ensure_directories()
    configure_hf_cache_env(layout)
    hf_token = resolve_hf_token()

    # Cache and verify model checkpoints in /data/hf_cache
    provenance = stage_cache_checkpoints(layout=layout, cfg=cfg, hf_token=hf_token)
    volume.commit()

    reports_out: list[dict[str, Any]] = []
    ablations_out: list[dict[str, Any]] = []
    target_categories = categories if categories is not None else cfg.dataset.categories

    for category in target_categories:
        report, ablations, _ = run_category_pipeline(
            category=category,
            layout=layout,
            cfg=cfg,
            hf_token=hf_token,
            force=force,
            run_ablations=run_ablations,
            on_checkpoint_cb=volume.commit,
        )
        reports_out.append(report.model_dump(mode="json"))
        ablations_out.extend([a.model_dump(mode="json") for a in ablations])
        volume.commit()

    # Create an in-memory tar archive of `/data/bundles` and `/data/manifests` (uncompressed tar
    # since .npz and .png files are already compressed, avoiding slow level-9 gzip CPU overhead).
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w") as tar:
        if layout.bundles_dir.is_dir():
            for cat in target_categories:
                cat_bundle = layout.bundles_dir / cat
                if cat_bundle.is_dir():
                    tar.add(cat_bundle, arcname=f"bundles/{cat}")
        if layout.manifests_dir.is_dir():
            tar.add(layout.manifests_dir, arcname="manifests")

    completed_at = datetime.now(UTC)
    return {
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "hardware_type": "NVIDIA L4 (Modal)",
        "provenance": provenance,
        "reports": reports_out,
        "ablations": ablations_out,
        "bundles_tar_gz": tar_buf.getvalue(),
    }
