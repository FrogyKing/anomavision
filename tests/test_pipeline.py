"""Integration tests for pipeline stages, MLflow tracking, DVC lineage, and InferenceEngine."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from anomavision.config import PipelineConfig
from anomavision.datasets.mvtec import build_category_manifest
from anomavision.modal_storage import VolumeLayout
from anomavision.models.dinov3 import DenseFeatureBatch
from anomavision.pipeline.inference import InferenceEngine
from anomavision.pipeline.stages import (
    stage_build_memory_bank,
    stage_calibrate,
    stage_evaluate_and_ablate,
    stage_package_bundle,
)
from anomavision.pipeline.tracking import (
    log_experiment_to_mlflow,
    record_modal_cost_estimate,
    write_dvc_and_lineage_manifests,
)


class _DummyExtractor:
    """Fast deterministic feature extractor fixture for CPU pipeline integration tests."""

    def __init__(self, grid_size: tuple[int, int] = (8, 8), dim: int = 32) -> None:
        self.grid_size = grid_size
        self.dim = dim

    def extract_batch(self, images: list[Any]) -> DenseFeatureBatch:
        n_patches = self.grid_size[0] * self.grid_size[1]
        patches_list: list[np.ndarray] = []
        cls_list: list[np.ndarray] = []
        global_list: list[np.ndarray] = []
        sizes: list[tuple[int, int]] = []

        for img in images:
            arr = np.asarray(img, dtype=np.float32) if not isinstance(img, np.ndarray) else img.astype(np.float32)
            sizes.append((arr.shape[0], arr.shape[1]))
            rng = np.random.default_rng(int(np.sum(arr)) % 10000)
            p = rng.normal(loc=0.0, scale=0.08, size=(n_patches, self.dim)).astype(np.float32)
            p[:, 0] += 1.0
            # If the image has a bright white defect square in [16:32, 16:32], perturb center patches
            if np.mean(arr[16:32, 16:32]) > 240.0:
                p[27:37, 0] = -1.0
                p[27:37, 1] = 2.0
            p /= np.linalg.norm(p, axis=-1, keepdims=True)
            c = np.mean(p, axis=0)
            c /= np.linalg.norm(c)
            g = np.concatenate([c, c])
            g /= np.linalg.norm(g)
            patches_list.append(p)
            cls_list.append(c)
            global_list.append(g)

        return DenseFeatureBatch(
            patch_embeddings=np.stack(patches_list, axis=0),
            cls_embeddings=np.stack(cls_list, axis=0),
            global_embeddings=np.stack(global_list, axis=0),
            grid_size=self.grid_size,
            original_sizes=sizes,
        )


def test_full_pipeline_stages_and_inference_engine(tmp_path: Path) -> None:
    layout = VolumeLayout(tmp_path / "volume")
    layout.ensure_directories()

    # Build synthetic MVTec category in layout.mvtec_root
    cat_dir = layout.mvtec_root / "bottle"
    for sub in ("train/good", "test/good", "test/broken_large", "ground_truth/broken_large"):
        (cat_dir / sub).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    for i in range(10):
        arr = rng.integers(40, 120, size=(64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(cat_dir / "train/good" / f"{i:03d}.png")
    for i in range(3):
        arr = rng.integers(40, 120, size=(64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(cat_dir / "test/good" / f"{i:03d}.png")
    for i in range(3):
        arr = rng.integers(40, 120, size=(64, 64, 3), dtype=np.uint8)
        arr[16:32, 16:32] = 255  # Bright defect square
        Image.fromarray(arr).save(cat_dir / "test/broken_large" / f"{i:03d}.png")
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[16:32, 16:32] = 255
        Image.fromarray(mask).save(cat_dir / "ground_truth/broken_large" / f"{i:03d}_mask.png")

    cfg = PipelineConfig()
    cfg.dataset.categories = ["bottle"]
    cfg.sam.enabled = False

    manifest = build_category_manifest(
        "bottle",
        layout.mvtec_root,
        cfg.dataset,
        manifest_out_path=layout.manifests_dir / "bottle_manifest.json",
    )

    dummy_ext = _DummyExtractor(grid_size=(8, 8), dim=32)
    fp = cfg.fingerprint()
    emb_dir = layout.embeddings_dir / "bottle" / fp
    emb_dir.mkdir(parents=True, exist_ok=True)
    emb_file = emb_dir / "dense_embeddings.npz"

    # Generate synthetic cached embeddings matching the dummy extractor
    from anomavision.datasets.mvtec import MVTecCategoryDataset

    ds = MVTecCategoryDataset(layout.mvtec_root, manifest)
    tr_b = dummy_ext.extract_batch([ds.load_sample_image(r) for r in ds.get_split("train_fit")])
    cal_b = dummy_ext.extract_batch([ds.load_sample_image(r) for r in ds.get_split("calibration")])
    te_b = dummy_ext.extract_batch([ds.load_sample_image(r) for r in ds.get_split("test")])

    np.savez_compressed(
        emb_file,
        train_patches_raw=tr_b.patch_embeddings,
        train_global=tr_b.global_embeddings,
        train_sizes=np.asarray(tr_b.original_sizes, dtype=np.int32),
        cal_patches_raw=cal_b.patch_embeddings,
        cal_global=cal_b.global_embeddings,
        cal_sizes=np.asarray(cal_b.original_sizes, dtype=np.int32),
        test_patches_raw=te_b.patch_embeddings,
        test_global=te_b.global_embeddings,
        test_sizes=np.asarray(te_b.original_sizes, dtype=np.int32),
        test_sam_masks=np.ones((len(te_b.original_sizes), 64, 64), dtype=np.uint8),
        grid_size=np.asarray([8, 8], dtype=np.int32),
        test_latencies_ms=np.ones(len(te_b.original_sizes), dtype=np.float32) * 5.0,
    )
    (emb_dir / "extraction_telemetry.json").write_text(
        json.dumps(
            {
                "category": "bottle",
                "fingerprint": fp,
                "mean_inference_latency_ms": 5.0,
                "p95_inference_latency_ms": 6.0,
                "peak_gpu_memory_mb": 256.0,
            }
        ),
        encoding="utf-8",
    )

    detector, retrieval_index, coreset_res = stage_build_memory_bank(
        "bottle", manifest, emb_file, layout, cfg
    )
    cal_params, _, cal_scores = stage_calibrate("bottle", emb_file, detector, layout, cfg)
    report, ablations = stage_evaluate_and_ablate(
        category="bottle",
        manifest=manifest,
        emb_file=emb_file,
        detector=detector,
        retrieval_index=retrieval_index,
        coreset_res=coreset_res,
        cal_params=cal_params,
        cal_scores=cal_scores,
        layout=layout,
        cfg=cfg,
        run_ablations=True,
    )
    assert report.image_auroc.value >= 0.99
    assert len(ablations) == 7

    bundle_dir = stage_package_bundle("bottle", manifest, layout, cfg)
    assert (bundle_dir / "patchcore_coreset.npz").is_file()

    # Verify InferenceEngine loads bundle and predicts accurately
    engine = InferenceEngine(
        bundles_root=layout.bundles_dir,
        feature_extractor=dummy_ext,
    )
    assert engine.available_categories == ["bottle"]

    test_anom_img = ds.load_sample_image(ds.get_split("test")[0])  # broken_large/000
    pred_res = engine.predict(test_anom_img, category="bottle", include_visualizations=True)
    assert pred_res.is_anomalous is True
    assert pred_res.decision_label == "ANOMALOUS"
    assert len(pred_res.similar_references) > 0
    assert pred_res.heatmap_base64_png is not None

    # Verify MLflow logging, cost estimator, and DVC lineage writer
    t0 = datetime.now(UTC)
    t1 = t0 + timedelta(seconds=90)
    cost_rec = record_modal_cost_estimate(
        job_name="unit_test_job",
        started_at=t0,
        completed_at=t1,
        hardware_type="NVIDIA L4",
        hourly_rate_usd=0.80,
        artifacts_root=tmp_path / "artifacts",
    )
    assert np.isclose(cost_rec.estimated_cost_usd, 0.02, atol=1e-4)

    run_id = log_experiment_to_mlflow(
        cfg=cfg,
        reports=[report],
        ablations=ablations,
        artifacts_root=tmp_path / "artifacts",
    )
    assert len(run_id) > 8

    lineage_path = write_dvc_and_lineage_manifests(
        cfg=cfg,
        reports=[report],
        artifacts_root=tmp_path / "artifacts",
        project_root=tmp_path,
    )
    assert lineage_path.is_file()
    assert (tmp_path / "dvc.yaml").is_file()
