"""Unit and integration tests for FastAPI inference service, Prometheus metrics, and drift monitoring."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from anomavision.api.app import create_app
from anomavision.config import PipelineConfig
from anomavision.datasets.mvtec import build_category_manifest
from anomavision.modal_storage import VolumeLayout
from anomavision.pipeline.inference import InferenceEngine
from anomavision.pipeline.stages import (
    stage_build_memory_bank,
    stage_calibrate,
    stage_evaluate_and_ablate,
    stage_package_bundle,
)
from tests.test_pipeline import _DummyExtractor


def _build_test_engine(tmp_path: Path) -> InferenceEngine:
    layout = VolumeLayout(tmp_path / "vol")
    layout.ensure_directories()
    cat_dir = layout.mvtec_root / "bottle"
    for sub in ("train/good", "test/good", "test/broken_large", "ground_truth/broken_large"):
        (cat_dir / sub).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    for i in range(8):
        Image.fromarray(rng.integers(40, 120, size=(64, 64, 3), dtype=np.uint8)).save(
            cat_dir / "train/good" / f"{i:03d}.png"
        )
    for i in range(2):
        Image.fromarray(rng.integers(40, 120, size=(64, 64, 3), dtype=np.uint8)).save(
            cat_dir / "test/good" / f"{i:03d}.png"
        )
    for i in range(2):
        arr = rng.integers(40, 120, size=(64, 64, 3), dtype=np.uint8)
        arr[16:32, 16:32] = 255
        Image.fromarray(arr).save(cat_dir / "test/broken_large" / f"{i:03d}.png")
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[16:32, 16:32] = 255
        Image.fromarray(mask).save(cat_dir / "ground_truth/broken_large" / f"{i:03d}_mask.png")

    cfg = PipelineConfig()
    cfg.dataset.categories = ["bottle"]
    cfg.sam.enabled = False
    manifest = build_category_manifest("bottle", layout.mvtec_root, cfg.dataset)

    dummy_ext = _DummyExtractor(grid_size=(8, 8), dim=32)
    fp = cfg.fingerprint()
    emb_dir = layout.embeddings_dir / "bottle" / fp
    emb_dir.mkdir(parents=True, exist_ok=True)
    emb_file = emb_dir / "dense_embeddings.npz"

    from anomavision.datasets.mvtec import MVTecCategoryDataset

    ds = MVTecCategoryDataset(layout.mvtec_root, manifest)
    tr_b = dummy_ext.extract_batch([ds.load_sample_image(r) for r in ds.get_split("train_fit")])
    cal_b = dummy_ext.extract_batch([ds.load_sample_image(r) for r in ds.get_split("calibration")])
    te_b = dummy_ext.extract_batch([ds.load_sample_image(r) for r in ds.get_split("test")])

    np.savez(
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
        test_latencies_ms=np.ones(len(te_b.original_sizes), dtype=np.float32) * 4.0,
    )
    (emb_dir / "extraction_telemetry.json").write_text(
        '{"category":"bottle","fingerprint":"x","mean_inference_latency_ms":4.0,"p95_inference_latency_ms":5.0,"peak_gpu_memory_mb":128.0}',
        encoding="utf-8",
    )

    detector, retrieval_index, coreset_res = stage_build_memory_bank(
        "bottle", manifest, emb_file, layout, cfg
    )
    cal_params, _, cal_scores = stage_calibrate("bottle", emb_file, detector, layout, cfg)
    stage_evaluate_and_ablate(
        "bottle",
        manifest,
        emb_file,
        detector,
        retrieval_index,
        coreset_res,
        cal_params,
        cal_scores,
        layout,
        cfg,
        run_ablations=False,
    )
    stage_package_bundle("bottle", manifest, layout, cfg)
    return InferenceEngine(bundles_root=layout.bundles_dir, feature_extractor=dummy_ext)


def test_fastapi_endpoints_and_prometheus_metrics(tmp_path: Path) -> None:
    engine = _build_test_engine(tmp_path)
    app = create_app(engine=engine)
    client = TestClient(app)

    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    r_ready = client.get("/ready")
    assert r_ready.status_code == 200
    assert "bottle" in r_ready.json()["available_categories"]

    r_meta = client.get("/metadata")
    assert r_meta.status_code == 200
    assert "bottle" in r_meta.json()["categories"]

    # Create synthetic defective image payload
    arr = np.full((64, 64, 3), 80, dtype=np.uint8)
    arr[16:32, 16:32] = 255
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    img_bytes = buf.getvalue()

    r_pred = client.post(
        "/predict",
        files={"file": ("defect.png", img_bytes, "image/png")},
        data={"category": "bottle", "top_k_similar": "3", "include_visualizations": "true"},
    )
    assert r_pred.status_code == 200
    payload = r_pred.json()
    assert payload["category"] == "bottle"
    assert payload["decision_label"] == "ANOMALOUS"
    assert len(payload["similar_references"]) == 3
    assert payload["heatmap_base64_png"] is not None

    r_ret = client.post(
        "/retrieve",
        files={"file": ("defect.png", img_bytes, "image/png")},
        data={"category": "bottle", "top_k": "2"},
    )
    assert r_ret.status_code == 200
    assert len(r_ret.json()) == 2

    # Error handling checks
    r_bad_cat = client.post(
        "/predict",
        files={"file": ("defect.png", img_bytes, "image/png")},
        data={"category": "nonexistent_cat"},
    )
    assert r_bad_cat.status_code == 404

    r_bad_img = client.post(
        "/predict",
        files={"file": ("corrupt.png", b"not_an_image", "image/png")},
        data={"category": "bottle"},
    )
    assert r_bad_img.status_code == 400

    r_drift = client.get("/monitoring/drift")
    assert r_drift.status_code == 200
    assert r_drift.json()["categories"]["bottle"]["window_samples"] == 1

    r_metrics = client.get("/metrics")
    assert r_metrics.status_code == 200
    assert "anomavision_requests_total" in r_metrics.text
    assert "anomavision_inference_latency_seconds" in r_metrics.text
