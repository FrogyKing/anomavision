"""Unit tests for greedy k-center coreset selection, anomaly scoring geometry, FAISS retrieval, and MMD drift."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anomavision.config import CoresetConfig
from anomavision.core.coreset import select_greedy_kcenter_coreset
from anomavision.core.detector import (
    GlobalCentroidBaseline,
    PatchCoreDetector,
    compute_pairwise_distances,
)
from anomavision.core.drift import DriftReferenceStats, compute_rbf_mmd_squared
from anomavision.core.retrieval import ReferenceImageSimilarityIndex
from anomavision.schemas import SampleRecord


def test_greedy_kcenter_monotone_covering_radius() -> None:
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(500, 64)).astype(np.float32)

    res = select_greedy_kcenter_coreset(
        embeddings=embeddings,
        coreset_ratio=0.10,
        projection_dim=32,
        seed=42,
        device="cpu",
    )
    assert res.coreset_embeddings.shape == (50, 64)
    assert len(np.unique(res.selected_indices)) == 50
    # In greedy k-center, the covering radius max_z min_{m in M_t} d(z, m) is non-increasing with t
    diffs = np.diff(res.covering_radii)
    assert np.all(diffs <= 1e-5)


def test_cosine_vs_euclidean_isometry_on_unit_sphere() -> None:
    rng = np.random.default_rng(7)
    q = rng.normal(size=(20, 32)).astype(np.float32)
    m = rng.normal(size=(40, 32)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    m /= np.linalg.norm(m, axis=1, keepdims=True)

    d_cos = compute_pairwise_distances(q, m, metric="cosine")
    d_l2 = compute_pairwise_distances(q, m, metric="euclidean")
    # On S^(d-1), ||u - v||_2^2 = 2 * (1 - <u, v>) = 2 * d_cos(u, v)
    assert np.allclose(d_l2 * d_l2, 2.0 * d_cos, atol=1e-4)


def test_patchcore_detector_separates_anomalies_and_persists(tmp_path: Path) -> None:
    rng = np.random.default_rng(101)
    # 12 normal training images, 8x8=64 patches, dim=32 clustered near unit vector e_0
    train_patches = rng.normal(loc=0.0, scale=0.1, size=(12, 64, 32)).astype(np.float32)
    train_patches[:, :, 0] += 1.0
    train_patches /= np.linalg.norm(train_patches, axis=-1, keepdims=True)

    cfg = CoresetConfig(coreset_ratio=0.2, metric="cosine", n_neighbors=5, gaussian_sigma=2.0)
    detector = PatchCoreDetector(cfg)
    detector.fit(train_patches, device="cpu")

    # Normal test image vs anomalous test image with local defect in patch (4, 4)
    normal_test = rng.normal(loc=0.0, scale=0.1, size=(1, 64, 32)).astype(np.float32)
    normal_test[:, :, 0] += 1.0
    normal_test /= np.linalg.norm(normal_test, axis=-1, keepdims=True)

    anom_test = normal_test.copy()
    defect_patch_idx = 4 * 8 + 4  # center of 8x8 grid
    anom_test[0, defect_patch_idx, :] = 0.0
    anom_test[0, defect_patch_idx, 1] = -1.0  # Orthogonal/opposite direction

    batch_in = np.concatenate([normal_test, anom_test], axis=0)
    pred = detector.predict_batch(
        patch_embeddings=batch_in,
        grid_size=(8, 8),
        original_sizes=[(64, 64), (64, 64)],
    )
    assert pred.image_scores[1] > pred.image_scores[0] * 3.0
    assert pred.anomaly_maps[1].shape == (64, 64)
    # Peak of upsampled anomaly map should align with center region (around pixel 36, 36)
    peak_y, peak_x = np.unravel_index(int(np.argmax(pred.anomaly_maps[1])), (64, 64))
    assert 24 <= peak_y <= 48
    assert 24 <= peak_x <= 48

    # Save & reload check
    save_path = tmp_path / "coreset.npz"
    detector.save(save_path)
    reloaded = PatchCoreDetector.load(save_path, cfg)
    pred2 = reloaded.predict_batch(batch_in, grid_size=(8, 8), original_sizes=[(64, 64), (64, 64)])
    assert np.allclose(pred.image_scores, pred2.image_scores, atol=1e-5)

    # Also test GlobalCentroidBaseline
    train_global = rng.normal(size=(12, 64)).astype(np.float32)
    test_global = rng.normal(size=(2, 64)).astype(np.float32)
    baseline = GlobalCentroidBaseline()
    baseline.fit(train_patches, train_global)
    b_scores, b_maps = baseline.predict_batch(
        patch_embeddings=batch_in,
        global_embeddings=test_global,
        grid_size=(8, 8),
        original_sizes=[(64, 64), (64, 64)],
    )
    assert b_scores.shape == (2,)
    assert len(b_maps) == 2


def test_faiss_retrieval_anti_leakage_and_ranking(tmp_path: Path) -> None:
    dim = 32
    rng = np.random.default_rng(55)
    embs = rng.normal(size=(5, dim)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)

    records = [
        SampleRecord(
            sample_id=f"bottle/train/good/{i:03d}",
            category="bottle",
            split="train_fit",
            defect_type="good",
            is_anomalous=False,
            image_relpath=f"bottle/train/good/{i:03d}.png",
            image_sha256=f"sha_{i}",
            width=64,
            height=64,
        )
        for i in range(5)
    ]

    index = ReferenceImageSimilarityIndex(dim=dim)
    index.fit(embs, records)
    neighbors, latency_ms = index.query(embs[2], top_k=3)
    assert len(neighbors) == 3
    assert neighbors[0].sample_id == "bottle/train/good/002"
    assert np.isclose(neighbors[0].similarity_score, 1.0, atol=1e-5)
    assert latency_ms >= 0.0

    # Save and load roundtrip
    idx_path = tmp_path / "ref.faiss"
    meta_path = tmp_path / "ref_meta.json"
    index.save(idx_path, meta_path)
    loaded = ReferenceImageSimilarityIndex.load(idx_path, meta_path)
    neighbors2, _ = loaded.query(embs[2], top_k=3)
    assert neighbors2[0].sample_id == "bottle/train/good/002"

    # Anti-leakage check: test or anomalous record must raise ValueError
    leaked_record = records[0].model_copy(update={"split": "test", "is_anomalous": True})
    with pytest.raises(ValueError, match="Anti-leakage violation"):
        index.fit(embs[:1], [leaked_record])


def test_mmd_drift_detection() -> None:
    rng = np.random.default_rng(99)
    train_patches = rng.normal(loc=0.0, scale=0.2, size=(400, 16)).astype(np.float32)
    cal_patches = rng.normal(loc=0.0, scale=0.2, size=(5, 64, 16)).astype(np.float32)

    stats = DriftReferenceStats.from_reference_patches(train_patches, cal_patches, seed=42)
    in_dist_query = rng.normal(loc=0.0, scale=0.2, size=(64, 16)).astype(np.float32)
    shifted_query = rng.normal(loc=2.5, scale=0.2, size=(64, 16)).astype(np.float32)

    mmd_in, alert_in = stats.score_query_patches(in_dist_query)
    mmd_shift, alert_shift = stats.score_query_patches(shifted_query)
    assert mmd_shift > mmd_in * 5.0
    assert not alert_in
    assert alert_shift
    assert compute_rbf_mmd_squared(in_dist_query, in_dist_query) < 1e-5
