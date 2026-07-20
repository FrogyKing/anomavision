"""Unit tests for MVTec AD dataset validation, deterministic splitting, and anti-leakage checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from anomavision.config import DatasetConfig
from anomavision.datasets.mvtec import (
    MVTecCategoryDataset,
    build_category_manifest,
    verify_no_split_leakage,
)
from anomavision.modal_storage import VolumeLayout, resolve_hf_token


def _create_synthetic_mvtec_category(root: Path, category: str = "bottle") -> Path:
    cat_dir = root / category
    train_good = cat_dir / "train" / "good"
    test_good = cat_dir / "test" / "good"
    test_broken = cat_dir / "test" / "broken_large"
    gt_broken = cat_dir / "ground_truth" / "broken_large"

    for d in (train_good, test_good, test_broken, gt_broken):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(123)
    for i in range(10):
        arr = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(train_good / f"{i:03d}.png")

    for i in range(3):
        arr = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(test_good / f"{i:03d}.png")

    for i in range(4):
        arr = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(test_broken / f"{i:03d}.png")
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[16:32, 16:32] = 255
        Image.fromarray(mask).save(gt_broken / f"{i:03d}_mask.png")

    return cat_dir


def test_build_category_manifest_and_splits(tmp_path: Path) -> None:
    _create_synthetic_mvtec_category(tmp_path, "bottle")
    cfg = DatasetConfig(categories=["bottle"], calibration_fraction=0.2, seed=42)
    manifest_path = tmp_path / "manifests" / "bottle_manifest.json"

    manifest = build_category_manifest("bottle", tmp_path, cfg, manifest_out_path=manifest_path)
    assert manifest_path.is_file()
    assert manifest.num_train_fit == 8
    assert manifest.num_calibration == 2
    assert manifest.num_test_good == 3
    assert manifest.num_test_anomalous == 4
    assert manifest.defect_types == ["broken_large", "good"]

    # Deterministic repeatability check
    manifest2 = build_category_manifest("bottle", tmp_path, cfg)
    assert manifest.manifest_sha256 == manifest2.manifest_sha256

    ds = MVTecCategoryDataset(tmp_path, manifest)
    assert len(ds.get_split("train_fit")) == 8
    assert len(ds.get_split("calibration")) == 2
    test_samples = ds.get_split("test")
    assert len(test_samples) == 7

    for sample in test_samples:
        img = ds.load_sample_image(sample)
        mask = ds.load_sample_mask(sample)
        assert img.shape == (64, 64, 3)
        assert mask.shape == (64, 64)
        if sample.is_anomalous:
            assert mask.max() == 1
        else:
            assert mask.max() == 0


def test_leakage_detection_raises(tmp_path: Path) -> None:
    _create_synthetic_mvtec_category(tmp_path, "bottle")
    cfg = DatasetConfig(categories=["bottle"], calibration_fraction=0.2, seed=42)
    manifest = build_category_manifest("bottle", tmp_path, cfg)

    # Inject duplicate SHA-256 across splits to ensure leakage detector catches it
    manifest.samples[0].split = "test"
    manifest.samples[1].image_sha256 = manifest.samples[0].image_sha256
    manifest.samples[1].split = "train_fit"

    with pytest.raises(ValueError, match="leakage detected"):
        verify_no_split_leakage(manifest)


def test_volume_layout_and_secret_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout = VolumeLayout(tmp_path / "data")
    layout.ensure_directories()
    assert layout.mvtec_root.is_dir()
    assert layout.hf_cache_dir.is_dir()
    assert layout.bundles_dir.is_dir()

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_test_dummy_token_do_not_log")
    resolved = resolve_hf_token()
    assert resolved == "hf_test_dummy_token_do_not_log"
