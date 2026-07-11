"""Unit tests for configuration validation and deterministic fingerprinting."""

from __future__ import annotations

from pathlib import Path

import pytest

from anomavision.config import BackboneConfig, PipelineConfig


def test_backbone_config_properties() -> None:
    cfg = BackboneConfig(image_size=448, patch_size=16, hidden_size=384)
    assert cfg.grid_size == (28, 28)
    assert cfg.num_patches == 784
    assert cfg.embedding_dim == 384

    cfg_concat = BackboneConfig(
        image_size=224,
        patch_size=16,
        hidden_size=384,
        extract_layers=(6, 9, 12),
        layer_aggregation="concat",
    )
    assert cfg_concat.grid_size == (14, 14)
    assert cfg_concat.num_patches == 196
    assert cfg_concat.embedding_dim == 384 * 3


def test_backbone_invalid_image_size_rejected() -> None:
    with pytest.raises(ValueError, match="multiple of 16"):
        BackboneConfig(image_size=250)


def test_backbone_invalid_kernel_rejected() -> None:
    with pytest.raises(ValueError, match="must be odd"):
        BackboneConfig(neighborhood_kernel=4)


def test_pipeline_config_yaml_roundtrip(tmp_path: Path) -> None:
    cfg = PipelineConfig()
    yaml_path = tmp_path / "config.yaml"
    cfg.to_yaml(yaml_path)
    loaded = PipelineConfig.from_yaml(yaml_path)
    assert loaded.fingerprint() == cfg.fingerprint()
    assert loaded.backbone.model_id == "affineisallyouneed/dinov3-vits16-pretrain-lvd1689m"
    assert loaded.sam.primary_model_id == "tkclam/sam3"
    assert loaded.sam.optional_sam31_model_id == "AEmotionStudio/sam3.1"
