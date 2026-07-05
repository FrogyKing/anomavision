"""Typed configuration models for AnomaVision pipelines, models, and infrastructure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class BackboneConfig(BaseModel):
    """DINOv3 ViT-S/16 feature extraction configuration."""

    model_id: str = Field(
        default="affineisallyouneed/dinov3-vits16-pretrain-lvd1689m",
        description="Public Hugging Face repository for DINOv3 ViT-S/16.",
    )
    revision: str = Field(
        default="1c33e2c713df1be43909a792e3ce324109412c21",
        description="Pinned commit SHA for deterministic checkpoint provenance.",
    )
    patch_size: int = Field(default=16, ge=8, le=32)
    hidden_size: int = Field(default=384, ge=64)
    num_register_tokens: int = Field(default=4, ge=0)
    image_size: int = Field(
        default=448,
        description="Input resolution (must be a multiple of patch_size). 448 yields a 28x28=784 patch grid.",
    )
    extract_layers: tuple[int, ...] = Field(
        default=(6, 9, 12),
        description="1-indexed transformer block outputs to concatenate/aggregate for multi-scale patch representations.",
    )
    layer_aggregation: Literal["mean", "concat"] = Field(
        default="mean",
        description="How multi-layer features are combined before local neighborhood smoothing.",
    )
    neighborhood_kernel: int = Field(
        default=3,
        ge=1,
        le=7,
        description="Spatial average pooling kernel over patch grid (PatchCore local neighborhood aggregation).",
    )
    l2_normalize: bool = Field(
        default=True,
        description="Project patch descriptors onto the unit hypersphere S^(d-1).",
    )
    batch_size: int = Field(default=16, ge=1, le=128)
    amp_dtype: Literal["bfloat16", "float16", "float32"] = Field(default="bfloat16")
    image_mean: tuple[float, float, float] = Field(default=(0.485, 0.456, 0.406))
    image_std: tuple[float, float, float] = Field(default=(0.229, 0.224, 0.225))

    @field_validator("image_size")
    @classmethod
    def validate_multiple_of_16(cls, v: int) -> int:
        if v <= 0 or v % 16 != 0:
            raise ValueError(f"image_size must be a positive multiple of 16, got {v}")
        return v

    @field_validator("neighborhood_kernel")
    @classmethod
    def validate_odd_kernel(cls, v: int) -> int:
        if v % 2 == 0:
            raise ValueError(f"neighborhood_kernel must be odd, got {v}")
        return v

    @property
    def grid_size(self) -> tuple[int, int]:
        side = self.image_size // self.patch_size
        return (side, side)

    @property
    def num_patches(self) -> int:
        h, w = self.grid_size
        return h * w

    @property
    def embedding_dim(self) -> int:
        if self.layer_aggregation == "concat":
            return self.hidden_size * len(self.extract_layers)
        return self.hidden_size


class SAMConfig(BaseModel):
    """SAM 3 and optional SAM 3.1 segmentation configuration for foreground object masking."""

    enabled: bool = Field(default=True)
    primary_model_id: str = Field(
        default="tkclam/sam3",
        description="Primary SAM 3 image/object segmentation repository in Transformers format.",
    )
    primary_revision: str = Field(
        default="4c7c7aa68a625b356934f4ed90597de58b74f3cf",
        description="Pinned commit SHA for tkclam/sam3.",
    )
    primary_weights_sha256: str = Field(
        default="6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a",
        description="Expected SHA-256 of model.safetensors in tkclam/sam3.",
    )
    optional_sam31_model_id: str = Field(
        default="AEmotionStudio/sam3.1",
        description="Optional SAM 3.1 multiplex repository (distinct video-multiplex architecture).",
    )
    optional_sam31_revision: str = Field(
        default="694239a1479aab8fd1317c87c433c58acd7c6eab",
        description="Pinned commit SHA for AEmotionStudio/sam3.1.",
    )
    adapter_variant: Literal["sam3", "sam3.1", "disabled"] = Field(default="sam3")
    score_threshold: float = Field(default=0.35, ge=0.05, le=0.99)
    mask_dilation_px: int = Field(
        default=5,
        ge=0,
        le=32,
        description="Morphological dilation applied to foreground object mask to retain boundary defects.",
    )
    background_suppression_factor: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Multiplier applied to anomaly scores outside the SAM foreground mask.",
    )


class DatasetConfig(BaseModel):
    """MVTec AD dataset and deterministic split configuration."""

    hf_dataset_id: str = Field(
        default="foersben/mvtec-ad",
        description="Authorized mirror of MVTec AD with unmodified directory tree and PNG masks.",
    )
    hf_revision: str = Field(
        default="c75b39616f84db43677bcc8228caaafaf5096d7f",
        description="Pinned commit SHA for dataset provenance.",
    )
    official_source_url: str = Field(
        default="https://www.mvtec.com/company/research/datasets/mvtec-ad",
        description="Upstream MVTec AD dataset attribution URL.",
    )
    license_id: str = Field(default="CC-BY-NC-SA-4.0")
    categories: list[str] = Field(default_factory=lambda: ["bottle", "metal_nut"])
    calibration_fraction: float = Field(
        default=0.15,
        ge=0.05,
        le=0.40,
        description="Fraction of normal training images deterministically reserved for unsupervised threshold calibration.",
    )
    seed: int = Field(default=42)


class CoresetConfig(BaseModel):
    """Greedy k-center coreset memory bank and nearest-neighbor scoring configuration."""

    coreset_ratio: float = Field(
        default=0.10,
        gt=0.0,
        le=1.0,
        description="Fraction of total training patch embeddings selected for the memory bank M.",
    )
    max_coreset_size: int = Field(default=15000, ge=128)
    projection_dim: int = Field(
        default=128,
        ge=16,
        le=1024,
        description="Johnson-Lindenstrauss random projection dimension for fast k-center distance updates.",
    )
    metric: Literal["cosine", "euclidean"] = Field(
        default="cosine",
        description="Distance metric used for patch-to-memory nearest neighbor scoring.",
    )
    n_neighbors: int = Field(
        default=9,
        ge=1,
        le=32,
        description="Number of nearest neighbors in M used for density-reweighted image anomaly scoring.",
    )
    score_top_k_patches: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Top-k highest-scoring patches averaged after local re-weighting for stable image-level score.",
    )
    gaussian_sigma: float = Field(
        default=4.0,
        ge=0.0,
        le=16.0,
        description="Gaussian smoothing sigma (in pixels) when upsampling patch anomaly maps to image resolution.",
    )
    seed: int = Field(default=42)


class ModalConfig(BaseModel):
    """Modal cloud execution, persistent Volume, and cost estimation settings."""

    app_name: str = Field(default="anomavision-pipeline")
    volume_name: str = Field(default="anomavision-data")
    mount_path: str = Field(default="/data")
    hf_secret_name: str = Field(default="huggingface")
    gpu_type: str = Field(default="L4")
    gpu_hourly_rate_usd: float = Field(
        default=0.80,
        description="Estimated Modal NVIDIA L4 hourly rate in USD for cost telemetry.",
    )
    cpu_hourly_rate_usd: float = Field(
        default=0.15,
        description="Estimated Modal CPU container hourly rate in USD.",
    )
    budget_max_usd: float = Field(default=30.0)
    budget_target_usd: float = Field(default=15.0)
    timeout_seconds: int = Field(default=1800)


class PipelineConfig(BaseModel):
    """Root configuration combining all AnomaVision subsystems."""

    backbone: BackboneConfig = Field(default_factory=BackboneConfig)
    sam: SAMConfig = Field(default_factory=SAMConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    coreset: CoresetConfig = Field(default_factory=CoresetConfig)
    modal: ModalConfig = Field(default_factory=ModalConfig)
    artifacts_dir: Path = Field(default=Path("artifacts"))

    def fingerprint(self) -> str:
        """Compute deterministic SHA-256 fingerprint of algorithmic configuration."""
        payload = {
            "backbone": self.backbone.model_dump(mode="json"),
            "sam": self.sam.model_dump(mode="json"),
            "dataset": self.dataset.model_dump(mode="json"),
            "coreset": self.coreset.model_dump(mode="json"),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def to_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads(self.model_dump_json())
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    @classmethod
    def from_yaml(cls, path: Path) -> PipelineConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(raw)
