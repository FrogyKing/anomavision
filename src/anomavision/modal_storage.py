"""Persistent Modal Volume directory layout, secret resolution, and synchronization utilities."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import modal

from anomavision.config import ModalConfig

logger = logging.getLogger(__name__)

CANDIDATE_HF_ENV_VARS: tuple[str, ...] = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HF_API_TOKEN",
)


@dataclass(frozen=True)
class VolumeLayout:
    """Canonical directory hierarchy inside the persistent Modal Volume (`anomavision-data`)."""

    root: Path

    @property
    def mvtec_root(self) -> Path:
        return self.root / "mvtec_ad"

    @property
    def manifests_dir(self) -> Path:
        return self.root / "manifests"

    @property
    def hf_cache_dir(self) -> Path:
        return self.root / "hf_cache"

    @property
    def embeddings_dir(self) -> Path:
        return self.root / "embeddings"

    @property
    def memory_banks_dir(self) -> Path:
        return self.root / "memory_banks"

    @property
    def calibration_dir(self) -> Path:
        return self.root / "calibration"

    @property
    def experiments_dir(self) -> Path:
        return self.root / "experiments"

    @property
    def visualizations_dir(self) -> Path:
        return self.root / "visualizations"

    @property
    def bundles_dir(self) -> Path:
        return self.root / "bundles"

    def ensure_directories(self) -> None:
        """Idempotently create all top-level directories inside the mounted Volume."""
        for directory in (
            self.mvtec_root,
            self.manifests_dir,
            self.hf_cache_dir,
            self.embeddings_dir,
            self.memory_banks_dir,
            self.calibration_dir,
            self.experiments_dir,
            self.visualizations_dir,
            self.bundles_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def resolve_hf_token() -> str | None:
    """Inspect environment variables injected by modal.Secret.from_name('huggingface') without logging values."""
    for env_name in CANDIDATE_HF_ENV_VARS:
        val = os.environ.get(env_name)
        if val and val.strip():
            token = val.strip()
            os.environ["HF_TOKEN"] = token
            logger.info("Authenticated Hugging Face token detected via environment variable '%s'", env_name)
            return token
    logger.info("No Hugging Face token environment variable found; proceeding with public Hub access")
    return None


def configure_hf_cache_env(layout: VolumeLayout) -> None:
    """Point Hugging Face Hub and Transformers cache directories to the persistent Modal Volume."""
    cache_path = str(layout.hf_cache_dir)
    os.environ["HF_HOME"] = cache_path
    os.environ["HF_HUB_CACHE"] = str(layout.hf_cache_dir / "hub")
    os.environ["TORCH_HOME"] = str(layout.hf_cache_dir / "torch")


def get_modal_volume(cfg: ModalConfig) -> modal.Volume:
    """Create or reuse the persistent Modal Volume `anomavision-data`."""
    return modal.Volume.from_name(cfg.volume_name, create_if_missing=True)


def get_hf_secret(cfg: ModalConfig) -> modal.Secret:
    """Reference the existing Modal Secret `huggingface`."""
    return modal.Secret.from_name(cfg.hf_secret_name)
