"""DINOv3 ViT-S/16 dense patch feature extractor with register-token handling and local aggregation."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from anomavision.config import BackboneConfig

logger = logging.getLogger(__name__)


@dataclass
class DenseFeatureBatch:
    """Output container for a batch of extracted DINOv3 representations."""

    patch_embeddings: np.ndarray  # Shape: (B, N_patches, d), float32
    cls_embeddings: np.ndarray  # Shape: (B, d_hidden), float32
    global_embeddings: np.ndarray  # Shape: (B, 2 * d_hidden), float32 (L2-normalized [CLS || mean_patch])
    grid_size: tuple[int, int]  # (H_p, W_p)
    original_sizes: list[tuple[int, int]]  # List of (H_orig, W_orig)


def process_token_sequence(
    layer_tokens: list[torch.Tensor],
    num_register_tokens: int,
    grid_size: tuple[int, int],
    layer_aggregation: str = "mean",
    neighborhood_kernel: int = 3,
    l2_normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transform raw ViT sequence tokens into locally aggregated spatial patch descriptors.

    Given sequence output T_l in R^(B x (1 + R + H_p*W_p) x C) from transformer layer l:
      1. Index 0 is the [CLS] token.
      2. Indices 1 .. R are the R=4 register tokens (discarded for spatial anomaly localization).
      3. Indices (1 + R) .. end are the N = H_p * W_p spatial patch tokens.
      4. Each layer's patch tokens are layer-normalized across channels, reshaped to (B, C, H_p, W_p),
         and smoothed with a 2D average pooling kernel of size p x p (stride 1, count_include_pad=False)
         to incorporate local geometric context (Roth et al., PatchCore CVPR 2022).
      5. Multi-layer features are aggregated via mean or channel concatenation and optionally
         projected onto the unit hypersphere S^(d-1).
    """
    if not layer_tokens:
        raise ValueError("layer_tokens must contain at least one layer tensor")

    h_p, w_p = grid_size
    expected_patches = h_p * w_p
    expected_seq_len = 1 + num_register_tokens + expected_patches

    pooled_layers: list[torch.Tensor] = []
    cls_layers: list[torch.Tensor] = []

    for tok in layer_tokens:
        if tok.ndim != 3:
            raise ValueError(f"Expected 3D token tensor (B, L, C), got shape {tuple(tok.shape)}")
        b, seq_len, c = tok.shape
        if seq_len != expected_seq_len:
            raise ValueError(
                f"Sequence length mismatch: got {seq_len}, expected {expected_seq_len} "
                f"(1 CLS + {num_register_tokens} registers + {expected_patches} patches for grid {grid_size})"
            )

        # Normalize layer representations so multi-depth features have matched variance
        norm_tok = F.layer_norm(tok.float(), (c,))
        cls_tok = norm_tok[:, 0, :]
        patch_tok = norm_tok[:, 1 + num_register_tokens :, :]

        # Reshape to spatial grid (B, C, H_p, W_p)
        spatial = patch_tok.transpose(1, 2).contiguous().view(b, c, h_p, w_p)

        # Local neighborhood aggregation (PatchCore locally aware patch features)
        if neighborhood_kernel > 1:
            pad = neighborhood_kernel // 2
            spatial = F.avg_pool2d(
                spatial,
                kernel_size=neighborhood_kernel,
                stride=1,
                padding=pad,
                count_include_pad=False,
            )

        # Flatten back to (B, N, C)
        flat_patches = spatial.view(b, c, expected_patches).transpose(1, 2).contiguous()
        pooled_layers.append(flat_patches)
        cls_layers.append(cls_tok)

    if layer_aggregation == "concat":
        patch_features = torch.cat(pooled_layers, dim=-1)
    elif layer_aggregation == "mean":
        patch_features = torch.stack(pooled_layers, dim=0).mean(dim=0)
    else:
        raise ValueError(f"Unsupported layer_aggregation: {layer_aggregation}")

    cls_features = torch.stack(cls_layers, dim=0).mean(dim=0)
    mean_patch = patch_features.mean(dim=1)

    if l2_normalize:
        patch_features = F.normalize(patch_features, p=2, dim=-1, eps=1e-12)
        cls_features = F.normalize(cls_features, p=2, dim=-1, eps=1e-12)

    global_features = torch.cat(
        [
            F.normalize(cls_features, p=2, dim=-1, eps=1e-12),
            F.normalize(mean_patch, p=2, dim=-1, eps=1e-12),
        ],
        dim=-1,
    )
    global_features = F.normalize(global_features, p=2, dim=-1, eps=1e-12)

    return patch_features, cls_features, global_features


class DINOv3FeatureExtractor:
    """Frozen DINOv3 ViT-S/16 backbone wrapper for deterministic dense patch embedding extraction."""

    def __init__(
        self,
        cfg: BackboneConfig,
        device: str | torch.device | None = None,
        hf_token: str | None = None,
    ) -> None:
        self.cfg = cfg
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(
            "Loading DINOv3 backbone '%s' (revision=%s) on device=%s",
            cfg.model_id,
            cfg.revision[:10],
            self.device,
        )
        self.processor = AutoImageProcessor.from_pretrained(
            cfg.model_id,
            revision=cfg.revision,
            token=hf_token,
        )
        self.model = AutoModel.from_pretrained(
            cfg.model_id,
            revision=cfg.revision,
            output_hidden_states=True,
            token=hf_token,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.to(self.device)

        # Verify model configuration matches expectations
        model_cfg = self.model.config
        actual_registers = getattr(model_cfg, "num_register_tokens", cfg.num_register_tokens)
        actual_patch = getattr(model_cfg, "patch_size", cfg.patch_size)
        if actual_registers != cfg.num_register_tokens or actual_patch != cfg.patch_size:
            raise RuntimeError(
                f"Backbone config mismatch: checkpoint has patch_size={actual_patch}, "
                f"num_register_tokens={actual_registers}, expected {cfg.patch_size} and {cfg.num_register_tokens}"
            )

    def preprocess_batch(
        self, images: Sequence[np.ndarray | Image.Image]
    ) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        """Deterministically resize and normalize a batch of images to (B, 3, image_size, image_size)."""
        pil_images: list[Image.Image] = []
        original_sizes: list[tuple[int, int]] = []
        target_hw = (self.cfg.image_size, self.cfg.image_size)

        for img in images:
            if isinstance(img, np.ndarray):
                if img.ndim == 2:
                    pil_img = Image.fromarray(img).convert("RGB")
                else:
                    pil_img = Image.fromarray(img.astype(np.uint8)).convert("RGB")
            else:
                pil_img = img.convert("RGB")
            original_sizes.append((pil_img.height, pil_img.width))
            if (pil_img.height, pil_img.width) != target_hw:
                pil_img = pil_img.resize((target_hw[1], target_hw[0]), resample=Image.Resampling.BICUBIC)
            pil_images.append(pil_img)

        # Convert to float32 tensor in [0, 1] and apply ImageNet normalization
        tensors: list[torch.Tensor] = []
        mean = torch.tensor(self.cfg.image_mean, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(self.cfg.image_std, dtype=torch.float32).view(3, 1, 1)

        for pil_img in pil_images:
            arr = np.asarray(pil_img, dtype=np.float32) / 255.0
            chw = torch.from_numpy(arr).permute(2, 0, 1)
            tensors.append((chw - mean) / std)

        batch_tensor = torch.stack(tensors, dim=0)
        return batch_tensor, original_sizes

    def extract_batch(self, images: Sequence[np.ndarray | Image.Image]) -> DenseFeatureBatch:
        """Extract dense patch embeddings for a batch of images."""
        pixel_values, original_sizes = self.preprocess_batch(images)
        pixel_values = pixel_values.to(self.device, non_blocking=True)

        use_amp = self.device.type == "cuda" and self.cfg.amp_dtype in ("bfloat16", "float16")
        amp_dtype = torch.bfloat16 if self.cfg.amp_dtype == "bfloat16" else torch.float16

        with torch.inference_mode():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)
            else:
                outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)

            hidden_states = outputs.hidden_states
            selected_layers = [hidden_states[layer_idx] for layer_idx in self.cfg.extract_layers]

            patch_feat, cls_feat, global_feat = process_token_sequence(
                layer_tokens=selected_layers,
                num_register_tokens=self.cfg.num_register_tokens,
                grid_size=self.cfg.grid_size,
                layer_aggregation=self.cfg.layer_aggregation,
                neighborhood_kernel=self.cfg.neighborhood_kernel,
                l2_normalize=self.cfg.l2_normalize,
            )

        return DenseFeatureBatch(
            patch_embeddings=patch_feat.cpu().numpy().astype(np.float32),
            cls_embeddings=cls_feat.cpu().numpy().astype(np.float32),
            global_embeddings=global_feat.cpu().numpy().astype(np.float32),
            grid_size=self.cfg.grid_size,
            original_sizes=original_sizes,
        )

    def extract_dataset(self, images: Sequence[np.ndarray | Image.Image]) -> DenseFeatureBatch:
        """Extract dense patch embeddings across an arbitrary list of images using configured batch size."""
        all_patches: list[np.ndarray] = []
        all_cls: list[np.ndarray] = []
        all_global: list[np.ndarray] = []
        all_sizes: list[tuple[int, int]] = []

        bs = self.cfg.batch_size
        for start in range(0, len(images), bs):
            batch_imgs = images[start : start + bs]
            batch_out = self.extract_batch(batch_imgs)
            all_patches.append(batch_out.patch_embeddings)
            all_cls.append(batch_out.cls_embeddings)
            all_global.append(batch_out.global_embeddings)
            all_sizes.extend(batch_out.original_sizes)

        return DenseFeatureBatch(
            patch_embeddings=np.concatenate(all_patches, axis=0),
            cls_embeddings=np.concatenate(all_cls, axis=0),
            global_embeddings=np.concatenate(all_global, axis=0),
            grid_size=self.cfg.grid_size,
            original_sizes=all_sizes,
        )

    def metadata(self) -> dict[str, Any]:
        """Return model provenance and preprocessing metadata."""
        return {
            "model_id": self.cfg.model_id,
            "revision": self.cfg.revision,
            "architecture": "DINOv3ViTModel",
            "patch_size": self.cfg.patch_size,
            "hidden_size": self.cfg.hidden_size,
            "num_register_tokens": self.cfg.num_register_tokens,
            "image_size": self.cfg.image_size,
            "grid_size": list(self.cfg.grid_size),
            "num_patches": self.cfg.num_patches,
            "extract_layers": list(self.cfg.extract_layers),
            "layer_aggregation": self.cfg.layer_aggregation,
            "neighborhood_kernel": self.cfg.neighborhood_kernel,
            "l2_normalize": self.cfg.l2_normalize,
            "embedding_dim": self.cfg.embedding_dim,
        }
