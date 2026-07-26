"""SAM 3 primary object segmentation and optional SAM 3.1 adapter for foreground region refinement.

Important architectural boundary:
  - SAM 3 (`tkclam/sam3`) is an open-vocabulary object segmenter used to isolate the foreground
    inspected component (e.g., bottle, metal nut, hazelnut) from conveyor/lighting backgrounds.
  - SAM 3 is NOT the anomaly detector itself; DINOv3 + PatchCore coreset scoring performs anomaly
    detection and localization, and the system remains 100% functional with SAM disabled.
  - SAM 3.1 (`AEmotionStudio/sam3.1`) is a distinct video-multiplex tracking checkpoint
    (`sam3.1_multiplex.safetensors`) whose architectural additions target multi-object video
    memory tracking while sharing the same image-detector weights. We provide an explicit
    optional adapter that inspects and loads the image-detector subset without conflating
    the two architectures.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file
from scipy.ndimage import binary_dilation, binary_fill_holes
from transformers import Sam3Model, Sam3Processor

from anomavision.config import SAMConfig

logger = logging.getLogger(__name__)

CATEGORY_TEXT_PROMPTS: dict[str, str] = {
    "bottle": "bottle",
    "cable": "cable",
    "capsule": "capsule",
    "carpet": "textured surface",
    "grid": "metal grid",
    "hazelnut": "hazelnut",
    "leather": "leather surface",
    "metal_nut": "metal nut",
    "pill": "pill",
    "screw": "metal screw",
    "tile": "tile surface",
    "toothbrush": "toothbrush",
    "transistor": "electronic transistor",
    "wood": "wood surface",
    "zipper": "zipper",
}


def dilate_binary_mask(mask: np.ndarray, dilation_px: int = 5) -> np.ndarray:
    """Fill interior holes and apply morphological dilation to preserve object rim defects."""
    binary = (mask > 0).astype(bool)
    if not np.any(binary):
        return np.ones_like(mask, dtype=np.uint8)
    filled = binary_fill_holes(binary)
    if dilation_px > 0:
        y, x = np.ogrid[-dilation_px : dilation_px + 1, -dilation_px : dilation_px + 1]
        struct = (x * x + y * y) <= dilation_px * dilation_px
        dilated = binary_dilation(filled, structure=struct)
        return dilated.astype(np.uint8)
    return filled.astype(np.uint8)


def apply_object_mask_refinement(
    anomaly_map: np.ndarray,
    object_mask: np.ndarray,
    background_suppression_factor: float = 0.15,
) -> np.ndarray:
    """Attenuate background false positives outside the dilated SAM object mask.

    Formula:
        A_refined(u, v) = A(u, v) * (S(u, v) + gamma_bg * (1 - S(u, v)))
    where S(u, v) in {0, 1} is the foreground object mask and gamma_bg in [0, 1].
    """
    if anomaly_map.shape != object_mask.shape:
        raise ValueError(
            f"Anomaly map shape {anomaly_map.shape} does not match object mask shape {object_mask.shape}"
        )
    s = (object_mask > 0).astype(np.float32)
    weights = s + float(background_suppression_factor) * (1.0 - s)
    return (anomaly_map.astype(np.float32) * weights).astype(np.float32)


class SAM3ObjectSegmenter:
    """Primary SAM 3 (`tkclam/sam3`) image segmenter for industrial foreground masking."""

    def __init__(
        self,
        cfg: SAMConfig,
        device: str | torch.device | None = None,
        hf_token: str | None = None,
    ) -> None:
        self.cfg = cfg
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(
            "Loading SAM 3 model '%s' (revision=%s) on device=%s",
            cfg.primary_model_id,
            cfg.primary_revision[:10],
            self.device,
        )
        self.processor = Sam3Processor.from_pretrained(
            cfg.primary_model_id,
            revision=cfg.primary_revision,
            token=hf_token,
        )
        self.model = Sam3Model.from_pretrained(
            cfg.primary_model_id,
            revision=cfg.primary_revision,
            token=hf_token,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.to(self.device)  # type: ignore[arg-type]

    def segment_object(
        self,
        image: np.ndarray | Image.Image,
        category: str = "bottle",
    ) -> tuple[np.ndarray, float]:
        """Generate a dilated binary foreground mask (H, W) in {0, 1} and foreground coverage ratio."""
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        else:
            pil_img = image.convert("RGB")

        h_orig, w_orig = pil_img.height, pil_img.width
        prompt = CATEGORY_TEXT_PROMPTS.get(category, category.replace("_", " "))

        inputs = self.processor(
            images=pil_img,
            text=prompt,
            return_tensors="pt",
        )
        model_inputs: dict[str, Any] = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.is_floating_point() and self.device.type == "cuda":
                    model_inputs[k] = v.to(self.device, dtype=torch.bfloat16)
                else:
                    model_inputs[k] = v.to(self.device)
            else:
                model_inputs[k] = v

        with (
            torch.inference_mode(),
            torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(self.device.type == "cuda"),
            ),
        ):
            outputs = self.model(**model_inputs)

        # Extract instance masks or semantic segmentation fallback
        mask_2d = self._extract_best_foreground_mask(outputs, target_hw=(h_orig, w_orig))
        dilated = dilate_binary_mask(mask_2d, dilation_px=self.cfg.mask_dilation_px)
        coverage = float(np.mean(dilated > 0))

        # Safety guard: if coverage is implausibly tiny (<3%), fall back to full-frame mask
        if coverage < 0.03:
            logger.warning(
                "SAM 3 foreground coverage (%.3f) below safety floor for '%s'; falling back to full image mask",
                coverage,
                category,
            )
            full_mask = np.ones((h_orig, w_orig), dtype=np.uint8)
            return full_mask, 1.0

        return dilated, coverage

    def _extract_best_foreground_mask(
        self,
        outputs: Any,
        target_hw: tuple[int, int],
    ) -> np.ndarray:
        """Combine high-confidence predicted query masks or semantic segmentation logits."""
        h_orig, w_orig = target_hw
        try:
            processed = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=self.cfg.score_threshold,
                mask_threshold=0.5,
                target_sizes=[(h_orig, w_orig)],
            )
            if processed and len(processed) > 0:
                masks = processed[0].get("masks")
                if masks is not None and len(masks) > 0:
                    if isinstance(masks, torch.Tensor):
                        union_mask = (masks.detach().cpu().numpy() > 0).any(axis=0)
                    else:
                        union_mask = (np.asarray(masks) > 0).any(axis=0)
                    if union_mask.shape == (h_orig, w_orig) and np.any(union_mask):
                        return union_mask.astype(np.uint8)
        except Exception as exc:
            logger.debug("post_process_instance_segmentation fallback triggered: %s", exc)

        # Direct tensor fallback using pred_masks and pred_logits
        pred_masks = getattr(outputs, "pred_masks", None)
        pred_logits = getattr(outputs, "pred_logits", None)
        if pred_masks is not None and pred_logits is not None:
            probs = torch.sigmoid(pred_logits.float())[0]  # (num_queries,) or (num_queries, C)
            if probs.ndim > 1:
                probs = probs.max(dim=-1).values
            best_idx = int(torch.argmax(probs).item())
            top_mask = pred_masks[0, best_idx].float().unsqueeze(0).unsqueeze(0)
            resized = torch.nn.functional.interpolate(
                top_mask,
                size=(h_orig, w_orig),
                mode="bilinear",
                align_corners=False,
            )
            binary = (torch.sigmoid(resized[0, 0]) > 0.5).cpu().numpy().astype(np.uint8)
            if np.any(binary):
                return binary

        return np.ones((h_orig, w_orig), dtype=np.uint8)


class SAM31MultiplexAdapter:
    """Optional adapter for `AEmotionStudio/sam3.1` (`sam3.1_multiplex.safetensors`).

    SAM 3.1 is a video-multiplex tracking architecture that introduces shared-memory multi-object
    tracking layers (`tracker_model.*`) while keeping the image-level detector weights unchanged.
    This adapter inspects the `sam3.1_multiplex.safetensors` checkpoint and allows extracting the
    image-detector subset without conflating SAM 3.1's multiplex tracker with SAM 3 image inference.
    """

    def __init__(self, cfg: SAMConfig, hf_token: str | None = None) -> None:
        self.cfg = cfg
        self.hf_token = hf_token

    def inspect_checkpoint_provenance(self, download_weights: bool = False) -> dict[str, Any]:
        """Inspect SAM 3.1 configuration and optionally tensor key structure from safetensors."""
        config_path = Path(
            hf_hub_download(
                repo_id=self.cfg.optional_sam31_model_id,
                filename="config.json",
                revision=self.cfg.optional_sam31_revision,
                token=self.hf_token,
            )
        )
        info: dict[str, Any] = {
            "model_id": self.cfg.optional_sam31_model_id,
            "revision": self.cfg.optional_sam31_revision,
            "architecture": "Sam3VideoModel (Multiplex Tracker)",
            "checkpoint_filename": "sam3.1_multiplex.safetensors",
            "config_cached_at": str(config_path),
            "architectural_distinction": (
                "SAM 3.1 replaces single-object memory attention with a joint multiplex video tracker "
                "for multi-object video streams while preserving the upstream SAM 3 image detector."
            ),
        }
        if download_weights:
            weights_path = Path(
                hf_hub_download(
                    repo_id=self.cfg.optional_sam31_model_id,
                    filename="sam3.1_multiplex.safetensors",
                    revision=self.cfg.optional_sam31_revision,
                    token=self.hf_token,
                )
            )
            state_dict = load_file(str(weights_path), device="cpu")
            detector_keys = [k for k in state_dict if "detector" in k or "vision" in k]
            tracker_keys = [k for k in state_dict if "tracker" in k or "memory" in k]
            info["total_tensors"] = len(state_dict)
            info["detector_tensor_count"] = len(detector_keys)
            info["tracker_tensor_count"] = len(tracker_keys)
        return info
