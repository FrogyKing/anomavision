"""Backbone feature extractors and object-aware segmentation adapters."""

from anomavision.models.dinov3 import (
    DenseFeatureBatch,
    DINOv3FeatureExtractor,
    process_token_sequence,
)
from anomavision.models.sam import (
    CATEGORY_TEXT_PROMPTS,
    SAM3ObjectSegmenter,
    SAM31MultiplexAdapter,
    apply_object_mask_refinement,
    dilate_binary_mask,
)

__all__ = [
    "CATEGORY_TEXT_PROMPTS",
    "DenseFeatureBatch",
    "DINOv3FeatureExtractor",
    "SAM3ObjectSegmenter",
    "SAM31MultiplexAdapter",
    "apply_object_mask_refinement",
    "dilate_binary_mask",
    "process_token_sequence",
]
