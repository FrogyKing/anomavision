"""Unit tests for DINOv3 token sequence processing and SAM mask refinement."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from anomavision.models.dinov3 import process_token_sequence
from anomavision.models.sam import apply_object_mask_refinement, dilate_binary_mask


def test_process_token_sequence_handles_cls_and_registers() -> None:
    batch_size = 2
    num_registers = 4
    grid_size = (14, 14)
    num_patches = 196
    seq_len = 1 + num_registers + num_patches  # 201 tokens
    hidden_dim = 384

    torch.manual_seed(42)
    layer_6 = torch.randn(batch_size, seq_len, hidden_dim)
    layer_9 = torch.randn(batch_size, seq_len, hidden_dim)
    layer_12 = torch.randn(batch_size, seq_len, hidden_dim)

    patches, cls_tok, global_tok = process_token_sequence(
        layer_tokens=[layer_6, layer_9, layer_12],
        num_register_tokens=num_registers,
        grid_size=grid_size,
        layer_aggregation="mean",
        neighborhood_kernel=3,
        l2_normalize=True,
    )

    assert patches.shape == (batch_size, num_patches, hidden_dim)
    assert cls_tok.shape == (batch_size, hidden_dim)
    assert global_tok.shape == (batch_size, hidden_dim * 2)

    # Verify unit L2 norms on the hypersphere S^(d-1)
    patch_norms = torch.linalg.norm(patches, dim=-1)
    assert torch.allclose(patch_norms, torch.ones_like(patch_norms), atol=1e-5)

    global_norms = torch.linalg.norm(global_tok, dim=-1)
    assert torch.allclose(global_norms, torch.ones_like(global_norms), atol=1e-5)


def test_process_token_sequence_rejects_wrong_seq_length() -> None:
    bad_tokens = [torch.randn(1, 197, 384)]  # Missing 4 register tokens
    with pytest.raises(ValueError, match="Sequence length mismatch"):
        process_token_sequence(
            layer_tokens=bad_tokens,
            num_register_tokens=4,
            grid_size=(14, 14),
        )


def test_sam_mask_dilation_and_refinement() -> None:
    raw_mask = np.zeros((64, 64), dtype=np.uint8)
    raw_mask[20:44, 20:44] = 1
    # Create an interior hole to verify hole filling
    raw_mask[28:34, 28:34] = 0

    dilated = dilate_binary_mask(raw_mask, dilation_px=3)
    assert dilated.shape == (64, 64)
    assert dilated[30, 30] == 1  # Hole filled
    assert dilated[18, 30] == 1  # Dilated boundary

    anomaly_map = np.ones((64, 64), dtype=np.float32) * 2.0
    refined = apply_object_mask_refinement(
        anomaly_map=anomaly_map,
        object_mask=dilated,
        background_suppression_factor=0.1,
    )
    assert np.isclose(refined[30, 30], 2.0)  # Foreground preserved 100%
    assert np.isclose(refined[0, 0], 0.2)  # Background attenuated by 0.1
