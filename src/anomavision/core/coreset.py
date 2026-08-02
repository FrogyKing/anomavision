"""Greedy k-Center coreset subsampling with Johnson-Lindenstrauss random projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class CoresetSelectionResult:
    """Output of greedy k-center coreset selection."""

    coreset_embeddings: np.ndarray  # Shape: (K, d), float32
    selected_indices: np.ndarray  # Shape: (K,), int64
    covering_radii: np.ndarray  # Shape: (K,), float32
    coreset_ratio_actual: float
    total_candidates: int


def select_greedy_kcenter_coreset(
    embeddings: np.ndarray,
    coreset_ratio: float = 0.10,
    max_coreset_size: int = 15000,
    projection_dim: int = 128,
    seed: int = 42,
    device: str | None = None,
) -> CoresetSelectionResult:
    """Select a greedy k-center coreset from candidate patch embeddings.

    Mathematical Formulation:
        Given candidate set P = {z_1, ..., z_M} subset R^d, we seek M_c subset P of size K
        minimizing the minimax covering radius:
            max_{z in P} min_{m in M_c} ||z - m||_2
        Using a Johnson-Lindenstrauss random Gaussian projection Pi in R^(d x d_proj) with
        entries N(0, 1/d_proj), pairwise distances are preserved within (1 +/- eps) while
        reducing per-step distance evaluation from O(M*d) to O(M*d_proj).
        At each step t, we pick:
            i_t = argmax_{i in {1..M}} min_{j < t} ||Pi^T z_i - Pi^T m_j||_2
        and update the running minimum distances in O(M * d_proj) time.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array (M, d), got shape {embeddings.shape}")

    total_m, dim = embeddings.shape
    if total_m == 0:
        raise ValueError("Cannot select coreset from empty embedding array")

    if not (0.0 < coreset_ratio <= 1.0):
        raise ValueError(f"coreset_ratio must be in (0, 1], got {coreset_ratio}")

    target_k = max(1, min(int(round(total_m * coreset_ratio)), max_coreset_size, total_m))
    if target_k == total_m:
        indices = np.arange(total_m, dtype=np.int64)
        radii = np.zeros(total_m, dtype=np.float32)
        return CoresetSelectionResult(
            coreset_embeddings=embeddings.astype(np.float32, copy=True),
            selected_indices=indices,
            covering_radii=radii,
            coreset_ratio_actual=1.0,
            total_candidates=total_m,
        )

    if device is None:
        exec_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        exec_device = torch.device(device)

    rng = np.random.default_rng(seed)
    tensor_emb = torch.from_numpy(embeddings.astype(np.float32, copy=False)).to(exec_device)

    # Apply Johnson-Lindenstrauss random projection when d > projection_dim
    proj_d = min(dim, projection_dim)
    if proj_d < dim:
        proj_matrix_np = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(proj_d),
            size=(dim, proj_d),
        ).astype(np.float32)
        proj_matrix = torch.from_numpy(proj_matrix_np).to(exec_device)
        projected = tensor_emb @ proj_matrix
    else:
        projected = tensor_emb

    selected_indices = np.empty(target_k, dtype=np.int64)
    covering_radii = np.empty(target_k, dtype=np.float32)

    # Initialize with the point farthest from the empirical centroid for deterministic stability
    centroid = projected.mean(dim=0, keepdim=True)
    init_dists = torch.linalg.norm(projected - centroid, dim=1)
    current_idx = int(torch.argmax(init_dists).item())

    min_distances = torch.full((total_m,), float("inf"), dtype=torch.float32, device=exec_device)

    for step in range(target_k):
        selected_indices[step] = current_idx
        pivot = projected[current_idx : current_idx + 1]  # (1, proj_d)
        dists_to_pivot = torch.linalg.norm(projected - pivot, dim=1)
        min_distances = torch.minimum(min_distances, dists_to_pivot)
        # Ensure already selected points have zero distance
        min_distances[selected_indices[: step + 1]] = 0.0

        max_dist_val, next_idx_tensor = torch.max(min_distances, dim=0)
        covering_radii[step] = float(max_dist_val.item())
        current_idx = int(next_idx_tensor.item())

    coreset_emb = embeddings[selected_indices].astype(np.float32, copy=True)
    return CoresetSelectionResult(
        coreset_embeddings=coreset_emb,
        selected_indices=selected_indices,
        covering_radii=covering_radii,
        coreset_ratio_actual=float(target_k) / float(total_m),
        total_candidates=total_m,
    )
