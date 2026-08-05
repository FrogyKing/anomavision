"""PatchCore-inspired dense nearest-neighbor anomaly detector and lightweight global baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from anomavision.config import CoresetConfig
from anomavision.core.coreset import CoresetSelectionResult, select_greedy_kcenter_coreset


@dataclass
class AnomalyPredictionBatch:
    """Batch output from PatchCoreDetector."""

    image_scores: np.ndarray  # Shape: (B,), float32
    max_patch_scores: np.ndarray  # Shape: (B,), float32
    top_k_patch_scores: np.ndarray  # Shape: (B,), float32
    patch_score_grids: np.ndarray  # Shape: (B, H_p, W_p), float32
    anomaly_maps: list[np.ndarray]  # List of (H_orig, W_orig) float32 pixel maps


def compute_pairwise_distances(
    queries: np.ndarray,
    memory_bank: np.ndarray,
    metric: Literal["cosine", "euclidean"] = "cosine",
) -> np.ndarray:
    """Compute exact distance matrix D in R^(N_q x K) between query patches and memory bank."""
    q = np.asarray(queries, dtype=np.float32)
    m = np.asarray(memory_bank, dtype=np.float32)

    if metric == "cosine":
        q_norm = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        m_norm = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
        sim = np.clip(q_norm @ m_norm.T, -1.0, 1.0)
        return (1.0 - sim).astype(np.float32)
    elif metric == "euclidean":
        q_sq = np.sum(q * q, axis=1, keepdims=True)
        m_sq = np.sum(m * m, axis=1, keepdims=True).T
        d_sq = np.maximum(0.0, q_sq + m_sq - 2.0 * (q @ m.T))
        return np.sqrt(d_sq).astype(np.float32)
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def upsample_patch_scores(
    patch_grid: np.ndarray,
    target_hw: tuple[int, int],
    gaussian_sigma: float = 4.0,
) -> np.ndarray:
    """Upsample 2D patch anomaly grid (H_p, W_p) to original image dimensions (H, W) with alignment."""
    if patch_grid.ndim != 2:
        raise ValueError(f"Expected 2D patch_grid, got shape {patch_grid.shape}")

    tensor_grid = torch.from_numpy(patch_grid.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    upsampled = F.interpolate(
        tensor_grid,
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()

    if gaussian_sigma > 0.0:
        upsampled = gaussian_filter(upsampled, sigma=gaussian_sigma, mode="nearest")
    return upsampled.astype(np.float32)


class PatchCoreDetector:
    """PatchCore-inspired anomaly detector operating on DINOv3 dense patch embeddings.

    Mathematical Formulation:
      1. Training / Memory Bank Construction:
         Given normal training images X_1, ..., X_{n_train}, extract dense locally-aggregated
         patch embeddings Z_i in R^(N x d). Concatenate all M_total = n_train * N patches into
         candidate pool P, and select greedy k-center coreset M subset P of size K.
      2. Nearest-Neighbor Patch Scoring:
         For each patch z_q (q = 1..N) of a test image, find its b nearest neighbors in M
         under metric d(., .):
             d_{(1)}(z_q) <= d_{(2)}(z_q) <= ... <= d_{(b)}(z_q)
         The spatial patch anomaly score is s(z_q) = d_{(1)}(z_q).
      3. Density-Reweighted Image Anomaly Score:
         For the top candidate patch z^* = argmax_{z_q} s(z_q), compute local density weight:
             w(z^*) = 1 - exp(d_{(1)}(z^*) / tau) / sum_{j=1}^b exp(d_{(j)}(z^*) / tau)
         Combine the reweighted peak score and the mean of the top-k patch scores:
             S(X) = 0.5 * (1 + w(z^*)) * d_{(1)}(z^*) + 0.5 * (1/k) * sum_{r=1}^k s_{(r)}
    """

    def __init__(self, cfg: CoresetConfig) -> None:
        self.cfg = cfg
        self.memory_bank: np.ndarray | None = None
        self.faiss_index: faiss.Index | None = None
        self.coreset_stats: CoresetSelectionResult | None = None

    def fit(
        self,
        train_patch_embeddings: np.ndarray,
        device: str | None = None,
    ) -> CoresetSelectionResult:
        """Construct the greedy k-center coreset memory bank from normal training patches.

        Args:
            train_patch_embeddings: Shape (N_images, N_patches, d) or (M_total, d).
        """
        if train_patch_embeddings.ndim == 3:
            n_img, n_p, dim = train_patch_embeddings.shape
            flat_patches = train_patch_embeddings.reshape(n_img * n_p, dim)
        elif train_patch_embeddings.ndim == 2:
            flat_patches = train_patch_embeddings
        else:
            raise ValueError(f"Unexpected train_patch_embeddings shape: {train_patch_embeddings.shape}")

        result = select_greedy_kcenter_coreset(
            embeddings=flat_patches,
            coreset_ratio=self.cfg.coreset_ratio,
            max_coreset_size=self.cfg.max_coreset_size,
            projection_dim=self.cfg.projection_dim,
            seed=self.cfg.seed,
            device=device,
        )
        self.coreset_stats = result
        self._set_memory_bank(result.coreset_embeddings)
        return result

    def _set_memory_bank(self, memory_bank: np.ndarray) -> None:
        mem = np.ascontiguousarray(memory_bank, dtype=np.float32)
        index: faiss.Index
        if self.cfg.metric == "cosine":
            norms = np.linalg.norm(mem, axis=1, keepdims=True)
            mem = np.ascontiguousarray(mem / np.maximum(norms, 1e-12), dtype=np.float32)
            index = faiss.IndexFlatIP(mem.shape[1])
            index.add(mem)
        elif self.cfg.metric == "euclidean":
            index = faiss.IndexFlatL2(mem.shape[1])
            index.add(mem)
        else:
            raise ValueError(f"Unsupported metric: {self.cfg.metric}")

        self.memory_bank = mem
        self.faiss_index = index

    def query_patch_knn(self, flat_queries: np.ndarray) -> np.ndarray:
        """Query top-b nearest neighbor distances in the coreset memory bank for each patch."""
        if self.faiss_index is None or self.memory_bank is None:
            raise RuntimeError("PatchCoreDetector must be fitted or loaded before scoring")

        b_neighbors = min(self.cfg.n_neighbors, self.faiss_index.ntotal)
        q = np.ascontiguousarray(flat_queries, dtype=np.float32)

        # Use GPU-accelerated exact k-NN search when CUDA is available for >100x throughput
        if torch.cuda.is_available():
            device = torch.device("cuda")
            mem_t = torch.from_numpy(self.memory_bank).to(device)
            n_q = len(q)
            out_dists = np.empty((n_q, b_neighbors), dtype=np.float32)
            chunk_size = 8192

            with torch.inference_mode():
                for start in range(0, n_q, chunk_size):
                    end = min(start + chunk_size, n_q)
                    q_t = torch.from_numpy(q[start:end]).to(device)
                    if self.cfg.metric == "cosine":
                        q_t = F.normalize(q_t, p=2, dim=1, eps=1e-12)
                        sims_t = q_t @ mem_t.T
                        top_sims_t, _ = torch.topk(sims_t, k=b_neighbors, dim=1, largest=True, sorted=True)
                        d_chunk = torch.clamp(1.0 - top_sims_t, min=0.0)
                    else:
                        d_mat = torch.cdist(q_t, mem_t, p=2.0)
                        d_chunk, _ = torch.topk(d_mat, k=b_neighbors, dim=1, largest=False, sorted=True)
                    out_dists[start:end] = d_chunk.cpu().numpy()
            return out_dists

        if self.cfg.metric == "cosine":
            norms = np.linalg.norm(q, axis=1, keepdims=True)
            q = np.ascontiguousarray(q / np.maximum(norms, 1e-12), dtype=np.float32)
            sims, _ = self.faiss_index.search(q, b_neighbors)
            dists = np.maximum(0.0, 1.0 - np.clip(sims, -1.0, 1.0))
            return dists.astype(np.float32)
        else:
            sq_dists, _ = self.faiss_index.search(q, b_neighbors)
            return np.sqrt(np.maximum(0.0, sq_dists)).astype(np.float32)

    def predict_batch(
        self,
        patch_embeddings: np.ndarray,
        grid_size: tuple[int, int],
        original_sizes: list[tuple[int, int]],
    ) -> AnomalyPredictionBatch:
        """Compute patch-level anomaly grids, upsampled pixel maps, and image-level scores."""
        if patch_embeddings.ndim != 3:
            raise ValueError(f"Expected (B, N, d) patch_embeddings, got {patch_embeddings.shape}")

        batch_size, n_patches, dim = patch_embeddings.shape
        h_p, w_p = grid_size
        if h_p * w_p != n_patches:
            raise ValueError(f"grid_size {grid_size} does not match n_patches={n_patches}")

        flat_queries = patch_embeddings.reshape(batch_size * n_patches, dim)
        knn_dists = self.query_patch_knn(flat_queries)  # (B * N, b)
        b_neighbors = knn_dists.shape[1]
        knn_dists_batch = knn_dists.reshape(batch_size, n_patches, b_neighbors)

        # Nearest neighbor distance per patch: d_{(1)}(z_q)
        patch_nn_dists = knn_dists_batch[:, :, 0]  # (B, N)
        patch_grids = patch_nn_dists.reshape(batch_size, h_p, w_p)

        image_scores = np.empty(batch_size, dtype=np.float32)
        max_patch_scores = np.empty(batch_size, dtype=np.float32)
        top_k_scores = np.empty(batch_size, dtype=np.float32)
        anomaly_maps: list[np.ndarray] = []

        k_top = min(self.cfg.score_top_k_patches, n_patches)

        for i in range(batch_size):
            p_dists = patch_nn_dists[i]  # (N,)
            max_idx = int(np.argmax(p_dists))
            d_max = float(p_dists[max_idx])
            max_patch_scores[i] = d_max

            sorted_p = np.sort(p_dists)[::-1]
            top_k_mean = float(np.mean(sorted_p[:k_top]))
            top_k_scores[i] = top_k_mean

            if b_neighbors > 1:
                neighbor_vec = knn_dists_batch[i, max_idx, :]  # (b,)
                # Scale temperature by median neighbor distance for scale-invariance across metrics
                tau = max(float(np.median(neighbor_vec)), 0.05)
                shifted = (neighbor_vec - neighbor_vec[0]) / tau
                exp_vals = np.exp(np.clip(shifted, -30.0, 30.0))
                w = float(1.0 - (exp_vals[0] / np.sum(exp_vals)))
            else:
                w = 0.0

            image_scores[i] = float(0.85 * (1.0 + w) * d_max + 0.15 * top_k_mean)

            orig_hw = original_sizes[i]
            amap = upsample_patch_scores(
                patch_grid=patch_grids[i],
                target_hw=orig_hw,
                gaussian_sigma=self.cfg.gaussian_sigma,
            )
            anomaly_maps.append(amap)

        return AnomalyPredictionBatch(
            image_scores=image_scores,
            max_patch_scores=max_patch_scores,
            top_k_patch_scores=top_k_scores,
            patch_score_grids=patch_grids,
            anomaly_maps=anomaly_maps,
        )

    @property
    def memory_bank_size_mb(self) -> float:
        if self.memory_bank is None:
            return 0.0
        return float(self.memory_bank.nbytes) / (1024.0 * 1024.0)

    def save(self, path: Path) -> None:
        if self.memory_bank is None:
            raise RuntimeError("Cannot save unfitted PatchCoreDetector")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            memory_bank=self.memory_bank,
            selected_indices=(
                self.coreset_stats.selected_indices
                if self.coreset_stats is not None
                else np.arange(len(self.memory_bank), dtype=np.int64)
            ),
            covering_radii=(
                self.coreset_stats.covering_radii
                if self.coreset_stats is not None
                else np.zeros(len(self.memory_bank), dtype=np.float32)
            ),
        )

    @classmethod
    def load(cls, path: Path, cfg: CoresetConfig) -> PatchCoreDetector:
        data = np.load(path)
        detector = cls(cfg)
        mem = data["memory_bank"].astype(np.float32)
        detector._set_memory_bank(mem)
        detector.coreset_stats = CoresetSelectionResult(
            coreset_embeddings=mem,
            selected_indices=data["selected_indices"].astype(np.int64),
            covering_radii=data["covering_radii"].astype(np.float32),
            coreset_ratio_actual=cfg.coreset_ratio,
            total_candidates=len(mem),
        )
        return detector


class GlobalCentroidBaseline:
    """Lightweight baseline detector: position-wise mean template + global shrinkage Mahalanobis.

    Rather than storing a k-center coreset memory bank of thousands of individual patches,
    this baseline stores only:
      1. The empirical global token mean mu_g in R^(2d) and regularized diagonal/shrinkage
         covariance for image-level scoring.
      2. A single spatial mean template M_bar in R^(N x d) (1 prototype per grid position)
         for pixel-level cosine distance scoring.
    This provides a transparent, fast baseline to quantify the gain from PatchCore's non-parametric
    multi-prototype memory bank.
    """

    def __init__(self) -> None:
        self.spatial_template: np.ndarray | None = None  # (N, d)
        self.global_mean: np.ndarray | None = None  # (2d,)
        self.global_inv_cov: np.ndarray | None = None  # (2d, 2d)

    def fit(self, train_patch_embeddings: np.ndarray, train_global_embeddings: np.ndarray) -> None:
        # Spatial position-wise mean template (N, d)
        mean_template = np.mean(train_patch_embeddings, axis=0).astype(np.float32)
        norms = np.linalg.norm(mean_template, axis=1, keepdims=True)
        self.spatial_template = mean_template / np.maximum(norms, 1e-12)

        # Global regularized covariance (Ledoit-Wolf style ridge shrinkage)
        g = train_global_embeddings.astype(np.float64)
        mu = np.mean(g, axis=0)
        centered = g - mu
        cov = (centered.T @ centered) / max(1, len(g) - 1)
        trace_scale = float(np.trace(cov)) / float(cov.shape[0])
        shrinkage_alpha = 0.20
        reg_cov = (1.0 - shrinkage_alpha) * cov + shrinkage_alpha * max(trace_scale, 1e-4) * np.eye(cov.shape[0])
        self.global_mean = mu.astype(np.float32)
        self.global_inv_cov = np.linalg.pinv(reg_cov).astype(np.float32)

    def predict_batch(
        self,
        patch_embeddings: np.ndarray,
        global_embeddings: np.ndarray,
        grid_size: tuple[int, int],
        original_sizes: list[tuple[int, int]],
        gaussian_sigma: float = 4.0,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        if self.spatial_template is None or self.global_mean is None or self.global_inv_cov is None:
            raise RuntimeError("GlobalCentroidBaseline must be fitted before prediction")

        b, n_p, _ = patch_embeddings.shape
        h_p, w_p = grid_size

        # Position-wise cosine distance to mean spatial template
        p_norm = patch_embeddings / np.maximum(np.linalg.norm(patch_embeddings, axis=-1, keepdims=True), 1e-12)
        pos_sim = np.sum(p_norm * self.spatial_template[None, :, :], axis=-1)  # (B, N)
        pos_dist = np.maximum(0.0, 1.0 - pos_sim)
        grids = pos_dist.reshape(b, h_p, w_p)

        # Global Mahalanobis distance combined with max template distance
        diff = global_embeddings.astype(np.float32) - self.global_mean[None, :]
        mahal = np.sqrt(np.maximum(0.0, np.sum((diff @ self.global_inv_cov) * diff, axis=1)))
        image_scores = (0.5 * mahal + 0.5 * np.max(pos_dist, axis=1)).astype(np.float32)

        maps: list[np.ndarray] = []
        for i in range(b):
            maps.append(
                upsample_patch_scores(
                    patch_grid=grids[i],
                    target_hw=original_sizes[i],
                    gaussian_sigma=gaussian_sigma,
                )
            )
        return image_scores, maps
