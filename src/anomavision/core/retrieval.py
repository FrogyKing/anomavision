"""FAISS vector similarity search for reference normal image retrieval."""

from __future__ import annotations

import json
import time
from pathlib import Path

import faiss
import numpy as np

from anomavision.schemas import SampleRecord, SimilarNeighbor


class ReferenceImageSimilarityIndex:
    """FAISS index over normal training (`train_fit`) global image embeddings.

    Design & Anti-Leakage Guarantee:
      - Indexes ONLY normal training (`train_fit`) samples. Test samples and defect labels
        never enter this index.
      - Uses L2-normalized global descriptors g(X) in R^(2d) with inner-product search
        (`faiss.IndexFlatIP`), where inner product equals cosine similarity:
            sim(q, r) = <q, r> / (||q||_2 ||r||_2) in [-1, 1]
        and cosine distance is d_cos(q, r) = 1 - sim(q, r).
      - Enables visual comparison of an inspected query image against its closest defect-free
        reference counterparts (matching object orientation, scale, and illumination).
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index: faiss.Index = faiss.IndexFlatIP(dim)
        self.metadata: list[dict[str, str]] = []

    @staticmethod
    def _normalize_rows(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return (x / np.maximum(norms, 1e-12)).astype(np.float32)

    def fit(self, global_embeddings: np.ndarray, records: list[SampleRecord]) -> None:
        """Build the FAISS reference index exclusively from normal training records."""
        if len(global_embeddings) != len(records):
            raise ValueError(
                f"Length mismatch: {len(global_embeddings)} embeddings vs {len(records)} records"
            )
        for rec in records:
            if rec.is_anomalous or rec.split != "train_fit":
                raise ValueError(
                    f"Anti-leakage violation: attempted to add sample '{rec.sample_id}' "
                    f"(split={rec.split}, is_anomalous={rec.is_anomalous}) to reference retrieval index"
                )

        normed = np.ascontiguousarray(self._normalize_rows(global_embeddings), dtype=np.float32)
        self.index.reset()
        self.index.add(normed)
        self.metadata = [
            {
                "sample_id": r.sample_id,
                "category": r.category,
                "image_relpath": r.image_relpath,
            }
            for r in records
        ]

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 4,
    ) -> tuple[list[SimilarNeighbor], float]:
        """Retrieve top_k most similar normal reference images and measure retrieval latency (ms)."""
        if self.index.ntotal == 0:
            return [], 0.0

        q = query_embedding.reshape(1, -1).astype(np.float32)
        q = np.ascontiguousarray(self._normalize_rows(q), dtype=np.float32)
        k = min(top_k, self.index.ntotal)

        t0 = time.perf_counter()
        similarities, indices = self.index.search(q, k)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        neighbors: list[SimilarNeighbor] = []
        for rank_idx in range(k):
            idx = int(indices[0, rank_idx])
            if idx < 0 or idx >= len(self.metadata):
                continue
            sim = float(np.clip(similarities[0, rank_idx], -1.0, 1.0))
            dist = float(max(0.0, 1.0 - sim))
            meta = self.metadata[idx]
            neighbors.append(
                SimilarNeighbor(
                    rank=rank_idx + 1,
                    sample_id=meta["sample_id"],
                    category=meta["category"],
                    image_relpath=meta["image_relpath"],
                    distance=dist,
                    similarity_score=sim,
                )
            )
        return neighbors, latency_ms

    def save(self, index_path: Path, meta_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        meta_path.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_path: Path, meta_path: Path) -> ReferenceImageSimilarityIndex:
        idx = faiss.read_index(str(index_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        instance = cls(dim=idx.d)
        instance.index = idx
        instance.metadata = meta
        return instance
