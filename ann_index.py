"""ANN index module supporting FAISS and HNSW algorithms with multiple distance metrics."""

import time
import numpy as np
from typing import Tuple


class ANNIndex:
    """Approximate Nearest Neighbor index wrapper for single-cell data.

    Supports three distance metrics:
      - l2: Euclidean / L2 distance (default)
      - cosine: Cosine distance (1 - cosine similarity, requires L2-normalized vectors)
      - ip: Inner product (negative inner product, so smaller = more similar)
    """

    METRICS = ("l2", "cosine", "ip")

    def __init__(self, embeddings: np.ndarray, method: str = "hnsw", metric: str = "l2"):
        if metric not in self.METRICS:
            raise ValueError(f"Unknown metric '{metric}'. Choose from {self.METRICS}")

        self.metric = metric
        self.embeddings = embeddings.astype(np.float32)
        self.n_cells, self.dim = embeddings.shape
        self.method = method
        self.index = None
        self._build_time_ms = 0.0

    def build(self, **kwargs) -> float:
        """Build the ANN index. Returns build time in milliseconds."""
        t0 = time.time()
        if self.method == "faiss_flat":
            self._build_faiss_flat()
        elif self.method == "faiss_ivf":
            self._build_faiss_ivf(**kwargs)
        elif self.method == "hnsw":
            self._build_hnsw(**kwargs)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        self._build_time_ms = (time.time() - t0) * 1000
        return self._build_time_ms

    # ── FAISS builders ──

    def _build_faiss_flat(self):
        import faiss
        self.index = faiss.IndexFlatL2(self.dim) if self.metric == "l2" else \
                     faiss.IndexFlatIP(self.dim)
        self.index.add(self.embeddings)

    def _build_faiss_ivf(self, nlist: int = 100, nprobe: int = 10):
        import faiss
        quantizer = faiss.IndexFlatL2(self.dim) if self.metric == "l2" else \
                    faiss.IndexFlatIP(self.dim)
        self.index = faiss.IndexIVFFlat(quantizer, self.dim, nlist)
        self.index.train(self.embeddings)
        self.index.add(self.embeddings)
        self.index.nprobe = nprobe

    # ── HNSW builder ──

    def _build_hnsw(self, M: int = 16, ef_construction: int = 200):
        try:
            import hnswlib
        except ImportError:
            raise ImportError(
                "hnswlib is not installed. Install it with: pip install hnswlib "
                "(requires Microsoft Visual C++ Build Tools on Windows)"
            )
        self.index = hnswlib.Index(space=self.metric, dim=self.dim)
        self.index.init_index(
            max_elements=self.n_cells,
            ef_construction=ef_construction,
            M=M,
        )
        self.index.add_items(self.embeddings, np.arange(self.n_cells))

    # ── Query ──

    def query(self, query_vec: np.ndarray, k: int = 10, ef: int = 50) -> Tuple[np.ndarray, np.ndarray, float]:
        """Query top-k similar cells. Returns (indices, distances, query_time_ms).

        For 'l2' metric: distances are squared Euclidean distances (smaller = more similar).
        For 'cosine' / 'ip': raw scores from the index; smaller = more similar for cosine distance,
        larger = more similar for inner product. The caller is responsible for interpretation.
        """
        query_vec = query_vec.astype(np.float32).reshape(1, -1)
        t0 = time.time()

        if self.method.startswith("faiss"):
            distances, indices = self.index.search(query_vec, k)
        elif self.method == "hnsw":
            self.index.set_ef(ef)
            indices, distances = self.index.knn_query(query_vec, k=k)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        query_time_ms = (time.time() - t0) * 1000
        return indices[0], distances[0], query_time_ms

    def batch_query(self, query_vecs: np.ndarray, k: int = 10, ef: int = 50) -> Tuple[np.ndarray, np.ndarray, float]:
        """Batch query top-k similar cells."""
        query_vecs = query_vecs.astype(np.float32)
        t0 = time.time()

        if self.method.startswith("faiss"):
            distances, indices = self.index.search(query_vecs, k)
        elif self.method == "hnsw":
            self.index.set_ef(ef)
            indices, distances = self.index.knn_query(query_vecs, k=k)

        query_time_ms = (time.time() - t0) * 1000
        return indices, distances, query_time_ms

    @property
    def build_time_ms(self) -> float:
        return self._build_time_ms
