from __future__ import annotations

import threading
import time

import faiss
import numpy as np
from numpy.typing import NDArray

from neural_cache.config import SearchConfig
from neural_cache.models import CacheEntry, SearchResult

class SimilaritySearchEngine:
    def __init__(self, config: SearchConfig, embedding_dim: int):
        self.config = config
        self.embedding_dim = embedding_dim
        self._index: faiss.Index | None = None
        self._id_map: dict[int, str] = {}
        self._reverse_id_map: dict[str, int] = {}
        self._lock = threading.RLock()
        self._next_id = 0
        self._is_trained = False
        self._reranker = None
        if config.enable_reranking:
            self._load_reranker()

    def _load_reranker(self) -> None:
        try:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(self.config.reranker_model)
        except Exception:
            self._reranker = None

    def _build_index(self) -> faiss.Index:
        dim = self.embedding_dim

        if self.config.index_type == "Flat":
            index = faiss.IndexFlatIP(dim)

        elif self.config.index_type == "IVF":
            quantizer = faiss.IndexFlatIP(dim)
            nlist = min(self.config.nlist, 1000)
            index = faiss.IndexIVFFlat(
                quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT
            )

        elif self.config.index_type == "HNSW":
            index = faiss.IndexHNSWFlat(dim, self.config.hnsw_m)
            index.hnsw.efConstruction = self.config.hnsw_ef_construction

        elif self.config.index_type == "LSH":
            index = faiss.IndexLSH(dim, nbits=dim // 2)

        else:
            raise ValueError(f"Unknown index type: {self.config.index_type}")

        return index

    def initialize_index(self) -> None:
        with self._lock:
            self._index = self._build_index()

            if self.config.index_type == "IVF":

                self._is_trained = False
            else:
                self._is_trained = True

    def add_entries(self, entries: list[CacheEntry]) -> None:
        if not entries:
            return

        with self._lock:
            embeddings = np.array(
                [e.embedding for e in entries], dtype=np.float32
            )

            if self._index is None:
                self.initialize_index()

            if self.config.index_type == "IVF" and not self._is_trained:
                if self._index.ntotal >= self.config.nlist:
                    self._index.train(embeddings)
                    self._is_trained = True
                else:

                    pass

            faiss_ids = np.arange(self._next_id, self._next_id + len(entries))
            self._index.add(embeddings)

            for i, entry in enumerate(entries):
                faiss_id = self._next_id + i
                self._id_map[faiss_id] = entry.entry_id
                self._reverse_id_map[entry.entry_id] = faiss_id

            self._next_id += len(entries)

            self._set_search_params()

    def rebuild_index(self, entries: list[CacheEntry]) -> None:

        with self._lock:
            self._index = self._build_index()
            self._id_map.clear()
            self._reverse_id_map.clear()
            self._next_id = 0
            self._is_trained = False

            if not entries:
                return

            embeddings = np.array(
                [e.embedding for e in entries], dtype=np.float32
            )

            if self.config.index_type == "IVF" and len(entries) >= self.config.nlist:
                self._index.train(embeddings)
                self._is_trained = True

            self._index.add(embeddings)

            for i, entry in enumerate(entries):
                self._id_map[i] = entry.entry_id
                self._reverse_id_map[entry.entry_id] = i

            self._next_id = len(entries)
            self._set_search_params()

    def search(
        self,
        query_embedding: NDArray[np.float32],
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:

        if self._index is None or self._index.ntotal == 0:
            return []

        top_k = top_k or self.config.top_k

        with self._lock:

            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)

            actual_k = min(top_k, self._index.ntotal)

            self._set_search_params()

            distances, indices = self._index.search(query_embedding, actual_k)

        results = []
        for i in range(actual_k):
            faiss_id = int(indices[0][i])
            score = float(distances[0][i])

            if faiss_id == -1:
                continue

            entry_id = self._id_map.get(faiss_id)
            if entry_id is not None:
                results.append((entry_id, score))

        return results

    def search_with_rerank(
        self,
        query_embedding: NDArray[np.float32],
        query_text: str,
        entries: dict[str, CacheEntry],
        top_k: int | None = None,
    ) -> list[SearchResult]:

        initial_k = (top_k or self.config.top_k) * 3
        raw_results = self.search(query_embedding, top_k=initial_k)

        if not raw_results or self._reranker is None:

            return [
                SearchResult(
                    entry=entries[eid],
                    similarity_score=score,
                    rank=i,
                )
                for i, (eid, score) in enumerate(raw_results)
            ]

        pairs = [
            (query_text, entries[eid].query)
            for eid, _ in raw_results
            if eid in entries
        ]

        if not pairs:
            return []

        rerank_scores = self._reranker.predict(pairs)

        search_results = []
        for i, (eid, faiss_score) in enumerate(raw_results):
            if eid not in entries:
                continue
            rr_score = float(rerank_scores[i]) if i < len(rerank_scores) else 0.0

            combined = 0.4 * faiss_score + 0.6 * rr_score
            search_results.append(
                SearchResult(
                    entry=entries[eid],
                    similarity_score=faiss_score,
                    rerank_score=rr_score,
                    rank=0,
                )
            )

        search_results.sort(
            key=lambda r: 0.4 * r.similarity_score + 0.6 * (r.rerank_score or 0),
            reverse=True,
        )

        for i, r in enumerate(search_results):
            r.rank = i

        return search_results[: (top_k or self.config.top_k)]

    def remove_entries(self, entry_ids: list[str]) -> None:

        with self._lock:
            for eid in entry_ids:
                faiss_id = self._reverse_id_map.pop(eid, None)
                if faiss_id is not None:
                    self._id_map.pop(faiss_id, None)

    def size(self) -> int:

        if self._index is None:
            return 0
        return self._index.ntotal

    def _set_search_params(self) -> None:

        if self._index is None:
            return
        if self.config.index_type == "HNSW":
            faiss_parameter = faiss.ParameterSpace()
            faiss_parameter.set_index_parameter(
                self._index, "efSearch", self.config.hnsw_ef_search
            )
        elif self.config.index_type == "IVF":
            faiss_parameter = faiss.ParameterSpace()
            faiss_parameter.set_index_parameter(
                self._index, "nprobe", self.config.nprobe
            )

    def get_stats(self) -> dict:

        if self._index is None:
            return {"status": "not_initialized"}

        stats = {
            "index_type": self.config.index_type,
            "total_vectors": self._index.ntotal,
            "embedding_dim": self.embedding_dim,
            "is_trained": self._is_trained,
        }

        if self.config.index_type == "HNSW":
            stats["hnsw_m"] = self.config.hnsw_m
            stats["ef_construction"] = self.config.hnsw_ef_construction
            stats["ef_search"] = self.config.hnsw_ef_search
        elif self.config.index_type == "IVF":
            stats["nlist"] = self.config.nlist
            stats["nprobe"] = self.config.nprobe

        return stats
