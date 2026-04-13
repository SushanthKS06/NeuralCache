from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

from neural_cache.models import CacheEntry, SearchResult

class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._doc_freq: dict[str, int] = defaultdict(int)
        self._term_freqs: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._doc_lengths: dict[str, int] = {}
        self._n_docs = 0
        self._avg_doc_length = 0.0

    def add_document(self, doc_id: str, text: str) -> None:
        tokens = self._tokenize(text)
        self._doc_lengths[doc_id] = len(tokens)
        self._n_docs += 1

        for token in tokens:
            self._term_freqs[doc_id][token] += 1

        for token in set(tokens):
            self._doc_freq[token] += 1

        self._avg_doc_length = sum(self._doc_lengths.values()) / self._n_docs

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        tokens = self._tokenize(query)

        if not tokens or self._n_docs == 0:
            return []

        scores: dict[str, float] = defaultdict(float)

        for token in tokens:
            df = self._doc_freq.get(token, 0)
            if df == 0:
                continue
            idf = math.log(
                1 + (self._n_docs - df + 0.5) / (df + 0.5)
            )

            for doc_id, tf in self._term_freqs.items():
                if token not in tf:
                    continue

                doc_len = self._doc_lengths.get(doc_id, 1)
                tf_val = tf[token]
                numerator = tf_val * (self.k1 + 1)
                denominator = tf_val + self.k1 * (
                    1 - self.b + self.b * doc_len / self._avg_doc_length
                )
                scores[doc_id] += idf * numerator / denominator

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        tokens = re.findall(r'\b\w{2,}\b', text)
        return tokens

class HybridRetriever:
    def __init__(
        self,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4,
        rrf_k: int = 60,
    ):
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k
        self._bm25_index = BM25Index()
        self._entries: dict[str, CacheEntry] = {}

    def add_entries(self, entries: list[CacheEntry]) -> None:
        for entry in entries:
            self._entries[entry.entry_id] = entry
            self._bm25_index.add_document(entry.entry_id, entry.query)

    def rebuild_keyword_index(self) -> None:
        self._bm25_index = BM25Index()
        for entry in self._entries.values():
            self._bm25_index.add_document(entry.entry_id, entry.query)

    def search(
        self,
        query_text: str,
        semantic_results: list[tuple[str, float]],
        top_k: int = 5,
    ) -> list[SearchResult]:
        keyword_results = self._bm25_index.search(query_text, top_k=top_k * 2)

        semantic_ranks: dict[str, int] = {
            eid: i for i, (eid, _) in enumerate(semantic_results)
        }
        keyword_ranks: dict[str, int] = {
            eid: i for i, (eid, _) in enumerate(keyword_results)
        }

        all_ids = set(semantic_ranks.keys()) | set(keyword_ranks.keys())

        rrf_scores: dict[str, float] = {}
        for eid in all_ids:
            semantic_rank = semantic_ranks.get(eid, len(semantic_results))
            keyword_rank = keyword_ranks.get(eid, len(keyword_results))

            semantic_rrf = 1.0 / (self.rrf_k + semantic_rank)
            keyword_rrf = 1.0 / (self.rrf_k + keyword_rank)

            rrf_scores[eid] = (
                self.semantic_weight * semantic_rrf
                + self.keyword_weight * keyword_rrf
            )

        results = []
        for eid, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            if len(results) >= top_k:
                break
            if eid in self._entries:
                sem_sim = dict(semantic_results).get(eid, 0.0)
                results.append(
                    SearchResult(
                        entry=self._entries[eid],
                        similarity_score=sem_sim,
                        rank=len(results),
                        rerank_score=score,
                    )
                )

        return results

    def get_stats(self) -> dict[str, Any]:
        return {
            "bm25_documents": self._bm25_index._n_docs,
            "bm25_vocabulary": len(self._bm25_index._doc_freq),
            "total_entries": len(self._entries),
            "semantic_weight": self.semantic_weight,
            "keyword_weight": self.keyword_weight,
        }
