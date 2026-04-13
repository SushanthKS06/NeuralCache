from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Awaitable

import structlog

from neural_cache.config import CacheConfig
from neural_cache.models import (
    CacheAction,
    CacheEntry,
    CacheResult,
    MetricsSnapshot,
    SearchResult,
)
from neural_cache.encoder import QueryEncoder
from neural_cache.storage import CacheStorage, InMemoryStorage, create_storage
from neural_cache.search import SimilaritySearchEngine
from neural_cache.decision import CacheDecisionPolicy
from neural_cache.adaptation import ResponseAdaptor
from neural_cache.llm_client import LLMClient, create_llm_client
from neural_cache.eviction import EvictionManager
from neural_cache.metrics import MetricsCollector
from neural_cache.hybrid_retrieval import HybridRetriever

logger = structlog.get_logger("neural_cache")

LLMGenerateFunc = Callable[[str], Awaitable[tuple[str, dict[str, Any]]]]

class NeuralCache:
    def __init__(self, config: CacheConfig | None = None):
        self.config = config or CacheConfig()

        self.encoder = QueryEncoder(self.config.encoder)
        self.storage: CacheStorage = create_storage(self.config.storage)
        self.search_engine = SimilaritySearchEngine(
            self.config.search,
            embedding_dim=self.encoder.dimension,
        )
        self.decision_policy = CacheDecisionPolicy(self.config.decision)
        self.adaptor = ResponseAdaptor(self.config.adaptation)
        self.eviction_manager = EvictionManager(self.storage, self.config.eviction)
        self.metrics = MetricsCollector(self.config.metrics)

        self.hybrid_retriever: HybridRetriever | None = None

        self._llm_client: LLMClient | None = None
        self._user_llm_func: LLMGenerateFunc | None = None

        self._entries: dict[str, CacheEntry] = {}

        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return

        self.encoder.warmup()

        self.search_engine.initialize_index()

        existing = self.storage.get_all()
        for entry in existing:
            self._entries[entry.entry_id] = entry

        if existing:
            self.search_engine.rebuild_index(existing)

        if any(e.tags for e in existing):
            self.hybrid_retriever = HybridRetriever()
            self.hybrid_retriever.add_entries(existing)

        if self.config.adaptation.mode.value in ("llm_refine", "hybrid"):
            if self._llm_client:
                self.adaptor.set_llm_client(self._llm_client)

        self._initialized = True
        logger.info("neural_cache_initialized", config=self.config.environment)

    def set_llm_client(self, client: LLMClient) -> None:
        self._llm_client = client
        self.adaptor.set_llm_client(client)

    def set_llm_function(self, func: LLMGenerateFunc) -> None:
        self._user_llm_func = func

    async def get(
        self,
        query: str,
        llm_generate: LLMGenerateFunc | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CacheResult:
        if not self.config.enabled:

            return await self._bypass_llm(query, llm_generate)

        if not self._initialized:
            await self.initialize()

        request_start = time.monotonic()
        request_id = str(uuid.uuid4())
        latency_breakdown: dict[str, float] = {}

        try:
            encode_start = time.monotonic()
            embedding = self.encoder.encode_single(query)
            latency_breakdown["embedding"] = (time.monotonic() - encode_start) * 1000

            search_start = time.monotonic()
            search_results = await self._search(query, embedding)
            latency_breakdown["search"] = (time.monotonic() - search_start) * 1000

            decision_start = time.monotonic()
            decision = self.decision_policy.decide(
                search_results=search_results,
                query=query,
                context=metadata,
            )
            latency_breakdown["decision"] = (time.monotonic() - decision_start) * 1000

            if decision.action == CacheAction.HIT:
                result = self._handle_hit(query, decision, request_id, request_start, latency_breakdown)

            elif decision.action == CacheAction.HIT_WITH_ADAPTATION:
                result = self._handle_adaptation(query, decision, request_start, latency_breakdown, request_id)

            elif decision.action == CacheAction.MISS:
                result = await self._handle_miss(
                    query, embedding, decision, request_start, latency_breakdown,
                    request_id, llm_generate, metadata,
                )
            else:
                result = CacheResult(
                    response="",
                    action=CacheAction.ERROR,
                    from_cache=False,
                    similarity_score=0.0,
                    total_latency_ms=(time.monotonic() - request_start) * 1000,
                    request_id=request_id,
                )

            self.metrics.record_request(result, latency_breakdown)
            self.metrics.update_cache_size(self.storage.count())

            return result

        except Exception as e:
            logger.error("cache_error", error=str(e), query=query[:100])
            if llm_generate or self._llm_client:
                return await self._bypass_llm(query, llm_generate)
            return CacheResult(
                response=f"Error: {e}",
                action=CacheAction.ERROR,
                from_cache=False,
                similarity_score=0.0,
                total_latency_ms=(time.monotonic() - request_start) * 1000,
                request_id=request_id,
                metadata={"error": str(e)},
            )

    def _handle_hit(
        self,
        query: str,
        decision: Any,
        request_id: str,
        request_start: float,
        latency_breakdown: dict,
    ) -> CacheResult:
        entry = decision.entry
        assert entry is not None

        updated_entry = entry.record_access()
        self._entries[entry.entry_id] = updated_entry
        self.storage.put(updated_entry)

        total_ms = (time.monotonic() - request_start) * 1000

        return CacheResult(
            response=entry.response,
            action=CacheAction.HIT,
            from_cache=True,
            similarity_score=decision.similarity_score,
            total_latency_ms=total_ms,
            cache_entry=entry,
            adapted=False,
            request_id=request_id,
        )

    def _handle_adaptation(
        self,
        query: str,
        decision: Any,
        request_start: float,
        latency_breakdown: dict,
        request_id: str,
    ) -> CacheResult:
        entry = decision.entry
        assert entry is not None

        adapt_start = time.monotonic()
        adapted_response = self.adaptor.adapt(
            cached_query=entry.query,
            cached_response=entry.response,
            new_query=query,
            similarity=decision.similarity_score,
        )
        latency_breakdown["adaptation"] = (time.monotonic() - adapt_start) * 1000

        updated_entry = entry.record_access()
        self._entries[entry.entry_id] = updated_entry
        self.storage.put(updated_entry)

        total_ms = (time.monotonic() - request_start) * 1000

        return CacheResult(
            response=adapted_response,
            action=CacheAction.HIT_WITH_ADAPTATION,
            from_cache=True,
            similarity_score=decision.similarity_score,
            total_latency_ms=total_ms,
            cache_entry=entry,
            adapted=True,
            request_id=request_id,
        )

    async def _handle_miss(
        self,
        query: str,
        embedding,
        decision: Any,
        request_start: float,
        latency_breakdown: dict,
        request_id: str,
        llm_generate: LLMGenerateFunc | None,
        metadata: dict[str, Any] | None,
    ) -> CacheResult:
        llm_start = time.monotonic()

        if llm_generate:
            response_text, response_meta = await llm_generate(query)
        elif self._user_llm_func:
            response_text, response_meta = await self._user_llm_func(query)
        elif self._llm_client:
            response_text, response_meta = await self._llm_client.generate(query)
        else:
            raise RuntimeError(
                "Cache miss but no LLM configured. "
                "Provide llm_generate or set_llm_client()."
            )

        latency_breakdown["llm"] = (time.monotonic() - llm_start) * 1000
        new_entry = CacheEntry(
            query=query,
            embedding=embedding.tolist(),
            response=response_text,
            response_metadata=response_meta,
            embedding_model=self.config.encoder.model_name.value,
            metadata=metadata or {},
        )

        if not self.config.dry_run:
            self._entries[new_entry.entry_id] = new_entry
            self.storage.put(new_entry)
            self.search_engine.add_entries([new_entry])

            if self.hybrid_retriever:
                self.hybrid_retriever.add_entries([new_entry])

            self.eviction_manager.check_and_evict()

        total_ms = (time.monotonic() - request_start) * 1000

        return CacheResult(
            response=response_text,
            action=CacheAction.MISS,
            from_cache=False,
            similarity_score=decision.similarity_score,
            total_latency_ms=total_ms,
            cache_entry=new_entry,
            request_id=request_id,
        )

    async def _search(self, query: str, embedding) -> list[SearchResult]:
        top_k = self.config.search.top_k

        if self.hybrid_retriever:

            raw_results = self.search_engine.search(embedding, top_k=top_k * 2)
            results = self.hybrid_retriever.search(query, raw_results, top_k=top_k)
            return results
        else:
            if self.config.decision.enable_reranking:
                results = self.search_engine.search_with_rerank(
                    embedding, query, self._entries, top_k=top_k,
                )
                return results
            else:
                raw_results = self.search_engine.search(embedding, top_k=top_k)
                results = [
                    SearchResult(
                        entry=self._entries[eid],
                        similarity_score=score,
                        rank=i,
                    )
                    for i, (eid, score) in enumerate(raw_results)
                    if eid in self._entries
                ]
                return results

    async def _bypass_llm(
        self,
        query: str,
        llm_generate: LLMGenerateFunc | None,
    ) -> CacheResult:
        start = time.monotonic()

        if llm_generate:
            response_text, response_meta = await llm_generate(query)
        elif self._user_llm_func:
            response_text, response_meta = await self._user_llm_func(query)
        elif self._llm_client:
            response_text, response_meta = await self._llm_client.generate(query)
        else:
            raise RuntimeError("No LLM configured")

        total_ms = (time.monotonic() - start) * 1000

        return CacheResult(
            response=response_text,
            action=CacheAction.MISS,
            from_cache=False,
            similarity_score=0.0,
            total_latency_ms=total_ms,
            metadata=response_meta,
        )

    def record_feedback(
        self,
        request_id: str,
        quality_score: float,
        was_good: bool,
    ) -> None:
        for entry in self._entries.values():
            if entry.entry_id == request_id or any(
                entry.entry_id == request_id for _ in [1]
            ):
                self.decision_policy.record_feedback(
                    similarity=0.0,
                    was_good=was_good,
                    quality_score=quality_score,
                )
                entry.update_quality(quality_score)
                self.storage.put(entry)
                break

    def get_metrics(self) -> MetricsSnapshot:
        snapshot = self.metrics.get_snapshot()
        snapshot.cache_size = self.storage.count()
        snapshot.eviction_count = self.eviction_manager.eviction_count
        return snapshot

    def get_stats(self) -> dict:
        return {
            "cache_size": self.storage.count(),
            "search_engine": self.search_engine.get_stats(),
            "decision_policy": self.decision_policy.get_stats(),
            "eviction": self.eviction_manager.get_stats(),
            "encoder": self.encoder.get_model_info(),
        }

    def clear(self) -> None:
        self._entries.clear()
        self.storage.clear()
        self.search_engine.initialize_index()

    def close(self) -> None:
        self.storage.close()

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.close()
