from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CacheAction(str, Enum):
    HIT = "hit"
    HIT_WITH_ADAPTATION = "hit_with_adaptation"
    MISS = "miss"
    ERROR = "error"


class EntryStatus(str, Enum):
    ACTIVE = "active"
    STALE = "stale"
    EVICTED = "evicted"
    FLAGGED = "flagged"


@dataclass(frozen=True)
class CacheEntry:
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    embedding: list[float] = field(default_factory=list)
    response: str = ""
    response_metadata: dict[str, Any] = field(default_factory=dict)
    embedding_model: str = ""
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    quality_score: float = 1.0
    status: EntryStatus = EntryStatus.ACTIVE
    tags: frozenset[str] = field(default_factory=frozenset)
    metadata: dict[str, Any] = field(default_factory=dict)

    def record_access(self) -> CacheEntry:
        return CacheEntry(
            entry_id=self.entry_id,
            query=self.query,
            embedding=self.embedding,
            response=self.response,
            response_metadata=self.response_metadata,
            embedding_model=self.embedding_model,
            created_at=self.created_at,
            last_accessed=time.time(),
            access_count=self.access_count + 1,
            quality_score=self.quality_score,
            status=self.status,
            tags=self.tags,
            metadata=self.metadata,
        )

    def update_quality(self, score: float) -> CacheEntry:
        alpha = 0.3
        new_score = alpha * score + (1 - alpha) * self.quality_score
        return CacheEntry(
            entry_id=self.entry_id,
            query=self.query,
            embedding=self.embedding,
            response=self.response,
            response_metadata=self.response_metadata,
            embedding_model=self.embedding_model,
            created_at=self.created_at,
            last_accessed=self.last_accessed,
            access_count=self.access_count,
            quality_score=new_score,
            status=self.status,
            tags=self.tags,
            metadata=self.metadata,
        )


@dataclass
class SearchResult:
    entry: CacheEntry
    similarity_score: float
    rank: int = 0
    rerank_score: float | None = None


@dataclass
class CacheDecision:
    action: CacheAction
    confidence: float
    entry: CacheEntry | None = None
    similarity_score: float = 0.0
    reasoning: str = ""
    should_explore: bool = False
    decision_latency_ms: float = 0.0


@dataclass
class CacheResult:
    response: str
    action: CacheAction
    from_cache: bool
    similarity_score: float
    total_latency_ms: float
    cache_entry: CacheEntry | None = None
    adapted: bool = False
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricsSnapshot:
    timestamp: float = field(default_factory=time.time)
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_with_adaptation: int = 0
    errors: int = 0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    avg_cache_decision_latency_ms: float = 0.0
    avg_embedding_latency_ms: float = 0.0
    avg_search_latency_ms: float = 0.0
    cache_hit_rate: float = 0.0
    avg_quality_score: float = 0.0
    avg_similarity_score: float = 0.0
    cache_size: int = 0
    eviction_count: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.cache_hits + self.cache_hit_with_adaptation) / self.total_requests
