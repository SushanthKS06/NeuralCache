from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import structlog

from neural_cache.config import MetricsConfig
from neural_cache.models import CacheAction, CacheResult, MetricsSnapshot


@dataclass
class LatencyRecord:
    timestamp: float
    total_ms: float
    embedding_ms: float = 0.0
    search_ms: float = 0.0
    decision_ms: float = 0.0
    adaptation_ms: float = 0.0
    llm_ms: float = 0.0


class MetricsCollector:
    def __init__(self, config: MetricsConfig):
        self.config = config
        self._lock = threading.Lock()

        self._timestamps: deque[float] = deque(maxlen=1_000_000)
        self._actions: deque[CacheAction] = deque(maxlen=1_000_000)
        self._latencies: deque[LatencyRecord] = deque(maxlen=1_000_000)
        self._similarity_scores: deque[tuple[float, float]] = deque(maxlen=1_000_000)
        self._quality_scores: deque[tuple[float, float]] = deque(maxlen=1_000_000)

        self._registry = None
        if config.enable_prometheus:
            self._setup_prometheus()

        self.logger = structlog.get_logger("neural_cache.metrics")

    def record_request(
        self,
        result: CacheResult,
        latency_breakdown: dict[str, float] | None = None,
    ) -> None:
        with self._lock:
            now = time.time()
            self._timestamps.append(now)
            self._actions.append(result.action)

            if latency_breakdown:
                self._latencies.append(LatencyRecord(
                    timestamp=now,
                    total_ms=result.total_latency_ms,
                    embedding_ms=latency_breakdown.get("embedding", 0.0),
                    search_ms=latency_breakdown.get("search", 0.0),
                    decision_ms=latency_breakdown.get("decision", 0.0),
                    adaptation_ms=latency_breakdown.get("adaptation", 0.0),
                    llm_ms=latency_breakdown.get("llm", 0.0),
                ))

            if result.similarity_score > 0:
                self._similarity_scores.append((now, result.similarity_score))

            self._update_prometheus(result)

    def record_quality_score(self, score: float) -> None:
        with self._lock:
            self._quality_scores.append((time.time(), score))

    def get_snapshot(self) -> MetricsSnapshot:
        with self._lock:
            now = time.time()
            cutoff = now - self.config.sliding_window_seconds

            actions_in_window = [
                a for a, t in zip(self._actions, self._timestamps)
                if t >= cutoff
            ]
            latencies_in_window = [
                lr for lr in self._latencies if lr.timestamp >= cutoff
            ]
            similarity_in_window = [
                s for t, s in self._similarity_scores if t >= cutoff
            ]
            quality_in_window = [
                s for t, s in self._quality_scores if t >= cutoff
            ]

            total = len(actions_in_window)
            hits = actions_in_window.count(CacheAction.HIT)
            misses = actions_in_window.count(CacheAction.MISS)
            adaptations = actions_in_window.count(CacheAction.HIT_WITH_ADAPTATION)
            errors = actions_in_window.count(CacheAction.ERROR)

            if latencies_in_window:
                total_latencies = sorted([lr.total_ms for lr in latencies_in_window])
                avg_latency = sum(total_latencies) / len(total_latencies)
                p50 = self._percentile(total_latencies, 50)
                p95 = self._percentile(total_latencies, 95)
                p99 = self._percentile(total_latencies, 99)

                decision_latencies = [lr.decision_ms for lr in latencies_in_window]
                avg_decision_ms = sum(decision_latencies) / len(decision_latencies)

                embedding_latencies = [lr.embedding_ms for lr in latencies_in_window]
                avg_embedding_ms = sum(embedding_latencies) / len(embedding_latencies)

                search_latencies = [lr.search_ms for lr in latencies_in_window]
                avg_search_ms = sum(search_latencies) / len(search_latencies)
            else:
                avg_latency = p50 = p95 = p99 = 0.0
                avg_decision_ms = avg_embedding_ms = avg_search_ms = 0.0

            hit_rate = (hits + adaptations) / total if total > 0 else 0.0
            avg_quality = (
                sum(quality_in_window) / len(quality_in_window)
                if quality_in_window
                else 0.0
            )
            avg_similarity = (
                sum(similarity_in_window) / len(similarity_in_window)
                if similarity_in_window
                else 0.0
            )

            return MetricsSnapshot(
                total_requests=total,
                cache_hits=hits,
                cache_misses=misses,
                cache_hit_with_adaptation=adaptations,
                errors=errors,
                avg_latency_ms=avg_latency,
                p50_latency_ms=p50,
                p95_latency_ms=p95,
                p99_latency_ms=p99,
                avg_cache_decision_latency_ms=avg_decision_ms,
                avg_embedding_latency_ms=avg_embedding_ms,
                avg_search_latency_ms=avg_search_ms,
                cache_hit_rate=hit_rate,
                avg_quality_score=avg_quality,
                avg_similarity_score=avg_similarity,
            )

    def _percentile(self, sorted_data: list[float], p: float) -> float:
        if not sorted_data:
            return 0.0
        k = (len(sorted_data) - 1) * (p / 100.0)
        f = int(k)
        c = f + 1
        if c >= len(sorted_data):
            return sorted_data[-1]
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])

    def _setup_prometheus(self) -> None:
        try:
            from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge

            self._registry = CollectorRegistry()

            self._prom_counter = Counter(
                "neural_cache_requests_total",
                "Total cache requests",
                ["action"],
                registry=self._registry,
            )
            self._prom_latency = Histogram(
                "neural_cache_latency_ms",
                "Request latency in milliseconds",
                registry=self._registry,
            )
            self._prom_hit_rate = Gauge(
                "neural_cache_hit_rate",
                "Current cache hit rate",
                registry=self._registry,
            )
            self._prom_cache_size = Gauge(
                "neural_cache_size",
                "Current cache size",
                registry=self._registry,
            )
        except ImportError:
            self._registry = None

    def _update_prometheus(self, result: CacheResult) -> None:
        if self._registry is None:
            return
        try:
            self._prom_counter.labels(action=result.action.value).inc()
            self._prom_latency.observe(result.total_latency_ms)
        except Exception:
            pass

    def update_cache_size(self, size: int) -> None:
        if self._registry and hasattr(self, "_prom_cache_size"):
            self._prom_cache_size.set(size)

    def get_prometheus_registry(self):
        return self._registry
