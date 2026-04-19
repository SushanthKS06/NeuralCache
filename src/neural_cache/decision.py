from __future__ import annotations

import random
import time
from collections import deque
from typing import Any

import numpy as np

from neural_cache.config import DecisionConfig, DecisionStrategy
from neural_cache.models import (
    CacheAction,
    CacheDecision,
    CacheEntry,
    SearchResult,
)

RERANK_WEIGHT_FAISS = 0.4
RERANK_WEIGHT_CROSS = 0.6


class CacheDecisionPolicy:
    def __init__(self, config: DecisionConfig):
        self.config = config
        self._feedback_history: deque[tuple[float, bool, float]] = deque(
            maxlen=config.adaptive_window_size
        )
        self._current_threshold = config.similarity_threshold
        self._total_decisions = 0
        self._exploration_count = 0

    def decide(
        self,
        search_results: list[SearchResult],
        query: str = "",
        context: dict[str, Any] | None = None,
    ) -> CacheDecision:
        start_time = time.monotonic()
        self._total_decisions += 1

        if not search_results:
            return self._make_miss_decision(time.monotonic() - start_time)

        best_result = search_results[0]
        similarity = best_result.similarity_score
        if best_result.rerank_score is not None:
            effective_score = (
                RERANK_WEIGHT_FAISS * similarity
                + RERANK_WEIGHT_CROSS * best_result.rerank_score
            )
        else:
            effective_score = similarity

        should_explore = self._should_explore()
        if should_explore:
            self._exploration_count += 1
            return CacheDecision(
                action=CacheAction.MISS,
                confidence=0.0,
                similarity_score=similarity,
                reasoning=f"Exploration: exploring despite similarity {similarity:.3f}",
                should_explore=True,
                decision_latency_ms=(time.monotonic() - start_time) * 1000,
            )

        if self.config.strategy == DecisionStrategy.FIXED_THRESHOLD:
            decision = self._fixed_threshold_decision(
                effective_score, similarity, best_result.entry, time.monotonic() - start_time
            )
        elif self.config.strategy == DecisionStrategy.ADAPTIVE_THRESHOLD:
            decision = self._adaptive_threshold_decision(
                effective_score, similarity, best_result.entry, time.monotonic() - start_time
            )
        elif self.config.strategy == DecisionStrategy.LEARNED_SCORING:
            decision = self._learned_scoring_decision(
                effective_score, similarity, best_result.entry, context, time.monotonic() - start_time
            )
        else:
            decision = self._fixed_threshold_decision(
                effective_score, similarity, best_result.entry, time.monotonic() - start_time
            )

        return decision

    def record_feedback(
        self,
        similarity: float,
        was_good: bool,
        quality_score: float,
    ) -> None:
        self._feedback_history.append((similarity, was_good, quality_score))
        if len(self._feedback_history) >= self.config.adaptive_window_size:
            self._recalibrate_threshold()

    def _fixed_threshold_decision(
        self,
        effective_score: float,
        raw_similarity: float,
        entry: CacheEntry,
        latency: float,
    ) -> CacheDecision:
        if effective_score >= self.config.high_confidence_threshold:
            return CacheDecision(
                action=CacheAction.HIT,
                confidence=effective_score,
                entry=entry,
                similarity_score=raw_similarity,
                reasoning=f"High confidence hit: {effective_score:.3f} >= {self.config.high_confidence_threshold}",
                decision_latency_ms=latency * 1000,
            )
        elif effective_score >= self._current_threshold:
            return CacheDecision(
                action=CacheAction.HIT_WITH_ADAPTATION,
                confidence=effective_score,
                entry=entry,
                similarity_score=raw_similarity,
                reasoning=f"Hit with adaptation: {effective_score:.3f} >= {self._current_threshold}",
                decision_latency_ms=latency * 1000,
            )
        else:
            return CacheDecision(
                action=CacheAction.MISS,
                confidence=1.0 - effective_score,
                similarity_score=raw_similarity,
                reasoning=f"Miss: {effective_score:.3f} < {self._current_threshold}",
                decision_latency_ms=latency * 1000,
            )

    def _adaptive_threshold_decision(
        self,
        effective_score: float,
        raw_similarity: float,
        entry: CacheEntry,
        latency: float,
    ) -> CacheDecision:
        return self._fixed_threshold_decision(
            effective_score, raw_similarity, entry, latency
        )

    def _learned_scoring_decision(
        self,
        effective_score: float,
        raw_similarity: float,
        entry: CacheEntry,
        context: dict[str, Any] | None,
        latency: float,
    ) -> CacheDecision:
        if len(self._feedback_history) < self.config.min_feedback_samples:
            return self._fixed_threshold_decision(
                effective_score, raw_similarity, entry, latency
            )

        similar_feedback = [
            (was_good, q)
            for sim, was_good, q in self._feedback_history
            if abs(sim - effective_score) < 0.05
        ]

        if similar_feedback:
            avg_quality = np.mean([q for _, q in similar_feedback])
            hit_rate = np.mean([int(g) for g, _ in similar_feedback])
            predicted_reliability = 0.5 * hit_rate + 0.5 * avg_quality

            if predicted_reliability >= 0.85:
                action = CacheAction.HIT
            elif predicted_reliability >= 0.7:
                action = CacheAction.HIT_WITH_ADAPTATION
            else:
                action = CacheAction.MISS
        else:
            predicted_reliability = effective_score
            action = CacheAction.HIT if effective_score >= self._current_threshold else CacheAction.MISS

        return CacheDecision(
            action=action,
            confidence=predicted_reliability,
            entry=entry if action != CacheAction.MISS else None,
            similarity_score=raw_similarity,
            reasoning=f"Learned scoring: predicted_reliability={predicted_reliability:.3f}",
            decision_latency_ms=latency * 1000,
        )

    def _should_explore(self) -> bool:
        if self.config.strategy != DecisionStrategy.ADAPTIVE_THRESHOLD:
            return False
        return random.random() < self.config.exploration_rate

    def _recalibrate_threshold(self) -> None:
        if len(self._feedback_history) < 10:
            return

        best_threshold = self._current_threshold
        best_score = -1.0

        for candidate in np.arange(0.5, 0.99, 0.01):
            correct = 0
            total = 0
            for sim, was_good, _ in self._feedback_history:
                predicted_hit = sim >= candidate
                if predicted_hit == was_good:
                    correct += 1
                total += 1

            accuracy = correct / total if total > 0 else 0
            if accuracy > best_score:
                best_score = accuracy
                best_threshold = candidate

        self._current_threshold = best_threshold

    def _make_miss_decision(self, latency: float) -> CacheDecision:
        return CacheDecision(
            action=CacheAction.MISS,
            confidence=1.0,
            similarity_score=0.0,
            reasoning="No similar entries found",
            decision_latency_ms=latency * 1000,
        )

    def get_stats(self) -> dict[str, Any]:
        return {
            "strategy": self.config.strategy.value,
            "current_threshold": self._current_threshold,
            "total_decisions": self._total_decisions,
            "exploration_count": self._exploration_count,
            "feedback_samples": len(self._feedback_history),
        }
