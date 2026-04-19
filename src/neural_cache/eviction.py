from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from neural_cache.config import EvictionConfig, EvictionPolicy

if TYPE_CHECKING:
    from neural_cache.storage import CacheStorage


class EvictionManager:
    def __init__(self, storage: CacheStorage, config: EvictionConfig):
        self.storage = storage
        self.config = config
        self._lock = threading.Lock()
        self._eviction_count = 0
        self._last_check = time.time()

    def should_evict(self) -> bool:
        current_size = self.storage.count()
        max_entries = self.storage.config.max_entries
        threshold = int(max_entries * self.config.high_watermark)
        return current_size >= threshold

    def check_and_evict(self) -> int:
        now = time.time()

        if now - self._last_check < self.config.check_interval:
            return 0

        self._last_check = now

        if not self.should_evict():
            return 0

        with self._lock:
            max_entries = self.storage.config.max_entries
            target_size = int(max_entries * self.config.low_watermark)
            n_to_evict = self.storage.count() - target_size

            if n_to_evict <= 0:
                return 0
            n_to_evict = min(n_to_evict, self.config.eviction_batch_size)

            candidates = self._get_eviction_candidates(n_to_evict)

            if candidates:
                deleted = self.storage.delete_batch(candidates)
                self._eviction_count += deleted
                return deleted

            return 0

    def _get_eviction_candidates(self, n: int) -> list[str]:
        if self.config.policy == EvictionPolicy.LRU:
            if hasattr(self.storage, "get_eviction_candidates_lru"):
                return self.storage.get_eviction_candidates_lru(n)
        elif self.config.policy == EvictionPolicy.LFU:
            if hasattr(self.storage, "get_eviction_candidates_lfu"):
                return self.storage.get_eviction_candidates_lfu(n)
        elif self.config.policy == EvictionPolicy.SCORE_BASED:
            if hasattr(self.storage, "get_eviction_candidates_score"):
                candidates = self.storage.get_eviction_candidates_score(n)
                if self.config.min_quality_score > 0:
                    all_entries = self.storage.get_all()
                    low_quality = {
                        e.entry_id
                        for e in all_entries
                        if e.quality_score < self.config.min_quality_score
                    }
                    candidates = [c for c in candidates if c in low_quality]
                return candidates
        elif self.config.policy == EvictionPolicy.TTL:
            return self._get_ttl_expired_candidates(n)

        return []

    def _get_ttl_expired_candidates(self, n: int) -> list[str]:
        all_entries = self.storage.get_all(limit=n * 2)
        now = time.time()
        expired = [
            e.entry_id
            for e in all_entries
            if (now - e.created_at) > self.config.ttl_seconds
        ]
        return expired[:n]

    def should_insert(self, entry_quality: float = 1.0) -> bool:
        if self.config.policy == EvictionPolicy.SCORE_BASED:
            return entry_quality >= self.config.min_quality_score
        return True

    @property
    def eviction_count(self) -> int:
        return self._eviction_count

    def get_stats(self) -> dict:
        return {
            "policy": self.config.policy.value,
            "cache_size": self.storage.count(),
            "max_entries": self.storage.config.max_entries,
            "eviction_count": self._eviction_count,
            "high_watermark": self.config.high_watermark,
            "low_watermark": self.config.low_watermark,
        }
