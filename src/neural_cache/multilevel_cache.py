from __future__ import annotations

from typing import Any

from neural_cache.config import StorageConfig, StorageBackend
from neural_cache.models import CacheEntry
from neural_cache.storage import CacheStorage, InMemoryStorage, create_storage


class MultiLevelCache:
    def __init__(
        self,
        l1_max_entries: int = 5000,
        l2_config: StorageConfig | None = None,
        promote_on_l2_hit: bool = True,
        async_l2_write: bool = False,
    ):
        l1_config = StorageConfig(
            backend=StorageBackend.IN_MEMORY,
            max_entries=l1_max_entries,
        )
        self.l1: InMemoryStorage = InMemoryStorage(l1_config)

        l2_config = l2_config or StorageConfig(
            backend=StorageBackend.SQLITE,
            max_entries=100_000,
        )
        self.l2: CacheStorage = create_storage(l2_config)

        self.promote_on_l2_hit = promote_on_l2_hit
        self.async_l2_write = async_l2_write

        self._l1_hits = 0
        self._l2_hits = 0
        self._misses = 0
        self._promotions = 0

    def get(self, entry_id: str) -> CacheEntry | None:
        entry = self.l1.get(entry_id)
        if entry is not None:
            self._l1_hits += 1
            self.l1.update_access(entry_id)
            return entry

        entry = self.l2.get(entry_id)
        if entry is not None:
            self._l2_hits += 1
            self.l2.update_access(entry_id)
            if self.promote_on_l2_hit:
                self._promote_to_l1(entry)
            return entry

        self._misses += 1
        return None

    def put(self, entry: CacheEntry, write_to_l2: bool = True) -> None:
        self.l1.put(entry)
        self._evict_l1_if_needed()

        if write_to_l2:
            self.l2.put(entry)

    def delete(self, entry_id: str) -> bool:
        l1_deleted = self.l1.delete(entry_id)
        l2_deleted = self.l2.delete(entry_id)
        return l1_deleted or l2_deleted

    def _promote_to_l1(self, entry: CacheEntry) -> None:
        self.l1.put(entry)
        self._promotions += 1
        self._evict_l1_if_needed()

    def _evict_l1_if_needed(self) -> None:
        max_entries = self.l1.config.max_entries
        current = self.l1.count()

        if current > max_entries:
            n_to_evict = current - max_entries
            candidates = self.l1.get_eviction_candidates_lru(n_to_evict)
            if candidates:
                self.l1.delete_batch(candidates)

    def get_stats(self) -> dict[str, Any]:
        total = self._l1_hits + self._l2_hits + self._misses
        return {
            "l1_size": self.l1.count(),
            "l1_max": self.l1.config.max_entries,
            "l2_size": self.l2.count(),
            "l2_max": self.l2.config.max_entries,
            "l1_hits": self._l1_hits,
            "l2_hits": self._l2_hits,
            "misses": self._misses,
            "promotions": self._promotions,
            "l1_hit_rate": self._l1_hits / total if total > 0 else 0.0,
            "combined_hit_rate": (self._l1_hits + self._l2_hits) / total if total > 0 else 0.0,
        }

    def clear(self) -> None:
        self.l1.clear()
        self.l2.clear()

    def close(self) -> None:
        self.l1.close()
        self.l2.close()
