
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any

import orjson

from neural_cache.config import StorageConfig, StorageBackend
from neural_cache.models import CacheEntry, EntryStatus

class CacheStorage(ABC):

    @abstractmethod
    def put(self, entry: CacheEntry) -> None:

    @abstractmethod
    def get(self, entry_id: str) -> CacheEntry | None:

    @abstractmethod
    def get_all(self, limit: int | None = None) -> list[CacheEntry]:

    @abstractmethod
    def delete(self, entry_id: str) -> bool:

    @abstractmethod
    def delete_batch(self, entry_ids: list[str]) -> int:

    @abstractmethod
    def count(self) -> int:

    @abstractmethod
    def clear(self) -> None:

    @abstractmethod
    def get_embeddings_batch(self, entry_ids: list[str]) -> dict[str, list[float]]:

    @abstractmethod
    def update_access(self, entry_id: str) -> None:

    @abstractmethod
    def update_quality(self, entry_id: str, score: float) -> None:

    @abstractmethod
    def close(self) -> None:

class InMemoryStorage(CacheStorage):

    def __init__(self, config: StorageConfig):
        self.config = config
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

    def put(self, entry: CacheEntry) -> None:
        with self._lock:

            if entry.entry_id in self._store:
                self._store.pop(entry.entry_id)
            self._store[entry.entry_id] = entry

    def get(self, entry_id: str) -> CacheEntry | None:
        with self._lock:
            if entry_id in self._store:

                self._store.move_to_end(entry_id)
                return self._store[entry_id]
            return None

    def get_all(self, limit: int | None = None) -> list[CacheEntry]:
        with self._lock:
            entries = list(self._store.values())
            if limit is not None:
                entries = entries[:limit]
            return entries

    def delete(self, entry_id: str) -> bool:
        with self._lock:
            if entry_id in self._store:
                self._store.pop(entry_id)
                return True
            return False

    def delete_batch(self, entry_ids: list[str]) -> int:
        with self._lock:
            count = 0
            for eid in entry_ids:
                if self._store.pop(eid, None) is not None:
                    count += 1
            return count

    def count(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def get_embeddings_batch(self, entry_ids: list[str]) -> dict[str, list[float]]:
        with self._lock:
            result = {}
            for eid in entry_ids:
                if eid in self._store:
                    result[eid] = self._store[eid].embedding
            return result

    def update_access(self, entry_id: str) -> None:
        with self._lock:
            if entry_id in self._store:
                old = self._store.pop(entry_id)
                new = old.record_access()
                self._store[entry_id] = new

    def update_quality(self, entry_id: str, score: float) -> None:
        with self._lock:
            if entry_id in self._store:
                old = self._store.pop(entry_id)
                new = old.update_quality(score)
                self._store[entry_id] = new

    def get_eviction_candidates_lru(self, n: int) -> list[str]:

        with self._lock:
            return list(self._store.keys())[:n]

    def get_eviction_candidates_lfu(self, n: int) -> list[str]:

        with self._lock:
            sorted_entries = sorted(
                self._store.values(), key=lambda e: e.access_count
            )
            return [e.entry_id for e in sorted_entries[:n]]

    def get_eviction_candidates_score(self, n: int) -> list[str]:

        with self._lock:
            sorted_entries = sorted(
                self._store.values(), key=lambda e: e.quality_score
            )
            return [e.entry_id for e in sorted_entries[:n]]

    def close(self) -> None:
        pass

class SQLiteStorage(CacheStorage):

    CREATE_TABLES_SQL =

    def __init__(self, config: StorageConfig):
        import sqlite3

        self.config = config
        self._db_path = Path(config.db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
        )
        if config.wal_mode:
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(self.CREATE_TABLES_SQL)
        self._conn.commit()
        self._lock = threading.RLock()

    def put(self, entry: CacheEntry) -> None:
        import sqlite3
        import numpy as np

        with self._lock:

            embedding_blob = np.array(entry.embedding, dtype=np.float32).tobytes()

            self._conn.execute(
                ,
                (
                    entry.entry_id,
                    entry.query,
                    entry.response,
                    orjson.dumps(entry.response_metadata).decode(),
                    entry.embedding_model,
                    entry.created_at,
                    entry.last_accessed,
                    entry.access_count,
                    entry.quality_score,
                    entry.status.value,
                    orjson.dumps(list(entry.tags)).decode(),
                    orjson.dumps(entry.metadata).decode(),
                ),
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO embeddings (entry_id, vector) VALUES (?, ?)",
                (entry.entry_id, embedding_blob),
            )
            self._conn.commit()

    def get(self, entry_id: str) -> CacheEntry | None:
        import numpy as np

        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM cache_entries WHERE entry_id = ?", (entry_id,)
            ).fetchone()
            if row is None:
                return None

            emb_row = self._conn.execute(
                "SELECT vector FROM embeddings WHERE entry_id = ?", (entry_id,)
            ).fetchone()
            embedding = (
                np.frombuffer(emb_row[0], dtype=np.float32).tolist()
                if emb_row
                else []
            )

            return self._row_to_entry(row, embedding)

    def get_all(self, limit: int | None = None) -> list[CacheEntry]:
        import numpy as np

        with self._lock:
            query = "SELECT * FROM cache_entries WHERE status = 'active'"
            if limit is not None:
                query += f" LIMIT {limit}"
            rows = self._conn.execute(query).fetchall()

            results = []
            for row in rows:
                entry_id = row[0]
                emb_row = self._conn.execute(
                    "SELECT vector FROM embeddings WHERE entry_id = ?", (entry_id,)
                ).fetchone()
                embedding = (
                    np.frombuffer(emb_row[0], dtype=np.float32).tolist()
                    if emb_row
                    else []
                )
                results.append(self._row_to_entry(row, embedding))
            return results

    def delete(self, entry_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM cache_entries WHERE entry_id = ?", (entry_id,)
            )
            self._conn.execute(
                "DELETE FROM embeddings WHERE entry_id = ?", (entry_id,)
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def delete_batch(self, entry_ids: list[str]) -> int:
        with self._lock:
            placeholders = ",".join("?" for _ in entry_ids)
            cursor = self._conn.execute(
                f"DELETE FROM cache_entries WHERE entry_id IN ({placeholders})",
                entry_ids,
            )
            self._conn.execute(
                f"DELETE FROM embeddings WHERE entry_id IN ({placeholders})",
                entry_ids,
            )
            self._conn.commit()
            return cursor.rowcount

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM cache_entries WHERE status = 'active'"
            ).fetchone()
            return row[0] if row else 0

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM cache_entries")
            self._conn.execute("DELETE FROM embeddings")
            self._conn.commit()

    def get_embeddings_batch(self, entry_ids: list[str]) -> dict[str, list[float]]:
        import numpy as np

        with self._lock:
            result = {}
            for eid in entry_ids:
                row = self._conn.execute(
                    "SELECT vector FROM embeddings WHERE entry_id = ?", (eid,)
                ).fetchone()
                if row:
                    result[eid] = np.frombuffer(row[0], dtype=np.float32).tolist()
            return result

    def update_access(self, entry_id: str) -> None:
        with self._lock:
            self._conn.execute(
                ,
                (time.time(), entry_id),
            )
            self._conn.commit()

    def update_quality(self, entry_id: str, score: float) -> None:
        with self._lock:

            self._conn.execute(
                ,
                (score, entry_id),
            )
            self._conn.commit()

    def get_eviction_candidates_lru(self, n: int) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT entry_id FROM cache_entries ORDER BY last_accessed ASC LIMIT ?",
                (n,),
            ).fetchall()
            return [r[0] for r in rows]

    def get_eviction_candidates_lfu(self, n: int) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT entry_id FROM cache_entries ORDER BY access_count ASC LIMIT ?",
                (n,),
            ).fetchall()
            return [r[0] for r in rows]

    def get_eviction_candidates_score(self, n: int) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT entry_id FROM cache_entries ORDER BY quality_score ASC LIMIT ?",
                (n,),
            ).fetchall()
            return [r[0] for r in rows]

    def _row_to_entry(self, row: tuple, embedding: list[float]) -> CacheEntry:
        import orjson

        return CacheEntry(
            entry_id=row[0],
            query=row[1],
            embedding=embedding,
            response=row[2],
            response_metadata=orjson.loads(row[3]) if row[3] else {},
            embedding_model=row[4],
            created_at=row[5],
            last_accessed=row[6],
            access_count=row[7],
            quality_score=row[8],
            status=EntryStatus(row[9]),
            tags=set(orjson.loads(row[10])) if row[10] else set(),
            metadata=orjson.loads(row[11]) if row[11] else {},
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

def create_storage(config: StorageConfig) -> CacheStorage:

    if config.backend == StorageBackend.IN_MEMORY:
        return InMemoryStorage(config)
    elif config.backend in (StorageBackend.SQLITE, StorageBackend.SQLITE_VEC):
        return SQLiteStorage(config)
    else:
        raise ValueError(f"Unknown storage backend: {config.backend}")
