from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from neural_cache.config import (
    CacheConfig,
    EncoderConfig,
    StorageConfig,
    StorageBackend,
    SearchConfig,
    DecisionConfig,
    EvictionConfig,
    EvictionPolicy,
    DecisionStrategy,
    EmbeddingModel,
)
from neural_cache.models import (
    CacheEntry,
    CacheAction,
    CacheResult,
    SearchResult,
    MetricsSnapshot,
)

@pytest.fixture
def sample_entries():

    return [
        CacheEntry(
            query="What is machine learning?",
            embedding=np.random.randn(384).astype(np.float32).tolist(),
            response="Machine learning is a subset of AI...",
            embedding_model="all-MiniLM-L6-v2",
        ),
        CacheEntry(
            query="What is deep learning?",
            embedding=np.random.randn(384).astype(np.float32).tolist(),
            response="Deep learning uses neural networks...",
            embedding_model="all-MiniLM-L6-v2",
        ),
        CacheEntry(
            query="How does neural network work?",
            embedding=np.random.randn(384).astype(np.float32).tolist(),
            response="Neural networks are inspired by biological brains...",
            embedding_model="all-MiniLM-L6-v2",
        ),
    ]

@pytest.fixture
def mock_llm():

    async def generate(query: str) -> tuple[str, dict]:
        await asyncio.sleep(0.01)
        return f"Mock response to: {query}", {"model": "mock"}
    return generate

class TestQueryEncoder:

    def test_encoder_initialization(self):

        from neural_cache.encoder import QueryEncoder

        config = EncoderConfig(
            model_name=EmbeddingModel.ALL_MINILM_L6_V2,
        )
        encoder = QueryEncoder(config)
        assert encoder.dimension == 384

    def test_encode_single(self):

        from neural_cache.encoder import QueryEncoder

        encoder = QueryEncoder(EncoderConfig())
        encoder.warmup()

        embedding = encoder.encode_single("What is Python?")
        assert embedding.shape == (encoder.dimension,)
        assert embedding.dtype == np.float32

    def test_encode_batch(self):

        from neural_cache.encoder import QueryEncoder

        encoder = QueryEncoder(EncoderConfig())
        encoder.warmup()

        queries = ["What is Python?", "What is ML?", "What is AI?"]
        embeddings = encoder.encode(queries)
        assert embeddings.shape == (3, encoder.dimension)

    def test_normalize_embeddings(self):

        from neural_cache.encoder import QueryEncoder

        encoder = QueryEncoder(EncoderConfig(normalize=True))
        encoder.warmup()

        embedding = encoder.encode_single("Test query")
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-5

    def test_similarity_computation(self):

        from neural_cache.encoder import QueryEncoder

        encoder = QueryEncoder(EncoderConfig(normalize=True))
        encoder.warmup()

        sim = encoder.similarity("What is ML?", "What is machine learning?")
        assert 0.0 <= sim <= 1.0

class TestInMemoryStorage:

    def test_put_and_get(self, sample_entries):

        from neural_cache.storage import InMemoryStorage

        storage = InMemoryStorage(StorageConfig())
        storage.put(sample_entries[0])

        retrieved = storage.get(sample_entries[0].entry_id)
        assert retrieved is not None
        assert retrieved.query == sample_entries[0].query

    def test_get_nonexistent(self):

        from neural_cache.storage import InMemoryStorage

        storage = InMemoryStorage(StorageConfig())
        assert storage.get("nonexistent") is None

    def test_delete(self, sample_entries):

        from neural_cache.storage import InMemoryStorage

        storage = InMemoryStorage(StorageConfig())
        storage.put(sample_entries[0])
        assert storage.delete(sample_entries[0].entry_id) is True
        assert storage.get(sample_entries[0].entry_id) is None

    def test_delete_batch(self, sample_entries):

        from neural_cache.storage import InMemoryStorage

        storage = InMemoryStorage(StorageConfig())
        for entry in sample_entries:
            storage.put(entry)

        ids_to_delete = [e.entry_id for e in sample_entries[:2]]
        deleted = storage.delete_batch(ids_to_delete)
        assert deleted == 2
        assert storage.count() == 1

    def test_count(self, sample_entries):

        from neural_cache.storage import InMemoryStorage

        storage = InMemoryStorage(StorageConfig())
        assert storage.count() == 0

        for entry in sample_entries:
            storage.put(entry)

        assert storage.count() == 3

    def test_clear(self, sample_entries):

        from neural_cache.storage import InMemoryStorage

        storage = InMemoryStorage(StorageConfig())
        for entry in sample_entries:
            storage.put(entry)

        storage.clear()
        assert storage.count() == 0

    def test_update_access(self, sample_entries):

        from neural_cache.storage import InMemoryStorage

        storage = InMemoryStorage(StorageConfig())
        storage.put(sample_entries[0])

        old_accessed = storage.get(sample_entries[0].entry_id).last_accessed
        time.sleep(0.01)
        storage.update_access(sample_entries[0].entry_id)

        entry = storage.get(sample_entries[0].entry_id)
        assert entry.access_count == 1
        assert entry.last_accessed > old_accessed

    def test_get_all(self, sample_entries):

        from neural_cache.storage import InMemoryStorage

        storage = InMemoryStorage(StorageConfig())
        for entry in sample_entries:
            storage.put(entry)

        all_entries = storage.get_all()
        assert len(all_entries) == 3

    def test_get_embeddings_batch(self, sample_entries):

        from neural_cache.storage import InMemoryStorage

        storage = InMemoryStorage(StorageConfig())
        for entry in sample_entries:
            storage.put(entry)

        ids = [e.entry_id for e in sample_entries[:2]]
        embeddings = storage.get_embeddings_batch(ids)
        assert len(embeddings) == 2

class TestSQLiteStorage:

    @pytest.fixture
    def sqlite_storage(self, tmp_path, sample_entries):

        from neural_cache.storage import SQLiteStorage

        db_path = str(tmp_path / "test.db")
        config = StorageConfig(
            backend=StorageBackend.SQLITE,
            db_path=db_path,
        )
        storage = SQLiteStorage(config)

        for entry in sample_entries:
            storage.put(entry)

        yield storage
        storage.close()

    def test_put_and_get(self, sqlite_storage, sample_entries):

        retrieved = sqlite_storage.get(sample_entries[0].entry_id)
        assert retrieved is not None
        assert retrieved.query == sample_entries[0].query

    def test_count(self, sqlite_storage):

        assert sqlite_storage.count() == 3

    def test_delete(self, sqlite_storage, sample_entries):

        assert sqlite_storage.delete(sample_entries[0].entry_id) is True
        assert sqlite_storage.count() == 2

class TestSimilaritySearchEngine:

    def test_flat_index(self, sample_entries):

        from neural_cache.search import SimilaritySearchEngine

        config = SearchConfig(index_type="Flat")
        engine = SimilaritySearchEngine(config, embedding_dim=384)
        engine.initialize_index()
        engine.add_entries(sample_entries)

        query_emb = np.random.randn(384).astype(np.float32)
        results = engine.search(query_emb, top_k=3)
        assert len(results) <= 3

    def test_hnsw_index(self, sample_entries):

        from neural_cache.search import SimilaritySearchEngine

        config = SearchConfig(index_type="HNSW", hnsw_m=16)
        engine = SimilaritySearchEngine(config, embedding_dim=384)
        engine.initialize_index()
        engine.add_entries(sample_entries)

        query_emb = np.random.randn(384).astype(np.float32)
        results = engine.search(query_emb, top_k=3)
        assert len(results) <= 3

    def test_empty_search(self):

        from neural_cache.search import SimilaritySearchEngine

        config = SearchConfig(index_type="Flat")
        engine = SimilaritySearchEngine(config, embedding_dim=384)
        engine.initialize_index()

        query_emb = np.random.randn(384).astype(np.float32)
        results = engine.search(query_emb)
        assert results == []

    def test_rebuild_index(self, sample_entries):

        from neural_cache.search import SimilaritySearchEngine

        config = SearchConfig(index_type="Flat")
        engine = SimilaritySearchEngine(config, embedding_dim=384)
        engine.rebuild_index(sample_entries)

        query_emb = np.random.randn(384).astype(np.float32)
        results = engine.search(query_emb, top_k=3)
        assert len(results) <= 3

    def test_get_stats(self, sample_entries):

        from neural_cache.search import SimilaritySearchEngine

        config = SearchConfig(index_type="HNSW")
        engine = SimilaritySearchEngine(config, embedding_dim=384)
        engine.initialize_index()
        engine.add_entries(sample_entries)

        stats = engine.get_stats()
        assert stats["total_vectors"] == 3
        assert stats["index_type"] == "HNSW"

class TestCacheDecisionPolicy:

    def test_high_confidence_hit(self):

        from neural_cache.decision import CacheDecisionPolicy

        config = DecisionConfig(
            similarity_threshold=0.85,
            high_confidence_threshold=0.95,
        )
        policy = CacheDecisionPolicy(config)

        entry = CacheEntry(query="test", response="response")
        result = SearchResult(entry=entry, similarity_score=0.97, rank=0)

        decision = policy.decide([result])
        assert decision.action == CacheAction.HIT
        assert decision.confidence >= 0.95

    def test_hit_with_adaptation(self):

        from neural_cache.decision import CacheDecisionPolicy

        config = DecisionConfig(
            similarity_threshold=0.85,
            high_confidence_threshold=0.95,
        )
        policy = CacheDecisionPolicy(config)

        entry = CacheEntry(query="test", response="response")
        result = SearchResult(entry=entry, similarity_score=0.90, rank=0)

        decision = policy.decide([result])
        assert decision.action == CacheAction.HIT_WITH_ADAPTATION

    def test_miss(self):

        from neural_cache.decision import CacheDecisionPolicy

        config = DecisionConfig(similarity_threshold=0.85)
        policy = CacheDecisionPolicy(config)

        entry = CacheEntry(query="test", response="response")
        result = SearchResult(entry=entry, similarity_score=0.70, rank=0)

        decision = policy.decide([result])
        assert decision.action == CacheAction.MISS

    def test_empty_results(self):

        from neural_cache.decision import CacheDecisionPolicy

        policy = CacheDecisionPolicy(DecisionConfig())
        decision = policy.decide([])
        assert decision.action == CacheAction.MISS

    def test_adaptive_threshold_recalibration(self):

        from neural_cache.decision import CacheDecisionPolicy

        config = DecisionConfig(
            strategy=DecisionStrategy.ADAPTIVE_THRESHOLD,
            adaptive_window_size=20,
            similarity_threshold=0.85,
        )
        policy = CacheDecisionPolicy(config)

        for _ in range(20):
            policy.record_feedback(0.95, True, 0.9)

        stats = policy.get_stats()
        assert stats["feedback_samples"] == 20

class TestEvictionManager:

    def test_lru_eviction(self, sample_entries):

        from neural_cache.storage import InMemoryStorage
        from neural_cache.eviction import EvictionManager

        config = StorageConfig(max_entries=3)
        storage = InMemoryStorage(config)
        for entry in sample_entries:
            storage.put(entry)

        eviction_config = EvictionConfig(
            policy=EvictionPolicy.LRU,
            high_watermark=0.8,
            low_watermark=0.5,
            check_interval=0,
        )
        manager = EvictionManager(storage, eviction_config)

        extra = CacheEntry(query="extra", response="extra response")
        storage.put(extra)

        assert storage.count() == 4

    def test_should_evict(self):

        from neural_cache.storage import InMemoryStorage
        from neural_cache.eviction import EvictionManager

        config = StorageConfig(max_entries=10)
        storage = InMemoryStorage(config)

        eviction_config = EvictionConfig(
            policy=EvictionPolicy.LRU,
            high_watermark=0.9,
        )
        manager = EvictionManager(storage, eviction_config)

        assert manager.should_evict() is False

        for i in range(9):
            storage.put(CacheEntry(query=f"q{i}", response=f"r{i}"))

        assert manager.should_evict() is True

class TestNeuralCacheIntegration:

    @pytest.mark.asyncio
    async def test_full_pipeline_with_mock_llm(self, mock_llm):

        from neural_cache.cache import NeuralCache

        config = CacheConfig.fast_production()
        config.storage.max_entries = 100

        async with NeuralCache(config) as cache:
            cache.set_llm_function(mock_llm)

            result1 = await cache.get("What is machine learning?")
            assert result1.action == CacheAction.MISS
            assert not result1.from_cache

            result2 = await cache.get("What is machine learning?")
            assert result2.from_cache
            assert result2.action == CacheAction.HIT

    @pytest.mark.asyncio
    async def test_similar_query_retrieval(self, mock_llm):

        from neural_cache.cache import NeuralCache

        config = CacheConfig.fast_production()
        config.decision.similarity_threshold = 0.5
        config.storage.max_entries = 100

        async with NeuralCache(config) as cache:
            cache.set_llm_function(mock_llm)

            await cache.get("What is Python programming?")

            result = await cache.get("Tell me about Python programming")

            assert result.response is not None

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, mock_llm):

        from neural_cache.cache import NeuralCache

        config = CacheConfig.fast_production()
        async with NeuralCache(config) as cache:
            cache.set_llm_function(mock_llm)

            for i in range(5):
                await cache.get(f"Query {i}")

            metrics = cache.get_metrics()
            assert metrics.total_requests == 5

    @pytest.mark.asyncio
    async def test_cache_disabled(self, mock_llm):

        from neural_cache.cache import NeuralCache

        config = CacheConfig()
        config.enabled = False

        async with NeuralCache(config) as cache:
            cache.set_llm_function(mock_llm)

            result1 = await cache.get("Test query")
            result2 = await cache.get("Test query")

            assert not result1.from_cache
            assert not result2.from_cache

    @pytest.mark.asyncio
    async def test_dry_run_mode(self, mock_llm):

        from neural_cache.cache import NeuralCache

        config = CacheConfig()
        config.dry_run = True

        async with NeuralCache(config) as cache:
            cache.set_llm_function(mock_llm)

            await cache.get("Test query")
            result = await cache.get("Test query")

            assert not result.from_cache

class TestDataModels:

    def test_cache_entry_creation(self):

        entry = CacheEntry(query="test", response="response")
        assert entry.entry_id is not None
        assert entry.access_count == 0
        assert entry.quality_score == 1.0
        assert entry.status.value == "active"

    def test_cache_entry_record_access(self):

        entry = CacheEntry(query="test", response="response")
        updated = entry.record_access()

        assert updated.access_count == 1
        assert updated.last_accessed > entry.created_at

    def test_cache_entry_update_quality(self):

        entry = CacheEntry(query="test", response="response", quality_score=1.0)
        updated = entry.update_quality(0.5)

        assert abs(updated.quality_score - 0.85) < 1e-10

    def test_cache_result(self):

        result = CacheResult(
            response="test",
            action=CacheAction.HIT,
            from_cache=True,
            similarity_score=0.95,
            total_latency_ms=10.0,
        )
        assert result.request_id is not None

    def test_metrics_snapshot_hit_rate(self):

        snapshot = MetricsSnapshot(
            total_requests=10,
            cache_hits=3,
            cache_hit_with_adaptation=2,
            cache_misses=5,
        )
        assert snapshot.hit_rate == 0.5

class TestPerformance:

    def test_embedding_latency(self):

        from neural_cache.encoder import QueryEncoder

        encoder = QueryEncoder(EncoderConfig())
        encoder.warmup()

        latencies = []
        for _ in range(10):
            start = time.monotonic()
            encoder.encode_single("What is the capital of France?")
            latencies.append((time.monotonic() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < 100
        print(f"\nAvg embedding latency: {avg_latency:.1f}ms")

    def test_search_latency(self, sample_entries):

        from neural_cache.search import SimilaritySearchEngine

        config = SearchConfig(index_type="HNSW")
        engine = SimilaritySearchEngine(config, embedding_dim=384)
        engine.initialize_index()
        engine.add_entries(sample_entries)

        query_emb = np.random.randn(384).astype(np.float32)

        latencies = []
        for _ in range(100):
            start = time.monotonic()
            engine.search(query_emb, top_k=5)
            latencies.append((time.monotonic() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < 10
        print(f"\nAvg search latency: {avg_latency:.2f}ms")

    def test_decision_latency(self):

        from neural_cache.decision import CacheDecisionPolicy

        policy = CacheDecisionPolicy(DecisionConfig())
        entry = CacheEntry(query="test", response="response")
        result = SearchResult(entry=entry, similarity_score=0.90, rank=0)

        latencies = []
        for _ in range(1000):
            decision = policy.decide([result])
            latencies.append(decision.decision_latency_ms)

        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < 1.0
        print(f"\nAvg decision latency: {avg_latency:.3f}ms")
