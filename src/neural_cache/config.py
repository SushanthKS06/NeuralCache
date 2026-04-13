from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

class EmbeddingModel(str, Enum):
    ALL_MINILM_L6_V2 = "sentence-transformers/all-MiniLM-L6-v2"

    ALL_MPNET_BASE_V2 = "sentence-transformers/all-mpnet-base-v2"

    PARAPHRASE_MINILM_L3 = "sentence-transformers/paraphrase-MiniLM-L3-v2"

    GTE_LARGE = "thenlper/gte-large"

class StorageBackend(str, Enum):
    IN_MEMORY = "in_memory"
    SQLITE = "sqlite"
    SQLITE_VEC = "sqlite_vec"

class EvictionPolicy(str, Enum):
    LRU = "lru"
    LFU = "lfu"
    SCORE_BASED = "score_based"
    TTL = "ttl"

class DecisionStrategy(str, Enum):
    FIXED_THRESHOLD = "fixed_threshold"
    ADAPTIVE_THRESHOLD = "adaptive_threshold"
    LEARNED_SCORING = "learned_scoring"

class ResponseAdaptationMode(str, Enum):
    NONE = "none"
    TEMPLATE_FILL = "template_fill"
    LLM_REFINE = "llm_refine"
    HYBRID = "hybrid"

@dataclass(frozen=True)
class EncoderConfig:
    model_name: EmbeddingModel = EmbeddingModel.ALL_MINILM_L6_V2
    embedding_dim: int = 384
    normalize: bool = True
    device: str = "cpu"
    batch_size: int = 32
    max_seq_length: int = 256

    pooling_strategy: str = "mean"

@dataclass(frozen=True)
class StorageConfig:
    backend: StorageBackend = StorageBackend.IN_MEMORY

    db_path: str = str(Path.home() / ".neural_cache" / "cache.db")

    max_entries: int = 100_000

    wal_mode: bool = True

    compress_vectors: bool = False

@dataclass(frozen=True)
class SearchConfig:
    index_type: str = "HNSW"

    nlist: int = 100

    hnsw_m: int = 32

    hnsw_ef_construction: int = 200

    hnsw_ef_search: int = 64

    top_k: int = 5

    use_gpu: bool = False

    nprobe: int = 10

@dataclass(frozen=True)
class DecisionConfig:
    strategy: DecisionStrategy = DecisionStrategy.FIXED_THRESHOLD

    similarity_threshold: float = 0.85

    high_confidence_threshold: float = 0.95

    adaptive_window_size: int = 100

    exploration_rate: float = 0.1

    min_feedback_samples: int = 500

    enable_reranking: bool = True

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

@dataclass(frozen=True)
class AdaptationConfig:
    mode: ResponseAdaptationMode = ResponseAdaptationMode.NONE

    refinement_prompt_template: str = (
        "Given the following cached response to a similar question,\n"
        "adapt it to answer this new query accurately.\n\n"
        "Cached Query: {cached_query}\n"
        "Cached Response: {cached_response}\n"
        "New Query: {new_query}\n\n"
        "Adapted Response:"
    )

    max_adaptation_tokens: int = 512

    passthrough_threshold: float = 0.98

@dataclass(frozen=True)
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout_seconds: float = 30.0

    max_retries: int = 3
    retry_backoff_factor: float = 2.0

@dataclass(frozen=True)
class EvictionConfig:
    policy: EvictionPolicy = EvictionPolicy.LRU

    eviction_batch_size: int = 100

    ttl_seconds: int = 7 * 24 * 60 * 60

    min_quality_score: float = 0.5

    check_interval: int = 300

    high_watermark: float = 0.9

    low_watermark: float = 0.7

@dataclass(frozen=True)
class MetricsConfig:
    enable_prometheus: bool = True
    prometheus_port: int = 9090

    log_level: str = "INFO"
    log_format: str = "json"

    sliding_window_seconds: int = 3600

    persist_metrics: bool = True
    metrics_path: str = str(Path.home() / ".neural_cache" / "metrics")

@dataclass(frozen=True)
class CacheConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    eviction: EvictionConfig = field(default_factory=EvictionConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    enabled: bool = True

    dry_run: bool = False

    environment: str = "development"

    @classmethod
    def fast_production(cls) -> CacheConfig:
        return cls(
            encoder=EncoderConfig(
                model_name=EmbeddingModel.ALL_MINILM_L6_V2,
                normalize=True,
            ),
            search=SearchConfig(
                index_type="HNSW",
                hnsw_m=16,
                hnsw_ef_search=32,
                top_k=3,
            ),
            decision=DecisionConfig(
                similarity_threshold=0.87,
                enable_reranking=False,
            ),
            storage=StorageConfig(
                backend=StorageBackend.IN_MEMORY,
                max_entries=50_000,
            ),
            environment="production",
        )

    @classmethod
    def high_accuracy(cls) -> CacheConfig:
        return cls(
            encoder=EncoderConfig(
                model_name=EmbeddingModel.ALL_MPNET_BASE_V2,
                embedding_dim=768,
            ),
            decision=DecisionConfig(
                similarity_threshold=0.82,
                high_confidence_threshold=0.93,
                enable_reranking=True,
            ),
            adaptation=AdaptationConfig(
                mode=ResponseAdaptationMode.LLM_REFINE,
            ),
            search=SearchConfig(
                top_k=5,
                hnsw_ef_search=128,
            ),
        )
