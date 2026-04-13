# Neural Cache System for Memory-Augmented AI

> A production-grade and research-grade system for reducing LLM latency through semantic caching with vector similarity retrieval.

## 🎯 Overview

Neural Cache intercepts queries to LLM-based systems, encodes them into embeddings, retrieves semantically similar past queries, and either reuses cached responses or adapts them — all while maintaining output quality. The system targets **<50ms retrieval latency** and **>40% cache hit rates** on repetitive query workloads.

### Key Results (Expected)

| Metric | Target | Expected |
|---|---|---|
| Cache Hit Rate | >30% | 40-60% |
| Avg Latency Reduction | >50% | 60-80% |
| P95 Latency (cached) | <20ms | 5-15ms |
| Quality Degradation | <5% | 1-3% |

---

## 🏗️ System Architecture

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        NEURAL CACHE SYSTEM                       │
│                                                                   │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────┐    │
│  │  QUERY   │───▶│   QUERY      │───▶│   SIMILARITY         │    │
│  │  INPUT   │    │   ENCODER    │    │   SEARCH ENGINE      │    │
│  │          │    │ (MiniLM/MPNet│    │   (FAISS HNSW/IVF)   │    │
│  └──────────┘    │  Embedding)  │    │                      │    │
│                  └──────────────┘    └──────────┬───────────┘    │
│                                                  │                │
│                                                  ▼                │
│                                        ┌──────────────────────┐   │
│                   ┌───────────────────▶│   CACHE DECISION     │   │
│                   │                    │   POLICY             │   │
│                   │                    │                      │   │
│                   │                    │  Threshold / Adaptive│   │
│                   │                    │  / Learned Scoring   │   │
│                   │                    └──┬────────┬──────────┘   │
│                   │                       │        │              │
│                   │              ┌────────┘        └────────┐     │
│                   │              ▼                          ▼     │
│                   │   ┌──────────────────┐    ┌───────────────┐  │
│                   │   │   HIT            │    │   MISS         │  │
│                   │   │   (return cached)│    │   (call LLM)   │  │
│                   │   │                  │    │                │  │
│                   │   │  ┌────────────┐  │    │  ┌──────────┐ │  │
│                   │   │  │ Adaptation │  │    │  │  LLM     │ │  │
│                   │   │  │   Layer    │  │    │  │ Fallback │ │  │
│                   │   │  └────────────┘  │    │  └────┬─────┘ │  │
│                   │   └──────────────────┘    │       │       │  │
│                   │                           │       ▼       │  │
│                   │                           │  ┌──────────┐ │  │
│                   │                           │  │  STORE   │ │  │
│                   │                           │  │  RESULT  │ │  │
│                   │                           │  └──────────┘ │  │
│                   │                           └───────┬───────┘  │
│                   │                                   │          │
│                   └───────────────────┬───────────────┘          │
│                                       ▼                           │
│                   ┌──────────────────────────────────────┐        │
│                   │   CACHE STORAGE (In-Memory / SQLite) │        │
│                   │   + EVICTION MANAGER (LRU/LFU/TTL)   │        │
│                   └──────────────────────────────────────┘        │
│                                                                   │
│                   ┌──────────────────────────────────────┐        │
│                   │   METRICS & LOGGING                  │        │
│                   │   (Prometheus + Structured Logs)     │        │
│                   └──────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

### Module Breakdown

| Module | File | Purpose |
|---|---|---|
| Query Encoder | `encoder.py` | Converts text → normalized embeddings |
| Cache Storage | `storage.py` | In-memory (OrderedDict) or SQLite backend |
| Search Engine | `search.py` | FAISS index (Flat/IVF/HNSW/LSH) + reranking |
| Decision Policy | `decision.py` | Fixed/Adaptive/Learned threshold logic |
| Adaptation Layer | `adaptation.py` | Template-fill or LLM-refine cached responses |
| LLM Fallback | `llm_client.py` | OpenAI/Anthropic/Local LLM clients |
| Eviction Manager | `eviction.py` | LRU/LFU/Score-based/TTL eviction |
| Metrics | `metrics.py` | Prometheus metrics + sliding window stats |
| Hybrid Retrieval | `hybrid_retrieval.py` | BM25 + semantic fusion (RRF) |
| Multi-Level Cache | `multilevel_cache.py` | L1 (hot) / L2 (cold) hierarchy |
| Experiments | `experiments.py` | Systematic evaluation framework |

---

## 📁 Project Structure

```
NeuralCache/
├── pyproject.toml                     # Dependencies and build config
├── README.md                          # This file
│
├── src/neural_cache/
│   ├── __init__.py                    # Package entry point
│   ├── __main__.py                    # python -m neural_cache
│   ├── config.py                      # All configuration (dataclasses)
│   ├── models.py                      # Core data models (CacheEntry, etc.)
│   │
│   ├── encoder.py                     # Query Encoder Module
│   ├── storage.py                     # Cache Storage System
│   ├── search.py                      # Similarity Search Engine
│   ├── decision.py                    # Cache Decision Policy
│   ├── adaptation.py                  # Response Adaptation Layer
│   ├── llm_client.py                  # LLM Fallback System
│   ├── eviction.py                    # Cache Update Policy
│   ├── metrics.py                     # Logging & Metrics System
│   │
│   ├── hybrid_retrieval.py            # Advanced: Hybrid BM25+Semantic
│   ├── multilevel_cache.py            # Advanced: L1/L2 Cache Hierarchy
│   ├── experiments.py                 # Experiment Framework
│   ├── cache.py                       # Main NeuralCache orchestrator
│   ├── core.py                        # Re-export for clean API
│   └── cli.py                         # CLI interface
│
└── tests/
    ├── conftest.py                    # Pytest fixtures
    └── test_neural_cache.py           # Full test suite
```

---

## ⚙️ Installation

```bash
# Clone or navigate to the project
cd NeuralCache

# Install with pip
pip install -e .

# Optional: GPU support
pip install -e ".[gpu]"

# Optional: Development tools
pip install -e ".[dev]"
```

### Dependencies

| Package | Purpose |
|---|---|
| `sentence-transformers` | Embedding models + cross-encoders |
| `faiss-cpu` | Vector similarity search |
| `numpy` | Numerical computations |
| `scikit-learn` | Additional ML utilities |
| `torch` | Deep learning backend for encoders |
| `orjson` | Fast JSON serialization |
| `structlog` | Structured logging |
| `prometheus-client` | Metrics export |
| `openai` | OpenAI API client |
| `tenacity` | Retry logic |
| `rich` + `click` | CLI interface |
| `pytest` | Testing framework |

---

## 🚀 Quick Start

### Basic Usage

```python
import asyncio
from neural_cache import NeuralCache, CacheConfig

async def main():
    # Use pre-tuned production config
    config = CacheConfig.fast_production()

    async with NeuralCache(config) as cache:
        # Set your LLM function
        async def my_llm(query: str) -> tuple[str, dict]:
            # Call your LLM here (OpenAI, Anthropic, local, etc.)
            response = await openai_call(query)
            return response, {"model": "gpt-4o-mini"}

        cache.set_llm_function(my_llm)

        # First call: cache miss → LLM
        result1 = await cache.get("What is machine learning?")
        print(f"Action: {result1.action}")  # MISS
        print(f"From cache: {result1.from_cache}")  # False

        # Second call: cache hit → instant
        result2 = await cache.get("What is machine learning?")
        print(f"Action: {result2.action}")  # HIT
        print(f"From cache: {result2.from_cache}")  # True
        print(f"Latency: {result2.total_latency_ms:.1f}ms")  # <10ms

        # Similar query: may hit with adaptation
        result3 = await cache.get("Explain machine learning")
        print(f"Action: {result3.action}")  # HIT or HIT_WITH_ADAPTATION

asyncio.run(main())
```

### Using Built-in LLM Clients

```python
from neural_cache import NeuralCache, CacheConfig
from neural_cache.llm_client import create_llm_client

config = CacheConfig.fast_production()
config.llm.provider = "openai"
config.llm.api_key = "sk-..."
config.llm.model = "gpt-4o-mini"

async with NeuralCache(config) as cache:
    # LLM client is used automatically on cache miss
    cache.set_llm_client(create_llm_client(config.llm))
    result = await cache.get("What is quantum computing?")
```

### Pre-Tuned Configurations

```python
# Fast production (MiniLM, no reranking, conservative threshold)
config = CacheConfig.fast_production()

# High accuracy (MPNet, reranking, LLM refinement)
config = CacheConfig.high_accuracy()

# Custom configuration
config = CacheConfig(
    encoder=EncoderConfig(
        model_name=EmbeddingModel.ALL_MPNET_BASE_V2,
        normalize=True,
    ),
    search=SearchConfig(
        index_type="HNSW",
        hnsw_m=32,
        top_k=5,
    ),
    decision=DecisionConfig(
        strategy=DecisionStrategy.ADAPTIVE_THRESHOLD,
        similarity_threshold=0.82,
    ),
    storage=StorageConfig(
        backend=StorageBackend.SQLITE,
        db_path="./my_cache.db",
        max_entries=50000,
    ),
)
```

---

## 🔧 Configuration Reference

### Embedding Models

| Model | Dimension | Speed | Quality | Use Case |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | ⚡⚡⚡ | Good | Production default |
| `paraphrase-MiniLM-L3-v2` | 384 | ⚡⚡⚡⚡ | Good | Short queries |
| `all-mpnet-base-v2` | 768 | ⚡⚡ | Excellent | Accuracy-critical |
| `gte-large` | 1024 | ⚡ | Best | Research |

### FAISS Index Types

| Index | Search Time | Accuracy | Memory | Best For |
|---|---|---|---|---|
| `Flat` | O(n) | Exact | Low | <100K entries |
| `IVF` | O(n/√clusters) | Approximate | Medium | 100K-1M entries |
| `HNSW` | O(log n) | Approximate | Medium-High | Production default |
| `LSH` | O(1) | Low | Low | Ultra-low latency |

### Decision Strategies

| Strategy | Description | When to Use |
|---|---|---|
| `FIXED_THRESHOLD` | Simple cosine similarity cutoff | Most workloads |
| `ADAPTIVE_THRESHOLD` | Learns threshold from feedback | Dynamic workloads |
| `LEARNED_SCORING` | Trains model to predict reliability | High-volume systems |

### Eviction Policies

| Policy | Strategy | Best For |
|---|---|---|
| `LRU` | Evict least recently used | General purpose |
| `LFU` | Evict least frequently used | Repetitive workloads |
| `SCORE_BASED` | Evict lowest quality | Quality-sensitive |
| `TTL` | Evict after time-to-live | Time-sensitive data |

---

## 🔬 Experiment Framework

### Running Experiments

```bash
# Run all experiments with mock LLM
neural-cache experiment --n-queries 50 --output-dir ./results

# Or programmatically
python -c "
from neural_cache.experiments import ExperimentRunner
import asyncio

async def mock_llm(query):
    return f'Response to: {query}', {}

runner = ExperimentRunner('./results')
queries = runner._generate_training_queries(100)
asyncio.run(runner.run_all(mock_llm, queries))
"
```

### Experiments Included

| # | Experiment | What It Measures |
|---|---|---|
| 1 | **Latency Comparison** | Cache vs no-cache latency reduction |
| 2 | **Threshold Sensitivity** | Optimal similarity threshold |
| 3 | **Cache Size Impact** | Memory vs hit rate tradeoff |
| 4 | **Eviction Policy Comparison** | LRU vs LFU vs Score-based |
| 5 | **Embedding Model Comparison** | MiniLM vs Paraphrase vs MPNet |

### Expected Outcomes

```
Experiment 1: Latency Comparison
  Avg latency without cache: ~500ms (LLM call)
  Avg latency with cache: ~50ms (embedding + search)
  Cache hit rate: 40-60%
  Latency reduction: 80-90%

Experiment 2: Threshold Sensitivity
  Optimal threshold: 0.82-0.90
  Too low → poor quality hits
  Too high → too many misses

Experiment 3: Cache Size Impact
  Hit rate plateaus at ~5K-10K entries
  Diminishing returns beyond that

Experiment 4: Eviction Policy
  LRU: Best for general workloads
  LFU: Best for repetitive queries
  Score-based: Best for quality-sensitive
```

---

## 📊 Metrics & Monitoring

### Prometheus Metrics

```
# Total requests by action
neural_cache_requests_total{action="hit"}
neural_cache_requests_total{action="miss"}
neural_cache_requests_total{action="hit_with_adaptation"}

# Latency histogram
neural_cache_latency_ms_bucket

# Cache hit rate gauge
neural_cache_hit_rate

# Cache size gauge
neural_cache_size
```

### Getting Metrics Snapshot

```python
metrics = cache.get_metrics()
print(f"Hit rate: {metrics.hit_rate:.2%}")
print(f"Avg latency: {metrics.avg_latency_ms:.1f}ms")
print(f"P95 latency: {metrics.p95_latency_ms:.1f}ms")
print(f"Cache size: {metrics.cache_size}")
```

### Structured Logging

All operations emit structured JSON logs:

```json
{
  "event": "cache_hit",
  "query": "What is ML?",
  "similarity_score": 0.95,
  "latency_ms": 5.2,
  "request_id": "uuid",
  "timestamp": "2026-04-13T10:30:00Z"
}
```

---

## 🌐 CLI Reference

```bash
# Initialize cache
neural-cache init --config fast

# View statistics
neural-cache stats

# Clear cache
neural-cache clear

# Ask a question
neural-cache ask "What is machine learning?"

# Run experiments
neural-cache experiment --n-queries 100

# Start HTTP server
neural-cache serve --port 8000
```

---

## 🔬 Research Paper Design

### Title
*"NeuralCache: Semantic Caching for Latency Reduction in Large Language Model Systems"*

### Abstract
We present NeuralCache, a production-grade semantic caching system that reduces LLM inference latency by 60-80% while maintaining output quality within 1-3% of fresh generation. The system combines dense retrieval (FAISS HNSW), adaptive threshold learning, and response adaptation to provide a complete caching layer for LLM-based systems.

### Evaluation Protocol

**Datasets:**
- MSMARCO Queries (relevance-focused)
- NaturalQuestions (factual QA)
- Custom repetitive query log (simulated enterprise)

**Metrics:**
1. Cache Hit Rate @ threshold τ
2. Latency Reduction Factor (LRF) = T_llm / T_cached
3. Response Quality Score (LLM-as-judge)
4. Semantic Similarity (BERTScore between cached and fresh)

**Ablation Studies:**
- Without re-ranking
- Without adaptation layer
- With different embedding models
- With different index types
- With different eviction policies

### Expected Contributions
1. First open-source production-ready semantic cache
2. Comprehensive evaluation framework
3. Adaptive threshold learning method
4. Hybrid retrieval (BM25 + semantic) for caching

---

## 🧪 Advanced Extensions

### 1. Adaptive Threshold Learning

The system automatically adjusts the similarity threshold based on user feedback:

```python
# Record feedback after response is evaluated
cache.record_feedback(
    request_id=result.request_id,
    quality_score=0.9,
    was_good=True,
)
# Threshold recalibrates every `adaptive_window_size` samples
```

### 2. Hybrid Retrieval (BM25 + Semantic)

Combines keyword matching with semantic similarity using Reciprocal Rank Fusion:

```python
config = CacheConfig()
# Hybrid retrieval activates automatically when entries have tags
```

### 3. Multi-Level Cache (L1/L2)

CPU-cache-like hierarchy for optimal performance:

```python
from neural_cache.multilevel_cache import MultiLevelCache

cache = MultiLevelCache(
    l1_max_entries=5000,    # Hot cache (in-memory)
    l2_config=...,           # Cold cache (SQLite)
    promote_on_l2_hit=True,  # Auto-promote
)
```

### 4. Cache Scoring Model

The `LEARNED_SCORING` decision strategy trains on feedback history to predict cache reliability — a learned model that outperforms fixed thresholds in dynamic workloads.

---

## 📝 Complete Example Workflow

```
User Query: "What is reinforcement learning?"
    │
    ▼
┌─ ENCODER ──────────────────────────────────┐
│  model: all-MiniLM-L6-v2                   │
│  input: "What is reinforcement learning?"   │
│  output: [0.12, -0.34, ..., 0.56] (384d)   │
│  latency: 8ms                              │
└─────────────────────────────────────────────┘
    │
    ▼
┌─ SEARCH ENGINE ────────────────────────────┐
│  index: HNSW (m=32, ef_search=64)          │
│  query: [0.12, -0.34, ..., 0.56]           │
│  top_k: 5                                  │
│  results:                                  │
│    1. "What is RL?"              (0.93)    │
│    2. "Explain reinforcement learning" (0.89)
│    3. "How does RL work?"        (0.81)    │
│  latency: 2ms                              │
└─────────────────────────────────────────────┘
    │
    ▼
┌─ DECISION POLICY ──────────────────────────┐
│  strategy: FIXED_THRESHOLD                 │
│  threshold: 0.85                           │
│  best match: 0.93 ≥ 0.85 → HIT             │
│  0.93 ≥ 0.95 (high_conf) → NO adaptation   │
│  action: HIT                               │
│  latency: 0.1ms                            │
└─────────────────────────────────────────────┘
    │
    ▼
┌─ RESPONSE ─────────────────────────────────┐
│  "Reinforcement learning is a type of      │
│   machine learning where an agent learns   │
│   to make decisions by interacting with    │
│   an environment..."                       │
│  total latency: 10.1ms (vs ~500ms LLM)    │
│  latency reduction: 98%                    │
└─────────────────────────────────────────────┘
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Run the test suite: `pytest tests/ -v`
5. Submit a pull request

---

## 📄 License

MIT License — see LICENSE file.

---

## 📚 References

1. Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*
2. Johnson, J. et al. (2019). *Billion-scale similarity search with GPUs* (FAISS)
3. Malkov, Y. & Yashunin, D. (2018). *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*
4. Robertson, S. & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*
