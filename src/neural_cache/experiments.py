from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

@dataclass
class ExperimentResult:
    name: str
    parameters: dict[str, Any]
    metrics: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "name": self.name,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> ExperimentResult:
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

class ExperimentRunner:
    def __init__(self, output_dir: str | Path = "./experiment_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[ExperimentResult] = []

    async def run_all(self, llm_generate, test_queries: list[str]) -> list[ExperimentResult]:
        experiments = [
            ("latency_comparison", {}),
            ("threshold_sensitivity", {}),
            ("cache_size_impact", {}),
            ("eviction_policy_comparison", {}),
            ("embedding_model_comparison", {}),
        ]

        all_results = []
        for name, params in experiments:
            print(f"\n{'='*60}")
            print(f"Running experiment: {name}")
            print(f"{'='*60}")

            method = getattr(self, f"run_{name}", None)
            if method:
                result = await method(llm_generate, test_queries)
                all_results.append(result)
                self.results.append(result)
                result.save(self.output_dir / f"{name}.json")
                self._print_metrics(result)

        return all_results

    async def run_latency_comparison(
        self,
        llm_generate,
        test_queries: list[str],
    ) -> ExperimentResult:

        from neural_cache.cache import NeuralCache
        from neural_cache.config import CacheConfig

        config = CacheConfig.fast_production()
        cache = NeuralCache(config)
        await cache.initialize()
        cache.set_llm_function(llm_generate)

        training_queries = self._generate_training_queries(200)
        print(f"Warming cache with {len(training_queries)} training queries...")
        for q in training_queries:
            await cache.get(q)

        print("Measuring latency WITH cache...")
        latencies_with_cache = []
        hit_count = 0
        for q in test_queries:
            start = time.monotonic()
            result = await cache.get(q)
            latency_ms = (time.monotonic() - start) * 1000
            latencies_with_cache.append(latency_ms)
            if result.from_cache:
                hit_count += 1

        hit_rate = hit_count / len(test_queries) if test_queries else 0
        print("Measuring latency WITHOUT cache...")
        latencies_without_cache = []
        for q in test_queries:
            start = time.monotonic()
            await llm_generate(q)
            latency_ms = (time.monotonic() - start) * 1000
            latencies_without_cache.append(latency_ms)

        metrics = {
            "avg_latency_with_cache_ms": float(np.mean(latencies_with_cache)),
            "avg_latency_without_cache_ms": float(np.mean(latencies_without_cache)),
            "p50_latency_with_cache_ms": float(np.percentile(latencies_with_cache, 50)),
            "p50_latency_without_cache_ms": float(np.percentile(latencies_without_cache, 50)),
            "p95_latency_with_cache_ms": float(np.percentile(latencies_with_cache, 95)),
            "p95_latency_without_cache_ms": float(np.percentile(latencies_without_cache, 95)),
            "p99_latency_with_cache_ms": float(np.percentile(latencies_with_cache, 99)),
            "p99_latency_without_cache_ms": float(np.percentile(latencies_without_cache, 99)),
            "cache_hit_rate": hit_rate,
            "latency_reduction_pct": (
                (1 - np.mean(latencies_with_cache) / np.mean(latencies_without_cache)) * 100
                if np.mean(latencies_without_cache) > 0
                else 0
            ),
            "n_queries": len(test_queries),
        }

        cache.close()
        return ExperimentResult(
            name="latency_comparison",
            parameters={
                "n_training_queries": 200,
                "n_test_queries": len(test_queries),
                "cache_config": "fast_production",
            },
            metrics=metrics,
        )

    async def run_threshold_sensitivity(
        self,
        llm_generate,
        test_queries: list[str],
    ) -> ExperimentResult:

        from neural_cache.cache import NeuralCache
        from neural_cache.config import CacheConfig

        thresholds = np.arange(0.70, 0.99, 0.03)
        results_by_threshold = {}
        for threshold in thresholds:
            threshold = round(float(threshold), 2)
            config = CacheConfig.fast_production()
            config.decision.similarity_threshold = threshold
            cache = NeuralCache(config)
            await cache.initialize()
            cache.set_llm_function(llm_generate)

            training = self._generate_training_queries(100)
            for q in training:
                await cache.get(q)

            hits = 0
            latencies = []
            for q in test_queries[:50]:
                start = time.monotonic()
                result = await cache.get(q)
                latencies.append((time.monotonic() - start) * 1000)
                if result.from_cache:
                    hits += 1

            results_by_threshold[str(threshold)] = {
                "hit_rate": hits / len(test_queries[:50]),
                "avg_latency_ms": float(np.mean(latencies)),
                "p95_latency_ms": float(np.percentile(latencies, 95)),
            }

            cache.close()
            print(f"  Threshold {threshold:.2f}: hit_rate={hits/len(test_queries[:50]):.3f}, "
                  f"avg_latency={np.mean(latencies):.1f}ms")
        best_threshold = max(
            results_by_threshold.items(),
            key=lambda x: x[1]["hit_rate"]
        )

        return ExperimentResult(
            name="threshold_sensitivity",
            parameters={"thresholds_tested": list(thresholds)},
            metrics={
                "best_threshold": float(best_threshold[0]),
                "best_hit_rate": best_threshold[1]["hit_rate"],
            },
            metadata={"results_by_threshold": results_by_threshold},
        )

    async def run_cache_size_impact(
        self,
        llm_generate,
        test_queries: list[str],
    ) -> ExperimentResult:

        from neural_cache.cache import NeuralCache
        from neural_cache.config import CacheConfig

        sizes = [100, 500, 1000, 2500, 5000, 10000]
        results_by_size = {}
        for size in sizes:
            config = CacheConfig.fast_production()
            config.storage.max_entries = size
            cache = NeuralCache(config)
            await cache.initialize()
            cache.set_llm_function(llm_generate)

            training = self._generate_training_queries(size)
            for q in training:
                await cache.get(q)

            hits = 0
            for q in test_queries[:100]:
                result = await cache.get(q)
                if result.from_cache:
                    hits += 1

            hit_rate = hits / len(test_queries[:100])
            results_by_size[str(size)] = {
                "hit_rate": hit_rate,
                "cache_size": cache.storage.count(),
            }

            cache.close()
            print(f"  Size {size}: hit_rate={hit_rate:.3f}")
        return ExperimentResult(
            name="cache_size_impact",
            parameters={"sizes_tested": sizes},
            metrics=results_by_size,
        )

    async def run_eviction_policy_comparison(
        self,
        llm_generate,
        test_queries: list[str],
    ) -> ExperimentResult:

        from neural_cache.cache import NeuralCache
        from neural_cache.config import CacheConfig, EvictionPolicy

        policies = [EvictionPolicy.LRU, EvictionPolicy.LFU, EvictionPolicy.SCORE_BASED]
        results = {}
        for policy in policies:
            config = CacheConfig.fast_production()
            config.storage.max_entries = 500
            config.eviction.policy = policy
            config.eviction.high_watermark = 0.8
            config.eviction.low_watermark = 0.6

            cache = NeuralCache(config)
            await cache.initialize()
            cache.set_llm_function(llm_generate)

            training = self._generate_training_queries(800)
            for q in training:
                await cache.get(q)

            hits = 0
            for q in test_queries[:50]:
                result = await cache.get(q)
                if result.from_cache:
                    hits += 1

            hit_rate = hits / len(test_queries[:50])
            results[policy.value] = {
                "hit_rate": hit_rate,
                "eviction_count": cache.eviction_manager.eviction_count,
            }

            cache.close()
            print(f"  Policy {policy.value}: hit_rate={hit_rate:.3f}")
        return ExperimentResult(
            name="eviction_policy_comparison",
            parameters={"policies": [p.value for p in policies]},
            metrics=results,
        )

    async def run_embedding_model_comparison(
        self,
        llm_generate,
        test_queries: list[str],
    ) -> ExperimentResult:

        from neural_cache.cache import NeuralCache
        from neural_cache.config import CacheConfig, EmbeddingModel

        models = [
            EmbeddingModel.ALL_MINILM_L6_V2,
            EmbeddingModel.PARAPHRASE_MINILM_L3,
        ]
        results = {}
        for model in models:
            config = CacheConfig()
            config.encoder.model_name = model
            cache = NeuralCache(config)
            await cache.initialize()
            cache.set_llm_function(llm_generate)

            training = self._generate_training_queries(100)
            for q in training:
                await cache.get(q)

            hits = 0
            latencies = []
            for q in test_queries[:30]:
                start = time.monotonic()
                result = await cache.get(q)
                latencies.append((time.monotonic() - start) * 1000)
                if result.from_cache:
                    hits += 1

            hit_rate = hits / len(test_queries[:30])
            results[model.value.split("/")[-1]] = {
                "hit_rate": hit_rate,
                "avg_latency_ms": float(np.mean(latencies)),
                "embedding_dim": cache.encoder.dimension,
            }

            cache.close()
            print(f"  Model {model.value.split('/')[-1]}: hit_rate={hit_rate:.3f}")
        return ExperimentResult(
            name="embedding_model_comparison",
            parameters={"models": [m.value for m in models]},
            metrics=results,
        )

    def _generate_training_queries(self, n: int) -> list[str]:
        topics = [
            "What is", "How does", "Explain", "Describe", "Compare",
            "What are the benefits of", "What is the difference between",
            "How to implement", "Why is", "When should I use",
        ]
        subjects = [
            "machine learning", "neural networks", "deep learning",
            "natural language processing", "computer vision",
            "reinforcement learning", "transformers", "attention mechanisms",
            "vector databases", "embedding models", "semantic search",
            "retrieval augmented generation", "fine-tuning", "transfer learning",
            "gradient descent", "backpropagation", "convolutional networks",
            "recurrent networks", "generative adversarial networks",
            "variational autoencoders", "diffusion models", "language models",
            "prompt engineering", "chain of thought", "few-shot learning",
            "zero-shot learning", "multimodal AI", "speech recognition",
            "image generation", "text summarization", "question answering",
            "sentiment analysis", "named entity recognition", "text classification",
            "recommendation systems", "anomaly detection", "time series forecasting",
            "graph neural networks", "knowledge graphs", "ontologies",
            "federated learning", "differential privacy", "model compression",
            "quantization", "pruning", "knowledge distillation",
            "Python programming", "software engineering", "distributed systems",
            "microservices", "container orchestration", "cloud computing",
            "database optimization", "API design", "system architecture",
            "data pipelines", "ETL processes", "stream processing",
            "event-driven architecture", "message queues", "caching strategies",
            "load balancing", "fault tolerance", "consensus algorithms",
        ]
        queries = []
        for i in range(n):
            topic = topics[i % len(topics)]
            subject = subjects[i % len(subjects)]
            queries.append(f"{topic} {subject}?")

        return queries

    def _print_metrics(self, result: ExperimentResult) -> None:
        print(f"\nExperiment: {result.name}")
        print(f"Parameters: {json.dumps(result.parameters, indent=2)}")
        print(f"Metrics:")
        for key, value in result.metrics.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
        print()
