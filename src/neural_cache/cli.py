from __future__ import annotations

import asyncio
import json
import sys

import click
from rich.console import Console
from rich.table import Table

from neural_cache.config import CacheConfig

console = Console()

@click.group()
@click.version_option(version="0.1.0")
def main():
    pass

@main.command()
@click.option("--config", type=click.Choice(["fast", "accurate", "default"]), default="default")
@click.option("--db-path", type=str, default=None)
def init(config: str, db_path: str | None):
    if config == "fast":
        cfg = CacheConfig.fast_production()
    elif config == "accurate":
        cfg = CacheConfig.high_accuracy()
    else:
        cfg = CacheConfig()

    if db_path:
        from neural_cache.config import StorageConfig, StorageBackend
        cfg.storage.backend = StorageBackend.SQLITE
        cfg.storage.db_path = db_path

    console.print(f"[green]Initialized Neural Cache with config: {config}[/green]")
    console.print(f"  Encoder: {cfg.encoder.model_name.value}")
    console.print(f"  Storage: {cfg.storage.backend.value}")
    console.print(f"  Search: {cfg.search.index_type}")
    console.print(f"  Decision: {cfg.decision.strategy.value}")

@main.command()
def stats():
    from neural_cache.cache import NeuralCache

    cache = NeuralCache(CacheConfig.fast_production())
    asyncio.run(cache.initialize())

    stats = cache.get_stats()
    metrics = cache.get_metrics()
    cache.close()

    table = Table(title="Cache Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Cache Size", str(stats["cache_size"]))
    table.add_row("Search Index Type", stats["search_engine"].get("index_type", "N/A"))
    table.add_row("Search Index Size", str(stats["search_engine"].get("total_vectors", 0)))
    table.add_row("Decision Strategy", stats["decision_policy"]["strategy"])
    table.add_row("Current Threshold", f"{stats['decision_policy']['current_threshold']:.3f}")
    table.add_row("Eviction Policy", stats["eviction"]["policy"])
    table.add_row("Eviction Count", str(stats["eviction"]["eviction_count"]))

    console.print(table)

    if metrics.total_requests > 0:
        metric_table = Table(title="Performance Metrics")
        metric_table.add_column("Metric", style="cyan")
        metric_table.add_column("Value", style="green")

        metric_table.add_row("Total Requests", str(metrics.total_requests))
        metric_table.add_row("Cache Hit Rate", f"{metrics.hit_rate:.2%}")
        metric_table.add_row("Avg Latency", f"{metrics.avg_latency_ms:.1f} ms")
        metric_table.add_row("P50 Latency", f"{metrics.p50_latency_ms:.1f} ms")
        metric_table.add_row("P95 Latency", f"{metrics.p95_latency_ms:.1f} ms")
        metric_table.add_row("Avg Quality Score", f"{metrics.avg_quality_score:.3f}")

        console.print(metric_table)

@main.command()
@click.confirmation_option(prompt="Are you sure you want to clear the cache?")
def clear():
    from neural_cache.cache import NeuralCache

    cache = NeuralCache(CacheConfig.fast_production())
    asyncio.run(cache.initialize())
    cache.clear()
    cache.close()

    console.print("[yellow]Cache cleared successfully.[/yellow]")

@main.command()
@click.option("--output-dir", type=str, default="./experiment_results")
@click.option("--n-queries", type=int, default=50)
def experiment(output_dir: str, n_queries: int):
    from neural_cache.experiments import ExperimentRunner

    async def mock_llm(query: str) -> tuple[str, dict]:
        import asyncio
        await asyncio.sleep(0.5)
        return f"Response to: {query}", {"model": "mock"}

    runner = ExperimentRunner(output_dir)
    queries = runner._generate_training_queries(n_queries)

    async def run():
        return await runner.run_all(mock_llm, queries)

    results = asyncio.run(run())
    console.print(f"[green]Experiments complete. Results saved to {output_dir}/[/green]")

@main.command()
@click.option("--host", type=str, default="127.0.0.1")
@click.option("--port", type=int, default=8000)
def serve(host: str, port: int):
    console.print(f"[yellow]Starting Neural Cache server on {host}:{port}[/yellow]")
    console.print("[dim]This feature requires fastapi and uvicorn.[/dim]")
    console.print("[dim]Install with: pip install fastapi uvicorn[/dim]")

    try:
        from fastapi import FastAPI
        import uvicorn
    except ImportError:
        console.print("[red]fastapi and uvicorn are required for server mode.[/red]")
        sys.exit(1)

    app = FastAPI(title="Neural Cache API")

    from neural_cache.cache import NeuralCache
    cache = NeuralCache(CacheConfig.fast_production())

    @app.on_event("startup")
    async def startup():
        await cache.initialize()

    @app.on_event("shutdown")
    async def shutdown():
        cache.close()

    @app.post("/query")
    async def query_endpoint(query: str):
        result = asyncio.run(cache.get(query))
        return {
            "response": result.response,
            "from_cache": result.from_cache,
            "similarity_score": result.similarity_score,
            "latency_ms": result.total_latency_ms,
            "request_id": result.request_id,
        }

    @app.get("/metrics")
    async def metrics_endpoint():
        metrics = cache.get_metrics()
        return {
            "total_requests": metrics.total_requests,
            "cache_hit_rate": metrics.hit_rate,
            "avg_latency_ms": metrics.avg_latency_ms,
            "p95_latency_ms": metrics.p95_latency_ms,
            "cache_size": metrics.cache_size,
        }

    @app.get("/stats")
    async def stats_endpoint():
        return cache.get_stats()

    uvicorn.run(app, host=host, port=port)

@main.command()
@click.argument("query")
@click.option("--json-output", is_flag=True, default=False)
def ask(query: str, json_output: bool):
    from neural_cache.cache import NeuralCache

    async def mock_llm(query: str) -> tuple[str, dict]:
        return f"Mock response to: {query}", {"model": "mock"}

    cache = NeuralCache(CacheConfig.fast_production())

    async def run():
        await cache.initialize()
        cache.set_llm_function(mock_llm)
        result = await cache.get(query)
        return result

    result = asyncio.run(run())
    cache.close()

    if json_output:
        click.echo(json.dumps({
            "response": result.response,
            "from_cache": result.from_cache,
            "similarity_score": result.similarity_score,
            "latency_ms": result.total_latency_ms,
        }, indent=2))
    else:
        console.print(f"\n[bold]Query:[/bold] {query}")
        console.print(f"[bold]Response:[/bold] {result.response}")
        console.print(f"[dim]From cache: {result.from_cache} | "
                      f"Similarity: {result.similarity_score:.3f} | "
                      f"Latency: {result.total_latency_ms:.1f}ms[/dim]")

if __name__ == "__main__":
    main()
