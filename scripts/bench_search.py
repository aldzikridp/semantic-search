"""Benchmark the search path.

Usage:
    # Run against live DB with real embeddings
    python scripts/bench_search.py --queries queries.txt --n 50

    # Run with mock embeddings (no API calls)
    python scripts/bench_search.py --mock --n 100

    # Output JSON for diffing
    python scripts/bench_search.py --queries queries.txt --n 50 --json

    # Benchmark HTTP server (requires `semsearch serve` running)
    python scripts/bench_search.py --http --host localhost --port 8383 --n 50
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path so `tests.conftest` is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import httpx

from semsearch.config import get_settings
from semsearch.service import SemanticSearchService


def _percentile(sorted_data: list[float], pct: float) -> float:
    """Return the value at the given percentile from a sorted list.

    Uses linear interpolation between closest ranks for accuracy.
    """
    if len(sorted_data) == 1:
        return sorted_data[0]
    k = (len(sorted_data) - 1) * pct
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    d = k - f
    return sorted_data[f] * (1 - d) + sorted_data[c] * d


def _run_service_benchmark(
    svc: SemanticSearchService,
    queries: list[str],
    n: int,
    k: int,
    rerank: bool,
    warmup: int,
) -> dict:
    """Run benchmark against SemanticSearchService directly."""
    # Warmup — discard results to let caches / connections settle
    for _ in range(warmup):
        svc.search(queries[0], k=k)

    all_latencies: list[float] = []
    per_query: dict[str, list[float]] = {q: [] for q in queries}
    errors = 0

    for _ in range(n):
        for q in queries:
            try:
                t0 = time.perf_counter()
                svc.search(q, k=k, rerank=rerank)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                all_latencies.append(elapsed_ms)
                per_query[q].append(elapsed_ms)
            except Exception as e:
                errors += 1
                print(f"Warning: search failed for '{q}': {e}", file=sys.stderr)

    if not all_latencies:
        print("Error: all searches failed", file=sys.stderr)
        sys.exit(1)

    sorted_lat = sorted(all_latencies)
    return {
        "mode": "service",
        "n_queries": len(queries),
        "n_repetitions": n,
        "k": k,
        "rerank": rerank,
        "total_requests": len(all_latencies),
        "errors": errors,
        "p50_ms": round(_percentile(sorted_lat, 0.50), 2),
        "p95_ms": round(_percentile(sorted_lat, 0.95), 2),
        "p99_ms": round(_percentile(sorted_lat, 0.99), 2),
        "mean_ms": round(statistics.mean(all_latencies), 2),
        "min_ms": round(min(all_latencies), 2),
        "max_ms": round(max(all_latencies), 2),
        "per_query": {
            q: {
                "mean_ms": round(statistics.mean(latencies), 2),
                "p50_ms": round(_percentile(sorted(latencies), 0.50), 2),
                "count": len(latencies),
            }
            for q, latencies in per_query.items()
            if latencies
        },
    }


def _run_http_benchmark(
    host: str,
    port: int,
    queries: list[str],
    n: int,
    k: int,
    rerank: bool,
    warmup: int,
) -> dict:
    """Run benchmark against HTTP server."""
    base_url = f"http://{host}:{port}"
    client = httpx.Client(base_url=base_url, timeout=60.0)

    # Health check
    try:
        r = client.get("/health")
        r.raise_for_status()
    except Exception as e:
        print(f"Error: cannot reach server at {base_url}: {e}", file=sys.stderr)
        sys.exit(1)

    # Warmup
    for _ in range(warmup):
        client.post("/search", json={"query": queries[0], "k": k})

    all_latencies: list[float] = []
    per_query: dict[str, list[float]] = {q: [] for q in queries}
    errors = 0

    for _ in range(n):
        for q in queries:
            try:
                t0 = time.perf_counter()
                r = client.post("/search", json={"query": q, "k": k, "rerank": rerank})
                r.raise_for_status()
                elapsed_ms = (time.perf_counter() - t0) * 1000
                all_latencies.append(elapsed_ms)
                per_query[q].append(elapsed_ms)
            except Exception as e:
                errors += 1
                print(f"Warning: request failed for '{q}': {e}", file=sys.stderr)

    client.close()

    if not all_latencies:
        print("Error: all requests failed", file=sys.stderr)
        sys.exit(1)

    sorted_lat = sorted(all_latencies)
    return {
        "mode": "http",
        "server_url": base_url,
        "n_queries": len(queries),
        "n_repetitions": n,
        "k": k,
        "rerank": rerank,
        "total_requests": len(all_latencies),
        "errors": errors,
        "p50_ms": round(_percentile(sorted_lat, 0.50), 2),
        "p95_ms": round(_percentile(sorted_lat, 0.95), 2),
        "p99_ms": round(_percentile(sorted_lat, 0.99), 2),
        "mean_ms": round(statistics.mean(all_latencies), 2),
        "min_ms": round(min(all_latencies), 2),
        "max_ms": round(max(all_latencies), 2),
        "per_query": {
            q: {
                "mean_ms": round(statistics.mean(latencies), 2),
                "p50_ms": round(_percentile(sorted(latencies), 0.50), 2),
                "count": len(latencies),
            }
            for q, latencies in per_query.items()
            if latencies
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark semsearch search path")
    p.add_argument("--queries", type=Path, help="File with queries (one per line)")
    p.add_argument("--n", type=int, default=50, help="Repetitions per query")
    p.add_argument("--k", type=int, default=5, help="Top-k results")
    p.add_argument("--rerank", action="store_true", help="Enable reranking")
    p.add_argument("--mock", action="store_true", help="Use mock embeddings")
    p.add_argument("--json", action="store_true", help="JSON output")
    p.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    p.add_argument(
        "--config", "-c", type=Path, default=None, help="Path to .env config file"
    )
    p.add_argument("--http", action="store_true", help="Benchmark HTTP server instead of direct service")
    p.add_argument("--host", default="localhost", help="HTTP server host (with --http)")
    p.add_argument("--port", type=int, default=8383, help="HTTP server port (with --http)")
    args = p.parse_args()

    # Load queries
    if args.queries:
        queries = [line.strip() for line in args.queries.read_text().splitlines() if line.strip()]
    else:
        queries = ["test query", "how to deploy", "database setup", "search optimization"]

    if not queries:
        print("No queries found", file=sys.stderr)
        sys.exit(1)

    # HTTP mode — benchmark against running server
    if args.http:
        report = _run_http_benchmark(
            host=args.host,
            port=args.port,
            queries=queries,
            n=args.n,
            k=args.k,
            rerank=args.rerank,
            warmup=args.warmup,
        )
    # Service mode — benchmark direct service
    else:
        settings = get_settings(args.config)

        if args.mock:
            from tests.conftest import MockEmbeddings
            from semsearch.store import build_engine, init_schema

            embedder = MockEmbeddings(dim=128)
            engine = build_engine(settings)
            init_schema(settings, engine, embedder.dim, recreate=True)
            svc = SemanticSearchService(settings, engine, embedder)
        else:
            svc = SemanticSearchService.from_settings(settings)

        with svc:
            report = _run_service_benchmark(
                svc=svc,
                queries=queries,
                n=args.n,
                k=args.k,
                rerank=args.rerank,
                warmup=args.warmup,
            )

    # Output
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        mode = report["mode"]
        if mode == "http":
            print(f"Benchmark (HTTP): {report['server_url']}")
        else:
            print("Benchmark (Service)")
        print(f"  Queries: {report['n_queries']}, Repetitions: {report['n_repetitions']}")
        print(f"  Total requests: {report['total_requests']}")
        if report.get("errors"):
            print(f"  Errors: {report['errors']}")
        print(f"  p50:  {report['p50_ms']}ms")
        print(f"  p95:  {report['p95_ms']}ms")
        print(f"  p99:  {report['p99_ms']}ms")
        print(f"  mean: {report['mean_ms']}ms")
        print(f"  min:  {report['min_ms']}ms")
        print(f"  max:  {report['max_ms']}ms")
        if "per_query" in report and report["per_query"]:
            print("\n  Per-query breakdown:")
            for q, qs in report["per_query"].items():
                print(f"    '{q}': mean={qs['mean_ms']}ms, p50={qs['p50_ms']}ms, n={qs['count']}")


if __name__ == "__main__":
    main()
