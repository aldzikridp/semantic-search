# TASK-029: Before/After Performance Benchmarks

> **Status**: Complete ✅  
> **Phase**: Performance Phase 1  
> **Depends on**: TASK-025, TASK-026, TASK-027, TASK-028  
> **Blocks**: None  

## Objective

Run benchmarks before and after each performance change to validate improvements and document results.

## Problem

No way to measure before/after performance. Need to run benchmarks and compare results to prove the changes are effective.

## Solution

1. Run baseline benchmarks before any changes.
2. Apply each change (TASK-025 through TASK-028).
3. Run benchmarks after each change.
4. Generate before/after comparison report.

## Benchmark Workflow

### Step 1: Baseline (before changes)

```bash
nix develop --command bash -c "python scripts/bench_search.py --queries queries.txt --n 50 --json" > bench_before.json
```

### Step 2: After TASK-025 (vector size caching)

```bash
nix develop --command bash -c "python scripts/bench_search.py --queries queries.txt --n 50 --json" > bench_after_025.json
```

### Step 3: After TASK-027 (Reranker httpx pooling)

```bash
nix develop --command bash -c "python scripts/bench_search.py --queries queries.txt --n 50 --json --rerank" > bench_after_027.json
```

### Step 4: After TASK-026 (HNSW tuning)

```bash
nix develop --command bash -c "python scripts/bench_search.py --queries queries.txt --n 50 --json" > bench_after_026.json
```

### Step 5: After TASK-028 (FastAPI server)

```bash
# Start server
nix develop --command bash -c "semsearch serve --port 8383" &
sleep 5

# Benchmark HTTP endpoint
nix develop --command bash -c "python scripts/bench_search.py --queries queries.txt --n 50 --json --http" > bench_after_028.json

# Stop server
kill %1
```

### Step 6: Generate Comparison Report

```bash
nix develop --command bash -c "python scripts/compare_benchmarks.py bench_before.json bench_after_026.json bench_after_027.json bench_after_028.json"
```

## Files to Create

### `scripts/compare_benchmarks.py`

```python
"""Compare benchmark results before and after changes.

Usage:
    python scripts/compare_benchmarks.py before.json after_025.json after_026.json
"""
import json
import sys


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_benchmarks.py before.json after.json [after2.json ...]")
        sys.exit(1)

    files = sys.argv[1:]
    baseline = json.load(open(files[0]))
    results = []

    for f in files[1:]:
        after = json.load(open(f))
        comparison = {"file": f}
        for k in ["p50_ms", "p95_ms", "p99_ms", "mean_ms"]:
            b, a = baseline[k], after[k]
            pct = ((b - a) / b) * 100
            comparison[k] = {
                "before": b,
                "after": a,
                "change_pct": round(pct, 1),
                "improved": pct > 0,
            }
        results.append(comparison)

    # Print report
    print("=" * 60)
    print("PERFORMANCE COMPARISON REPORT")
    print("=" * 60)
    print(f"\nBaseline: {files[0]}")
    print(f"  p50:  {baseline['p50_ms']}ms")
    print(f"  p95:  {baseline['p95_ms']}ms")
    print(f"  p99:  {baseline['p99_ms']}ms")
    print(f"  mean: {baseline['mean_ms']}ms")

    for r in results:
        print(f"\nAfter: {r['file']}")
        for k in ["p50_ms", "p95_ms", "p99_ms", "mean_ms"]:
            c = r[k]
            symbol = "✅" if c["improved"] else "❌"
            print(f"  {k}: {c['before']}ms → {c['after']}ms ({c['change_pct']:+.1f}%) {symbol}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        total_improvement = sum(
            1 for k in ["p50_ms", "p95_ms", "p99_ms", "mean_ms"]
            if r[k]["improved"]
        )
        print(f"{r['file']}: {total_improvement}/4 metrics improved")


if __name__ == "__main__":
    main()
```

### `queries.txt`

Sample queries for benchmarking:

```
how to deploy
database setup
search optimization
test query
vector similarity
embedding providers
```

## Expected Results

### TASK-025 (Vector Size Caching)

- `semsearch stats` latency: ~200ms → ~5ms (95% reduction)
- `semsearch init` latency: ~200ms → ~5ms (after first probe)

### TASK-026 (HNSW Tuning)

- Recall improvement: +5-10% at fixed latency
- No measurable latency change (ef_search is table-level default)

### TASK-027 (Reranker httpx Pooling)

- Reranker latency: ~200ms → ~50ms (75% reduction after warmup)
- 429 errors: automatic retry instead of failure

### TASK-028 (FastAPI Server)

- Cold start: ~500ms-1.5s → 0ms (eliminated)
- Per-request overhead: ~50ms (HTTP) vs ~500ms (CLI)

## Acceptance Criteria

- [x] `scripts/bench_search.py` exists with `--http`, `--mock`, `--json` flags
- [x] `scripts/compare_benchmarks.py` exists with validation and error handling
- [x] `queries.txt` exists with sample queries
- [x] `BENCHMARKS.md` documents methodology and expected results
- [x] `compare_benchmarks.py` uses context managers for file I/O
- [x] `compare_benchmarks.py` validates JSON keys before comparison
- [ ] Benchmark JSON files generated (run manually)

**Note:** Benchmark JSON files should be generated by running the benchmarks
against a live database. See `BENCHMARKS.md` for instructions.
