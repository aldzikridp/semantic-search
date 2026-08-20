"""Compare benchmark results before and after changes.

Usage:
    python scripts/compare_benchmarks.py before.json after_025.json after_026.json
"""
from __future__ import annotations

import json
import sys

REQUIRED_KEYS = ["p50_ms", "p95_ms", "p99_ms", "mean_ms"]


def _load_json(path: str) -> dict:
    """Load JSON file with proper resource management."""
    with open(path) as f:
        return json.load(f)


def _validate(data: dict, path: str) -> None:
    """Validate that JSON contains required keys."""
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        print(f"Error: {path} missing keys: {missing}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python compare_benchmarks.py before.json after.json [after2.json ...]")
        sys.exit(1)

    files = sys.argv[1:]
    baseline = _load_json(files[0])
    _validate(baseline, files[0])
    results: list[dict] = []

    for f in files[1:]:
        after = _load_json(f)
        _validate(after, f)
        comparison: dict = {"file": f}
        for k in REQUIRED_KEYS:
            b, a = baseline[k], after[k]
            pct = ((b - a) / b) * 100 if b != 0 else 0.0
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
        for k in REQUIRED_KEYS:
            c = r[k]
            symbol = "✅" if c["improved"] else "❌"
            print(f"  {k}: {c['before']}ms → {c['after']}ms ({c['change_pct']:+.1f}%) {symbol}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        total_improvement = sum(
            1 for k in REQUIRED_KEYS
            if r[k]["improved"]
        )
        print(f"{r['file']}: {total_improvement}/4 metrics improved")


if __name__ == "__main__":
    main()
