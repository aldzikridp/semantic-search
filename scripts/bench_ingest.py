"""Micro-benchmark: wall time to ingest one generated N-chunk text file.

Usage (inside `nix develop`):
    python scripts/bench_ingest.py --chunks 500 --repeats 3
"""

import argparse
import hashlib
import os
import statistics
import time
from pathlib import Path

from langchain_core.documents import Document
from pydantic import SecretStr

from semsearch.config import EmbeddingProviderConfig, Settings


def get_db_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        return url
    pgsocket = Path("~/Project/semantic-search/.pgsocket").expanduser()
    return (
        "postgresql+psycopg://semsearch:test@/semsearch"
        f"?host={pgsocket}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    from semsearch.service import SemanticSearchService
    from semsearch.splitter import split_documents

    # Calibrate lines so the file produces >= --chunks chunks.
    line = "benchmark chunk content " * 100  # ~2400 chars per line
    probe_settings = Settings(chunk_size=1000, chunk_overlap=200)
    probe_chunks = split_documents(
        [Document(page_content=line)],
        probe_settings.chunk_size,
        probe_settings.chunk_overlap,
    )
    chunks_per_line = len(probe_chunks)
    n_lines = -(-args.chunks // max(chunks_per_line, 1))

    tmp = Path("/tmp/bench_ingest_file.txt")
    tmp.write_text((line + "\n") * n_lines)

    settings = Settings(
        database_url=get_db_url(),
        collection_name="semsearch_bench",
        chunk_size=1000,
        chunk_overlap=200,
        embedding_provider=EmbeddingProviderConfig(
            type="openai",
            model="text-embedding-3-small",
            api_key=SecretStr("bench-key"),
        ),
    )

    class BenchEmbeddings:
        """Deterministic hash-based embeddings (same scheme as test MockEmbeddings)."""

        dim = 128

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [self._embed(t) for t in texts]

        def embed_query(self, text: str) -> list[float]:
            return self._embed(text)

        def _embed(self, text: str) -> list[float]:
            h = hashlib.sha256(text.encode()).digest()
            raw = (h * ((self.dim // 32) + 1))[: self.dim]
            return [b / 255.0 for b in raw]

    from semsearch.store import build_engine, init_schema

    embedder = BenchEmbeddings()
    engine = build_engine(settings)
    init_schema(settings, engine, embedder.dim, recreate=True)

    svc = SemanticSearchService(settings, engine, embedder)
    try:
        times: list[float] = []
        total_chunks = 0
        for i in range(args.repeats):
            if i > 0:
                tmp.write_text((line + "\n") * n_lines + f"variant {i}\n")
            start = time.perf_counter()
            result = svc.ingest(tmp)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            total_chunks = max(total_chunks, result.chunks_added)
            print(
                f"run {i + 1}: {elapsed * 1000:.1f} ms "
                f"(chunks={result.chunks_added}, updated={result.chunks_updated})"
            )
        if total_chunks < args.chunks:
            raise SystemExit(
                f"expected >= {args.chunks} chunks, got {total_chunks}"
            )
        print(
            f"\n{args.chunks}+ chunk file, {args.repeats} runs: "
            f"median={statistics.median(times) * 1000:.1f} ms "
            f"min={min(times) * 1000:.1f} ms"
        )
    finally:
        svc.close()
        tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
