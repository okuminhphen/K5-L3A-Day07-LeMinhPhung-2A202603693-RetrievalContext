"""Search Day 6 chunks with the same multilingual E5 model and pgvector."""
import argparse
import json
import os
from functools import cached_property
from pathlib import Path
from typing import Any


MODEL_NAME = "intfloat/multilingual-e5-small"
DIMENSIONS = 384


class DenseRetriever:
    """Expose the Day 6 index through the starter agent's retriever interface."""

    def __init__(self, database_url: str, strategy: str) -> None:
        self.database_url = database_url
        self.strategy = strategy

    @cached_property
    def model(self) -> Any:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(MODEL_NAME, model_kwargs={"use_safetensors": True})

    def load_chunks(self) -> list[dict[str, Any]]:
        """Read the same strategy's chunks to build a lexical index."""
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(self.database_url, connect_timeout=5, row_factory=dict_row) as conn:
            conn.read_only = True
            chunks = conn.execute(
                """
                SELECT chunk_id AS id, document_id AS source_id, source, page, text, metadata
                FROM ingestion_chunks
                WHERE strategy = %s
                ORDER BY chunk_id
                """,
                (self.strategy,),
            ).fetchall()
        if not chunks:
            raise ValueError("No chunks found for the selected Day 6 strategy")
        return chunks

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")

        import psycopg
        from psycopg.rows import dict_row
        from psycopg.types.json import Jsonb

        vector = self.model.encode(
            "query: " + query, normalize_embeddings=True, show_progress_bar=False
        ).tolist()
        if len(vector) != DIMENSIONS:
            raise ValueError(f"Expected a {DIMENSIONS}-dimensional E5 query vector")

        with psycopg.connect(self.database_url, connect_timeout=5, row_factory=dict_row) as conn:
            conn.read_only = True
            return conn.execute(
                """
                SELECT chunk_id AS id, document_id AS source_id, source, page, text, metadata,
                       1 - (embedding <=> %s::vector) AS score
                FROM ingestion_chunks
                WHERE strategy = %s AND metadata @> %s
                ORDER BY score DESC, chunk_id
                LIMIT %s
                """,
                (json.dumps(vector, allow_nan=False), self.strategy, Jsonb(filters or {}), top_k),
            ).fetchall()


def add_connection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--strategy", required=True, help="Full Day 6 strategy ID")
    parser.add_argument("--env-file", type=Path, help="Read DATABASE_URL from this file")


def database_url_from(env_file: Path | None) -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url and env_file:
        from dotenv import dotenv_values

        database_url = dotenv_values(env_file).get("DATABASE_URL")
    if not database_url:
        raise ValueError("Set DATABASE_URL or pass --env-file with the Day 6 .env path")
    return database_url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    add_connection_arguments(parser)
    parser.add_argument("--top-k", default=3, type=int)
    args = parser.parse_args()

    try:
        database_url = database_url_from(args.env_file)
        results = DenseRetriever(database_url, args.strategy).search(args.query, top_k=args.top_k)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
