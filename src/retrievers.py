"""Keyword and hybrid retrievers for the Day 7 agent interface."""
from typing import Any

from starter.baseline import TfidfIndex


RRF_CONSTANT = 60
HYBRID_CANDIDATES = 10
RERANK_CANDIDATES = 10
RERANK_TFIDF_WEIGHT = 0.20


class TfidfRetriever:
    """Score the shared corpus with the starter TF-IDF algorithm."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.index = TfidfIndex(chunks)

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")

        results = self.index.search(query, top_k=len(self.index.chunks))
        if filters:
            results = [
                chunk for chunk in results
                if all(key in chunk["metadata"] and chunk["metadata"][key] == value
                       for key, value in filters.items())
            ]
        return results[:top_k]


class HybridRetriever:
    """Combine dense and lexical ranks with reciprocal rank fusion."""

    def __init__(self, dense, tfidf) -> None:
        self.dense = dense
        self.tfidf = tfidf

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        depth = max(HYBRID_CANDIDATES, top_k)
        fused: dict[str, dict[str, Any]] = {}
        for name, retriever in (("dense", self.dense), ("tfidf", self.tfidf)):
            for rank, chunk in enumerate(retriever.search(query, filters=filters, top_k=depth), 1):
                result = fused.setdefault(chunk["id"], {**chunk, "score": 0.0, "component_ranks": {}})
                result["component_ranks"][name] = rank
                result["score"] += 1 / (RRF_CONSTANT + rank)
        return sorted(fused.values(), key=lambda chunk: (-chunk["score"], chunk["id"]))[:top_k]


class DenseReranker:
    """Rerank only the dense shortlist using a lexical similarity signal."""

    def __init__(self, dense, tfidf: TfidfRetriever) -> None:
        self.dense = dense
        self.tfidf = tfidf

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        candidates = self.dense.search(query, filters=filters, top_k=max(RERANK_CANDIDATES, top_k))
        lexical_scores = {
            chunk["id"]: chunk["score"]
            for chunk in self.tfidf.search(query, filters=filters, top_k=len(self.tfidf.index.chunks))
        }
        results = []
        for rank, chunk in enumerate(candidates, 1):
            dense_score = chunk["score"]
            tfidf_score = lexical_scores.get(chunk["id"], 0.0)
            score = (1 - RERANK_TFIDF_WEIGHT) * dense_score + RERANK_TFIDF_WEIGHT * tfidf_score
            results.append({
                **chunk,
                "score": score,
                "dense_rank": rank,
                "dense_score": dense_score,
                "tfidf_score": tfidf_score,
            })
        return sorted(results, key=lambda chunk: (-chunk["score"], chunk["dense_rank"]))[:top_k]
