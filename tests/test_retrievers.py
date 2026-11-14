"""Check that keyword search and rank fusion obey the retriever contract."""
import unittest

from src.retrievers import DenseReranker, HybridRetriever, TfidfRetriever


class RetrieverTests(unittest.TestCase):
    def test_tfidf_applies_filter_before_cutoff(self):
        retriever = TfidfRetriever([
            {"id": "a", "text": "function", "metadata": {"chapter": "Intro"}},
            {"id": "b", "text": "function", "metadata": {"chapter": "Functions"}},
        ])
        results = retriever.search("function", filters={"chapter": "Functions"}, top_k=1)
        self.assertEqual([row["id"] for row in results], ["b"])

    def test_hybrid_fuses_by_rank_and_returns_unique_chunks(self):
        class Ranked:
            def __init__(self, ids):
                self.ids = ids

            def search(self, query, *, filters, top_k):
                if filters != {"chapter": "Functions"}:
                    raise AssertionError("Missing chapter filter")
                scores = [1000, 0.01]
                return [
                    {"id": chunk_id, "score": score}
                    for chunk_id, score in zip(self.ids[:top_k], scores)
                ]

        retriever = HybridRetriever(Ranked(["a", "shared"]), Ranked(["b", "shared"]))
        results = retriever.search("function", filters={"chapter": "Functions"}, top_k=3)
        self.assertEqual([row["id"] for row in results], ["shared", "a", "b"])
        self.assertEqual(results[0]["component_ranks"], {"dense": 2, "tfidf": 2})

    def test_reranker_promotes_lexical_match_only_if_dense_retrieved_it(self):
        class Dense:
            def search(self, query, *, filters, top_k):
                if top_k != 10 or filters != {"chapter": "Functions"}:
                    raise AssertionError("Unexpected candidate search")
                return [{"id": "generic", "score": 0.83}, {"id": "specific", "score": 0.81}]

        chunks = [
            {"id": "generic", "text": "example", "metadata": {"chapter": "Functions"}},
            {"id": "specific", "text": "function", "metadata": {"chapter": "Functions"}},
            {"id": "lexical-only", "text": "function", "metadata": {"chapter": "Functions"}},
        ]
        retriever = DenseReranker(Dense(), TfidfRetriever(chunks))
        results = retriever.search("function", filters={"chapter": "Functions"}, top_k=2)

        self.assertEqual([row["id"] for row in results], ["specific", "generic"])
        self.assertEqual(results[0]["dense_rank"], 2)
        self.assertEqual(results[1]["tfidf_score"], 0.0)
