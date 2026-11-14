"""Exercise evidence scoring without PostgreSQL or an embedding model."""
import unittest

from src.retrieval_benchmark import evaluate


class RetrievalBenchmarkTests(unittest.TestCase):
    def test_translated_query_preserves_filter_and_original_question_in_trace(self):
        class Retriever:
            def search(self, query, *, filters, top_k):
                if (query, filters, top_k) != ("What is a module?", {"chapter": "Functions"}, 3):
                    raise AssertionError("Unexpected retrieval input")
                return []

        requested_translations = []

        def translate(question):
            requested_translations.append(question)
            return "What is a module?"

        cases = [
            {"id": "Q1", "question": "Module là gì?", "answerable": True,
             "filter": {"chapter": "Functions"}, "gold": []},
            {"id": "Q2", "question": "Thiếu dữ liệu?", "answerable": False},
        ]
        report = evaluate(cases, Retriever(), context_budget=1600, translate=translate)
        self.assertEqual(requested_translations, ["Module là gì?"])
        self.assertEqual(report["traces"][0]["original_query"], "Module là gì?")
        self.assertEqual(report["traces"][0]["query"], "What is a module?")
        self.assertGreaterEqual(report["median_translation_ms"], 0)

    def test_scores_rank_two_evidence_and_excludes_unanswerable_cases(self):
        gold = {"source_id": "book", "page": 42, "evidence": "matching evidence"}
        chunk = {"source_id": "book", "source": "book.pdf"}

        class Retriever:
            def search(self, query, *, filters, top_k):
                if (query, filters, top_k) != ("Question?", {"chapter": "Functions"}, 3):
                    raise AssertionError("Unexpected retrieval input")
                return [
                    {**chunk, "id": "wrong-page", "page": 41, "text": "matching evidence"},
                    {**chunk, "id": "right-page", "page": 42, "text": "matching evidence"},
                ]

        cases = [
            {"id": "Q1", "question": "Question?", "answerable": True,
             "filter": {"chapter": "Functions"}, "gold": [gold]},
            {"id": "Q2", "question": "Missing?", "answerable": False},
        ]
        report = evaluate(cases, Retriever(), context_budget=800)

        self.assertEqual(report["metrics"]["hit_at_1"]["hits"], 0)
        self.assertEqual(report["metrics"]["evidence_hit_at_3"]["hits"], 1)
        self.assertEqual(report["metrics"]["evidence_hit_at_3"]["total"], 1)
        self.assertEqual(report["metrics"]["packed_context_evidence"]["hits"], 1)
        self.assertEqual(report["traces"][0]["first_evidence_rank"], 2)
        self.assertEqual(report["skipped_unanswerable_ids"], ["Q2"])

        report = evaluate(cases, Retriever(), context_budget=1)
        self.assertEqual(report["metrics"]["packed_context_evidence"]["hits"], 0)
