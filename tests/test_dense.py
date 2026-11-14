"""Validate dense search inputs without loading a model or connecting to PostgreSQL."""
import unittest

from src.dense import DenseRetriever


class DenseRetrieverTests(unittest.TestCase):
    def test_empty_query_is_rejected_without_loading_model(self):
        retriever = DenseRetriever("postgresql://localhost/day6_lab", "index:B")

        with self.assertRaisesRegex(ValueError, "query"):
            retriever.search("  ")

    def test_nonpositive_top_k_is_rejected_without_loading_model(self):
        retriever = DenseRetriever("postgresql://localhost/day6_lab", "index:B")

        with self.assertRaisesRegex(ValueError, "top_k"):
            retriever.search("What is a variable?", top_k=0)
