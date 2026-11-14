import unittest

from src.agent import run_agent


CHUNK = {
    "id": "book:42:1",
    "source": "book.pdf",
    "page": 42,
    "text": "The documented value is 7.",
}


class Controller:
    def __init__(self, *actions):
        self.actions = iter(actions)

    def decide(self, question, context, queries, allowed_actions):
        if queries:
            assert CHUNK["text"] in context
        return next(self.actions)


class Retriever:
    def search(self, query, *, filters, top_k):
        assert query == "documented value"
        assert filters == {"chapter": "Example"}
        assert top_k == 3
        return [CHUNK]


class AgentTests(unittest.TestCase):
    def test_numbered_citation_resolves_to_the_chunk_and_page(self):
        class NumberedController:
            def decide(self, question, context, queries, allowed_actions, *, source_numbers):
                if not queries:
                    return {"action": "search", "query": "documented value"}
                assert source_numbers == [1]
                assert context.startswith("[1 | book.pdf | page 42]")
                return {"action": "answer", "answer": "7", "source_number": 1, "quote": CHUNK["text"]}

        result = run_agent(
            "What is the value?", NumberedController(), Retriever(),
            filters={"chapter": "Example"}, numbered_citations=True,
        )
        self.assertEqual(result["citations"], [CHUNK["id"]])
        self.assertEqual(result["sources"], [{"id": CHUNK["id"], "source": "book.pdf", "page": 42}])

    def test_translates_first_search_only_and_keeps_the_question_for_the_model(self):
        question = "Giá trị là gì?"
        requested = []
        queries = []

        def translate(text):
            requested.append(text)
            return "What is the documented value?"

        class SearchTwice:
            def decide(self, original, context, previous_queries, allowed_actions):
                if original != question:
                    raise AssertionError("Expected the original Vietnamese question")
                if not previous_queries:
                    return {"action": "search", "query": original}
                if len(previous_queries) == 1:
                    return {"action": "search", "query": "documented value"}
                return {"action": "abstain"}

        class RecordingRetriever:
            def search(self, query, *, filters, top_k):
                queries.append((query, filters, top_k))
                return [CHUNK]

        filters = {"chapter": "Example"}
        report = run_agent(question, SearchTwice(), RecordingRetriever(), filters=filters, translate=translate)

        self.assertEqual(requested, [question])
        self.assertEqual(queries, [
            ("What is the documented value?", filters, 3), ("documented value", filters, 3),
        ])
        first = next(event for event in report["events"] if event["event"] == "search")
        self.assertEqual(first["original_query"], question)
        self.assertEqual(first["query"], "What is the documented value?")

    def run_actions(self, *actions):
        return run_agent(
            "What is the value?",
            Controller(*actions),
            Retriever(),
            filters={"chapter": "Example"},
        )

    def test_renders_citation_from_retrieved_context(self):
        result = self.run_actions(
            {"action": "search", "query": "documented value"},
            {"action": "answer", "answer": "7", "citations": [CHUNK["id"]], "quotes": [CHUNK["text"]]},
        )
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["answer"], "7 [book:42:1]")
        self.assertEqual(result["events"][1]["context_ids"], [CHUNK["id"]])

    def test_invalid_citation_becomes_abstention_with_trace(self):
        result = self.run_actions(
            {"action": "search", "query": "documented value"},
            {"action": "answer", "answer": "7", "citations": ["invented"]},
        )
        self.assertEqual(result["status"], "abstain")
        self.assertEqual(result["events"][-1]["event"], "invalid_citations")

    def test_model_can_abstain_after_search(self):
        result = self.run_actions(
            {"action": "search", "query": "documented value"},
            {"action": "abstain"},
        )
        self.assertEqual(result["status"], "abstain")
        self.assertEqual(result["events"][-1]["event"], "abstain")

    def test_quote_must_be_present_in_its_cited_chunk(self):
        result = self.run_actions(
            {"action": "search", "query": "documented value"},
            {"action": "answer", "answer": "8", "citations": [CHUNK["id"]], "quotes": ["The value is 8."]},
        )
        self.assertEqual(result["status"], "abstain")
        self.assertEqual(result["events"][-1]["event"], "invalid_evidence_quotes")

    def test_model_cannot_answer_before_search(self):
        with self.assertRaisesRegex(ValueError, "action"):
            self.run_actions({"action": "answer", "answer": "7", "citations": [CHUNK["id"]]})
