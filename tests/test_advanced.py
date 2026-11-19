import unittest

from src.advanced_agent import (
    ABSTENTION_TEXT,
    check_corpus_absence,
    expand_bilingual_query,
    extract_query_hints,
    hit_multimodal,
    pack_multimodal_context,
    select_supporting_source_numbers,
)
from src.advanced_evaluate import evaluate_advanced_run


class AdvancedChallengeTests(unittest.TestCase):
    def test_extract_query_hints_parses_pages_tables_figures_and_images(self):
        q = "Kết hợp Bảng 1 (trang 5) và Hình 11 ở trang 98 (Im13.png)"
        hints = extract_query_hints(q)
        self.assertEqual(hints["pages"], [5, 98])
        self.assertEqual(hints["tables"], ["Table 1"])
        self.assertEqual(hints["figures"], ["Figure 11"])
        self.assertEqual(hints["images"], ["im13.png"])

    def test_expand_bilingual_query_adds_english_report_terms(self):
        expanded = expand_bilingual_query("Theo Bảng 8 (trang 28), hiệu chuẩn và nhiễm dữ liệu là gì?")
        self.assertIn("Table 8", expanded)
        self.assertIn("Page 28", expanded)
        self.assertIn("Contamination", expanded)
        self.assertIn("Calibration plot", expanded)

    def test_select_supporting_source_numbers_uses_word_boundaries_for_table_1_vs_11(self):
        source_map = {
            1: {"id": "c1", "page": 32, "region_id": "Table 11"},
            2: {"id": "c2", "page": 5, "region_id": "Table 1"},
        }
        nums = select_supporting_source_numbers("Theo Bảng 1 (trang 5)...", source_map, [1])
        self.assertEqual(nums[0], 2)

    def test_check_corpus_absence_detects_missing_entities_in_context(self):
        reason = check_corpus_absence(
            "Điểm MMLU tiếng Việt (Vietnamese) trong Hình 5 là bao nhiêu?",
            "Note: Vietnamese is NOT included among the languages evaluated in Figure 5.",
        )
        self.assertIsNotNone(reason)
        self.assertIn("Vietnamese", reason)

    def test_pack_multimodal_context_skips_oversized_and_packs_remaining(self):
        chunks = [
            {
                "id": "c1", "source": "doc.pdf", "page": 1, "modality": "text",
                "region_id": "P1", "bbox": [0, 0, 10, 10], "crop_path": None,
                "text": "A" * 50,
            },
            {
                "id": "c2", "source": "doc.pdf", "page": 2, "modality": "table",
                "region_id": "T1", "bbox": [0, 0, 10, 10], "crop_path": None,
                "text": "B" * 500,
            },
            {
                "id": "c3", "source": "doc.pdf", "page": 3, "modality": "chart",
                "region_id": "F1", "bbox": [0, 0, 10, 10], "crop_path": "crop.png",
                "text": "C" * 40,
            },
        ]
        context, ids, smap = pack_multimodal_context(chunks, budget=380)
        self.assertEqual(ids, ["c1", "c3"])
        self.assertEqual(set(smap.keys()), {1, 2})
        self.assertIn("crop=crop.png", context)

    def test_evaluate_advanced_run_enforces_strict_pass_fail_and_90_percent_threshold(self):
        cases = [
            {
                "id": "AT01",
                "primary_group": "table",
                "answerable": True,
                "required_pages": [5],
                "gold": [{"page": 5, "modality": "table"}],
            },
            {
                "id": "AT25",
                "primary_group": "unanswerable",
                "answerable": False,
            },
        ]
        run = {
            "split": "test",
            "mode": "multimodal-hybrid",
            "model": "llama3.2:latest",
            "embedding_model": "intfloat/multilingual-e5-small",
            "top_k": 5,
            "context_budget_chars": 6000,
            "traces": [
                {
                    "query_id": "AT01",
                    "status": "answered",
                    "answer": "163 / 170",
                    "sources": [{"id": "t1", "page": 5, "modality": "table", "region_id": "Table 1"}],
                    "ollama_usage": [{"prompt_tokens": 100, "output_tokens": 30}],
                    "elapsed_seconds": 1.5,
                },
                {
                    "query_id": "AT25",
                    "status": "abstain",
                    "answer": ABSTENTION_TEXT,
                    "sources": [],
                    "ollama_usage": [{"prompt_tokens": 90, "output_tokens": 20}],
                    "elapsed_seconds": 1.2,
                },
            ],
        }
        reviews = [
            {
                "query_id": "AT01",
                "answer_accurate_and_complete": True,
                "citation_locates_page_and_region": True,
                "evidence_supports_full_answer": True,
                "passed": True,
                "rationale": "Exact match with Table 1 p.5",
                "original_pdf_verification": "Verified on page 5",
            },
            {
                "query_id": "AT25",
                "abstained_without_speculation": True,
                "passed": True,
                "rationale": "Properly abstained on undisclosed API pricing",
                "original_pdf_verification": "Verified across 100 pages",
            },
        ]
        summary = evaluate_advanced_run(cases, run, reviews)
        self.assertEqual(summary["overall"]["passed"], 2)
        self.assertEqual(summary["overall"]["total"], 2)
        self.assertTrue(summary["overall"]["meets_90_percent_bonus_threshold"])
        self.assertEqual(summary["timing_and_cost"]["total_tokens"], 240)
        self.assertTrue(hit_multimodal({"id": "x", "page": 5, "modality": "table"}, cases[0]["gold"]))


if __name__ == "__main__":
    unittest.main()
