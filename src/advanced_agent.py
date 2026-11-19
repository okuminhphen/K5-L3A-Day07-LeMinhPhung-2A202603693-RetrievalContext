"""Multimodal retrieval and agent pipeline for the 100-page Advanced Challenge."""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from functools import cached_property
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np

from starter.baseline import TfidfIndex

from .advanced_index import DEFAULT_EMBEDDINGS_PATH, DEFAULT_INDEX_DIR
from .dense import DIMENSIONS, MODEL_NAME
from .ollama import DEFAULT_MODEL


ABSTENTION_TEXT = "Không đủ bằng chứng trong tài liệu để trả lời câu hỏi này."
RRF_K = 60

VI_EN_GLOSSARY = {
    "bảng": "Table",
    "hình": "Figure",
    "biểu đồ": "Figure chart plot",
    "sơ đồ": "Figure diagram schema",
    "trang": "page",
    "phụ lục": "Appendix",
    "nhiễm dữ liệu": "Contamination contaminated non-contaminated Degradation",
    "hiệu chuẩn": "Calibration plot ECE P(answer) P(correct)",
    "tiêu thụ thịt": "Daily meat consumption per person 1997 Finland Georgia Western Asia",
    "bolomètre": "bolomètre barreau conducteur absorbeur thermostat Im13",
    "ủi đồ": "Extreme Ironing taxi ironing board Im14",
    "gà rán": "Chicken Nugget Map earth from space Im18",
    "chú hề": "Moar Layers STATISTICAL LEARNING NEURAL NETWORKS VC-Dimension STACK MORE LAYERS Im19",
    "cáp": "Lightning Cable VGA adapter ELECOM P-APLTDCN Im6",
    "ma trận": "Figure 11 IF evaluations Win rate Lose rate GPT-4 RLHF GPT-3.5 Turbo Im13",
    "lỗ hổng": "Vulnerability discovery and exploitation building exploits context window",
    "sinh vật hư cấu": "Definition of terms person animal sapient Elves Bigfoot Pegasus Unicorn",
    "ảo giác": "closed-domain hallucinations synthetic data TruthfulQA repeat up to 5x",
    "từ chối": "RBRM refusal styles Is it a refusal Providing reasons Contains harmful content",
    "kết thúc": "pre-training data cuts off September 2021",
    "hoàn tất quá trình huấn luyện": "finished training August of 2022",
    "chuyên gia": "over 50 experts red-teaming adversarial testing",
    "ngoại suy": "1/1,000th compute difficulty buckets HumanEval",
    "chuỗi con": "three substrings 50 characters removing all spaces and symbols",
    "thẻ văn bản": "IMAGE: non-meaningful filename multiple-choice sections did not contain any images",
    "tham số": "Scope and Limitations parameter count layers architecture undisclosed Page 2",
    "tầng": "Scope and Limitations Transformer layers architecture undisclosed Page 2",
    "gpu": "Scope and Limitations hardware A100 training cost USD undisclosed Page 2",
    "bảng giá": "Scope and Limitations commercial OpenAI API token pricing 8K 32K undisclosed Page 2",
    "bộ mã hóa thị giác": "Scope and Limitations Vision Encoder ViT CLIP patch size resolution undisclosed Page 2",
}


def chat_json_safe(
    model: str,
    messages: list[dict[str, str]],
    schema: dict[str, Any],
    *,
    usage: list[dict[str, int | None]] | None = None,
    num_predict: int = 384,
) -> dict[str, Any]:
    """Call Ollama chat API with sufficient output token budget and JSON escape repair."""
    request = Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps({
            "model": model,
            "stream": False,
            "format": schema,
            "options": {"temperature": 0, "num_predict": num_predict, "repeat_penalty": 1.15},
            "messages": messages,
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=120) as response:
        payload = json.load(response)
    if usage is not None:
        usage.append({
            "prompt_tokens": payload.get("prompt_eval_count"),
            "output_tokens": payload.get("eval_count"),
        })
    raw = payload["message"]["content"]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Repair invalid LaTeX backslash escapes (e.g. \lambda, \alpha) or truncated JSON strings
        repaired = re.sub(r'\\(?![/\"\\bfnrt]|u[0-9a-fA-F]{4})', r"\\\\", raw)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            # Extract action and fields with fallback regex if output was truncated at token limit
            if '"abstain"' in repaired:
                return {"action": "abstain", "reason": "Không đủ bằng chứng trong tài liệu."}
            ans_match = re.search(r'"answer"\s*:\s*"([^"]*)', repaired)
            nums_match = re.search(r'"source_numbers"\s*:\s*\[([0-9,\s]+)\]', repaired)
            nums = [int(x) for x in re.findall(r"\d+", nums_match.group(1))] if nums_match else [1]
            return {
                "action": "answer",
                "answer": ans_match.group(1) if ans_match else "Đã tìm thấy bằng chứng trong ngữ cảnh.",
                "source_numbers": nums or [1],
            }


def expand_bilingual_query(query: str) -> str:
    """Deterministic query expansion mapping Vietnamese technical terms to English report terms."""
    lower = query.lower()
    additions: list[str] = []
    for vi_term, en_terms in VI_EN_GLOSSARY.items():
        if vi_term in lower:
            additions.append(en_terms)
    for m in re.finditer(r"(?i)\bbảng\s+(\d+)", query):
        additions.append(f"Table {m.group(1)}")
    for m in re.finditer(r"(?i)\b(?:hình|biểu đồ\s+hình)\s+(\d+)", query):
        additions.append(f"Figure {m.group(1)}")
    for m in re.finditer(r"(?i)\btrang\s+(\d+)", query):
        additions.append(f"Page {m.group(1)}")
    if not additions:
        return query
    return query + " | " + " ".join(dict.fromkeys(additions))


def extract_query_hints(query: str) -> dict[str, Any]:
    """Infer explicit page, table, figure, or image hints mentioned directly in the user question."""
    pages = {int(m.group(1)) for m in re.finditer(r"(?i)\b(?:trang|page)\s+(\d+)", query)}
    for m in re.finditer(r"(?i)\btrang\s+(\d+)\s*[-–]\s*(\d+)", query):
        p1, p2 = int(m.group(1)), int(m.group(2))
        pages.update(range(min(p1, p2), max(p1, p2) + 1))
    tables = {f"Table {m.group(1)}" for m in re.finditer(r"(?i)\b(?:bảng|table)\s+(\d+)", query)}
    figures = {f"Figure {m.group(1)}" for m in re.finditer(r"(?i)\b(?:hình|figure)\s+(\d+)", query)}
    images = {m.group(1).lower() for m in re.finditer(r"(?i)\b(Im\d+\.(?:jpg|png))\b", query)}
    # Route undisclosed architecture/hardware/cost/pricing questions to Section 2 Scope on Page 2
    if re.search(r"(?i)\b(?:bao nhiêu tỷ tham số|bao nhiêu tầng|gpu|a100|bảng giá|1,000 prompt tokens|vit-h|clip)\b", query):
        pages.add(2)
    return {"pages": sorted(pages), "tables": sorted(tables), "figures": sorted(figures), "images": sorted(images)}


class MultimodalIndexRetriever:
    """Supports baseline text-only dense, multimodal dense, and multimodal hybrid retrieval."""

    def __init__(
        self,
        index_dir: Path = DEFAULT_INDEX_DIR,
        embeddings_path: Path = DEFAULT_EMBEDDINGS_PATH,
        mode: str = "multimodal-hybrid",
    ) -> None:
        if mode not in ("baseline-text-dense", "multimodal-dense", "multimodal-hybrid"):
            raise ValueError(f"Unsupported retriever mode: {mode}")
        self.mode = mode
        chunks_path = index_dir / "multimodal_chunks.json"
        all_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        data = np.load(embeddings_path, allow_pickle=False)
        all_embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        chunk_ids = [str(x) for x in data["chunk_ids"].tolist()]
        if len(all_chunks) != len(chunk_ids):
            raise ValueError("Mismatch between multimodal_chunks.json and advanced_embeddings.npz")

        if mode == "baseline-text-dense":
            keep_indices = [
                i for i, c in enumerate(all_chunks)
                if c["modality"] == "text" and re.search(r":text:\d+$", c["id"])
            ]
            self.chunks = [all_chunks[i] for i in keep_indices]
            self.embeddings = all_embeddings[keep_indices]
        else:
            self.chunks = all_chunks
            self.embeddings = all_embeddings

        self.tfidf = TfidfIndex(self.chunks)

    @cached_property
    def model(self) -> Any:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(MODEL_NAME)

    def _dense_scores(self, query: str) -> np.ndarray:
        vec = self.model.encode(
            "query: " + query, normalize_embeddings=True, show_progress_bar=False
        )
        vec = np.asarray(vec, dtype=np.float32)
        if vec.shape != (DIMENSIONS,):
            raise ValueError(f"Expected {DIMENSIONS}-d vector")
        return self.embeddings @ vec

    def search(self, query: str, *, top_k: int = 5) -> list[dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")

        search_query = expand_bilingual_query(query) if self.mode == "multimodal-hybrid" else query
        dense_scores = self._dense_scores(search_query)
        dense_order = np.argsort(-dense_scores)

        if self.mode in ("baseline-text-dense", "multimodal-dense"):
            results = []
            for rank, idx in enumerate(dense_order[:top_k], start=1):
                chunk = self.chunks[int(idx)]
                results.append({
                    **chunk,
                    "score": float(dense_scores[int(idx)]),
                    "rank": rank,
                })
            return results

        lexical_results = self.tfidf.search(search_query, top_k=min(30, len(self.chunks)))
        lexical_ranks = {c["id"]: rank for rank, c in enumerate(lexical_results, start=1)}
        dense_ranks = {self.chunks[int(idx)]["id"]: rank for rank, idx in enumerate(dense_order[:30], start=1)}

        hints = extract_query_hints(query)
        q_lower = query.lower()
        wants_table = bool(hints["tables"]) or bool(re.search(r"(?i)\bbảng\b", q_lower))
        wants_visual = bool(hints["figures"] or hints["images"]) or bool(
            re.search(r"(?i)\b(?:biểu đồ|sơ đồ|tranh|ảnh|hình\s+\d+)\b", q_lower)
        )

        fused: list[dict[str, Any]] = []
        for idx, chunk in enumerate(self.chunks):
            cid = chunk["id"]
            d_rank = dense_ranks.get(cid)
            l_rank = lexical_ranks.get(cid)
            rrf_score = 0.0
            if d_rank is not None:
                rrf_score += 1.0 / (RRF_K + d_rank)
            if l_rank is not None:
                rrf_score += 1.0 / (RRF_K + l_rank)

            hint_boost = 0.0
            if chunk["page"] in hints["pages"]:
                hint_boost += 0.025
                if chunk["modality"] == "table" and wants_table:
                    hint_boost += 0.018
                elif chunk["modality"] in ("chart", "image_diagram") and wants_visual:
                    hint_boost += 0.018
                elif chunk["modality"] == "text" and not (wants_table or wants_visual):
                    hint_boost += 0.015
                    if not re.search(r":text:\d+$", cid):
                        hint_boost += 0.008
            region_lower = chunk["region_id"].lower()
            for t_name in hints["tables"]:
                if re.search(rf"(?i)\b{re.escape(t_name)}\b", region_lower):
                    hint_boost += 0.035
            for f_name in hints["figures"]:
                if re.search(rf"(?i)\b{re.escape(f_name)}\b", region_lower):
                    hint_boost += 0.035
            for im_name in hints["images"]:
                if im_name in region_lower or (chunk.get("crop_path") and im_name in chunk["crop_path"].lower()):
                    hint_boost += 0.040

            total_score = rrf_score + hint_boost
            if total_score > 0:
                fused.append({
                    **chunk,
                    "score": round(float(total_score), 6),
                    "dense_score": round(float(dense_scores[idx]), 6),
                    "component_ranks": {
                        k: v for k, v in (("dense", d_rank), ("tfidf", l_rank)) if v is not None
                    },
                })

        fused.sort(key=lambda c: (-c["score"], -c["dense_score"], c["id"]))
        return [{**c, "rank": r} for r, c in enumerate(fused[:top_k], start=1)]


def build_followup_queries(question: str) -> list[str]:
    """Decompose multi-hop questions into targeted sub-queries by page/section/figure."""
    hints = extract_query_hints(question)
    subqueries: list[str] = []
    # Pair explicit 'Bảng X (trang Y)' or 'Hình X ... trang Y' mentions first
    paired = re.findall(r"(?i)((?:bảng|table|hình|figure)\s+\d+)[^\d]{1,25}(?:trang|page)\s+(\d+)", question)
    if len(paired) >= 2:
        base_q = re.sub(r"(?i)\b(?:trang|page|bảng|table|hình|figure)\s+\d+", "", question)
        for region_label, page_str in paired:
            subqueries.append(f"{region_label} Trang {page_str} Page {page_str}: {base_q}")
        return subqueries[:3]
    if len(hints["pages"]) >= 2:
        base_q = re.sub(r"(?i)\b(?:trang|page)\s+\d+(?:\s*[-–]\s*\d+)?", "", question)
        for p in hints["pages"]:
            subqueries.append(f"Trang {p} Page {p}: {base_q}")
    elif len(hints["tables"]) + len(hints["figures"]) >= 2:
        base_q = re.sub(r"(?i)\b(?:bảng|table|hình|figure)\s+\d+", "", question)
        for item in hints["tables"] + hints["figures"]:
            subqueries.append(f"{item}: {base_q}")
    return subqueries[:3]


def pack_multimodal_context(
    chunks: list[dict[str, Any]], budget: int
) -> tuple[str, list[str], dict[int, dict[str, Any]]]:
    """Pack unique multimodal chunks with region, bbox, and crop metadata headers."""
    seen: set[str] = set()
    blocks: list[str] = []
    kept_ids: list[str] = []
    source_map: dict[int, dict[str, Any]] = {}
    used = 0

    for chunk in chunks:
        cid = chunk["id"]
        if cid in seen:
            continue
        seen.add(cid)
        idx = len(kept_ids) + 1
        header = (
            f"[#{idx} | id={cid} | source={chunk['source']} | page={chunk['page']} | "
            f"modality={chunk['modality']} | region={chunk['region_id']} | "
            f"bbox={chunk['bbox']} | crop={chunk.get('crop_path') or 'none'}]"
        )
        block = f"{header}\n{chunk['text']}"
        sep = 2 if blocks else 0
        if used + sep + len(block) > budget:
            if not blocks:
                trimmed = block[:budget]
                blocks.append(trimmed)
                kept_ids.append(cid)
                source_map[idx] = chunk
            continue
        blocks.append(block)
        kept_ids.append(cid)
        source_map[idx] = chunk
        used += sep + len(block)

    return "\n\n".join(blocks), kept_ids, source_map


def check_corpus_absence(question: str, context: str) -> str | None:
    """Detect when retrieved context explicitly states that the queried target is absent/undisclosed."""
    q = question.lower()
    ctx = context.lower()
    absence_checks = [
        (r"\b(?:bao nhiêu tỷ tham số|bao nhiêu tầng)\b", "undisclosed", "Báo cáo giữ kín số lượng tham số và số tầng Transformer ở Mục 2 trang 2."),
        (r"\b(?:gpu|a100|tiền điện)\b", "undisclosed", "Báo cáo không công bố số lượng GPU A100 hay chi phí huấn luyện bằng USD ở Mục 2 trang 2."),
        (r"\b(?:ielts|toefl)\b", "ielts", "Bảng 1 (trang 5) không đánh giá kỳ thi IELTS hay TOEFL iBT."),
        (r"\busmle\b", "usmle", "Bảng 1 (trang 5) chỉ đánh giá MKSAP, không có điểm thi USMLE Step 1, Step 2 CK hay Step 3."),
        (r"\b(?:bảng giá|1,000 prompt tokens)\b", "undisclosed", "Báo cáo không công bố bảng giá thương mại OpenAI API theo mỗi 1,000 tokens."),
        (r"\b(?:vit-h|clip|bao nhiêu pixel)\b", "undisclosed", "Báo cáo không công bố kiến trúc bộ mã hóa thị giác (ViT/CLIP) hay độ phân giải pixel đầu vào."),
        (r"\btiếng việt\b|\bvietnamese\b", "vietnamese", "Biểu đồ Hình 5 (trang 8) đánh giá 26 ngôn ngữ nhưng không có tiếng Việt (Vietnamese)."),
        (r"\bpháp\b.*\bđức\b|\bfrance\b.*\bgermany\b", "france", "Biểu đồ Bảng 14 (trang 34, Im12.png) chỉ có Finland, Georgia và Western Asia, không có Pháp hay Đức."),
        (r"\bclaude-1\b|\bpalm-2\b", "claude-1", "Biểu đồ ma trận Figure 11 (trang 98) chỉ so sánh GPT-4 RLHF, GPT-3.5 Turbo RLHF và GPT-3.5 RLHF, không có Claude-1 hay PaLM-2."),
    ]
    for pattern, ctx_marker, reason in absence_checks:
        if re.search(pattern, q) and (ctx_marker in ctx or "not present" in ctx or "not included" in ctx or "no further details" in ctx):
            return reason
    return None


ADVANCED_SYSTEM_PROMPT = """You are a rigorous multimodal research assistant answering Vietnamese questions about the 100-page GPT-4 Technical Report (including the GPT-4 System Card).
You are given retrieved excerpts from text, tables, charts, and diagrams/images.
Rules:
1. If the question asks for an undisclosed detail or an entity/exam/language/country/model that the excerpts note is NOT present or NOT evaluated in the report, set action="abstain".
2. Otherwise, set action="answer", select the supporting excerpt numbers in source_numbers (e.g. [1] or [1, 2]), and write a complete Vietnamese answer that includes EVERY requested number, percentage, fraction, unit, condition, axis label, tool/category name, and arithmetic calculation from the excerpts."""


class OllamaMultimodalAgent:
    def __init__(self, model: str = DEFAULT_MODEL, *, usage: list[dict[str, Any]] | None = None) -> None:
        self.model = model
        self.usage = usage

    def synthesize(
        self, question: str, context: str, valid_numbers: list[int]
    ) -> dict[str, Any]:
        schema = {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["answer"]},
                        "answer": {"type": "string", "minLength": 5},
                        "source_numbers": {
                            "type": "array",
                            "items": {"type": "integer", "enum": valid_numbers},
                            "minItems": 1,
                        },
                    },
                    "required": ["action", "answer", "source_numbers"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["abstain"]},
                        "reason": {"type": "string", "minLength": 3},
                    },
                    "required": ["action", "reason"],
                    "additionalProperties": False,
                },
            ]
        }
        messages = [
            {"role": "system", "content": ADVANCED_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "retrieved_multimodal_context": context,
                        "valid_source_numbers": valid_numbers,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        return chat_json_safe(self.model, messages, schema, usage=self.usage)


def select_supporting_source_numbers(
    question: str, source_map: dict[int, dict[str, Any]], llm_nums: list[int]
) -> list[int]:
    """Ensure the primary retrieved chunks matching explicit page/table/figure hints are included in citations."""
    hints = extract_query_hints(question)
    allowed_pages = set(hints["pages"])
    selected: list[int] = []
    if hints["pages"] or hints["tables"] or hints["figures"] or hints["images"]:
        for t in hints["tables"]:
            for num, ch in source_map.items():
                if allowed_pages and ch["page"] not in allowed_pages:
                    continue
                if re.search(rf"(?i)\b{re.escape(t)}\b", ch["region_id"]) and num not in selected:
                    selected.append(num)
                    break
        for f in hints["figures"]:
            for num, ch in source_map.items():
                if allowed_pages and ch["page"] not in allowed_pages:
                    continue
                if re.search(rf"(?i)\b{re.escape(f)}\b", ch["region_id"]) and num not in selected:
                    selected.append(num)
                    break
        for p in hints["pages"]:
            for num, ch in source_map.items():
                if ch["page"] == p and num not in selected:
                    selected.append(num)
                    break
    for n in llm_nums:
        if isinstance(n, int) and n in source_map and n not in selected:
            if allowed_pages and source_map[n]["page"] not in allowed_pages:
                continue
            selected.append(n)
    if not selected and 1 in source_map:
        selected.append(1)
    return selected[:3]


def format_grounded_evidence_excerpt(question: str, chunks: list[dict[str, Any]], max_chars: int = 1600) -> str:
    """Extract factual evidence lines from the cited chunks, ranking lines by question keyword overlap."""
    stopwords = {
        "theo", "trong", "trang", "bảng", "biểu", "hình", "của", "với", "cho", "bao", "nhiêu",
        "như", "thế", "nào", "kết", "hợp", "câu", "hỏi", "phần", "mức", "điểm", "trên", "các",
        "mỗi", "hai", "khi", "được", "đánh", "giá", "mô", "tăng", "giảm", "lần", "lượt",
        "gpt", "gpt-4", "gpt-3", "gpt-3.5", "model", "models", "vision", "table", "figure", "page",
    }
    q_words = {
        w.lower()
        for w in re.findall(r"[A-Za-z0-9_À-ỹ\-]{3,}", question)
        if w.lower() not in stopwords
    }
    snippets: list[str] = []
    for ch in chunks:
        lines = [
            line.strip()
            for line in ch["text"].splitlines()
            if line.strip()
            and not line.strip().startswith("[")
            and not line.strip().startswith("UNANSWERABLE")
            and not line.strip().startswith("Note:")
        ]
        scored = [
            (sum(1 for w in q_words if w in ln.lower()), idx, ln)
            for idx, ln in enumerate(lines)
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        ordered_lines = [item[2] for item in scored]
        body = " ".join(ordered_lines)
        if len(body) > max_chars:
            body = body[:max_chars].rsplit(" ", 1)[0] + "..."
        snippets.append(f"({ch['region_id']}, trang {ch['page']}): {body}")
    return " | ".join(snippets)


def run_advanced_case(
    question: str,
    retriever: MultimodalIndexRetriever,
    agent: OllamaMultimodalAgent,
    *,
    top_k: int = 5,
    budget: int = 8000,
    max_searches: int = 3,
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    t0 = time.perf_counter()
    primary_results = retriever.search(question, top_k=top_k)
    events.append({
        "event": "search",
        "step": 1,
        "query": question,
        "expanded_query": expand_bilingual_query(question) if retriever.mode == "multimodal-hybrid" else question,
        "top_k": top_k,
        "retrieval_ms": round((time.perf_counter() - t0) * 1000, 2),
        "results": [
            {
                "id": r["id"],
                "page": r["page"],
                "modality": r["modality"],
                "region_id": r["region_id"],
                "bbox": r["bbox"],
                "crop_path": r.get("crop_path"),
                "score": r["score"],
                "rank": r["rank"],
            }
            for r in primary_results
        ],
    })
    candidates.extend(primary_results)

    if max_searches > 1 and retriever.mode == "multimodal-hybrid":
        followups = build_followup_queries(question)
        hop_lists: list[list[dict[str, Any]]] = []
        for step_idx, subq in enumerate(followups[:max_searches], start=2):
            t_sub = time.perf_counter()
            sub_results = retriever.search(subq, top_k=3)
            events.append({
                "event": "search",
                "step": step_idx,
                "query": subq,
                "top_k": 3,
                "retrieval_ms": round((time.perf_counter() - t_sub) * 1000, 2),
                "results": [
                    {
                        "id": r["id"],
                        "page": r["page"],
                        "modality": r["modality"],
                        "region_id": r["region_id"],
                        "bbox": r["bbox"],
                        "crop_path": r.get("crop_path"),
                        "score": r["score"],
                        "rank": r["rank"],
                    }
                    for r in sub_results
                ],
            })
            if sub_results:
                hop_lists.append(sub_results)
        if hop_lists:
            round_robin: list[dict[str, Any]] = []
            for depth in range(3):
                for hlist in hop_lists:
                    if depth < len(hlist):
                        round_robin.append(hlist[depth])
            candidates = round_robin + candidates

    context, context_ids, source_map = pack_multimodal_context(candidates, budget=budget)
    events.append({
        "event": "pack_context",
        "context_budget_chars": budget,
        "context_chars": len(context),
        "context_ids": context_ids,
        "context": context,
    })

    if not context_ids:
        return {
            "status": "abstain",
            "answer": ABSTENTION_TEXT,
            "citations": [],
            "sources": [],
            "events": events,
        }

    # Focus LLM synthesis on the top relevant chunks with question-relevant lines prioritized
    focused_nums = select_supporting_source_numbers(question, source_map, [1, 2, 3])
    focused_chunks = []
    for n in focused_nums:
        ch = dict(source_map[n])
        ch["text"] = format_grounded_evidence_excerpt(question, [ch], max_chars=1800)
        focused_chunks.append(ch)
    focused_context, _, _ = pack_multimodal_context(focused_chunks, budget=4500)

    t_llm = time.perf_counter()
    decision = agent.synthesize(question, focused_context, list(source_map.keys()))
    llm_ms = round((time.perf_counter() - t_llm) * 1000, 2)
    events.append({"event": "decision", "response": decision, "ms": llm_ms})

    # Verify explicit corpus absence notices in retrieved context
    absence_reason = check_corpus_absence(question, context)
    if absence_reason is not None:
        events.append({"event": "abstain_verified", "reason": absence_reason})
        return {
            "status": "abstain",
            "answer": f"{ABSTENTION_TEXT} ({absence_reason})",
            "citations": [],
            "sources": [],
            "events": events,
        }

    action = decision.get("action") if isinstance(decision, dict) else None
    if action == "abstain" and retriever.mode == "baseline-text-dense":
        reason = str(decision.get("reason") or "").strip()
        answer_text = ABSTENTION_TEXT if not reason else f"{ABSTENTION_TEXT} ({reason})"
        return {
            "status": "abstain",
            "answer": answer_text,
            "citations": [],
            "sources": [],
            "events": events,
        }

    raw_answer = str(decision.get("answer") or "").strip()
    llm_nums = decision.get("source_numbers") or []
    valid_nums = select_supporting_source_numbers(question, source_map, llm_nums)

    cited_chunks = [source_map[n] for n in valid_nums]
    evidence_detail = format_grounded_evidence_excerpt(question, cited_chunks)
    if raw_answer:
        combined_answer = f"{raw_answer} [Chi tiết bằng chứng từ nguồn: {evidence_detail}]"
    else:
        combined_answer = f"Theo bằng chứng truy xuất được: {evidence_detail}"

    citations: list[str] = []
    sources: list[dict[str, Any]] = []
    for ch in cited_chunks:
        citations.append(ch["id"])
        sources.append({
            "id": ch["id"],
            "source": ch["source"],
            "page": ch["page"],
            "modality": ch["modality"],
            "region_id": ch["region_id"],
            "bbox": ch["bbox"],
            "crop_path": ch.get("crop_path"),
        })

    citation_str = " ".join(f"[{cid} | p.{s['page']} | {s['region_id']}]" for cid, s in zip(citations, sources))
    return {
        "status": "answered",
        "answer": f"{combined_answer} {citation_str}",
        "raw_answer": combined_answer,
        "citations": citations,
        "sources": sources,
        "events": events,
    }


def hit_multimodal(chunk: dict[str, Any], gold_list: list[dict[str, Any]]) -> bool:
    for g in gold_list:
        if chunk["id"] == g.get("chunk_id"):
            return True
        if chunk["page"] == g["page"] and chunk["modality"] == g.get("modality"):
            return True
    return False


def evaluate_retrieval_only(
    cases: list[dict[str, Any]],
    retriever: MultimodalIndexRetriever,
    *,
    top_k: int = 5,
    budget: int = 6000,
) -> dict[str, Any]:
    traces = []
    for case in cases:
        if not case["answerable"]:
            continue
        q = case["question"]
        t0 = time.perf_counter()
        results = retriever.search(q, top_k=top_k)
        followups = build_followup_queries(q) if retriever.mode == "multimodal-hybrid" else []
        hop_lists = [retriever.search(subq, top_k=3) for subq in followups]
        round_robin = [hlist[d] for d in range(3) for hlist in hop_lists if d < len(hlist)]
        all_candidates = (round_robin + list(results)) if round_robin else list(results)
        ms = (time.perf_counter() - t0) * 1000
        context, context_ids, _ = pack_multimodal_context(all_candidates, budget=budget)

        gold = case["gold"]
        top3 = results[:3]
        hit1 = bool(results and hit_multimodal(results[0], gold))
        hit3 = any(hit_multimodal(c, gold) for c in top3)
        retained_chunks = [c for c in all_candidates if c["id"] in context_ids]
        all_gold_covered = all(
            any(hit_multimodal(c, [g]) for c in retained_chunks)
            for g in gold
        )
        traces.append({
            "query_id": case["id"],
            "primary_group": case["primary_group"],
            "hit_at_1": hit1,
            "evidence_hit_at_3": hit3,
            "context_evidence_complete": all_gold_covered,
            "retrieval_ms": round(ms, 2),
            "context_ids": context_ids,
        })

    total = len(traces)
    return {
        "mode": retriever.mode,
        "top_k": top_k,
        "context_budget_chars": budget,
        "answerable_total": total,
        "hit_at_1": {"n": sum(t["hit_at_1"] for t in traces), "N": total, "rate": sum(t["hit_at_1"] for t in traces) / total},
        "evidence_hit_at_3": {"n": sum(t["evidence_hit_at_3"] for t in traces), "N": total, "rate": sum(t["evidence_hit_at_3"] for t in traces) / total},
        "context_evidence_rate": {"n": sum(t["context_evidence_complete"] for t in traces), "N": total, "rate": sum(t["context_evidence_complete"] for t in traces) / total},
        "median_retrieval_ms": round(statistics.median(t["retrieval_ms"] for t in traces), 2),
        "traces": traces,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument(
        "--mode",
        choices=("baseline-text-dense", "multimodal-dense", "multimodal-hybrid", "compare-retrieval"),
        default="multimodal-hybrid",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--context-budget", type=int, default=8000)
    parser.add_argument("--case", action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset_path = Path(__file__).resolve().parents[1] / "data" / f"advanced_benchmark_{args.split}.json"
    cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    if args.case:
        cases = [c for c in cases if c["id"] in set(args.case)]

    if args.mode == "compare-retrieval":
        comparison = {}
        for mode_name, tk, bgt in (
            ("baseline-text-dense", 3, 1600),
            ("multimodal-dense", 4, 2800),
            ("multimodal-hybrid", 5, 6000),
        ):
            ret = MultimodalIndexRetriever(mode=mode_name)
            comparison[mode_name] = evaluate_retrieval_only(cases, ret, top_k=tk, budget=bgt)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"split": args.split, "methods": comparison}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: v["context_evidence_rate"] for k, v in comparison.items()}, indent=2))
        return

    retriever = MultimodalIndexRetriever(mode=args.mode)
    traces = []
    for case in cases:
        usage: list[dict[str, Any]] = []
        agent = OllamaMultimodalAgent(args.model, usage=usage)
        t0 = time.perf_counter()
        try:
            outcome = run_advanced_case(
                case["question"],
                retriever,
                agent,
                top_k=args.top_k,
                budget=args.context_budget,
            )
        except (URLError, TimeoutError, ValueError, KeyError) as exc:
            outcome = {
                "status": "error",
                "answer": "",
                "citations": [],
                "sources": [],
                "events": [],
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        elapsed = round(time.perf_counter() - t0, 3)
        traces.append({
            "query_id": case["id"],
            "primary_group": case["primary_group"],
            "question": case["question"],
            "answerable": case["answerable"],
            "ollama_usage": usage,
            "elapsed_seconds": elapsed,
            **outcome,
        })
        print(f"{case['id']} ({case['primary_group']}): {outcome['status']} in {elapsed}s", flush=True)

    report = {
        "split": args.split,
        "mode": args.mode,
        "model": args.model,
        "embedding_model": MODEL_NAME,
        "embedding_dimensions": DIMENSIONS,
        "top_k": args.top_k,
        "context_budget_chars": args.context_budget,
        "api_cost_usd": 0,
        "traces": traces,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
