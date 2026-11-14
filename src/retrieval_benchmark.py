"""Compare retrieval methods on the answerable Day 7 dev cases."""
import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

from starter.baseline import hit, pack_context

from .dense import DIMENSIONS, MODEL_NAME, DenseRetriever, add_connection_arguments, database_url_from
from .ollama import DEFAULT_MODEL, chat_json
from .retrievers import (
    HYBRID_CANDIDATES,
    RERANK_CANDIDATES,
    RERANK_TFIDF_WEIGHT,
    RRF_CONSTANT,
    DenseReranker,
    HybridRetriever,
    TfidfRetriever,
)


TOP_K = 3


def translate_query(question: str, model: str, *, usage=None) -> str:
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string", "minLength": 1}},
        "required": ["query"],
        "additionalProperties": False,
    }
    messages = [
        {"role": "system", "content": (
            "Translate the Vietnamese question into a natural English search question for "
            "the book Think Python. Preserve Python identifiers, literals, operators, "
            "chapter names and all conditions. Do not answer or add likely answers. "
            "Return only the translated question in the query field."
        )},
        {"role": "user", "content": json.dumps({"question": question}, ensure_ascii=False)},
    ]
    return chat_json(model, messages, schema, usage=usage)["query"]


def evaluate(
    cases, retriever, *, context_budget: int, translate: Callable[[str], str] | None = None
) -> dict:
    traces = []
    skipped = []
    for case in cases:
        if not case["answerable"]:
            skipped.append(case["id"])
            continue

        original_query, filters = case["question"], case.get("filter")
        query = original_query
        translation = {}
        if translate is not None:
            start = time.perf_counter()
            query = translate(original_query)
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f'{case["id"]}: translation must be a nonempty string')
            query = query.strip()
            translation = {
                "original_query": original_query,
                "translation_ms": (time.perf_counter() - start) * 1000,
            }
        start = time.perf_counter()
        results = retriever.search(query, filters=filters, top_k=TOP_K)
        retrieval_ms = (time.perf_counter() - start) * 1000
        context, context_ids = pack_context(results, budget=context_budget)
        gold = case["gold"]
        ranked = [
            {**chunk, "rank": rank, "evidence_hit": hit(chunk, gold)}
            for rank, chunk in enumerate(results, 1)
        ]
        first_rank = next((chunk["rank"] for chunk in ranked if chunk["evidence_hit"]), None)
        traces.append({
            "query_id": case["id"],
            "query": query,
            "filter": filters,
            "gold": gold,
            "results": ranked,
            "first_evidence_rank": first_rank,
            "hit_at_1": first_rank == 1,
            "evidence_hit_at_3": first_rank is not None,
            "packed_context_evidence": any(
                chunk["evidence_hit"] and chunk["id"] in context_ids for chunk in ranked
            ),
            "context_ids": context_ids,
            "context": context,
            "retrieval_ms": retrieval_ms,
            **translation,
        })

    if not traces:
        raise ValueError("Dev benchmark must contain answerable cases")
    total = len(traces)
    metrics = {}
    for key in ("hit_at_1", "evidence_hit_at_3", "packed_context_evidence"):
        hits = sum(trace[key] for trace in traces)
        metrics[key] = {"hits": hits, "total": total, "rate": hits / total}
    report = {
        "metrics": metrics,
        "median_retrieval_ms": statistics.median(trace["retrieval_ms"] for trace in traces),
        "skipped_unanswerable_ids": skipped,
        "traces": traces,
    }
    if translate is not None:
        report["median_translation_ms"] = statistics.median(t["translation_ms"] for t in traces)
        report["median_translate_and_retrieve_ms"] = statistics.median(
            t["translation_ms"] + t["retrieval_ms"] for t in traces
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_connection_arguments(parser)
    parser.add_argument(
        "--method", choices=("dense", "tfidf", "hybrid", "dense-rerank", "translation-ab", "all"),
        default="dense",
    )
    parser.add_argument("--translation-model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-budget", type=int, default=800, help="Character budget")
    args = parser.parse_args()
    if args.context_budget < 1:
        parser.error("--context-budget must be positive")

    questions = Path(__file__).resolve().parents[1] / "data" / "benchmark_dev.json"
    cases = json.loads(questions.read_text(encoding="utf-8"))
    try:
        dense = DenseRetriever(database_url_from(args.env_file), args.strategy)
        chunks = dense.load_chunks()
    except ValueError as exc:
        parser.error(str(exc))

    methods = ("dense", "tfidf", "hybrid") if args.method == "all" else (args.method,)
    retrievers = {"dense": dense}
    setup = {}
    if args.method == "translation-ab":
        methods = ("dense", "dense-english")
        retrievers["dense-english"] = dense
        setup["translation_model"] = args.translation_model
    if any(name in methods for name in ("dense", "hybrid", "dense-rerank")):
        start = time.perf_counter()
        _ = dense.model
        setup.update(
            embedding_model=MODEL_NAME,
            embedding_dimensions=DIMENSIONS,
            model_load_seconds=time.perf_counter() - start,
        )
    if any(name in methods for name in ("tfidf", "hybrid", "dense-rerank")):
        start = time.perf_counter()
        tfidf = TfidfRetriever(chunks)
        setup["tfidf_index_seconds"] = time.perf_counter() - start
        retrievers.update(tfidf=tfidf, hybrid=HybridRetriever(dense, tfidf))
    if "hybrid" in methods:
        setup.update(rrf_constant=RRF_CONSTANT, hybrid_candidates_per_method=HYBRID_CANDIDATES)
    if "dense-rerank" in methods:
        retrievers["dense-rerank"] = DenseReranker(dense, tfidf)
        setup.update(
            rerank_dense_candidates=RERANK_CANDIDATES,
            rerank_tfidf_weight=RERANK_TFIDF_WEIGHT,
        )

    common = {
        "split": "dev",
        "day6_strategy": args.strategy,
        "chunk_count": len(chunks),
        "top_k": TOP_K,
        "context_budget_chars": args.context_budget,
        "api_cost_usd": 0,
        **setup,
    }
    reports = {}
    for name in methods:
        start = time.perf_counter()
        translator = None
        if name == "dense-english":
            def translator(question):
                return translate_query(question, args.translation_model)
        report = evaluate(
            cases, retrievers[name], context_budget=args.context_budget, translate=translator
        )
        reports[name] = {"evaluation_seconds": time.perf_counter() - start, **report}
    if len(methods) > 1:
        report = {**common, "methods": reports}
    else:
        report = {**common, "method": args.method, **reports[args.method]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: result["metrics"] for name, result in reports.items()}, indent=2))


if __name__ == "__main__":
    main()
