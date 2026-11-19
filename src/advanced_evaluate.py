"""Aggregate and verify pass/fail evaluations for the Advanced Challenge benchmark."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


GROUP_ORDER = ("table", "image_diagram", "chart", "multi_hop", "text", "unanswerable")
GROUP_LABELS = {
    "table": "Bảng",
    "image_diagram": "Ảnh, sơ đồ",
    "chart": "Biểu đồ",
    "multi_hop": "Multi-hop",
    "text": "Text",
    "unanswerable": "Thiếu bằng chứng",
}


def evaluate_advanced_run(
    cases: list[dict[str, Any]],
    run: dict[str, Any],
    reviews: list[dict[str, Any]],
) -> dict[str, Any]:
    traces_by_id = {t["query_id"]: t for t in run["traces"]}
    reviews_by_id = {r["query_id"]: r for r in reviews}
    if set(traces_by_id) != {c["id"] for c in cases}:
        raise ValueError("Case IDs in benchmark and run traces do not match")
    if set(reviews_by_id) != {c["id"] for c in cases}:
        raise ValueError("Review IDs must cover every benchmark case")

    group_stats: dict[str, dict[str, Any]] = {
        g: {"label": GROUP_LABELS[g], "passed": 0, "total": 0, "failed_ids": []}
        for g in GROUP_ORDER
    }
    case_results: list[dict[str, Any]] = []
    system_errors = 0
    prompt_tokens = 0
    output_tokens = 0

    for case in cases:
        cid = case["id"]
        group = case["primary_group"]
        trace = traces_by_id[cid]
        review = reviews_by_id[cid]
        group_stats[group]["total"] += 1

        for u in trace.get("ollama_usage") or []:
            prompt_tokens += int(u.get("prompt_tokens") or 0)
            output_tokens += int(u.get("output_tokens") or 0)

        if trace["status"] == "error":
            system_errors += 1
            passed = False
        elif case["answerable"]:
            cited_pages = {s["page"] for s in (trace.get("sources") or [])}
            req_pages = set(case.get("required_pages") or [g["page"] for g in case["gold"]])
            pages_covered = req_pages.issubset(cited_pages)
            passed = bool(
                trace["status"] == "answered"
                and pages_covered
                and review.get("answer_accurate_and_complete") is True
                and review.get("citation_locates_page_and_region") is True
                and review.get("evidence_supports_full_answer") is True
                and review.get("passed") is True
            )
        else:
            passed = bool(
                trace["status"] == "abstain"
                and review.get("abstained_without_speculation") is True
                and review.get("passed") is True
            )

        if passed:
            group_stats[group]["passed"] += 1
        else:
            group_stats[group]["failed_ids"].append(cid)

        case_results.append({
            "query_id": cid,
            "primary_group": group,
            "answerable": case["answerable"],
            "status": trace["status"],
            "passed": passed,
            "cited_sources": trace.get("sources") or [],
            "elapsed_seconds": trace.get("elapsed_seconds", 0.0),
            "rationale": review.get("rationale", ""),
            "original_pdf_verification": review.get("original_pdf_verification", ""),
        })

    total_cases = len(cases)
    total_passed = sum(g["passed"] for g in group_stats.values())
    pass_rate = total_passed / total_cases if total_cases else 0.0

    for g in group_stats.values():
        g["rate"] = round(g["passed"] / g["total"], 4) if g["total"] else 0.0
        g["n_over_N"] = f"{g['passed']}/{g['total']}"

    elapsed_list = [t.get("elapsed_seconds", 0.0) for t in run["traces"]]
    return {
        "split": run["split"],
        "mode": run["mode"],
        "model": run["model"],
        "embedding_model": run["embedding_model"],
        "top_k": run["top_k"],
        "context_budget_chars": run["context_budget_chars"],
        "overall": {
            "passed": total_passed,
            "total": total_cases,
            "n_over_N": f"{total_passed}/{total_cases}",
            "pass_rate": round(pass_rate, 4),
            "pass_percentage": round(pass_rate * 100, 2),
            "meets_90_percent_bonus_threshold": pass_rate >= 0.90,
        },
        "by_group": group_stats,
        "system_errors": system_errors,
        "timing_and_cost": {
            "median_elapsed_seconds": round(statistics.median(elapsed_list), 3) if elapsed_list else 0.0,
            "total_elapsed_seconds": round(sum(elapsed_list), 3),
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": prompt_tokens + output_tokens,
            "api_cost_usd": 0.0,
        },
        "case_results": case_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    run = json.loads(args.run.read_text(encoding="utf-8"))
    reviews = json.loads(args.reviews.read_text(encoding="utf-8"))
    summary = evaluate_advanced_run(cases, run, reviews)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
