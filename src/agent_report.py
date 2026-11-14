"""Summarize an agent trace with independently reviewed answer and citation labels."""
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from starter.agent_metrics import aggregate
from starter.baseline import hit


def summarize(cases, run, reviews):
    metrics = aggregate(cases, run["traces"], reviews)
    traces = {trace["query_id"]: trace for trace in run["traces"]}
    ranks = []
    context_hits = []
    counts = Counter()
    for case in cases:
        trace = traces[case["id"]]
        searches = [event for event in trace["events"] if event["event"] == "search"]
        counts[len(searches)] += 1
        if not case["answerable"]:
            continue
        gold = case["gold"]
        initial = searches[0]["results"] if searches else []
        ranks.append(next((rank for rank, chunk in enumerate(initial, 1) if hit(chunk, gold)), None))
        if searches:
            retained = set(searches[-1]["context_ids"])
            if any(hit(chunk, gold) and chunk["id"] in retained
                   for search in searches for chunk in search["results"]):
                context_hits.append(case["id"])

    report = {key: value for key, value in run.items() if key != "traces"}
    report.update(
        metrics=metrics,
        fully_grounded_out_of_all_answerable={"n": metrics["grounded_answer"]["2"], "N": len(ranks)},
        hit_at_1={"n": ranks.count(1), "N": len(ranks)},
        evidence_hit_at_3={"n": sum(rank is not None and rank <= 3 for rank in ranks), "N": len(ranks)},
        final_context_exact_evidence={"n": len(context_hits), "N": len(ranks), "query_ids": context_hits},
        search_call_distribution=dict(sorted(counts.items())),
        median_elapsed_seconds=statistics.median(trace["elapsed_seconds"] for trace in traces.values()),
        total_case_elapsed_seconds=sum(trace["elapsed_seconds"] for trace in traces.values()),
    )
    context_grades = [review["context_grade"] for review in reviews if "context_grade" in review]
    if context_grades:
        report["reviewed_context_grades"] = {str(grade): context_grades.count(grade) for grade in range(3)}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--reviews", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cases, run, reviews = [
        json.loads(path.read_text(encoding="utf-8")) for path in (args.cases, args.run, args.reviews)
    ]
    if {case["id"] for case in cases} != {trace["query_id"] for trace in run["traces"]}:
        parser.error("The case IDs must match the run")
    summary = summarize(cases, run, reviews)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
