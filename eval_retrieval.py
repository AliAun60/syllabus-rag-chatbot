from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
warnings.filterwarnings("ignore", message="Relevance scores must be between 0 and 1.*")

from retriever import retrieve_documents

DEFAULT_EVAL_PATH = Path("eval/retrieval_eval_cases.json")
DEFAULT_K = 4


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list):
        raise ValueError("Evaluation file must contain a list of cases.")

    return cases


def expected_sources(case: dict[str, Any]) -> set[str]:
    if "expected_sources" in case:
        return {str(source) for source in case["expected_sources"]}
    raise ValueError(f"Case is missing expected_sources: {case.get('question', '<unknown>')}")


def source_name(document: Any) -> str:
    return str(document.metadata.get("document_name", "Unknown Document"))


def evaluate_case(case: dict[str, Any], k: int, use_reranking: bool) -> dict[str, Any]:
    expected = expected_sources(case)
    with contextlib.redirect_stderr(io.StringIO()):
        documents = retrieve_documents(
            case["question"],
            top_k=k,
            course_filter=case.get("expected_course"),
            use_reranking=use_reranking,
        )
    retrieved = [source_name(document) for document in documents]
    unique_retrieved = set(retrieved)
    hits = unique_retrieved.intersection(expected)
    relevant_chunks = sum(1 for source in retrieved if source in expected)

    precision = relevant_chunks / k if k else 0
    recall = len(hits) / len(expected) if expected else 0

    return {
        "question": case["question"],
        "expected": sorted(expected),
        "retrieved": retrieved,
        "precision": precision,
        "recall": recall,
        "hit": bool(hits),
    }


def evaluate(cases: list[dict[str, Any]], k: int, use_reranking: bool) -> list[dict[str, Any]]:
    return [evaluate_case(case, k=k, use_reranking=use_reranking) for case in cases]


def print_results(label: str, results: list[dict[str, Any]], k: int) -> None:
    print(f"\n{label}")
    print("=" * len(label))

    for index, result in enumerate(results, start=1):
        print(f"{index}. {result['question']}")
        print(f"   Expected: {', '.join(result['expected'])}")
        print(f"   Retrieved: {', '.join(result['retrieved'])}")
        print(f"   Precision@{k}: {result['precision']:.2f}")
        print(f"   Recall@{k}: {result['recall']:.2f}")
        print(f"   Hit: {'yes' if result['hit'] else 'no'}")

    if not results:
        print("No evaluation cases found.")
        return

    avg_precision = sum(result["precision"] for result in results) / len(results)
    avg_recall = sum(result["recall"] for result in results) / len(results)
    hit_rate = sum(1 for result in results if result["hit"]) / len(results)

    print("\nAverages")
    print(f"Precision@{k}: {avg_precision:.2f}")
    print(f"Recall@{k}: {avg_recall:.2f}")
    print(f"Hit rate: {hit_rate:.2f}")


def average_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {"precision": 0, "recall": 0, "hit_rate": 0}

    return {
        "precision": sum(result["precision"] for result in results) / len(results),
        "recall": sum(result["recall"] for result in results) / len(results),
        "hit_rate": sum(1 for result in results if result["hit"]) / len(results),
    }


def print_comparison(enabled_results: list[dict[str, Any]], disabled_results: list[dict[str, Any]], k: int) -> None:
    enabled = average_metrics(enabled_results)
    disabled = average_metrics(disabled_results)
    precision_delta = enabled["precision"] - disabled["precision"]
    recall_delta = enabled["recall"] - disabled["recall"]

    print("\nReranking comparison")
    print("====================")
    print(f"Precision@{k} delta: {precision_delta:+.2f}")
    print(f"Recall@{k} delta: {recall_delta:+.2f}")
    if precision_delta > 0 or recall_delta > 0:
        print("Reranking improved retrieval on this eval set.")
    elif precision_delta == 0 and recall_delta == 0:
        print("Reranking matched non-reranked retrieval on this eval set.")
    else:
        print("Reranking reduced retrieval quality on this eval set.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality with Precision@K and Recall@K.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_EVAL_PATH, help="Path to retrieval eval cases JSON.")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Number of retrieved chunks to evaluate.")
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    cases = load_cases(args.cases)
    print(f"Loaded {len(cases)} evaluation cases from {args.cases}.")
    print(f"Using K={args.k}.")

    enabled_results = evaluate(cases, k=args.k, use_reranking=True)
    disabled_results = evaluate(cases, k=args.k, use_reranking=False)

    print_results("Reranking enabled", enabled_results, k=args.k)
    print_results("Reranking disabled", disabled_results, k=args.k)
    print_comparison(enabled_results, disabled_results, k=args.k)
