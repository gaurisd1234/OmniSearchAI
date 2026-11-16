"""Shared latency and ranking-metric comparison for retrieval experiments."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from omnisearch_ai.evaluation.retrieval_metrics import evaluate_at_k, macro_average


@dataclass(frozen=True, slots=True)
class MethodEvaluation:
    method: str
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    mean_query_latency_ms: float
    median_query_latency_ms: float
    per_query: tuple[dict[str, Any], ...]


def compare_retrievers(
    retrievers: Mapping[str, Any],
    queries: Mapping[str, str],
    relevance: Mapping[str, Sequence[str]],
    *,
    k: int = 5,
    repeats: int = 3,
    warmup: bool = True,
) -> tuple[MethodEvaluation, ...]:
    """Compare any compatible retrievers against document-level qrels.

    Query keys must match between ``queries`` and ``relevance``. Latency times
    each search call and excludes model loading, indexing, and startup.
    """
    if k <= 0 or repeats <= 0:
        raise ValueError("k and repeats must be positive")
    query_ids = set(queries)
    if query_ids != set(relevance):
        raise ValueError("query and relevance mappings must have identical keys")
    reports = []
    for name, retriever in retrievers.items():
        metric_rows = []
        per_query = []
        latencies = []
        for query_id, query in queries.items():
            if warmup:
                retriever.search(query, top_k=k)
            durations = []
            results = []
            for _ in range(repeats):
                start = time.perf_counter()
                results = retriever.search(query, top_k=k)
                durations.append((time.perf_counter() - start) * 1000.0)
            metrics = evaluate_at_k(results, relevance[query_id], k=k)
            metric_rows.append(metrics)
            latency = statistics.median(durations)
            latencies.append(latency)
            per_query.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "relevant_document_ids": list(relevance[query_id]),
                    "metrics": metrics.to_dict(),
                    "median_latency_ms": latency,
                    "retrieved_document_ids": [item.document_id for item in results],
                }
            )
        aggregate = macro_average(metric_rows)
        reports.append(
            MethodEvaluation(
                method=name,
                precision_at_k=aggregate.precision_at_k,
                recall_at_k=aggregate.recall_at_k,
                mrr=aggregate.mrr,
                ndcg_at_k=aggregate.ndcg_at_k,
                mean_query_latency_ms=statistics.mean(latencies) if latencies else 0.0,
                median_query_latency_ms=statistics.median(latencies) if latencies else 0.0,
                per_query=tuple(per_query),
            )
        )
    return tuple(reports)


def write_evaluation_report(
    reports: Sequence[MethodEvaluation], path: str | Path, *, metadata: dict[str, Any]
) -> None:
    """Persist configuration, judgments provenance, and observed results."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"metadata": metadata, "methods": [asdict(report) for report in reports]},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
