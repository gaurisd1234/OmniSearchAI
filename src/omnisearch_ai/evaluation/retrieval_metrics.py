"""Document-level IR metrics for explicit relevance judgments."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from omnisearch_ai.retrieval.results import RetrievalResult


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def evaluate_at_k(
    results: Sequence[RetrievalResult],
    relevant_document_ids: Iterable[str],
    *,
    k: int = 5,
) -> RetrievalMetrics:
    """Compute binary document-level P@K, R@K, reciprocal rank, and nDCG@K.

    Repeated chunks from one document count once. Precision is divided by K,
    even when fewer than K documents were returned.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(relevant_document_ids)
    seen: set[str] = set()
    ranked_ids: list[str] = []
    for result in results:
        if result.document_id not in seen:
            ranked_ids.append(result.document_id)
            seen.add(result.document_id)
        if len(ranked_ids) >= k:
            break
    hits = [1 if doc_id in relevant else 0 for doc_id in ranked_ids[:k]]
    relevant_hits = sum(hits)
    precision = relevant_hits / k
    recall = relevant_hits / len(relevant) if relevant else 0.0
    reciprocal_rank = next(
        (1.0 / rank for rank, hit in enumerate(hits, start=1) if hit), 0.0
    )
    dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, start=1))
    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0
    return RetrievalMetrics(precision, recall, reciprocal_rank, ndcg)


def evaluate_ranked(
    results: Sequence[Any],
    relevance: Mapping[str, int] | Iterable[str],
    *,
    k: int = 5,
    id_field: str = "document_id",
) -> RetrievalMetrics:
    """Evaluate generic text/image ranked results against binary or graded qrels.

    Results without the configured ID field are skipped. Duplicate IDs count
    once. Graded nDCG uses gain ``2**grade - 1``; any positive grade is relevant
    for precision, recall, and MRR.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    grades = ({str(key): int(value) for key, value in relevance.items()}
              if isinstance(relevance, Mapping)
              else {str(key): 1 for key in relevance})
    ranked: list[str] = []
    seen: set[str] = set()
    for result in results:
        value = result.get(id_field) if isinstance(result, Mapping) else getattr(result, id_field, None)
        if value is None or str(value) in seen:
            continue
        seen.add(str(value))
        ranked.append(str(value))
        if len(ranked) == k:
            break
    actual = [grades.get(item, 0) for item in ranked]
    hits = sum(grade > 0 for grade in actual)
    relevant_count = sum(grade > 0 for grade in grades.values())
    reciprocal_rank = next((1.0 / rank for rank, grade in enumerate(actual, 1) if grade > 0), 0.0)
    dcg = sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(actual, 1))
    ideal = sorted((grade for grade in grades.values() if grade > 0), reverse=True)[:k]
    ideal_dcg = sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(ideal, 1))
    return RetrievalMetrics(hits / k, hits / relevant_count if relevant_count else 0.0,
                            reciprocal_rank, dcg / ideal_dcg if ideal_dcg else 0.0)


def macro_average(metrics: Sequence[RetrievalMetrics]) -> RetrievalMetrics:
    """Macro-average per-query metrics; an empty input returns zeros."""
    if not metrics:
        return RetrievalMetrics(0.0, 0.0, 0.0, 0.0)
    count = len(metrics)
    return RetrievalMetrics(
        precision_at_k=sum(item.precision_at_k for item in metrics) / count,
        recall_at_k=sum(item.recall_at_k for item in metrics) / count,
        mrr=sum(item.mrr for item in metrics) / count,
        ndcg_at_k=sum(item.ndcg_at_k for item in metrics) / count,
    )
