"""Deterministic query/document pair preparation and leakage-aware splitting."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class QueryDocumentPair:
    query: str
    document_id: str


@dataclass(frozen=True, slots=True)
class PairSplits:
    train: tuple[QueryDocumentPair, ...]
    validation: tuple[QueryDocumentPair, ...]


def create_pairs(
    judgments: Mapping[str, Iterable[str]],
    *,
    known_document_ids: Iterable[str] | None = None,
) -> tuple[QueryDocumentPair, ...]:
    """Expand query qrels into unique pairs and reject IDs outside the corpus."""
    known = set(known_document_ids) if known_document_ids is not None else None
    pairs: set[QueryDocumentPair] = set()
    for query, document_ids in judgments.items():
        query = query.strip()
        if not query:
            raise ValueError("queries must not be empty")
        for document_id in document_ids:
            if known is not None and document_id not in known:
                raise ValueError(f"unknown document ID in judgments: {document_id}")
            pairs.add(QueryDocumentPair(query, str(document_id)))
    return tuple(sorted(pairs, key=lambda pair: (pair.query, pair.document_id)))


def split_pairs(
    pairs: Sequence[QueryDocumentPair],
    *,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> PairSplits:
    """Split by query (never by pair) so one query cannot leak across splits."""
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    unique_queries = sorted({pair.query for pair in pairs})
    if validation_fraction and len(unique_queries) < 2:
        raise ValueError("at least two distinct queries are required for validation")
    shuffled = list(unique_queries)
    random.Random(seed).shuffle(shuffled)
    validation_count = (
        min(len(shuffled) - 1, max(1, round(len(shuffled) * validation_fraction)))
        if validation_fraction
        else 0
    )
    validation_queries = set(shuffled[:validation_count])
    train = tuple(pair for pair in pairs if pair.query not in validation_queries)
    validation = tuple(pair for pair in pairs if pair.query in validation_queries)
    return PairSplits(train=train, validation=validation)


def write_pairs_jsonl(pairs: Sequence[QueryDocumentPair], path: str | Path) -> None:
    """Write stable UTF-8 query/document JSONL for experiment provenance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for pair in pairs:
            stream.write(
                json.dumps(
                    {"query": pair.query, "document_id": pair.document_id},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
