"""Small shared metadata filter evaluator for in-memory retrieval."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def matches_where(metadata: Mapping[str, Any], where: Mapping[str, Any] | None) -> bool:
    """Match Chroma-style equality, comparison, membership, and boolean filters."""
    if not where:
        return True
    for key, expected in where.items():
        if key == "$and":
            if not all(matches_where(metadata, clause) for clause in expected):
                return False
            continue
        if key == "$or":
            if not any(matches_where(metadata, clause) for clause in expected):
                return False
            continue
        actual = metadata.get(key)
        if isinstance(expected, Mapping):
            for operator, value in expected.items():
                if operator == "$eq" and actual != value:
                    return False
                if operator == "$ne" and actual == value:
                    return False
                if operator == "$in" and actual not in value:
                    return False
                if operator == "$nin" and actual in value:
                    return False
                if operator in {"$gt", "$gte", "$lt", "$lte"}:
                    if actual is None:
                        return False
                    try:
                        if operator == "$gt" and not actual > value:
                            return False
                        if operator == "$gte" and not actual >= value:
                            return False
                        if operator == "$lt" and not actual < value:
                            return False
                        if operator == "$lte" and not actual <= value:
                            return False
                    except TypeError:
                        return False
                if operator not in {
                    "$eq", "$ne", "$in", "$nin", "$gt", "$gte", "$lt", "$lte"
                }:
                    raise ValueError(f"Unsupported metadata filter operator: {operator}")
        elif actual != expected:
            return False
    return True
