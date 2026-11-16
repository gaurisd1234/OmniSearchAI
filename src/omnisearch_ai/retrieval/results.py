"""Shared result shape for keyword and dense retrieval experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """One ranked chunk; scores are method-specific and higher is better."""

    chunk_id: str
    document_id: str
    document_name: str
    source: str
    page_number: int
    chunk_index: int
    score: float
    content: str
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)
