"""Stable intermediate records emitted by document ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """A character window from one source page with a traceable reference."""

    chunk_id: str
    document_id: str
    text: str
    page_number: int
    chunk_index: int
    start_offset: int
    end_offset: int
    reference: str


@dataclass(frozen=True, slots=True)
class PageContent:
    """Extracted text for a source page (or logical page for non-PDF files)."""

    page_number: int
    text: str


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    """Consistent extracted document representation shared by all readers."""

    document_id: str
    source: Path
    content_hash: str
    file_type: str
    pages: tuple[PageContent, ...]
    chunks: tuple[DocumentChunk, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Return page text in source order, separated by a newline."""
        return "\n".join(page.text for page in self.pages)
