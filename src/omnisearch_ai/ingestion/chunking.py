"""Configurable character-based chunking that never crosses page boundaries."""

from __future__ import annotations

from dataclasses import dataclass

from omnisearch_ai.ingestion.models import DocumentChunk, PageContent


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Chunk size and overlap are measured in Unicode characters."""

    chunk_size: int = 1000
    overlap: int = 100

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError("overlap must be >= 0 and smaller than chunk_size")


def chunk_pages(
    pages: tuple[PageContent, ...], document_id: str, config: ChunkingConfig
) -> tuple[DocumentChunk, ...]:
    """Split each page independently and retain character offsets and refs."""
    chunks: list[DocumentChunk] = []
    stride = config.chunk_size - config.overlap
    for page in pages:
        text = page.text
        start = 0
        while start < len(text):
            end = min(start + config.chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                index = len(chunks)
                ref = f"{document_id}#page={page.page_number}&chunk={index}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{document_id}:chunk:{index}",
                        document_id=document_id,
                        text=chunk_text,
                        page_number=page.page_number,
                        chunk_index=index,
                        start_offset=start,
                        end_offset=end,
                        reference=ref,
                    )
                )
            if end == len(text):
                break
            start += stride
    return tuple(chunks)
