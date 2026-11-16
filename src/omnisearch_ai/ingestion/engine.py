"""High-level file ingestion and deterministic duplicate detection."""

from __future__ import annotations

import hashlib
from pathlib import Path

from omnisearch_ai.config import load_config
from omnisearch_ai.ingestion.chunking import ChunkingConfig, chunk_pages
from omnisearch_ai.ingestion.models import DocumentRecord
from omnisearch_ai.ingestion.readers import READERS


class IngestionError(ValueError):
    """Base error for unsupported, unreadable, or empty source documents."""


class EmptyDocumentError(IngestionError):
    """Raised when a file has no extractable non-whitespace text."""


class UnsupportedFormatError(IngestionError):
    """Raised when a file extension has no registered reader."""


class DocumentIngestor:
    """Extract PDF, TXT, and DOCX files into a shared record format.

    Document IDs are content-addressed SHA-256 identifiers. Re-ingesting
    identical bytes through this instance returns the first record, preventing
    duplicate records in a corpus while preserving the original source path.
    """

    def __init__(
        self,
        chunking: ChunkingConfig | None = None,
        *,
        config_path: str | Path | None = None,
    ) -> None:
        if chunking is None:
            settings = load_config(config_path)
            chunking = ChunkingConfig(settings.chunk_size, settings.chunk_overlap)
        self.chunking = chunking
        self._documents: dict[str, DocumentRecord] = {}

    def ingest(self, path: str | Path) -> DocumentRecord:
        source = Path(path)
        suffix = source.suffix.lower()
        reader = READERS.get(suffix)
        if reader is None:
            raise UnsupportedFormatError(f"Unsupported document format: {suffix or '(none)'}")
        if not source.is_file():
            raise IngestionError(f"Source file does not exist or is not a file: {source}")
        try:
            raw = source.read_bytes()
        except OSError as exc:
            raise IngestionError(f"Could not read source file {source}: {exc}") from exc
        if not raw:
            raise EmptyDocumentError(f"Document is empty: {source}")
        try:
            extracted = reader(source, raw)
        except Exception as exc:
            raise IngestionError(f"Failed to ingest {source}: {exc}") from exc
        if not any(page.text.strip() for page in extracted.pages):
            raise EmptyDocumentError(f"Document has no extractable text: {source}")

        # Hash normalized extracted text rather than container bytes so a copied
        # document with different file metadata still resolves to the same ID.
        canonical_text = "\n\f\n".join(page.text for page in extracted.pages)
        content_hash = hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()
        document_id = f"doc_{content_hash}"
        existing = self._documents.get(document_id)
        if existing is not None:
            return existing

        metadata = dict(extracted.metadata)
        metadata.update(
            {
                "source": str(source.resolve()),
                "content_hash": content_hash,
                "page_count": len(extracted.pages),
            }
        )
        chunks = chunk_pages(extracted.pages, document_id, self.chunking)
        record = DocumentRecord(
            document_id=document_id,
            source=source.resolve(),
            content_hash=content_hash,
            file_type=suffix[1:],
            pages=extracted.pages,
            chunks=chunks,
            metadata=metadata,
        )
        self._documents[document_id] = record
        return record

    def ingest_many(self, paths: list[str | Path]) -> list[DocumentRecord]:
        """Ingest paths in order and include each content ID only once."""
        results: list[DocumentRecord] = []
        seen: set[str] = set()
        for path in paths:
            document = self.ingest(path)
            if document.document_id not in seen:
                results.append(document)
                seen.add(document.document_id)
        return results
