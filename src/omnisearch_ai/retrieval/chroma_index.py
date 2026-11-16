"""Persistent Chroma vector index for ingestion document chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from omnisearch_ai.config import load_config
from omnisearch_ai.embeddings.encoder import (
    EmbeddingMetrics,
    SentenceTransformerEmbedder,
)
from omnisearch_ai.ingestion.models import DocumentChunk, DocumentRecord


@dataclass(frozen=True, slots=True)
class IndexingReport:
    indexed_count: int
    skipped_count: int
    updated_count: int
    embedding_metrics: tuple[EmbeddingMetrics, ...]


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    distance: float


class ChromaVectorIndex:
    """Persist normalized vectors and chunk provenance in a Chroma collection."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        collection_name: str | None = None,
        embedder: Any | None = None,
        config_path: str | Path | None = None,
        operation_batch_size: int = 128,
    ) -> None:
        if operation_batch_size <= 0:
            raise ValueError("operation_batch_size must be positive")
        config = load_config(config_path)
        self.path = Path(path or config.vector_store_dir).expanduser()
        self.collection_name = collection_name or config.vector_collection
        self.operation_batch_size = operation_batch_size
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "ChromaDB is required for persistent indexing; install omnisearch-ai[vector]"
            ) from exc
        self.embedder = embedder or SentenceTransformerEmbedder.from_config(config_path)
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    def close(self) -> None:
        """Stop the embedded Chroma system and release persistent file handles."""
        self.client.close()

    def __enter__(self) -> ChromaVectorIndex:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _batched(items: list[Any], batch_size: int):
        for start in range(0, len(items), batch_size):
            yield items[start : start + batch_size]

    @staticmethod
    def _metadata(chunk: DocumentChunk, document: DocumentRecord) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "document_id": chunk.document_id,
            "source": str(document.source),
            "document_name": document.source.name,
            "page_number": int(chunk.page_number),
            "chunk_index": int(chunk.chunk_index),
            "start_offset": int(chunk.start_offset),
            "end_offset": int(chunk.end_offset),
            "reference": chunk.reference,
            "file_type": document.file_type,
            "content_hash": document.content_hash,
        }
        for key, value in document.metadata.items():
            if key not in metadata and isinstance(value, (str, int, float, bool)):
                metadata[key] = value
        return metadata

    def index_documents(self, documents: Iterable[DocumentRecord]) -> IndexingReport:
        """Index unseen chunks, skip identical records, and replace changed IDs."""
        pending_by_id: dict[str, tuple[DocumentChunk, DocumentRecord]] = {}
        duplicate_inputs = 0
        for document in documents:
            for chunk in document.chunks:
                if chunk.chunk_id in pending_by_id:
                    duplicate_inputs += 1
                else:
                    pending_by_id[chunk.chunk_id] = (chunk, document)

        indexed = updated = 0
        skipped = duplicate_inputs
        metrics: list[EmbeddingMetrics] = []
        candidates = list(pending_by_id.values())
        for candidate_batch in self._batched(candidates, self.operation_batch_size):
            ids = [chunk.chunk_id for chunk, _ in candidate_batch]
            current = self.collection.get(ids=ids, include=["documents"])
            existing_docs = dict(zip(current["ids"], current.get("documents") or []))
            to_write: list[tuple[DocumentChunk, DocumentRecord]] = []
            for chunk, document in candidate_batch:
                old_text = existing_docs.get(chunk.chunk_id)
                if old_text == chunk.text:
                    skipped += 1
                else:
                    to_write.append((chunk, document))
                    if old_text is not None:
                        updated += 1
            if not to_write:
                continue
            embedding_batch = self.embedder.encode_texts(
                [chunk.text for chunk, _ in to_write]
            )
            metrics.append(embedding_batch.metrics)
            self.collection.upsert(
                ids=[chunk.chunk_id for chunk, _ in to_write],
                documents=[chunk.text for chunk, _ in to_write],
                embeddings=embedding_batch.vectors.tolist(),
                metadatas=[self._metadata(chunk, doc) for chunk, doc in to_write],
            )
            indexed += len(to_write) - sum(
                1 for chunk, _ in to_write if chunk.chunk_id in existing_docs
            )
        return IndexingReport(indexed, skipped, updated, tuple(metrics))

    def delete_document(self, document_id: str) -> int:
        """Delete all stored chunks for one ingestion document ID."""
        result = self.collection.get(
            where={"document_id": document_id}, include=["metadatas"]
        )
        ids = result["ids"]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def query(
        self,
        text: str,
        *,
        limit: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Embed a query and return nearest chunks under cosine distance."""
        if limit <= 0:
            raise ValueError("limit must be positive")
        if not text.strip():
            return []
        if self.collection.count() == 0:
            return []
        query_batch = self.embedder.encode_texts([text])
        result = self.collection.query(
            query_embeddings=query_batch.vectors.tolist(),
            n_results=min(limit, self.collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        texts = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        return [
            SearchHit(chunk_id=chunk_id, text=chunk_text, metadata=metadata, distance=float(distance))
            for chunk_id, chunk_text, metadata, distance in zip(
                ids, texts, metadatas, distances
            )
        ]

    def count(self) -> int:
        return int(self.collection.count())
