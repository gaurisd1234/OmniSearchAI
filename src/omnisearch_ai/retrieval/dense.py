"""Independent dense semantic retriever backed by the persistent Chroma index."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omnisearch_ai.config import load_config
from omnisearch_ai.retrieval.chroma_index import ChromaVectorIndex
from omnisearch_ai.retrieval.results import RetrievalResult


class DenseRetriever:
    """Embed natural-language queries and search Chroma by cosine similarity."""

    def __init__(
        self,
        index: ChromaVectorIndex | None = None,
        *,
        top_k: int | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        self.top_k = load_config(config_path).retrieval_top_k if top_k is None else top_k
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        self.index = index if index is not None else ChromaVectorIndex(config_path=config_path)
        self._owns_index = index is None

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Return nearest chunks; scores are cosine similarities."""
        limit = self.top_k if top_k is None else top_k
        if limit <= 0:
            raise ValueError("top_k must be positive")
        if not query.strip():
            return []
        hits = self.index.query(query, limit=limit, where=where)
        results: list[RetrievalResult] = []
        for hit in hits:
            metadata = dict(hit.metadata)
            source = str(metadata.get("source", ""))
            results.append(
                RetrievalResult(
                    chunk_id=hit.chunk_id,
                    document_id=str(metadata.get("document_id", "")),
                    document_name=str(
                        metadata.get("document_name")
                        or Path(source).name
                        or metadata.get("title", "unknown")
                    ),
                    source=source,
                    page_number=int(metadata.get("page_number", 1)),
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    score=1.0 - hit.distance,
                    content=hit.text,
                    method="dense",
                    metadata=metadata,
                )
            )
        return results

    def close(self) -> None:
        if self._owns_index:
            self.index.close()

    def __enter__(self) -> DenseRetriever:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
