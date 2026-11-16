"""Independent BM25 keyword retriever over ingestion document chunks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from omnisearch_ai.config import load_config
from omnisearch_ai.ingestion.models import DocumentRecord
from omnisearch_ai.retrieval.filters import matches_where
from omnisearch_ai.retrieval.results import RetrievalResult


def tokenize(text: str) -> list[str]:
    """Lowercase Unicode word tokens; no stemming or stopword removal."""
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


class BM25Retriever:
    """BM25Okapi retriever using ``rank-bm25`` and configurable k1/b/top-k."""

    def __init__(
        self,
        documents: Iterable[DocumentRecord],
        *,
        top_k: int | None = None,
        k1: float | None = None,
        b: float | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        config = load_config(config_path)
        self.top_k = config.retrieval_top_k if top_k is None else top_k
        self.k1 = config.bm25_k1 if k1 is None else k1
        self.b = config.bm25_b if b is None else b
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= self.b <= 1:
            raise ValueError("b must be between zero and one")

        self._records: list[tuple[Any, dict[str, Any], list[str]]] = []
        for document in documents:
            for chunk in document.chunks:
                tokens = tokenize(chunk.text)
                if not tokens:
                    continue
                metadata = dict(document.metadata)
                metadata.update(
                    {
                        "document_id": chunk.document_id,
                        "source": str(document.source),
                        "document_name": document.source.name,
                        "file_type": document.file_type,
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                        "start_offset": chunk.start_offset,
                        "end_offset": chunk.end_offset,
                        "reference": chunk.reference,
                        "content_hash": document.content_hash,
                    }
                )
                self._records.append((chunk, metadata, tokens))

        self._bm25 = None
        if self._records:
            try:
                from rank_bm25 import BM25Okapi
            except ImportError as exc:
                raise RuntimeError("Install rank-bm25 to use keyword retrieval") from exc
            self._bm25 = BM25Okapi(
                [tokens for _, _, tokens in self._records], k1=self.k1, b=self.b
            )

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Return matching chunks ranked by BM25 score (highest first)."""
        limit = self.top_k if top_k is None else top_k
        if limit <= 0:
            raise ValueError("top_k must be positive")
        query_tokens = tokenize(query)
        if not query_tokens or self._bm25 is None:
            return []

        scores = self._bm25.get_scores(query_tokens)
        query_terms = set(query_tokens)
        candidates = [
            i
            for i, (chunk, metadata, tokens) in enumerate(self._records)
            if query_terms.intersection(tokens) and matches_where(metadata, where)
        ]
        candidates.sort(key=lambda i: (-float(scores[i]), i))
        return [self._result(i, float(scores[i])) for i in candidates[:limit]]

    def _result(self, position: int, score: float) -> RetrievalResult:
        chunk, metadata, _ = self._records[position]
        return RetrievalResult(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            document_name=str(metadata["document_name"]),
            source=str(metadata["source"]),
            page_number=int(chunk.page_number),
            chunk_index=int(chunk.chunk_index),
            score=score,
            content=chunk.text,
            method="bm25",
            metadata=metadata,
        )
