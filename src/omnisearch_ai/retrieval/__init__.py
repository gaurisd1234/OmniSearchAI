"""Sparse, dense, and hybrid candidate retrieval and ranking interfaces."""

from omnisearch_ai.retrieval.chroma_index import (
    ChromaVectorIndex,
    IndexingReport,
    SearchHit,
)
from omnisearch_ai.retrieval.bm25 import BM25Retriever
from omnisearch_ai.retrieval.dense import DenseRetriever
from omnisearch_ai.retrieval.hybrid import CrossEncoderReranker, HybridRetriever
from omnisearch_ai.retrieval.results import RetrievalResult

__all__ = [
    "BM25Retriever",
    "ChromaVectorIndex",
    "CrossEncoderReranker",
    "DenseRetriever",
    "HybridRetriever",
    "IndexingReport",
    "RetrievalResult",
    "SearchHit",
]
