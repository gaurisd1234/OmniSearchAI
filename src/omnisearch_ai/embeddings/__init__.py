"""Embedding model interfaces, batching, caching, and vector encoding."""

from omnisearch_ai.embeddings.encoder import (
    DEFAULT_MODEL,
    EmbeddingBatch,
    EmbeddingMetrics,
    SentenceTransformerEmbedder,
    embed_texts,
)

__all__ = [
    "DEFAULT_MODEL",
    "EmbeddingBatch",
    "EmbeddingMetrics",
    "SentenceTransformerEmbedder",
    "embed_texts",
]
