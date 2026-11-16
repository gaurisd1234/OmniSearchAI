"""Reciprocal Rank Fusion and optional cross-encoder reranking."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol, Sequence

from omnisearch_ai.config import load_config
from omnisearch_ai.retrieval.results import RetrievalResult


class Retriever(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]: ...


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: Sequence[RetrievalResult]
    ) -> list[RetrievalResult]: ...


class CrossEncoderReranker:
    """Optional Sentence Transformers cross-encoder for a fused candidate set."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        device: str = "auto",
        batch_size: int = 8,
        model: Any | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
        if model is None:
            import torch
            from sentence_transformers import CrossEncoder

            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            elif device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested for cross-encoder but is unavailable")
            model = CrossEncoder(model_name, device=device)
        self.model = model
        self.model_name = model_name
        self.batch_size = batch_size

    def rerank(
        self, query: str, candidates: Sequence[RetrievalResult]
    ) -> list[RetrievalResult]:
        if not candidates:
            return []
        pairs = [(query, result.content) for result in candidates]
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        reranked = []
        for result, score in zip(candidates, scores):
            metadata = dict(result.metadata)
            metadata["rrf_score"] = float(result.score)
            metadata["cross_encoder_score"] = float(score)
            reranked.append(
                replace(
                    result,
                    score=float(score),
                    method="hybrid_reranked",
                    metadata=metadata,
                )
            )
        return sorted(reranked, key=lambda result: -result.score)


class HybridRetriever:
    """Fuse independent BM25 and dense rankings with weighted RRF."""

    def __init__(
        self,
        bm25_retriever: Retriever,
        dense_retriever: Retriever,
        *,
        top_k: int | None = None,
        candidate_k: int | None = None,
        rrf_k: int | None = None,
        bm25_weight: float | None = None,
        dense_weight: float | None = None,
        reranker: Reranker | None = None,
        use_cross_encoder: bool = False,
        cross_encoder_model: str | None = None,
        cross_encoder_device: str = "auto",
        cross_encoder_batch_size: int | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        config = load_config(config_path)
        self.top_k = config.retrieval_top_k if top_k is None else top_k
        self.candidate_k = config.rrf_candidate_k if candidate_k is None else candidate_k
        self.rrf_k = config.rrf_k if rrf_k is None else rrf_k
        self.bm25_weight = config.bm25_rrf_weight if bm25_weight is None else bm25_weight
        self.dense_weight = config.dense_rrf_weight if dense_weight is None else dense_weight
        if self.top_k <= 0 or self.candidate_k <= 0:
            raise ValueError("top_k and candidate_k must be positive")
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if self.bm25_weight < 0 or self.dense_weight < 0:
            raise ValueError("RRF method weights cannot be negative")
        if self.bm25_weight + self.dense_weight == 0:
            raise ValueError("At least one RRF method weight must be positive")
        if reranker is not None and use_cross_encoder:
            raise ValueError("Pass a reranker or request the default cross-encoder, not both")
        if use_cross_encoder:
            model_name = cross_encoder_model or config.cross_encoder_model
            reranker = CrossEncoderReranker(
                model_name,
                device=cross_encoder_device,
                batch_size=(
                    config.cross_encoder_batch_size
                    if cross_encoder_batch_size is None
                    else cross_encoder_batch_size
                ),
            )
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.reranker = reranker

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        candidate_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve from both methods, fuse ranks, then optionally rerank."""
        final_k = self.top_k if top_k is None else top_k
        candidates_limit = self.candidate_k if candidate_k is None else candidate_k
        if final_k <= 0 or candidates_limit <= 0:
            raise ValueError("top_k and candidate_k must be positive")

        bm25_results = self.bm25_retriever.search(
            query, top_k=candidates_limit, where=where
        )
        dense_results = self.dense_retriever.search(
            query, top_k=candidates_limit, where=where
        )
        fused_scores: dict[str, float] = {}
        fused_results: dict[str, RetrievalResult] = {}
        ranks: dict[str, dict[str, int]] = {}
        for method, results, weight in (
            ("bm25", bm25_results, self.bm25_weight),
            ("dense", dense_results, self.dense_weight),
        ):
            seen: set[str] = set()
            for rank, result in enumerate(results, start=1):
                if result.chunk_id in seen:
                    continue
                seen.add(result.chunk_id)
                fused_scores[result.chunk_id] = fused_scores.get(result.chunk_id, 0.0) + (
                    weight / (self.rrf_k + rank)
                )
                fused_results.setdefault(result.chunk_id, result)
                ranks.setdefault(result.chunk_id, {})[method] = rank

        fused = []
        for chunk_id, result in fused_results.items():
            metadata = dict(result.metadata)
            metadata["rrf_bm25_rank"] = ranks[chunk_id].get("bm25", 0)
            metadata["rrf_dense_rank"] = ranks[chunk_id].get("dense", 0)
            metadata["rrf_score"] = fused_scores[chunk_id]
            fused.append(
                replace(
                    result,
                    score=fused_scores[chunk_id],
                    method="hybrid",
                    metadata=metadata,
                )
            )
        fused.sort(key=lambda result: (-result.score, result.chunk_id))
        if self.reranker is not None:
            fused = self.reranker.rerank(query, fused)
        return fused[:final_k]
