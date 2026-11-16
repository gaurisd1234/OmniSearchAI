"""Evaluate BM25, dense, and RRF hybrid retrieval on labeled sample queries.

This is a small synthetic research smoke evaluation, not a benchmark dataset.
It reports macro-averaged document-level metrics and warm-query latency. Pass
``--rerank`` to additionally measure the optional cross-encoder variant.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import statistics
import sys
import time
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import chromadb
import sentence_transformers
import torch

from omnisearch_ai.config import load_config
from omnisearch_ai.embeddings import SentenceTransformerEmbedder
from omnisearch_ai.evaluation import evaluate_at_k, macro_average
from omnisearch_ai.ingestion.models import DocumentChunk, DocumentRecord, PageContent
from omnisearch_ai.retrieval import BM25Retriever, ChromaVectorIndex, DenseRetriever, HybridRetriever


CORPUS = (
    (
        "bm25_keyword_search.txt",
        "BM25 ranks exact keyword matches through term frequency and inverse document frequency.",
        {"topic": "lexical"},
    ),
    (
        "dense_semantic_search.txt",
        "Semantic vector search finds passages that mean similar things despite different words.",
        {"topic": "semantic"},
    ),
    (
        "vector_deletion.txt",
        "Chroma delete operations remove stale vectors and records from persistent collections.",
        {"topic": "storage"},
    ),
    (
        "gpu_cuda.txt",
        "CUDA executes tensor operations on NVIDIA graphics hardware for GPU acceleration.",
        {"topic": "hardware"},
    ),
    (
        "bm25_probability.txt",
        "BM25 is a probabilistic ranking function that scores query term overlap.",
        {"topic": "lexical"},
    ),
    (
        "embedding_vectors.txt",
        "Dense embeddings represent sentences as numerical vectors for similarity search.",
        {"topic": "semantic"},
    ),
    (
        "persistent_indexing.txt",
        "Persistent Chroma collections store chunk metadata and embeddings across program runs.",
        {"topic": "storage"},
    ),
    (
        "laptop_memory.txt",
        "Laptop temperature and VRAM limits can influence neural model inference speed.",
        {"topic": "hardware"},
    ),
)

QUERY_JUDGMENTS = (
    {
        "query_id": "lexical-1",
        "query": "term frequency inverse document frequency BM25 ranking",
        "relevant_names": {"bm25_keyword_search.txt"},
    },
    {
        "query_id": "semantic-1",
        "query": "find passages that express the same idea",
        "relevant_names": {"dense_semantic_search.txt"},
    },
    {
        "query_id": "storage-1",
        "query": "remove obsolete entries from the stored collection",
        "relevant_names": {"vector_deletion.txt"},
    },
    {
        "query_id": "hardware-1",
        "query": "NVIDIA GPU computing with CUDA",
        "relevant_names": {"gpu_cuda.txt"},
    },
)


def make_record(name: str, text: str, metadata: dict[str, str]) -> DocumentRecord:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    document_id = f"doc_{digest}"
    chunk = DocumentChunk(
        chunk_id=f"{document_id}:chunk:0",
        document_id=document_id,
        text=text,
        page_number=1,
        chunk_index=0,
        start_offset=0,
        end_offset=len(text),
        reference=f"{document_id}#page=1&chunk=0",
    )
    return DocumentRecord(
        document_id=document_id,
        source=Path(name),
        content_hash=digest,
        file_type="txt",
        pages=(PageContent(1, text),),
        chunks=(chunk,),
        metadata=metadata,
    )


def result_summary(results) -> list[dict[str, object]]:
    return [
        {
            "document_name": item.document_name,
            "document_id": item.document_id,
            "page_number": item.page_number,
            "score": round(item.score, 6),
            "content": item.content,
        }
        for item in results
    ]


def measure_method(name, retriever, judgments, relevant_by_name, k, repeats):
    for judgment in judgments:
        retriever.search(judgment["query"], top_k=k)
    per_query = []
    metric_rows = []
    latency_rows = []
    for judgment in judgments:
        durations = []
        results = []
        for _ in range(repeats):
            started = time.perf_counter()
            results = retriever.search(judgment["query"], top_k=k)
            durations.append(time.perf_counter() - started)
        relevant_ids = {
            relevant_by_name[doc_name] for doc_name in judgment["relevant_names"]
        }
        metrics = evaluate_at_k(results, relevant_ids, k=k)
        metric_rows.append(metrics)
        latency_ms = statistics.median(durations) * 1000.0
        latency_rows.append(latency_ms)
        per_query.append(
            {
                "query_id": judgment["query_id"],
                "query": judgment["query"],
                "relevant_document_names": sorted(judgment["relevant_names"]),
                "metrics": metrics.to_dict(),
                "median_latency_ms": latency_ms,
                "results": result_summary(results),
            }
        )
    return {
        "method": name,
        "macro_metrics": macro_average(metric_rows).to_dict(),
        "mean_query_latency_ms": statistics.mean(latency_rows),
        "median_query_latency_ms": statistics.median(latency_rows),
        "per_query": per_query,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="also evaluate the configured cross-encoder on fused candidates",
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")

    config_path = ROOT / "configs" / "baseline.toml"
    config = load_config(config_path)
    documents = [make_record(*item) for item in CORPUS]
    relevant_by_name = {doc.source.name: doc.document_id for doc in documents}
    bm25 = BM25Retriever(documents, config_path=config_path)
    encoder = SentenceTransformerEmbedder.from_config(config_path)
    systems = [
        (
            "bm25",
            bm25,
        ),
    ]

    with tempfile.TemporaryDirectory(prefix="omnisearch-hybrid-eval-") as directory:
        with ChromaVectorIndex(Path(directory) / "chroma", embedder=encoder) as index:
            index.index_documents(documents)
            dense = DenseRetriever(index, top_k=config.retrieval_top_k)
            hybrid = HybridRetriever(
                bm25,
                dense,
                top_k=config.retrieval_top_k,
                candidate_k=config.rrf_candidate_k,
                rrf_k=config.rrf_k,
                bm25_weight=config.bm25_rrf_weight,
                dense_weight=config.dense_rrf_weight,
            )
            systems = [
                ("bm25", bm25),
                ("dense", dense),
                ("hybrid_rrf", hybrid),
            ]
            if args.rerank:
                reranked = HybridRetriever(
                    bm25,
                    dense,
                    top_k=config.retrieval_top_k,
                    candidate_k=config.rrf_candidate_k,
                    rrf_k=config.rrf_k,
                    bm25_weight=config.bm25_rrf_weight,
                    dense_weight=config.dense_rrf_weight,
                    use_cross_encoder=True,
                    cross_encoder_model=config.cross_encoder_model,
                    cross_encoder_batch_size=config.cross_encoder_batch_size,
                )
                systems.append(("hybrid_cross_encoder", reranked))

            reports = [
                measure_method(
                    name,
                    retriever,
                    QUERY_JUDGMENTS,
                    relevant_by_name,
                    k=3,
                    repeats=args.repeats,
                )
                for name, retriever in systems
            ]

    print(
        json.dumps(
            {
                "corpus": "8 synthetic one-chunk documents",
                "qrels": "4 hand-labeled queries; one relevant document per query",
                "top_k": 3,
                "repeats_per_query": args.repeats,
                "warmup": "one unmeasured call per query and method",
                "model": config.embedding_model,
                "torch_version": torch.__version__,
                "sentence_transformers_version": sentence_transformers.__version__,
                "chromadb_version": chromadb.__version__,
                "rank_bm25_version": version("rank-bm25"),
                "dense_device": encoder.device,
                "rrf": {
                    "k": config.rrf_k,
                    "candidate_k": config.rrf_candidate_k,
                    "bm25_weight": config.bm25_rrf_weight,
                    "dense_weight": config.dense_rrf_weight,
                },
                "cross_encoder_evaluated": args.rerank,
                "warning": "Small synthetic evaluation; results are not general effectiveness claims.",
                "score_note": "BM25, RRF, and cross-encoder scores have different scales.",
                "systems": reports,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
