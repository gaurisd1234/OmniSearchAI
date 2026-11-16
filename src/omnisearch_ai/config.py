"""Central application and experiment configuration.

Configuration values are loaded from TOML, then overridden by OMNISEARCH_*
environment variables. Keeping this module dependency-free makes experiments
easy to reproduce in a clean Python 3.12 environment.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class AppConfig:
    seed: int = 42
    log_level: str = "INFO"
    data_dir: Path = Path("datasets")
    device: str = "auto"
    chunk_size: int = 1000
    chunk_overlap: int = 100
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cuda_batch_size: int = 8
    cpu_batch_size: int = 32
    embedding_max_seq_length: int = 256
    normalize_embeddings: bool = True
    vector_store_dir: Path = Path("vector_store/chroma")
    vector_collection: str = "omnisearch_chunks"
    retrieval_top_k: int = 5
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    rrf_k: int = 60
    rrf_candidate_k: int = 20
    bm25_rrf_weight: float = 1.0
    dense_rrf_weight: float = 1.0
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_batch_size: int = 8
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b-instruct-q4_K_M"
    ollama_timeout_seconds: float = 120.0
    ollama_context_window: int = 4096
    ollama_max_tokens: int = 512
    ollama_temperature: float = 0.1
    rag_max_context_chars: int = 12000
    rag_max_context_chunks: int = 8
    clip_model: str = "openai/clip-vit-base-patch32"
    clip_cuda_batch_size: int = 2
    clip_cpu_batch_size: int = 4
    image_collection: str = "omnisearch_images"
    generative_model: str = "google/flan-t5-small"
    generative_epochs: int = 3
    generative_batch_size: int = 1
    generative_gradient_accumulation: int = 8
    generative_learning_rate: float = 5e-5
    generative_max_input_tokens: int = 256
    generative_max_output_tokens: int = 4
    generative_validation_fraction: float = 0.2
    generative_beam_candidates: int = 20


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load TOML settings and apply supported environment overrides."""
    raw: dict[str, Any] = {}
    if path is not None:
        with Path(path).open("rb") as config_file:
            raw = tomllib.load(config_file)
    section = raw.get("omnisearch", raw)

    seed = os.getenv("OMNISEARCH_SEED", section.get("seed", 42))
    log_level = os.getenv("OMNISEARCH_LOG_LEVEL", section.get("log_level", "INFO"))
    data_dir = os.getenv("OMNISEARCH_DATA_DIR", section.get("data_dir", "datasets"))
    device = os.getenv("OMNISEARCH_DEVICE", section.get("device", "auto"))
    ingestion = raw.get("ingestion", {})
    chunk_size = os.getenv("OMNISEARCH_CHUNK_SIZE", ingestion.get("chunk_size", 1000))
    chunk_overlap = os.getenv(
        "OMNISEARCH_CHUNK_OVERLAP", ingestion.get("chunk_overlap", 100)
    )
    embeddings = raw.get("embeddings", {})
    embedding_model = os.getenv(
        "OMNISEARCH_EMBEDDING_MODEL",
        embeddings.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
    )
    cuda_batch_size = os.getenv(
        "OMNISEARCH_CUDA_BATCH_SIZE", embeddings.get("cuda_batch_size", 8)
    )
    cpu_batch_size = os.getenv(
        "OMNISEARCH_CPU_BATCH_SIZE", embeddings.get("cpu_batch_size", 32)
    )
    max_seq_length = os.getenv(
        "OMNISEARCH_EMBEDDING_MAX_SEQ_LENGTH", embeddings.get("max_seq_length", 256)
    )
    normalize = os.getenv(
        "OMNISEARCH_NORMALIZE_EMBEDDINGS",
        str(embeddings.get("normalize", True)),
    )
    vector_store = raw.get("vector_store", {})
    vector_store_dir = os.getenv(
        "OMNISEARCH_VECTOR_STORE_DIR", vector_store.get("path", "vector_store/chroma")
    )
    vector_collection = os.getenv(
        "OMNISEARCH_VECTOR_COLLECTION",
        vector_store.get("collection", "omnisearch_chunks"),
    )
    retrieval = raw.get("retrieval", {})
    retrieval_top_k = os.getenv(
        "OMNISEARCH_RETRIEVAL_TOP_K", retrieval.get("top_k", 5)
    )
    bm25_k1 = os.getenv("OMNISEARCH_BM25_K1", retrieval.get("bm25_k1", 1.5))
    bm25_b = os.getenv("OMNISEARCH_BM25_B", retrieval.get("bm25_b", 0.75))
    rrf_k = os.getenv("OMNISEARCH_RRF_K", retrieval.get("rrf_k", 60))
    rrf_candidate_k = os.getenv(
        "OMNISEARCH_RRF_CANDIDATE_K", retrieval.get("rrf_candidate_k", 20)
    )
    bm25_rrf_weight = os.getenv(
        "OMNISEARCH_BM25_RRF_WEIGHT", retrieval.get("bm25_rrf_weight", 1.0)
    )
    dense_rrf_weight = os.getenv(
        "OMNISEARCH_DENSE_RRF_WEIGHT", retrieval.get("dense_rrf_weight", 1.0)
    )
    cross_encoder_model = os.getenv(
        "OMNISEARCH_CROSS_ENCODER_MODEL",
        retrieval.get("cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    )
    cross_encoder_batch_size = os.getenv(
        "OMNISEARCH_CROSS_ENCODER_BATCH_SIZE",
        retrieval.get("cross_encoder_batch_size", 8),
    )
    llm = raw.get("llm", {})
    ollama_base_url = os.getenv(
        "OMNISEARCH_OLLAMA_BASE_URL", llm.get("ollama_base_url", "http://localhost:11434")
    )
    ollama_model = os.getenv(
        "OMNISEARCH_OLLAMA_MODEL",
        llm.get("ollama_model", "qwen2.5:3b-instruct-q4_K_M"),
    )
    ollama_timeout = os.getenv(
        "OMNISEARCH_OLLAMA_TIMEOUT_SECONDS", llm.get("timeout_seconds", 120.0)
    )
    ollama_context = os.getenv(
        "OMNISEARCH_OLLAMA_CONTEXT_WINDOW", llm.get("context_window", 4096)
    )
    ollama_max_tokens = os.getenv(
        "OMNISEARCH_OLLAMA_MAX_TOKENS", llm.get("max_tokens", 512)
    )
    ollama_temperature = os.getenv(
        "OMNISEARCH_OLLAMA_TEMPERATURE", llm.get("temperature", 0.1)
    )
    rag = raw.get("rag", {})
    rag_max_context_chars = os.getenv(
        "OMNISEARCH_RAG_MAX_CONTEXT_CHARS", rag.get("max_context_chars", 12000)
    )
    rag_max_context_chunks = os.getenv(
        "OMNISEARCH_RAG_MAX_CONTEXT_CHUNKS", rag.get("max_context_chunks", 8)
    )
    multimodal = raw.get("multimodal", {})
    clip_model = os.getenv(
        "OMNISEARCH_CLIP_MODEL",
        multimodal.get("clip_model", "openai/clip-vit-base-patch32"),
    )
    clip_cuda_batch_size = os.getenv(
        "OMNISEARCH_CLIP_CUDA_BATCH_SIZE", multimodal.get("cuda_batch_size", 2)
    )
    clip_cpu_batch_size = os.getenv(
        "OMNISEARCH_CLIP_CPU_BATCH_SIZE", multimodal.get("cpu_batch_size", 4)
    )
    image_collection = os.getenv(
        "OMNISEARCH_IMAGE_COLLECTION",
        multimodal.get("image_collection", "omnisearch_images"),
    )
    generative = raw.get("generative_retrieval", {})
    generative_model = os.getenv(
        "OMNISEARCH_GENERATIVE_MODEL",
        generative.get("model", "google/flan-t5-small"),
    )
    generative_epochs = os.getenv(
        "OMNISEARCH_GENERATIVE_EPOCHS", generative.get("epochs", 3)
    )
    generative_batch_size = os.getenv(
        "OMNISEARCH_GENERATIVE_BATCH_SIZE", generative.get("batch_size", 1)
    )
    generative_gradient_accumulation = os.getenv(
        "OMNISEARCH_GENERATIVE_GRADIENT_ACCUMULATION",
        generative.get("gradient_accumulation", 8),
    )
    generative_learning_rate = os.getenv(
        "OMNISEARCH_GENERATIVE_LEARNING_RATE",
        generative.get("learning_rate", 5e-5),
    )
    generative_max_input_tokens = os.getenv(
        "OMNISEARCH_GENERATIVE_MAX_INPUT_TOKENS",
        generative.get("max_input_tokens", 256),
    )
    generative_max_output_tokens = os.getenv(
        "OMNISEARCH_GENERATIVE_MAX_OUTPUT_TOKENS",
        generative.get("max_output_tokens", 4),
    )
    generative_validation_fraction = os.getenv(
        "OMNISEARCH_GENERATIVE_VALIDATION_FRACTION",
        generative.get("validation_fraction", 0.2),
    )
    generative_beam_candidates = os.getenv(
        "OMNISEARCH_GENERATIVE_BEAM_CANDIDATES",
        generative.get("beam_candidates", 20),
    )
    normalized_bool = str(normalize).strip().lower()
    if normalized_bool not in {"true", "false", "1", "0", "yes", "no"}:
        raise ValueError("normalize_embeddings must be a boolean value")
    return AppConfig(
        seed=int(seed),
        log_level=str(log_level).upper(),
        data_dir=Path(data_dir).expanduser(),
        device=str(device),
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
        embedding_model=str(embedding_model),
        cuda_batch_size=int(cuda_batch_size),
        cpu_batch_size=int(cpu_batch_size),
        embedding_max_seq_length=int(max_seq_length),
        normalize_embeddings=normalized_bool in {"true", "1", "yes"},
        vector_store_dir=Path(vector_store_dir).expanduser(),
        vector_collection=str(vector_collection),
        retrieval_top_k=int(retrieval_top_k),
        bm25_k1=float(bm25_k1),
        bm25_b=float(bm25_b),
        rrf_k=int(rrf_k),
        rrf_candidate_k=int(rrf_candidate_k),
        bm25_rrf_weight=float(bm25_rrf_weight),
        dense_rrf_weight=float(dense_rrf_weight),
        cross_encoder_model=str(cross_encoder_model),
        cross_encoder_batch_size=int(cross_encoder_batch_size),
        ollama_base_url=str(ollama_base_url).rstrip("/"),
        ollama_model=str(ollama_model),
        ollama_timeout_seconds=float(ollama_timeout),
        ollama_context_window=int(ollama_context),
        ollama_max_tokens=int(ollama_max_tokens),
        ollama_temperature=float(ollama_temperature),
        rag_max_context_chars=int(rag_max_context_chars),
        rag_max_context_chunks=int(rag_max_context_chunks),
        clip_model=str(clip_model),
        clip_cuda_batch_size=int(clip_cuda_batch_size),
        clip_cpu_batch_size=int(clip_cpu_batch_size),
        image_collection=str(image_collection),
        generative_model=str(generative_model),
        generative_epochs=int(generative_epochs),
        generative_batch_size=int(generative_batch_size),
        generative_gradient_accumulation=int(generative_gradient_accumulation),
        generative_learning_rate=float(generative_learning_rate),
        generative_max_input_tokens=int(generative_max_input_tokens),
        generative_max_output_tokens=int(generative_max_output_tokens),
        generative_validation_fraction=float(generative_validation_fraction),
        generative_beam_candidates=int(generative_beam_candidates),
    )
