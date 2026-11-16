"""Sentence Transformers encoder with CUDA sizing, CPU fallback, and metrics."""

from __future__ import annotations

import importlib
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from omnisearch_ai.config import load_config


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True, slots=True)
class EmbeddingMetrics:
    """Observed cost of one encoding call; memory values are bytes."""

    device: str
    num_texts: int
    batch_size: int
    elapsed_seconds: float
    rss_before_bytes: int
    peak_rss_bytes: int
    rss_delta_bytes: int
    rss_sampling_interval_ms: int = 10
    peak_gpu_allocated_bytes: int = 0
    peak_gpu_reserved_bytes: int = 0
    fallback_reason: str | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    vectors: Any
    metrics: EmbeddingMetrics


class _RSSSampler:
    """Sample process resident memory while model inference is running."""

    def __init__(self, interval: float = 0.01) -> None:
        try:
            self.psutil = importlib.import_module("psutil")
        except ImportError as exc:
            raise RuntimeError("Install omnisearch-ai[vector] to measure memory use") from exc
        self.process = self.psutil.Process()
        self.interval = interval
        self.before = self.process.memory_info().rss
        self.peak = self.before
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.peak = max(self.peak, self.process.memory_info().rss)
            except self.psutil.Error:
                return

    def __enter__(self) -> _RSSSampler:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()
        try:
            self.peak = max(self.peak, self.process.memory_info().rss)
        except self.psutil.Error:
            pass


class SentenceTransformerEmbedder:
    """Encode texts using a Sentence Transformers model.

    CUDA starts with a conservative batch of 8 for a 4 GB GPU. CUDA out-of-
    memory errors reduce that batch geometrically; if batch size 1 still fails,
    the model retries on CPU. Model downloads happen only when this object is
    constructed without an injected model.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str = "auto",
        cuda_batch_size: int = 8,
        cpu_batch_size: int = 32,
        max_seq_length: int = 256,
        normalize_embeddings: bool = True,
        fallback_to_cpu: bool = True,
        model: Any | None = None,
    ) -> None:
        if cuda_batch_size <= 0 or cpu_batch_size <= 0:
            raise ValueError("Batch sizes must be positive")
        if max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        try:
            self._torch = importlib.import_module("torch")
            self._np = importlib.import_module("numpy")
        except ImportError as exc:
            raise RuntimeError(
                "Install omnisearch-ai[vector] and a PyTorch build to encode text"
            ) from exc
        cuda_available = bool(self._torch.cuda.is_available())
        if device == "auto":
            self.device = "cuda" if cuda_available else "cpu"
        elif device == "cuda" and not cuda_available:
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        elif device in {"cpu", "cuda"}:
            self.device = device
        else:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")

        self.model_name = model_name
        self.cuda_batch_size = cuda_batch_size
        self.cpu_batch_size = cpu_batch_size
        self.max_seq_length = max_seq_length
        self.normalize_embeddings = normalize_embeddings
        self.fallback_to_cpu = fallback_to_cpu
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name, device=self.device)
        else:
            model.to(self.device)
        self.model = model
        if hasattr(self.model, "max_seq_length"):
            self.model.max_seq_length = max_seq_length

    @classmethod
    def from_config(
        cls,
        config_path: str | Path | None = None,
        *,
        model: Any | None = None,
    ) -> SentenceTransformerEmbedder:
        """Create an encoder from TOML and ``OMNISEARCH_*`` overrides."""
        config = load_config(config_path)
        return cls(
            config.embedding_model,
            device=config.device,
            cuda_batch_size=config.cuda_batch_size,
            cpu_batch_size=config.cpu_batch_size,
            max_seq_length=config.embedding_max_seq_length,
            normalize_embeddings=config.normalize_embeddings,
            model=model,
        )

    @staticmethod
    def _is_cuda_oom(error: BaseException) -> bool:
        message = str(error).lower()
        return "out of memory" in message and ("cuda" in message or "gpu" in message)

    def _encode(self, texts: Sequence[str], batch_size: int) -> Any:
        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        return self._np.asarray(vectors, dtype=self._np.float32)

    def encode_texts(self, texts: Sequence[str]) -> EmbeddingBatch:
        """Encode texts and return vectors with measured time and memory use."""
        psutil = importlib.import_module("psutil")
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("All embedding inputs must be strings")
        if not texts:
            dimension = int(self.model.get_sentence_embedding_dimension() or 0)
            return EmbeddingBatch(
                vectors=self._np.empty((0, dimension), dtype=self._np.float32),
                metrics=EmbeddingMetrics(
                    device=self.device,
                    num_texts=0,
                    batch_size=self.cuda_batch_size if self.device == "cuda" else self.cpu_batch_size,
                    elapsed_seconds=0.0,
                    rss_before_bytes=psutil.Process().memory_info().rss,
                    peak_rss_bytes=psutil.Process().memory_info().rss,
                    rss_delta_bytes=0,
                ),
            )

        configured_batch = self.cuda_batch_size if self.device == "cuda" else self.cpu_batch_size
        effective_batch = configured_batch
        fallback_reason: str | None = None
        gpu_allocated = 0
        gpu_reserved = 0
        if self.device == "cuda":
            self._torch.cuda.reset_peak_memory_stats()
        timer_start = time.perf_counter()
        with _RSSSampler() as rss:
            while True:
                try:
                    vectors = self._encode(texts, effective_batch)
                    if self.device == "cuda":
                        self._torch.cuda.synchronize()
                    break
                except (RuntimeError, MemoryError) as exc:
                    if self.device != "cuda" or not self._is_cuda_oom(exc):
                        raise
                    try:
                        gpu_allocated = max(
                            gpu_allocated, int(self._torch.cuda.max_memory_allocated())
                        )
                        gpu_reserved = max(
                            gpu_reserved, int(self._torch.cuda.max_memory_reserved())
                        )
                        self._torch.cuda.empty_cache()
                    except Exception:
                        pass
                    if effective_batch > 1:
                        effective_batch = max(1, effective_batch // 2)
                        continue
                    if not self.fallback_to_cpu:
                        raise
                    fallback_reason = str(exc)
                    self.model.to("cpu")
                    self.device = "cpu"
                    effective_batch = self.cpu_batch_size
            if self.device == "cuda":
                gpu_allocated = max(
                    gpu_allocated, int(self._torch.cuda.max_memory_allocated())
                )
                gpu_reserved = max(
                    gpu_reserved, int(self._torch.cuda.max_memory_reserved())
                )
        elapsed = time.perf_counter() - timer_start
        peak_rss = rss.peak
        return EmbeddingBatch(
            vectors=vectors,
            metrics=EmbeddingMetrics(
                device=self.device,
                num_texts=len(texts),
                batch_size=effective_batch,
                elapsed_seconds=elapsed,
                rss_before_bytes=rss.before,
                peak_rss_bytes=peak_rss,
                rss_delta_bytes=max(0, peak_rss - rss.before),
                peak_gpu_allocated_bytes=gpu_allocated,
                peak_gpu_reserved_bytes=gpu_reserved,
                fallback_reason=fallback_reason,
            ),
        )

    def encode_chunks(self, chunks: Sequence[Any]) -> EmbeddingBatch:
        """Convenience wrapper for ingestion ``DocumentChunk`` sequences."""
        return self.encode_texts([chunk.text for chunk in chunks])


def embed_texts(texts: Sequence[str]) -> EmbeddingBatch:
    """Use the project-configured model to encode a batch of texts."""
    return SentenceTransformerEmbedder.from_config().encode_texts(texts)
