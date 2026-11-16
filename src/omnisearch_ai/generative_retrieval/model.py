"""Lazy loading helpers for the small T5 generative retrieval baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omnisearch_ai.config import load_config
from omnisearch_ai.generative_retrieval.vocabulary import DocumentIdVocabulary


@dataclass(slots=True)
class T5Components:
    model: Any
    tokenizer: Any
    vocabulary: DocumentIdVocabulary
    device: str
    model_name: str


def resolve_device(device: str = "auto", *, minimum_free_cuda_bytes: int = 0) -> str:
    """Choose CUDA only when available memory clears an optional safety floor."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("T5 retrieval requires PyTorch; install the generative extra") from exc
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if device == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        if device == "cuda":
            raise RuntimeError("CUDA was requested but is not available")
        return "cpu"
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < minimum_free_cuda_bytes:
        if device == "cuda":
            raise RuntimeError(
                "CUDA has insufficient free memory for the configured T5 workload"
            )
        return "cpu"
    return "cuda"


def load_t5_components(
    vocabulary: DocumentIdVocabulary,
    *,
    model_name: str | None = None,
    checkpoint: str | Path | None = None,
    device: str = "auto",
    config_path: str | Path | None = None,
    minimum_free_cuda_bytes: int = 1_500_000_000,
) -> T5Components:
    """Load tokenizer/model on demand and install the stable document tokens.

    Loading a named model may download its checkpoint. Loading a local
    ``checkpoint`` does not need a Hub model lookup when its files are present.
    """
    try:
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Install omnisearch-ai[generative] to load and fine-tune T5 models"
        ) from exc
    config = load_config(config_path)
    # New document-token embeddings are randomly initialized; seed before
    # resizing so equal model/vocabulary/config runs begin from equal weights.
    torch.manual_seed(config.seed)
    source = str(checkpoint or model_name or config.generative_model)
    selected_device = resolve_device(device, minimum_free_cuda_bytes=minimum_free_cuda_bytes)
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = T5ForConditionalGeneration.from_pretrained(source)
    added_tokens = vocabulary.add_to_tokenizer(tokenizer)
    if added_tokens:
        model.resize_token_embeddings(len(tokenizer))
    if selected_device == "cuda":
        try:
            model.to(selected_device)
        except (RuntimeError, MemoryError) as exc:
            if not _is_cuda_oom(exc) or device == "cuda":
                raise
            torch.cuda.empty_cache()
            model.to("cpu")
            selected_device = "cpu"
    else:
        model.to("cpu")
    return T5Components(
        model=model,
        tokenizer=tokenizer,
        vocabulary=vocabulary,
        device=selected_device,
        model_name=source,
    )


def _is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message and ("cuda" in message or "gpu" in message)
