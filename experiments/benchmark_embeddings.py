"""Measure a small, reproducible embedding batch on the configured device.

Run with ``python experiments/benchmark_embeddings.py`` after installing the
``vector`` extra and downloading the configured model. The report measures
embedding calls only; model loading and the warm-up are excluded.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import sentence_transformers
import torch

from omnisearch_ai.config import load_config
from omnisearch_ai.embeddings import SentenceTransformerEmbedder


SAMPLE_TEXTS = [
    "Dense retrieval encodes documents and queries in a shared vector space.",
    "A persistent vector index stores chunk text, embeddings, and source metadata.",
    "Retrieval augmented generation uses selected passages to ground an answer.",
    "Small batches reduce activation memory use on a laptop graphics card.",
    "Normalized vectors make cosine similarity comparisons straightforward.",
    "Reproducible experiments record the model, data split, seed, and metrics.",
    "PDF page numbers help researchers inspect the original source passage.",
    "Incremental indexing avoids recomputing embeddings for unchanged chunks.",
]


def main() -> None:
    config = load_config(ROOT / "configs" / "baseline.toml")
    encoder = SentenceTransformerEmbedder.from_config(
        ROOT / "configs" / "baseline.toml"
    )
    encoder.encode_texts(["Warm-up sentence for the configured encoder."])
    runs = [encoder.encode_texts(SAMPLE_TEXTS) for _ in range(3)]
    times = [run.metrics.elapsed_seconds for run in runs]
    last = runs[-1]
    from huggingface_hub import try_to_load_from_cache

    cached_config = try_to_load_from_cache(config.embedding_model, "config.json")
    model_revision = (
        Path(cached_config).resolve().parent.name
        if isinstance(cached_config, str)
        else None
    )
    report = {
        "model": config.embedding_model,
        "model_revision": model_revision,
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": last.metrics.device,
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if last.metrics.device == "cuda" else None
        ),
        "text_count_per_run": len(SAMPLE_TEXTS),
        "embedding_shape": list(last.vectors.shape),
        "normalization_enabled": config.normalize_embeddings,
        "effective_batch_size": last.metrics.batch_size,
        "elapsed_seconds_each_run": times,
        "median_elapsed_seconds": statistics.median(times),
        "throughput_texts_per_second_median": len(SAMPLE_TEXTS)
        / statistics.median(times),
        "rss_before_bytes": last.metrics.rss_before_bytes,
        "peak_rss_bytes": last.metrics.peak_rss_bytes,
        "rss_delta_bytes": last.metrics.rss_delta_bytes,
        "rss_sampling_interval_ms": last.metrics.rss_sampling_interval_ms,
        "peak_gpu_allocated_bytes": last.metrics.peak_gpu_allocated_bytes,
        "peak_gpu_reserved_bytes": last.metrics.peak_gpu_reserved_bytes,
        "measurement_scope": "warm model; encoding calls only; three repeated runs",
        "rss_note": "peak process RSS is sampled every 10 ms and may miss shorter spikes",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
