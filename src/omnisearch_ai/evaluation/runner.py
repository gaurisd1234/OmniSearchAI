"""Measured, modality-neutral retrieval experiments and reproducible outputs."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from omnisearch_ai.evaluation.datasets import EvaluationDataset
from omnisearch_ai.evaluation.retrieval_metrics import evaluate_ranked


@dataclass(frozen=True, slots=True)
class MethodSpec:
    """An independently callable retrieval method and its experiment settings."""

    name: str
    retriever: Any
    id_field: str = "document_id"
    config: Mapping[str, Any] = field(default_factory=dict)
    embedding_probe: Callable[[str], Any] | None = None


def run_experiment(
    dataset: EvaluationDataset,
    methods: Sequence[MethodSpec],
    output_dir: str | Path,
    *,
    k: int = 5,
    repeats: int = 3,
    warmup: bool = True,
    seed: int = 42,
    config: Mapping[str, Any] | None = None,
    mlflow_tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Run search calls, calculate metrics from their returned rankings, and persist.

    Search latency includes query encoding performed inside ``search``; a
    separate embedding probe is optional and reported separately. Model load,
    index construction, and warmup are excluded from latency. Median over
    ``repeats`` calls is used per query.
    """
    if k < 1 or repeats < 1:
        raise ValueError("k and repeats must be positive")
    if not methods:
        raise ValueError("at least one retrieval method is required")
    if len({method.name for method in methods}) != len(methods):
        raise ValueError("method names must be unique")
    _seed_runtime(seed)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    method_rows: dict[str, list[dict[str, Any]]] = {method.name: [] for method in methods}

    for method in methods:
        for query_id, query in dataset.queries.items():
            if query_id not in dataset.qrels:
                continue
            durations: list[float] = []
            results: Sequence[Any] = ()
            if warmup:
                method.retriever.search(query, top_k=k)
            for _ in range(repeats):
                started = time.perf_counter()
                results = method.retriever.search(query, top_k=k)
                durations.append((time.perf_counter() - started) * 1000.0)
            latency = statistics.median(durations)
            embedding_ms = None
            if method.embedding_probe is not None:
                started = time.perf_counter()
                method.embedding_probe(query)
                embedding_ms = (time.perf_counter() - started) * 1000.0
            metrics = evaluate_ranked(results, dataset.qrels[query_id], k=k, id_field=method.id_field)
            metric_row = {
                "method": method.name, "query_id": query_id, "query": query,
                **metrics.to_dict(), "search_latency_ms": latency,
                "embedding_generation_ms": embedding_ms,
            }
            query_rows.append(metric_row)
            method_rows[method.name].append(metric_row)
            for rank, result in enumerate(results, start=1):
                result_id = _get(result, method.id_field)
                if result_id is None:
                    continue
                raw_rows.append({
                    **metric_row, "rank": rank, "result_id": str(result_id),
                    "relevance_grade": dataset.qrels[query_id].get(str(result_id), 0),
                    "score": _get(result, "score"),
                    "source": _get(result, "source", _get(result, "path", "")),
                    "document_name": _get(result, "document_name", ""),
                    "document_reference": _get(result, "document_reference", ""),
                    "page_number": _get(result, "page_number", ""),
                    "content": _get(result, "content", _get(result, "ocr_text", "")),
                    "metadata_json": json.dumps(_get(result, "metadata", {}), ensure_ascii=False, default=str),
                })

    aggregates = {}
    for method, rows in method_rows.items():
        count = len(rows)
        aggregates[method] = {
            key: (sum(float(row[key]) for row in rows) / count if count else 0.0)
            for key in ("precision_at_k", "recall_at_k", "mrr", "ndcg_at_k", "search_latency_ms")
        }
        observed = [row["embedding_generation_ms"] for row in rows if row["embedding_generation_ms"] is not None]
        aggregates[method]["embedding_generation_ms"] = sum(observed) / len(observed) if observed else None

    _write_csv(root / "raw_results.csv", raw_rows)
    _write_csv(root / "per_query_metrics.csv", query_rows)
    summary = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": asdict(dataset.manifest),
        "dataset_sha256": _dataset_fingerprint(dataset),
        "dataset_counts": {
            "queries": len(dataset.queries),
            "judged_queries": len(dataset.qrels),
            "corpus_items": len(dataset.corpus),
            "judgment_pairs": sum(len(judgments) for judgments in dataset.qrels.values()),
        },
        "seed": seed,
        "k": k,
        "repeats": repeats,
        "warmup": warmup,
        "runtime": _runtime_metadata(),
        "command": sys.argv,
        "configuration": dict(config or {}),
        "methods": {method.name: {"id_field": method.id_field, "config": dict(method.config)} for method in methods},
        "aggregates": aggregates,
        "limitations": [
            "Results describe only the supplied queries, relevance judgments, split, corpus and hardware.",
            "Search latency excludes model loading and index construction; it includes work done inside search.",
            "Embedding generation time is reported only when a separate embedding_probe is configured.",
            "Macro metrics treat positive grades as relevant; graded gains are used only for nDCG.",
        ],
    }
    (root / "experiment_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (root / "experiment_report.md").write_text(_markdown_report(summary), encoding="utf-8")
    try:
        generate_charts(aggregates, root / "charts")
    except ImportError:
        summary["chart_status"] = "not generated: install the research optional dependencies"
        (root / "experiment_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if mlflow_tracking_uri:
        log_mlflow(summary, root, mlflow_tracking_uri)
    return summary


def generate_charts(aggregates: Mapping[str, Mapping[str, Any]], output_dir: str | Path) -> None:
    """Write observed metric and latency charts. Requires matplotlib."""
    import matplotlib.pyplot as plt

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for metric in ("precision_at_k", "recall_at_k", "mrr", "ndcg_at_k", "search_latency_ms", "embedding_generation_ms"):
        methods = [name for name, row in aggregates.items() if row.get(metric) is not None]
        if not methods:
            continue
        values = [aggregates[name][metric] for name in methods]
        fig, axis = plt.subplots(figsize=(max(6, len(methods) * 1.25), 4))
        axis.bar(methods, values)
        axis.set_ylabel(metric)
        axis.set_title(f"Observed {metric}")
        axis.tick_params(axis="x", labelrotation=25)
        fig.tight_layout()
        fig.savefig(root / f"{metric}.png", dpi=150)
        plt.close(fig)


def log_mlflow(summary: Mapping[str, Any], output_dir: str | Path, tracking_uri: str) -> None:
    """Log actual aggregate metrics and raw artifacts to a local/remote MLflow URI."""
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("MLflow tracking requires `pip install -e '.[research]'`") from exc
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("OmniSearch AI retrieval evaluation")
    with mlflow.start_run(run_name=str(summary["run_id"])):
        mlflow.log_param("dataset", summary["dataset"]["name"])
        mlflow.log_param("split", summary["dataset"]["split"])
        mlflow.log_param("dataset_sha256", summary["dataset_sha256"])
        mlflow.log_param("seed", summary["seed"])
        mlflow.log_param("k", summary["k"])
        for method, values in summary["methods"].items():
            for key, value in values["config"].items():
                mlflow.log_param(f"{method}.{key}", str(value))
        for method, metrics in summary["aggregates"].items():
            for key, value in metrics.items():
                if value is not None:
                    mlflow.log_metric(f"{method}.{key}", float(value))
        mlflow.log_artifacts(str(output_dir))


def _get(result: Any, name: str, default: Any = None) -> Any:
    return result.get(name, default) if isinstance(result, Mapping) else getattr(result, name, default)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or ["empty"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _dataset_fingerprint(dataset: EvaluationDataset) -> str:
    payload = json.dumps({"queries": dataset.queries, "qrels": dataset.qrels, "corpus": dataset.corpus}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _runtime_metadata() -> dict[str, Any]:
    packages = ("omnisearch-ai", "torch", "sentence-transformers", "chromadb", "transformers", "rank-bm25", "mlflow", "matplotlib", "Pillow")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    revision = None
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True,
            timeout=2, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    hardware: dict[str, Any] = {"processor": platform.processor() or None}
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            timeout=2, check=True,
        ).stdout.strip()
        hardware["nvidia_smi"] = gpu.splitlines()
    except (OSError, subprocess.SubprocessError):
        hardware["nvidia_smi"] = None
    try:
        import torch

        hardware["cuda_available"] = bool(torch.cuda.is_available())
        hardware["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        hardware["cuda_available"] = None
    return {"python": sys.version, "platform": platform.platform(), "hardware": hardware, "packages": versions, "git_revision": revision}


def _seed_runtime(seed: int) -> None:
    """Seed available inference/runtime libraries without requiring ML extras."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _markdown_report(summary: Mapping[str, Any]) -> str:
    dataset = summary["dataset"]
    lines = ["# OmniSearch AI retrieval experiment", "", f"- Run: `{summary['run_id']}`", f"- Dataset: {dataset['name']} {dataset['version']} ({dataset['split']})", f"- License: {dataset['license']}", f"- Dataset SHA-256: `{summary['dataset_sha256']}`", f"- Random seed: {summary['seed']}", f"- K: {summary['k']}", "", "## Observed results", "", "| Method | P@K | R@K | MRR | nDCG@K | Search latency (ms) | Embedding time (ms) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for method, values in summary["aggregates"].items():
        display = ["—" if values[key] is None else f"{values[key]:.4f}" for key in ("precision_at_k", "recall_at_k", "mrr", "ndcg_at_k", "search_latency_ms", "embedding_generation_ms")]
        lines.append(f"| {method} | " + " | ".join(display) + " |")
    lines += ["", "These values are calculated from the retrieval results stored in `raw_results.csv`; they are specific to this dataset, split, judgments and configuration.", "", "## Limitations", ""]
    lines.extend(f"- {item}" for item in summary["limitations"])
    lines += ["", f"Dataset source: {dataset['source_url']}", f"Citation: {dataset['citation']}", ""]
    return "\n".join(lines)
