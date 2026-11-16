"""Evaluation dataset manifests and local BEIR-format dataset loading."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    name: str
    version: str
    license: str
    source_url: str
    citation: str
    split: str
    modality: str = "text"
    license_url: str = ""
    notes: str = ""

    def validate(self) -> None:
        missing = [key for key in ("name", "version", "license", "source_url", "citation", "split") if not getattr(self, key).strip()]
        if missing:
            raise ValueError(f"Dataset manifest is missing required fields: {', '.join(missing)}")


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    manifest: DatasetManifest
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    corpus: dict[str, dict[str, Any]]


def load_dataset(directory: str | Path, *, split: str = "test") -> EvaluationDataset:
    """Load local corpus/queries JSONL and BEIR qrels TSV; never downloads data.

    ``manifest.json`` is required and must document the dataset-specific license
    and provenance. BEIR itself cautions that licenses vary by constituent set.
    """
    root = Path(directory)
    raw_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest = DatasetManifest(**raw_manifest)
    manifest.validate()
    queries = {str(row["_id"]): str(row["text"]) for row in _jsonl(root / "queries.jsonl")}
    corpus = {str(row["_id"]): row for row in _jsonl(root / "corpus.jsonl")}
    qrels_path = root / "qrels" / f"{split}.tsv"
    qrels: dict[str, dict[str, int]] = {}
    with qrels_path.open(encoding="utf-8") as handle:
        header = handle.readline().strip().split("\t")
        if header != ["query-id", "corpus-id", "score"]:
            raise ValueError("qrels TSV header must be query-id, corpus-id, score")
        for line in handle:
            query_id, document_id, grade = line.rstrip("\n").split("\t")
            qrels.setdefault(query_id, {})[document_id] = int(grade)
    if not qrels or set(qrels) - set(queries):
        raise ValueError("qrels must include judgments for known queries")
    unknown_documents = {doc_id for judgments in qrels.values() for doc_id in judgments} - set(corpus)
    if unknown_documents:
        raise ValueError(f"qrels refer to unknown corpus IDs: {sorted(unknown_documents)[:5]}")
    manifest = replace(manifest, split=split)
    return EvaluationDataset(manifest, queries, qrels, corpus)


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
