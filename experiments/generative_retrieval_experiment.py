"""Fine-tune small T5 doc-ID generation and compare it to BM25/dense baselines.

Input schemas are documented in docs/architecture.md. Running this script
explicitly loads the configured T5 and embedding checkpoints; it is not part of
the offline test suite.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omnisearch_ai.config import load_config
from omnisearch_ai.embeddings import SentenceTransformerEmbedder
from omnisearch_ai.generative_retrieval.data import (
    PairSplits,
    create_pairs,
    split_pairs,
    write_pairs_jsonl,
)
from omnisearch_ai.generative_retrieval.evaluation import (
    compare_retrievers,
    write_evaluation_report,
)
from omnisearch_ai.generative_retrieval.model import load_t5_components
from omnisearch_ai.generative_retrieval.retriever import GenerativeRetriever
from omnisearch_ai.generative_retrieval.trainer import fine_tune_t5
from omnisearch_ai.generative_retrieval.vocabulary import DocumentIdVocabulary
from omnisearch_ai.ingestion.models import DocumentChunk, DocumentRecord, PageContent
from omnisearch_ai.retrieval import BM25Retriever, ChromaVectorIndex, DenseRetriever


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on {path}:{line_number}") from exc


def load_corpus(path: Path) -> list[DocumentRecord]:
    documents = []
    seen_ids = set()
    for item in read_jsonl(path):
        document_id = str(item["document_id"])
        text = str(item["text"])
        source = Path(item.get("source", f"{document_id}.txt"))
        page_number = int(item.get("page_number", 1))
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if document_id in seen_ids:
            raise ValueError(f"duplicate document ID in corpus: {document_id}")
        seen_ids.add(document_id)
        chunk = DocumentChunk(
            chunk_id=f"{document_id}:chunk:0",
            document_id=document_id,
            text=text,
            page_number=page_number,
            chunk_index=0,
            start_offset=0,
            end_offset=len(text),
            reference=f"{source}#page={page_number}&chunk=0",
        )
        documents.append(
            DocumentRecord(
                document_id=document_id,
                source=source,
                content_hash=digest,
                file_type=source.suffix.lstrip(".") or "txt",
                pages=(PageContent(page_number, text),),
                chunks=(chunk,),
                metadata=item.get("metadata", {}),
            )
        )
    if not documents:
        raise ValueError("corpus JSONL is empty")
    return documents


def read_judgments(
    path: Path,
    known_ids: set[str],
    *,
    validation_fraction: float,
    seed: int,
):
    grouped = {"train": {}, "validation": {}}
    test_queries = {}
    test_relevance = {}
    seen_query_splits = {}
    for row in read_jsonl(path):
        split = row.get("split", "train")
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"unknown judgment split: {split}")
        query = str(row["query"]).strip()
        if not query:
            raise ValueError("judgment queries must not be empty")
        previous_split = seen_query_splits.setdefault(query, split)
        if previous_split != split:
            raise ValueError(
                f"query appears in both {previous_split!r} and {split!r} splits: {query}"
            )
        document_ids = [str(value) for value in row["relevant_document_ids"]]
        for document_id in document_ids:
            if document_id not in known_ids:
                raise ValueError(f"judgment references unknown document {document_id!r}")
        if split == "test":
            if query in test_relevance:
                raise ValueError(f"duplicate test query: {query}")
            query_id = f"test-{len(test_queries) + 1:04d}"
            test_queries[query_id] = query
            test_relevance[query_id] = document_ids
        else:
            grouped[split].setdefault(query, []).extend(document_ids)
    train_pairs = create_pairs(grouped["train"], known_document_ids=known_ids)
    if grouped["validation"]:
        splits = PairSplits(
            train=train_pairs,
            validation=create_pairs(
                grouped["validation"], known_document_ids=known_ids
            ),
        )
    else:
        splits = split_pairs(
            train_pairs,
            validation_fraction=validation_fraction,
            seed=seed,
        )
    if not test_queries:
        raise ValueError("judgments JSONL must include at least one split='test' query")
    return splits, test_queries, test_relevance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True, help="corpus JSONL")
    parser.add_argument("--judgments", type=Path, required=True, help="qrels JSONL")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints" / "generative_t5")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "baseline.toml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    config = load_config(args.config)

    documents = load_corpus(args.corpus)
    known_ids = {document.document_id for document in documents}
    splits, queries, relevance = read_judgments(
        args.judgments,
        known_ids,
        validation_fraction=config.generative_validation_fraction,
        seed=config.seed,
    )
    vocabulary = DocumentIdVocabulary.from_document_ids(known_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vocabulary.save(args.output_dir / "document_id_vocabulary.json")
    write_pairs_jsonl(splits.train, args.output_dir / "train_pairs.jsonl")
    write_pairs_jsonl(splits.validation, args.output_dir / "validation_pairs.jsonl")

    components = load_t5_components(
        vocabulary,
        model_name=config.generative_model,
        device=args.device,
        config_path=args.config,
    )
    training = fine_tune_t5(
        splits,
        vocabulary,
        args.output_dir,
        components=components,
        config_path=args.config,
    )
    bm25 = BM25Retriever(documents, config_path=args.config)
    generative = GenerativeRetriever(
        components.model,
        components.tokenizer,
        vocabulary,
        {document.document_id: document for document in documents},
        device=components.device,
        max_input_tokens=config.generative_max_input_tokens,
        max_output_tokens=config.generative_max_output_tokens,
        candidate_sequences=config.generative_beam_candidates,
        model_name=components.model_name,
    )
    embedder = SentenceTransformerEmbedder.from_config(args.config)
    with tempfile.TemporaryDirectory(prefix="omnisearch-generative-eval-") as temp_dir:
        with ChromaVectorIndex(
            Path(temp_dir) / "chroma",
            embedder=embedder,
            config_path=args.config,
        ) as index:
            index.index_documents(documents)
            dense = DenseRetriever(index, config_path=args.config)
            reports = compare_retrievers(
                {"bm25": bm25, "dense": dense, "generative_t5": generative},
                queries,
                relevance,
                k=args.k,
                repeats=args.repeats,
            )
    software = {}
    for package in ("torch", "transformers", "sentence-transformers", "chromadb", "rank-bm25"):
        try:
            software[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            software[package] = "not installed"
    write_evaluation_report(
        reports,
        args.output_dir / "evaluation.json",
        metadata={
            "model": config.generative_model,
            "seed": config.seed,
            "training": asdict(training),
            "training_config": {
                "epochs": config.generative_epochs,
                "batch_size": config.generative_batch_size,
                "gradient_accumulation": config.generative_gradient_accumulation,
                "learning_rate": config.generative_learning_rate,
                "max_input_tokens": config.generative_max_input_tokens,
                "max_output_tokens": config.generative_max_output_tokens,
                "gradient_checkpointing": True,
                "seed": config.seed,
            },
            "corpus_path": str(args.corpus),
            "corpus_sha256": sha256_file(args.corpus),
            "judgments_path": str(args.judgments),
            "judgments_sha256": sha256_file(args.judgments),
            "config_path": str(args.config),
            "config_sha256": sha256_file(args.config),
            "train_queries": len({pair.query for pair in splits.train}),
            "validation_queries": len({pair.query for pair in splits.validation}),
            "test_queries": len(queries),
            "k": args.k,
            "repeats": args.repeats,
            "candidate_sequences": config.generative_beam_candidates,
            "command": sys.argv,
            "device": components.device,
            "gpu_name": (
                __import__("torch").cuda.get_device_name(0)
                if components.device == "cuda"
                else None
            ),
            "latency_note": "Warm search only; excludes model load, indexing, and training.",
            "limitations": "Small simplified query-to-ID T5 baseline; not a state-of-the-art reproduction.",
            "software_versions": software,
        },
    )
    print(f"Wrote training and retrieval evaluation artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
