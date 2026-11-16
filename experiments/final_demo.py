"""Offline-first end-to-end demonstration of OmniSearch AI.

Default mode runs tested ingestion, BM25 retrieval, and the evaluation runner
against the small manually judged fixture. Optional flags explicitly enable
model-backed semantic, image, RAG, and generative demonstrations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omnisearch_ai.config import load_config
from omnisearch_ai.evaluation import MethodSpec, load_dataset, run_experiment
from omnisearch_ai.ingestion.models import DocumentChunk, DocumentRecord, PageContent
from omnisearch_ai.retrieval import BM25Retriever, ChromaVectorIndex, DenseRetriever, HybridRetriever


def fixture_records(dataset):
    records = []
    for document_id, item in dataset.corpus.items():
        body = f"{item.get('title', '')}\n{item.get('text', '')}".strip()
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        source = Path(f"{document_id}.txt")
        chunk = DocumentChunk(
            chunk_id=f"{document_id}:chunk:0", document_id=document_id, text=body,
            page_number=1, chunk_index=0, start_offset=0, end_offset=len(body),
            reference=f"{document_id}#page=1&chunk=0",
        )
        records.append(DocumentRecord(
            document_id=document_id, source=source, content_hash=digest,
            file_type="txt", pages=(PageContent(1, body),), chunks=(chunk,),
            metadata={"title": item.get("title", document_id)},
        ))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "final_demo")
    parser.add_argument("--semantic", action="store_true", help="load Sentence Transformers and run dense plus RRF retrieval")
    parser.add_argument("--multimodal", action="store_true", help="load CLIP and demonstrate image search")
    parser.add_argument("--rag", action="store_true", help="call the configured local Ollama model")
    parser.add_argument("--generative-checkpoint", type=Path, help="run query-to-document-ID retrieval from a compatible checkpoint")
    args = parser.parse_args()

    dataset = load_dataset(ROOT / "datasets" / "evaluation" / "manual_demo", split="test")
    records = fixture_records(dataset)
    config_path = ROOT / "configs" / "baseline.toml"
    keyword = BM25Retriever(records, config_path=config_path)
    methods = [MethodSpec("bm25", keyword, config={"implementation": "rank-bm25"})]
    dense_index = None
    temporary = tempfile.TemporaryDirectory(prefix="omnisearch-final-demo-")
    try:
        if args.semantic:
            from omnisearch_ai.embeddings import SentenceTransformerEmbedder

            embedder = SentenceTransformerEmbedder.from_config(config_path)
            dense_index = ChromaVectorIndex(Path(temporary.name) / "text_chroma", embedder=embedder, config_path=config_path)
            dense_index.index_documents(records)
            dense = DenseRetriever(dense_index, config_path=config_path)
            methods.extend([
                MethodSpec("dense", dense, config={"model": load_config(config_path).embedding_model}, embedding_probe=lambda query: embedder.encode_texts([query])),
                MethodSpec("hybrid_rrf", HybridRetriever(keyword, dense, config_path=config_path), config={"fusion": "weighted_rrf"}, embedding_probe=lambda query: embedder.encode_texts([query])),
            ])

        report = run_experiment(
            dataset, methods, args.output, k=3, repeats=3,
            seed=load_config(config_path).seed,
            config={"fixture": "manual_demo", "scope": "smoke demonstration"},
        )
        output: dict[str, object] = {
            "report_path": str(args.output / "experiment_report.md"),
            "dataset": dataset.manifest.name,
            "methods": report["aggregates"],
            "interpretation": "Synthetic fixture only; do not treat as general effectiveness evidence.",
        }

        if args.multimodal:
            from omnisearch_ai.multimodal.image_index import ChromaImageIndex
            from omnisearch_ai.multimodal.retrievers import ImageToImageRetriever, TextToImageRetriever

            image_dir = ROOT / "tests" / "fixtures" / "images"
            image_paths = sorted(image_dir.glob("*.png"))
            image_index = ChromaImageIndex(path=Path(temporary.name) / "image_chroma", config_path=config_path)
            try:
                indexing = image_index.index_images(image_paths)
                output["multimodal"] = {
                    "indexed_images": indexing.indexed_count,
                    "text_to_image": [item.image_id for item in TextToImageRetriever(image_index).search("red square", top_k=3)],
                    "image_to_image": [item.image_id for item in ImageToImageRetriever(image_index).search(image_paths[0], top_k=3)],
                    "note": "Model-backed demonstration, not a relevance benchmark.",
                }
            finally:
                image_index.close()

        if args.rag:
            from omnisearch_ai.llm.ollama import OllamaClient
            from omnisearch_ai.rag.pipeline import RAGPipeline

            response = RAGPipeline(keyword, OllamaClient(config_path=config_path), config_path=config_path).answer(
                next(iter(dataset.queries.values()))
            )
            output["rag"] = {
                "answer": response.answer,
                "citations": [citation.__dict__ if hasattr(citation, "__dict__") else {
                    "marker": citation.marker, "document_name": citation.document_name,
                    "page_number": citation.page_number, "source": citation.source,
                } for citation in response.citations],
                "insufficient_evidence": response.insufficient_evidence,
                "model": response.model,
            }

        if args.generative_checkpoint:
            from omnisearch_ai.generative_retrieval.retriever import GenerativeRetriever

            generative = GenerativeRetriever.from_checkpoint(args.generative_checkpoint, {record.document_id: record for record in records}, device="auto")
            output["generative"] = {
                "model": generative.model_name,
                "results": [{"document_id": item.document_id, "score": item.score} for item in generative.search(next(iter(dataset.queries.values()), ""), top_k=3)],
                "note": "The checkpoint vocabulary must match this fixture's document IDs.",
            }
        print(json.dumps(output, indent=2, ensure_ascii=False, default=str))
    finally:
        if dense_index is not None:
            dense_index.close()
        temporary.cleanup()


if __name__ == "__main__":
    main()
