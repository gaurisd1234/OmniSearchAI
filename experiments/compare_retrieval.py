"""Compare BM25 and dense results on the same tiny synthetic corpus/queries.

This is a qualitative smoke comparison, not an effectiveness benchmark: no
relevance judgments are available and the two methods' scores are not calibrated
to the same scale.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omnisearch_ai.config import load_config
from omnisearch_ai.embeddings import SentenceTransformerEmbedder
from omnisearch_ai.ingestion.models import DocumentChunk, DocumentRecord, PageContent
from omnisearch_ai.retrieval import BM25Retriever, ChromaVectorIndex, DenseRetriever


CORPUS = (
    (
        "bm25_keyword_search.txt",
        "BM25 keyword retrieval ranks documents using term frequency and inverse document frequency.",
        {"topic": "lexical", "author": "Research notes"},
    ),
    (
        "dense_semantic_search.txt",
        "Dense semantic retrieval uses sentence embeddings to find passages with similar meaning.",
        {"topic": "semantic", "author": "Research notes"},
    ),
    (
        "gpu_hardware.txt",
        "NVIDIA GPU CUDA acceleration can speed up neural model inference.",
        {"topic": "hardware", "author": "Lab notes"},
    ),
    (
        "vector_database.txt",
        "Chroma stores persistent vectors and metadata for document chunks.",
        {"topic": "storage", "author": "Lab notes"},
    ),
)

QUERIES = (
    "term frequency and inverse document frequency ranking",
    "find passages that express similar meaning",
    "NVIDIA CUDA GPU acceleration",
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


def render(results) -> list[dict[str, object]]:
    return [
        {
            "document_name": item.document_name,
            "page_number": item.page_number,
            "score": round(item.score, 6),
            "content": item.content,
        }
        for item in results
    ]


def main() -> None:
    documents = [make_record(*item) for item in CORPUS]
    config = load_config(ROOT / "configs" / "baseline.toml")
    keyword = BM25Retriever(documents, config_path=ROOT / "configs" / "baseline.toml")
    encoder = SentenceTransformerEmbedder.from_config(ROOT / "configs" / "baseline.toml")
    comparison = []
    with tempfile.TemporaryDirectory(prefix="omnisearch-retrieval-") as directory:
        with ChromaVectorIndex(
            Path(directory) / "chroma", embedder=encoder
        ) as index:
            index.index_documents(documents)
            dense = DenseRetriever(index, top_k=config.retrieval_top_k)
            for query in QUERIES:
                comparison.append(
                    {
                        "query": query,
                        "bm25": render(keyword.search(query)),
                        "dense": render(dense.search(query)),
                    }
                )
    print(
        json.dumps(
            {
                "top_k": config.retrieval_top_k,
                "note": "Qualitative sample only; BM25 and cosine scores use different scales.",
                "comparisons": comparison,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
