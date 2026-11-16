import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from omnisearch_ai.embeddings import EmbeddingBatch, EmbeddingMetrics
from omnisearch_ai.ingestion.models import DocumentChunk, DocumentRecord, PageContent
from omnisearch_ai.retrieval import BM25Retriever, ChromaVectorIndex, DenseRetriever


def make_record(
    name: str,
    text: str,
    *,
    page: int = 1,
    metadata: dict | None = None,
) -> DocumentRecord:
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    document_id = f"doc_{content_hash}"
    chunk = DocumentChunk(
        chunk_id=f"{document_id}:chunk:0",
        document_id=document_id,
        text=text,
        page_number=page,
        chunk_index=0,
        start_offset=0,
        end_offset=len(text),
        reference=f"{document_id}#page={page}&chunk=0",
    )
    return DocumentRecord(
        document_id=document_id,
        source=Path(name),
        content_hash=content_hash,
        file_type=Path(name).suffix.lstrip("."),
        pages=(PageContent(page, text),),
        chunks=(chunk,),
        metadata=metadata or {},
    )


class SemanticFixtureEmbedder:
    """Tiny deterministic stand-in that maps vehicle synonyms together."""

    @staticmethod
    def _vector(text: str) -> list[float]:
        words = set(text.lower().split())
        if words & {"car", "automobile", "vehicle"}:
            return [1.0, 0.0, 0.0]
        if words & {"banana", "fruit"}:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def encode_texts(self, texts):
        vectors = np.asarray([self._vector(text) for text in texts], dtype=np.float32)
        return EmbeddingBatch(
            vectors,
            EmbeddingMetrics(
                device="cpu",
                num_texts=len(texts),
                batch_size=16,
                elapsed_seconds=0.001,
                rss_before_bytes=1,
                peak_rss_bytes=1,
                rss_delta_bytes=0,
            ),
        )


class RetrievalTests(unittest.TestCase):
    def test_bm25_ranks_keyword_matches_and_preserves_result_fields(self) -> None:
        documents = [
            make_record(
                "bm25_guide.txt",
                "BM25 keyword ranking uses term frequency and inverse document frequency.",
                page=3,
                metadata={"topic": "lexical", "author": "Ada"},
            ),
            make_record(
                "vector_guide.txt",
                "Semantic retrieval compares vectors to find similar meaning.",
                metadata={"topic": "semantic", "author": "Lin"},
            ),
            make_record(
                "indexing_guide.txt",
                "Chroma stores vector records with metadata and persistent IDs.",
                metadata={"topic": "storage", "author": "Sam"},
            ),
        ]
        retriever = BM25Retriever(documents, top_k=2, k1=1.4, b=0.7)

        results = retriever.search(
            "inverse document frequency", where={"topic": "lexical"}
        )

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.document_name, "bm25_guide.txt")
        self.assertEqual(result.page_number, 3)
        self.assertEqual(result.method, "bm25")
        self.assertGreater(result.score, 0.0)
        self.assertIn("frequency", result.content)
        self.assertEqual(result.metadata["author"], "Ada")
        self.assertEqual(retriever.search("terms not in corpus"), [])

    def test_dense_retrieves_semantic_synonym_and_filters_metadata(self) -> None:
        documents = [
            make_record(
                "vehicle_notes.pdf",
                "A car is a road vehicle with an engine.",
                page=2,
                metadata={"topic": "transport", "author": "Ada"},
            ),
            make_record(
                "food_notes.txt",
                "Banana is a sweet fruit.",
                metadata={"topic": "food", "author": "Lin"},
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            with ChromaVectorIndex(
                directory, embedder=SemanticFixtureEmbedder()
            ) as index:
                index.index_documents(documents)
                retriever = DenseRetriever(index, top_k=2)

                results = retriever.search(
                    "automobile",
                    where={
                        "$and": [
                            {"author": "Ada"},
                            {"page_number": {"$gte": 2}},
                        ]
                    },
                )

                self.assertEqual(len(results), 1)
                self.assertEqual(results[0].document_name, "vehicle_notes.pdf")
                self.assertEqual(results[0].page_number, 2)
                self.assertEqual(results[0].method, "dense")
                self.assertAlmostEqual(results[0].score, 1.0, places=5)
                self.assertIn("road vehicle", results[0].content)

    def test_top_k_and_bm25_parameters_are_validated(self) -> None:
        document = make_record("one.txt", "A simple test document.")
        with self.assertRaises(ValueError):
            BM25Retriever([document], top_k=0)
        with self.assertRaises(ValueError):
            BM25Retriever([document], b=1.5)


if __name__ == "__main__":
    unittest.main()
