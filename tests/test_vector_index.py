import tempfile
import unittest
from pathlib import Path

import numpy as np

from omnisearch_ai.embeddings import EmbeddingBatch, EmbeddingMetrics
from omnisearch_ai.ingestion.models import DocumentChunk, DocumentRecord, PageContent
from omnisearch_ai.retrieval import ChromaVectorIndex


class FakeEmbedder:
    def encode_texts(self, texts):
        vectors = np.asarray(
            [[float(len(text) or 1), float(sum(map(ord, text)) % 97 + 1)] for text in texts],
            dtype=np.float32,
        )
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
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


def make_record(texts: tuple[str, ...] = ("first chunk", "second chunk")) -> DocumentRecord:
    document_id = "doc_test_content_id"
    chunks = tuple(
        DocumentChunk(
            chunk_id=f"{document_id}:chunk:{index}",
            document_id=document_id,
            text=text,
            page_number=index + 1,
            chunk_index=index,
            start_offset=0,
            end_offset=len(text),
            reference=f"{document_id}#page={index + 1}&chunk={index}",
        )
        for index, text in enumerate(texts)
    )
    return DocumentRecord(
        document_id=document_id,
        source=Path("sample.pdf"),
        content_hash="test_content_id",
        file_type="pdf",
        pages=tuple(PageContent(i + 1, text) for i, text in enumerate(texts)),
        chunks=chunks,
        metadata={"title": "Test document"},
    )


class ChromaIndexTests(unittest.TestCase):
    def test_incremental_persistence_provenance_query_and_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "chroma"
            record = make_record()
            with ChromaVectorIndex(path, embedder=FakeEmbedder()) as index:
                first_report = index.index_documents([record])
                self.assertEqual(first_report.indexed_count, 2)
                self.assertEqual(index.count(), 2)
                self.assertEqual(first_report.embedding_metrics[0].num_texts, 2)

                second_report = index.index_documents([record])
                self.assertEqual(second_report.indexed_count, 0)
                self.assertEqual(second_report.skipped_count, 2)
                self.assertEqual(second_report.embedding_metrics, ())

                stored = index.collection.get(
                    ids=[record.chunks[0].chunk_id], include=["metadatas"]
                )
                metadata = stored["metadatas"][0]
                self.assertEqual(metadata["document_id"], record.document_id)
                self.assertEqual(metadata["source"], str(record.source))
                self.assertEqual(metadata["page_number"], 1)
                self.assertEqual(metadata["chunk_index"], 0)
                self.assertTrue(index.query("first chunk", limit=1))

            with ChromaVectorIndex(path, embedder=FakeEmbedder()) as reopened:
                self.assertEqual(reopened.count(), 2)
                self.assertEqual(reopened.delete_document(record.document_id), 2)
                self.assertEqual(reopened.count(), 0)

    def test_reindex_of_changed_chunk_updates_existing_id_without_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with ChromaVectorIndex(temporary_directory, embedder=FakeEmbedder()) as index:
                initial = make_record(("original content",))
                changed = make_record(("updated content",))
                index.index_documents([initial])

                report = index.index_documents([changed])

                self.assertEqual(report.updated_count, 1)
                self.assertEqual(report.indexed_count, 0)
                self.assertEqual(index.count(), 1)


if __name__ == "__main__":
    unittest.main()
