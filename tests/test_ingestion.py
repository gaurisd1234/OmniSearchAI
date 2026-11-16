import shutil
import tempfile
import unittest
from pathlib import Path

import pymupdf
from docx import Document as WordDocument

from omnisearch_ai.ingestion import (
    ChunkingConfig,
    DocumentIngestor,
    EmptyDocumentError,
    IngestionError,
)


ROOT = Path(__file__).resolve().parents[1]


class DocumentIngestionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_dir.name)
        self.ingestor = DocumentIngestor(ChunkingConfig(chunk_size=32, overlap=8))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_pdf(self, path: Path) -> None:
        pdf = pymupdf.open()
        pdf.set_metadata({"title": "Sample PDF", "author": "OmniSearch tests"})
        for content in ("First PDF page with searchable words.", "Second page reference text."):
            page = pdf.new_page()
            page.insert_text((72, 72), content)
        pdf.save(path)
        pdf.close()

    def make_docx(self, path: Path) -> None:
        doc = WordDocument()
        doc.core_properties.title = "Sample Word document"
        doc.core_properties.author = "OmniSearch tests"
        doc.add_paragraph("First DOCX logical page.")
        doc.add_page_break()
        doc.add_paragraph("Second DOCX logical page.")
        doc.add_table(rows=1, cols=1).cell(0, 0).text = "DOCX table content."
        doc.save(path)

    def test_extracts_pdf_text_metadata_pages_and_references(self) -> None:
        path = self.directory / "sample.pdf"
        self.make_pdf(path)

        record = self.ingestor.ingest(path)

        self.assertEqual(record.file_type, "pdf")
        self.assertIn("First PDF page", record.pages[0].text)
        self.assertEqual([page.page_number for page in record.pages], [1, 2])
        self.assertEqual(record.metadata["title"], "Sample PDF")
        self.assertEqual(record.metadata["author"], "OmniSearch tests")
        self.assertTrue(record.chunks[0].reference.startswith(record.document_id + "#page=1"))
        self.assertTrue(all(chunk.document_id == record.document_id for chunk in record.chunks))

    def test_extracts_docx_text_metadata_and_explicit_page_breaks(self) -> None:
        path = self.directory / "sample.docx"
        self.make_docx(path)

        record = self.ingestor.ingest(path)

        self.assertEqual(record.file_type, "docx")
        self.assertEqual([page.page_number for page in record.pages], [1, 2])
        self.assertIn("First DOCX", record.pages[0].text)
        self.assertIn("Second DOCX", record.pages[1].text)
        self.assertIn("DOCX table content", record.pages[1].text)
        self.assertEqual(record.metadata["author"], "OmniSearch tests")
        self.assertEqual({chunk.page_number for chunk in record.chunks}, {1, 2})

    def test_extracts_txt_and_form_feed_page_numbers(self) -> None:
        path = self.directory / "sample.txt"
        path.write_text("First TXT page.\r\n\fSecond TXT page.", encoding="utf-8")

        record = self.ingestor.ingest(path)

        self.assertEqual(record.file_type, "txt")
        self.assertEqual([page.page_number for page in record.pages], [1, 2])
        self.assertIn("First TXT page.", record.pages[0].text)
        self.assertIn("Second TXT page.", record.pages[1].text)
        self.assertEqual(record.metadata["page_count"], 2)

    def test_configurable_chunk_size_overlap_and_document_ids(self) -> None:
        path = self.directory / "long.txt"
        path.write_text("abcdefghijklmnopqrstuvwxyz0123456789", encoding="utf-8")
        ingestor = DocumentIngestor(ChunkingConfig(chunk_size=12, overlap=4))

        record = ingestor.ingest(path)

        self.assertGreater(len(record.chunks), 1)
        self.assertEqual(record.chunks[0].start_offset, 0)
        self.assertEqual(record.chunks[1].start_offset, 8)
        self.assertTrue(record.document_id.startswith("doc_"))
        self.assertEqual(record.document_id, ingestor.ingest(path).document_id)

    def test_chunking_can_be_loaded_from_experiment_toml(self) -> None:
        config_path = self.directory / "experiment.toml"
        config_path.write_text(
            "[ingestion]\nchunk_size = 10\nchunk_overlap = 2\n", encoding="utf-8"
        )
        source = self.directory / "configured.txt"
        source.write_text("abcdefghijklmnopqrstuvwxyz", encoding="utf-8")
        ingestor = DocumentIngestor(config_path=config_path)

        record = ingestor.ingest(source)

        self.assertEqual(record.chunks[0].end_offset, 10)
        self.assertEqual(record.chunks[1].start_offset, 8)

    def test_duplicate_text_is_deduplicated_across_paths(self) -> None:
        original = self.directory / "one.txt"
        duplicate = self.directory / "two.txt"
        original.write_text("Same normalized content", encoding="utf-8")
        duplicate.write_text("Same normalized content", encoding="utf-8")

        records = self.ingestor.ingest_many([original, duplicate])

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].source, original.resolve())

    def test_empty_and_whitespace_documents_raise_clear_error(self) -> None:
        empty = self.directory / "empty.txt"
        whitespace = self.directory / "whitespace.txt"
        empty.write_bytes(b"")
        whitespace.write_text(" \n\t ", encoding="utf-8")
        for path in (empty, whitespace):
            with self.subTest(path=path.name), self.assertRaises(EmptyDocumentError):
                self.ingestor.ingest(path)

    def test_corrupted_pdf_and_docx_raise_ingestion_error(self) -> None:
        broken_pdf = self.directory / "broken.pdf"
        broken_docx = self.directory / "broken.docx"
        broken_pdf.write_bytes(b"not a PDF")
        broken_docx.write_bytes(b"not a DOCX")
        for path in (broken_pdf, broken_docx):
            with self.subTest(path=path.name), self.assertRaises(IngestionError):
                self.ingestor.ingest(path)

    def test_sample_text_fixture_is_usable(self) -> None:
        fixture = ROOT / "tests" / "fixtures" / "sample.txt"
        copy = self.directory / fixture.name
        shutil.copyfile(fixture, copy)

        record = self.ingestor.ingest(copy)

        self.assertIn("OmniSearch AI sample text", record.text)
        self.assertTrue(record.chunks)


if __name__ == "__main__":
    unittest.main()
