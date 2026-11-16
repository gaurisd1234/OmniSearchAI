from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from omnisearch_ai.app.api import create_app
from omnisearch_ai.app.service import AppService
from omnisearch_ai.rag.pipeline import RAGResponse, SourceCitation


class ApplicationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.config_path = root / "config.toml"
        vector_path = str(root / "vectors").replace("\\", "/")
        self.config_path.write_text(
            f'[vector_store]\npath = "{vector_path}"\ncollection = "test_chunks"\n'
            '[multimodal]\nimage_collection = "test_images"\n',
            encoding="utf-8",
        )
        self.service = AppService(storage_dir=self.temp.name, config_path=self.config_path)
        self.client = TestClient(create_app(self.service))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.service.close()
        self.temp.cleanup()

    def test_document_upload_list_keyword_search_duplicate_and_delete(self):
        response = self.client.post(
            "/api/documents",
            files={"file": ("guide.txt", b"CUDA enables GPU tensor operations in PyTorch.", "text/plain")},
        )
        self.assertEqual(response.status_code, 201, response.text)
        document_id = response.json()["document_id"]
        self.assertGreater(response.json()["chunk_count"], 0)
        self.assertEqual(self.client.get("/api/documents").json()["documents"][0]["document_id"], document_id)

        result = self.client.post("/api/search", json={"query": "GPU tensor", "method": "keyword", "top_k": 3})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["results"][0]["document_id"], document_id)
        self.assertIn("page_number", result.json()["results"][0])

        duplicate = self.client.post("/api/documents", files={"file": ("copy.txt", b"CUDA enables GPU tensor operations in PyTorch.", "text/plain")})
        self.assertTrue(duplicate.json()["duplicate"])
        self.assertEqual(len(self.client.get("/api/documents").json()["documents"]), 1)
        deleted = self.client.delete(f"/api/documents/{document_id}")
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(self.client.get("/api/dashboard").json()["document_count"], 0)

    def test_document_upload_rejects_unknown_and_empty_files(self):
        unknown = self.client.post("/api/documents", files={"file": ("code.py", b"print('x')", "text/x-python")})
        self.assertEqual(unknown.status_code, 400)
        empty = self.client.post("/api/documents", files={"file": ("empty.txt", b"", "text/plain")})
        self.assertEqual(empty.status_code, 400)

    def test_search_stats_are_actual_and_persist_across_service_restart(self):
        self.client.post("/api/documents", files={"file": ("doc.txt", b"Search statistics are recorded after a query.", "text/plain")})
        self.client.post("/api/search", json={"query": "statistics", "method": "keyword"})
        dashboard = self.client.get("/api/dashboard").json()
        self.assertEqual(dashboard["search_count"], 1)
        self.assertEqual(dashboard["method_counts"], {"keyword": 1})
        restored = AppService(storage_dir=self.temp.name, config_path=self.config_path)
        self.assertEqual(len(restored.list_documents()), 1)
        restored.close()

    def test_insufficient_dense_index_is_reported_without_model_load(self):
        response = self.client.post("/api/search", json={"query": "meaning", "method": "semantic"})
        self.assertEqual(response.status_code, 409)
        self.assertIn("No documents have dense embeddings", response.json()["detail"])

    def test_health_and_analytics_schema(self):
        self.assertEqual(self.client.get("/api/health").json()["status"], "ok")
        analytics = self.client.get("/api/analytics").json()
        self.assertIn("runs", analytics)
        self.assertIn("search_events", analytics)

    def test_rag_response_serializes_source_citations(self):
        self.service.answer = lambda question, top_k=5: RAGResponse(
            question=question,
            answer="Supported statement [S1]",
            citations=(SourceCitation("S1", "doc-1", "guide.txt", "guide.txt", 2, "chunk-1"),),
            retrieved=(),
            insufficient_evidence=False,
            model="mock-ollama",
        )
        response = self.client.post("/api/rag", json={"question": "What is supported?", "top_k": 2})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["citations"][0]["page_number"], 2)
        self.assertEqual(response.json()["model"], "mock-ollama")


if __name__ == "__main__":
    unittest.main()
