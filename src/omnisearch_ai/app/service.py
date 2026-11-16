"""Application service joining the tested ingestion and retrieval modules."""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from omnisearch_ai.config import load_config
from omnisearch_ai.ingestion import DocumentIngestor
from omnisearch_ai.retrieval.bm25 import BM25Retriever


class AppService:
    """Owns persisted document catalog and lazily constructed model clients."""

    def __init__(self, *, config_path: str | Path | None = None, storage_dir: str | Path | None = None) -> None:
        self.config_path = Path(config_path).resolve() if config_path else None
        self.config = load_config(self.config_path)
        self.root = Path(storage_dir or os.getenv("OMNISEARCH_APP_DATA_DIR", "data/app")).resolve()
        self.docs_dir = self.root / "documents"
        self.images_dir = self.root / "images"
        self.registry_path = self.root / "catalog.json"
        self.stats_path = self.root / "search_events.jsonl"
        self.root.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self.documents: dict[str, Any] = {}
        self._entries: dict[str, dict[str, Any]] = {}
        self._bm25: BM25Retriever | None = None
        self._vector_index: Any = None
        self._dense: Any = None
        self._hybrid: Any = None
        self._image_index: Any = None
        self._generative: Any = None
        self._rag: Any = None
        self._load_catalog()

    def _load_catalog(self) -> None:
        if self.registry_path.exists():
            try:
                entries = json.loads(self.registry_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                entries = []
            for entry in entries:
                source = Path(entry.get("path", ""))
                if not source.is_file():
                    continue
                try:
                    document = DocumentIngestor(config_path=self.config_path).ingest(source)
                except Exception:
                    continue
                if document.document_id == entry.get("document_id"):
                    self.documents[document.document_id] = document
                    self._entries[document.document_id] = {**entry, "path": str(source)}
        self._persist_catalog()
        self._rebuild_bm25()

    def _persist_catalog(self) -> None:
        payload = [self._entries[key] for key in sorted(self._entries)]
        temporary = self.registry_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.registry_path)

    def _rebuild_bm25(self) -> None:
        self._bm25 = BM25Retriever(self.documents.values(), config_path=self.config_path)
        self._hybrid = None
        self._rag = None
        self._generative = None

    @staticmethod
    def _allowed_file(filename: str, extensions: set[str]) -> str:
        ext = Path(filename or "").suffix.lower()
        if ext not in extensions:
            raise ValueError(f"Unsupported file type {ext or '(none)'}")
        return ext

    def add_document(self, filename: str, content: bytes, *, embed: bool = False) -> dict[str, Any]:
        ext = self._allowed_file(filename, {".pdf", ".txt", ".docx"})
        if not content:
            raise ValueError("Uploaded file is empty")
        with self._lock:
            staging = self.root / f"upload_{uuid.uuid4().hex}{ext}"
            staging.write_bytes(content)
            try:
                parsed = DocumentIngestor(config_path=self.config_path).ingest(staging)
                existing = self.documents.get(parsed.document_id)
                if existing is not None:
                    staging.unlink(missing_ok=True)
                    return self.document_info(existing, duplicate=True)
                destination_dir = self.docs_dir / parsed.document_id
                destination_dir.mkdir(parents=True, exist_ok=True)
                safe_name = Path(filename).name
                destination = destination_dir / safe_name
                if destination.exists():
                    destination = destination_dir / f"{uuid.uuid4().hex}_{safe_name}"
                shutil.move(str(staging), destination)
                record = DocumentIngestor(config_path=self.config_path).ingest(destination)
                entry = {"document_id": record.document_id, "path": str(destination), "filename": destination.name,
                         "dense_indexed": False, "added_utc": datetime.now(timezone.utc).isoformat()}
                self.documents[record.document_id] = record
                self._entries[record.document_id] = entry
                self._rebuild_bm25()
                embedding_error = None
                if embed:
                    try:
                        self._get_vector_index().index_documents([record])
                        entry["dense_indexed"] = True
                    except Exception as exc:
                        embedding_error = str(exc)
                        entry["embedding_error"] = embedding_error
                self._persist_catalog()
                result = self.document_info(record)
                result["embedding_error"] = embedding_error
                return result
            except Exception:
                staging.unlink(missing_ok=True)
                raise

    def document_info(self, record: Any, *, duplicate: bool = False) -> dict[str, Any]:
        entry = self._entries.get(record.document_id, {})
        return {"document_id": record.document_id, "name": record.source.name, "source": str(record.source),
                "file_type": record.file_type, "page_count": len(record.pages), "chunk_count": len(record.chunks),
                "dense_indexed": bool(entry.get("dense_indexed")), "added_utc": entry.get("added_utc"),
                "duplicate": duplicate, "metadata": record.metadata}

    def list_documents(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self.document_info(record) for record in sorted(self.documents.values(), key=lambda item: item.source.name.casefold())]

    def delete_document(self, document_id: str) -> bool:
        with self._lock:
            record = self.documents.get(document_id)
            entry = self._entries.get(document_id)
            if record is None:
                return False
            if self._vector_index is not None:
                self._vector_index.delete_document(document_id)
            elif entry and entry.get("dense_indexed"):
                self._delete_from_collection(self.config.vector_collection, {"document_id": document_id})
            if self._image_count():
                self._delete_from_collection(self.config.image_collection, {"document_id": document_id})
            self.documents.pop(document_id, None)
            self._entries.pop(document_id, None)
            Path(entry["path"]).unlink(missing_ok=True)
            try:
                Path(entry["path"]).parent.rmdir()
            except OSError:
                pass
            self._persist_catalog()
            self._rebuild_bm25()
            return True

    def _delete_from_collection(self, collection_name: str, where: dict[str, Any]) -> None:
        client = self._active_chroma_client()
        owns_client = client is None
        if client is None:
            import chromadb
            client = chromadb.PersistentClient(path=str(self.config.vector_store_dir))
        try:
            names = {getattr(item, "name", str(item)) for item in client.list_collections()}
            if collection_name not in names:
                return
            collection = client.get_collection(collection_name)
            existing = collection.get(where=where, include=["metadatas"])
            if existing["ids"]:
                collection.delete(ids=existing["ids"])
        finally:
            if owns_client:
                client.close()

    def _get_vector_index(self):
        if self._vector_index is None:
            from omnisearch_ai.retrieval.chroma_index import ChromaVectorIndex
            self._vector_index = ChromaVectorIndex(config_path=self.config_path)
        return self._vector_index

    def _get_dense(self):
        if self._dense is None:
            from omnisearch_ai.retrieval.dense import DenseRetriever
            self._dense = DenseRetriever(self._get_vector_index(), config_path=self.config_path)
        return self._dense

    def _get_hybrid(self):
        if self._hybrid is None:
            from omnisearch_ai.retrieval.hybrid import HybridRetriever
            self._hybrid = HybridRetriever(self._bm25, self._get_dense(), config_path=self.config_path)
        return self._hybrid

    def _get_generative(self):
        if self._generative is None:
            checkpoint = self.root / "generative_checkpoint"
            if not checkpoint.is_dir():
                raise RuntimeError("Generative retrieval is unavailable: train or configure data/app/generative_checkpoint first")
            from omnisearch_ai.generative_retrieval.retriever import GenerativeRetriever
            self._generative = GenerativeRetriever.from_checkpoint(checkpoint, self.documents, device=self.config.device)
        return self._generative

    def search(self, query: str, method: str = "hybrid", *, top_k: int = 5, document_id: str | None = None) -> list[Any]:
        if not query.strip():
            return []
        started = time.perf_counter()
        where = {"document_id": document_id} if document_id else None
        if method == "keyword":
            retriever = self._bm25
        elif method == "semantic":
            if not self._any_dense_documents():
                raise RuntimeError("No documents have dense embeddings. Enable semantic indexing when uploading documents.")
            retriever = self._get_dense()
        elif method == "hybrid":
            if not self._any_dense_documents():
                raise RuntimeError("Hybrid search requires semantic vectors. Enable semantic indexing when uploading documents.")
            retriever = self._get_hybrid()
        elif method == "generative":
            retriever = self._get_generative()
        else:
            raise ValueError("method must be keyword, semantic, hybrid, or generative")
        results = retriever.search(query, top_k=top_k, where=where)
        self._record_search(method, (time.perf_counter() - started) * 1000)
        return results

    def _any_dense_documents(self) -> bool:
        if self._vector_index is not None:
            return self._vector_index.count() > 0
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(self.config.vector_store_dir))
            try:
                collection = client.get_collection(self.config.vector_collection)
                return collection.count() > 0
            finally:
                client.close()
        except Exception:
            return False

    def _record_search(self, method: str, latency_ms: float) -> None:
        row = {"timestamp": datetime.now(timezone.utc).isoformat(), "method": method, "latency_ms": latency_ms}
        with self._lock, self.stats_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def dashboard(self) -> dict[str, Any]:
        events = self._read_search_events()
        methods = Counter(row.get("method", "unknown") for row in events)
        return {"document_count": len(self.documents), "chunk_count": sum(len(doc.chunks) for doc in self.documents.values()),
                "image_count": self._image_count(), "search_count": len(events), "method_counts": dict(methods),
                "recent_searches": events[-20:], "models": {"embedding": self.config.embedding_model,
                "clip": self.config.clip_model, "generative_retrieval": self.config.generative_model,
                "llm": self.config.ollama_model, "ollama_url": self.config.ollama_base_url}}

    def _read_search_events(self) -> list[dict[str, Any]]:
        try:
            lines = self.stats_path.read_text(encoding="utf-8").splitlines()[-1000:]
            return [json.loads(line) for line in lines]
        except (OSError, json.JSONDecodeError):
            return []

    def _image_count(self) -> int:
        if self._image_index is not None:
            return self._image_index.count()
        try:
            client = self._active_chroma_client()
            owns_client = client is None
            if client is None:
                import chromadb
                client = chromadb.PersistentClient(path=str(self.config.vector_store_dir))
            try:
                return int(client.get_collection(self.config.image_collection).count())
            finally:
                if owns_client:
                    client.close()
        except Exception:
            return 0

    def _active_chroma_client(self):
        if self._vector_index is not None:
            return self._vector_index.client
        if self._image_index is not None:
            return self._image_index.client
        return None

    def _get_image_index(self):
        if self._image_index is None:
            from omnisearch_ai.multimodal.image_index import ChromaImageIndex
            self._image_index = ChromaImageIndex(config_path=self.config_path)
        return self._image_index

    def add_image(self, filename: str, content: bytes, *, document_id: str | None = None, ocr: bool = False) -> dict[str, Any]:
        ext = self._allowed_file(filename, {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})
        if not content:
            raise ValueError("Uploaded image is empty")
        target = self.images_dir / f"{uuid.uuid4().hex}{ext}"
        target.write_bytes(content)
        try:
            association = {}
            if document_id:
                doc = self.documents.get(document_id)
                if doc is None:
                    raise ValueError("Associated document was not found")
                association[str(target)] = {"document_id": document_id, "document_name": doc.source.name}
            report = self._get_image_index().index_images([target], associations=association, extract_ocr=ocr)
            image_record = self._get_image_index().processor.record(target)
            return {"image_id": image_record.image_id, "path": str(target), "indexed_count": report.indexed_count,
                    "skipped_count": report.skipped_count, "document_id": document_id}
        except Exception:
            target.unlink(missing_ok=True)
            raise

    def search_images(self, query: str, *, top_k: int = 5):
        if self._image_count() == 0:
            raise RuntimeError("No images have been indexed yet.")
        return self._get_image_index().search_text(query, top_k=top_k)

    def search_similar_images(self, path: str | Path, *, top_k: int = 5):
        if self._image_count() == 0:
            raise RuntimeError("No images have been indexed yet.")
        return self._get_image_index().search_image(path, top_k=top_k)

    def answer(self, question: str, *, top_k: int = 5):
        from omnisearch_ai.llm.ollama import OllamaClient
        from omnisearch_ai.rag.pipeline import RAGPipeline
        if not self._any_dense_documents():
            raise RuntimeError("RAG requires semantic vectors. Enable semantic indexing when uploading documents.")
        if self._rag is None:
            self._rag = RAGPipeline(self._get_hybrid(), OllamaClient(config_path=self.config_path), config_path=self.config_path, top_k=top_k)
        started = time.perf_counter()
        response = self._rag.answer(question)
        self._record_search("rag", (time.perf_counter() - started) * 1000)
        return response

    def analytics(self) -> dict[str, Any]:
        reports = []
        for path in Path("outputs").glob("**/experiment_report.json"):
            try:
                reports.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return {"runs": reports, "search_events": self._read_search_events()}

    def close(self) -> None:
        clients = {}
        if self._vector_index is not None:
            clients[id(self._vector_index.client)] = self._vector_index.client
        if self._image_index is not None:
            clients[id(self._image_index.client)] = self._image_index.client
        for client in clients.values():
            client.close()
