"""Persistent image vector indexing and cross-modal search with ChromaDB."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from omnisearch_ai.config import load_config
from omnisearch_ai.multimodal.images import (
    CLIPImageTextEmbedder,
    EasyOCRTextExtractor,
    ImageProcessor,
    ImageRecord,
    ImageSearchResult,
)


@dataclass(frozen=True, slots=True)
class ImageIndexReport:
    indexed_count: int
    skipped_count: int
    ocr_extracted_count: int


class ChromaImageIndex:
    """Image-only collection; vectors share CLIP space for text and images."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        collection_name: str | None = None,
        embedder: Any | None = None,
        processor: ImageProcessor | None = None,
        ocr_extractor: Any | None = None,
        index_batch_size: int = 16,
        config_path: str | Path | None = None,
    ) -> None:
        if index_batch_size <= 0:
            raise ValueError("index_batch_size must be positive")
        config = load_config(config_path)
        self.path = Path(path or config.vector_store_dir).expanduser()
        self.collection_name = collection_name or config.image_collection
        if self.collection_name == config.vector_collection:
            raise ValueError("image and text vectors must use separate Chroma collections")
        self.processor = processor or ImageProcessor()
        self.ocr_extractor = ocr_extractor
        self.index_batch_size = index_batch_size
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "ChromaDB is required; install omnisearch-ai[vector,multimodal]"
            ) from exc
        self.client = chromadb.PersistentClient(path=str(self.path))
        requested_model = getattr(
            embedder, "model_name", type(embedder).__name__ if embedder is not None else config.clip_model
        )
        collection_names = {
            getattr(item, "name", str(item)) for item in self.client.list_collections()
        }
        if self.collection_name in collection_names:
            self.collection = self.client.get_collection(
                name=self.collection_name, embedding_function=None
            )
            stored_model = (self.collection.metadata or {}).get("embedding_model")
            if stored_model != requested_model:
                self.client.close()
                raise ValueError(
                    f"image collection uses {stored_model!r}, not {requested_model!r}; "
                    "use a new collection for a different embedding model"
                )
        else:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "modality": "image",
                    "embedding_model": requested_model,
                },
                embedding_function=None,
            )
        self.embedder = embedder or CLIPImageTextEmbedder(
            config.clip_model,
            device=config.device,
            cuda_batch_size=config.clip_cuda_batch_size,
            cpu_batch_size=config.clip_cpu_batch_size,
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> ChromaImageIndex:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _extractor(self, *, gpu: bool):
        if self.ocr_extractor is None:
            self.ocr_extractor = EasyOCRTextExtractor(gpu=gpu)
        return self.ocr_extractor

    def _record(
        self,
        path: str | Path,
        *,
        association: Mapping[str, Any] | None,
        extract_ocr: bool,
        ocr_gpu: bool,
    ) -> ImageRecord:
        image, image_format = self.processor.open(path)
        association = association or {}
        ocr_text = ""
        if extract_ocr:
            import numpy as np

            ocr_text = self._extractor(gpu=ocr_gpu).extract(np.asarray(image))
        # Reuse validated image dimensions/format but avoid reopening for OCR.
        resolved = Path(path).expanduser().resolve()
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        record = ImageRecord(
            image_id=f"img_{digest}",
            path=resolved,
            content_hash=digest,
            width=image.width,
            height=image.height,
            image_format=image_format,
            ocr_text=ocr_text,
            document_id=association.get("document_id"),
            document_name=association.get("document_name"),
            document_reference=association.get("document_reference"),
            page_number=association.get("page_number"),
            metadata=dict(association.get("metadata", {})),
        )
        return record

    @staticmethod
    def _metadata(record: ImageRecord) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "image_id": record.image_id,
            "source": str(record.path),
            "image_name": record.path.name,
            "content_hash": record.content_hash,
            "width": record.width,
            "height": record.height,
            "image_format": record.image_format,
            "ocr_text": record.ocr_text,
        }
        optional = {
            "document_id": record.document_id,
            "document_name": record.document_name,
            "document_reference": record.document_reference,
            "page_number": record.page_number,
        }
        metadata.update({key: value for key, value in optional.items() if value is not None})
        for key, value in record.metadata.items():
            if key not in metadata and isinstance(value, (str, int, float, bool)):
                metadata[key] = value
        return metadata

    @staticmethod
    def _stored_document(record: ImageRecord) -> str:
        # Chroma keeps a non-empty display document even when OCR is disabled.
        return record.ocr_text or record.path.name

    def index_images(
        self,
        paths: Iterable[str | Path],
        *,
        associations: Mapping[str, Mapping[str, Any]] | None = None,
        extract_ocr: bool = False,
        ocr_gpu: bool = False,
    ) -> ImageIndexReport:
        """Index paths incrementally; associations are keyed by path string."""
        association_map = associations or {}
        records_by_id: dict[str, ImageRecord] = {}
        duplicate_inputs = 0
        for path in paths:
            association = association_map.get(str(path)) or association_map.get(path)
            record = self._record(
                path,
                association=association,
                extract_ocr=extract_ocr,
                ocr_gpu=ocr_gpu,
            )
            if record.image_id in records_by_id:
                duplicate_inputs += 1
            else:
                records_by_id[record.image_id] = record
        if not records_by_id:
            return ImageIndexReport(0, 0, 0)

        candidates = list(records_by_id.values())
        skipped = duplicate_inputs
        indexed = 0
        for start in range(0, len(candidates), self.index_batch_size):
            batch = candidates[start : start + self.index_batch_size]
            ids = [record.image_id for record in batch]
            existing = self.collection.get(ids=ids, include=["documents", "metadatas"])
            existing_documents = dict(zip(existing["ids"], existing.get("documents") or []))
            existing_metadata = dict(zip(existing["ids"], existing.get("metadatas") or []))
            pending: list[tuple[ImageRecord, dict[str, Any]]] = []
            for record in batch:
                metadata = self._metadata(record)
                old_id = record.image_id
                old_metadata = existing_metadata.get(old_id)
                if old_metadata:
                    preserved = {}
                    for field_name in (
                        "document_id",
                        "document_name",
                        "document_reference",
                        "page_number",
                    ):
                        if getattr(record, field_name) is None and old_metadata.get(field_name) is not None:
                            preserved[field_name] = old_metadata[field_name]
                    if not extract_ocr and old_metadata.get("ocr_text"):
                        preserved["ocr_text"] = str(old_metadata["ocr_text"])
                    if preserved:
                        record = replace(record, **preserved)
                    metadata = self._metadata(record)
                    reserved = set(metadata)
                    for key, value in old_metadata.items():
                        if key not in reserved:
                            metadata[key] = value
                if (
                    old_id in existing_documents
                    and existing_documents[old_id] == self._stored_document(record)
                    and old_metadata == metadata
                ):
                    skipped += 1
                else:
                    pending.append((record, metadata))

            if pending:
                images = [self.processor.open(record.path)[0] for record, _ in pending]
                vectors = self.embedder.encode_images(images)
                self.collection.upsert(
                    ids=[record.image_id for record, _ in pending],
                    embeddings=vectors.tolist(),
                    documents=[self._stored_document(record) for record, _ in pending],
                    metadatas=[metadata for _, metadata in pending],
                )
                indexed += len(pending)
        return ImageIndexReport(
            indexed_count=indexed,
            skipped_count=skipped,
            ocr_extracted_count=(len(records_by_id) if extract_ocr else 0),
        )

    def search_text(
        self,
        query: str,
        *,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[ImageSearchResult]:
        """Find images by natural-language text in CLIP's joint vector space."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not query.strip() or self.collection.count() == 0:
            return []
        vector = self.embedder.encode_texts([query])[0].tolist()
        return self._query(vector, top_k=top_k, where=where)

    def search_image(
        self,
        image_path: str | Path,
        *,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
        exclude_self: bool = True,
    ) -> list[ImageSearchResult]:
        """Find visually similar images to an uploaded/query image."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.collection.count() == 0:
            return []
        image, _ = self.processor.open(image_path)
        vector = self.embedder.encode_images([image])[0].tolist()
        results = self._query(vector, top_k=top_k + int(exclude_self), where=where)
        if exclude_self:
            query_hash = self.processor.record(image_path).image_id
            results = [item for item in results if item.image_id != query_hash]
        return results[:top_k]

    def _query(
        self,
        vector: list[float],
        *,
        top_k: int,
        where: dict[str, Any] | None,
    ) -> list[ImageSearchResult]:
        result = self.collection.query(
            query_embeddings=[vector],
            n_results=min(top_k, self.collection.count()),
            where=where,
            include=["metadatas", "distances"],
        )
        hits = []
        for metadata, distance in zip(result["metadatas"][0], result["distances"][0]):
            hits.append(
                ImageSearchResult(
                    image_id=metadata["image_id"],
                    path=metadata["source"],
                    score=1.0 - float(distance),
                    ocr_text=metadata.get("ocr_text", ""),
                    document_id=metadata.get("document_id"),
                    document_name=metadata.get("document_name"),
                    document_reference=metadata.get("document_reference"),
                    page_number=metadata.get("page_number"),
                    metadata=metadata,
                )
            )
        return hits

    def delete_image(self, image_id: str) -> bool:
        existing = self.collection.get(ids=[image_id], include=["metadatas"])
        if not existing["ids"]:
            return False
        self.collection.delete(ids=[image_id])
        return True

    def delete_document(self, document_id: str) -> int:
        existing = self.collection.get(
            where={"document_id": document_id}, include=["metadatas"]
        )
        ids = existing["ids"]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def count(self) -> int:
        return int(self.collection.count())
