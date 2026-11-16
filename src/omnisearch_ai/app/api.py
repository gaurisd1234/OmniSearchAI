"""FastAPI HTTP interface for OmniSearch AI."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from omnisearch_ai.app.service import AppService


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    method: str = "hybrid"
    top_k: int = Field(default=5, ge=1, le=100)
    document_id: str | None = None


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=30)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def create_app(service: AppService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service or AppService()
        yield
        app.state.service.close()

    api = FastAPI(title="OmniSearch AI", version="0.1.0", lifespan=lifespan)
    if service is not None:
        api.state.service = service

    def svc() -> AppService:
        instance = getattr(api.state, "service", None)
        if instance is None:
            raise HTTPException(503, "Application service is not ready")
        return instance

    @api.get("/api/health")
    def health():
        return {"status": "ok", "service": "omnisearch-ai"}

    @api.get("/api/dashboard")
    def dashboard():
        return svc().dashboard()

    @api.get("/api/documents")
    def documents():
        return {"documents": svc().list_documents()}

    @api.post("/api/documents", status_code=201)
    async def upload_document(file: UploadFile = File(...), embed: bool = Query(False)):
        content = await file.read(50 * 1024 * 1024 + 1)
        if len(content) > 50 * 1024 * 1024:
            raise HTTPException(413, "Maximum document upload size is 50 MiB")
        try:
            return svc().add_document(file.filename or "upload", content, embed=embed)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(500, str(exc)) from exc

    @api.delete("/api/documents/{document_id}")
    def delete_document(document_id: str):
        if not svc().delete_document(document_id):
            raise HTTPException(404, "Document not found")
        return {"deleted": True, "document_id": document_id}

    @api.post("/api/search")
    def search(request: SearchRequest):
        try:
            results = svc().search(request.query, request.method, top_k=request.top_k, document_id=request.document_id)
            return {"method": request.method, "query": request.query, "results": _jsonable(results)}
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @api.post("/api/images", status_code=201)
    async def upload_image(file: UploadFile = File(...), document_id: str | None = Query(None), ocr: bool = Query(False)):
        content = await file.read(20 * 1024 * 1024 + 1)
        if len(content) > 20 * 1024 * 1024:
            raise HTTPException(413, "Maximum image upload size is 20 MiB")
        try:
            return svc().add_image(file.filename or "image", content, document_id=document_id, ocr=ocr)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(500, str(exc)) from exc

    @api.get("/api/images/search")
    def search_images(query: str = Query(min_length=1), top_k: int = Query(5, ge=1, le=100)):
        try:
            return {"results": _jsonable(svc().search_images(query, top_k=top_k))}
        except Exception as exc:
            raise HTTPException(409, str(exc)) from exc

    @api.post("/api/images/search-by-image")
    async def search_images_by_image(file: UploadFile = File(...), top_k: int = Query(5, ge=1, le=100)):
        content = await file.read(20 * 1024 * 1024 + 1)
        if len(content) > 20 * 1024 * 1024:
            raise HTTPException(413, "Maximum image upload size is 20 MiB")
        image_root = svc().images_dir
        temporary = image_root / f"query_{__import__('uuid').uuid4().hex}{Path(file.filename or 'image.png').suffix.lower()}"
        temporary.write_bytes(content)
        try:
            return {"results": _jsonable(svc().search_similar_images(temporary, top_k=top_k))}
        except Exception as exc:
            raise HTTPException(409, str(exc)) from exc
        finally:
            temporary.unlink(missing_ok=True)

    @api.post("/api/rag")
    def ask(request: QuestionRequest):
        try:
            return _jsonable(svc().answer(request.question, top_k=request.top_k))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @api.get("/api/analytics")
    def analytics():
        return svc().analytics()

    return api


app = create_app()
