# FastAPI Reference

Base URL in the documented local setup: `http://127.0.0.1:8000`. Open the
generated OpenAPI/Swagger page at `/docs` while the server is running.

The API is intended for a local development machine. It has no authentication
or user isolation. The launch command binds to `127.0.0.1`; do not expose it to
an untrusted network without adding suitable access controls.

## Health and dashboard

### `GET /api/health`

Returns `{"status":"ok","service":"omnisearch-ai"}` when the API process is
serving requests.

### `GET /api/dashboard`

Returns local application status, including:

- `document_count`, `chunk_count`, `image_count`
- `search_count`, `method_counts`, and up to 20 recent method/latency events
- configured embedding, CLIP, generative retrieval, and Ollama model identifiers

Search events store method, UTC timestamp, and latency; query text is not
persisted.

## Documents

### `GET /api/documents`

Returns `{"documents": [...]}`. Each item includes document ID/name/source,
file type, page and chunk counts, whether dense vectors were indexed, added time,
and extracted metadata.

### `POST /api/documents?embed=false`

Multipart form field `file` accepts PDF, TXT, or DOCX. Maximum upload size is
50 MiB. The backend extracts text/chunks and updates BM25's source catalog.
Set `embed=true` to additionally generate and persist semantic vectors. This
may initialize/download the configured Sentence Transformer on first use. If
embedding fails, the document remains keyword-searchable and the response
contains `embedding_error`.

Example:

```powershell
curl.exe -F "file=@C:\docs\guide.pdf" "http://127.0.0.1:8000/api/documents?embed=false"
```

### `DELETE /api/documents/{document_id}`

Deletes the catalog entry, stored source file, and indexed vector/image entries
associated with the document where available. Returns 404 if the ID is unknown.

## Text search

### `POST /api/search`

Request JSON:

```json
{
  "query": "How does semantic search work?",
  "method": "hybrid",
  "top_k": 5,
  "document_id": null
}
```

Methods are `keyword`, `semantic`, `hybrid`, and `generative`. `document_id` is
an optional metadata filter. `top_k` is bounded to 1–100. The response has the
method, query, and ordered results containing document name/ID, page, source,
chunk content, score, retrieval method, and metadata.

Semantic search requires at least one indexed dense vector. Hybrid search also
requires vectors in addition to the in-memory BM25 index. Generative retrieval
requires a trained checkpoint at `data/app/generative_checkpoint` with its
document-ID vocabulary. Missing readiness returns HTTP 409 with an explanation.

## Multimodal search

### `POST /api/images?document_id=&ocr=false`

Multipart field `file` accepts PNG, JPEG, WEBP, BMP, or TIFF up to 20 MiB.
Optional `document_id` associates the image with an existing document. Set
`ocr=true` to extract OCR text using EasyOCR. Indexing loads CLIP weights and
may load OCR weights when enabled.

### `GET /api/images/search?query=red%20square&top_k=5`

Searches indexed images from a natural-language query in CLIP's joint image/text
space. Returns image ID/path, score, OCR text, optional source document/page,
and metadata.

### `POST /api/images/search-by-image?top_k=5`

Multipart field `file` contains a query image. Finds visually similar indexed
images and excludes the identical image ID when it is already in the index.
The uploaded query image is temporary and removed after the search.

## RAG assistant

### `POST /api/rag`

Request JSON:

```json
{"question":"What does the guide say about chunking?","top_k":5}
```

The response serializes `RAGResponse`: question, answer, citation list,
retrieved results, `insufficient_evidence`, model name, and optional provider
error. RAG uses hybrid retrieval and therefore requires dense-indexed documents.
The configured Ollama server/model must also be available for generation.

## Research analytics

### `GET /api/analytics`

Returns evaluation reports found under `outputs/**/experiment_report.json` and
the recent application retrieval-latency events. The Streamlit Analytics page
charts saved actual report metrics. It shows an empty state when no report is
present; it does not synthesize missing method results.

## Status codes

| Status | Meaning |
|---|---|
| 200 | Successful read/search/delete/answer operation. |
| 201 | Upload accepted; a duplicate document may resolve to the existing ID. |
| 400 | Unsupported type, invalid/empty file, or invalid input. |
| 404 | Document ID not found for deletion. |
| 409 | Feature prerequisite is missing or an optional model/index is unavailable. |
| 413 | Upload exceeds the route's size limit. |

FastAPI's validation layer may return 422 for malformed JSON or query values.
