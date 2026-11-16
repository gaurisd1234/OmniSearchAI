# OmniSearch AI application

The application pairs a FastAPI service with a Streamlit research interface.
Reusable retrieval, ingestion, RAG, multimodal, generative retrieval, and
evaluation logic remains in `src/omnisearch_ai/`; the app layer coordinates
those tested components and exposes them over HTTP.

## Components

- `omnisearch_ai.app.service.AppService` owns the durable document catalog,
  rebuilds the BM25 in-memory index from its stored files, records privacy-safe
  method/latency search events, and constructs model-backed indexes lazily.
- `omnisearch_ai.app.api.create_app` provides the REST API. Uploads are limited
  to 50 MiB for documents and 20 MiB for images. The health endpoint is
  `GET /api/health`.
- `ui/streamlit_app.py` provides six pages: Dashboard, Document Management, AI
  Search, Multimodal Search, RAG Assistant, and Research Analytics.
- `data/app/` stores uploaded documents, images, the catalog, and search events.
  Chroma collections remain in the configured vector-store path. Do not commit
  these runtime files.

## Retrieval and model readiness

Document upload extracts text and creates page-aware chunks. Keyword retrieval
is available immediately. The upload checkbox `Generate semantic vectors on
upload` is off by default; enabling it loads the configured Sentence
Transformer and indexes vectors. Dense and hybrid search report a clear error
if there are no vectors. Image indexing loads CLIP on demand; OCR is separately
opt-in and loads EasyOCR weights. Generative retrieval expects a trained
checkpoint under `data/app/generative_checkpoint`. RAG uses hybrid retrieval
and the configured local Ollama model. These model requirements are surfaced as
feature errors when not available; the app does not fabricate results.

Research Analytics reads actual `experiment_report.json` outputs under
`outputs/`, produced by the evaluation framework, and shows application search
latency events recorded by successful retrieval calls. It displays an empty
state when no experiment report exists.

## Windows 11 setup and launch

Run from PowerShell in the repository root. Activate the project environment
or use the explicit interpreter below:

```powershell
cd C:\OmniSearchAI
.\.venv\Scripts\python.exe -m pip install -e ".[app]"
```

For model-backed semantic and multimodal indexing, install their optional
dependencies too. This does not fetch model weights; the first enabled model
operation may do so. Keep the already configured CUDA-enabled PyTorch build if
it is working in this virtual environment.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[app,vector,multimodal,generative,research]"
```

Start the API in PowerShell terminal 1:

```powershell
cd C:\OmniSearchAI
$env:OMNISEARCH_APP_DATA_DIR = "C:\OmniSearchAI\data\app"
.\.venv\Scripts\python.exe -m uvicorn omnisearch_ai.app.api:app --host 127.0.0.1 --port 8000
```

Start the UI in PowerShell terminal 2:

```powershell
cd C:\OmniSearchAI
$env:OMNISEARCH_API_URL = "http://127.0.0.1:8000"
.\.venv\Scripts\streamlit.exe run ui\streamlit_app.py
```

Open the local address printed by Streamlit (normally `http://localhost:8501`).
The API can be checked at `http://127.0.0.1:8000/docs`.

For RAG, install Ollama separately and pull the configured small quantized
model if it is not already available, then keep the Ollama service running:

```powershell
ollama pull qwen2.5:3b-instruct-q4_K_M
ollama serve
```

The 4 GB GPU may not hold the complete model plus its context cache; Ollama
chooses placement based on available memory. Check actual placement with
`ollama ps`. Generative retrieval remains unavailable until a compatible
checkpoint and document-ID vocabulary are trained and saved at the configured
checkpoint directory.

## Automated checks

Run the API and service integration tests, then the full suite:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_application -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests use temporary storage, sample text files, and real local HTTP
serialization through FastAPI's test client. They do not load or download
model weights. Model-backed semantic, image, generative, and Ollama execution
still depends on the optional dependencies, data, and local services listed
above.
