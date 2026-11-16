# OmniSearch AI

OmniSearch AI is a local-first, student-oriented research project for document
search, multimodal retrieval, generative retrieval, and evidence-grounded
question answering. It combines independently accessible retrieval baselines
with a FastAPI backend and a Streamlit interface. Its goal is to make
experiments inspectable and reproducible, not to claim a universal winning
retriever.

## What is implemented

| Area | Current implementation | Evidence/status |
|---|---|---|
| Ingestion | PDF, TXT, DOCX; page-aware configurable chunks; content IDs; duplicate checks | Automated tests pass. |
| Keyword search | BM25 via rank-bm25 | Tests and smoke comparisons. |
| Semantic search | MiniLM embeddings with persistent Chroma, CUDA/CPU selection | Tests; measured warm embedding and dense retrieval smoke reports. |
| Hybrid search | Weighted RRF; optional cross-encoder | RRF tests and a tiny measured synthetic comparison; no general quality claim. |
| Generative retrieval | Experimental T5 query-to-document-ID pipeline | Data/trainer/retriever tests use mocked model components; no committed real training metrics. |
| Multimodal | CLIP text/image space, image indexing, optional EasyOCR | Interfaces and index tests use injected components; no committed real-model quality metrics. |
| RAG | Evidence-bounded pipeline, citation markers, replaceable Ollama adapter | Mocked generation/grounding tests; no human-rated answer-quality study. |
| Application | FastAPI API and six-page Streamlit UI | API integration tests and local server smoke checks. |
| Evaluation | P@K, R@K, MRR, nDCG, latency, CSV/report output, optional MLflow/charts | Framework tests and small actual smoke reports. |

The full methods, literature, experimental settings, actual recorded results,
and limitations are in [the technical report](docs/technical_report.md).
It explicitly separates measured results from unmeasured capabilities.

## Quick start on Windows 11

Use Python 3.12 and the repository virtual environment. Full install options and
the recorded CUDA setup are in the [installation guide](docs/installation.md).

```powershell
cd C:\OmniSearchAI
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install -e ".[app,vector,multimodal,generative,research]"
```

Start the API in PowerShell terminal 1:

```powershell
cd C:\OmniSearchAI
.\.venv\Scripts\python.exe -m uvicorn omnisearch_ai.app.api:app --host 127.0.0.1 --port 8000
```

Start the UI in terminal 2:

```powershell
cd C:\OmniSearchAI
.\.venv\Scripts\streamlit.exe run ui\streamlit_app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.
The API's interactive OpenAPI page is `http://127.0.0.1:8000/docs`.
Model weights are loaded only when a model-backed function is used. A trained
generative checkpoint and a running Ollama service are separate prerequisites.

## Reproduce the final smoke demonstration

The default command does not load neural weights. It ingests the small
project-authored judged fixture, runs BM25, computes real metrics and latency,
and writes CSV and JSON/Markdown reports under the ignored `outputs/` folder:

```powershell
cd C:\OmniSearchAI
.\.venv\Scripts\python.exe experiments\final_demo.py
```

Optional model-backed demonstrations are explicit:

```powershell
# Dense and RRF hybrid: configured Sentence Transformer and Chroma
.\.venv\Scripts\python.exe experiments\final_demo.py --semantic

# CLIP text-to-image and image-to-image over the sample PNGs
.\.venv\Scripts\python.exe experiments\final_demo.py --multimodal

# RAG through the configured local Ollama model
.\.venv\Scripts\python.exe experiments\final_demo.py --rag
```

## Documentation map

- [Technical and research report](docs/technical_report.md)
- [Installation and Windows launch guide](docs/installation.md)
- [HTTP API reference](docs/api.md)
- [Project folder guide](docs/project_structure.md)
- [Architecture and configuration map](docs/architecture.md)
- [Application details](docs/application.md)
- [Experiment instructions](experiments/README.md)
- [Final demo and actual result note](experiments/final_demo_report_2026-11-16.md)
- [Measured hybrid smoke report](experiments/hybrid_retrieval_report_2026-11-13.md)
- [Measured embedding smoke report](experiments/embedding_benchmark_2026-11-13.md)

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The test suite uses small fixtures and mocks for optional model providers. A
passing test suite verifies interfaces and behavior; it does not imply that a
model was downloaded, trained, or evaluated on a real-world benchmark.
