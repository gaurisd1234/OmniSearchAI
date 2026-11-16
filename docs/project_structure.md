# Project Folder Guide

| Path | Purpose |
|---|---|
| `src/omnisearch_ai/` | Installable Python package with shared domain code. |
| `src/omnisearch_ai/ingestion/` | PDF/TXT/DOCX readers, page-aware records, chunking, and duplicate IDs. |
| `src/omnisearch_ai/embeddings/` | Sentence Transformer encoding, device selection, batch fallback, and timing/memory metrics. |
| `src/omnisearch_ai/retrieval/` | BM25, dense Chroma index, shared results, filters, RRF hybrid, optional cross-encoder. |
| `src/omnisearch_ai/generative_retrieval/` | Query-document pairs, split logic, document-ID vocabulary, T5 model/trainer, candidate mapping, and baseline evaluator. |
| `src/omnisearch_ai/multimodal/` | Pillow image handling, CLIP vectors, EasyOCR adapter, Chroma image collection, text/image retrievers. |
| `src/omnisearch_ai/rag/` | Context packing, evidence prompting, citation handling, and explicit insufficient-evidence behavior. |
| `src/omnisearch_ai/llm/` | Replaceable LLM protocol and local Ollama HTTP client. |
| `src/omnisearch_ai/evaluation/` | Local dataset/manifest loading, ranking metrics, experiment runner, CSV/report/chart output, and optional MLflow. |
| `src/omnisearch_ai/app/` | FastAPI endpoints and the service that coordinates package modules and durable local catalog state. |
| `ui/` | Streamlit frontend entry point (`streamlit_app.py`). |
| `configs/` | Versioned TOML configuration. `baseline.toml` is the application/model baseline; `research_evaluation.toml` is an evaluation template. |
| `datasets/` | Local data location. `datasets/evaluation/manual_demo/` is a tiny project-authored CC0 smoke dataset; public corpora are not bundled. |
| `experiments/` | Reproducible scripts and committed notes/results. `final_demo.py` is the offline-first combined demonstration. |
| `tests/` | Unit and integration tests. Test fixtures are intentionally small; model-backed behavior is usually injected/mocked. |
| `tests/fixtures/` | Sample TXT, synthetic images, and small generative query/corpus pairs. |
| `docs/` | Technical report, installation, API, architecture, and folder documentation. |
| `hardware_info.py` | Interpreter, NVIDIA driver, PyTorch/CUDA diagnostics and a real CUDA compute smoke test. |
| `.venv/` | Project-specific Python virtual environment; never commit it. |
| `data/app/` | Runtime application uploads, catalog, and query-free usage events; gitignored. |
| `vector_store/` | Persistent Chroma data created by local experiments/app; gitignored. |
| `outputs/` | Generated CSV/JSON/Markdown/charts and experiment runs; gitignored. |
| `checkpoints/` | Optional model training checkpoints when a script is run; do not commit. |

## Configuration locations

`configs/baseline.toml` describes chunk size/overlap, embedding model and
batches, vector-store path/collection, BM25/RRF settings, Ollama, RAG context,
CLIP, and T5 defaults. `load_config()` supports the documented
`OMNISEARCH_*` environment overrides; see
[architecture.md](architecture.md) for the full list.

## Data and generated artifacts

Keep source data, downloaded weights/caches, private uploads, Chroma databases,
and generated metrics out of source control. For each benchmark, retain a
manifest with dataset-specific license, version, source, citation, and split.
Raw experimental CSV should accompany any result table that is shared.
