# Installation Guide

## Supported development setup

The repository targets Python 3.12. The recorded verification machine is
Windows 11 with Python 3.12.5, an NVIDIA RTX 3050 Laptop GPU (4 GiB VRAM), and
16 GiB RAM. The saved reports record NVIDIA driver 581.83 and PyTorch
2.14.1+cu130 for their measured CUDA runs. These versions document the tested
environment; check the project's currently configured artifacts before
changing versions.

## Create the virtual environment

Open PowerShell and run from the repository root:

```powershell
cd C:\OmniSearchAI
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Install the recorded CUDA PyTorch wheel into this environment if it is not
already installed and verified:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe hardware_info.py
```

`hardware_info.py` reports the active interpreter, environment, `nvidia-smi`,
Torch/CUDA versions, and runs an actual CUDA matrix multiplication. CUDA should
be considered available only if that computation succeeds. Do not replace a
working environment's CUDA wheel with a CPU wheel.

## Install project dependencies

The base package supports ingestion and BM25. Install the app and feature
extras for the full local interface:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[app,vector,multimodal,generative,research]"
```

Extras provide:

- `app`: FastAPI, Uvicorn, Streamlit, multipart uploads, requests, and pandas.
- `vector`: PyTorch, Sentence Transformers, ChromaDB, and process-memory
  measurement.
- `multimodal`: PyTorch/Transformers, Pillow, EasyOCR, and ChromaDB.
- `generative`: Transformers and SentencePiece for T5 fine-tuning/inference.
- `research`: Matplotlib charts and MLflow tracking.

When PyTorch is already installed in `.venv` and satisfies the declared
requirement, pip should keep that installed build. Re-run `hardware_info.py`
after dependency changes. Optional-package installation does not itself fetch
Hugging Face model weights or benchmark datasets; model constructors download
weights only when the relevant model-backed feature is explicitly used.

## Launch FastAPI and Streamlit

Use separate PowerShell terminals from the project root.

Terminal 1:

```powershell
cd C:\OmniSearchAI
$env:OMNISEARCH_APP_DATA_DIR = "C:\OmniSearchAI\data\app"
.\.venv\Scripts\python.exe -m uvicorn omnisearch_ai.app.api:app --host 127.0.0.1 --port 8000
```

Terminal 2:

```powershell
cd C:\OmniSearchAI
$env:OMNISEARCH_API_URL = "http://127.0.0.1:8000"
.\.venv\Scripts\streamlit.exe run ui\streamlit_app.py
```

Open the URL printed by Streamlit, normally `http://localhost:8501`. FastAPI's
OpenAPI reference is at `http://127.0.0.1:8000/docs`. The API command binds to
loopback. No authentication is included; do not bind it to a public interface
without adding authentication and deployment controls.

## Optional local language model

RAG uses the Ollama HTTP API. Install Ollama separately, then download the
configured model explicitly if needed:

```powershell
ollama pull qwen2.5:3b-instruct-q4_K_M
ollama serve
```

Ollama may split execution between GPU and CPU. A 4 GiB GPU does not guarantee
the model and its KV cache fit entirely in VRAM. Inspect actual placement with
`ollama ps`. Change `OMNISEARCH_OLLAMA_MODEL` or `[llm].ollama_model` in
`configs/baseline.toml` to use a different Ollama model.

## Run the project checks

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_application -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests use small local fixtures and mocked model interfaces. They do not
download model weights or datasets, fine-tune T5, or validate live Ollama
answers.

## Run demonstrations and experiments

Offline-first end-to-end fixture:

```powershell
.\.venv\Scripts\python.exe experiments\final_demo.py
```

Optional model-backed demonstrations:

```powershell
.\.venv\Scripts\python.exe experiments\final_demo.py --semantic
.\.venv\Scripts\python.exe experiments\final_demo.py --multimodal
.\.venv\Scripts\python.exe experiments\final_demo.py --rag
```

The first semantic or image indexing operation may download the configured
model. `--rag` requires a running local Ollama service and model. A generative
retrieval checkpoint must first be trained; then pass
`--generative-checkpoint <checkpoint-directory>` to the demo, and ensure that
its stored document-ID vocabulary matches the selected corpus.

For the dedicated T5 workflow and its data schema, see
[`experiments/README.md`](../experiments/README.md) and
[`generative_retrieval_experiment.py`](../experiments/generative_retrieval_experiment.py).

## Runtime data locations

- Upload catalog/files/events: `data/app/` (override with
  `OMNISEARCH_APP_DATA_DIR`).
- Text and image Chroma collections: the configured
  `vector_store.path` (`vector_store/chroma` by default).
- Model checkpoints: Hugging Face cache or an explicitly supplied checkpoint.
- Experiment reports: choose an output directory; `outputs/` is gitignored.

Do not commit uploaded/private documents, vector stores, model weights,
credentials, or generated run artifacts.
