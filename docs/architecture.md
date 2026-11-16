# OmniSearch AI architecture

## Design

The repository uses a Python 3.12 `src` layout. Domain code lives in
`src/omnisearch_ai/`. The base install includes PyMuPDF and python-docx for
ingestion and rank-bm25 for keyword retrieval. Model and vector database
dependencies remain in the optional `vector` extra.

## Package responsibilities

| Path | Responsibility |
| --- | --- |
| `config.py` | Loads TOML settings and supported `OMNISEARCH_*` environment overrides into an immutable `AppConfig`. |
| `logging_utils.py` | Sets up the shared `omnisearch_ai` logger with a consistent format and configured level. |
| `ingestion/` | Reads PDF, TXT, and DOCX; normalizes text, extracts metadata, assigns content-based document IDs, detects duplicates, and produces page-aware configurable chunks with references. |
| `embeddings/` | Encodes chunks and queries with Sentence Transformers, selects CUDA or CPU, adapts batch size on GPU out-of-memory, normalizes vectors, and records observed runtime and memory metrics. |
| `retrieval/chroma_index.py` | Persists vectors and chunk provenance in Chroma, supports incremental upsert, deduplication, query, and document deletion. |
| `retrieval/bm25.py` | Independently ranks in-memory document chunks with rank-bm25 BM25Okapi and metadata filters. |
| `retrieval/dense.py` | Independently embeds natural-language queries and performs metadata-filtered cosine search through Chroma. |
| `retrieval/results.py` | Defines shared result fields for document names, source/page references, content, method, and method-specific scores. |
| `retrieval/hybrid.py` | Independently requests BM25 and dense candidate lists, combines ranks with configurable weighted Reciprocal Rank Fusion, and optionally reranks fused candidates with a Sentence Transformers cross-encoder. |
| `evaluation/retrieval_metrics.py` | Computes binary and graded Precision@K, Recall@K, MRR, and nDCG@K with configurable text/image result identifiers and duplicate-ID handling. |
| `evaluation/datasets.py` | Loads local project or BEIR-layout JSONL/TSV evaluation data and requires dataset-specific license and provenance metadata; it never downloads datasets. |
| `evaluation/runner.py` | Measures independent retrieval methods, optional embedding probes, writes ranked raw CSV and query metrics, records reproducibility metadata, creates charts, and optionally logs to MLflow. |
| `app/service.py` | Coordinates tested package components, persists the uploaded-document catalog and search usage events, and initializes model-backed components only when needed. |
| `app/api.py` | Exposes document, search, image, RAG, dashboard, and evaluation analytics routes through FastAPI. |
| `ui/streamlit_app.py` | Provides the six-page local research UI and calls the FastAPI backend. |
| `llm/base.py` | Defines the replaceable provider protocol, generation settings, and provider error type. |
| `llm/ollama.py` | Calls a locally running Ollama server through its JSON HTTP API using only the Python standard library. |
| `rag/pipeline.py` | Retrieves and optionally reranks evidence, bounds the context, builds a source-marked prompt, returns explicit abstentions, and attaches document/page citations. |
| `multimodal/images.py` | Defines image records, Pillow EXIF/RGB preprocessing, lazy EasyOCR, and normalized CLIP ViT-B/32 image/text embeddings in one shared joint space with CUDA batch reduction and CPU fallback. |
| `multimodal/image_index.py` | Stores content-addressed images in a separate persistent Chroma collection and supports text-to-image search, image-to-image search, provenance filters, and deletion. |
| `multimodal/retrievers.py` | Exposes separate `TextToImageRetriever` and `ImageToImageRetriever` interfaces over the image index. |
| `generative_retrieval/data.py` | Expands judgments into deterministic query/document pairs and splits by query to prevent train/validation leakage. |
| `generative_retrieval/vocabulary.py` | Sorts document IDs, assigns atomic T5 special tokens, and fingerprints/saves the mapping. |
| `generative_retrieval/model.py` | Lazily loads a T5 model/tokenizer and selects CPU/CUDA based on availability and free VRAM. |
| `generative_retrieval/trainer.py` | Fine-tunes T5 with small batches, gradient accumulation, gradient checkpointing, validation, epoch checkpoints, and CPU fallback after CUDA OOM. |
| `generative_retrieval/retriever.py` | Generates candidate ID tokens, rejects malformed/duplicate/unmapped IDs, and maps valid IDs to indexed source-aware results. |
| `generative_retrieval/evaluation.py` | Compares BM25, dense, and generative retrievers using P@K, R@K, MRR, nDCG@K, and measured search latency. |

## Repository responsibilities

| Path | Responsibility |
| --- | --- |
| `configs/` | Versioned TOML experiment settings, including seeds, methods, dataset/split identifiers, and runtime options. |
| `datasets/` | Location for local datasets; files are not committed. Dataset provenance and split policy belong in experiment records. |
| `experiments/` | Experiment-specific code and run records that identify configuration, code revision, dependencies, data, hardware, metrics, and limitations. |
| `ui/` | Streamlit application entry point; install the `app` optional dependencies to launch it. |
| `tests/` | Fast checks for imports, embeddings, and persistent indexing; Chroma uses temporary directories and encoder behavior is stubbed, so tests do not download models or data. CUDA-specific fallback tests run only when CUDA is available. |
| `docs/architecture.md` | This map of the package boundaries and repository layout. |
| `hardware_info.py` | Existing standalone machine, NVIDIA driver, and PyTorch CUDA diagnostic. |

## Ingestion output and page semantics

`DocumentIngestor` emits `DocumentRecord`, `PageContent`, and `DocumentChunk`
records. IDs use SHA-256 of normalized extracted text, so copies
with different filenames or container metadata deduplicate within one ingestor
instance. Chunks carry character offsets, their source page number, and a
`document_id#page=N&chunk=M` reference. Chunk size and overlap are Unicode
character counts; chunks never cross a page boundary.

PDF page numbers are physical page numbers. TXT files use form-feed characters
as page boundaries and otherwise have one logical page. DOCX does not contain
reliable rendered pagination without a layout engine, so the reader numbers
logical pages at explicit page breaks and `page_break_before` markers; ordinary
paragraphs share a logical page. Empty or textless documents and invalid input
raise `IngestionError` subclasses. Scanned image-only PDFs need OCR and are not
currently converted into text.

## Configuration and reproducibility

`load_config(path)` reads TOML and then applies environment overrides. Supported
overrides include `OMNISEARCH_SEED`, `OMNISEARCH_LOG_LEVEL`,
`OMNISEARCH_DATA_DIR`, `OMNISEARCH_DEVICE`, `OMNISEARCH_CHUNK_SIZE`,
`OMNISEARCH_CHUNK_OVERLAP`, `OMNISEARCH_RETRIEVAL_TOP_K`, `OMNISEARCH_BM25_K1`,
`OMNISEARCH_BM25_B`, `OMNISEARCH_RRF_K`,
`OMNISEARCH_RRF_CANDIDATE_K`, `OMNISEARCH_BM25_RRF_WEIGHT`,
`OMNISEARCH_DENSE_RRF_WEIGHT`, `OMNISEARCH_CROSS_ENCODER_MODEL`,
`OMNISEARCH_CROSS_ENCODER_BATCH_SIZE`, `OMNISEARCH_OLLAMA_BASE_URL`,
`OMNISEARCH_OLLAMA_MODEL`, `OMNISEARCH_OLLAMA_TIMEOUT_SECONDS`,
`OMNISEARCH_OLLAMA_CONTEXT_WINDOW`, `OMNISEARCH_OLLAMA_MAX_TOKENS`,
`OMNISEARCH_OLLAMA_TEMPERATURE`, `OMNISEARCH_RAG_MAX_CONTEXT_CHARS`,
`OMNISEARCH_RAG_MAX_CONTEXT_CHUNKS`, `OMNISEARCH_CLIP_MODEL`,
`OMNISEARCH_CLIP_CUDA_BATCH_SIZE`, `OMNISEARCH_CLIP_CPU_BATCH_SIZE`, and
`OMNISEARCH_IMAGE_COLLECTION`, `OMNISEARCH_GENERATIVE_MODEL`,
`OMNISEARCH_GENERATIVE_EPOCHS`, `OMNISEARCH_GENERATIVE_BATCH_SIZE`,
`OMNISEARCH_GENERATIVE_GRADIENT_ACCUMULATION`,
`OMNISEARCH_GENERATIVE_LEARNING_RATE`,
`OMNISEARCH_GENERATIVE_MAX_INPUT_TOKENS`,
`OMNISEARCH_GENERATIVE_MAX_OUTPUT_TOKENS`,
`OMNISEARCH_GENERATIVE_VALIDATION_FRACTION`, and
`OMNISEARCH_GENERATIVE_BEAM_CANDIDATES`. Chunk size and overlap are measured in characters;
the ingestor reads them from the `[ingestion]` TOML section unless a
`ChunkingConfig` is passed directly. Commit experiment TOML files;
do not commit secrets. Record dataset version and provenance, data split,
model revision, software versions, seed, hardware, command, and metrics with
each run. Environment overrides are useful for local changes but should be
captured in the run record when they affect results.

## Dependency policy

PyMuPDF, python-docx, and rank-bm25 are the base dependencies. PyTorch, Sentence
Transformers, ChromaDB, and psutil are grouped in the optional `vector` extra.
Transformers, Pillow, EasyOCR, and ChromaDB are declared in the optional
`multimodal` extra; install the GPU-specific PyTorch wheel before extras on
CUDA systems. Transformers and SentencePiece are in the optional `generative`
extra; no new dependencies are required by default.
The NVIDIA CUDA PyTorch wheel should be installed from the official CUDA index
before extras on GPU systems. Model weights download only when a model-backed
encoder or OCR reader is explicitly instantiated; package imports and tests do
not download model weights or datasets.

BM25 Okapi scores and dense cosine similarities are method-specific and are
not calibrated to one another. Hybrid retrieval combines ranks, not raw scores.
`experiments/compare_retrieval.py` provides qualitative BM25/dense inspection.
`experiments/compare_hybrid_retrieval.py` compares BM25, dense, and RRF hybrid
using hand-labeled synthetic queries, macro P@K, R@K, MRR, nDCG@K, and warm
query latency; `--rerank` adds the optional cross-encoder variant. Its tiny
synthetic fixture is only a functional evaluation example, not general
effectiveness evidence. Use representative held-out relevance judgments before
drawing conclusions about which method performs better.

## RAG and local generation

`RAGPipeline` accepts any `Searcher` and `LLMProvider`, making the Ollama
adapter replaceable for model comparisons. The intended composition is
`RAGPipeline(hybrid_retriever, OllamaClient(config_path="configs/baseline.toml"))`.
The pipeline sends source-marked retrieved excerpts, a question, and explicit
grounding/abstention instructions to the provider. It bounds prompt evidence by
chunk count and characters, returns cited source names and page numbers, skips
generation when retrieval yields no usable chunks, and preserves citations
when provider generation fails. This instruction-following setup encourages
grounding but is not a formal guarantee that a small language model will never
hallucinate; answers with no citation marker or an unknown marker are rejected
as insufficient evidence, but citation presence alone does not verify semantic
support. Verify answers against the returned sources.

The default local model is `qwen2.5:3b-instruct-q4_K_M`; its Ollama library
listing is about 1.9 GB. The default API context is 4,096 tokens, the generation
limit is 512 tokens, and RAG context is capped at 12,000 characters/eight
chunks. The character cap is not an exact token count; language and tokenization
affect prompt size, while Ollama enforces the configured token context. Model
weights do not equal runtime VRAM use: KV cache and display/GPU
workloads also consume memory. Ollama selects CPU/GPU placement based on
available memory; check `ollama ps` to see the actual split. No model weights
are included or downloaded by the project code.

## Image retrieval and embedding compatibility

`CLIPImageTextEmbedder` uses `openai/clip-vit-base-patch32`. It calls the text
and image towers from the same pretrained CLIP model and L2-normalizes both
projected outputs. CLIP jointly trained these outputs for image/text alignment,
so natural-language query vectors can be compared to image vectors, and image
vectors can be compared to one another. Do not mix these vectors with
Sentence Transformers vectors or other CLIP checkpoints in the same collection.

`ChromaImageIndex` persists image vectors in its own `omnisearch_images`
collection (under the configured Chroma path), keyed by SHA-256 content IDs.
Metadata includes file path/name, dimensions, format, OCR text, optional source
document ID/name/reference/page, and caller metadata. Re-indexing unchanged
images is idempotent; image and source-document deletion are supported. Pass
associations keyed by image path to `index_images` when an image came from an
ingested document. OCR is opt-in at indexing so model weights are never fetched
implicitly; `EasyOCRTextExtractor` defaults to CPU to leave VRAM available for
CLIP. CUDA CLIP batches default to 2 and fall back to smaller batches then CPU
on CUDA out-of-memory.

Install the optional dependencies with `python -m pip install -e ".[multimodal]"`
(install the appropriate PyTorch CUDA wheel first on GPU systems). Run
`python experiments/demo_multimodal_retrieval.py` for a text-to-image and
image-to-image demo over the small synthetic fixtures. Its first run loads the
CLIP checkpoint and may download model files; `--ocr` also loads/downloads
EasyOCR recognition weights. These are user-invoked actions,
not performed by tests or package imports. Tests use sample images and injected
fake embeddings/OCR, so they verify indexing and compatibility contracts
without claiming model retrieval quality.

## Generative retrieval experiment

This repository includes a simplified T5 query-to-document-ID baseline, not a
reproduction of a state-of-the-art generative retrieval method. The default
`google/flan-t5-small` checkpoint has 77M parameters. Training pairs are
expanded from relevance judgments; the document-ID vocabulary is sorted and
saved with a SHA-256 fingerprint, so token-to-ID mappings are reproducible for
a fixed indexed corpus. A dedicated special token represents each indexed ID.
Split by query to avoid query leakage, and include all searchable document IDs
in the output vocabulary.

Fine-tuning defaults to batch size 1, accumulation 8, gradient checkpointing,
256 input tokens, and short ID targets. CUDA is used only when enough memory is
available; auto mode falls back to CPU, and a CUDA OOM during training retries
the current accumulation group on CPU. Epoch model/tokenizer checkpoints,
training/validation pair JSONL, the ID vocabulary, validation losses, and the
evaluation JSON are saved under the output directory. For this 4 GB GPU, actual
memory depends on other GPU processes and sequence lengths; the script does not
claim training will always fit in VRAM.

Install the optional dependencies, then run the tiny synthetic workflow:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[generative]"
.\.venv\Scripts\python.exe experiments\generative_retrieval_experiment.py `
  --corpus tests\fixtures\generative\corpus.jsonl `
  --judgments tests\fixtures\generative\judgments.jsonl
```

The command explicitly loads the configured T5 checkpoint and the existing
Sentence Transformer model; the first run can download model weights. It trains
on the `train` split, validates on `validation`, and reports results on the
query-disjoint `test` split. The synthetic fixture is only a workflow smoke
example and is far too small for meaningful quality conclusions.

Corpus JSONL rows contain `document_id`, `source`, `text`, optional
`page_number`, and optional scalar `metadata`. Judgment JSONL rows contain
`query`, `relevant_document_ids`, and `split` (`train`, `validation`, or
`test`). If no explicit validation rows are provided, train queries are split
deterministically using the configured seed and validation fraction.

Generation is ordinary beam search over the extended T5 tokenizer vocabulary;
it is not constrained decoding. Free-form output may fail to match an ID token.
The retriever records and skips malformed outputs, duplicate IDs, and IDs that
are not present in the current index. It returns the first chunk for each
generated document ID. Its query-to-ID mapping has limited capacity, is
dependent on the fixed corpus vocabulary, and can overfit small pair sets; it
does not model large-corpus identifiers, hierarchical retrieval, or the full
training/evaluation setup of published generative retrieval systems. Compare
with BM25 and dense retrieval using held-out queries and recorded relevance
judgments before drawing conclusions.
