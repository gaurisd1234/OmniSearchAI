# Experiments

Store each experiment's code or run notes here, with configuration based on a
versioned TOML file from `configs/`. Record the code revision, random seed,
software versions, dataset provenance and split, model identifier/revision,
hardware, command, metrics, and known limitations. Keep generated outputs and
large assets out of version control.

`demo_multimodal_retrieval.py` demonstrates text-to-image and image-to-image
search. Its CLIP checkpoint (and optional EasyOCR weights) are loaded only when
you run the demo; automated tests use lightweight injected components instead.

`final_demo.py` is the consolidated offline-first demonstration. Its default
run evaluates actual BM25 results on the small manual fixture and writes
`outputs/final_demo/experiment_report.md`, JSON, and CSV. `--semantic` adds
dense and RRF methods; `--multimodal` explicitly loads CLIP; `--rag` requires a
running Ollama model; `--generative-checkpoint PATH` requires a checkpoint whose
document-ID vocabulary matches the manual fixture. See the committed
[`final_demo_report_2026-11-16.md`](final_demo_report_2026-11-16.md) for the
observed default run and its limits.

## Research retrieval evaluation

The reusable framework is in `omnisearch_ai.evaluation`. It accepts independent
BM25, dense, hybrid, generative, and image retrievers through `MethodSpec`;
image methods select `id_field="image_id"`. Searchers must expose
`search(query, top_k=...)`. Use `embedding_probe` to measure an isolated query
embedding call; this is reported separately because indexing and query encoding
are otherwise method-specific. Search latency includes encoding performed
inside `search`, and excludes model/index startup and the configurable warm-up.

The manually judged smoke dataset is at
`datasets/evaluation/manual_demo/`. It is project-authored CC0 synthetic data,
with graded query-document judgments. Its metrics are only a framework check,
not effectiveness evidence.

For public benchmarks, prepare a local directory in BEIR layout with
`corpus.jsonl`, `queries.jsonl`, `qrels/<split>.tsv`, and a `manifest.json`.
Manifest fields are `name`, `version`, `license`, `source_url`, `citation`, and
`split`; optional fields include `modality`, `license_url`, and `notes`. Use
`load_dataset(path, split="test")`. This code does not download datasets.
Dataset licenses vary; BEIR explicitly leaves dataset licensing to each
dataset owner, so verify and record the individual dataset license before use:
[BEIR dataset documentation](https://github.com/beir-cellar/beir/wiki/Datasets-available).

Construct retriever instances in an experiment-specific script, then call
`run_experiment(dataset, methods, output_dir, k=10, repeats=3, seed=42,
config={...})`. `MethodSpec` captures the ID field and model/retrieval settings.
`configs/research_evaluation.toml` is a reproducibility template for the five
requested method families; concrete retriever adapters remain experiment-
specific because each needs its own corpus, index, or trained checkpoint.
The runner produces per-query metrics, `raw_results.csv` (ranked result IDs,
scores, relevance grades and evidence), JSON and Markdown reports, and charts
when Matplotlib is installed. Configure an MLflow tracking URI to log observed
metrics and artifacts. Install optional tracking/plotting packages with
`python -m pip install -e ".[research]"`.

Metrics are macro-averaged over queries with qrels; positive grades count as
relevant for Precision@K, Recall@K and MRR, while graded nDCG uses
`2**grade - 1`. Duplicate result IDs are deduplicated. Latency is median per
query over actual timed retrieval calls; embedding generation is only recorded
when an embedding probe is provided. Reports record the dataset fingerprint,
license/provenance, split, random seed, method configuration, Python/runtime
details (installed package versions and Git revision when available), and known
limits. The framework does not make universal claims from a
single dataset or split.
