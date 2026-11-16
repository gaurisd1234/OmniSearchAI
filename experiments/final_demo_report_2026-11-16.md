# Final demo: measured BM25 smoke run

This report records one actual run of `experiments/final_demo.py`. It evaluates
BM25 only; it is separate from the eight-document BM25/dense/hybrid comparison
in `hybrid_retrieval_report_2026-11-13.md`.

## Configuration and provenance

- Command: `.venv\Scripts\python.exe experiments\final_demo.py`
- Dataset: `omnisearch-manual-demo` v1.0.0, project-authored synthetic text,
  CC0-1.0; 3 queries, 3 corpus items, 4 judged pairs.
- Split: test; manually assigned grades; dataset fingerprint:
  `9a73bb52ca4a212cbc6ce6dc0ee6ec1a9926b10b02172412d20ff1c3ef5e8f086`.
- Method: rank-bm25 0.2.2 BM25Okapi; K=3, seed 42, one warmup, 3 timed calls
  per query. Embedding and index construction are not used in this method.
- Run timestamp: 2026-11-16 03:26:13 UTC; Python 3.12.5 on Windows 11.
- Runtime record: PyTorch 2.14.1+cu130, Sentence Transformers 6.3.0, ChromaDB
  1.5.9, Transformers 5.23.0; RTX 3050 Laptop GPU and driver 581.83 were
  available, but BM25 itself ran on CPU.
- The checkout did not provide a Git revision (`git_revision: null`).

## Results

| Query | Retrieved at rank 1 | Grade | P@3 | R@3 | RR | nDCG@3 | Median search latency |
|---|---|---:|---:|---:|---:|---:|---:|
| q_python | doc_python | 2 | 0.3333 | 0.5000 | 1.0000 | 0.8262 | 0.1782 ms |
| q_cuda | doc_gpu | 2 | 0.3333 | 1.0000 | 1.0000 | 1.0000 | 0.2304 ms |
| q_metrics | doc_retrieval | 2 | 0.3333 | 1.0000 | 1.0000 | 1.0000 | 0.1618 ms |
| **Macro mean** | — | — | **0.3333** | **0.8333** | **1.0000** | **0.9421** | **0.1901 ms** |

Positive grades count as relevant for P/R/MRR. Graded nDCG uses gain
`2**grade - 1`. Precision divides by K even if fewer than K results are
returned; this BM25 run returned one lexical match per query. Latency is the
mean of per-query medians; timing excludes model/index startup.

## Interpretation and limitations

This verifies the demo's local dataset loader, BM25 result serialization,
metric calculations, and report generation. Three synthetic queries cannot
support a general retrieval-quality claim. The raw result file from the run is
written to the gitignored `outputs/final_demo/raw_results.csv`; rerun the command
to recreate it. The committed report retains the per-query observed outcomes
and runtime configuration without committing generated experiment artifacts.
