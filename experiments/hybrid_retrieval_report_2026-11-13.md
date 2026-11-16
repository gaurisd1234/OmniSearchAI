# Hybrid retrieval smoke evaluation

## Setup

Command: `.venv\Scripts\python.exe experiments\compare_hybrid_retrieval.py --repeats 3`

This run uses 8 synthetic, one-chunk documents and 4 hand-labeled queries, with
one relevant document per query. It reports document-level macro metrics at
K=3. Each query/method had one unmeasured warm-up and three timed searches;
latency is the median of the three runs, then summarized across the four query
medians. The sentence embedding model was already available locally. Dense
search ran on CUDA (RTX 3050 Laptop GPU); model loading and initial index
creation are excluded from query latency. Cross-encoder reranking was not
enabled.

Software versions: Python 3.12.5, PyTorch 2.14.1+cu130, Sentence Transformers
6.3.0, ChromaDB 1.5.9, rank-bm25 0.2.2. RRF used `k=60`, candidate depth 20,
and equal weights of 1.0.

## Results

| Method | Precision@3 | Recall@3 | MRR | nDCG@3 | Mean per-query median latency | Median per-query median latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.333 | 1.000 | 1.000 | 1.000 | 0.193 ms | 0.209 ms |
| Dense | 0.333 | 1.000 | 1.000 | 1.000 | 18.264 ms | 18.653 ms |
| Hybrid RRF | 0.333 | 1.000 | 1.000 | 1.000 | 16.155 ms | 15.185 ms |

All systems placed the single judged relevant document first for each query,
so the ranking metrics tie on this fixture. Precision is one third because
only one of the three requested results is relevant. Hybrid was slower than
BM25 and somewhat faster than dense in this run; this is a machine-specific
observation, not a stable performance claim.

## Limits

The fixture is tiny, synthetic, and deliberately simple. Four judgments cannot
support a general conclusion about retrieval quality, latency at scale, or
which method is preferable. Latency includes query embedding and vector lookup
for dense/hybrid, but excludes model startup, indexing, and disk/network
initialization. Use representative held-out queries, multiple relevance grades,
and a larger corpus before making research or product decisions. Raw BM25,
cosine, and RRF scores have different scales and are not compared directly.

Run `experiments/compare_hybrid_retrieval.py --rerank` to include the optional
cross-encoder variant; that requires loading the configured cross-encoder model
and may download its weights on first use.
