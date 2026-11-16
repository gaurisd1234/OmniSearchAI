# Embedding smoke benchmark — 2026-11-13

This records one small engineering smoke run, not a corpus benchmark. The
reproducible script is `experiments/benchmark_embeddings.py`; it uses eight
fixed, short research-related sentences and no dataset.

## Configuration

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Model revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- Max sequence length: 256 tokens
- CUDA batch size: 8; normalized 384-dimensional embeddings
- Hardware: NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM; driver 581.83
- Python: 3.12.5; PyTorch: 2.14.1+cu130 (CUDA runtime 13.0)
- Sentence Transformers: 6.3.0; Transformers: 5.23.0
- Warm-up: one unmeasured sentence; then three measured batches of eight

## Observed results

| Run | Time |
| --- | ---: |
| 1 | 0.0718 s |
| 2 | 0.0137 s |
| 3 | 0.0106 s |
| Median | 0.0137 s |

Median throughput for these short strings was 581.9 texts/s. This is specific
to this small warm-model run and should not be projected onto longer chunks or
larger corpora.

- Peak CUDA allocated: 101,956,608 bytes
- Peak CUDA reserved: 111,149,056 bytes
- Process RSS before encoding: 1,312,772,096 bytes
- Sampled peak process RSS: 1,312,927,744 bytes (delta 155,648 bytes)
- RSS sampling interval: 10 ms; brief peaks can be missed
- Device selected: `cuda`, `NVIDIA GeForce RTX 3050 Laptop GPU`

The memory figures include the loaded model resident during encoding. They are
observations from the measured calls, not guaranteed upper bounds. The CUDA
peak also reflects the model allocation already present when per-call peak
tracking began.
