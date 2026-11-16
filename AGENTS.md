# OmniSearch AI development instructions

## Project goals

Build a student-friendly research codebase for generative retrieval, multimodal
retrieval, RAG, LLMs, and AI-driven search. Prefer small, reproducible
experiments that explain their assumptions and limitations.

## Development practices

- Target Python 3.12 and keep reusable package code under `src/omnisearch_ai/`.
- Put automated checks in `tests/`; keep experiments reproducible and record
  configuration, data provenance, and evaluation metrics.
- Explain non-obvious research and engineering choices in comments or docs.
- Avoid adding dependencies until needed. Never download large models or datasets
  without an explicit project need; document expected disk, RAM, and VRAM use.
- Do not commit secrets, private data, model weights, datasets, caches, or outputs.
- Preserve existing files. Inspect a file before modifying it, and keep changes
  focused on the requested work.
- Prefer CPU-capable smoke checks and small fixtures. Treat GPU availability as
  optional and check it at runtime before using CUDA.
- Report commands run, outcomes, and known limitations when finishing a task.

## Research quality

- Separate training, validation, and test data where relevant; avoid test leakage.
- Compare retrieval methods with explicit baselines and appropriate metrics.
- Record random seeds and software versions when they affect results.
- Do not present benchmark scores without documenting datasets and evaluation
  settings.

## Mandatory Development Workflow

For every development task:

- Inspect existing code before modifying it.
- Implement only the requested scope.
- Do not create duplicate modules.
- Do not leave placeholder functions presented as completed features.
- Do not fabricate test results.
- Run relevant unit and integration tests.
- Fix errors before declaring completion.
- Record actual GPU and CPU resource usage when relevant.
- Maintain documentation and dependency versions.
- Explain all important architectural decisions.
- Provide a summary of modified files and remaining issues.

Never begin the next major development phase until the current phase has a working, tested implementation.