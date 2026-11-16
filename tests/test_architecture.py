import os
import unittest
from pathlib import Path
from unittest.mock import patch

import omnisearch_ai
from omnisearch_ai.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class ArchitectureSmokeTest(unittest.TestCase):
    def test_domain_modules_import(self) -> None:
        modules = (
            "ingestion",
            "embeddings",
            "retrieval",
            "multimodal",
            "rag",
            "llm",
            "generative_retrieval",
            "evaluation",
        )
        for module in modules:
            with self.subTest(module=module):
                __import__(f"omnisearch_ai.{module}")

    def test_baseline_toml_and_environment_override(self) -> None:
        config = load_config(ROOT / "configs" / "baseline.toml")
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.device, "auto")
        self.assertEqual(config.data_dir, Path("datasets"))
        self.assertEqual(config.chunk_size, 1000)
        self.assertEqual(config.chunk_overlap, 100)
        self.assertEqual(config.retrieval_top_k, 5)
        self.assertEqual(config.bm25_k1, 1.5)
        self.assertEqual(config.bm25_b, 0.75)
        self.assertEqual(config.rrf_k, 60)
        self.assertEqual(config.rrf_candidate_k, 20)
        self.assertEqual(config.cross_encoder_batch_size, 8)
        self.assertEqual(config.ollama_model, "qwen2.5:3b-instruct-q4_K_M")
        self.assertEqual(config.ollama_context_window, 4096)
        self.assertEqual(config.rag_max_context_chars, 12000)
        self.assertEqual(config.clip_model, "openai/clip-vit-base-patch32")
        self.assertEqual(config.clip_cuda_batch_size, 2)
        self.assertEqual(config.generative_model, "google/flan-t5-small")
        self.assertEqual(config.generative_batch_size, 1)
        self.assertEqual(config.generative_gradient_accumulation, 8)

        with patch.dict(
            os.environ,
            {
                "OMNISEARCH_SEED": "123",
                "OMNISEARCH_CHUNK_SIZE": "512",
                "OMNISEARCH_RETRIEVAL_TOP_K": "2",
                "OMNISEARCH_OLLAMA_MODEL": "local-test-model",
                "OMNISEARCH_RAG_MAX_CONTEXT_CHUNKS": "3",
                "OMNISEARCH_CLIP_CUDA_BATCH_SIZE": "1",
                "OMNISEARCH_GENERATIVE_BATCH_SIZE": "2",
            },
        ):
            overridden = load_config(ROOT / "configs" / "baseline.toml")
        self.assertEqual(overridden.seed, 123)
        self.assertEqual(overridden.chunk_size, 512)
        self.assertEqual(overridden.retrieval_top_k, 2)
        self.assertEqual(overridden.ollama_model, "local-test-model")
        self.assertEqual(overridden.rag_max_context_chunks, 3)
        self.assertEqual(overridden.clip_cuda_batch_size, 1)
        self.assertEqual(overridden.generative_batch_size, 2)

    def test_version_and_architecture_document(self) -> None:
        self.assertEqual(omnisearch_ai.__version__, "0.1.0")
        self.assertTrue((ROOT / "docs" / "architecture.md").is_file())


if __name__ == "__main__":
    unittest.main()
