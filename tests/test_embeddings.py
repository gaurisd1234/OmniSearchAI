from __future__ import annotations

import unittest

import numpy as np
import torch

from omnisearch_ai.embeddings import SentenceTransformerEmbedder


class FakeSentenceModel:
    max_seq_length = 128

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.normalized: list[bool] = []
        self.device = "cpu"

    def to(self, device: str) -> FakeSentenceModel:
        self.device = str(device)
        return self

    def encode(
        self,
        texts,
        *,
        batch_size,
        show_progress_bar,
        convert_to_numpy,
        normalize_embeddings,
    ):
        self.batch_sizes.append(batch_size)
        self.normalized.append(normalize_embeddings)
        vectors = np.asarray(
            [[float(len(text)), 1.0, 2.0] for text in texts], dtype=np.float32
        )
        if normalize_embeddings:
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors

    def get_sentence_embedding_dimension(self) -> int:
        return 3


class EmbeddingTests(unittest.TestCase):
    def test_cpu_embedding_shape_normalization_and_measurements(self) -> None:
        fake_model = FakeSentenceModel()
        encoder = SentenceTransformerEmbedder(
            device="cpu",
            cpu_batch_size=4,
            max_seq_length=128,
            model=fake_model,
        )

        result = encoder.encode_texts(["short text", "a longer sentence"])

        self.assertEqual(result.vectors.shape, (2, 3))
        np.testing.assert_allclose(np.linalg.norm(result.vectors, axis=1), [1.0, 1.0])
        self.assertEqual(fake_model.max_seq_length, 128)
        self.assertEqual(fake_model.batch_sizes, [4])
        self.assertEqual(fake_model.normalized, [True])
        self.assertEqual(result.metrics.device, "cpu")
        self.assertEqual(result.metrics.num_texts, 2)
        self.assertGreaterEqual(result.metrics.elapsed_seconds, 0.0)
        self.assertGreater(result.metrics.peak_rss_bytes, 0)

    def test_empty_input_returns_correct_embedding_dimension(self) -> None:
        encoder = SentenceTransformerEmbedder(device="cpu", model=FakeSentenceModel())

        result = encoder.encode_texts([])

        self.assertEqual(result.vectors.shape, (0, 3))
        self.assertEqual(result.metrics.num_texts, 0)

    def test_batch_size_and_device_validation(self) -> None:
        with self.assertRaises(ValueError):
            SentenceTransformerEmbedder(device="cpu", cuda_batch_size=0, model=FakeSentenceModel())
        with self.assertRaises(ValueError):
            SentenceTransformerEmbedder(device="tpu", model=FakeSentenceModel())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_batch_is_reduced_after_out_of_memory(self) -> None:
        class LimitedFakeModel(FakeSentenceModel):
            def encode(self, texts, **kwargs):
                if self.device == "cuda" and kwargs["batch_size"] > 2:
                    raise RuntimeError("CUDA out of memory. simulated test condition")
                return super().encode(texts, **kwargs)

        model = LimitedFakeModel()
        encoder = SentenceTransformerEmbedder(
            device="cuda", cuda_batch_size=8, model=model
        )

        result = encoder.encode_texts(["sample"] * 3)

        self.assertEqual(result.metrics.device, "cuda")
        self.assertEqual(result.metrics.batch_size, 2)
        self.assertEqual(model.batch_sizes, [2])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_out_of_memory_falls_back_to_cpu_at_batch_one(self) -> None:
        class CudaUnavailableFakeModel(FakeSentenceModel):
            def encode(self, texts, **kwargs):
                if self.device == "cuda":
                    raise RuntimeError("CUDA out of memory. simulated test condition")
                return super().encode(texts, **kwargs)

        model = CudaUnavailableFakeModel()
        encoder = SentenceTransformerEmbedder(
            device="cuda", cuda_batch_size=1, cpu_batch_size=4, model=model
        )

        result = encoder.encode_texts(["sample"])

        self.assertEqual(result.metrics.device, "cpu")
        self.assertIn("out of memory", result.metrics.fallback_reason or "")
        self.assertEqual(model.batch_sizes, [4])


if __name__ == "__main__":
    unittest.main()
