import unittest
from omnisearch_ai.evaluation import evaluate_at_k, macro_average
from omnisearch_ai.retrieval import (
    CrossEncoderReranker,
    HybridRetriever,
    RetrievalResult,
)


def result(chunk_id: str, document_id: str | None = None) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=document_id or chunk_id,
        document_name=f"{chunk_id}.txt",
        source=f"{chunk_id}.txt",
        page_number=2,
        chunk_index=0,
        score=0.5,
        content=f"content for {chunk_id}",
        method="test",
        metadata={"author": "Ada"},
    )


class FixedRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, *, top_k=None, where=None):
        self.calls.append((query, top_k, where))
        return self.results[:top_k]


class HybridRetrievalTests(unittest.TestCase):
    def test_rrf_combines_independent_rankings_and_preserves_metadata(self) -> None:
        bm25 = FixedRetriever([result("a"), result("b")])
        dense = FixedRetriever([result("b"), result("c")])
        hybrid = HybridRetriever(
            bm25,
            dense,
            top_k=3,
            candidate_k=7,
            rrf_k=10,
            bm25_weight=1.0,
            dense_weight=1.0,
        )

        results = hybrid.search("test query", where={"author": "Ada"})

        self.assertEqual([item.chunk_id for item in results], ["b", "a", "c"])
        self.assertAlmostEqual(results[0].score, 1 / 12 + 1 / 11)
        self.assertEqual(results[0].metadata["rrf_bm25_rank"], 2)
        self.assertEqual(results[0].metadata["rrf_dense_rank"], 1)
        self.assertEqual(results[0].method, "hybrid")
        self.assertEqual(bm25.calls, [("test query", 7, {"author": "Ada"})])
        self.assertEqual(dense.calls, [("test query", 7, {"author": "Ada"})])

    def test_optional_cross_encoder_reranks_fused_candidates(self) -> None:
        class FakeCrossEncoder:
            def predict(self, pairs, **_kwargs):
                return [0.1 if "a" in pair[1] else 0.9 for pair in pairs]

        bm25 = FixedRetriever([result("a"), result("b")])
        dense = FixedRetriever([result("a"), result("b")])
        hybrid = HybridRetriever(
            bm25,
            dense,
            top_k=2,
            candidate_k=2,
            reranker=CrossEncoderReranker(model=FakeCrossEncoder()),
        )

        results = hybrid.search("test query")

        self.assertEqual([item.chunk_id for item in results], ["b", "a"])
        self.assertEqual(results[0].method, "hybrid_reranked")
        self.assertIn("rrf_score", results[0].metadata)
        self.assertEqual(results[0].metadata["author"], "Ada")

    def test_rrf_parameters_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            HybridRetriever(FixedRetriever([]), FixedRetriever([]), rrf_k=0)
        with self.assertRaises(ValueError):
            HybridRetriever(
                FixedRetriever([]), FixedRetriever([]), bm25_weight=0, dense_weight=0
            )

    def test_metrics_use_document_level_deduplication_and_macro_average(self) -> None:
        ranked = [
            result("a-chunk-1", "a"),
            result("a-chunk-2", "a"),
            result("b-chunk", "b"),
        ]
        metrics = evaluate_at_k(ranked, {"b", "c"}, k=2)
        self.assertEqual(metrics.precision_at_k, 0.5)
        self.assertEqual(metrics.recall_at_k, 0.5)
        self.assertEqual(metrics.mrr, 0.5)
        self.assertAlmostEqual(metrics.ndcg_at_k, 0.3868528072)
        average = macro_average([metrics, metrics])
        self.assertEqual(average, metrics)
        self.assertEqual(macro_average([]).mrr, 0.0)


if __name__ == "__main__":
    unittest.main()
