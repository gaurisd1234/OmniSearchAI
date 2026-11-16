from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from omnisearch_ai.evaluation import (
    DatasetManifest,
    EvaluationDataset,
    MethodSpec,
    evaluate_ranked,
    load_dataset,
    run_experiment,
)


class EvaluationFrameworkTests(unittest.TestCase):
    def test_graded_metrics_and_image_id_field(self):
        results = [{"image_id": "img-a"}, {"image_id": "img-b"}]
        metrics = evaluate_ranked(results, {"img-a": 2, "img-c": 1}, k=2, id_field="image_id")
        self.assertEqual(metrics.precision_at_k, 0.5)
        self.assertEqual(metrics.recall_at_k, 0.5)
        self.assertEqual(metrics.mrr, 1.0)
        self.assertGreater(metrics.ndcg_at_k, 0.5)

    def test_local_beir_dataset_manifest_and_qrels(self):
        root = Path(__file__).parents[1] / "datasets" / "evaluation" / "manual_demo"
        dataset = load_dataset(root, split="test")
        self.assertEqual(dataset.manifest.license, "CC0-1.0")
        self.assertEqual(dataset.qrels["q_python"]["doc_python"], 2)
        self.assertEqual(len(dataset.queries), 3)

    def test_experiment_writes_actual_results_and_report(self):
        class Retriever:
            def search(self, query, *, top_k):
                doc_id = "d1" if "one" in query else "d2"
                return [{"document_id": doc_id, "score": 0.8, "source": "fixture.txt", "content": query}]

        dataset = EvaluationDataset(
            DatasetManifest("unit", "1", "CC0-1.0", "local://unit", "unit test", "test"),
            {"q1": "one", "q2": "two"},
            {"q1": {"d1": 2}, "q2": {"d2": 1}},
            {},
        )
        with tempfile.TemporaryDirectory() as temp:
            summary = run_experiment(
                dataset,
                [MethodSpec("bm25", Retriever(), config={"k1": 1.2}, embedding_probe=lambda _q: [1.0])],
                temp,
                k=1,
                repeats=1,
                warmup=False,
                seed=9,
            )
            self.assertEqual(summary["aggregates"]["bm25"]["precision_at_k"], 1.0)
            self.assertGreaterEqual(summary["aggregates"]["bm25"]["search_latency_ms"], 0.0)
            self.assertGreaterEqual(summary["aggregates"]["bm25"]["embedding_generation_ms"], 0.0)
            self.assertTrue((Path(temp) / "experiment_report.md").exists())
            with (Path(temp) / "raw_results.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["result_id"] for row in rows], ["d1", "d2"])
            self.assertEqual([row["relevance_grade"] for row in rows], ["2", "1"])


if __name__ == "__main__":
    unittest.main()
