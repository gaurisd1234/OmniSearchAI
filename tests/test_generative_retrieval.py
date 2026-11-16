import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from omnisearch_ai.generative_retrieval import (
    DocumentIdVocabulary,
    FineTuneConfig,
    GenerativeRetriever,
    PairSplits,
    QueryDocumentPair,
    T5Components,
    compare_retrievers,
    create_pairs,
    fine_tune_t5,
    split_pairs,
)
from omnisearch_ai.ingestion.models import DocumentChunk, DocumentRecord, PageContent
from omnisearch_ai.retrieval import RetrievalResult
from experiments.generative_retrieval_experiment import load_corpus, read_judgments


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, values, **_kwargs):
        if isinstance(values, str):
            values = [values]
        return {"input_ids": torch.ones((len(values), 2), dtype=torch.long)}

    def batch_decode(self, outputs, *, skip_special_tokens):
        self.last_skip_special_tokens = skip_special_tokens
        return list(outputs)

    def save_pretrained(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "tokenizer.fixture").write_text("fake", encoding="utf-8")


class FakeGeneratingModel:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def eval(self):
        return self

    def to(self, _device):
        return self

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.outputs


def document(document_id, content="example content"):
    chunk = DocumentChunk(
        chunk_id=f"{document_id}:0",
        document_id=document_id,
        text=content,
        page_number=3,
        chunk_index=0,
        start_offset=0,
        end_offset=len(content),
        reference=f"{document_id}.pdf#page=3&chunk=0",
    )
    return DocumentRecord(
        document_id=document_id,
        source=Path(f"{document_id}.pdf"),
        content_hash=f"hash-{document_id}",
        file_type="pdf",
        pages=(PageContent(3, content),),
        chunks=(chunk,),
        metadata={"topic": "research"},
    )


class PairAndVocabularyTests(unittest.TestCase):
    def test_synthetic_jsonl_fixture_keeps_test_queries_out_of_training(self):
        fixture_dir = Path(__file__).resolve().parent / "fixtures" / "generative"
        records = load_corpus(fixture_dir / "corpus.jsonl")
        known_ids = {record.document_id for record in records}
        splits, queries, relevance = read_judgments(
            fixture_dir / "judgments.jsonl",
            known_ids,
            validation_fraction=0.25,
            seed=42,
        )

        train_queries = {pair.query for pair in splits.train}
        validation_queries = {pair.query for pair in splits.validation}
        self.assertEqual(len(records), 3)
        self.assertEqual(len(queries), 3)
        self.assertEqual(set(queries), set(relevance))
        self.assertFalse(train_queries & validation_queries)
        self.assertFalse(train_queries & set(queries.values()))
        self.assertFalse(validation_queries & set(queries.values()))

    def test_pairs_are_deduplicated_and_split_by_query_reproducibly(self):
        pairs = create_pairs(
            {
                "query c": ["doc-c"],
                "query a": ["doc-a", "doc-a"],
                "query b": ["doc-b"],
                "query d": ["doc-d"],
            },
            known_document_ids={"doc-a", "doc-b", "doc-c", "doc-d"},
        )
        split = split_pairs(pairs, validation_fraction=0.25, seed=7)
        again = split_pairs(pairs, validation_fraction=0.25, seed=7)

        self.assertEqual(split, again)
        self.assertEqual(len(pairs), 4)
        self.assertFalse({pair.query for pair in split.train} & {pair.query for pair in split.validation})
        with self.assertRaisesRegex(ValueError, "unknown document ID"):
            create_pairs({"bad query": ["missing"]}, known_document_ids={"known"})

    def test_document_id_vocabulary_is_sorted_stable_and_round_trips(self):
        first = DocumentIdVocabulary.from_document_ids(["z", "a", "m", "a"])
        second = DocumentIdVocabulary.from_document_ids(["m", "z", "a"])
        self.assertEqual(first.document_ids, ("a", "m", "z"))
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(first.encode("m"), "<|docid_00000001|>")
        self.assertEqual(first.decode("<|docid_00000002|>"), "z")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vocabulary.json"
            first.save(path)
            self.assertEqual(DocumentIdVocabulary.load(path), first)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["document_ids"][0] = "tampered"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                DocumentIdVocabulary.load(path)


class GenerativeRetrieverTests(unittest.TestCase):
    def test_generates_valid_candidates_skipping_invalid_duplicate_and_unmapped_ids(self):
        vocabulary = DocumentIdVocabulary.from_document_ids(["doc-a", "doc-b", "doc-c"])
        token = vocabulary.document_to_token
        model = FakeGeneratingModel(
            [
                "not an identifier",
                "<|docid_00000000|> extra text",
                token["doc-a"],
                token["doc-a"],
                token["doc-b"],
                token["doc-c"],
                "<|docid_99999999|>",
            ]
        )
        indexed = {"doc-a": document("doc-a"), "doc-b": document("doc-b")}
        retriever = GenerativeRetriever(
            model,
            FakeTokenizer(),
            vocabulary,
            indexed,
            candidate_sequences=6,
        )

        results = retriever.search("find research", top_k=3)

        self.assertEqual([item.document_id for item in results], ["doc-a", "doc-b"])
        self.assertEqual(results[0].method, "generative")
        self.assertEqual(results[0].source, "doc-a.pdf")
        self.assertEqual(results[0].page_number, 3)
        self.assertEqual(results[0].metadata["generated_document_id_token"], token["doc-a"])
        self.assertEqual(retriever.last_stats.invalid_sequences, 3)
        self.assertEqual(retriever.last_stats.duplicate_ids, 1)
        self.assertEqual(retriever.last_stats.unmapped_ids, 1)
        self.assertEqual(model.calls[0]["num_return_sequences"], 3)

    def test_retrieval_filter_and_blank_query(self):
        vocabulary = DocumentIdVocabulary.from_document_ids(["doc-a"])
        model = FakeGeneratingModel([vocabulary.encode("doc-a")])
        retriever = GenerativeRetriever(
            model,
            FakeTokenizer(),
            vocabulary,
            {"doc-a": document("doc-a")},
        )
        self.assertEqual(retriever.search("topic", where={"topic": "missing"}), [])
        self.assertEqual(retriever.generate_candidate_ids("  "), [])
        self.assertEqual(len(model.calls), 1)


class FakeTrainTokenizer(FakeTokenizer):
    pass


class FakeTrainModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(use_cache=True)
        self.checkpointing_enabled = False

    def forward(self, input_ids, labels):
        return SimpleNamespace(loss=(self.weight - 0.25).square() + 0.5)

    def gradient_checkpointing_enable(self):
        self.checkpointing_enabled = True

    def gradient_checkpointing_disable(self):
        self.checkpointing_enabled = False

    def save_pretrained(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.fixture").write_text("fake", encoding="utf-8")


class TrainingAndEvaluationTests(unittest.TestCase):
    def test_fine_tuning_uses_validation_and_writes_epoch_and_final_checkpoints(self):
        pairs = PairSplits(
            train=(QueryDocumentPair("question one", "doc-a"), QueryDocumentPair("question two", "doc-b")),
            validation=(QueryDocumentPair("held out question", "doc-a"),),
        )
        vocabulary = DocumentIdVocabulary.from_document_ids(["doc-a", "doc-b"])
        model = FakeTrainModel()
        components = T5Components(
            model=model,
            tokenizer=FakeTrainTokenizer(),
            vocabulary=vocabulary,
            device="cpu",
            model_name="fake-t5",
        )
        config = FineTuneConfig(
            epochs=1,
            batch_size=1,
            gradient_accumulation=2,
            max_input_tokens=12,
            max_output_tokens=4,
        )
        with tempfile.TemporaryDirectory() as directory:
            report = fine_tune_t5(pairs, vocabulary, directory, components=components, config=config)

            self.assertEqual(report.train_pairs, 2)
            self.assertEqual(report.validation_pairs, 1)
            self.assertEqual(report.effective_batch_size, 2)
            self.assertIsNotNone(report.epoch_metrics[0].validation_loss)
            self.assertTrue((Path(directory) / "checkpoint-epoch-1" / "model.fixture").exists())
            self.assertTrue((Path(directory) / "model.fixture").exists())
            self.assertTrue((Path(directory) / "document_id_vocabulary.json").exists())
            self.assertTrue(model.config.use_cache)

    def test_comparison_reports_ranking_metrics_and_latency_for_methods(self):
        record = document("doc-a")
        result = RetrievalResult(
            chunk_id=record.chunks[0].chunk_id,
            document_id="doc-a",
            document_name=record.source.name,
            source=str(record.source),
            page_number=3,
            chunk_index=0,
            score=1.0,
            content=record.chunks[0].text,
            method="test",
        )

        class FixedRetriever:
            def search(self, _query, *, top_k=5, where=None):
                return [result][:top_k]

        reports = compare_retrievers(
            {"bm25": FixedRetriever(), "dense": FixedRetriever(), "generative": FixedRetriever()},
            {"q1": "query"},
            {"q1": ["doc-a"]},
            k=1,
            repeats=2,
        )
        self.assertEqual([row.method for row in reports], ["bm25", "dense", "generative"])
        for row in reports:
            self.assertEqual(row.recall_at_k, 1.0)
            self.assertEqual(row.mrr, 1.0)
            self.assertEqual(row.ndcg_at_k, 1.0)
            self.assertGreaterEqual(row.mean_query_latency_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
