import json
import unittest
from io import BytesIO
from unittest.mock import patch

from omnisearch_ai.llm import GenerationSettings, LLMError, OllamaClient
from omnisearch_ai.rag import (
    INSUFFICIENT_ANSWER,
    INSUFFICIENT_EVIDENCE,
    RAGPipeline,
)
from omnisearch_ai.retrieval import RetrievalResult


def result(chunk_id="chunk-1", content="A blue whale is a mammal.", score=0.8):
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        document_name="whales.pdf",
        source="research/whales.pdf",
        page_number=4,
        chunk_index=0,
        score=score,
        content=content,
        method="hybrid",
        metadata={"category": "biology"},
    )


class MockRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, *, top_k=None, where=None):
        self.calls.append((query, top_k, where))
        return self.results[:top_k]


class MockLLM:
    model_name = "mock:unit-test"

    def __init__(self, response="The blue whale is a mammal [S1].", error=None):
        self.response = response
        self.error = error
        self.calls = []

    def generate(self, *, system_prompt, prompt):
        self.calls.append((system_prompt, prompt))
        if self.error:
            raise self.error
        return self.response


class RAGPipelineTests(unittest.TestCase):
    def test_generates_grounded_answer_and_source_page(self):
        retriever = MockRetriever([result()])
        llm = MockLLM()
        pipeline = RAGPipeline(retriever, llm)

        response = pipeline.answer("What is a blue whale?", where={"category": "biology"})

        self.assertIn("The blue whale is a mammal [S1].", response.answer)
        self.assertIn("whales.pdf, page 4", response.answer)
        self.assertEqual(response.citations[0].marker, "S1")
        self.assertEqual(response.citations[0].source, "research/whales.pdf")
        self.assertFalse(response.insufficient_evidence)
        self.assertEqual(
            retriever.calls,
            [("What is a blue whale?", 5, {"category": "biology"})],
        )
        self.assertIn("A blue whale is a mammal.", llm.calls[0][1])
        self.assertIn("only the supplied retrieved evidence", llm.calls[0][0])

    def test_no_retrieved_evidence_skips_generation(self):
        llm = MockLLM()
        response = RAGPipeline(MockRetriever([]), llm).answer("Unknown?")

        self.assertEqual(response.answer, INSUFFICIENT_ANSWER)
        self.assertTrue(response.insufficient_evidence)
        self.assertEqual(llm.calls, [])

    def test_model_can_explicitly_decline_insufficient_evidence(self):
        response = RAGPipeline(
            MockRetriever([result()]), MockLLM(INSUFFICIENT_EVIDENCE)
        ).answer("What is not in this evidence?")

        self.assertEqual(response.answer, INSUFFICIENT_ANSWER)
        self.assertTrue(response.insufficient_evidence)

    def test_uncited_or_unknown_source_answer_is_rejected(self):
        for text in ("An unsupported answer.", "Answer [S9]."):
            with self.subTest(text=text):
                response = RAGPipeline(
                    MockRetriever([result()]), MockLLM(text)
                ).answer("Question")
                self.assertEqual(response.answer, INSUFFICIENT_ANSWER)
                self.assertTrue(response.insufficient_evidence)

    def test_reranker_and_context_limits_are_applied(self):
        class ReverseReranker:
            def __init__(self):
                self.called = False

            def rerank(self, query, candidates):
                self.called = True
                return list(reversed(candidates))

        first = result("first", "first evidence")
        second = result("second", "second evidence")
        reranker = ReverseReranker()
        llm = MockLLM("Answer [S1].")
        response = RAGPipeline(
            MockRetriever([first, second]),
            llm,
            reranker=reranker,
            top_k=2,
            max_context_chunks=1,
            max_context_chars=200,
        ).answer("Question")

        self.assertTrue(reranker.called)
        self.assertEqual(response.citations[0].chunk_id, "second")
        self.assertIn("second evidence", llm.calls[0][1])
        self.assertNotIn("first evidence", llm.calls[0][1])

    def test_long_context_is_bounded_and_empty_question_rejected(self):
        long_result = result(content="word " * 1000)
        llm = MockLLM("Short answer [S1].")
        pipeline = RAGPipeline(
            MockRetriever([long_result]), llm, max_context_chars=180
        )
        pipeline.answer("Question")
        prompt = llm.calls[0][1]
        context = prompt.split("Evidence:\n", 1)[1].split("\n\nQuestion:", 1)[0]
        self.assertLessEqual(len(context), 180)
        with self.assertRaises(ValueError):
            pipeline.answer("  ")

    def test_provider_failure_returns_sources_and_error(self):
        response = RAGPipeline(
            MockRetriever([result()]), MockLLM(error=LLMError("server unavailable"))
        ).answer("Question")

        self.assertIn("could not generate", response.answer)
        self.assertEqual(response.error, "server unavailable")
        self.assertFalse(response.insufficient_evidence)
        self.assertEqual(response.citations[0].page_number, 4)


class OllamaClientTests(unittest.TestCase):
    def test_posts_non_streaming_request_with_bounded_generation_options(self):
        client = OllamaClient(
            "test-model",
            base_url="http://localhost:11434/",
            settings=GenerationSettings(
                max_tokens=48,
                temperature=0.2,
                context_window=2048,
                timeout_seconds=3,
            ),
        )
        response = BytesIO(b'{"response":" generated text "}')

        with patch("omnisearch_ai.llm.ollama.urlopen", return_value=response) as open_url:
            # BytesIO itself does not implement the context manager protocol in
            # every supported Python version, so supply a small response mock.
            class Response:
                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return False

                def read(self):
                    return response.read()

            open_url.return_value = Response()
            generated = client.generate(system_prompt="system", prompt="question")

        request = open_url.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(generated, "generated text")
        self.assertEqual(payload["model"], "test-model")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["options"]["num_ctx"], 2048)
        self.assertEqual(payload["options"]["num_predict"], 48)
        self.assertEqual(open_url.call_args.kwargs["timeout"], 3)


if __name__ == "__main__":
    unittest.main()
