"""Generate document-ID candidates with T5 and map only valid IDs to records."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from omnisearch_ai.generative_retrieval.model import _is_cuda_oom, load_t5_components
from omnisearch_ai.generative_retrieval.vocabulary import DocumentIdVocabulary
from omnisearch_ai.ingestion.models import DocumentRecord
from omnisearch_ai.retrieval.results import RetrievalResult


TOKEN_PATTERN = re.compile(r"<\|docid_\d{8}\|>")


@dataclass(frozen=True, slots=True)
class CandidateGenerationStats:
    generated_sequences: int
    invalid_sequences: int
    duplicate_ids: int
    unmapped_ids: int


class GenerativeRetriever:
    """T5 query-to-document-ID generation as a normal retrieval interface."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        vocabulary: DocumentIdVocabulary,
        indexed_documents: Mapping[str, DocumentRecord | RetrievalResult],
        *,
        device: str = "cpu",
        max_input_tokens: int = 256,
        max_output_tokens: int = 4,
        candidate_sequences: int = 20,
        model_name: str = "t5",
    ) -> None:
        if min(max_input_tokens, max_output_tokens, candidate_sequences) <= 0:
            raise ValueError("token limits and candidate_sequences must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.device = device
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.candidate_sequences = candidate_sequences
        self.model_name = model_name
        self.documents = {
            document_id: self._as_result(document_id, value)
            for document_id, value in indexed_documents.items()
            if document_id in vocabulary.document_to_token
        }
        self.last_stats = CandidateGenerationStats(0, 0, 0, 0)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        indexed_documents: Mapping[str, DocumentRecord | RetrievalResult],
        *,
        device: str = "auto",
        max_input_tokens: int = 256,
        max_output_tokens: int = 4,
        candidate_sequences: int = 20,
    ) -> GenerativeRetriever:
        checkpoint = Path(checkpoint)
        vocabulary = DocumentIdVocabulary.load(
            checkpoint / "document_id_vocabulary.json"
        )
        components = load_t5_components(
            vocabulary, checkpoint=checkpoint, device=device
        )
        return cls(
            components.model,
            components.tokenizer,
            vocabulary,
            indexed_documents,
            device=components.device,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            candidate_sequences=candidate_sequences,
            model_name=components.model_name,
        )

    @staticmethod
    def _as_result(
        document_id: str, value: DocumentRecord | RetrievalResult
    ) -> RetrievalResult:
        if isinstance(value, RetrievalResult):
            return value
        if not value.chunks:
            raise ValueError(f"indexed document {document_id} contains no chunks")
        chunk = value.chunks[0]
        return RetrievalResult(
            chunk_id=chunk.chunk_id,
            document_id=document_id,
            document_name=value.source.name,
            source=str(value.source),
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            score=0.0,
            content=chunk.text,
            method="generative",
            metadata={
                **value.metadata,
                "reference": chunk.reference,
                "file_type": value.file_type,
            },
        )

    def generate_candidate_ids(self, query: str, *, top_k: int = 5) -> list[str]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not query.strip():
            self.last_stats = CandidateGenerationStats(0, 0, 0, 0)
            return []
        torch = __import__("torch")
        count = min(
            max(top_k, self.candidate_sequences), len(self.vocabulary.document_ids)
        )
        batch = self.tokenizer(
            f"retrieve document: {query.strip()}",
            max_length=self.max_input_tokens,
            truncation=True,
            return_tensors="pt",
        )
        model_inputs = {key: value.to(self.device) for key, value in batch.items()}
        try:
            with torch.inference_mode():
                outputs = self.model.generate(
                    **model_inputs,
                    num_beams=count,
                    num_return_sequences=count,
                    max_new_tokens=self.max_output_tokens,
                    early_stopping=True,
                )
        except (RuntimeError, MemoryError) as exc:
            if self.device != "cuda" or not _is_cuda_oom(exc):
                raise
            torch.cuda.empty_cache()
            self.model.to("cpu")
            self.device = "cpu"
            model_inputs = {key: value.to("cpu") for key, value in batch.items()}
            with torch.inference_mode():
                outputs = self.model.generate(
                    **model_inputs,
                    num_beams=count,
                    num_return_sequences=count,
                    max_new_tokens=self.max_output_tokens,
                    early_stopping=True,
                )

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        token_to_id = self.vocabulary.token_to_document
        candidate_ids: list[str] = []
        seen: set[str] = set()
        invalid = duplicates = unmapped = 0
        for sequence in decoded:
            generated_tokens = TOKEN_PATTERN.findall(sequence)
            control_tokens = [
                token
                for token in getattr(self.tokenizer, "all_special_tokens", ())
                if token not in token_to_id
            ]
            cleaned = sequence
            for control_token in control_tokens:
                cleaned = cleaned.replace(control_token, "")
            cleaned = cleaned.strip()
            if len(generated_tokens) != 1 or cleaned != generated_tokens[0]:
                invalid += 1
                continue
            document_id = token_to_id.get(generated_tokens[0])
            if document_id is None:
                invalid += 1
                continue
            if document_id not in self.documents:
                unmapped += 1
                continue
            if document_id in seen:
                duplicates += 1
                continue
            seen.add(document_id)
            candidate_ids.append(document_id)
        self.last_stats = CandidateGenerationStats(
            generated_sequences=len(decoded),
            invalid_sequences=invalid,
            duplicate_ids=duplicates,
            unmapped_ids=unmapped,
        )
        return candidate_ids[:top_k]

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        document_ids = self.generate_candidate_ids(query, top_k=top_k)
        results = []
        for rank, document_id in enumerate(document_ids, start=1):
            result = self.documents[document_id]
            if where and any(result.metadata.get(key) != value for key, value in where.items()):
                continue
            metadata = dict(result.metadata)
            metadata["generated_document_id_token"] = self.vocabulary.encode(document_id)
            metadata["generation_rank"] = rank
            results.append(
                replace(
                    result,
                    score=1.0 / rank,
                    method="generative",
                    metadata=metadata,
                )
            )
        return results
