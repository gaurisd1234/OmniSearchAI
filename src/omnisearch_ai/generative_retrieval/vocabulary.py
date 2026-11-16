"""Stable special-token vocabulary mapping indexed document IDs to T5 tokens."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class DocumentIdVocabulary:
    document_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(set(self.document_ids)) != len(self.document_ids):
            raise ValueError("document IDs in vocabulary must be unique")
        if not self.document_ids:
            raise ValueError("document ID vocabulary cannot be empty")

    @classmethod
    def from_document_ids(cls, document_ids: Iterable[str]) -> DocumentIdVocabulary:
        ordered = tuple(sorted({str(document_id) for document_id in document_ids}))
        return cls(ordered)

    @property
    def tokens(self) -> tuple[str, ...]:
        return tuple(f"<|docid_{index:08d}|>" for index in range(len(self.document_ids)))

    @property
    def document_to_token(self) -> dict[str, str]:
        return dict(zip(self.document_ids, self.tokens))

    @property
    def token_to_document(self) -> dict[str, str]:
        return dict(zip(self.tokens, self.document_ids))

    @property
    def fingerprint(self) -> str:
        payload = "\n".join(self.document_ids).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def encode(self, document_id: str) -> str:
        try:
            return self.document_to_token[document_id]
        except KeyError as exc:
            raise KeyError(f"document ID is not in vocabulary: {document_id}") from exc

    def decode(self, token: str) -> str | None:
        return self.token_to_document.get(token)

    def add_to_tokenizer(self, tokenizer: Any) -> int:
        """Append doc ID markers as atomic special tokens to a T5 tokenizer."""
        return int(tokenizer.add_tokens(list(self.tokens), special_tokens=True))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "fingerprint": self.fingerprint,
            "document_ids": list(self.document_ids),
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> DocumentIdVocabulary:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_version") != 1:
            raise ValueError("unsupported document ID vocabulary format")
        vocab = cls(tuple(payload["document_ids"]))
        if payload.get("fingerprint") != vocab.fingerprint:
            raise ValueError("document ID vocabulary fingerprint mismatch")
        return vocab
