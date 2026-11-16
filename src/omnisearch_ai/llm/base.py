"""Replaceable language-model generation interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class LLMProvider(Protocol):
    """Minimal provider contract used by RAG pipelines and experiments."""

    @property
    def model_name(self) -> str: ...

    def generate(self, *, system_prompt: str, prompt: str) -> str: ...


class LLMError(RuntimeError):
    """Raised when a language-model provider cannot return a valid response."""


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    """Generation controls shared by local and remote provider adapters."""

    max_tokens: int = 512
    temperature: float = 0.1
    context_window: int = 4096
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        if self.max_tokens <= 0 or self.context_window <= 0:
            raise ValueError("max_tokens and context_window must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative")
