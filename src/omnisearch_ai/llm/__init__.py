"""Provider-neutral language model interfaces and generation utilities."""

from omnisearch_ai.llm.base import GenerationSettings, LLMError, LLMProvider
from omnisearch_ai.llm.ollama import OllamaClient

__all__ = ["GenerationSettings", "LLMError", "LLMProvider", "OllamaClient"]
