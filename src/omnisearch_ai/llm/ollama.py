"""Small standard-library client for a locally running Ollama server."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from omnisearch_ai.config import load_config
from omnisearch_ai.llm.base import GenerationSettings, LLMError


class OllamaClient:
    """Generate non-streaming text through Ollama's local ``/api/generate``."""

    def __init__(
        self,
        model: str | None = None,
        *,
        base_url: str | None = None,
        settings: GenerationSettings | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        config = load_config(config_path)
        self.model_name = model or config.ollama_model
        self.base_url = (base_url or config.ollama_base_url).rstrip("/")
        self.settings = settings or GenerationSettings(
            max_tokens=config.ollama_max_tokens,
            temperature=config.ollama_temperature,
            context_window=config.ollama_context_window,
            timeout_seconds=config.ollama_timeout_seconds,
        )

    def generate(self, *, system_prompt: str, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "system": system_prompt,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": self.settings.context_window,
                "num_predict": self.settings.max_tokens,
                "temperature": self.settings.temperature,
            },
        }
        request = Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.settings.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise LLMError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise LLMError(
                f"Could not reach Ollama at {self.base_url}; ensure Ollama is running"
            ) from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise LLMError("Ollama returned an invalid JSON response") from exc

        generated = body.get("response") if isinstance(body, dict) else None
        if not isinstance(generated, str) or not generated.strip():
            raise LLMError("Ollama response did not contain generated text")
        return generated.strip()
