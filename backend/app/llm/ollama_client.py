from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


class OllamaClientError(Exception):
    """Raised when the Ollama API returns an error or malformed payload."""


class OllamaClient:
    """Client for local Ollama API."""

    def __init__(self, base_url: str, model: str, timeout: float = 60.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._endpoint = f"{self._base_url}/api/generate"

    def generate_text(self, prompt: str) -> str:
        """Send a prompt to Ollama and return the text response."""

        payload: Dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for planning
            }
        }

        try:
            response = httpx.post(
                self._endpoint,
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("response", "")

        except httpx.TimeoutException:
            raise OllamaClientError(f"Ollama request timed out after {self._timeout}s")
        except httpx.HTTPStatusError as e:
            raise OllamaClientError(f"Ollama API error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise OllamaClientError(f"Unexpected error communicating with Ollama: {str(e)}")
