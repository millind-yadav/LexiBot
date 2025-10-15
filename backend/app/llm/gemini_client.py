from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class GeminiClientError(Exception):
    """Raised when the Gemini API returns an error or malformed payload."""


class GeminiClient:
    """Minimal client for Google's Gemini Generative Language API."""

    def __init__(self, api_key: str, model: str, timeout: float = 30.0) -> None:
        if not api_key:
            raise ValueError("Gemini API key must be provided")

        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        )

    def generate_text(self, prompt: str) -> str:
        """Send a prompt to Gemini and return the concatenated text response."""

        payload: dict[str, Any] = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt,
                        }
                    ]
                }
            ],
        }

        try:
            response = httpx.post(
                self._endpoint,
                params={"key": self._api_key},
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - network failure paths
            logger.exception("Gemini API request failed")
            raise GeminiClientError("Unable to reach Gemini API") from exc

        data = response.json()
        try:
            candidates = data["candidates"]
            first_candidate = candidates[0]
            parts = first_candidate.get("content", {}).get("parts", [])
            text_response = "".join(part.get("text", "") for part in parts).strip()
        except (KeyError, IndexError) as exc:
            logger.debug("Unexpected Gemini payload: %s", data)
            raise GeminiClientError("Gemini API returned an unexpected payload structure") from exc

        if not text_response:
            logger.debug("Empty Gemini response: %s", data)
            raise GeminiClientError("Gemini API returned an empty response")

        return text_response
