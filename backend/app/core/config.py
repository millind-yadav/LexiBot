from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
	"""Centralised application configuration."""

	planner_model: str = os.getenv("PLANNER_MODEL", "rule_based")
	gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
	gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
	ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
	ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
	return Settings()
