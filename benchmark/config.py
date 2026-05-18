"""Configuration centralisée du benchmark."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from benchmark.providers import normalize_ollama_base_host

ROOT_DIR = Path(__file__).resolve().parent.parent
for candidate in (ROOT_DIR / ".env", ROOT_DIR / "benchmark" / ".env"):
    if candidate.exists():
        load_dotenv(candidate, override=False)

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
GROQ_BASE = "https://api.groq.com/openai/v1"
OLLAMA_DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_TAGS_URL = (
    normalize_ollama_base_host(OLLAMA_DEFAULT_HOST).rstrip("/") + "/api/tags"
)
