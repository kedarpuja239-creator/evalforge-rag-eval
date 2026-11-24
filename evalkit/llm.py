"""One place that creates the LLM client, used by answer generation, the judge and agentic chunking."""
from __future__ import annotations

from . import config

_client = None


def client():
    """OpenAI client. Reads OPENAI_API_KEY from the environment/.env. Point EVALKIT_LLM_BASE_URL at any
    OpenAI-compatible server (for example Ollama: http://localhost:11434/v1, key can be any text)."""
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(base_url=config.LLM_BASE_URL)
    return _client


def friendly_error(e: Exception) -> str:
    name = type(e).__name__
    if name == "AuthenticationError":
        return "OpenAI rejected the API key (401). Put a valid key in .env as OPENAI_API_KEY=..."
    if name == "RateLimitError":
        return "OpenAI says quota or rate limit exceeded (429). Check billing credit."
    return f"{name}: {e}"
