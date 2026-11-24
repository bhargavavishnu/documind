"""Configuration and provider resolution for DocuMind."""

from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

_PLACEHOLDER_KEYS = {"", "your_openai_api_key_here", "sk-your-key"}


@dataclass(frozen=True)
class Settings:
    """Runtime settings for DocuMind ingestion, retrieval, and generation."""

    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    provider_mode: str = "auto"
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 4

    @classmethod
    def from_env(cls, openai_api_key_override: str | None = None) -> "Settings":
        """Build settings from environment variables with optional key override."""
        key = (
            openai_api_key_override
            if openai_api_key_override is not None
            else os.getenv("OPENAI_API_KEY", "")
        ).strip()
        return cls(
            openai_api_key=key,
            openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip(),
            openai_embedding_model=os.getenv(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
            ).strip(),
            provider_mode=os.getenv("DOCUMIND_PROVIDER", "auto").strip().lower(),
            local_embedding_model=os.getenv(
                "LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ).strip(),
        )

    def has_valid_openai_key(self) -> bool:
        """Return True when a non-placeholder OpenAI API key is configured."""
        return self.openai_api_key not in _PLACEHOLDER_KEYS

    def resolve_provider(self) -> str:
        """Determine active provider ('openai' or 'local')."""
        if self.provider_mode == "local":
            return "local"
        if self.provider_mode == "openai":
            if not self.has_valid_openai_key():
                raise ValueError(
                    "DOCUMIND_PROVIDER=openai requires a valid OPENAI_API_KEY."
                )
            return "openai"
        return "openai" if self.has_valid_openai_key() else "local"
