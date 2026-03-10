"""Embedding model initialization for the RAG pipeline."""

from __future__ import annotations

import os

from langchain_ollama import OllamaEmbeddings

DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


def get_embeddings(
    model: str = DEFAULT_EMBEDDING_MODEL, base_url: str | None = None
) -> OllamaEmbeddings:
    """Return an Ollama embedding client for local embedding generation."""
    resolved_base_url = base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
    return OllamaEmbeddings(model=model, base_url=resolved_base_url)


def check_embeddings_ready(embeddings: OllamaEmbeddings) -> bool:
    """Quick connectivity check against the embedding model."""
    try:
        _ = embeddings.embed_query("health-check")
        return True
    except Exception:
        return False
