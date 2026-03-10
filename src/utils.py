"""Shared helpers for the HR policy RAG project."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterable


DEFAULT_FALLBACK_RESPONSE = (
    "I could not find this information in the uploaded HR policy documents."
)


@dataclass(frozen=True)
class PageRecord:
    """Represents extracted text for one PDF page."""

    text: str
    page_number: int
    source_file: str

    def to_metadata(self) -> dict[str, Any]:
        """Convert the page record to metadata used by chunked documents."""
        return {"source": self.source_file, "page": self.page_number}


def ensure_directory(path: str | Path) -> Path:
    """Create directory when missing and return a resolved path."""
    directory = Path(path).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def is_pdf_file(path: str | Path) -> bool:
    """Return True when the file extension is .pdf."""
    return Path(path).suffix.lower() == ".pdf"


def normalize_text(text: str) -> str:
    """Normalize extracted text while preserving paragraph breaks."""
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def build_chunk_id(source: str, page: int, chunk_index: int, text: str) -> str:
    """Build a stable chunk id based on source metadata and content."""
    payload = f"{source}|{page}|{chunk_index}|{text[:160]}"
    digest = sha1(payload.encode("utf-8")).hexdigest()  # noqa: S324
    return f"{source}-p{page}-c{chunk_index}-{digest[:12]}"


def format_source_reference(metadata: dict[str, Any]) -> str:
    """Create a source label from document metadata."""
    source = str(metadata.get("source", "unknown"))
    page = metadata.get("page")
    if page is None:
        return source
    return f"{source} - page {page}"


def unique_sources_from_documents(documents: Iterable[Any]) -> list[str]:
    """Return unique source labels from LangChain documents."""
    seen: set[str] = set()
    ordered: list[str] = []
    for doc in documents:
        metadata = getattr(doc, "metadata", {}) or {}
        label = format_source_reference(metadata)
        if label not in seen:
            seen.add(label)
            ordered.append(label)
    return ordered
