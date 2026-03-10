"""Text splitting utilities for RAG chunk creation."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

from langchain_core.documents import Document

from src.utils import PageRecord, build_chunk_id

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover
    from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_text_splitter(
    chunk_size: int = 900, chunk_overlap: int = 150
) -> RecursiveCharacterTextSplitter:
    """Create the default recursive character splitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def page_records_to_documents(records: Iterable[PageRecord]) -> list[Document]:
    """Convert page records into LangChain Document objects."""
    documents: list[Document] = []
    for record in records:
        if not record.text.strip():
            continue
        documents.append(
            Document(page_content=record.text, metadata=record.to_metadata())
        )
    return documents


def split_documents(
    documents: Sequence[Document],
    splitter: RecursiveCharacterTextSplitter | None = None,
) -> list[Document]:
    """Split documents while preserving and extending metadata."""
    active_splitter = splitter or get_text_splitter()
    chunks = active_splitter.split_documents(list(documents))

    chunk_counters: dict[tuple[str, int], int] = defaultdict(int)
    for chunk in chunks:
        metadata = chunk.metadata or {}
        source = str(metadata.get("source", "unknown"))
        page = int(metadata.get("page", -1))

        key = (source, page)
        chunk_index = chunk_counters[key]
        chunk_counters[key] += 1

        chunk.metadata["chunk_index"] = chunk_index
        chunk.metadata["chunk_id"] = build_chunk_id(
            source=source,
            page=page,
            chunk_index=chunk_index,
            text=chunk.page_content,
        )

    return chunks


def split_page_records(
    records: Iterable[PageRecord],
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Convert page records to chunked documents in one step."""
    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    page_docs = page_records_to_documents(records)
    return split_documents(page_docs, splitter=splitter)
