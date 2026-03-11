"""Chroma vector store setup and ingestion helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document

from src.embeddings import get_embeddings
from src.utils import ensure_directory

try:
    from langchain_chroma import Chroma
except ImportError:  # pragma: no cover
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:  # pragma: no cover
        from langchain.vectorstores import Chroma


DEFAULT_DB_DIR = "chroma_db"
DEFAULT_COLLECTION_NAME = "hr_policy_docs"


def get_vectorstore(
    persist_directory: str | Path = DEFAULT_DB_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_function=None,
) -> Chroma:
    """Create or load a persistent Chroma vector store."""
    db_path = ensure_directory(persist_directory)
    embeddings = embedding_function or get_embeddings()
    return Chroma(
        collection_name=collection_name,
        persist_directory=str(db_path),
        embedding_function=embeddings,
    )


def get_vector_count(vectorstore: Chroma) -> int:
    """Return total indexed vectors in the Chroma collection."""
    try:
        return int(vectorstore._collection.count())  # noqa: SLF001
    except Exception:
        return 0


def is_vectorstore_empty(vectorstore: Chroma) -> bool:
    """Return True when the collection has no vectors."""
    return get_vector_count(vectorstore) == 0


def ingest_documents(
    documents: Sequence[Document],
    persist_directory: str | Path = DEFAULT_DB_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_function=None,
    reset: bool = False,
) -> Chroma:
    """Ingest chunked documents into persistent Chroma storage."""
    if not documents:
        raise ValueError("No documents were provided for ingestion.")

    db_path = Path(persist_directory).expanduser()
    vectorstore = get_vectorstore(
        persist_directory=db_path,
        collection_name=collection_name,
        embedding_function=embedding_function,
    )

    if reset:
        try:
            if hasattr(vectorstore, "delete_collection"):
                vectorstore.delete_collection()
            vectorstore = get_vectorstore(
                persist_directory=db_path,
                collection_name=collection_name,
                embedding_function=embedding_function,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to reset Chroma collection. "
                "Close other running app instances and try again."
            ) from exc

    ids: list[str] = []
    for idx, doc in enumerate(documents):
        chunk_id = str(doc.metadata.get("chunk_id", f"doc-{idx}"))
        ids.append(chunk_id)

    vectorstore.add_documents(list(documents), ids=ids)
    if hasattr(vectorstore, "persist"):
        vectorstore.persist()

    return vectorstore
