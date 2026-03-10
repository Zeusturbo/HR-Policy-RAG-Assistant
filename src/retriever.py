"""Retriever creation and query helpers."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


def create_retriever(
    vectorstore,
    k: int = 4,
    search_type: str = "similarity",
) -> BaseRetriever:
    """Create a retriever from a Chroma vectorstore."""
    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )


def retrieve_documents(retriever: BaseRetriever, query: str) -> list[Document]:
    """Retrieve top-k relevant documents for a query."""
    results = retriever.invoke(query)
    return list(results or [])
