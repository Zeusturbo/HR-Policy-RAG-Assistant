"""LangGraph workflow for retrieval-augmented answering."""

from __future__ import annotations

import os
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from src.retriever import retrieve_documents
from src.utils import DEFAULT_FALLBACK_RESPONSE

DEFAULT_LLM_MODEL = "qwen2.5:7b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an HR policy assistant. "
                "Answer only using the provided context. "
                "If the answer is not explicitly present in the context, "
                "reply exactly with: {fallback_response}"
            ),
        ),
        (
            "human",
            "Question:\n{question}\n\nContext:\n{context}",
        ),
    ]
)


class RAGState(TypedDict, total=False):
    """State passed through the LangGraph workflow."""

    question: str
    documents: list[Document]
    context: str
    answer: str


def _render_context(documents: list[Document]) -> str:
    """Render retrieved documents into a single context string."""
    context_blocks: list[str] = []
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")
        context_blocks.append(
            f"[Source: {source}, Page: {page}]\n{doc.page_content.strip()}"
        )
    return "\n\n".join(context_blocks).strip()


def build_llm(
    model: str = DEFAULT_LLM_MODEL, base_url: str | None = None
) -> ChatOllama:
    """Create the Ollama chat model used for answer generation."""
    resolved_base_url = base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
    return ChatOllama(model=model, base_url=resolved_base_url, temperature=0)


def build_rag_graph(
    retriever,
    llm: ChatOllama | None = None,
    fallback_response: str = DEFAULT_FALLBACK_RESPONSE,
):
    """Build and compile a simple retrieve -> answer LangGraph flow."""
    model = llm or build_llm()

    def retrieve_node(state: RAGState) -> RAGState:
        question = state.get("question", "").strip()
        if not question:
            return {"documents": [], "context": ""}

        docs = retrieve_documents(retriever, question)
        context = _render_context(docs)
        return {"documents": docs, "context": context}

    def answer_node(state: RAGState) -> RAGState:
        question = state.get("question", "").strip()
        context = state.get("context", "").strip()
        docs = state.get("documents", [])

        if not question or not context or not docs:
            return {"answer": fallback_response}

        prompt_value = RAG_PROMPT.invoke(
            {
                "question": question,
                "context": context,
                "fallback_response": fallback_response,
            }
        )
        response = model.invoke(prompt_value)
        answer = str(getattr(response, "content", "")).strip()
        if not answer:
            answer = fallback_response
        return {"answer": answer}

    workflow = StateGraph(RAGState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("answer", answer_node)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "answer")
    workflow.add_edge("answer", END)
    return workflow.compile()
