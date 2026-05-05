from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from retriever import retrieve_documents
from utils import format_context, format_sources

load_dotenv()


def _get_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Add it to your environment or .env file.")

    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key,
    )


def _format_history(chat_history: list[dict]) -> str:
    lines: list[str] = []
    for message in chat_history[-8:]:
        role = message.get("role", "user").capitalize()
        content = message.get("content", "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _format_memory(conversation_summary: str | None, chat_history: list[dict]) -> str:
    history = _format_history(chat_history)
    summary = (conversation_summary or "").strip()

    if summary and history:
        return f"Summary of earlier conversation:\n{summary}\n\nRecent messages:\n{history}"
    if summary:
        return f"Summary of earlier conversation:\n{summary}"
    return history


def _append_sources(answer: str, sources: list[str]) -> str:
    if not sources:
        return answer

    source_lines = "\n".join(f"- {source}" for source in sources)
    return f"{answer}\n\nSources:\n{source_lines}"


def _is_insufficient_quota_error(exc: Exception) -> bool:
    """Detect OpenAI quota failures without depending on a specific SDK exception class."""
    message = str(exc).lower()
    status_code = getattr(exc, "status_code", None)

    if status_code == 429 and "insufficient_quota" in message:
        return True

    return "insufficient_quota" in message or (
        "429" in message and "quota" in message and "openai" in message
    )


def _build_retrieved_context(documents: list[Any]) -> list[dict[str, Any]]:
    context_items: list[dict[str, Any]] = []

    for document in documents:
        metadata = document.metadata
        context_items.append(
            {
                "content": document.page_content,
                "source": metadata.get("document_name", "Unknown Document"),
                "page": metadata.get("page"),
                "course": metadata.get("course"),
            }
        )

    return context_items


def answer_question(
    question: str,
    chat_history: list[dict] | None = None,
    course_filter: str | None = None,
    top_k: int = 4,
    retrieval_only: bool = False,
    use_reranking: bool = True,
    conversation_summary: str | None = None,
) -> dict:
    """
    Answer a question using only retrieved context.

    Returns a response payload suitable for the Streamlit chat UI.
    """
    history = chat_history or []
    documents = retrieve_documents(
        question,
        top_k=top_k,
        course_filter=course_filter,
        use_reranking=use_reranking,
    )
    sources = format_sources(documents)
    retrieved_context = _build_retrieved_context(documents)

    if not documents:
        return {
            "answer": "I don't know",
            "sources": [],
            "documents": [],
            "retrieved_context": [],
            "mode": "no_context",
        }

    if retrieval_only:
        return {
            "answer": (
                "Retrieval-only mode is enabled. Generation was skipped, "
                "but the top matching context is shown below."
            ),
            "sources": sources,
            "documents": documents,
            "retrieved_context": retrieved_context,
            "mode": "retrieval_only",
        }

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a university syllabus and policy assistant. "
                    "Answer the user using only the provided context. "
                    "Do not use outside knowledge. "
                    "If the answer is not fully supported by the context, reply exactly with: I don't know"
                ),
            ),
            (
                "human",
                (
                    "Conversation memory:\n{history}\n\n"
                    "Retrieved context:\n{context}\n\n"
                    "Question: {question}"
                ),
            ),
        ]
    )

    chain = prompt | _get_llm() | StrOutputParser()
    try:
        raw_answer = chain.invoke(
            {
                "history": _format_memory(conversation_summary, history),
                "context": format_context(documents),
                "question": question,
            }
        ).strip()
    except Exception as exc:
        if _is_insufficient_quota_error(exc):
            return {
                "answer": (
                    "Generation is temporarily unavailable because the OpenAI API quota "
                    "has been exceeded. The retrieval pipeline still worked, so the top "
                    "matching context is shown below."
                ),
                "sources": sources,
                "documents": documents,
                "retrieved_context": retrieved_context,
                "mode": "quota_fallback",
            }
        raise

    if raw_answer == "I don't know":
        return {
            "answer": "I don't know",
            "sources": sources,
            "documents": documents,
            "retrieved_context": [],
            "mode": "answer",
        }

    return {
        "answer": _append_sources(raw_answer, sources),
        "sources": sources,
        "documents": documents,
        "retrieved_context": [],
        "mode": "answer",
    }
