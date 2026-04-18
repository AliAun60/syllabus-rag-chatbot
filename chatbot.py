from __future__ import annotations

import os

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


def _append_sources(answer: str, sources: list[str]) -> str:
    if not sources:
        return answer

    source_lines = "\n".join(f"- {source}" for source in sources)
    return f"{answer}\n\nSources:\n{source_lines}"


def answer_question(
    question: str,
    chat_history: list[dict] | None = None,
    course_filter: str | None = None,
    top_k: int = 3,
) -> dict:
    """
    Answer a question using only retrieved context.

    Returns a response payload suitable for the Streamlit chat UI.
    """
    history = chat_history or []
    documents = retrieve_documents(question, top_k=top_k, course_filter=course_filter)
    sources = format_sources(documents)

    if not documents:
        return {
            "answer": "I don't know",
            "sources": [],
            "documents": [],
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
                    "Conversation history:\n{history}\n\n"
                    "Retrieved context:\n{context}\n\n"
                    "Question: {question}"
                ),
            ),
        ]
    )

    chain = prompt | _get_llm() | StrOutputParser()
    raw_answer = chain.invoke(
        {
            "history": _format_history(history),
            "context": format_context(documents),
            "question": question,
        }
    ).strip()

    if raw_answer == "I don't know":
        return {
            "answer": "I don't know",
            "sources": sources,
            "documents": documents,
        }

    return {
        "answer": _append_sources(raw_answer, sources),
        "sources": sources,
        "documents": documents,
    }
