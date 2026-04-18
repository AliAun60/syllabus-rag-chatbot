from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from ingest import get_vectorstore


def is_vectorstore_empty() -> bool:
    """Return True when the Chroma collection has no indexed chunks."""
    vectorstore = get_vectorstore()

    try:
        return vectorstore._collection.count() == 0
    except Exception:
        return True


def retrieve_documents(question: str, top_k: int = 3, course_filter: str | None = None) -> list[Document]:
    """Run semantic similarity search with an optional metadata filter."""
    if not question.strip() or is_vectorstore_empty():
        return []

    vectorstore = get_vectorstore()
    search_kwargs: dict[str, Any] = {"k": top_k}
    if course_filter:
        search_kwargs["filter"] = {"course": course_filter}

    try:
        results = vectorstore.similarity_search_with_relevance_scores(question, **search_kwargs)
        filtered_results = [document for document, score in results if score >= 0.2]
        if filtered_results:
            return filtered_results
    except Exception:
        pass

    try:
        return vectorstore.similarity_search(question, **search_kwargs)
    except Exception:
        return []


def get_available_courses() -> list[str]:
    """List unique course/folder metadata currently stored in the vector DB."""
    if is_vectorstore_empty():
        return []

    vectorstore = get_vectorstore()
    try:
        records = vectorstore.get(include=["metadatas"])
    except Exception:
        return []

    courses = {
        metadata.get("course")
        for metadata in records.get("metadatas", [])
        if metadata and metadata.get("course")
    }
    return sorted(courses)
