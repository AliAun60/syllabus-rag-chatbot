from __future__ import annotations

import re
from collections import Counter
from typing import Any

from langchain_core.documents import Document

from ingest import get_vectorstore

DEFAULT_DENSE_K = 20
DEFAULT_KEYWORD_K = 20
DEFAULT_FINAL_K = 4
MIN_RELEVANCE_SCORE = 0.2
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


def is_vectorstore_empty() -> bool:
    """Return True when the Chroma collection has no indexed chunks."""
    vectorstore = get_vectorstore()

    try:
        return vectorstore._collection.count() == 0
    except Exception:
        return True


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in STOP_WORDS
    ]


def _course_codes(text: str) -> set[str]:
    return {
        f"{prefix}{number}"
        for prefix, number in re.findall(r"\b([a-z]{2,4})[\s_-]*(\d{3})\b", text.lower())
    }


def _document_search_text(document: Document) -> str:
    metadata = document.metadata
    metadata_text = " ".join(
        str(metadata.get(key, ""))
        for key in ("document_name", "source", "course")
        if metadata.get(key)
    )
    return f"{metadata_text} {document.page_content}"


def _document_key(document: Document) -> tuple[str, str]:
    metadata = document.metadata
    source = str(metadata.get("source") or metadata.get("document_name") or "")
    return source, document.page_content


def _dense_retrieve(
    question: str,
    dense_k: int,
    course_filter: str | None,
    include_low_scores: bool,
) -> list[Document]:
    """Use Chroma embeddings for semantic candidate retrieval."""
    vectorstore = get_vectorstore()
    search_kwargs: dict[str, Any] = {"k": dense_k}
    if course_filter:
        search_kwargs["filter"] = {"course": course_filter}

    try:
        results = vectorstore.similarity_search_with_relevance_scores(question, **search_kwargs)
        if include_low_scores:
            return [document for document, _score in results]
        return [document for document, score in results if score >= MIN_RELEVANCE_SCORE]
    except Exception:
        pass

    try:
        return vectorstore.similarity_search(question, **search_kwargs)
    except Exception:
        return []


def _keyword_retrieve(question: str, keyword_k: int, course_filter: str | None) -> list[Document]:
    """
    Use lightweight keyword scoring over stored Chroma chunks.

    Hybrid retrieval helps when embeddings miss exact terms. Keyword retrieval is
    especially useful for course numbers, filenames, instructor names, and policy
    phrases such as "CSE 274" or "late assignment".
    """
    query_tokens = _tokenize(question)
    if not query_tokens:
        return []

    vectorstore = get_vectorstore()
    try:
        records = vectorstore.get(include=["documents", "metadatas"])
    except Exception:
        return []

    query_counts = Counter(query_tokens)
    query_terms = set(query_counts)
    query_phrase = " ".join(query_tokens)
    query_course_codes = _course_codes(question)
    scored_documents: list[tuple[float, int, Document]] = []

    raw_documents = records.get("documents", [])
    raw_metadatas = records.get("metadatas", [])

    for index, (content, metadata) in enumerate(zip(raw_documents, raw_metadatas)):
        metadata = metadata or {}
        if course_filter and metadata.get("course") != course_filter:
            continue

        document = Document(page_content=content or "", metadata=metadata)
        metadata_text = _document_search_text(Document(page_content="", metadata=metadata))
        searchable_text = _document_search_text(document)
        content_tokens = _tokenize(searchable_text)
        if not content_tokens:
            continue

        content_counts = Counter(content_tokens)
        overlap = sum(min(query_counts[token], content_counts[token]) for token in query_terms)
        if not overlap:
            continue

        term_frequency = sum(min(content_counts[token], 3) for token in query_terms)
        coverage = len(query_terms.intersection(content_counts)) / len(query_terms)
        density = overlap / len(content_tokens)
        phrase_bonus = 1.0 if query_phrase and query_phrase in " ".join(content_tokens) else 0
        metadata_course_bonus = 25.0 if query_course_codes.intersection(_course_codes(metadata_text)) else 0
        content_course_bonus = 4.0 if query_course_codes.intersection(_course_codes(document.page_content)) else 0

        score = term_frequency + (4 * coverage) + density + phrase_bonus + metadata_course_bonus + content_course_bonus
        scored_documents.append((score, -index, document))

    scored_documents.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [document for _score, _index, document in scored_documents[:keyword_k]]


def _merge_documents(document_groups: list[list[Document]]) -> list[Document]:
    merged: list[Document] = []
    seen: set[tuple[str, str]] = set()

    for documents in document_groups:
        for document in documents:
            key = _document_key(document)
            if key in seen:
                continue
            seen.add(key)
            merged.append(document)

    return merged


def _rerank_documents(question: str, documents: list[Document], final_k: int) -> list[Document]:
    """
    Rerank Chroma candidates with keyword overlap and phrase matching.

    Chroma is still the candidate generator. Reranking helps when many syllabi use
    similar wording by giving extra weight to chunks that share concrete query
    terms, repeated terms, and the exact query phrase.
    """
    query_tokens = _tokenize(question)
    if not query_tokens:
        return documents[:final_k]

    query_counts = Counter(query_tokens)
    query_terms = set(query_counts)
    query_phrase = " ".join(query_tokens)
    query_course_codes = _course_codes(question)
    scored_documents: list[tuple[float, int, Document]] = []

    for index, document in enumerate(documents):
        metadata = document.metadata
        metadata_text = _document_search_text(Document(page_content="", metadata=metadata))
        searchable_text = _document_search_text(document)
        content_tokens = _tokenize(searchable_text)
        content_counts = Counter(content_tokens)

        overlap = sum(min(query_counts[token], content_counts[token]) for token in query_terms)
        coverage = overlap / len(query_tokens)
        unique_coverage = len(query_terms.intersection(content_counts)) / len(query_terms)
        density = overlap / max(len(content_tokens), 1)
        phrase_bonus = 0.25 if query_phrase and query_phrase in " ".join(content_tokens) else 0
        metadata_course_bonus = 12.0 if query_course_codes.intersection(_course_codes(metadata_text)) else 0
        content_course_bonus = 1.0 if query_course_codes.intersection(_course_codes(document.page_content)) else 0

        score = coverage + unique_coverage + density + phrase_bonus + metadata_course_bonus + content_course_bonus
        scored_documents.append((score, -index, document))

    scored_documents.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [document for _score, _index, document in scored_documents[:final_k]]


def retrieve_documents(
    question: str,
    top_k: int = DEFAULT_FINAL_K,
    course_filter: str | None = None,
    use_reranking: bool = True,
    dense_k: int = DEFAULT_DENSE_K,
    keyword_k: int = DEFAULT_KEYWORD_K,
    initial_k: int | None = None,
) -> list[Document]:
    """
    Run hybrid retrieval, then optionally rerank candidates.

    dense_k controls how many Chroma embedding candidates are fetched.
    keyword_k controls how many keyword-scored candidates are fetched.
    top_k/final_k controls how many chunks are returned to the chatbot.
    """
    if not question.strip() or is_vectorstore_empty():
        return []

    final_k = top_k
    effective_dense_k = initial_k if initial_k is not None else dense_k

    dense_results = _dense_retrieve(
        question,
        dense_k=max(effective_dense_k, final_k),
        course_filter=course_filter,
        include_low_scores=use_reranking,
    )
    keyword_results = _keyword_retrieve(question, keyword_k=max(keyword_k, final_k), course_filter=course_filter)
    documents = _merge_documents([dense_results, keyword_results])

    if use_reranking:
        return _rerank_documents(question, documents, final_k=final_k)

    return documents[:final_k]


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
