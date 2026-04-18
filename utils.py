from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
DB_DIR = Path("db")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_COLLECTION_NAME = "university_rag"

_SEMANTIC_MODEL: SentenceTransformer | None = None


def ensure_directories() -> None:
    """Create the required project directories if they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)


def get_course_from_source(source: str | Path) -> str | None:
    """
    Infer an optional course/folder value from a file path relative to ./data.

    Examples:
    - data/CSE434/syllabus.pdf -> CSE434
    - data/CSE434/week1/policy.pdf -> CSE434/week1
    - data/syllabus.pdf -> None
    """
    source_path = Path(source)

    try:
        relative_parent = source_path.resolve().parent.relative_to(DATA_DIR.resolve())
    except ValueError:
        try:
            relative_parent = source_path.parent.relative_to(DATA_DIR)
        except ValueError:
            return None

    course = relative_parent.as_posix().strip(".")
    return course or None


def normalize_document_metadata(document: Document) -> Document:
    """Ensure each document carries consistent metadata used by retrieval and citations."""
    source = document.metadata.get("source", "")
    source_path = Path(source)

    document.metadata["source"] = str(source_path)
    document.metadata["document_name"] = source_path.name or "Unknown Document"
    document.metadata["course"] = document.metadata.get("course") or get_course_from_source(source_path)

    return document


def split_into_sentences(text: str) -> list[str]:
    """A lightweight sentence tokenizer that avoids external downloads."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", cleaned)
    if len(parts) == 1:
        parts = re.split(r"(?<=[.!?])\s+", cleaned)

    sentences = [part.strip() for part in parts if part.strip()]
    return sentences


def _get_semantic_model() -> SentenceTransformer:
    global _SEMANTIC_MODEL

    if _SEMANTIC_MODEL is None:
        _SEMANTIC_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return _SEMANTIC_MODEL


def _clone_metadata(document: Document) -> dict:
    return dict(document.metadata)


def _finalize_chunk(
    document: Document,
    chunk_text: str,
    chunk_index: int,
    method: str,
) -> Document | None:
    content = chunk_text.strip()
    if not content:
        return None

    metadata = _clone_metadata(document)
    metadata["chunk_index"] = chunk_index
    metadata["chunk_method"] = method
    return Document(page_content=content, metadata=metadata)


def _sentence_chunks(documents: Iterable[Document], target_words: int = 180, overlap_sentences: int = 1) -> list[Document]:
    chunks: list[Document] = []

    for document in documents:
        sentences = split_into_sentences(document.page_content)
        if not sentences:
            finalized = _finalize_chunk(document, document.page_content, 0, "sentence")
            if finalized:
                chunks.append(finalized)
            continue

        current_sentences: list[str] = []
        current_words = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())
            if current_sentences and current_words + sentence_words > target_words:
                finalized = _finalize_chunk(
                    document,
                    " ".join(current_sentences),
                    chunk_index,
                    "sentence",
                )
                if finalized:
                    chunks.append(finalized)
                    chunk_index += 1

                current_sentences = current_sentences[-overlap_sentences:] if overlap_sentences else []
                current_words = sum(len(item.split()) for item in current_sentences)

            current_sentences.append(sentence)
            current_words += sentence_words

        finalized = _finalize_chunk(document, " ".join(current_sentences), chunk_index, "sentence")
        if finalized:
            chunks.append(finalized)

    return chunks


def _semantic_chunks(
    documents: Iterable[Document],
    similarity_threshold: float = 0.58,
    max_words: int = 220,
) -> list[Document]:
    """
    Group semantically similar adjacent sentences using sentence embeddings.

    This keeps local topical cohesion without requiring an external semantic splitter.
    """
    chunks: list[Document] = []
    model = _get_semantic_model()

    for document in documents:
        sentences = split_into_sentences(document.page_content)
        if not sentences:
            finalized = _finalize_chunk(document, document.page_content, 0, "semantic")
            if finalized:
                chunks.append(finalized)
            continue

        if len(sentences) == 1:
            finalized = _finalize_chunk(document, sentences[0], 0, "semantic")
            if finalized:
                chunks.append(finalized)
            continue

        embeddings = model.encode(sentences, normalize_embeddings=True)

        current_sentences = [sentences[0]]
        current_embedding = embeddings[0]
        current_words = len(sentences[0].split())
        chunk_index = 0

        for idx in range(1, len(sentences)):
            sentence = sentences[idx]
            sentence_embedding = embeddings[idx]
            similarity = float(np.dot(current_embedding, sentence_embedding))
            sentence_words = len(sentence.split())

            should_split = similarity < similarity_threshold or current_words + sentence_words > max_words

            if should_split:
                finalized = _finalize_chunk(
                    document,
                    " ".join(current_sentences),
                    chunk_index,
                    "semantic",
                )
                if finalized:
                    chunks.append(finalized)
                    chunk_index += 1

                current_sentences = [sentence]
                current_embedding = sentence_embedding
                current_words = sentence_words
                continue

            current_sentences.append(sentence)
            current_words += sentence_words
            current_embedding = np.mean(embeddings[idx - len(current_sentences) + 1 : idx + 1], axis=0)
            norm = np.linalg.norm(current_embedding)
            if norm:
                current_embedding = current_embedding / norm

        finalized = _finalize_chunk(document, " ".join(current_sentences), chunk_index, "semantic")
        if finalized:
            chunks.append(finalized)

    return chunks


def chunk_documents(documents: list[Document], method: str = "word") -> list[Document]:
    """
    Chunk input documents using one of three strategies:
    - word: RecursiveCharacterTextSplitter
    - sentence: sentence-aware grouping
    - semantic: embedding-based sentence grouping
    """
    normalized_documents = [normalize_document_metadata(document) for document in documents]
    normalized_method = method.strip().lower()

    if normalized_method == "word":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(normalized_documents)
        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index
            chunk.metadata["chunk_method"] = "word"
        return chunks

    if normalized_method == "sentence":
        return _sentence_chunks(normalized_documents)

    if normalized_method == "semantic":
        return _semantic_chunks(normalized_documents)

    raise ValueError("Unsupported chunking method. Use 'word', 'sentence', or 'semantic'.")


def format_sources(documents: list[Document]) -> list[str]:
    """Return unique source document names in retrieval order."""
    unique_sources: list[str] = []
    seen: set[str] = set()

    for document in documents:
        document_name = document.metadata.get("document_name", "Unknown Document")
        if document_name not in seen:
            seen.add(document_name)
            unique_sources.append(document_name)

    return unique_sources


def format_context(documents: list[Document]) -> str:
    """Format retrieved documents into a compact prompt context."""
    sections: list[str] = []

    for document in documents:
        source = document.metadata.get("document_name", "Unknown Document")
        page = document.metadata.get("page")
        course = document.metadata.get("course")

        header_bits = [f"Source: {source}"]
        if page is not None:
            header_bits.append(f"Page: {page + 1}")
        if course:
            header_bits.append(f"Course: {course}")

        sections.append(f"{' | '.join(header_bits)}\n{document.page_content}")

    return "\n\n".join(sections)
