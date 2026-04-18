from __future__ import annotations

import argparse
from pathlib import Path
from uuid import uuid4

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from utils import (
    DATA_DIR,
    DB_DIR,
    DEFAULT_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    chunk_documents,
    ensure_directories,
    normalize_document_metadata,
)

VALID_METADATA_TYPES = (str, int, float, bool)


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore() -> Chroma:
    ensure_directories()
    return Chroma(
        collection_name=DEFAULT_COLLECTION_NAME,
        persist_directory=str(DB_DIR),
        embedding_function=get_embeddings(),
    )


def load_pdf_documents(file_paths: list[Path] | None = None) -> list[Document]:
    """Load PDFs from explicit paths or recursively from ./data."""
    ensure_directories()

    if file_paths is None:
        file_paths = sorted(DATA_DIR.rglob("*.pdf"))

    documents: list[Document] = []
    for file_path in file_paths:
        if not file_path.exists() or file_path.suffix.lower() != ".pdf":
            continue

        loader = PyPDFLoader(str(file_path))
        for page_document in loader.load():
            documents.append(normalize_document_metadata(page_document))

    return documents


def sanitize_document_metadata(document: Document) -> Document:
    """
    Remove unsupported metadata values before upsert into Chroma.

    Chroma accepts only scalar metadata values: str, int, float, or bool.
    """
    document.metadata = {
        key: value
        for key, value in document.metadata.items()
        if value is not None and isinstance(value, VALID_METADATA_TYPES)
    }
    return document


def ingest_documents(documents: list[Document], method: str = "word") -> int:
    """Chunk documents, embed them, and persist them to Chroma."""
    if not documents:
        return 0

    vectorstore = get_vectorstore()
    chunks = chunk_documents(documents, method=method)
    if not chunks:
        return 0

    chunks = [sanitize_document_metadata(chunk) for chunk in chunks]

    ids = [str(uuid4()) for _ in chunks]
    vectorstore.add_documents(documents=chunks, ids=ids)
    return len(chunks)


def ingest_data(method: str = "word", file_paths: list[Path] | None = None) -> tuple[int, int]:
    """
    Load PDFs, chunk them, and persist chunks.

    Returns:
        (document_count, chunk_count)
    """
    documents = load_pdf_documents(file_paths=file_paths)
    chunk_count = ingest_documents(documents, method=method)
    return len(documents), chunk_count


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest university PDFs into a local Chroma database.")
    parser.add_argument(
        "--method",
        choices=["word", "sentence", "semantic"],
        default="word",
        help="Chunking strategy to use.",
    )
    parser.add_argument(
        "--path",
        nargs="*",
        help="Optional PDF file paths to ingest instead of all PDFs in ./data.",
    )
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    paths = [Path(path) for path in args.path] if args.path else None
    document_count, chunk_count = ingest_data(method=args.method, file_paths=paths)

    print(f"Loaded {document_count} document pages.")
    print(f"Persisted {chunk_count} chunks to {DB_DIR.resolve()}.")
