from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from uuid import uuid4

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
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
SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


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


def reset_vectorstore() -> None:
    """Safely remove the configured Chroma database directory before rebuilding."""
    configured_db_path = DB_DIR.resolve()
    project_path = Path.cwd().resolve()

    if configured_db_path == project_path or not configured_db_path.is_relative_to(project_path):
        raise ValueError(f"Refusing to reset unsafe DB path: {configured_db_path}")

    if not configured_db_path.exists():
        ensure_directories()
        return

    if not configured_db_path.is_dir():
        raise ValueError(f"Refusing to reset non-directory DB path: {configured_db_path}")

    shutil.rmtree(configured_db_path)
    ensure_directories()


def get_collection_count() -> int | None:
    """Return the current Chroma collection count when available."""
    try:
        vectorstore = get_vectorstore()
        return vectorstore._collection.count()
    except Exception:
        return None


def _get_document_paths() -> list[Path]:
    paths: list[Path] = []
    for extension in SUPPORTED_EXTENSIONS:
        paths.extend(DATA_DIR.rglob(f"*{extension}"))
    return sorted(paths)


def load_documents(file_paths: list[Path] | None = None) -> list[Document]:
    """Load supported documents from explicit paths or recursively from ./data."""
    ensure_directories()

    if file_paths is None:
        file_paths = _get_document_paths()

    documents: list[Document] = []
    for file_path in file_paths:
        if not file_path.exists() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
        else:
            loader = Docx2txtLoader(str(file_path))

        try:
            loaded_documents = loader.load()
        except Exception:
            continue

        for document in loaded_documents:
            if file_path.suffix.lower() == ".docx":
                document.metadata.setdefault("page", 0)
            documents.append(normalize_document_metadata(document))

    return documents


def load_pdf_documents(file_paths: list[Path] | None = None) -> list[Document]:
    """Load PDFs from explicit paths or recursively from ./data."""
    if file_paths is None:
        file_paths = sorted(DATA_DIR.rglob("*.pdf"))
    return load_documents(file_paths=file_paths)


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
    Load supported documents, chunk them, and persist chunks.

    Returns:
        (document_count, chunk_count)
    """
    documents = load_documents(file_paths=file_paths)
    chunk_count = ingest_documents(documents, method=method)
    return len(documents), chunk_count


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest university PDFs and DOCX files into a local Chroma database.")
    parser.add_argument(
        "--method",
        choices=["word", "sentence", "semantic"],
        default="word",
        help="Chunking strategy to use.",
    )
    parser.add_argument(
        "--path",
        nargs="*",
        help="Optional PDF or DOCX file paths to ingest instead of all supported files in ./data.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the configured Chroma DB directory before ingesting.",
    )
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    paths = [Path(path) for path in args.path] if args.path else None
    print(f"Reset mode: {'enabled' if args.reset else 'disabled'}")
    if args.reset:
        reset_vectorstore()
        print(f"Reset Chroma DB at {DB_DIR.resolve()}.")

    document_count, chunk_count = ingest_data(method=args.method, file_paths=paths)
    collection_count = get_collection_count()

    print(f"Loaded {document_count} document pages.")
    print(f"Persisted {chunk_count} chunks to {DB_DIR.resolve()}.")
    if collection_count is None:
        print("Final Chroma collection count: unavailable.")
    else:
        print(f"Final Chroma collection count: {collection_count}.")
