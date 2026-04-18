from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

from chatbot import answer_question
from ingest import ingest_data
from retriever import get_available_courses, is_vectorstore_empty
from utils import DATA_DIR, ensure_directories

load_dotenv()
ensure_directories()

st.set_page_config(page_title="University RAG Chatbot", page_icon="📚", layout="wide")

CHUNKING_OPTIONS = {
    "Word-Based": "word",
    "Sentence-Based": "sentence",
    "Semantic": "semantic",
}


def create_chat_session(name: str | None = None) -> str:
    session_id = str(uuid4())
    session_name = name or f"Chat {len(st.session_state.chat_sessions) + 1}"
    st.session_state.chat_sessions[session_id] = {
        "name": session_name,
        "messages": [],
        "created_at": datetime.now().isoformat(),
    }
    return session_id


def initialize_state() -> None:
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}

    if "active_chat_id" not in st.session_state or st.session_state.active_chat_id not in st.session_state.chat_sessions:
        st.session_state.active_chat_id = create_chat_session()

    if "last_upload_signature" not in st.session_state:
        st.session_state.last_upload_signature = None


def save_uploaded_file(uploaded_file) -> Path:
    target_path = DATA_DIR / uploaded_file.name
    target_path.write_bytes(uploaded_file.getbuffer())
    return target_path


def process_uploads(uploaded_files, chunk_method: str) -> tuple[int, int, list[str]]:
    valid_paths: list[Path] = []
    errors: list[str] = []

    for uploaded_file in uploaded_files:
        if not uploaded_file.name.lower().endswith(".pdf"):
            errors.append(f"{uploaded_file.name}: invalid file type")
            continue

        try:
            valid_paths.append(save_uploaded_file(uploaded_file))
        except Exception as exc:
            errors.append(f"{uploaded_file.name}: {exc}")

    if not valid_paths:
        return 0, 0, errors

    try:
        document_count, chunk_count = ingest_data(method=chunk_method, file_paths=valid_paths)
    except Exception as exc:
        errors.append(str(exc))
        return 0, 0, errors

    return document_count, chunk_count, errors


def render_sidebar() -> tuple[str | None, str]:
    with st.sidebar:
        st.title("University RAG")

        if st.button("New Chat", use_container_width=True):
            st.session_state.active_chat_id = create_chat_session()

        session_options = list(st.session_state.chat_sessions.keys())
        labels = {
            session_id: st.session_state.chat_sessions[session_id]["name"]
            for session_id in session_options
        }

        active_chat_id = st.selectbox(
            "Chats",
            options=session_options,
            format_func=lambda session_id: labels[session_id],
            index=session_options.index(st.session_state.active_chat_id),
        )
        st.session_state.active_chat_id = active_chat_id

        selected_chunk_label = st.selectbox(
            "Chunking Strategy",
            options=list(CHUNKING_OPTIONS.keys()),
            index=0,
        )
        selected_chunk_method = CHUNKING_OPTIONS[selected_chunk_label]

        available_courses = get_available_courses()
        course_options = ["All Courses", *available_courses]
        selected_course = st.selectbox("Course Filter", options=course_options, index=0)
        course_filter = None if selected_course == "All Courses" else selected_course

        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="Uploaded PDFs are saved into ./data and added to the existing vector DB immediately.",
        )

        if uploaded_files:
            upload_signature = tuple((file.name, file.size) for file in uploaded_files)
            if upload_signature != st.session_state.last_upload_signature:
                with st.spinner("Processing uploaded PDFs..."):
                    document_count, chunk_count, errors = process_uploads(uploaded_files, selected_chunk_method)
                st.session_state.last_upload_signature = upload_signature

                if chunk_count:
                    st.success(f"Ingested {document_count} pages into {chunk_count} chunks.")
                elif not errors:
                    st.warning("No new content was ingested from the uploaded files.")

                for error in errors:
                    st.error(error)

        if st.button("Ingest All PDFs from ./data", use_container_width=True):
            with st.spinner("Ingesting PDFs from ./data..."):
                try:
                    document_count, chunk_count = ingest_data(method=selected_chunk_method)
                    if chunk_count:
                        st.success(f"Ingested {document_count} pages into {chunk_count} chunks.")
                    else:
                        st.warning("No PDFs were found in ./data or no chunks were created.")
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")

    return course_filter, selected_chunk_method


def render_chat() -> None:
    active_session = st.session_state.chat_sessions[st.session_state.active_chat_id]
    messages = active_session["messages"]

    st.title("University Syllabus and Policy Chatbot")
    st.caption("Ask questions about course syllabi, policies, deadlines, grading, and university documents.")

    if is_vectorstore_empty():
        st.info("The vector database is empty. Add PDFs to ./data or upload them from the sidebar, then ingest them.")

    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources") and message["content"] == "I don't know":
                st.markdown("**Sources**")
                for source in message["sources"]:
                    st.markdown(f"- {source}")


def main() -> None:
    initialize_state()
    course_filter, _selected_chunk_method = render_sidebar()
    render_chat()

    prompt = st.chat_input("Ask a question about a syllabus or policy document")
    if not prompt:
        return

    active_session = st.session_state.chat_sessions[st.session_state.active_chat_id]
    active_session["messages"].append({"role": "user", "content": prompt})

    if active_session["name"].startswith("Chat ") and len(active_session["messages"]) == 1:
        active_session["name"] = prompt[:40] + ("..." if len(prompt) > 40 else "")

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = answer_question(
                    question=prompt,
                    chat_history=active_session["messages"][:-1],
                    course_filter=course_filter,
                    top_k=3,
                )
            except Exception as exc:
                error_message = f"Error: {exc}"
                st.error(error_message)
                active_session["messages"].append({"role": "assistant", "content": error_message, "sources": []})
                return

        if response["answer"] == "I don't know" and response["sources"]:
            st.markdown("I don't know")
            st.markdown("**Sources**")
            for source in response["sources"]:
                st.markdown(f"- {source}")
            message_content = "I don't know"
        else:
            st.markdown(response["answer"])
            message_content = response["answer"]

    active_session["messages"].append(
        {
            "role": "assistant",
            "content": message_content,
            "sources": response.get("sources", []),
        }
    )


if __name__ == "__main__":
    main()
