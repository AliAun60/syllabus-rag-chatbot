# University Syllabus and Policy RAG Chatbot

This project is a local Streamlit chatbot for querying university syllabi and policy PDFs with retrieval-augmented generation (RAG).

## Features

- Loads PDFs from `./data`
- Chunks documents with word-based, sentence-based, or semantic chunking
- Stores embeddings in a persistent local Chroma database in `./db`
- Uses `all-MiniLM-L6-v2` sentence-transformer embeddings
- Uses `gpt-4o-mini` for answer generation
- Supports PDF uploads directly from the Streamlit UI
- Supports optional course/folder filtering through metadata
- Supports multiple chat sessions in Streamlit
- Shows source document citations for answers

## Setup

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and set your OpenAI API key:

```bash
OPENAI_API_KEY=your_api_key_here
```

4. Place your PDF files inside `./data`.

## Run

Optionally pre-ingest all PDFs from `./data`:

```bash
python ingest.py --method word
```

Start the app:

```bash
streamlit run app.py
```

## Notes

- Uploaded PDFs are saved into `./data` and immediately added to the current vector database.
- If the retriever cannot find relevant context, the chatbot returns `I don't know`.
- The vector store persists locally in `./db`.
