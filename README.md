# University Syllabus and Policy RAG Chatbot

This project is a local Streamlit chatbot for querying university syllabi and policy documents with retrieval-augmented generation (RAG).

## Features

- Loads PDFs and DOCX files from `./data`
- Chunks documents with word-based, sentence-based, or semantic chunking
- Stores embeddings in a persistent local Chroma database in `./db`
- Uses `all-MiniLM-L6-v2` sentence-transformer embeddings
- Uses `gpt-4o-mini` for answer generation
- Supports PDF and DOCX uploads directly from the Streamlit UI
- Supports optional course/folder filtering through metadata
- Combines dense Chroma retrieval with lightweight keyword retrieval before reranking
- Includes a retrieval evaluation script with Precision@K and Recall@K
- Keeps a lightweight per-chat conversation summary in Streamlit session state
- Supports multiple chat sessions in Streamlit
- Shows source document citations for answers

## Setup

Create and activate a Python 3.10+ virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set your OpenAI API key:

```bash
cp .env.example .env
```

Required environment variable:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

Optional environment variable:

```bash
ANONYMIZED_TELEMETRY=False
```

Place PDF and DOCX files inside `./data`.

## Ingestion

Normal ingestion appends chunks to the existing Chroma database:

```bash
python3 ingest.py --method word
```

To rebuild the vector database cleanly and avoid duplicate chunks, use reset mode:

```bash
python3 ingest.py --method word --reset
```

The reset command only clears the configured local `./db` directory before rebuilding.

## Run Streamlit

Start the app:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Retrieval Evaluation

Run the lightweight retrieval evaluation script:

```bash
python3 eval_retrieval.py
```

The evaluation cases live in `eval/retrieval_eval_cases.json`. The script compares retrieval with reranking enabled and disabled.

- Precision@K means: out of the top K retrieved chunks, how many came from an expected source?
- Recall@K means: out of the expected sources for that question, how many appeared in the top K retrieved chunks?

The default is `K=4`. You can change it with:

```bash
python3 eval_retrieval.py --k 5
```

## Deployment Notes

A simple Dockerfile is included for container-based deployment.

Build the image:

```bash
docker build -t university-rag-chatbot .
```

Run the container:

```bash
docker run --env-file .env -p 8501:8501 university-rag-chatbot
```

For a clean production/demo database, run ingestion before deployment or mount a persistent `db` volume and run:

```bash
python3 ingest.py --method word --reset
```

If deploying to a hosted service, configure `OPENAI_API_KEY` as a secret/environment variable and make sure the `data` and `db` directories are included or mounted according to that platform's storage model.

## Notes

- Uploaded PDFs and DOCX files are saved into `./data` and immediately added to the current vector database.
- Use `python3 ingest.py --method word --reset` before demos or submission when you want a clean Chroma rebuild without duplicate chunks.
- Hybrid retrieval combines dense embedding search with keyword scoring over stored chunks. This improves recall for exact matches such as course numbers, filenames, instructor names, and policy phrases.
- Reranking reorders the merged candidates by query/document keyword overlap. This helps when multiple syllabi use similar wording.
- Conversation memory is session-only. It is not saved after the Streamlit session ends.
- If the retriever cannot find relevant context, the chatbot returns `I don't know`.
- The vector store persists locally in `./db`.
