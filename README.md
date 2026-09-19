# RAG Streamlit App

A simple Retrieval-Augmented Generation (RAG) app: upload a document, ask questions about it, get answers grounded in its content — with a fallback to general knowledge when the document doesn't have the answer.

## EXAMPLES
<img width="2549" height="1354" alt="image" src="https://github.com/user-attachments/assets/05fee577-f73b-4fae-92c6-e6474b940423" />

CHANGED THE DOCUMENT, CLEAR COLLECTION REMOVED THE OLD EMBEDDINGS 
<img width="2549" height="1354" alt="image" src="https://github.com/user-attachments/assets/a11e137d-d65b-4ae8-8626-9c176a5cd164" />
<img width="2549" height="1354" alt="image" src="https://github.com/user-attachments/assets/ca0efdac-007e-4f6f-b784-ac9d44291c3c" />



## How it works

1. Upload a `.txt`, `.pdf`, or `.docx` file
2. The document is chunked, embedded, and stored in a local vector database (ChromaDB)
3. Ask a question — the app retrieves the most relevant chunks and asks a local LLM (via Ollama) to answer using only that context
4. If the retrieved context doesn't answer the question, the app says so and falls back to the model's general knowledge, clearly labeled

**Note:** uploading a new document replaces the previous one's data — this is a single-document app, not a multi-document knowledge base.

## Stack

- **UI:** Streamlit (chat interface)
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector DB:** ChromaDB (local, persisted to `vectordb/`)
- **LLM:** Ollama running `llama3.1:8b` locally
- **Package management:** [uv](https://docs.astral.sh/uv/)

## Setup

### 1. Install Ollama

**Linux / macOS:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download and run the installer from [ollama.com/download](https://ollama.com/download).

### 2. Pull the model

```bash
ollama pull llama3.1:8b
```

Confirm it's there:
```bash
ollama list
```

### 3. Install Python dependencies

This project uses [uv](https://docs.astral.sh/uv/):
```bash
uv sync
```

### 4. Run the app

```bash
uv run streamlit run streamlit_app.py
```

This opens the app in your browser at `http://localhost:8501`.

## Project structure

```
rag_app/
├── core/
│   ├── ingest.py      # file loading, chunking, embedding, vector DB upsert
│   ├── retrieve.py    # query embedding + similarity search
│   ├── generate.py    # LLM prompt construction + Ollama call
│   └── config.py      # chunk size, model names, TOP_K, paths
├── streamlit_app.py   # UI
├── pyproject.toml
└── uv.lock
```

## Configuration

Tunable values live in `core/config.py`:

| Setting | Default | Notes |
|---|---|---|
| `CHUNK_SIZE` | 700 | characters per chunk |
| `CHUNK_OVERLAP` | 15% of chunk size | overlap between adjacent chunks |
| `TOP_K` | 8 | number of chunks retrieved per query |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `OLLAMA_MODEL` | `llama3.1:8b` | must be pulled locally first |

## Known limitations

- Single document at a time — new uploads clear previous data
- Local-only — no auth, no deployment config
- 8B model can be imprecise at following exact-format instructions (e.g. the `NO_CONTEXT` signal), so fallback behavior isn't 100% reliable



