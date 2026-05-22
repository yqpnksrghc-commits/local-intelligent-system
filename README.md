# Local Intelligent System

Personal assistant with RAG, long-term memory, and multi-agent routing. Runs fully local via Ollama.

## Quick start

```bash
# 1. Copy env and configure
cp .env.example .env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Pull models (Ollama must be running)
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# 4. (Optional) Ingest your documents
#    Drop PDFs/text into data/documents/ then:
python -m core.rag.ingest

# 5. Run
python run.py
```

## Structure

```
core/
  orchestrator.py   LangGraph main graph (tool-calling loop)
  agents/           Planner, Researcher, Writer, Coder nodes
  rag/              ingest.py — PDF/text → Chroma
  memory/           store.py  — SQLite long-term memory
data/
  documents/        Drop source files here for RAG
  vector_db/        Chroma persistent store
  long_term/        SQLite memory database
```

## Environment variables (.env)

| Variable | Default | Description |
|---|---|---|
| OLLAMA_HOST | http://localhost:11434 | Ollama endpoint |
| OLLAMA_MODEL | llama3.2:3b | Chat model |
| EMBED_MODEL | nomic-embed-text | Embedding model |
| VECTOR_DB_PATH | ./data/vector_db | Chroma directory |
| LONG_TERM_DB | ./data/long_term/memory.db | SQLite path |

## Phase 2 (next)

- RAG retrieval wired into orchestrator before tool use
- Planner → Researcher → Writer/Coder supervisor graph
- Gradio frontend with file upload and chat history
