import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from core.orchestrator import create_orchestrator
from core.self import health, watcher, summarizer

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL       = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
VECTOR_PATH = os.getenv("VECTOR_DB_PATH", "./data/vector_db")
DOCS_DIR    = Path(os.getenv("DOCS_DIR", "data/documents"))
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# block until Ollama is reachable — no crash on cold start
health.wait_for_ollama(OLLAMA_HOST)

llm = ChatOllama(model=MODEL, temperature=0.3, base_url=OLLAMA_HOST)
embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_HOST)

vectorstore = Chroma(
    collection_name="personal_knowledge",
    embedding_function=embeddings,
    persist_directory=VECTOR_PATH,
)

orchestrator = create_orchestrator(llm, vectorstore)

# self-sufficiency daemons
watcher.start(DOCS_DIR, poll_interval=10)
summarizer.start(llm, every_n=10)

print(f"\nLocal Intelligent System ready  (model: {MODEL})")
print("Type 'exit' to quit.\n")

while True:
    try:
        query = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not query:
        continue
    if query.lower() in ("exit", "quit"):
        break

    # auto-recover if Ollama went away mid-session
    if not health.is_alive(OLLAMA_HOST):
        print("[health] Ollama is down — waiting for recovery …")
        health.wait_for_ollama(OLLAMA_HOST)

    result = orchestrator.invoke({"messages": [("user", query)]})
    print(f"Assistant: {result['messages'][-1].content}\n")
