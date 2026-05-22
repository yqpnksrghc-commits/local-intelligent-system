"""
File-system watcher — monitors data/documents/ and auto-ingests
any new PDF/TXT/MD dropped there, without restarting the system.
"""
import threading
import time
from pathlib import Path


SUPPORTED = {".pdf", ".txt", ".md"}


def _ingest_path(path: Path) -> None:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain_community.embeddings import OllamaEmbeddings
    import os
    from dotenv import load_dotenv
    load_dotenv()

    loaders = {".pdf": PyPDFLoader, ".txt": TextLoader, ".md": TextLoader}
    loader_cls = loaders.get(path.suffix.lower())
    if not loader_cls:
        return

    try:
        docs = loader_cls(str(path)).load()
    except Exception as e:
        print(f"[watcher] failed to load {path.name}: {e}")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(
        model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    )
    db = Chroma(
        collection_name="personal_knowledge",
        embedding_function=embeddings,
        persist_directory=os.getenv("VECTOR_DB_PATH", "./data/vector_db"),
    )
    db.add_documents(chunks)
    print(f"[watcher] ingested {len(chunks)} chunks from {path.name}")


def _watch_loop(docs_dir: Path, poll_interval: int = 10) -> None:
    seen: set[Path] = set()
    # seed with already-present files so we don't re-ingest on startup
    for p in docs_dir.rglob("*"):
        if p.suffix.lower() in SUPPORTED:
            seen.add(p)

    while True:
        time.sleep(poll_interval)
        for p in docs_dir.rglob("*"):
            if p.suffix.lower() in SUPPORTED and p not in seen:
                seen.add(p)
                print(f"[watcher] new file detected: {p.name}")
                _ingest_path(p)


def start(docs_dir: Path, poll_interval: int = 10) -> threading.Thread:
    t = threading.Thread(target=_watch_loop, args=(docs_dir, poll_interval), daemon=True)
    t.start()
    print(f"[watcher] watching {docs_dir} every {poll_interval}s")
    return t
