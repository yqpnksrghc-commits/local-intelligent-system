"""
Drop PDFs / text files into data/documents/ then run:
    python -m core.rag.ingest
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

load_dotenv()

DOCS_DIR    = Path(os.getenv("DOCS_DIR", "data/documents"))
VECTOR_PATH = os.getenv("VECTOR_DB_PATH", "./data/vector_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

LOADERS = {".pdf": PyPDFLoader, ".txt": TextLoader, ".md": TextLoader}
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)


def load_documents(docs_dir: Path):
    docs = []
    for path in docs_dir.rglob("*"):
        loader_cls = LOADERS.get(path.suffix.lower())
        if loader_cls:
            try:
                docs.extend(loader_cls(str(path)).load())
            except Exception as e:
                print(f"  skipped {path.name}: {e}")
    return docs


def ingest():
    print(f"Loading documents from {DOCS_DIR} ...")
    raw = load_documents(DOCS_DIR)
    if not raw:
        print("No documents found.")
        return
    chunks = SPLITTER.split_documents(raw)
    print(f"  {len(raw)} documents → {len(chunks)} chunks")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_HOST)
    db = Chroma(
        collection_name="personal_knowledge",
        embedding_function=embeddings,
        persist_directory=VECTOR_PATH,
    )
    db.add_documents(chunks)
    print(f"Ingested {len(chunks)} chunks into {VECTOR_PATH}")


if __name__ == "__main__":
    ingest()
