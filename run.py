import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from core.orchestrator import create_orchestrator

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL       = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
VECTOR_PATH = os.getenv("VECTOR_DB_PATH", "./data/vector_db")

llm = ChatOllama(model=MODEL, temperature=0.3, base_url=OLLAMA_HOST)
embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_HOST)

vectorstore = Chroma(
    collection_name="personal_knowledge",
    embedding_function=embeddings,
    persist_directory=VECTOR_PATH,
)

orchestrator = create_orchestrator(llm, vectorstore)

print(f"Local Intelligent System ready  (model: {MODEL})")
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
    result = orchestrator.invoke({"messages": [("user", query)]})
    print(f"Assistant: {result['messages'][-1].content}\n")
