"""
FastAPI service — unified REST interface over all system capabilities.

Endpoints:
  POST /chat                — main assistant (RAG + memory)
  POST /translate           — universal translator
  POST /transduce           — multiversal: any modality → any modality
  POST /action              — natural language → real-world action
  GET  /status              — system health
  GET  /memory              — recent long-term memories
  POST /inbound/call        — Twilio inbound call webhook
  POST /inbound/sms         — Twilio inbound SMS webhook
  WS   /stream              — streaming chat (WebSocket)
"""
from __future__ import annotations
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

load_dotenv()

# ── State ──────────────────────────────────────────────────────────────────────

_llm         = None
_vectorstore = None
_orchestrator = None
_bridge      = None


def _get_llm():
    global _llm
    if _llm is None:
        from langchain_ollama import ChatOllama
        _llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
            temperature=0.3,
            base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        )
    return _llm


def _get_bridge():
    global _bridge
    if _bridge is None:
        from core.multiversal.nbridge import NBridge
        _bridge = NBridge(
            llm=_get_llm(),
            embed_model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
            ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        )
    return _bridge


def _get_orchestrator():
    global _orchestrator, _vectorstore
    if _orchestrator is None:
        from langchain_chroma import Chroma
        from langchain_community.embeddings import OllamaEmbeddings
        from core.orchestrator import create_orchestrator
        emb = OllamaEmbeddings(
            model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
            base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        )
        _vectorstore = Chroma(
            collection_name="personal_knowledge",
            embedding_function=emb,
            persist_directory=os.getenv("VECTOR_DB_PATH", "./data/vector_db"),
        )
        _orchestrator = create_orchestrator(_get_llm(), _vectorstore)
    return _orchestrator


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    from pathlib import Path
    from core.self.health import wait_for_ollama
    from core.self import watcher, summarizer

    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    wait_for_ollama(host)

    llm = _get_llm()
    _get_orchestrator()
    _get_bridge()

    docs_dir = Path(os.getenv("DOCS_DIR", "data/documents"))
    docs_dir.mkdir(parents=True, exist_ok=True)
    watcher.start(docs_dir)
    summarizer.start(llm)

    print("[api] all systems ready")
    yield


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Local Intelligent System", lifespan=lifespan)


# ── Request / response models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str

class TranslateRequest(BaseModel):
    text: str
    target_language: str
    source_language: str | None = None

class TransduceRequest(BaseModel):
    source: str
    targets: list[str]
    source_modality: str = "text"
    source_language: str | None = None

class ActionRequest(BaseModel):
    request: str
    confirm: bool = True


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/status")
def status():
    from core.self.health import is_alive
    return {
        "ollama": is_alive(os.getenv("OLLAMA_HOST", "http://localhost:11434")),
        "model":  os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
    }


@app.get("/memory")
def memory(n: int = 20):
    from core.memory.store import recent
    return {"memories": recent(n)}


@app.post("/chat")
def chat(req: ChatRequest):
    orch   = _get_orchestrator()
    result = orch.invoke({"messages": [("user", req.message)]})
    reply  = result["messages"][-1].content
    return {"reply": reply}


@app.post("/translate")
def translate(req: TranslateRequest):
    from core.translator.translator import translate as _translate
    result = _translate(req.text, req.target_language, req.source_language, _get_llm())
    return {"translation": result, "target": req.target_language}


@app.post("/transduce")
def transduce(req: TransduceRequest):
    bridge  = _get_bridge()
    results = bridge.transduce(
        req.source, req.targets,
        source_modality=req.source_modality,
        source_language=req.source_language,
    )
    return results


@app.post("/action")
def action(req: ActionRequest):
    from core.actions.agent import run
    result = run(req.request, _get_llm(), confirm_first=req.confirm)
    return {"result": result}


# ── Twilio webhooks ────────────────────────────────────────────────────────────

@app.post("/inbound/call")
async def inbound_call(request):
    from fastapi import Request
    form  = await request.form()
    frm   = form.get("From", "?")
    body  = form.get("SpeechResult", form.get("Body", ""))
    print(f"[inbound call] from {frm}: {body!r}")
    orch  = _get_orchestrator()
    if body:
        result = orch.invoke({"messages": [("user", body)]})
        reply  = result["messages"][-1].content
    else:
        reply = "Hello, how can I help you?"
    twiml = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<Response>'
        f'<Gather input="speech" action="/inbound/call" timeout="3">'
        f'<Say>{reply}</Say>'
        f'</Gather>'
        f'</Response>'
    )
    from fastapi.responses import Response
    return Response(content=twiml, media_type="text/xml")


@app.post("/inbound/sms")
async def inbound_sms(request):
    form  = await request.form()
    frm   = form.get("From", "?")
    body  = form.get("Body", "")
    print(f"[inbound SMS] from {frm}: {body!r}")
    orch  = _get_orchestrator()
    result = orch.invoke({"messages": [("user", body)]})
    reply  = result["messages"][-1].content[:1600]   # SMS limit
    twiml = f'<?xml version="1.0"?><Response><Message>{reply}</Message></Response>'
    from fastapi.responses import Response
    return Response(content=twiml, media_type="text/xml")


# ── WebSocket streaming chat ───────────────────────────────────────────────────

@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    orch = _get_orchestrator()
    try:
        while True:
            message = await ws.receive_text()
            result  = orch.invoke({"messages": [("user", message)]})
            reply   = result["messages"][-1].content
            await ws.send_text(reply)
    except WebSocketDisconnect:
        pass
