"""
Gradio frontend — unified interface over all system capabilities.

Tabs:
  Chat        — main assistant with RAG + long-term memory
  Translate   — universal translator (any language, voice output)
  Transduce   — multiversal: encode meaning, render to any modality
  Actions     — natural language → real-world actions (call, SMS, order)
  Status      — system health, memory, trajectory
"""
import os
import requests
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

API = os.getenv("API_URL", "http://localhost:8000")


def _post(endpoint: str, payload: dict) -> dict:
    try:
        r = requests.post(f"{API}{endpoint}", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ── Chat ───────────────────────────────────────────────────────────────────────

def chat(message: str, history: list) -> tuple[str, list]:
    result = _post("/chat", {"message": message})
    reply  = result.get("reply", result.get("error", "No response."))
    history.append((message, reply))
    return "", history


# ── Translate ──────────────────────────────────────────────────────────────────

def translate(text: str, target: str, source: str) -> str:
    result = _post("/translate", {
        "text": text,
        "target_language": target or "english",
        "source_language": source or None,
    })
    return result.get("translation", result.get("error", ""))


# ── Transduce ──────────────────────────────────────────────────────────────────

def transduce(source: str, modality: str, targets_raw: str) -> str:
    targets = [t.strip() for t in targets_raw.split(",") if t.strip()]
    result  = _post("/transduce", {
        "source":          source,
        "source_modality": modality or "text",
        "targets":         targets,
    })
    if "error" in result:
        return result["error"]
    lines = [f"Meaning (ℝ{result.get('dim','?')}):\n{result.get('meaning','')}"]
    skip  = {"source", "modality", "meaning", "dim"}
    for k, v in result.items():
        if k not in skip:
            lines.append(f"\n── {k.upper()} ──\n{v}")
    return "\n".join(lines)


# ── Actions ────────────────────────────────────────────────────────────────────

def action(request: str, auto_confirm: bool) -> str:
    result = _post("/action", {"request": request, "confirm": not auto_confirm})
    return result.get("result", result.get("error", ""))


# ── Status ─────────────────────────────────────────────────────────────────────

def status() -> str:
    try:
        r = requests.get(f"{API}/status", timeout=5).json()
        m = requests.get(f"{API}/memory?n=10", timeout=5).json()
    except Exception as e:
        return f"API unreachable: {e}"
    lines = [
        f"Ollama: {'✓ online' if r.get('ollama') else '✗ offline'}",
        f"Model:  {r.get('model', '?')}",
        "",
        "── Recent memory ──",
    ]
    for mem in m.get("memories", []):
        lines.append(f"  [{mem['role']}] {mem['content'][:80]}")
    return "\n".join(lines)


# ── Layout ─────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Local Intelligent System", theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# Local Intelligent System")

    with gr.Tab("Chat"):
        chatbot = gr.Chatbot(height=500)
        with gr.Row():
            msg = gr.Textbox(placeholder="Message…", scale=9, show_label=False)
            btn = gr.Button("Send", scale=1)
        btn.click(chat, [msg, chatbot], [msg, chatbot])
        msg.submit(chat, [msg, chatbot], [msg, chatbot])

    with gr.Tab("Translate"):
        with gr.Row():
            t_in  = gr.Textbox(label="Input", lines=4, scale=2)
            t_out = gr.Textbox(label="Translation", lines=4, scale=2, interactive=False)
        with gr.Row():
            t_src = gr.Textbox(label="Source language (blank = auto)", scale=1)
            t_tgt = gr.Textbox(label="Target language", value="english", scale=1)
            t_btn = gr.Button("Translate", scale=1)
        t_btn.click(translate, [t_in, t_tgt, t_src], t_out)

    with gr.Tab("Transduce"):
        with gr.Row():
            td_in  = gr.Textbox(label="Input", lines=3, scale=2)
            td_mod = gr.Textbox(label="Source modality", value="text", scale=1)
        td_tgt = gr.Textbox(
            label="Target modalities (comma-separated)",
            value="english, math, music, symbol",
        )
        td_btn = gr.Button("Transduce")
        td_out = gr.Textbox(label="Output", lines=12, interactive=False)
        td_btn.click(transduce, [td_in, td_mod, td_tgt], td_out)

    with gr.Tab("Actions"):
        ac_req  = gr.Textbox(
            label="What do you want the system to do?",
            placeholder="Find me an iPhone 15 Pro under $1100 on Amazon",
            lines=2,
        )
        ac_auto = gr.Checkbox(label="Auto-confirm (skip gate)", value=False)
        ac_btn  = gr.Button("Execute")
        ac_out  = gr.Textbox(label="Result", lines=6, interactive=False)
        ac_btn.click(action, [ac_req, ac_auto], ac_out)

    with gr.Tab("Status"):
        st_btn = gr.Button("Refresh")
        st_out = gr.Textbox(label="System status", lines=20, interactive=False)
        st_btn.click(status, [], st_out)
        demo.load(status, [], st_out)


def launch(port: int = 7860, share: bool = False) -> None:
    demo.launch(server_port=port, share=share, inbrowser=True)


if __name__ == "__main__":
    launch()
