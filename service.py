"""
service.py — the unified launcher.

Starts everything as one process:
  · FastAPI backend   (port 8000)
  · Gradio frontend   (port 7860)
  · Self-sufficiency daemons (health, watcher, summarizer)
  · Twilio inbound webhook (port 5001, optional)

Usage:
    python service.py
    python service.py --no-ui          # API only, no browser
    python service.py --port 8080      # custom API port
    python service.py --ngrok          # expose webhook via ngrok
"""
import argparse
import threading
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()


def _start_api(port: int) -> threading.Thread:
    def run():
        uvicorn.run(
            "core.api:app",
            host="0.0.0.0",
            port=port,
            log_level="warning",
        )
    t = threading.Thread(target=run, daemon=True)
    t.start()
    print(f"[service] API running on http://localhost:{port}")
    return t


def _start_ui(api_url: str, port: int) -> threading.Thread:
    os.environ["API_URL"] = api_url

    def run():
        from frontend.app import launch
        launch(port=port)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    print(f"[service] UI running on http://localhost:{port}")
    return t


def _start_webhook(ngrok: bool = False) -> threading.Thread:
    webhook_port = int(os.getenv("WEBHOOK_PORT", "5001"))
    if not os.getenv("TWILIO_ACCOUNT_SID"):
        return None
    from core.actions.telephony import start_webhook
    return start_webhook(port=webhook_port, ngrok=ngrok)


def main():
    parser = argparse.ArgumentParser(description="Local Intelligent System service")
    parser.add_argument("--port",    type=int, default=8000, help="API port")
    parser.add_argument("--ui-port", type=int, default=7860, help="UI port")
    parser.add_argument("--no-ui",   action="store_true", help="Start API only")
    parser.add_argument("--ngrok",   action="store_true", help="Expose webhook via ngrok")
    args = parser.parse_args()

    print("\nLocal Intelligent System")
    print("─" * 40)

    api_thread = _start_api(args.port)

    if not args.no_ui:
        import time; time.sleep(2)   # give API a moment to bind
        _start_ui(f"http://localhost:{args.port}", args.ui_port)

    _start_webhook(ngrok=args.ngrok)

    print("\nSystem ready. Press Ctrl+C to stop.\n")
    try:
        api_thread.join()
    except KeyboardInterrupt:
        print("\n[service] shutting down.")


if __name__ == "__main__":
    main()
