"""
Telephony — make calls, send SMS, receive inbound via webhook.

Outbound calls use Twilio's TTS engine (supports 50+ languages/voices).
The meaning system can render content into any language before dialing.

Required env vars:
  TWILIO_ACCOUNT_SID
  TWILIO_AUTH_TOKEN
  TWILIO_PHONE_NUMBER   — your Twilio number (E.164 format, e.g. +12025551234)

Optional:
  WEBHOOK_URL           — public URL for inbound call/SMS webhook
  WEBHOOK_PORT          — local port for webhook server (default 5001)
"""
from __future__ import annotations
import os
import threading
from dotenv import load_dotenv
from core.actions.gate import Action, confirm, ActionDenied

load_dotenv()

_SID    = os.getenv("TWILIO_ACCOUNT_SID", "")
_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN", "")
_FROM   = os.getenv("TWILIO_PHONE_NUMBER", "")


def _client():
    from twilio.rest import Client
    if not _SID or not _TOKEN:
        raise RuntimeError(
            "Twilio credentials missing. Set TWILIO_ACCOUNT_SID and "
            "TWILIO_AUTH_TOKEN in .env"
        )
    return Client(_SID, _TOKEN)


# ── Outbound call ─────────────────────────────────────────────────────────────

def call(
    to: str,
    message: str,
    language: str = "en-US",
    voice: str = "Polly.Joanna",
    confirm_first: bool = True,
) -> str:
    """
    Dial `to` and speak `message` using TTS.
    Returns the Twilio call SID.
    """
    action = Action(
        kind="call",
        description=f"Call {to} and say: {message[:80]!r}",
        cost=0.015,   # ~$0.015/min Twilio outbound
        reversible=False,
        payload={"to": to, "message": message, "language": language},
    )
    confirm(action, auto_approve=not confirm_first)

    twiml = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<Response><Say language="{language}" voice="{voice}">{message}</Say></Response>'
    )
    call_obj = _client().calls.create(
        twiml=twiml,
        to=to,
        from_=_FROM,
    )
    print(f"[telephony] call initiated: {call_obj.sid}")
    return call_obj.sid


# ── SMS ───────────────────────────────────────────────────────────────────────

def sms(
    to: str,
    message: str,
    confirm_first: bool = True,
) -> str:
    """Send an SMS. Returns the message SID."""
    action = Action(
        kind="sms",
        description=f"SMS to {to}: {message[:80]!r}",
        cost=0.0079,
        reversible=False,
        payload={"to": to, "message": message},
    )
    confirm(action, auto_approve=not confirm_first)

    msg = _client().messages.create(body=message, from_=_FROM, to=to)
    print(f"[telephony] SMS sent: {msg.sid}")
    return msg.sid


# ── Inbound webhook server ─────────────────────────────────────────────────────

_inbound_handlers: list = []


def on_inbound(handler) -> None:
    """Register a callback for inbound calls/SMS. handler(event: dict) -> str (TwiML)."""
    _inbound_handlers.append(handler)


def start_webhook(port: int | None = None, ngrok: bool = False) -> threading.Thread:
    """
    Start a Flask webhook server for inbound Twilio events.
    Set WEBHOOK_PORT in .env or pass port directly.
    If ngrok=True, attempts to expose via pyngrok (must be installed).
    """
    port = port or int(os.getenv("WEBHOOK_PORT", "5001"))

    try:
        from flask import Flask, request, Response
    except ImportError:
        raise RuntimeError("pip install flask  to use inbound webhook")

    app = Flask(__name__)

    @app.route("/inbound/call", methods=["POST"])
    def inbound_call():
        event = request.form.to_dict()
        print(f"[telephony] inbound call from {event.get('From', '?')}")
        twiml = '<Response><Say>Message received.</Say></Response>'
        for h in _inbound_handlers:
            result = h({"type": "call", **event})
            if result:
                twiml = result
                break
        return Response(twiml, mimetype="text/xml")

    @app.route("/inbound/sms", methods=["POST"])
    def inbound_sms():
        event = request.form.to_dict()
        body  = event.get("Body", "")
        frm   = event.get("From", "?")
        print(f"[telephony] inbound SMS from {frm}: {body!r}")
        for h in _inbound_handlers:
            h({"type": "sms", "from": frm, "body": body})
        return Response('<Response/>', mimetype="text/xml")

    def _run():
        if ngrok:
            try:
                from pyngrok import ngrok as ng
                tunnel = ng.connect(port)
                print(f"[telephony] ngrok tunnel: {tunnel.public_url}")
            except Exception as e:
                print(f"[telephony] ngrok failed: {e}")
        app.run(port=port, debug=False, use_reloader=False)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    print(f"[telephony] webhook server on port {port}")
    return t


# ── Status / listing ──────────────────────────────────────────────────────────

def recent_calls(n: int = 5) -> list[dict]:
    calls = _client().calls.list(limit=n)
    return [{"sid": c.sid, "to": c.to, "status": c.status,
             "duration": c.duration, "start": str(c.start_time)} for c in calls]


def recent_sms(n: int = 5) -> list[dict]:
    msgs = _client().messages.list(limit=n)
    return [{"sid": m.sid, "to": m.to, "body": m.body,
             "status": m.status, "date": str(m.date_sent)} for m in msgs]
