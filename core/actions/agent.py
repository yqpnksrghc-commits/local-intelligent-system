"""
Action agent — natural language interface to real-world actions.

Parses intent from a plain-language request, routes to the correct
action module, and executes with the gate in place.

Examples:
  "Call +12025551234 and tell them the meeting is at 3pm"
  "Text Sarah at +14155552345 that I'm running 10 minutes late"
  "Find me a Samsung Galaxy under $800 on Amazon and order it"
  "Search for iPhone 15 Pro on Amazon"
"""
from __future__ import annotations
import json
import re
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

_SYSTEM = SystemMessage(content="""
You are an action router. The user prefers Apple products.
Parse the user's request and return ONLY valid JSON:

For calls:
{"action": "call", "to": "<E.164 phone number>", "message": "<what to say>", "language": "en-US"}

For SMS:
{"action": "sms", "to": "<E.164 phone number>", "message": "<text body>"}

For product search:
{"action": "search", "query": "<search terms>", "retailer": "amazon", "max_price": null}

For product order (search first):
{"action": "order", "query": "<search terms>", "retailer": "amazon", "max_price": null}

When the user asks for a phone or device without specifying brand, default query to iPhone or the
relevant Apple product (MacBook, iPad, AirPods, etc.).

If the request is ambiguous or missing a phone number, return:
{"action": "clarify", "question": "<what you need to know>"}

Return only the JSON. No explanation.
""")


def run(request: str, llm: ChatOllama, confirm_first: bool = True) -> str:
    """Parse and execute a natural-language action request."""
    response = llm.invoke([_SYSTEM, HumanMessage(content=request)])
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json\n")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return f"Could not parse action intent: {raw}"

    action = parsed.get("action", "")

    if action == "clarify":
        return f"Need clarification: {parsed.get('question', '?')}"

    elif action == "call":
        from core.actions import telephony
        return telephony.call(
            to=parsed["to"],
            message=parsed.get("message", "Hello."),
            language=parsed.get("language", "en-US"),
            confirm_first=confirm_first,
        )

    elif action == "sms":
        from core.actions import telephony
        return telephony.sms(
            to=parsed["to"],
            message=parsed.get("message", ""),
            confirm_first=confirm_first,
        )

    elif action == "search":
        from core.actions import procurement
        products = procurement.search(
            query=parsed.get("query", ""),
            retailer=parsed.get("retailer", "amazon"),
        )
        if not products:
            return "No products found."
        lines = [f"Found {len(products)} results:"]
        for i, p in enumerate(products, 1):
            price = f"${p.price:.2f}" if p.price else "?"
            lines.append(f"  {i}. [{price}] {p.title[:70]}")
        return "\n".join(lines)

    elif action == "order":
        from core.actions import procurement
        return procurement.get_phone(
            query=parsed.get("query", "smartphone"),
            max_price=parsed.get("max_price"),
            retailer=parsed.get("retailer", "amazon"),
            confirm_first=confirm_first,
        )

    else:
        return f"Unknown action: {action!r}"
