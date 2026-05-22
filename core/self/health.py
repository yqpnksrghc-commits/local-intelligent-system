"""
Ollama health — blocks until the endpoint is reachable, with exponential backoff.
Call wait_for_ollama() before constructing any LLM object.
"""
import time
import requests


def wait_for_ollama(host: str, max_wait: int = 120) -> None:
    url = f"{host.rstrip('/')}/api/tags"
    delay = 2
    elapsed = 0
    while elapsed < max_wait:
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                return
        except requests.exceptions.ConnectionError:
            pass
        print(f"[health] Ollama not ready — retrying in {delay}s …")
        time.sleep(delay)
        elapsed += delay
        delay = min(delay * 2, 30)
    raise RuntimeError(f"Ollama unreachable at {host} after {max_wait}s")


def is_alive(host: str) -> bool:
    try:
        return requests.get(f"{host.rstrip('/')}/api/tags", timeout=3).status_code == 200
    except Exception:
        return False
