"""
Memory summarizer — runs in the background and compresses old memories
into a rolling summary every N turns, so the context window stays clean.
"""
import threading
import time
from core.memory import store


def _summarize(llm, every_n: int) -> None:
    count = 0
    while True:
        time.sleep(30)
        count += 1
        if count < every_n:
            continue
        count = 0

        memories = store.recent(40)
        if len(memories) < every_n:
            continue

        # build a condensed narrative from the oldest half
        to_compress = memories[: len(memories) // 2]
        text = "\n".join(f"{m['role']}: {m['content']}" for m in to_compress)
        prompt = (
            "Summarize the following conversation exchange into a single concise paragraph "
            "preserving all key facts, decisions, and context:\n\n" + text
        )
        try:
            summary = llm.invoke(prompt).content
            store.save("summary", summary)
            print(f"[summarizer] compressed {len(to_compress)} memories into 1 summary")
        except Exception as e:
            print(f"[summarizer] error: {e}")


def start(llm, every_n: int = 10) -> threading.Thread:
    t = threading.Thread(target=_summarize, args=(llm, every_n), daemon=True)
    t.start()
    print(f"[summarizer] running — compresses every {every_n} turns")
    return t
