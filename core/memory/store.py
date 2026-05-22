"""
SQLite-backed long-term memory: store and retrieve conversation summaries.
"""
import sqlite3
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
DB_PATH = os.getenv("LONG_TERM_DB", "./data/long_term/memory.db")


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            ts        TEXT NOT NULL,
            role      TEXT NOT NULL,
            content   TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def save(role: str, content: str):
    with _conn() as conn:
        conn.execute(
            "INSERT INTO memories (ts, role, content) VALUES (?, ?, ?)",
            (datetime.utcnow().isoformat(), role, content),
        )


def recent(n: int = 20) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT ts, role, content FROM memories ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
    return [{"ts": r[0], "role": r[1], "content": r[2]} for r in reversed(rows)]
