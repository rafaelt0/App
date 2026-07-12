"""SQLite-backed persistent cache and watchlist for B3 Explorer."""
import sqlite3
import json
import time
import os

_DB = os.path.join(os.path.dirname(__file__), '..', 'b3_data.db')


def _conn():
    conn = sqlite3.connect(_DB, check_same_thread=False, timeout=10)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            key  TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            ts   REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            ticker    TEXT PRIMARY KEY,
            added_at  REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            id         INTEGER PRIMARY KEY CHECK (id = 1),
            tickers    TEXT NOT NULL,
            weights    TEXT,
            updated_at REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


# ── Cache ──────────────────────────────────────────────────────────────────────

def cache_get(key: str, ttl: int = 3600):
    """Return cached value if fresh, else None."""
    try:
        with _conn() as c:
            row = c.execute(
                "SELECT data, ts FROM cache WHERE key = ?", (key,)
            ).fetchone()
            if row and time.time() - row[1] < ttl:
                return json.loads(row[0])
    except Exception:
        pass
    return None


def cache_set(key: str, data):
    """Persist data in cache."""
    try:
        with _conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO cache VALUES (?, ?, ?)",
                (key, json.dumps(data, default=str), time.time()),
            )
    except Exception:
        pass


def cache_clear_expired(max_age: int = 86400):
    """Delete entries older than max_age seconds (default 24h)."""
    try:
        with _conn() as c:
            c.execute("DELETE FROM cache WHERE ts < ?", (time.time() - max_age,))
    except Exception:
        pass


# ── Watchlist ──────────────────────────────────────────────────────────────────

def wl_get() -> list[str]:
    """Return watchlist tickers ordered by insertion time."""
    try:
        with _conn() as c:
            rows = c.execute(
                "SELECT ticker FROM watchlist ORDER BY added_at"
            ).fetchall()
            return [r[0] for r in rows]
    except Exception:
        return []


def wl_add(ticker: str):
    try:
        with _conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO watchlist VALUES (?, ?)",
                (ticker.upper(), time.time()),
            )
    except Exception:
        pass


def wl_remove(ticker: str):
    try:
        with _conn() as c:
            c.execute("DELETE FROM watchlist WHERE ticker = ?", (ticker.upper(),))
    except Exception:
        pass


def wl_has(ticker: str) -> bool:
    try:
        with _conn() as c:
            row = c.execute(
                "SELECT 1 FROM watchlist WHERE ticker = ?", (ticker.upper(),)
            ).fetchone()
            return row is not None
    except Exception:
        return False


# ── Portfolio (última carteira montada) ────────────────────────────────────────

def portfolio_get() -> tuple[list[str], dict]:
    """Return (tickers, weights) from the last saved portfolio, or ([], {})."""
    try:
        with _conn() as c:
            row = c.execute(
                "SELECT tickers, weights FROM portfolio WHERE id = 1"
            ).fetchone()
            if row:
                tickers = json.loads(row[0])
                weights = json.loads(row[1]) if row[1] else {}
                return tickers, weights
    except Exception:
        pass
    return [], {}


def portfolio_save(tickers: list[str], weights: dict | None = None):
    """Persist the current ticker selection and (optional) manual weights."""
    try:
        with _conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO portfolio VALUES (1, ?, ?, ?)",
                (
                    json.dumps(tickers),
                    json.dumps(weights) if weights else None,
                    time.time(),
                ),
            )
    except Exception:
        pass


def portfolio_clear():
    try:
        with _conn() as c:
            c.execute("DELETE FROM portfolio WHERE id = 1")
    except Exception:
        pass
