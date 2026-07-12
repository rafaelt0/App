"""SQLite-backed persistent cache, watchlist and portfolio for B3 Explorer.

Watchlist and portfolio are keyed by an anonymous per-browser `uid`
(see utils/identity.py) so different visitors don't share the same data.
"""
import sqlite3
import json
import time
import os

_DB = os.path.join(os.path.dirname(__file__), '..', 'b3_data.db')


def _migrate_legacy_watchlist(conn):
    """Add the `uid` column to a pre-existing single-user watchlist table.

    Older deployments created `watchlist(ticker PRIMARY KEY, added_at)`
    shared by every visitor. Existing rows are kept under a "legacy" uid
    instead of being dropped.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(watchlist)").fetchall()}
    if cols and "uid" not in cols:
        conn.execute("ALTER TABLE watchlist RENAME TO watchlist_legacy")
        conn.execute("""
            CREATE TABLE watchlist (
                uid       TEXT NOT NULL,
                ticker    TEXT NOT NULL,
                added_at  REAL NOT NULL,
                PRIMARY KEY (uid, ticker)
            )
        """)
        conn.execute("""
            INSERT INTO watchlist (uid, ticker, added_at)
            SELECT 'legacy', ticker, added_at FROM watchlist_legacy
        """)
        conn.execute("DROP TABLE watchlist_legacy")


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
            uid       TEXT NOT NULL,
            ticker    TEXT NOT NULL,
            added_at  REAL NOT NULL,
            PRIMARY KEY (uid, ticker)
        )
    """)
    _migrate_legacy_watchlist(conn)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            uid        TEXT PRIMARY KEY,
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

def wl_get(uid: str) -> list[str]:
    """Return this visitor's watchlist tickers ordered by insertion time."""
    try:
        with _conn() as c:
            rows = c.execute(
                "SELECT ticker FROM watchlist WHERE uid = ? ORDER BY added_at",
                (uid,),
            ).fetchall()
            return [r[0] for r in rows]
    except Exception:
        return []


def wl_add(uid: str, ticker: str):
    try:
        with _conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO watchlist VALUES (?, ?, ?)",
                (uid, ticker.upper(), time.time()),
            )
    except Exception:
        pass


def wl_remove(uid: str, ticker: str):
    try:
        with _conn() as c:
            c.execute(
                "DELETE FROM watchlist WHERE uid = ? AND ticker = ?",
                (uid, ticker.upper()),
            )
    except Exception:
        pass


def wl_has(uid: str, ticker: str) -> bool:
    try:
        with _conn() as c:
            row = c.execute(
                "SELECT 1 FROM watchlist WHERE uid = ? AND ticker = ?",
                (uid, ticker.upper()),
            ).fetchone()
            return row is not None
    except Exception:
        return False


# ── Portfolio (última carteira montada) ────────────────────────────────────────

def portfolio_get(uid: str) -> tuple[list[str], dict]:
    """Return (tickers, weights) from this visitor's last saved portfolio."""
    try:
        with _conn() as c:
            row = c.execute(
                "SELECT tickers, weights FROM portfolio WHERE uid = ?", (uid,)
            ).fetchone()
            if row:
                tickers = json.loads(row[0])
                weights = json.loads(row[1]) if row[1] else {}
                return tickers, weights
    except Exception:
        pass
    return [], {}


def portfolio_save(uid: str, tickers: list[str], weights: dict | None = None):
    """Persist this visitor's current ticker selection and manual weights."""
    try:
        with _conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO portfolio VALUES (?, ?, ?, ?)",
                (
                    uid,
                    json.dumps(tickers),
                    json.dumps(weights) if weights else None,
                    time.time(),
                ),
            )
    except Exception:
        pass


def portfolio_clear(uid: str):
    try:
        with _conn() as c:
            c.execute("DELETE FROM portfolio WHERE uid = ?", (uid,))
    except Exception:
        pass
