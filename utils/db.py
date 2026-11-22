"""SQLite cache and anonymous-owner watchlist and allocation storage.

Owner identity is resolved by backend.state, never from a caller's URL or body.
"""
import sqlite3
import json
import time
import os
import logging
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)

_DB = os.environ.get("B3LAB_DB", os.path.join(os.path.dirname(__file__), '..', 'b3_data.db'))
_READY_TTL = 15.0
_ready_lock = threading.Lock()
_ready_at = 0.0
_ready_value = False
_ready_key = None


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


@contextmanager
def _conn():
    conn = sqlite3.connect(_DB, check_same_thread=False, timeout=10)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                ts REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                uid TEXT NOT NULL,
                ticker TEXT NOT NULL,
                added_at REAL NOT NULL,
                PRIMARY KEY (uid, ticker)
            )
        """)
        _migrate_legacy_watchlist(conn)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS portfolio (
                uid TEXT PRIMARY KEY,
                tickers TEXT NOT NULL,
                weights TEXT,
                updated_at REAL NOT NULL
            )
        """)
        conn.commit()
        with conn:
            yield conn
    finally:
        conn.close()


def ready() -> bool:
    """Whether the store accepts a write; a read-only file answers SELECT but cannot persist sessions.

    The probe is cached because the platform asks every few seconds and each check takes a write lock.
    """
    global _ready_at, _ready_value, _ready_key
    now = time.monotonic()
    with _ready_lock:
        if _ready_key == _DB and now - _ready_at < _READY_TTL:
            return _ready_value
        _ready_value = _write_probe()
        _ready_at = now
        _ready_key = _DB
        return _ready_value


def _write_probe() -> bool:
    connection = None
    try:
        connection = sqlite3.connect(_DB, timeout=2)
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("CREATE TABLE IF NOT EXISTS __b3lab_ready (ok INTEGER)")
        connection.rollback()
        return True
    except sqlite3.Error:
        logger.exception("Readiness probe failed for %s", _DB)
        return False
    finally:
        if connection is not None:
            connection.close()


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
        logger.warning("cache_get failed for key=%s", key, exc_info=True)
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
        logger.warning("cache_set failed for key=%s", key, exc_info=True)


def cache_clear_expired(max_age: int = 86400):
    """Delete entries older than max_age seconds (default 24h)."""
    try:
        with _conn() as c:
            c.execute("DELETE FROM cache WHERE ts < ?", (time.time() - max_age,))
    except Exception:
        logger.warning("cache_clear_expired failed", exc_info=True)


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
        logger.warning("wl_get failed for uid=%s", uid, exc_info=True)
        return []


def wl_add(uid: str, ticker: str):
    try:
        with _conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO watchlist VALUES (?, ?, ?)",
                (uid, ticker.upper(), time.time()),
            )
        return True
    except Exception:
        logger.warning("wl_add failed for uid=%s ticker=%s", uid, ticker, exc_info=True)
        return False


def wl_remove(uid: str, ticker: str):
    try:
        with _conn() as c:
            c.execute(
                "DELETE FROM watchlist WHERE uid = ? AND ticker = ?",
                (uid, ticker.upper()),
            )
        return True
    except Exception:
        logger.warning("wl_remove failed for uid=%s ticker=%s", uid, ticker, exc_info=True)
        return False


def wl_has(uid: str, ticker: str) -> bool:
    try:
        with _conn() as c:
            row = c.execute(
                "SELECT 1 FROM watchlist WHERE uid = ? AND ticker = ?",
                (uid, ticker.upper()),
            ).fetchone()
            return row is not None
    except Exception:
        logger.warning("wl_has failed for uid=%s ticker=%s", uid, ticker, exc_info=True)
        return False


def cache_clear_prefix(prefix: str) -> int:
    """Delete cached entries whose keys start with prefix and return the count."""
    if not prefix:
        raise ValueError("cache prefix must not be empty")
    try:
        escaped_prefix = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        with _conn() as c:
            cursor = c.execute(
                "DELETE FROM cache WHERE key LIKE ? ESCAPE '\\'",
                (f"{escaped_prefix}%",),
            )
            return cursor.rowcount
    except Exception:
        logger.warning("cache_clear_prefix failed for prefix=%s", prefix, exc_info=True)
        return 0



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
        logger.warning("portfolio_get failed for uid=%s", uid, exc_info=True)
    return [], {}


def portfolio_save(uid: str, tickers: list[str], weights: dict | None = None):
    """Persist the allocation and report whether the transaction committed."""
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
        return True
    except Exception:
        logger.warning("portfolio_save failed for uid=%s", uid, exc_info=True)
        return False


def portfolio_clear(uid: str):
    try:
        with _conn() as c:
            c.execute("DELETE FROM portfolio WHERE uid = ?", (uid,))
        return True
    except Exception:
        logger.warning("portfolio_clear failed for uid=%s", uid, exc_info=True)
        return False
