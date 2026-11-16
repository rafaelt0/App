"""Keeps the Home page's shared market caches warm so visitors never wait on a cold fetch.

The Yahoo download for the whole universe costs tens of seconds of CPU on the
host. Priming at boot and touching the caches during B3 hours moves that cost
off the request path; the caches' stale windows cover the remaining gaps.
"""

import datetime
import logging
import threading
import time
from zoneinfo import ZoneInfo

from backend.news import _market_headlines
from utils.market_data import get_listed_stocks, get_liquidity_by_ticker, get_universe_changes
from utils.market_strip import market_strip

logger = logging.getLogger(__name__)
BRASILIA = ZoneInfo("America/Sao_Paulo")
INTERVAL_SECONDS = 240
# Regular session with margin for the opening auction and the closing call.
SESSION_HOURS = (datetime.time(9, 45), datetime.time(18, 30))
TASKS = (
    ("universe changes", lambda: get_universe_changes(tuple(get_listed_stocks()["Ticker"]))),
    ("market strip", market_strip),
    ("liquidity", get_liquidity_by_ticker),
    ("headlines", _market_headlines),
)


def in_session(now: datetime.datetime) -> bool:
    local = now.astimezone(BRASILIA)
    return local.weekday() < 5 and SESSION_HOURS[0] <= local.time() <= SESSION_HOURS[1]


def warm() -> None:
    """Touch each Home cache: fills a missing entry, or starts a background refresh of a stale one.

    Each provider group has its own thread, so priming costs the slowest group
    instead of their sum. Yahoo stays serialized by its own lock, and the
    universe download goes first because every visitor's heatmap waits on it.
    """
    threads = [threading.Thread(target=_warm_one, args=(name, task), name=f"warm-{name}", daemon=True)
               for name, task in TASKS]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def _warm_one(name, task) -> None:
    try:
        task()
    except Exception:
        logger.warning("Cache warm-up of %s failed", name, exc_info=True)


def run(stopping: threading.Event) -> None:
    started = time.monotonic()
    warm()
    logger.info("Home caches primed in %.1fs", time.monotonic() - started)
    while not stopping.wait(INTERVAL_SECONDS):
        if in_session(datetime.datetime.now(datetime.timezone.utc)):
            warm()


def start() -> threading.Event:
    stopping = threading.Event()
    threading.Thread(target=run, args=(stopping,), name="cache-warmup", daemon=True).start()
    return stopping
