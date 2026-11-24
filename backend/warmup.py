"""Keeps shared market caches warm so visitors never wait on a cold fetch.

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
from utils.market_data import (
    get_fii_market_data,
    get_liquidity_by_ticker,
    get_listed_stocks,
    get_universe_changes,
    get_us_market_data,
)
from utils.market_strip import market_strip
from utils.treasury_data import get_treasury_history

logger = logging.getLogger(__name__)
BRASILIA = ZoneInfo("America/Sao_Paulo")
INTERVAL_SECONDS = 240
# Regular session with margin for the opening auction and the closing call.
SESSION_HOURS = (datetime.time(9, 45), datetime.time(18, 30))


def _warm_yahoo_downloads() -> None:
    """Warm Yahoo caches in order: fast strip, US equities, then bulk B3 heatmap.

    Running the bulk 332-ticker B3 download last keeps the market strip instant and
    prevents crumb/401 contention while the 73 US equities prime in parallel.
    """
    _warm_one("market strip", market_strip)
    _warm_one("US equities", get_us_market_data)
    _warm_one("universe changes", lambda: get_universe_changes(tuple(get_listed_stocks()["Ticker"])))


TASKS = (
    ("yahoo caches", _warm_yahoo_downloads),
    ("liquidity", get_liquidity_by_ticker),
    ("FIIs", get_fii_market_data),
    ("headlines", _market_headlines),
    ("Tesouro Direto", get_treasury_history),
)


def in_session(now: datetime.datetime) -> bool:
    local = now.astimezone(BRASILIA)
    return local.weekday() < 5 and SESSION_HOURS[0] <= local.time() <= SESSION_HOURS[1]


def warm() -> None:
    """Touch each market data cache: fills a missing entry, or starts a background refresh of a stale one.

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
    logger.info("Shared market caches primed in %.1fs", time.monotonic() - started)
    while not stopping.wait(INTERVAL_SECONDS):
        if in_session(datetime.datetime.now(datetime.timezone.utc)):
            warm()


def start() -> threading.Event:
    stopping = threading.Event()
    threading.Thread(target=run, args=(stopping,), name="cache-warmup", daemon=True).start()
    return stopping
