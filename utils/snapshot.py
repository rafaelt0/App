import logging
import threading
import time

logger = logging.getLogger(__name__)


class Snapshot:
    """Last good result of `fetch`, served stale while a single background refresh replaces it.

    Process-local and in memory: a restart or deploy starts without a snapshot, and
    the first read then waits for a real fetch instead of inventing one.
    """

    def __init__(self, fetch, ttl):
        self._fetch, self._ttl = fetch, ttl
        self._lock = threading.Lock()
        self._value = None
        self._stored_at = None
        self._error = None
        self._worker = None

    def _run(self):
        try:
            value = self._fetch()
        except Exception as exc:
            logger.warning("Snapshot refresh failed", exc_info=True)
            with self._lock:
                self._error = str(exc) or type(exc).__name__
        else:
            with self._lock:
                self._value, self._stored_at, self._error = value, time.monotonic(), None
        finally:
            with self._lock:
                self._worker = None

    def _start(self):
        if self._worker is None:
            self._error = None
            self._worker = threading.Thread(target=self._run, daemon=True)
            self._worker.start()
        return self._worker

    def _state(self, refreshing):
        if self._value is None:
            raise ValueError(self._error or "Sem dados disponíveis.")
        stale = time.monotonic() - self._stored_at >= self._ttl
        return {**self._value, "stale": stale, "refreshing": refreshing, "refresh_error": self._error}

    def read(self, wait=False):
        """Fresh value as is; a stale one immediately with a refresh started, unless `wait` joins it."""
        with self._lock:
            if self._value is not None and time.monotonic() - self._stored_at < self._ttl:
                return self._state(False)
            worker = self._start()
            if self._value is not None and not wait:
                return self._state(True)
        worker.join()
        with self._lock:
            return self._state(False)

    def refresh(self):
        """Fetch now, joining a refresh already running, and return the outcome."""
        with self._lock:
            worker = self._start()
        worker.join()
        with self._lock:
            return self._state(False)

    def clear(self):
        with self._lock:
            self._value = self._stored_at = self._error = None
