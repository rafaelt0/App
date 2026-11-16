import datetime
import threading

import pytest

from backend import warmup
from utils import cache


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(cache.time, "monotonic", lambda: now[0])
    return now


def join_refreshes():
    for thread in threading.enumerate():
        if thread.name.startswith("refresh-"):
            thread.join(5)


def counter(ttl, stale, gate=None):
    calls = []

    @cache.cache_data(ttl=ttl, stale=stale)
    def value():
        calls.append(None)
        if gate is not None:
            gate.wait(5)
        return len(calls)

    return value, calls


def test_fresh_entry_is_reused(clock):
    value, calls = counter(ttl=10, stale=0)
    assert value() == 1
    clock[0] += 9
    assert value() == 1
    assert len(calls) == 1


def test_expired_entry_without_stale_window_blocks_on_recompute(clock):
    value, _ = counter(ttl=10, stale=0)
    value()
    clock[0] += 10
    assert value() == 2


def test_stale_entry_is_served_while_one_background_refresh_runs(clock):
    gate = threading.Event()
    value, calls = counter(ttl=10, stale=100, gate=gate)
    gate.set()
    assert value() == 1
    gate.clear()
    clock[0] += 50
    assert value() == 1, "stale value returned without waiting"
    assert value() == 1
    gate.set()
    join_refreshes()
    assert len(calls) == 2, "concurrent stale reads share one refresh"
    assert value() == 2


def test_entry_past_the_stale_window_is_recomputed_in_line(clock):
    value, _ = counter(ttl=10, stale=100)
    value()
    clock[0] += 110
    assert value() == 2


def test_failed_background_refresh_keeps_the_stale_entry(clock):
    attempts = []

    @cache.cache_data(ttl=10, stale=100)
    def flaky():
        attempts.append(None)
        if len(attempts) > 1:
            raise RuntimeError("provider down")
        return "first"

    assert flaky() == "first"
    clock[0] += 20
    assert flaky() == "first"
    join_refreshes()
    assert flaky() == "first"


def test_clear_discards_a_refresh_that_started_before_it(clock):
    gate = threading.Event()
    value, _ = counter(ttl=10, stale=100, gate=gate)
    gate.set()
    value()
    gate.clear()
    clock[0] += 20
    value()
    value.clear()
    gate.set()
    join_refreshes()
    assert value() == 3


@pytest.mark.parametrize(("utc", "expected"), [
    ("2026-11-16T13:00:00", True),   # Monday 10:00 in Brasília
    ("2026-11-16T12:30:00", False),  # Monday 09:30, before the opening margin
    ("2026-11-16T21:00:00", True),   # Monday 18:00
    ("2026-11-16T22:00:00", False),  # Monday 19:00
    ("2026-11-14T15:00:00", False),  # Saturday
])
def test_warmup_runs_only_around_the_b3_session(utc, expected):
    now = datetime.datetime.fromisoformat(utc).replace(tzinfo=datetime.timezone.utc)
    assert warmup.in_session(now) is expected
