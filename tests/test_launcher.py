import os
from pathlib import Path
import signal
import threading
import time

import run


def wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_first_restart_is_quick_and_backoff_is_capped():
    assert run.restart_delay(0) == 1
    assert run.restart_delay(1) == 2
    assert run.restart_delay(5) == 30
    assert run.restart_delay(9) == 30


def test_early_exits_accumulate_until_the_container_restarts():
    crashes, delay = run.plan_restart(0, 2)
    assert (crashes, delay) == (1, 1)
    crashes, delay = run.plan_restart(3, 2)
    assert (crashes, delay) == (4, 10)
    assert run.plan_restart(4, 2) == (5, None)


def test_a_long_run_earns_a_fresh_restart_count():
    assert run.plan_restart(4, run.STABLE_SECONDS) == (1, 1)


def test_supervise_replaces_a_child_that_exits(monkeypatch):
    monkeypatch.setattr(run, "BACKOFF_SECONDS", (0,))
    child = run.Managed("teste", ["sleep", "30"], Path("/tmp"))
    stopping = threading.Event()
    supervisor = threading.Thread(target=run.supervise, args=([child], os.environ.copy(), stopping), daemon=True)
    supervisor.start()
    try:
        assert wait_for(lambda: child.process is not None)
        first = child.process.pid
        os.kill(first, signal.SIGKILL)
        assert wait_for(lambda: child.process is not None and child.process.pid != first)
        assert child.process.poll() is None
    finally:
        stopping.set()
        supervisor.join(timeout=10)
        run.shutdown([child], timeout=2)
    assert not supervisor.is_alive()


def test_a_transient_spawn_failure_is_retried(monkeypatch):
    monkeypatch.setattr(run, "BACKOFF_SECONDS", (0,))

    class Flaky(run.Managed):
        calls = 0

        def start(self, environment):
            self.calls += 1
            if self.calls == 1:
                raise OSError(11, "Resource temporarily unavailable")
            super().start(environment)

    child = Flaky("teste", ["sleep", "30"], Path("/tmp"))
    stopping = threading.Event()
    supervisor = threading.Thread(target=run.supervise, args=([child], os.environ.copy(), stopping), daemon=True)
    supervisor.start()
    try:
        assert wait_for(lambda: child.calls >= 2 and child.process is not None)
        assert child.process.poll() is None
    finally:
        stopping.set()
        supervisor.join(timeout=10)
        run.shutdown([child], timeout=2)
    assert not supervisor.is_alive()
