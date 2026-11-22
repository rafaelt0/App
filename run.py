import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.request


ROOT = Path(__file__).resolve().parent
KEEPALIVE_SECONDS = 600
READY_TIMEOUT_SECONDS = 300
# A process that survives this long is considered healthy, so its next exit starts a fresh count.
STABLE_SECONDS = 60
# Consecutive early exits before the launcher gives up and lets the platform restart the container.
MAX_CRASHES = 5
BACKOFF_SECONDS = (1, 2, 5, 10, 20, 30)


def restart_delay(crashes: int) -> float:
    """Seconds to wait before the next attempt after `crashes` early exits."""
    return BACKOFF_SECONDS[min(max(crashes, 0), len(BACKOFF_SECONDS) - 1)]


def plan_restart(crashes: int, ran_seconds: float) -> tuple[int, float | None]:
    """Next (crashes, delay) after an exit; a delay of None means the container should restart."""
    if ran_seconds >= STABLE_SECONDS:
        crashes = 0
    crashes += 1
    if crashes >= MAX_CRASHES:
        return crashes, None
    return crashes, restart_delay(crashes - 1)


def keep_alive(url, stopping):
    # O plano gratuito do Render suspende o serviço após 15 min sem tráfego externo.
    while not stopping.wait(KEEPALIVE_SECONDS):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                response.read(1)
        except Exception as error:
            print(f"Keep-alive falhou: {error}", flush=True)


def stop_group(process, timeout=5) -> None:
    """Stop a child and every descendant it started, freeing its port for a restart."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass
    # npm keeps `next-server` in the same group; the leader may exit before its descendants.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass


def shutdown(managed: list["Managed"], timeout=28) -> None:
    """Let in-flight requests finish inside the platform's grace period, then force the rest."""
    running = [item.process for item in managed if item.process is not None]
    for process in running:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    deadline = time.monotonic() + timeout
    for process in running:
        try:
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for process in running:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass


@dataclass
class Managed:
    name: str
    args: list[str]
    cwd: Path
    process: subprocess.Popen | None = None
    started: float = 0.0
    crashes: int = 0
    resume_at: float = 0.0

    def start(self, environment) -> None:
        self.process = subprocess.Popen(self.args, cwd=self.cwd, env=environment, start_new_session=True)
        self.started = time.monotonic()
        self.resume_at = 0.0

    def exited(self) -> bool:
        return self.process is not None and self.process.poll() is not None


class CrashLoop(Exception):
    """A child keeps exiting before it can serve, so the whole container must restart."""


def start_child(item: "Managed", environment, now: float) -> None:
    """Start a child; a transient spawn failure retries on the same bounded backoff as an exit."""
    try:
        item.start(environment)
    except OSError as error:
        item.crashes, delay = plan_restart(item.crashes, 0)
        if delay is None:
            raise CrashLoop(f"{item.name} não iniciou após {item.crashes} tentativas ({error}).") from error
        item.resume_at = now + delay
        print(f"{item.name} não iniciou ({error}); nova tentativa em {delay:.0f}s.", flush=True)
        return
    if item.crashes:
        print(f"{item.name} reiniciado.", flush=True)


def supervise(managed: list[Managed], environment, stopping) -> None:
    """Start each child once and bring back the ones that exit, without taking down its siblings."""
    now = time.monotonic()
    for item in managed:
        start_child(item, environment, now)
    while not stopping.wait(0.4):
        now = time.monotonic()
        for item in managed:
            if item.process is None:
                if now >= item.resume_at:
                    start_child(item, environment, now)
                continue
            if not item.exited():
                continue
            code = item.process.returncode
            stop_group(item.process)
            item.process = None
            item.crashes, delay = plan_restart(item.crashes, now - item.started)
            if delay is None:
                raise CrashLoop(f"{item.name} encerrou {item.crashes} vezes seguidas (código {code}).")
            item.resume_at = now + delay
            print(f"{item.name} encerrou (código {code}); nova tentativa em {delay:.0f}s.", flush=True)


def healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status < 400
    except Exception:
        return False


def announce_ready(urls: list[str], stopping) -> None:
    """Log once when every process answers in two consecutive rounds, so the boot log is honest."""
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    consecutive = 0
    while not stopping.is_set() and time.monotonic() < deadline:
        with ThreadPoolExecutor(max_workers=len(urls)) as pool:
            answers = list(pool.map(healthy, urls))
        consecutive = consecutive + 1 if all(answers) else 0
        if consecutive >= 2:
            print("Pronto: painel e API respondendo.", flush=True)
            return
        stopping.wait(1)
    print(f"Prontidão não confirmada para: {', '.join(urls)}.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Inicia o B3Lab em Next.js e a API Python.")
    parser.add_argument("--production", action="store_true", help="Usa o build Next.js produzido por npm run build.")
    args = parser.parse_args()
    port = os.environ.get("B3LAB_PORT", "3000")
    api_port = os.environ.get("B3LAB_API_PORT", "8000")
    environment = os.environ.copy()
    environment.setdefault("NEXT_TELEMETRY_DISABLED", "1")
    environment.setdefault("B3LAB_API_URL", f"http://127.0.0.1:{api_port}")
    environment.setdefault("B3LAB_ORIGIN", f"http://localhost:{port}")
    environment.setdefault("B3LAB_WARM_CACHES", "1")
    if not (ROOT / "frontend/node_modules/next").exists():
        parser.error("Execute npm ci no diretório frontend antes de iniciar.")
    stopping = threading.Event()
    for event in (signal.SIGTERM, signal.SIGINT):
        signal.signal(event, lambda *_: stopping.set())
    # Invoking Next directly skips the npm wrapper process, which costs about 70 MB on a 512 MB instance.
    next_cli = ROOT / "frontend/node_modules/.bin/next"
    if not next_cli.exists():
        parser.error("Execute npm ci no diretório frontend antes de iniciar.")
    managed = [
        Managed("API Python", [sys.executable, "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", api_port, "--workers", "1"], ROOT),
        Managed("Next.js", [str(next_cli), "start" if args.production else "dev", "--hostname", os.environ.get("B3LAB_HOST", "127.0.0.1"), "--port", port], ROOT / "frontend"),
    ]
    try:
        print(f"Iniciando B3Lab em http://localhost:{port}. API privada em 127.0.0.1:{api_port}.", flush=True)
        if keepalive_url := os.environ.get("B3LAB_KEEPALIVE_URL"):
            threading.Thread(target=keep_alive, args=(keepalive_url, stopping), daemon=True).start()
        threading.Thread(target=announce_ready, args=([f"http://127.0.0.1:{port}/", f"http://127.0.0.1:{api_port}/health"], stopping), daemon=True).start()
        supervise(managed, environment, stopping)
        return 0
    except CrashLoop as loop:
        print(str(loop), flush=True)
        return 1
    finally:
        shutdown(managed)


if __name__ == "__main__":
    raise SystemExit(main())
