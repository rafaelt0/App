"""Prove the launcher and readiness contract against the production build.

Starts run.py with an isolated database, waits for a serving app, kills the API
process, and requires /health to stay non-2xx until the API answers again while
the panel keeps serving. Run from the repository root after npm run build:

    PYTHONPATH=. python scripts/smoke_runtime.py
"""

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parent.parent
READY_TIMEOUT = 240
RECOVERY_TIMEOUT = 240


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def status(url: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()
    except Exception:
        return 0, ""


def wait_for(label: str, probe, timeout: float):
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        value = probe()
        if value:
            print(f"ok   {label} ({time.monotonic() - started:.1f}s)", flush=True)
            return value
        time.sleep(0.1)
    raise SystemExit(f"FAIL {label}: not observed within {timeout:.0f}s")


def children(launcher: int) -> dict[str, int]:
    try:
        pids = subprocess.run(["pgrep", "-P", str(launcher)], capture_output=True, text=True, check=True).stdout.split()
    except subprocess.CalledProcessError:
        return {}
    found = {}
    for pid in map(int, pids):
        try:
            with open(f"/proc/{pid}/stat") as handle:
                if handle.read().split()[2] == "Z":
                    continue
            command = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode()
        except OSError:
            continue
        found["api" if "uvicorn" in command else "web"] = pid
    return found


def main() -> int:
    if not (ROOT / "frontend/.next/BUILD_ID").exists():
        raise SystemExit("FAIL build the frontend first: npm --prefix frontend run build")
    web_port, api_port = free_port(), free_port()
    base = f"http://127.0.0.1:{web_port}"
    work = Path(tempfile.mkdtemp(prefix="b3lab-runtime-"))
    log_path = work / "launcher.log"
    environment = dict(
        os.environ,
        B3LAB_PORT=str(web_port),
        B3LAB_API_PORT=str(api_port),
        B3LAB_ORIGIN=base,
        B3LAB_DB=str(work / "smoke.db"),
        B3LAB_ENV="production",
        B3LAB_WARM_CACHES="0",
        NEXT_TELEMETRY_DISABLED="1",
    )
    log = log_path.open("w")
    launcher = subprocess.Popen([sys.executable, "run.py", "--production"], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_for("readiness reaches 200", lambda: status(f"{base}/health")[0] == 200, READY_TIMEOUT)
        pids = wait_for("both children are running", lambda: children(launcher.pid) if len(children(launcher.pid)) == 2 else None, 30)
        web = pids["web"]
        os.kill(pids["api"], signal.SIGKILL)
        print(f"killed api pid={pids['api']}", flush=True)

        wait_for(
            "readiness leaves 200 while the API is down",
            lambda: status(f"{base}/health") if status(f"{base}/health")[0] != 200 else None,
            10,
        )
        code, body = status(f"{base}/health")
        assert code == 503, f"expected 503 during the API outage, got {code}"
        assert json.loads(body)["status"] in {"starting", "degraded"}, body
        assert status(f"{base}/")[0] == 200, "the panel must keep serving during the API outage"
        print(f"ok   panel served while /health said {body}", flush=True)

        wait_for("readiness returns to 200", lambda: status(f"{base}/health")[0] == 200, RECOVERY_TIMEOUT)
        assert children(launcher.pid).get("web") == web, "the frontend process must not be replaced by an API kill"
        print("ok   frontend PID unchanged across the API restart", flush=True)
        return 0
    finally:
        launcher.send_signal(signal.SIGTERM)
        try:
            launcher.wait(timeout=40)
        except subprocess.TimeoutExpired:
            launcher.kill()
            launcher.wait()
        log.close()
        if launcher.returncode not in (0, -signal.SIGTERM):
            print(f"launcher exited {launcher.returncode}; tail of {log_path}:", flush=True)
            print("\n".join(log_path.read_text().splitlines()[-20:]), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
