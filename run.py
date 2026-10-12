import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import urllib.request


ROOT = Path(__file__).resolve().parent
KEEPALIVE_SECONDS = 600


def keep_alive(url, stopping):
    # O plano gratuito do Render suspende o serviço após 15 min sem tráfego externo.
    while not stopping.wait(KEEPALIVE_SECONDS):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                response.read(1)
        except Exception as error:
            print(f"Keep-alive falhou: {error}", flush=True)


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
    if not (ROOT / "frontend/node_modules/next").exists():
        parser.error("Execute npm ci no diretório frontend antes de iniciar.")
    stopping = threading.Event()
    for event in (signal.SIGTERM, signal.SIGINT):
        signal.signal(event, lambda *_: stopping.set())
    processes = []
    try:
        processes.append(subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", api_port, "--workers", "1"], cwd=ROOT, env=environment, start_new_session=True))
        frontend_args = ["--port", port]
        if host := os.environ.get("B3LAB_HOST"):
            frontend_args += ["--hostname", host]
        processes.append(subprocess.Popen(["npm", "run", "start" if args.production else "dev", "--", *frontend_args], cwd=ROOT / "frontend", env=environment, start_new_session=True))
        print(f"B3Lab em http://localhost:{port}. API privada em 127.0.0.1:{api_port}.", flush=True)
        if keepalive_url := os.environ.get("B3LAB_KEEPALIVE_URL"):
            threading.Thread(target=keep_alive, args=(keepalive_url, stopping), daemon=True).start()
        while not stopping.wait(.25):
            failed = next((process for process in processes if process.poll() is not None), None)
            if failed is not None:
                return failed.returncode or 1
        return 0
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
