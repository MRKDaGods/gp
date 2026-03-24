"""
Launch backend (FastAPI on port 8002) and frontend (Next.js on port 3000).
Run:  python start.py
Stop: Ctrl+C

Behaviour:
  - Any existing process on port 8002 or 3000 is killed first.
  - Backend is started and health-checked before the frontend boots.
  - Ctrl+C cleanly stops both servers.
"""

import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"

VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
BACKEND_PY = str(VENV_PY) if VENV_PY.exists() else sys.executable

BACKEND_PORT = 8002
FRONTEND_PORT = 3000
IS_WIN = os.name == "nt"
NPM_CMD = "npm.cmd" if IS_WIN else "npm"

# Running child processes (filled in main)
_procs: list[subprocess.Popen] = []


# ── Helpers ───────────────────────────────────────────────────────────────

def _pids_on_port(port: int) -> list[int]:
    """Return PIDs that are LISTENING on *port* (Windows netstat)."""
    pids: list[int] = []
    try:
        out = subprocess.check_output(
            ["netstat", "-ano"],
            text=True, errors="replace",
            stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            if f":{port} " in line and "LISTENING" in line:
                parts = line.strip().split()
                try:
                    pids.append(int(parts[-1]))
                except ValueError:
                    pass
    except Exception:
        pass
    return list(set(pids))


def _kill_port(port: int) -> None:
    """Kill every process listening on *port*."""
    pids = _pids_on_port(port)
    if not pids:
        return
    print(f"[launcher] Killing {len(pids)} existing process(es) on port {port}: {pids}", flush=True)
    for pid in pids:
        try:
            subprocess.call(
                ["taskkill", "/F", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass
    # Wait up to 5 s for the port to clear
    for _ in range(10):
        time.sleep(0.5)
        if not _pids_on_port(port):
            break


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) != 0


def _wait_http(url: str, timeout: int = 60) -> bool:
    """Poll *url* until it returns HTTP 200 or *timeout* seconds elapse."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status < 500:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _reset_frontend_cache() -> None:
    """Delete frontend build cache to prevent stale chunk load errors."""
    next_dir = FRONTEND_DIR / ".next"
    if not next_dir.exists():
        return
    print(f"[launcher] Removing stale Next.js cache: {next_dir}", flush=True)
    # Retry a few times in case files are briefly locked while old process exits.
    for attempt in range(1, 6):
        try:
            import shutil
            shutil.rmtree(next_dir)
            print("[launcher] Frontend cache removed.", flush=True)
            return
        except Exception as e:
            if attempt == 5:
                print(f"[launcher] WARNING: Could not remove .next cache: {e}", flush=True)
                return
            time.sleep(0.8)


def _stream(proc: subprocess.Popen, label: str) -> None:
    """Forward stdout + stderr of *proc* to the console with a [label] prefix."""
    def _drain(stream):
        try:
            for line in stream:
                print(f"[{label}] {line}", end="", flush=True)
        except Exception:
            pass

    if proc.stdout:
        threading.Thread(target=_drain, args=(proc.stdout,), daemon=True).start()
    if proc.stderr:
        threading.Thread(target=_drain, args=(proc.stderr,), daemon=True).start()


# ── Shutdown ──────────────────────────────────────────────────────────────

def _shutdown(signum=None, frame=None) -> None:
    print("\n[launcher] Shutting down…", flush=True)
    for p in _procs:
        try:
            p.terminate()
        except Exception:
            pass
    time.sleep(2)
    for p in _procs:
        try:
            if p.poll() is None:
                p.kill()
        except Exception:
            pass
    # Ensure ports are freed via taskkill too (belt-and-suspenders)
    for port in (BACKEND_PORT, FRONTEND_PORT):
        _kill_port(port)
    print("[launcher] All servers stopped.", flush=True)
    sys.exit(0)


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    # Windows only supports SIGINT reliably
    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, "SIGTERM"):
        try:
            signal.signal(signal.SIGTERM, _shutdown)
        except (OSError, ValueError):
            pass  # Not supported on this platform

    print("=" * 60, flush=True)
    print("  GP Pipeline Launcher")
    print(f"  Backend  →  http://localhost:{BACKEND_PORT}")
    print(f"  Frontend →  http://localhost:{FRONTEND_PORT}")
    print("  Press Ctrl+C to stop both servers")
    print("=" * 60, flush=True)

    # ── 1. Kill any lingering servers on our ports ────────────────────────
    print("\n[launcher] Checking for existing processes…", flush=True)
    _kill_port(BACKEND_PORT)
    _kill_port(FRONTEND_PORT)
    _reset_frontend_cache()

    # ── 2. Start backend ──────────────────────────────────────────────────
    print(f"\n[launcher] Starting backend ({BACKEND_PY})…", flush=True)

    # CREATE_NEW_PROCESS_GROUP keeps the child in its own console group on
    # Windows so our Ctrl+C handler (not Windows) decides when to stop it.
    extra_flags = subprocess.CREATE_NEW_PROCESS_GROUP if IS_WIN else 0

    backend = subprocess.Popen(
        [
            BACKEND_PY, "-m", "uvicorn",
            "backend_api:app",
            "--host", "0.0.0.0",
            "--port", str(BACKEND_PORT),
        ],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=extra_flags,
    )
    _procs.append(backend)
    _stream(backend, "backend")

    # ── 3. Wait for backend to be healthy ─────────────────────────────────
    health_url = f"http://localhost:{BACKEND_PORT}/api/health"
    print(f"[launcher] Waiting for backend health check ({health_url})…", flush=True)
    if not _wait_http(health_url, timeout=60):
        print("[launcher] ERROR: Backend did not become healthy within 60 s.", flush=True)
        _shutdown()
    print("[launcher] Backend is healthy ✓", flush=True)

    # ── 4. Start frontend ─────────────────────────────────────────────────
    print(f"\n[launcher] Starting frontend (npm run dev)…", flush=True)
    frontend = subprocess.Popen(
        [NPM_CMD, "run", "dev"],
        cwd=str(FRONTEND_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=extra_flags,
    )
    _procs.append(frontend)
    _stream(frontend, "frontend")

    print("\n[launcher] Both servers are running.", flush=True)
    print(f"  → Open http://localhost:{FRONTEND_PORT} in your browser\n", flush=True)

    # ── 5. Monitor: if either crashes, stop the other ─────────────────────
    while True:
        for p in _procs:
            if p.poll() is not None:
                print(
                    f"\n[launcher] A server exited unexpectedly (code {p.returncode})."
                    " Stopping everything.",
                    flush=True,
                )
                _shutdown()
        time.sleep(1)


if __name__ == "__main__":
    main()
