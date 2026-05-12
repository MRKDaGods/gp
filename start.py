"""
GP Pipeline Launcher — starts backend (port 8004) and frontend (port 3001).

Usage:  python start.py
Stop:   Ctrl+C
"""

import os
import signal
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT          = Path(__file__).resolve().parent
FRONTEND_DIR  = ROOT / "frontend"
VENV_PY       = ROOT / ".venv" / "Scripts" / "python.exe"
BACKEND_PY    = str(VENV_PY) if VENV_PY.exists() else sys.executable
BACKEND_PORT  = 8004
FRONTEND_PORT = 3001
IS_WIN        = os.name == "nt"
NPM_CMD       = "npm.cmd" if IS_WIN else "npm"

_procs: list[subprocess.Popen] = []


# ── Port management ───────────────────────────────────────────────────────

def _pids_on_port(port: int) -> list[int]:
    pids: list[int] = []
    try:
        out = subprocess.check_output(
            ["netstat", "-ano"], text=True, errors="replace",
            stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            if f":{port} " in line and "LISTENING" in line:
                try:
                    pids.append(int(line.strip().split()[-1]))
                except ValueError:
                    pass
    except Exception:
        pass
    return list(set(pids))


def _kill_port(port: int) -> bool:
    """Kill every process listening on *port*. Returns True when port is free."""
    deadline = time.time() + 15
    while time.time() < deadline:
        pids = _pids_on_port(port)
        if not pids:
            return True
        print(f"[launcher] Killing PID(s) on port {port}: {pids}", flush=True)
        for p in pids:
            subprocess.call(
                ["taskkill", "/F", "/T", "/PID", str(p)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        time.sleep(0.8)
    remaining = _pids_on_port(port)
    if remaining:
        print(
            f"[launcher] ERROR: port {port} still occupied (PIDs: {remaining}).\n"
            f"[launcher] Open Task Manager → end all python.exe processes, then retry.",
            flush=True,
        )
        return False
    return True



def _wait_http(url: str, timeout: int) -> bool:
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


def _clear_next_cache() -> None:
    next_dir = FRONTEND_DIR / ".next"
    if not next_dir.exists():
        return
    print("[launcher] Clearing stale Next.js cache…", flush=True)
    for _ in range(5):
        try:
            shutil.rmtree(next_dir)
            return
        except Exception:
            time.sleep(0.8)


def _stream(proc: subprocess.Popen, label: str) -> None:
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
        try: p.terminate()
        except Exception: pass
    time.sleep(2)
    for p in _procs:
        try:
            if p.poll() is None: p.kill()
        except Exception: pass
    _kill_port(BACKEND_PORT)
    _kill_port(FRONTEND_PORT)
    print("[launcher] Done.", flush=True)
    sys.exit(0)


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, "SIGTERM"):
        try: signal.signal(signal.SIGTERM, _shutdown)
        except (OSError, ValueError): pass

    print("=" * 52, flush=True)
    print("  GP Pipeline Launcher")
    print(f"  Backend  → http://localhost:{BACKEND_PORT}")
    print(f"  Frontend → http://localhost:{FRONTEND_PORT}")
    print("  Ctrl+C to stop")
    print("=" * 52, flush=True)

    # ── 1. Free ports ─────────────────────────────────────────────────────
    print("\n[launcher] Freeing ports…", flush=True)
    if not _kill_port(BACKEND_PORT):
        sys.exit(1)
    if not _kill_port(FRONTEND_PORT):
        sys.exit(1)
    _clear_next_cache()

    # ── 2. Start backend ──────────────────────────────────────────────────
    print(f"\n[launcher] Starting backend…", flush=True)
    backend = subprocess.Popen(
        [BACKEND_PY, "-m", "uvicorn", "backend_api:app",
         "--host", "0.0.0.0", "--port", str(BACKEND_PORT)],
        cwd=str(ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    _procs.append(backend)
    _stream(backend, "backend")

    if not _wait_http(f"http://localhost:{BACKEND_PORT}/api/health", timeout=60):
        print("[launcher] ERROR: Backend did not become healthy within 60s.", flush=True)
        _shutdown()
    print("[launcher] Backend ready ✓", flush=True)

    # ── 3. Start frontend ─────────────────────────────────────────────────
    print(f"\n[launcher] Starting frontend…", flush=True)
    frontend = subprocess.Popen(
        [NPM_CMD, "run", "dev", "--", "--port", str(FRONTEND_PORT)],
        cwd=str(FRONTEND_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    _procs.append(frontend)
    _stream(frontend, "frontend")

    if not _wait_http(f"http://localhost:{FRONTEND_PORT}", timeout=90):
        print("[launcher] ERROR: Frontend did not start within 90s.", flush=True)
        _shutdown()
    print("[launcher] Frontend ready ✓", flush=True)

    print(f"\n[launcher] Open http://localhost:{FRONTEND_PORT} in your browser\n", flush=True)

    # ── 4. Monitor: shut down if either server exits unexpectedly ─────────
    while True:
        for p in list(_procs):
            if p.poll() is not None:
                print(f"\n[launcher] Server exited (code {p.returncode}). Stopping.", flush=True)
                _shutdown()
        time.sleep(2)


if __name__ == "__main__":
    main()
