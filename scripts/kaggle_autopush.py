"""Automated Kaggle pipeline monitor and pusher.

Monitors running/queued Kaggle kernels and automatically pushes the next
stage when each upstream stage completes.

Pipeline (Cycle 1 - existing runs):
  10b v6 (QUEUED)  ->  complete  ->  push 10c v9  (144-combo scan w/ HOTA fix)
  10a v3 (QUEUED)  ->  complete  ->  push 10b v7
  10b v7           ->  complete  ->  push 10c v10  (min=5 tracklets + 144-combo scan)

Pipeline (Cycle 2 - confidence=0.60 detection improvement):
  10c v10 done     ->  push 10a v4  (confidence 0.50->0.60)
  10a v4           ->  complete  ->  push 10b v8
  10b v8           ->  complete  ->  push 10c v11  (144-combo scan)
  10c v11          ->  complete  ->  download logs + report best params

Usage:
    python scripts/kaggle_autopush.py [--interval 60]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Kernel slugs ────────────────────────────────────────────────────────────
KERNEL_10A = "mrkdagods/mtmc-10a-stages-0-2-tracking-reid-features"
KERNEL_10B = "mrkdagods/mtmc-10b-stage-3-faiss-indexing"
KERNEL_10C = "mrkdagods/mtmc-10c-stages-4-5-association-eval"

NOTEBOOK_DIR_10A = Path("notebooks/kaggle/10a_stages012")
NOTEBOOK_DIR_10B = Path("notebooks/kaggle/10b_stage3")
NOTEBOOK_DIR_10C = Path("notebooks/kaggle/10c_stages45")

LOG_DIR = Path("data/outputs")

# ── State tracking ───────────────────────────────────────────────────────────
# Format: {"kernel_slug": "version_tag"}
pushed: dict[str, str] = {}


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def kaggle_status(kernel: str) -> str:
    """Returns raw status string from kaggle CLI."""
    r = subprocess.run(
        ["kaggle", "kernels", "status", kernel],
        capture_output=True, text=True,
    )
    out = r.stdout + r.stderr
    if "COMPLETE" in out:
        return "COMPLETE"
    if "RUNNING" in out:
        return "RUNNING"
    if "QUEUED" in out:
        return "QUEUED"
    if "ERROR" in out or "CANCEL" in out:
        return "ERROR"
    return "UNKNOWN"


def kaggle_push(notebook_dir: Path, label: str) -> bool:
    """Push a Kaggle notebook. Returns True on success."""
    log(f"  Pushing {label} ...")
    r = subprocess.run(
        ["kaggle", "kernels", "push"],
        cwd=str(notebook_dir),
        capture_output=True, text=True,
    )
    out = r.stdout + r.stderr
    if "successfully pushed" in out.lower():
        log(f"  OK {label} pushed!")
        return True
    elif "maximum batch gpu session" in out.lower():
        log(f"  ✗ {label}: max GPU sessions reached, will retry")
        return False
    else:
        log(f"  ✗ {label} push failed: {out.strip()[:200]}")
        return False


def download_logs(kernel: str, label: str) -> None:
    """Download the most recent kernel logs."""
    outfile = LOG_DIR / f"{label}_logs.txt"
    r = subprocess.run(
        [sys.executable, "scripts/kaggle_logs.py", kernel, "--out", str(outfile)],
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        log(f"  OK Logs saved to {outfile}")
        _print_key_metrics(outfile)
    else:
        log(f"  ✗ Log download failed: {r.stderr[:200]}")


def _print_key_metrics(logfile: Path) -> None:
    """Extract and print key metric lines from a log file."""
    import json
    lines_out = []
    try:
        with open(logfile, encoding="utf-8", errors="replace") as f:
            raw = f.read()
        for line in raw.splitlines():
            line = line.strip().lstrip(",")
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "data" in obj:
                    lines_out.append(obj["data"])
            except Exception:
                pass
    except Exception:
        return

    text = "\n".join(lines_out)
    for line in text.splitlines():
        lo = line.lower()
        if any(k in lo for k in ["idf1", "mota", "hota", "best", "scan result", "[mtmc]", "complete"]):
            print(f"    > {line.strip()}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=60,
                        help="Poll interval in seconds (default: 60)")
    parser.add_argument("--max-hours", type=float, default=10.0,
                        help="Max run time in hours before exiting")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    deadline = time.time() + args.max_hours * 3600
    log(f"Autopush monitor started (interval={args.interval}s, max={args.max_hours}h)")

    # State tracking using "transition count" pattern:
    # Each time a kernel status transitions INTO "COMPLETE" that's a new completion event.
    # v10a: QUEUED->RUNNING->COMPLETE = v3 done
    # v10b: first COMPLETE = v6 done; second COMPLETE (after we push v7) = v7 done
    # v10c: first COMPLETE = v9 done; second COMPLETE (after we push v10) = v10 done

    prev_10a = "UNKNOWN"
    prev_10b = "UNKNOWN"
    prev_10c = "UNKNOWN"

    # Completion events (count) — tracks transitions INTO "COMPLETE".
    # NOTE: 10c v8 is COMPLETE at startup, so done_10c starts at 0 but will
    # immediately increment to 1 on the first poll. Thresholds:
    #   done_10a: 1=v3  done_10b: 1=v6, 2=v7
    #   done_10c: 1=v8(startup), 2=v9, 3=v10
    done_10a = 0
    done_10b = 0
    done_10c = 0

    # Push flags (prevent double-pushing)
    pushed_10c_v9 = False
    pushed_10b_v7 = False
    pushed_10c_v10 = False
    # Cycle 2: confidence=0.60 detection improvement pipeline
    pushed_10a_v4 = False
    pushed_10b_v8 = False
    pushed_10c_v11 = False

    # Log download flags (prevent double-downloading)
    logged_10c_v9 = False
    logged_10c_v10 = False
    logged_10a_v4 = False
    logged_10c_v11 = False

    # After pushing each job, the kernel status briefly stays COMPLETE until the
    # new version is queued. Use these flags to "arm" for the next completion.
    armed_10b_v7 = False   # True once 10b transitions away from COMPLETE (v7 started)
    armed_10c_v9 = False   # True once 10c transitions away from COMPLETE (v9 started)
    armed_10c_v10 = False  # True once 10c transitions away from COMPLETE (v10 started)
    # Cycle 2 arms
    armed_10a_v4 = False   # True once 10a transitions away from COMPLETE (v4 started)
    armed_10b_v8 = False   # True once 10b transitions away from COMPLETE (v8 started)
    armed_10c_v11 = False  # True once 10c transitions away from COMPLETE (v11 started)

    try:
        while time.time() < deadline:
            status_10a = kaggle_status(KERNEL_10A)
            status_10b = kaggle_status(KERNEL_10B)
            status_10c = kaggle_status(KERNEL_10C)

            log(f"Status: 10a={status_10a}  10b={status_10b}  10c={status_10c}")

            # ── Detect new COMPLETE transitions ───────────────────────────────
            if status_10a == "COMPLETE" and prev_10a != "COMPLETE":
                done_10a += 1
                log(f"-> 10a completion #{done_10a} detected!")

            if status_10b == "COMPLETE" and prev_10b != "COMPLETE":
                done_10b += 1
                log(f"-> 10b completion #{done_10b} detected!")

            if status_10c == "COMPLETE" and prev_10c != "COMPLETE":
                done_10c += 1
                log(f"-> 10c completion #{done_10c} detected!")

            # ── Arm flags: kernel transitioned away from COMPLETE (new run started)
            if pushed_10b_v7 and prev_10b == "COMPLETE" and status_10b != "COMPLETE":
                armed_10b_v7 = True
                log("  10b v7 now running/queued (armed for completion)")
            if pushed_10c_v9 and prev_10c == "COMPLETE" and status_10c != "COMPLETE":
                armed_10c_v9 = True
                log("  10c v9 now running/queued (armed for completion)")
            if pushed_10c_v10 and prev_10c == "COMPLETE" and status_10c != "COMPLETE":
                armed_10c_v10 = True
                log("  10c v10 now running/queued (armed for completion)")
            # Cycle 2 arm detection
            if pushed_10a_v4 and prev_10a == "COMPLETE" and status_10a != "COMPLETE":
                armed_10a_v4 = True
                log("  10a v4 now running/queued (armed for completion)")
            if pushed_10b_v8 and prev_10b == "COMPLETE" and status_10b != "COMPLETE":
                armed_10b_v8 = True
                log("  10b v8 now running/queued (armed for completion)")
            if pushed_10c_v11 and prev_10c == "COMPLETE" and status_10c != "COMPLETE":
                armed_10c_v11 = True
                log("  10c v11 now running/queued (armed for completion)")

            # ── 10b v6 COMPLETE -> push 10c v9 ────────────────────────────────
            if done_10b >= 1 and not pushed_10c_v9:
                log("  <- 10b v6 done. Downloading logs ...")
                download_logs(KERNEL_10B, "10b_v6")
                log("  -> Pushing 10c v9 (108-combo scan) ...")
                ok = kaggle_push(NOTEBOOK_DIR_10C, "10c v9")
                if ok:
                    pushed_10c_v9 = True

            # ── 10a v3 COMPLETE -> download logs + push 10b v7 ────────────────
            if done_10a >= 1 and not pushed_10b_v7:
                log("  <- 10a v3 done. Downloading logs ...")
                download_logs(KERNEL_10A, "10a_v3")
                log("  -> Pushing 10b v7 ...")
                ok = kaggle_push(NOTEBOOK_DIR_10B, "10b v7")
                if ok:
                    pushed_10b_v7 = True

            # ── 10b v7 COMPLETE -> push 10c v10 ────────────────────────────────
            # Detect via "armed" flag + new COMPLETE event
            if pushed_10b_v7:
                if armed_10b_v7 and status_10b == "COMPLETE" and not pushed_10c_v10:
                    # v7 completion detected via arm+complete cycle
                    log("  <- 10b v7 done. Downloading logs ...")
                    download_logs(KERNEL_10B, "10b_v7")
                    log("  -> Pushing 10c v10 (min=5 tracklets scan) ...")
                    ok = kaggle_push(NOTEBOOK_DIR_10C, "10c v10")
                    if ok:
                        pushed_10c_v10 = True
                elif not armed_10b_v7 and done_10b >= 2 and not pushed_10c_v10:
                    # Fallback: detected via completion count (if we missed the transition)
                    log("  <- 10b v7 done (via count). Downloading logs ...")
                    download_logs(KERNEL_10B, "10b_v7")
                    log("  -> Pushing 10c v10 (min=5 tracklets scan) ...")
                    ok = kaggle_push(NOTEBOOK_DIR_10C, "10c v10")
                    if ok:
                        pushed_10c_v10 = True

            # ── 10c v9 COMPLETE -> download results (done_10c reaches 2) ──────
            if pushed_10c_v9 and not pushed_10c_v10 and not logged_10c_v9:
                v9_done = (armed_10c_v9 and status_10c == "COMPLETE") or (done_10c >= 2 and not armed_10c_v9)
                if v9_done:
                    log("  <- 10c v9 done! Downloading scan results ...")
                    download_logs(KERNEL_10C, "10c_v9")
                    logged_10c_v9 = True

            # ── 10c v10 COMPLETE -> download results + push 10a v4 (cycle 2) ──
            if pushed_10c_v10 and not logged_10c_v10:
                v10_done = (armed_10c_v10 and status_10c == "COMPLETE") or (done_10c >= 3 and not armed_10c_v10)
                if v10_done:
                    log("  <- 10c v10 done! Downloading scan results ...")
                    download_logs(KERNEL_10C, "10c_v10")
                    logged_10c_v10 = True
                    # Start Cycle 2: push 10a v4 with confidence=0.60 detection
                    if not pushed_10a_v4:
                        log("  -> Starting Cycle 2: pushing 10a v4 (confidence=0.60 detection) ...")
                        ok = kaggle_push(NOTEBOOK_DIR_10A, "10a v4")
                        if ok:
                            pushed_10a_v4 = True

            # ── Cycle 2: 10a v4 COMPLETE -> push 10b v8 ─────────────────────
            if pushed_10a_v4 and not pushed_10b_v8:
                v4_done = (armed_10a_v4 and status_10a == "COMPLETE") or (done_10a >= 2 and not armed_10a_v4)
                if v4_done:
                    if not logged_10a_v4:
                        log("  <- 10a v4 done! Downloading logs ...")
                        download_logs(KERNEL_10A, "10a_v4")
                        logged_10a_v4 = True
                    log("  -> Pushing 10b v8 (FAISS index of 10a v4 tracklets) ...")
                    ok = kaggle_push(NOTEBOOK_DIR_10B, "10b v8")
                    if ok:
                        pushed_10b_v8 = True

            # ── Cycle 2: 10b v8 COMPLETE -> push 10c v11 ─────────────────────
            if pushed_10b_v8 and not pushed_10c_v11:
                v8_done = (armed_10b_v8 and status_10b == "COMPLETE") or (done_10b >= 3 and not armed_10b_v8)
                if v8_done:
                    log("  <- 10b v8 done. Downloading logs ...")
                    download_logs(KERNEL_10B, "10b_v8")
                    log("  -> Pushing 10c v11 (confidence=0.60 tracklets + 144-combo scan) ...")
                    ok = kaggle_push(NOTEBOOK_DIR_10C, "10c v11")
                    if ok:
                        pushed_10c_v11 = True

            # ── Cycle 2: 10c v11 COMPLETE -> download results + exit ──────────
            if pushed_10c_v11 and not logged_10c_v11:
                v11_done = (armed_10c_v11 and status_10c == "COMPLETE") or (done_10c >= 4 and not armed_10c_v11)
                if v11_done:
                    log("  <- 10c v11 done! Downloading scan results ...")
                    download_logs(KERNEL_10C, "10c_v11")
                    logged_10c_v11 = True
                    log("All two pipeline cycles complete! Exiting.")
                    break

            # ── All done check ────────────────────────────────────────────────
            if (pushed_10c_v9 and pushed_10b_v7 and pushed_10c_v10
                    and logged_10c_v9 and logged_10c_v10
                    and pushed_10a_v4 and pushed_10b_v8 and pushed_10c_v11
                    and logged_10c_v11):
                log("All pipeline cycles pushed and completed. Exiting.")
                break

            prev_10a, prev_10b, prev_10c = status_10a, status_10b, status_10c
            time.sleep(args.interval)

    except KeyboardInterrupt:
        log("Interrupted by user.")

    log("Autopush monitor finished.")


if __name__ == "__main__":
    main()

