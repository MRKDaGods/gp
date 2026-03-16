"""Auto-chain monitor for the 3-notebook MTMC pipeline.

Polls 10a → once complete, pushes 10b → polls 10b → pushes 10c → polls 10c → shows results.

Usage:
    python scripts/kaggle_chain.py              # start from monitoring 10a (already running)
    python scripts/kaggle_chain.py --from 10b   # skip 10a, start by pushing 10b
    python scripts/kaggle_chain.py --from 10c   # skip 10a+10b, just push 10c

Options:
    --owner OWNER        Kaggle username (default: mrkdagods)
    --poll-interval N    Seconds between status polls (default: 60)
    --logs               Fetch and print execution log when each kernel finishes
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent

OWNER = "mrkdagods"
SLUG_10A = "mtmc-10a-stages-0-2-tracking-reid-features"
SLUG_10B = "mtmc-10b-stage-3-faiss-indexing"
SLUG_10C = "mtmc-10c-stages-4-5-association-eval"

NB_DIRS = {
    "10a": ROOT / "notebooks" / "kaggle" / "10a_stages012",
    "10b": ROOT / "notebooks" / "kaggle" / "10b_stage3",
    "10c": ROOT / "notebooks" / "kaggle" / "10c_stages45",
}
SLUGS = {
    "10a": SLUG_10A,
    "10b": SLUG_10B,
    "10c": SLUG_10C,
}


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def kaggle(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["kaggle", *args], capture_output=True, text=True)


def get_status(slug: str) -> str:
    r = kaggle("kernels", "status", f"{OWNER}/{slug}")
    out = (r.stdout + r.stderr).strip()
    for word in ["RUNNING", "QUEUED", "COMPLETE", "ERROR", "CANCEL"]:
        if word in out.upper():
            return word
    return out


def push(notebook_id: str):
    nb_dir = NB_DIRS[notebook_id]
    print(f"\n[{ts()}] Pushing {notebook_id} ({nb_dir.name}) ...")
    r = subprocess.run(["kaggle", "kernels", "push", "-p", str(nb_dir)],
                       capture_output=False)
    if r.returncode != 0:
        print(f"[{ts()}] ERROR: push failed for {notebook_id}")
        sys.exit(1)


def wait_for(notebook_id: str, poll: int) -> bool:
    """Poll until kernel reaches terminal state. Returns True on COMPLETE."""
    slug = SLUGS[notebook_id]
    print(f"[{ts()}] Monitoring {notebook_id} ({OWNER}/{slug}) ...")
    dots = 0
    while True:
        status = get_status(slug)
        if status in ("COMPLETE", "SUCCESS"):
            print(f"\n[{ts()}] ✓ {notebook_id} COMPLETE")
            return True
        if status in ("ERROR", "CANCEL", "CANCELLED"):
            print(f"\n[{ts()}] ✗ {notebook_id} FAILED with status: {status}")
            print(f"  Fetch logs: python scripts/kaggle_logs.py {OWNER}/{slug} --tail 100")
            return False
        dots += 1
        elapsed_min = dots * poll / 60
        print(f"[{ts()}] {notebook_id} {status} ({elapsed_min:.0f} min elapsed) ...",
              flush=True)
        time.sleep(poll)


def fetch_logs(notebook_id: str, tail: int = 80):
    slug = SLUGS[notebook_id]
    log_file = ROOT / "data" / "outputs" / f"{notebook_id}_logs.txt"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{ts()}] Fetching logs for {notebook_id} -> {log_file}")
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "kaggle_logs.py"),
         f"{OWNER}/{slug}", "--tail", str(tail), "--out", str(log_file)],
        capture_output=False,
    )
    if log_file.exists():
        print(f"\n{'=' * 70}")
        print(log_file.read_text(encoding="utf-8", errors="replace"))
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="start_from", default="10a",
                        choices=["10a", "10b", "10c"],
                        help="Which kernel to start from")
    parser.add_argument("--owner", default=OWNER)
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--logs", action="store_true",
                        help="Fetch logs when each kernel finishes")
    args = parser.parse_args()

    poll = args.poll_interval
    stages = ["10a", "10b", "10c"]
    start_idx = stages.index(args.start_from)

    print(f"MTMC auto-chain starting from {args.start_from}")
    print(f"Poll interval: {poll}s")
    print(f"Chain: 10a ({SLUG_10A})")
    print(f"    -> 10b ({SLUG_10B})")
    print(f"    -> 10c ({SLUG_10C})")
    print()

    # --- 10a ---
    if start_idx <= 0:
        # 10a was already pushed; just monitor it
        ok = wait_for("10a", poll)
        if args.logs:
            fetch_logs("10a")
        if not ok:
            print("Aborting chain — fix 10a errors first.")
            sys.exit(1)

    # --- 10b ---
    if start_idx <= 1:
        push("10b")
        time.sleep(10)  # give Kaggle a moment to register the new version
        ok = wait_for("10b", poll)
        if args.logs:
            fetch_logs("10b")
        if not ok:
            print("Aborting chain — fix 10b errors first.")
            sys.exit(1)

    # --- 10c ---
    push("10c")
    time.sleep(10)
    ok = wait_for("10c", poll)
    if args.logs:
        fetch_logs("10c", tail=150)
    if not ok:
        print("10c failed — fetch logs above for details.")
        sys.exit(1)

    print(f"\n[{ts()}] ✓ Full pipeline complete!")
    print(f"  To re-run 10c with different params:")
    print(f"    edit notebooks/kaggle/10c_stages45/{SLUG_10C}.ipynb (tuning cell)")
    print(f"    then: python scripts/kaggle_chain.py --from 10c")


if __name__ == "__main__":
    main()
