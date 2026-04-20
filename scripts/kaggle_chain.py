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
import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent

DEFAULT_OWNER = "mrkdagods"
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
CHAIN_SLUGS = frozenset(SLUGS.values())


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def kaggle(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["kaggle", *args], capture_output=True, text=True)


def build_kernel_ref(owner: str, *, notebook_id: str | None = None, slug: str | None = None) -> str:
    if (notebook_id is None) == (slug is None):
        raise ValueError("Provide exactly one of notebook_id or slug")

    kernel_slug = SLUGS[notebook_id] if notebook_id is not None else slug
    return f"{owner}/{kernel_slug}"


def apply_owner_to_metadata(metadata: dict, notebook_id: str, owner: str) -> tuple[dict, bool]:
    updated_metadata = dict(metadata)
    changed = False

    desired_kernel_ref = build_kernel_ref(owner, notebook_id=notebook_id)
    if updated_metadata.get("id") != desired_kernel_ref:
        updated_metadata["id"] = desired_kernel_ref
        changed = True

    kernel_sources = list(updated_metadata.get("kernel_sources", []))
    rewritten_sources = []
    for source in kernel_sources:
        source_slug = source.split("/", 1)[1] if "/" in source else source
        if source_slug in CHAIN_SLUGS:
            rewritten_sources.append(build_kernel_ref(owner, slug=source_slug))
        else:
            rewritten_sources.append(source)

    if rewritten_sources != kernel_sources:
        updated_metadata["kernel_sources"] = rewritten_sources
        changed = True

    return updated_metadata, changed


def prepare_push_dir(notebook_id: str, owner: str) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    nb_dir = NB_DIRS[notebook_id]
    metadata_path = nb_dir / "kernel-metadata.json"

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    updated_metadata, changed = apply_owner_to_metadata(metadata, notebook_id, owner)
    if not changed:
        return nb_dir, None

    temp_dir = tempfile.TemporaryDirectory(prefix=f"kaggle_chain_{notebook_id}_")
    temp_nb_dir = Path(temp_dir.name) / nb_dir.name
    shutil.copytree(nb_dir, temp_nb_dir)

    with (temp_nb_dir / "kernel-metadata.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(updated_metadata, handle, ensure_ascii=True, indent=2)
        handle.write("\n")

    return temp_nb_dir, temp_dir


def get_status(owner: str, notebook_id: str) -> str:
    r = kaggle("kernels", "status", build_kernel_ref(owner, notebook_id=notebook_id))
    out = (r.stdout + r.stderr).strip()
    for word in ["RUNNING", "QUEUED", "COMPLETE", "ERROR", "CANCEL"]:
        if word in out.upper():
            return word
    return out


def push(notebook_id: str, owner: str):
    nb_dir = NB_DIRS[notebook_id]
    push_dir, temp_dir = prepare_push_dir(notebook_id, owner)
    print(f"\n[{ts()}] Pushing {notebook_id} ({nb_dir.name}) as {build_kernel_ref(owner, notebook_id=notebook_id)} ...")
    try:
        r = subprocess.run(["kaggle", "kernels", "push", "-p", str(push_dir)],
                           capture_output=False)
        if r.returncode != 0:
            print(f"[{ts()}] ERROR: push failed for {notebook_id}")
            sys.exit(1)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def wait_for(notebook_id: str, poll: int, owner: str) -> bool:
    """Poll until kernel reaches terminal state. Returns True on COMPLETE."""
    kernel_ref = build_kernel_ref(owner, notebook_id=notebook_id)
    print(f"[{ts()}] Monitoring {notebook_id} ({kernel_ref}) ...")
    dots = 0
    while True:
        status = get_status(owner, notebook_id)
        if status in ("COMPLETE", "SUCCESS"):
            print(f"\n[{ts()}] ✓ {notebook_id} COMPLETE")
            return True
        if status in ("ERROR", "CANCEL", "CANCELLED"):
            print(f"\n[{ts()}] ✗ {notebook_id} FAILED with status: {status}")
            print(f"  Fetch logs: python scripts/kaggle_logs.py {kernel_ref} --tail 100")
            return False
        dots += 1
        elapsed_min = dots * poll / 60
        print(f"[{ts()}] {notebook_id} {status} ({elapsed_min:.0f} min elapsed) ...",
              flush=True)
        time.sleep(poll)


def fetch_logs(notebook_id: str, owner: str, tail: int = 80):
    log_file = ROOT / "data" / "outputs" / f"{notebook_id}_logs.txt"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{ts()}] Fetching logs for {notebook_id} ({build_kernel_ref(owner, notebook_id=notebook_id)}) -> {log_file}")
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "kaggle_logs.py"),
         build_kernel_ref(owner, notebook_id=notebook_id), "--tail", str(tail), "--out", str(log_file)],
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
    parser.add_argument("--owner", default=DEFAULT_OWNER)
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--logs", action="store_true",
                        help="Fetch logs when each kernel finishes")
    args = parser.parse_args()

    owner = args.owner
    poll = args.poll_interval
    stages = ["10a", "10b", "10c"]
    start_idx = stages.index(args.start_from)

    print(f"MTMC auto-chain starting from {args.start_from}")
    print(f"Owner: {owner}")
    print(f"Poll interval: {poll}s")
    print(f"Chain: 10a ({build_kernel_ref(owner, notebook_id='10a')})")
    print(f"    -> 10b ({build_kernel_ref(owner, notebook_id='10b')})")
    print(f"    -> 10c ({build_kernel_ref(owner, notebook_id='10c')})")
    print()

    # --- 10a ---
    if start_idx <= 0:
        # 10a was already pushed; just monitor it
        ok = wait_for("10a", poll, owner)
        if args.logs:
            fetch_logs("10a", owner)
        if not ok:
            print("Aborting chain — fix 10a errors first.")
            sys.exit(1)

    # --- 10b ---
    if start_idx <= 1:
        push("10b", owner)
        time.sleep(10)  # give Kaggle a moment to register the new version
        ok = wait_for("10b", poll, owner)
        if args.logs:
            fetch_logs("10b", owner)
        if not ok:
            print("Aborting chain — fix 10b errors first.")
            sys.exit(1)

    # --- 10c ---
    push("10c", owner)
    time.sleep(10)
    ok = wait_for("10c", poll, owner)
    if args.logs:
        fetch_logs("10c", owner, tail=150)
    if not ok:
        print("10c failed — fetch logs above for details.")
        sys.exit(1)

    print(f"\n[{ts()}] ✓ Full pipeline complete!")
    print(f"  To re-run 10c with different params:")
    print(f"    edit notebooks/kaggle/10c_stages45/{SLUG_10C}.ipynb (tuning cell)")
    print(f"    then: python scripts/kaggle_chain.py --owner {owner} --from 10c")


if __name__ == "__main__":
    main()
