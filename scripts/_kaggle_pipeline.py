"""Monitor Kaggle notebook pipeline: wait for 10a → push 10b → wait → push 10c.

Usage: python scripts/_kaggle_pipeline.py [--start-from 10a|10b|10c]
"""
import subprocess, sys, time, argparse
from pathlib import Path

PROJECT = Path(__file__).parent.parent

KERNELS = {
    "10a": "mrkdagods/mtmc-10a-stages-0-2-tracking-reid-features",
    "10b": "mrkdagods/mtmc-10b-stage-3-faiss-indexing",
    "10c": "mrkdagods/mtmc-10c-stages-4-5-association-eval",
}

PATHS = {
    "10a": "notebooks/kaggle/10a_stages012",
    "10b": "notebooks/kaggle/10b_stage3",
    "10c": "notebooks/kaggle/10c_stages45",
}


def kaggle_status(slug: str) -> str:
    result = subprocess.run(
        ["kaggle", "kernels", "status", slug],
        capture_output=True, text=True
    )
    out = result.stdout + result.stderr
    for keyword in ("COMPLETE", "ERROR", "RUNNING", "QUEUED", "CANCEL"):
        if keyword in out:
            return keyword
    return "UNKNOWN"


def kaggle_push(notebook: str):
    path = PROJECT / PATHS[notebook]
    print(f"  Pushing {notebook} ({PATHS[notebook]})...")
    result = subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(path)],
        capture_output=True, text=True
    )
    print(f"  {result.stdout.strip()}")
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}")
        return False
    return True


def wait_for_completion(notebook: str, poll_interval: int = 60, timeout_minutes: int = 180):
    slug = KERNELS[notebook]
    deadline = time.time() + timeout_minutes * 60
    print(f"\nWaiting for {notebook} ({slug}) to complete...")
    
    while time.time() < deadline:
        status = kaggle_status(slug)
        now = time.strftime("%H:%M:%S")
        print(f"  [{now}] {notebook} status: {status}", flush=True)
        
        if status == "COMPLETE":
            return True
        elif status == "ERROR":
            print(f"  ERROR: {notebook} failed!")
            return False
        elif status in ("CANCEL", "UNKNOWN"):
            print(f"  {notebook} status is {status} — stopping wait.")
            return False
        
        time.sleep(poll_interval)
    
    print(f"  Timeout after {timeout_minutes} minutes!")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-from", choices=["10a", "10b", "10c"], default="10a",
                        help="Which stage to start monitoring from (10a is already running)")
    parser.add_argument("--poll", type=int, default=60, help="Poll interval in seconds")
    args = parser.parse_args()
    
    pipeline = ["10a", "10b", "10c"]
    start_idx = pipeline.index(args.start_from)
    
    for i, nb in enumerate(pipeline[start_idx:]):
        actual_idx = start_idx + i
        
        if actual_idx == 0 or (actual_idx > 0 and nb == args.start_from and i == 0):
            # This notebook is assumed to already be running — just monitor it
            print(f"\n=== Waiting for {nb} (already running) ===")
        else:
            # Push this notebook (previous one just completed)
            print(f"\n=== Pushing {nb} ===")
            if not kaggle_push(nb):
                print(f"Failed to push {nb}. Stopping.")
                sys.exit(1)
            time.sleep(30)  # Give Kaggle time to queue the run
        
        success = wait_for_completion(nb, poll_interval=args.poll)
        if not success:
            print(f"\n{nb} did not complete successfully. Stopping pipeline.")
            sys.exit(1)
        print(f"\n{nb} COMPLETE!")
    
    print("\n=== Pipeline complete! All stages done. ===")


if __name__ == "__main__":
    main()
