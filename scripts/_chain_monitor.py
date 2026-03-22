"""
v72 Full Chain Monitor: 10a → 10b → 10c
Polls each stage, pushes next when complete, prints final results.
"""
import subprocess, time, sys, os

slug_10a = "gumfreddy/mtmc-10a-stages-0-2-tracking-reid-features"
slug_10b = "gumfreddy/mtmc-10b-stage-3-faiss-indexing"
slug_10c = "gumfreddy/mtmc-10c-stages-4-5-association-eval"

def poll_until_done(slug, label):
    """Poll a kernel until COMPLETE or ERROR."""
    while True:
        r = subprocess.run(["kaggle", "kernels", "status", slug], capture_output=True, text=True)
        out = r.stdout.strip()
        ts = time.strftime("%H:%M:%S")
        if "COMPLETE" in out:
            print(f"[{ts}] {label} COMPLETE!", flush=True)
            return True
        elif "ERROR" in out or "CANCEL" in out:
            print(f"[{ts}] {label} FAILED: {out}", flush=True)
            # Get logs
            r2 = subprocess.run([sys.executable, "scripts/kaggle_logs.py", slug, "--tail", "30"],
                              capture_output=True, text=True)
            print(r2.stdout, flush=True)
            return False
        else:
            print(f"[{ts}] {label}: RUNNING", flush=True)
            time.sleep(60)

def push_kernel(path, label):
    """Push a kernel and return success."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] Pushing {label}...", flush=True)
    r = subprocess.run(["kaggle", "kernels", "push", "-p", path], capture_output=True, text=True)
    print(r.stdout.strip(), flush=True)
    if r.stderr:
        print(r.stderr.strip(), flush=True)
    return "successfully pushed" in r.stdout.lower()

# Stage 1: Monitor 10a (PCA 512D)
print("=" * 60)
print("PHASE 1: Monitoring 10a (PCA 384→512D)...")
print("=" * 60, flush=True)
if not poll_until_done(slug_10a, "10a"):
    print("10a FAILED — aborting chain")
    sys.exit(1)

# Get 10a logs
r = subprocess.run([sys.executable, "scripts/kaggle_logs.py", slug_10a, "--tail", "20"],
                  capture_output=True, text=True)
print(r.stdout, flush=True)

# Stage 2: Push and monitor 10b
print("\n" + "=" * 60)
print("PHASE 2: Pushing and monitoring 10b (FAISS indexing)...")
print("=" * 60, flush=True)
if not push_kernel("notebooks/kaggle/10b_stage3", "10b"):
    print("10b push failed — aborting")
    sys.exit(1)
time.sleep(30)  # Give Kaggle time to queue

if not poll_until_done(slug_10b, "10b"):
    print("10b FAILED — aborting chain")
    sys.exit(1)

# Stage 3: Push and monitor 10c
print("\n" + "=" * 60)
print("PHASE 3: Pushing and monitoring 10c (PCA 512D + intra-merge wins)...")
print("=" * 60, flush=True)
if not push_kernel("notebooks/kaggle/10c_stages45", "10c"):
    print("10c push failed — aborting")
    sys.exit(1)
time.sleep(30)

if not poll_until_done(slug_10c, "10c"):
    print("10c FAILED!")
    sys.exit(1)

# Get final results
print("\n" + "=" * 60)
print("FINAL RESULTS (10c with PCA 512D + intra-merge v72):")
print("=" * 60, flush=True)
r = subprocess.run([sys.executable, "scripts/kaggle_logs.py", slug_10c, "--tail", "80"],
                  capture_output=True, text=True)
print(r.stdout, flush=True)

print("\nChain complete!")
