"""Wait for GPU slots to free up, then push corrected 09b+09c notebooks."""
from kaggle.api.kaggle_api_extended import KaggleApi
import time
import subprocess
import sys
from pathlib import Path

LOG = Path(__file__).resolve().parent.parent / "data" / "outputs" / "_push_log.txt"

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

api = KaggleApi()
api.authenticate()

KAGGLE_EXE = str(Path(sys.executable).parent / "kaggle.exe")

KERNELS = {
    "09b": {
        "slug": "yahiaakhalafallah/09b-vehicle-reid-384px-fine-tune",
        "dir": "notebooks/kaggle/09b_vehicle_reid_384px",
    },
}

pushed = set()
MAX_ATTEMPTS = 90  # 90 * 2min = 3h max

log("Waiting for GPU slots to free up...")
for attempt in range(MAX_ATTEMPTS):
    for name, info in KERNELS.items():
        if name in pushed:
            continue
        status = str(api.kernels_status(info["slug"]).status)
        if "RUNNING" not in status:
            log(f"{name} finished: {status}")
            r = subprocess.run(
                [KAGGLE_EXE, "kernels", "push", "-p", info["dir"]],
                capture_output=True, text=True,
            )
            out = (r.stdout + r.stderr).strip()
            if "successfully pushed" in out.lower():
                log(f"-> Pushed {name}! {out}")
                pushed.add(name)
            else:
                log(f"-> Push attempt: {out[:200]}")

    if len(pushed) == len(KERNELS):
        log("All pushed successfully!")
        break

    remaining = [n for n in KERNELS if n not in pushed]
    log(f"Still waiting on: {remaining}")
    time.sleep(120)
else:
    unpushed = [n for n in KERNELS if n not in pushed]
    log(f"Timed out. Not pushed: {unpushed}")
