"""VehicleX fetch PROBE (Kaggle CPU + internet).

Goal: determine whether the public VehicleX synthetic ReID images are fetchable
from the yorkeyao/VehicleX Google-Drive links, and report their archive structure
so we can write a clean extraction/restructure step next. Downloads nothing huge to
the repo — runs entirely on Kaggle, leaves artifacts in /kaggle/working.

The two Drive IDs below are the VeRi-776-adapted and VehicleID-adapted VehicleX
image sets listed in the VehicleX README (1,362 synthetic identities each, rendered
in Unity then domain-adapted). The CityFlow-adapted split is AIC-signup-gated and not
used here.
"""

import glob
import os
import subprocess
import sys

WORK = "/kaggle/working"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=False)
import gdown  # noqa: E402

DRIVE_IDS = {
    "vehiclex_veri_adapted": "1wLmUWY5clm88Jcmu1e5ITMYNCht_mnds",
    "vehiclex_vehicleid_adapted": "1C6VAf_Z19HuVPuUlb738HPRxpZKwWGx_",
}


def attempt(name: str, fid: str) -> str | None:
    """Try file download, then folder download. Return local path or None."""
    print("=" * 64)
    print(f"ATTEMPT {name}  (drive id {fid})")
    out = os.path.join(WORK, name)
    # 1) single-file (most adapted sets are a single .zip/.tar)
    try:
        path = gdown.download(id=fid, output=out, quiet=False, fuzzy=True)
        if path and os.path.isfile(path) and os.path.getsize(path) > 4096:
            print(f"  [file] downloaded -> {path}  ({os.path.getsize(path):,} bytes)")
            return path
        print(f"  [file] returned {path!r} (too small / not a file)")
    except Exception as e:  # noqa: BLE001
        print(f"  [file] failed: {e!r}")
    # 2) folder fallback
    try:
        os.makedirs(out, exist_ok=True)
        gdown.download_folder(id=fid, output=out, quiet=False, use_cookies=False)
        n = sum(len(f) for _, _, f in os.walk(out))
        print(f"  [folder] downloaded -> {out}  ({n} files)")
        return out if n else None
    except Exception as e:  # noqa: BLE001
        print(f"  [folder] failed: {e!r}")
    return None


def inspect(path: str) -> None:
    """Print archive/dir structure: type, entry count, sample names, id histogram."""
    import collections
    import tarfile
    import zipfile

    print("-" * 48, "INSPECT", path)
    names = []
    if os.path.isfile(path) and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
        print(f"  ZIP with {len(names)} entries")
    elif os.path.isfile(path) and tarfile.is_tarfile(path):
        with tarfile.open(path) as t:
            names = t.getnames()
        print(f"  TAR with {len(names)} entries")
    elif os.path.isdir(path):
        names = [
            os.path.relpath(os.path.join(d, f), path)
            for d, _, fs in os.walk(path)
            for f in fs
        ]
        print(f"  DIR with {len(names)} files")
    else:
        print(f"  unknown artifact type, size={os.path.getsize(path):,}")
        return

    for n in names[:40]:
        print("    ", n)
    imgs = [n for n in names if n.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"  image files: {len(imgs)}")
    # leading-token id histogram (VehicleX naming is id_cam_num.jpg)
    ids = collections.Counter()
    for n in imgs:
        base = os.path.basename(n).split("_")[0].split(".")[0]
        if base.isdigit():
            ids[base] += 1
    print(f"  distinct leading-int ids: {len(ids)}  (e.g. {list(ids)[:8]})")


def main() -> None:
    found = []
    for name, fid in DRIVE_IDS.items():
        p = attempt(name, fid)
        if p:
            found.append(p)
    print("=" * 64)
    print("TOP-LEVEL /kaggle/working:")
    for p in sorted(glob.glob(WORK + "/*")):
        size = os.path.getsize(p) if os.path.isfile(p) else "<dir>"
        print(f"  {p}  {size}")
    for p in found:
        inspect(p)
    print("PROBE DONE — downloaded:", found)


if __name__ == "__main__":
    main()
