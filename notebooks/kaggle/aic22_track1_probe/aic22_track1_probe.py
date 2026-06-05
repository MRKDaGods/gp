"""AIC22 Track1 MTMC dataset probe (Kaggle CPU + internet).

Two goals, run in one cheap CPU kernel:

  PART 1 - inventory the ALREADY-mounted ``thanhnguyenle/data-aicity-2023-track-2``
           Kaggle dataset that our training notebooks (09g, 13f, ...) already use:
           which scenes/cameras, GT presence, any ReID-crop / VehicleX subset.

  PART 2 - download the user-provided Google-Drive ``AICity22_Track1_MTMC_Tracking.zip``
           (drive id 13wNJpS_Oaoe-7y5Dzexg_Ol7bKu1OWuC), list its archive structure
           WITHOUT a full extract, and report whether it ADDS anything beyond Part 1
           (more train scenes S03/S04, the test split S05/S06, or VehicleX synthetic).

Leaves only small text artifacts in /kaggle/working; deletes the big zip when done.
"""

import collections
import glob
import os
import shutil
import subprocess
import sys
import zipfile

INPUT = "/kaggle/input"
WORK = "/kaggle/working"
DRIVE_ID = "13wNJpS_Oaoe-7y5Dzexg_Ol7bKu1OWuC"

# Tokens that flag interesting content when seen in archive/dir entry names.
REID_HINTS = ("image_train", "image_query", "image_test", "reid", "train_label", "name_train")
SYNTH_HINTS = ("vehiclex", "synthetic", "/sim", "sim_", "unity", "gan", "augment")


def banner(text: str) -> None:
    print("=" * 72)
    print(text)
    print("=" * 72, flush=True)


def best_scratch_dir(min_free_gb: float = 25.0) -> str:
    """Pick a writable dir with the most free space for the big download."""
    candidates = ["/kaggle/temp", "/tmp", "/kaggle/working", os.getcwd()]
    best, best_free = None, -1.0
    for d in candidates:
        try:
            os.makedirs(d, exist_ok=True)
            free_gb = shutil.disk_usage(d).free / 1e9
        except Exception as e:  # noqa: BLE001
            print(f"  scratch candidate {d!r}: unusable ({e!r})")
            continue
        print(f"  scratch candidate {d!r}: {free_gb:,.1f} GB free")
        if free_gb > best_free:
            best, best_free = d, free_gb
    print(f"  -> chose {best!r} ({best_free:,.1f} GB free; need ~{min_free_gb} GB)")
    return best


def summarize_names(names, total_uncompressed=None) -> None:
    """Print scene/camera structure + content flags for a list of entry paths."""
    n = len(names)
    print(f"  entries: {n:,}")
    if total_uncompressed is not None:
        print(f"  uncompressed size: {total_uncompressed / 1e9:,.2f} GB")

    # top-level dirs
    tops = collections.Counter(p.replace("\\", "/").split("/")[0] for p in names if p.strip())
    print(f"  top-level entries: {dict(list(tops.items())[:20])}")

    lower = [p.replace("\\", "/").lower() for p in names]

    # split roots
    for split in ("train", "validation", "test"):
        hit = sum(1 for p in lower if f"/{split}/" in p or p.startswith(f"{split}/"))
        if hit:
            print(f"  split '{split}': {hit:,} entries")

    # scenes Sxx and cameras cxxx
    import re

    scenes = collections.Counter(m.group(0) for p in names for m in re.finditer(r"S0\d\b", p))
    cams = collections.Counter(m.group(0) for p in names for m in re.finditer(r"\bc0\d\d\b", p))
    print(f"  scenes: {dict(sorted(scenes.items()))}")
    print(f"  distinct cameras: {len(cams)}  e.g. {sorted(cams)[:12]}")

    # key files
    for key in ("gt.txt", "vdo.avi", "calibration.txt", "roi.jpg", "det", "mtsc"):
        hit = sum(1 for p in lower if p.endswith(key) or f"/{key}" in p)
        if hit:
            print(f"  '{key}': {hit:,}")

    # ReID / synthetic flags
    reid = sorted({h for h in REID_HINTS for p in lower if h in p})
    synth = sorted({h for h in SYNTH_HINTS for p in lower if h.strip("/") in p})
    imgs = sum(1 for p in lower if p.endswith((".jpg", ".jpeg", ".png")))
    print(f"  image files: {imgs:,}")
    print(f"  ReID-subset hints present: {reid or 'NONE'}")
    print(f"  SYNTHETIC/VehicleX hints present: {synth or 'NONE'}")

    print("  --- sample entries ---")
    for p in names[:40]:
        print("    ", p)


def inventory_kaggle() -> None:
    banner("PART 1: inventory already-mounted Kaggle datasets under /kaggle/input")
    if not os.path.isdir(INPUT):
        print("  /kaggle/input missing")
        return
    print("  mounted dataset roots:")
    for d in sorted(glob.glob(INPUT + "/*")):
        print("    ", d)

    gts = glob.glob(INPUT + "/**/gt.txt", recursive=True)
    vids = glob.glob(INPUT + "/**/vdo.avi", recursive=True)
    print(f"\n  gt.txt files found: {len(gts):,}")
    print(f"  vdo.avi files found: {len(vids):,}")

    # build "name list" analog from the on-disk tree (depth-limited walk)
    names = []
    for root, dirs, files in os.walk(INPUT):
        depth = root[len(INPUT):].count(os.sep)
        if depth > 5:
            dirs[:] = []
            continue
        for f in files:
            names.append(os.path.relpath(os.path.join(root, f), INPUT))
        if len(names) > 400000:
            print("  (walk truncated at 400k entries)")
            break
    summarize_names(names)


def download_drive(drive_id: str) -> str | None:
    banner("PART 2a: download Google-Drive zip via gdown")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=False)
    import gdown  # noqa: E402

    scratch = best_scratch_dir()
    out = os.path.join(scratch, "aic22_track1_mtmc.zip")
    try:
        path = gdown.download(id=drive_id, output=out, quiet=False, fuzzy=True)
    except Exception as e:  # noqa: BLE001
        print(f"  gdown FAILED: {e!r}")
        print("  (large Drive files often hit quota / virus-scan interstitial)")
        return None
    if path and os.path.isfile(path) and os.path.getsize(path) > 1_000_000:
        print(f"  downloaded -> {path}  ({os.path.getsize(path) / 1e9:,.2f} GB)")
        return path
    print(f"  gdown returned {path!r} (missing / too small)")
    return None


def inspect_zip(path: str) -> None:
    banner("PART 2b: inspect the downloaded archive (no full extract)")
    if not zipfile.is_zipfile(path):
        print(f"  NOT a zip: {path} (size {os.path.getsize(path):,})")
        return
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        total = sum(i.file_size for i in infos)
        summarize_names(names, total_uncompressed=total)
        # extract just the small text/meta files for verification
        small_exts = (".txt", ".json", ".xml", ".cfg", ".ini")
        extracted = 0
        sample_dir = os.path.join(WORK, "aic22_sample_textfiles")
        os.makedirs(sample_dir, exist_ok=True)
        for i in infos:
            if i.filename.lower().endswith(small_exts) and i.file_size < 2_000_000:
                try:
                    z.extract(i, sample_dir)
                    extracted += 1
                except Exception:  # noqa: BLE001
                    pass
                if extracted >= 60:
                    break
        print(f"\n  extracted {extracted} small text/meta files -> {sample_dir}")


def main() -> None:
    inventory_kaggle()
    zip_path = download_drive(DRIVE_ID)
    if zip_path:
        try:
            inspect_zip(zip_path)
        finally:
            banner("cleanup: deleting big zip to keep kernel output small")
            try:
                os.remove(zip_path)
                print(f"  removed {zip_path}")
            except Exception as e:  # noqa: BLE001
                print(f"  could not remove {zip_path}: {e!r}")
    else:
        banner("PART 2 SKIPPED — Drive download unavailable")
        print("  Verdict relies on PART 1. If the Drive zip is needed, it likely")
        print("  hit Drive quota; retry later or ingest via Kaggle 'Add Data > URL'.")

    banner("PROBE DONE")
    print("Check PART 1 vs PART 2 above:")
    print("  * Does the Drive zip add train scenes (S03/S04) or the test split (S05/S06)")
    print("    beyond what data-aicity-2023-track-2 already provides?")
    print("  * Does EITHER source contain VehicleX / synthetic / ReID crops?")


if __name__ == "__main__":
    main()
