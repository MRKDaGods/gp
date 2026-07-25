"""Stage VeRi-Wild TRAIN images as a private Kaggle dataset.

Sibling of the July veriwild-prep-bin kernel (which staged the 138,517 TEST
images as mrkdagods/veriwild-test-bin). Same 23-part GDrive rar archive and
extraction flow -- but this one tars the 277,797 TRAIN images listed in
train_list_start0.txt (30,671 identities) for the Phase 6 joint multi-domain
vehicle ReID retrain.

Output: private dataset mrkdagods/veriwild-train containing
  veriwild_train_a.tarbin, veriwild_train_b.tarbin  (~8.5GB each; .tarbin so
  Kaggle mounts them as opaque files -- consumers untar to local NVMe)
  train_list_start0.txt, vehicle_info.txt
If both tarbins together exceed the per-dataset cap, falls back to two
datasets (veriwild-train-a / veriwild-train-b), one tarbin each.

Secrets: the auth token below is a placeholder replaced by push_kernel.py
from ~/.kaggle/kaggle.json at push time -- the committed copy carries none.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time

# -- 0. In-kernel Kaggle auth (needed to create the output dataset) ----------
os.makedirs("/root/.kaggle", exist_ok=True)
with open("/root/.kaggle/kaggle.json", "w") as f:
    f.write("__KGAT_INJECTED_AT_PUSH__")
os.chmod("/root/.kaggle/kaggle.json", 0o600)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle==2.0.1"], check=True)


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, **kw)


chk = sh("kaggle datasets list --mine")
print("AUTH rc=", chk.returncode)
print(chk.stdout[:300], chk.stderr[:300])
assert chk.returncode == 0, "Kaggle auth FAILED in-kernel -- aborting before download."
print("AUTH OK -> proceeding to download.")

# -- 1. Download the 23-part rar archive (full VeRi-Wild image set) ----------
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "gdown"], check=True)
import gdown  # noqa: E402

PARTS = {
    "images.part01.rar": "1p8kjYVI1Bkj2LcC2y2Kv5lD0FalzgkpR",
    "images.part02.rar": "1lAwOdb5qVytL239YzbvxMcEihkjAM92m",
    "images.part03.rar": "19NnMttzJ1x6n8i48tm6R_LM_wXgo9AnL",
    "images.part04.rar": "155_m-3MUYuXKJ9F4bV5Av4bN_JG8P-6s",
    "images.part05.rar": "1G5yikBZxSeoDE7CMECe0zjJq4dy8YNlQ",
    "images.part06.rar": "1tlGa71UkAklRCroLX9WSGQiIOqYoxqC9",
    "images.part07.rar": "1zjmoIEiYJqa1ERWRs_M1to4sqr9DCKpI",
    "images.part08.rar": "1pmi6PTs5fZLxfp-pgkh0SobrjWPpXBPx",
    "images.part09.rar": "1hZ8Lh4ZiDsfEAdMlpZ6MTsrzSUMobCtb",
    "images.part10.rar": "1ExzVtVEMUVOMnAdLY1qPvhZyjjdjMEm3",
    "images.part11.rar": "164gk4u3YqwwZBJqYE-Qek1aB3EhuAkFJ",
    "images.part12.rar": "1vv2fJH9iY6DDSqAqBRWbbWMP2BoWNlNG",
    "images.part13.rar": "1qIFrDyQmrlu2vKeIFmjAW9GznneNXSr0",
    "images.part14.rar": "1LqvQrTzwe2M8Jk844Vgzvin6eGRaQgdr",
    "images.part15.rar": "1ZCCibZgkG0cvmdoZbDe73dh2nTcp9ArK",
    "images.part16.rar": "1CI4dCssHITKbOl76JJWMAn5aK5IeCg3I",
    "images.part17.rar": "1f4W0ctQXxs5oiue5NmYvwrm_FzTR8mOf",
    "images.part18.rar": "1jHCpMWn2xS5kl3ObSwiJ0Vrqs2dMeDZb",
    "images.part19.rar": "1nPQ11uARfH6Y8UZBn7Hln29zYu_HLjXA",
    "images.part20.rar": "14ScRkLLiQLtqMfki7UNHrWpFQZp9cddL",
    "images.part21.rar": "1AJ50JM-QIrbf9Jn6l3hxSKOOWG0gtb9w",
    "images.part22.rar": "1Yaab4Zl4fyNQihGhB-T6e7VK5MJY-R9P",
    "images.part23.rar": "16C-Wu8rVj7TutUZ9hrd_3GZN1ANPcLm8",
}
os.makedirs("/tmp/rars", exist_ok=True)
for name, fid in PARTS.items():
    out = f"/tmp/rars/{name}"
    if os.path.exists(out) and os.path.getsize(out) > 700_000_000:
        print("skip existing", name, os.path.getsize(out))
        continue
    ok = False
    for attempt in range(3):
        try:
            gdown.download(id=fid, output=out, quiet=False)
            if os.path.getsize(out) > 700_000_000:
                ok = True
                break
        except Exception as e:  # noqa: BLE001
            print("retry", name, attempt, repr(e)[:200])
            time.sleep(5)
    assert ok, f"failed to download {name}"
sizes = {n: os.path.getsize(f"/tmp/rars/{n}") for n in PARTS}
print("downloaded", len(sizes), "parts, total GB=", round(sum(sizes.values()) / 1e9, 2))

# -- 2. Extract the multi-volume archive --------------------------------------
subprocess.run("apt-get -qq update && apt-get -qq install -y unar unrar >/dev/null 2>&1", shell=True)
print("DISK before extract:\n", sh("df -h /tmp /kaggle/working").stdout)
os.makedirs("/tmp/ex", exist_ok=True)
part01 = "/tmp/rars/images.part01.rar"


def try_extract():
    # unrar first (clean multi-volume join), then unar fallback
    r = sh(f"unrar x -o+ -idq {part01} /tmp/ex/")
    if r.returncode == 0:
        return "unrar", r
    print("unrar rc", r.returncode, r.stderr[-500:])
    r = sh(f"unar -quiet -force-overwrite -output-directory /tmp/ex {part01}")
    return "unar", r


tool, r = try_extract()
print("extract tool=", tool, "rc=", r.returncode)
print((r.stdout or "")[-800:])
print((r.stderr or "")[-800:])
found = glob.glob("/tmp/ex/**/*.jpg", recursive=True)
print("extracted jpgs found:", len(found))
print("sample paths:", found[:3])
print("DISK after extract:\n", sh("df -h /tmp").stdout)

# -- 3. Collect the TRAIN image set from train_list_start0.txt ----------------
gdown.download_folder(id="1NP-wo6lHBRQeQFJoVGHhOeMeqNIpBgjT", output="/tmp/tts", quiet=True, use_cookies=False)
need = []
seen = set()
for line in open("/tmp/tts/train_list_start0.txt"):
    if not line.strip():
        continue
    rel = line.split()[0]  # "<vehid>/<img>.jpg <label> <camid>"
    if rel not in seen:
        seen.add(rel)
        need.append(rel)
print("unique train images needed:", len(need))  # expect 277797
assert len(need) > 250_000, f"train list suspiciously short: {len(need)}"

sample = need[0]
root = None
for c in ["/tmp/ex/images", "/tmp/ex"]:
    if os.path.exists(os.path.join(c, sample)):
        root = c
        break
if root is None:
    hits = glob.glob(f"/tmp/ex/**/{sample}", recursive=True)
    assert hits, f"could not locate sample image {sample} under /tmp/ex"
    root = hits[0][: -len(sample) - 1]
print("images root =", root)

missing = [p for p in need if not os.path.exists(os.path.join(root, p))]
print("missing after extract:", len(missing), missing[:5])
assert not missing, f"{len(missing)} needed images missing from extraction"

# free the rars now that extraction succeeded
shutil.rmtree("/tmp/rars", ignore_errors=True)

# -- 4. Tar in two halves (upload-cap safety + parallel consumer untar) -------
os.makedirs("/tmp/out", exist_ok=True)
need_sorted = sorted(need)
halves = {
    "veriwild_train_a.tarbin": need_sorted[: len(need_sorted) // 2],
    "veriwild_train_b.tarbin": need_sorted[len(need_sorted) // 2 :],
}
for tar_name, members in halves.items():
    lst = f"/tmp/{tar_name}.list"
    with open(lst, "w") as f:
        f.write("\n".join(members) + "\n")
    rc = subprocess.run(["tar", "-C", root, "-T", lst, "-cf", f"/tmp/out/{tar_name}"]).returncode
    assert rc == 0, f"tar failed for {tar_name}"
    print(tar_name, "GB=", round(os.path.getsize(f"/tmp/out/{tar_name}") / 1e9, 2), "members=", len(members))

shutil.copy("/tmp/tts/train_list_start0.txt", "/tmp/out/train_list_start0.txt")
if os.path.exists("/tmp/tts/vehicle_info.txt"):
    shutil.copy("/tmp/tts/vehicle_info.txt", "/tmp/out/vehicle_info.txt")
print("out dir:", sorted(os.listdir("/tmp/out")))
print("DISK before upload:\n", sh("df -h /tmp").stdout)


# -- 5. Create the private dataset (fallback: split into two datasets) --------
def create_dataset(dirpath, ds_id, title):
    meta = {"title": title, "id": ds_id, "licenses": [{"name": "other"}]}
    with open(os.path.join(dirpath, "dataset-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    for cmd in [
        f"kaggle datasets create -p {dirpath} -r tar",
        f"kaggle datasets create -p {dirpath}",
    ]:
        res = sh(cmd)
        print("\n$", cmd, "\n rc=", res.returncode)
        print(res.stdout[-800:])
        print(res.stderr[-800:])
        if res.returncode == 0:
            return True
        if "already exists" in (res.stdout + res.stderr):
            v = sh(f"kaggle datasets version -p {dirpath} -m update -r tar")
            print("version rc=", v.returncode, v.stdout[-400:], v.stderr[-400:])
            return v.returncode == 0
    return False


ok = create_dataset("/tmp/out", "mrkdagods/veriwild-train", "VeRi-Wild Train (images + list)")
if not ok:
    print("single-dataset upload failed -> splitting into two datasets")
    results = []
    for suffix, tar_name in [("a", "veriwild_train_a.tarbin"), ("b", "veriwild_train_b.tarbin")]:
        d = f"/tmp/out_{suffix}"
        os.makedirs(d, exist_ok=True)
        shutil.move(f"/tmp/out/{tar_name}", f"{d}/{tar_name}")
        shutil.copy("/tmp/out/train_list_start0.txt", f"{d}/train_list_start0.txt")
        results.append(
            create_dataset(d, f"mrkdagods/veriwild-train-{suffix}", f"VeRi-Wild Train {suffix.upper()} (images)")
        )
    ok = all(results)

print("\nDATASET UPLOAD OK =", ok)
assert ok, "dataset upload failed on all paths"
print("URL: https://www.kaggle.com/datasets/mrkdagods/veriwild-train")
