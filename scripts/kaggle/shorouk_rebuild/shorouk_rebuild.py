"""Rebuild the Shorouk MTMC dataset from the legacy Kaggle upload.

Input : gumfreddy/seif-dataset  (Hikvision DVR exports named D{ch}_{ts}.mp4,
        which are NOT real mp4 containers, plus DVR event-log .txt sidecars)
Output: a new public dataset  mrkdagods/shorouk-dataset  reproducing the
        local layout:  c0XX/vdo.mp4  (HEVC 1080p25 ~1.7 Mbps, ~1295 s,
        synchronized) + camera_coordinates.json

How the legacy upload maps to the local dataset (reverse-engineered from the
local files' burned-in OSD):
  * Each D{ch}_{YYYYMMDDHHMMSS}.mp4 is an ~80 min MPEG-PS export that rolls
    over at ~1 GB; four channels therefore have a second continuation file
    (same channel, later timestamp) - they are rollovers, not duplicates.
  * The D channel number is NOT the physical camera id. The true id is the
    "Camera XX" OSD burned into the bottom-right of the frame.
  * The local dataset is a synchronized ~21.6 min window: it starts at
    T0 = the latest segment start across cameras (17:40:51 DVR time, the
    start of D07_20320610174051) and runs ~1295.08 s = until the earliest-
    ending export (D01) runs out. Files that end sooner just stop early,
    which reproduces the 1293.8-1295.1 s spread seen locally.

Pipeline:
  1. Sniff the real container of every D*.mp4 (magic bytes + ffprobe).
  2. Read the true camera id from the "Camera XX" OSD digits by glyph
     TEMPLATE MATCHING against templates harvested from the ground-truth
     local dataset (tesseract systematically misreads this blocky font:
     7->2, 1->0; templates scored 56/56 on the local videos).
  3. Per camera keep the LATEST segment (earlier rollovers are superseded),
     drop cameras not in CAMERA_COORDS, derive T0 = max(kept start times).
  4. Trim each kept file from (T0 - file_start) for TARGET_DURATION seconds
     and re-encode to HEVC 1080p25 ~1.73 Mbps (NVENC, libx264 fallback) as
     c0XX/vdo.mp4. Self-checks: the output's frame-0 OSD digits must
     template-match the same camera id (hard error), and its OSD clock is
     OCR'd (warn-only) - it must sit in the 17:40:49-56 band.
  5. Write camera_coordinates.json and publish everything as a public
     dataset via the Kaggle API (credentials from Kaggle notebook secrets
     KAGGLE_USERNAME / KAGGLE_KEY).

Kernel needs: GPU (for NVENC), internet, and the two secrets toggled on.
"""

import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- config

INPUT_DIR = Path("/kaggle/input/seif-dataset")
WORK = Path("/kaggle/working")
OUT = WORK / "shorouk"
FRAMES = WORK / "_frames"
REPORT_PATH = WORK / "rebuild_report.json"

DATASET_ID = "mrkdagods/shorouk-dataset"
DATASET_TITLE = "Shorouk MTMC Dataset"

TARGET_BITRATE = "1730k"      # local reference vdo.mp4 files are ~1.73 Mbps HEVC
MAXRATE = "3500k"
BUFSIZE = "7000k"
TARGET_DURATION = "1295.08"   # local max duration; short sources stop early
EXPECTED_T0 = datetime(2032, 6, 10, 17, 40, 51)  # sanity check for derived T0
OCR_FRAME_CANDIDATES = (9, 100, 250, 500)        # 10th frame first, then fallbacks

# Verbatim copy of the local data/raw/shorouk/camera_coordinates.json
CAMERA_COORDS = {
    "17": {"lat": 30.140325, "lng": 31.619793, "label": "Camera 17"},
    "18": {"lat": 30.140526, "lng": 31.619919, "label": "Camera 18"},
    "19": {"lat": 30.140664, "lng": 31.619873, "label": "Camera 19"},
    "20": {"lat": 30.140557, "lng": 31.619148, "label": "Camera 20"},
    "21": {"lat": 30.140480, "lng": 31.618640, "label": "Camera 21"},
    "22": {"lat": 30.139732, "lng": 31.618444, "label": "Camera 22"},
    "23": {"lat": 30.140271, "lng": 31.618757, "label": "Camera 23"},
    "26": {"lat": 30.139564, "lng": 31.618054, "label": "Camera 26"},
    "27": {"lat": 30.139541, "lng": 31.618379, "label": "Camera 27"},
    "28": {"lat": 30.138971, "lng": 31.618153, "label": "Camera 28"},
    "29": {"lat": 30.139121, "lng": 31.618749, "label": "Camera 29"},
    "30": {"lat": 30.139047, "lng": 31.619275, "label": "Camera 30"},
    "31": {"lat": 30.139192, "lng": 31.619175, "label": "Camera 31"},
    "32": {"lat": 30.139504, "lng": 31.619205, "label": "Camera 32"},
}
EXPECTED_CAMERAS = set(CAMERA_COORDS)


def sh(cmd, check=True, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], check=check, **kw)


# ---------------------------------------------------------------- tooling

def setup_tools():
    """Install tesseract + pytesseract and fetch a static ffmpeg with NVENC."""
    sh(["apt-get", "install", "-y", "-qq", "tesseract-ocr"], check=False,
       capture_output=True)
    sh([sys.executable, "-m", "pip", "install", "-q", "pytesseract"])

    ffdir = WORK / "ffmpeg-static"
    ffmpeg, ffprobe = ffdir / "ffmpeg", ffdir / "ffprobe"
    if not ffmpeg.exists():
        url = ("https://github.com/BtbN/FFmpeg-Builds/releases/download/"
               "latest/ffmpeg-master-latest-linux64-gpl.tar.xz")
        tar_path = WORK / "ffmpeg.tar.xz"
        sh(["wget", "-q", "-O", tar_path, url])
        ffdir.mkdir(exist_ok=True)
        with tarfile.open(tar_path) as tf:
            for member in tf.getmembers():
                name = Path(member.name).name
                if member.isfile() and name in ("ffmpeg", "ffprobe"):
                    member.name = name
                    tf.extract(member, ffdir)
        tar_path.unlink()
        for binp in (ffmpeg, ffprobe):
            binp.chmod(binp.stat().st_mode | stat.S_IEXEC)
    return ffmpeg, ffprobe


def pick_encoder(ffmpeg):
    """Prefer NVENC HEVC (matches the local files); fall back gracefully."""
    for enc in ("hevc_nvenc", "h264_nvenc", "libx264"):
        probe = subprocess.run(
            [str(ffmpeg), "-v", "error", "-f", "lavfi",
             "-i", "testsrc=size=256x256:rate=25",
             "-frames:v", "3", "-c:v", enc, "-f", "null", "-"],
            capture_output=True)
        if probe.returncode == 0:
            print(f"encoder: {enc}")
            return enc
    raise RuntimeError("no working video encoder found")


# ---------------------------------------------------------------- sniffing

def sniff_container(path, ffprobe):
    with open(path, "rb") as fh:
        head = fh.read(16)
    if head[4:8] == b"ftyp":
        magic = "mp4/iso-bmff (genuine mp4)"
    elif head[:4] == b"\x00\x00\x01\xba":
        magic = "mpeg-ps (MPEG program stream)"
    elif head[:4] == b"DHAV":
        magic = "dav (Dahua DVR)"
    elif head[:3] == b"\x00\x00\x01":
        magic = "mpeg elementary/program stream"
    else:
        magic = f"unknown ({head[:8].hex()})"
    probe = subprocess.run(
        [str(ffprobe), "-v", "error", "-show_entries",
         "format=format_name,duration:stream=codec_name,codec_type",
         "-of", "json", str(path)],
        capture_output=True, text=True)
    fmt = {}
    if probe.returncode == 0:
        info = json.loads(probe.stdout)
        fmt = {
            "ffprobe_format": info.get("format", {}).get("format_name"),
            "duration_s": float(info.get("format", {}).get("duration") or 0),
            "codecs": [s.get("codec_name") for s in info.get("streams", [])],
        }
    return {"magic": magic, **fmt}


# ---------------------------------------------------------------- OCR

# ---- camera-id via glyph template matching ------------------------------
# The OSD layout is pixel-identical across all cameras (verified on the
# local 1080p videos): the two id digits of "Camera XX" occupy fixed 30x34
# cells. Templates below were harvested from the ground-truth local dataset
# (scripts/kaggle/shorouk_rebuild is self-contained: they are embedded as
# hex-packed bitmasks). Digits 4/5 never occur in the roster and have no
# template; an unmatchable glyph returns None rather than a guess.
OSD_CELL_Y = (964, 998)
OSD_CELL_XS = ((1556, 1586), (1582, 1612))
OSD_SHIFT = 4               # +/- px alignment search
OSD_ACCEPT_DIST = 90        # max Hamming distance (cell = 1020 px)
OSD_ACCEPT_MARGIN = 40      # required lead over the runner-up digit
OSD_TEMPLATE_HEX = {
    "0": "00000000000000000ff800003fe00000ffc0001f8fc0007e3f0001f8fc003f007e00fc01f803f007e00fc01f803f007e00fc01f803f1c7e00fc71f803f1c7e00fc71f803f1c7e00fc71f803f007e00fc01f803f007e00fc01f803f007e00fc01f8007e3f0001f8fc0007e3f00003fe00000ff800003fe000c000000000000000",
    "1": "0000000000000000007e000001f8000007e00000ff803003fe00c00ff80301ffe00c07ff80301ffe00c001f8030007e00c001f8030007e000001f8000007e000001f8000007e000001f8000007e000001f8000007e000001f8000007e000001f8000007e000001f8000007e00007fffe001ffff8007fffe00000000000000000",
    "2": "00000000000000001fffc0007fff0001fffc003f007e30fc01f8c3f007e300001f8c00007e300001f8c0003f030000fc0c0003f020007e000001f8000007e00000fc000003f000000fc00001f8000007e000001f000003f000030fc0000c3f000030fc01f843f007e30fc00f843ffffe00fffff803ffffe00000000000000000",
    "3": "00000000000000001fffc0007fff0001fffc003f007e00fc01f803f007e000001f8000007e000001f8000007e000001f8000007e0003ffc0000fff00003ffc0000007e000001f8000007e000001f8000007e000001f8000007e300001f8c00007e30fc01f803f007e00fc01f8007fff0001fffc0007fff000000000000000000",
    "6": "000000000000000007f800003fe00000ff00001f8000007e000001f800003f000000fc000003f000000fc000003f000000fc000003ffff000ffffc003ffff000fc01f803f007e00fc01f803f007e00fc01f803f007e00fc01f803f007e00fc01f803f007e00fc01f803f007e001fffc0007fff0001fffc000000000000000000",
    "7": "0000000000000003ffffe00fffff803ffffe00fc01f803f007e00fc01f8010007e004001f8010007e000001f8000007e000001f806003f000800fc000003f000007e000001f8000c07e00030fc000003f000000fc004203f000000fc000003f000000fc000003f000040fc000003f038000fc100003f70000007000000700000",
    "8": "00000000000000003fff0000fffc0007ffe000fc01f803f007e00fc01f803f007e00fc01f803f007e00fc01f803f007e00fc01f8007fff0001fffc0007fff000fc01f803f007e00fc01f803f007e00fc01f803f007e00fc01f803f007e00fc01f803f007e00fc01f803f003e001fffc0007fff0001fffc000000000000000000",
    "9": "00000000000000003fff0001fffc0007fff000fc01f803f007e00fc01f803f007e00fc01f803f007e00fc01f803f007e00fc01f8007fffe001ffff8007fffe000001f8000007e000001f8000007e000001f8000007e000001f8000007e000001f800003f000000fc000003f0001ffe00007ff80001ffe0000000000000000000",
}
CLOCK_CROPS = [(0.00, 0.36, 0.02, 0.12)]  # top-left "... Thu HH:MM:SS"

_TEMPLATE_CACHE = {}


def _osd_templates():
    if not _TEMPLATE_CACHE:
        h = OSD_CELL_Y[1] - OSD_CELL_Y[0]
        w = OSD_CELL_XS[0][1] - OSD_CELL_XS[0][0]
        for d, hx in OSD_TEMPLATE_HEX.items():
            bits = np.unpackbits(np.frombuffer(bytes.fromhex(hx), dtype=np.uint8))
            _TEMPLATE_CACHE[d] = bits[:h * w].reshape(h, w).astype(bool)
    return _TEMPLATE_CACHE


def _match_cell(gray, x0, x1):
    """Best-digit template match for one OSD cell, or None if unsure.
    Tries both OSD polarities (white-on-dark and auto-inverted black) and a
    small shift search to absorb sub-cell misalignment."""
    s = OSD_SHIFT
    pad = gray[OSD_CELL_Y[0] - s:OSD_CELL_Y[1] + s, x0 - s:x1 + s]
    scores = []
    for cand in (pad > 200, pad < 60):
        for d, t in _osd_templates().items():
            th, tw = t.shape
            best = th * tw
            for dy in range(2 * s + 1):
                for dx in range(2 * s + 1):
                    dist = int((cand[dy:dy + th, dx:dx + tw] ^ t).sum())
                    if dist < best:
                        best = dist
            scores.append((best, d))
    scores.sort()
    best_dist, best_d = scores[0]
    margin = next((sc for sc, d in scores if d != best_d), 10 ** 6) - best_dist
    if best_dist <= OSD_ACCEPT_DIST and margin >= OSD_ACCEPT_MARGIN:
        return best_d
    return None


def _grab_frame(video, frame_idx, png, ffmpeg):
    grab = subprocess.run(
        [str(ffmpeg), "-y", "-v", "error", "-i", str(video),
         "-vf", f"select=eq(n\\,{frame_idx})", "-frames:v", "1", str(png)],
        capture_output=True)
    return grab.returncode == 0 and png.exists()


def _ocr_variants(img):
    """Binarizations that always present dark-text-on-light to tesseract.
    The Hikvision OSD inverts polarity per character block against the local
    background, so a single global threshold can drop half the characters -
    every plausible polarity/threshold gets its own variant and the caller
    aggregates votes across all of them."""
    from PIL import Image, ImageOps
    big = img.convert("L").resize((img.width * 3, img.height * 3), Image.LANCZOS)
    yield ImageOps.invert(big.point(lambda p: 255 if p > 200 else 0))  # white glyphs
    yield big.point(lambda p: 0 if p < 60 else 255)                    # black glyphs
    yield ImageOps.invert(big.point(lambda p: 255 if p > 160 else 0))
    yield big


def _ocr_texts(img, crops, whitelist):
    """Yield raw OCR text for every (crop, variant, psm) combination."""
    import pytesseract
    w, h = img.size
    for x0, x1, y0, y1 in crops:
        crop = img.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
        for variant in _ocr_variants(crop):
            for psm in ("7", "6"):
                # whitelist must stay space/quote-free: pytesseract shlex-splits
                # the config string and chokes on quoted args on some platforms
                yield pytesseract.image_to_string(
                    variant,
                    config=f"--psm {psm} -c tessedit_char_whitelist={whitelist}")


CLOCK_RE = re.compile(r"\d{2}:(\d{2}):(\d{2})")


def _vote(counter, min_votes=2):
    """Return the winner only when it has enough votes and a unique lead."""
    ranked = counter.most_common(2)
    if not ranked or ranked[0][1] < min_votes:
        return None
    if len(ranked) > 1 and ranked[1][1] == ranked[0][1]:
        return None
    return ranked[0][0]


def read_camera_id(video, ffmpeg):
    """Read the camera id from the OSD digit cells by glyph template
    matching, retrying later frames if a passing car obscures the label."""
    from PIL import Image
    for frame_idx in OCR_FRAME_CANDIDATES:
        png = FRAMES / f"{video.parent.name}_{video.stem}_f{frame_idx}.png"
        if not _grab_frame(video, frame_idx, png, ffmpeg):
            continue
        gray = np.asarray(Image.open(png).convert("L"))
        if gray.shape != (1080, 1920):
            print(f"  {video.name}: unexpected resolution {gray.shape}")
            return None, frame_idx
        digits = [_match_cell(gray, x0, x1) for x0, x1 in OSD_CELL_XS]
        if None not in digits:
            return str(int("".join(digits))), frame_idx
    return None, None


def ocr_osd_clock(video, ffmpeg):
    """Consensus-OCR the OSD clock (top-left, frame 0) - warn-only check
    that a trimmed output starts inside the sync window. Only MM:SS is
    trusted: the blocky '7' in the hour field reads as '2' against busy
    backgrounds. Returns None (never raises) when OCR is unavailable."""
    try:
        from collections import Counter
        from PIL import Image
        png = FRAMES / f"{video.parent.name}_{video.stem}_check_f0.png"
        if not _grab_frame(video, 0, png, ffmpeg):
            return None
        votes = Counter()
        for text in _ocr_texts(Image.open(png), CLOCK_CROPS, "0123456789:-"):
            for mm, ss in CLOCK_RE.findall(text):
                votes[f"{mm}:{ss}"] += 1
        return _vote(votes, min_votes=1)
    except Exception as exc:
        print(f"  clock OCR unavailable: {exc}")
        return None


# ---------------------------------------------------------------- convert

def convert(src, dst, seek_s, ffmpeg, encoder):
    dst.parent.mkdir(parents=True, exist_ok=True)
    enc_opts = {
        "hevc_nvenc": ["-preset", "p5", "-rc", "vbr"],
        "h264_nvenc": ["-preset", "p5", "-rc", "vbr"],
        "libx264": ["-preset", "veryfast"],
    }[encoder]
    sh([ffmpeg, "-y", "-v", "warning", "-stats", "-stats_period", "60",
        "-fflags", "+genpts", "-ss", f"{seek_s:.2f}", "-i", src,
        "-t", TARGET_DURATION,
        "-map", "0:v:0", "-an",
        "-c:v", encoder, *enc_opts,
        "-b:v", TARGET_BITRATE, "-maxrate", MAXRATE, "-bufsize", BUFSIZE,
        "-r", "25", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", dst])


def probe_output(path, ffprobe):
    probe = subprocess.run(
        [str(ffprobe), "-v", "error", "-show_entries",
         "format=duration,size:stream=codec_name,width,height,r_frame_rate",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    info = json.loads(probe.stdout)
    s = info["streams"][0]
    return {
        "codec": s["codec_name"],
        "res": f'{s["width"]}x{s["height"]}',
        "fps": s["r_frame_rate"],
        "duration_s": round(float(info["format"]["duration"]), 1),
        "size_mb": round(int(info["format"]["size"]) / 1e6, 1),
    }


# ---------------------------------------------------------------- publish

def setup_credentials():
    """Fetch API creds (notebook secrets first, then environment) and write
    ~/.kaggle/kaggle.json. Called FIRST thing in main so a missing-secrets
    run fails in seconds instead of after an hour of encoding."""
    user = key = None
    try:
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        user, key = (secrets.get_secret("KAGGLE_USERNAME"),
                     secrets.get_secret("KAGGLE_KEY"))
    except Exception as exc:
        print(f"notebook secrets unavailable ({exc}); trying environment")
        user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if not (user and key):
        raise RuntimeError(
            "No Kaggle credentials - aborting BEFORE the heavy work. In the "
            "notebook editor open Add-ons > Secrets, add KAGGLE_USERNAME and "
            "KAGGLE_KEY for the mrkdagods account, tick them for this "
            "notebook, then rerun.")
    cred_dir = Path.home() / ".kaggle"
    cred_dir.mkdir(exist_ok=True)
    cred_file = cred_dir / "kaggle.json"
    cred_file.write_text(json.dumps({"username": user, "key": key}))
    cred_file.chmod(0o600)
    print(f"kaggle credentials configured for user: {user}")


def publish(out_dir):
    (out_dir / "dataset-metadata.json").write_text(json.dumps({
        "title": DATASET_TITLE,
        "id": DATASET_ID,
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2))

    create = subprocess.run(
        ["kaggle", "datasets", "create", "-p", str(out_dir),
         "--public", "--dir-mode", "zip"],
        capture_output=True, text=True)
    print(create.stdout, create.stderr)
    if create.returncode == 0 and "already exists" not in create.stdout.lower():
        return
    print("create failed or dataset exists; pushing a new version instead")
    sh(["kaggle", "datasets", "version", "-p", out_dir,
        "-m", "rebuild from seif-dataset (mime fix + OCR camera map + sync trim)",
        "--dir-mode", "zip"])


# ---------------------------------------------------------------- main

def parse_export_ts(name):
    return datetime.strptime(name.split("_", 1)[1].split(".")[0], "%Y%m%d%H%M%S")


def main():
    setup_credentials()  # fail fast if publish would be impossible
    OUT.mkdir(exist_ok=True)
    FRAMES.mkdir(exist_ok=True)
    ffmpeg, ffprobe = setup_tools()
    encoder = pick_encoder(ffmpeg)

    global INPUT_DIR
    videos = sorted(INPUT_DIR.glob("*.mp4"))
    if not videos:
        # mount landed somewhere else (renamed slug / nested layout) - search
        videos = sorted(Path("/kaggle/input").glob("**/D*.mp4"))
        if videos:
            INPUT_DIR = videos[0].parent
            print(f"input found at {INPUT_DIR} instead")
    if not videos:
        raise RuntimeError(
            f"no videos found under {INPUT_DIR} or /kaggle/input/**. Attach "
            "the gumfreddy/seif-dataset data source to this notebook.")
    print(f"{len(videos)} source videos")

    # 1+2: sniff containers and OCR the true camera ids
    records = []
    for video in videos:
        rec = {"file": video.name,
               "start": parse_export_ts(video.name),
               **sniff_container(video, ffprobe)}
        cam, frame_idx = read_camera_id(video, ffmpeg)
        rec.update({"camera": cam, "ocr_frame": frame_idx})
        print(f'{video.name}: {rec["magic"]} -> camera {cam} '
              f'(frame {frame_idx})', flush=True)
        records.append(rec)

    # 3: per camera keep the latest rollover segment; drop non-roster cameras
    chosen, dropped = {}, []
    for rec in sorted(records, key=lambda r: r["start"]):
        cam = rec["camera"]
        if cam is None:
            rec["status"] = "OCR FAILED"
            dropped.append(rec)
        elif cam not in EXPECTED_CAMERAS:
            rec["status"] = f"camera {cam} not in roster - skipped"
            dropped.append(rec)
        else:
            if cam in chosen:
                prev = chosen[cam]
                prev["status"] = f'superseded by later segment {rec["file"]}'
                dropped.append(prev)
            rec["status"] = "selected"
            chosen[cam] = rec  # later start overwrites earlier rollover

    missing = sorted(EXPECTED_CAMERAS - set(chosen), key=int)
    for rec in dropped:
        print(f'DROPPED {rec["file"]}: {rec["status"]}')
    if missing:
        print(f"MISSING cameras (no source mapped): {missing}")

    # derive the sync window start; all cameras get trimmed to [T0, T0+~1295s]
    t0 = max(rec["start"] for rec in chosen.values())
    print(f"\nsync window start T0 = {t0}  (expected {EXPECTED_T0})")
    if abs((t0 - EXPECTED_T0).total_seconds()) > 120:
        print("WARNING: derived T0 far from expected - check the OCR mapping")

    # 4: trim + re-encode the selected videos
    for cam in sorted(chosen, key=int):
        rec = chosen[cam]
        seek_s = (t0 - rec["start"]).total_seconds()
        dst = OUT / f"c{int(cam):03d}" / "vdo.mp4"
        print(f'\n=== camera {cam}: {rec["file"]} '
              f'@ +{seek_s:.0f}s -> {dst.relative_to(WORK)}')
        convert(INPUT_DIR / rec["file"], dst, seek_s, ffmpeg, encoder)
        rec["seek_s"] = seek_s
        rec["output"] = str(dst.relative_to(OUT))
        rec["output_probe"] = probe_output(dst, ffprobe)
        # hard self-check: the output's own OSD digits must read back as cam
        check_cam, _ = read_camera_id(dst, ffmpeg)
        if check_cam != cam:
            raise RuntimeError(
                f"{dst} OSD reads back as camera {check_cam}, expected {cam}")
        # soft self-check: OSD clock should sit in the ~40:49-40:59 band
        rec["output_osd_clock"] = ocr_osd_clock(dst, ffmpeg)
        if rec["output_osd_clock"] and not rec["output_osd_clock"].startswith("40:"):
            print(f"  WARNING: frame-0 OSD clock {rec['output_osd_clock']} "
                  "outside expected 40:49-40:59 band - check the trim")
        print(rec["output_probe"], "| frame-0 OSD clock:", rec["output_osd_clock"])

    # 5: coordinates + report
    (OUT / "camera_coordinates.json").write_text(
        json.dumps(CAMERA_COORDS, indent=2) + "\n")
    for rec in records:
        rec["start"] = rec["start"].isoformat()
    REPORT_PATH.write_text(json.dumps(
        {"encoder": encoder, "sync_window_start": t0.isoformat(),
         "records": records, "missing": missing}, indent=2))
    shutil.rmtree(FRAMES, ignore_errors=True)

    print("\n=== final tree ===")
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            print(f"{path.relative_to(OUT)}  {path.stat().st_size/1e6:.1f} MB")

    unmapped = [r["file"] for r in records if r["camera"] is None]
    if unmapped:
        print(f"NOTE: {len(unmapped)} file(s) had no confident OSD read and "
              f"were not used: {unmapped}")
    if missing:
        raise RuntimeError(
            "camera roster incomplete - NOT publishing. Inspect "
            f"{REPORT_PATH.name}, then adjust the template matcher or "
            "hardcode the failing file's camera id.")

    publish(OUT)
    print(f"\ndone: https://www.kaggle.com/datasets/{DATASET_ID}")


if __name__ == "__main__":
    main()
