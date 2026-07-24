"""Fit per-stream score calibrations on CityFlowV2 validation ground truth.

GPU kernel (T4). Runs `athar run --profile production` over ALL cameras of
the first CityFlowV2 validation scene (S02, c006-c009), labels every
tracklet with its ground-truth vehicle id (IoU majority vote against the
scene's gt.txt files — the validation split ships GT, so no human
annotation is needed), builds cross-camera same/different pairs per
embedding stream, and fits the Platt calibrations through
``athar.search.calibration.ScoreCalibration.fit`` — the exact code path
the serving API uses.

Person streams get no labels here (AIC GT is vehicles only) and are
skipped by the pair-count guard rather than being given a made-up fit.

Inputs: same three datasets as the production-validation kernel.
Outputs: stream_calibrations.json (StreamCalibrations schema, provenance
in ``fitted_on``) + calibration_fit_report.json (pair counts, score
separation, per-stream sanity probes).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
PROJECT = Path("/tmp/athar-v2")
RUNS_ROOT = Path("/tmp/runs")
GDRIVE_ID = "13wNJpS_Oaoe-7y5Dzexg_Ol7bKu1OWuC"  # AIC22_Track1_MTMC_Tracking.zip

IOU_MATCH = 0.5        # observation matches a GT box
MIN_PURITY = 0.8       # majority GT id must own this share of matched obs
MIN_COVERAGE = 0.5     # share of observations that must match any GT box
NEG_PER_POS = 20       # cap negative pairs (they dominate ~99:1 otherwise)
MIN_POSITIVES = 5      # below this a stream fit is meaningless

REID_CKPTS = {
    "vehicle_transreid_vit_base_veri776.pth": "vehicle_transreid_vit_base_veri776.pth",
    "clipsenet_v6_veri776.pth": "clipsenet_v6_veri776_best.pth",
    "clipsenet_v6_veri776_best.pth": "clipsenet_v6_veri776_best.pth",
    "vehicle_transreid_dinov2_large_cityflowv2_final.pth":
        "vehicle_transreid_dinov2_large_cityflowv2_final.pth",
    "person_transreid_vit_base_market1501.pth":
        "person_transreid_vit_base_market1501.pth",
}


def run(cmd, cwd=None, env=None):
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=cwd, env=env, check=True)


def find_dir(marker: str) -> Path:
    for path in INPUT.rglob(marker):
        return path.parent if path.is_file() else path
    raise FileNotFoundError(f"marker {marker!r} not found under {INPUT}")


def fetch_validation_scene() -> list[tuple[str, Path]]:
    """All cameras of the first validation scene (GT ships with the split)."""
    search_roots = [
        Path("/tmp/cityflowv2"),
        Path("/kaggle/input/cityflowv2"),
        Path("/kaggle/input/aic22-track1-mtmc-tracking"),
    ]
    root = next(
        (r for r in search_roots if r.exists() and any(r.rglob("vdo.avi"))), None
    )
    if root is None:
        archive = Path("/tmp/AIC22_Track1_MTMC_Tracking.zip")
        print(f"Downloading CityFlowV2 (gdrive id={GDRIVE_ID}, ~20GB)...")
        import gdown

        gdown.download(f"https://drive.google.com/uc?id={GDRIVE_ID}",
                       str(archive), quiet=False)
        staging = Path("/tmp/_aic22_staging")
        staging.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(str(archive)) as zf:
            members = [m for m in zf.namelist()
                       if m.startswith("validation/") or "/validation/" in m]
            zf.extractall(str(staging), members=members or None)
        archive.unlink()
        root = staging

    by_scene: dict[Path, list[Path]] = {}
    for vdo in sorted(root.rglob("vdo.avi")):
        by_scene.setdefault(vdo.parent.parent, []).append(vdo)
    ordered = sorted(
        by_scene.items(),
        key=lambda item: (0 if "validation" in str(item[0]) else 1, str(item[0])),
    )
    for scene, videos in ordered:
        if len(videos) >= 2 and all((v.parent / "gt" / "gt.txt").is_file() for v in videos):
            print(f"scene {scene.name}: {len(videos)} cameras with GT")
            return [(v.parent.name, v) for v in videos]
    raise RuntimeError("no validation scene with >= 2 GT-bearing cameras found")


# ---------------------------------------------------------------------------
# GT labeling
# ---------------------------------------------------------------------------

def load_gt(gt_path: Path) -> dict[int, list[tuple[int, float, float, float, float]]]:
    """MOT gt.txt -> {frame: [(gt_id, x1, y1, x2, y2), ...]}. Frames 1-based."""
    frames: dict[int, list] = defaultdict(list)
    for line in gt_path.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue
        frame, gid = int(parts[0]), int(parts[1])
        x, y, w, h = (float(v) for v in parts[2:6])
        frames[frame].append((gid, x, y, x + w, y + h))
    return frames


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def match_score(observations, gt_frames, shift: int) -> int:
    hits = 0
    for obs in observations:
        boxes = gt_frames.get(obs["frame_index"] + shift)
        if not boxes:
            continue
        bb = obs["bbox"]
        box = (bb["x1"], bb["y1"], bb["x2"], bb["y2"])
        if any(iou(box, g[1:]) >= IOU_MATCH for g in boxes):
            hits += 1
    return hits


def label_tracklets(tracklets_json: Path, gt_path: Path) -> dict[int, int]:
    """{track_id: gt_id} for tracklets that map cleanly onto one GT vehicle."""
    payload = json.loads(tracklets_json.read_text())
    gt_frames = load_gt(gt_path)
    all_obs = [o for obs in payload["observations"].values() for o in obs]
    # our frame_index is 0-based, MOT frames are 1-based; probe the shift
    # on a sample rather than assuming (the P4 lesson)
    sample = all_obs[:: max(1, len(all_obs) // 500)]
    shift = max((-1, 0, 1, 2), key=lambda s: match_score(sample, gt_frames, s))
    print(f"{payload['camera_id']}: frame shift {shift:+d}")

    labels: dict[int, int] = {}
    for track_id_str, observations in payload["observations"].items():
        votes: Counter = Counter()
        matched = 0
        for obs in observations:
            boxes = gt_frames.get(obs["frame_index"] + shift)
            if not boxes:
                continue
            bb = obs["bbox"]
            box = (bb["x1"], bb["y1"], bb["x2"], bb["y2"])
            best = max(boxes, key=lambda g: iou(box, g[1:]))
            if iou(box, best[1:]) >= IOU_MATCH:
                votes[best[0]] += 1
                matched += 1
        if not votes or matched / len(observations) < MIN_COVERAGE:
            continue
        gt_id, top = votes.most_common(1)[0]
        if top / matched >= MIN_PURITY:
            labels[int(track_id_str)] = gt_id
    return labels


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    bundle = find_dir("GIT_SHA.txt")
    if bundle.name == "src":
        bundle = bundle.parent
    if not PROJECT.exists():
        shutil.copytree(bundle / "src", PROJECT)
    git_sha = (PROJECT / "GIT_SHA.txt").read_text().strip()
    print("src head:", git_sha)

    run([sys.executable, "-m", "pip", "install", "-q",
         "faiss-cpu", "lapx", "pydantic>=2", "pydantic-settings", "pyyaml",
         "timm", "filterpy", "gdown", "av", "torchcodec"])
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
         "ultralytics", "boxmot==22.0.0"])
    env = {**os.environ, "PYTHONPATH": str(PROJECT)}
    sys.path.insert(0, str(PROJECT))

    import torch

    print("torch", torch.__version__, "device", torch.cuda.get_device_name(0))
    assert float((torch.ones(8, device="cuda") * 2).sum().item()) == 16.0

    for sub in ("detection", "tracker", "reid"):
        (PROJECT / "models" / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy2(bundle / "models" / "yolo26m.pt",
                 PROJECT / "models" / "detection" / "yolo26m.pt")
    shutil.copy2(bundle / "models" / "osnet_x0_25_msmt17.pt",
                 PROJECT / "models" / "tracker" / "osnet_x0_25_msmt17.pt")
    placed: set[str] = set()
    for path in INPUT.rglob("*.pth"):
        target = REID_CKPTS.get(path.name)
        if target and target not in placed:
            shutil.copy2(path, PROJECT / "models" / "reid" / target)
            placed.add(target)
    missing = set(REID_CKPTS.values()) - placed
    assert not missing, f"missing checkpoints: {missing}"

    cams = fetch_validation_scene()

    cmd = [sys.executable, "-m", "athar.cli.main", "run",
           "--profile", "production", "--role", "gallery",
           "--runs-root", str(RUNS_ROOT),
           # cuda:0 NEVER bare "cuda" (ultralytics select_device writes it
           # into CUDA_VISIBLE_DEVICES and masks the GPU)
           "--set", "detect_track.device=cuda:0",
           "--set", "embed.device=cuda:0"]
    for cam_name, video in cams:
        cmd += ["--video", f"{cam_name}={video}"]
    run(cmd, cwd=PROJECT, env=env)

    run_dir = next(p for p in sorted(RUNS_ROOT.iterdir())
                   if (p / "manifest.json").exists())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "completed", manifest["status"]

    # ---- label tracklets against GT -----------------------------------
    labels: dict[tuple[str, int], int] = {}
    per_cam_counts = {}
    for cam_name, video in cams:
        cam_labels = label_tracklets(
            run_dir / "tracklets" / f"{cam_name}.json",
            video.parent / "gt" / "gt.txt",
        )
        for track_id, gt_id in cam_labels.items():
            labels[(cam_name, track_id)] = gt_id
        total = len(json.loads(
            (run_dir / "tracklets" / f"{cam_name}.json").read_text()
        )["tracklets"])
        per_cam_counts[cam_name] = {"labeled": len(cam_labels), "tracklets": total}
        print(f"{cam_name}: {len(cam_labels)}/{total} tracklets labeled")

    # ---- per-stream cross-camera pairs + Platt fit --------------------
    import numpy as np

    from athar.search.calibration import (
        CalibrationError,
        ScoreCalibration,
        StreamCalibrations,
    )

    summary = json.loads((run_dir / "embed_summary.json").read_text())
    fitted = StreamCalibrations()
    report_streams = {}
    rng = np.random.default_rng(0)
    fitted_on = f"cityflowv2-validation-{cams[0][1].parent.parent.name}@{git_sha[:12]}"

    for stream in sorted(summary["streams"]):
        vecs: dict[tuple[str, int], np.ndarray] = {}
        for cam_name, _ in cams:
            npz_path = run_dir / "embeddings" / f"{cam_name}.{stream}.npz"
            if not npz_path.exists():
                continue
            data = np.load(npz_path)
            for row, track_id in enumerate(data["track_ids"]):
                key = (cam_name, int(track_id))
                if key in labels:
                    v = data["embeddings"][row].astype(np.float64)
                    norm = np.linalg.norm(v)
                    if norm > 0:
                        vecs[key] = v / norm
        keys = sorted(vecs)
        pos, neg = [], []
        for i, ka in enumerate(keys):
            for kb in keys[i + 1:]:
                if ka[0] == kb[0]:
                    continue  # calibrating CROSS-camera association scores
                score = float(vecs[ka] @ vecs[kb])
                (pos if labels[ka] == labels[kb] else neg).append(score)
        if len(pos) < MIN_POSITIVES or not neg:
            report_streams[stream] = {
                "skipped": True, "positives": len(pos), "negatives": len(neg),
            }
            print(f"{stream}: SKIPPED ({len(pos)} pos / {len(neg)} neg)")
            continue
        if len(neg) > NEG_PER_POS * len(pos):
            neg = list(rng.choice(neg, size=NEG_PER_POS * len(pos), replace=False))
        scores = pos + neg
        y = [1] * len(pos) + [0] * len(neg)
        try:
            calibration = ScoreCalibration.fit(scores, y, fitted_on=fitted_on)
        except CalibrationError as exc:
            report_streams[stream] = {
                "skipped": True, "positives": len(pos), "negatives": len(neg),
                "error": str(exc),
            }
            print(f"{stream}: FIT FAILED - {exc}")
            continue
        fitted.streams[stream] = calibration
        report_streams[stream] = {
            "skipped": False,
            "positives": len(pos), "negatives": len(neg),
            "mean_pos": float(np.mean(pos)), "mean_neg": float(np.mean(neg)),
            "midpoint": calibration.midpoint, "scale": calibration.scale,
            "probe": {f"{s:.2f}": round(calibration.probability(s), 4)
                      for s in (0.3, 0.5, 0.7, 0.9)},
        }
        print(f"{stream}: fit ok - {len(pos)} pos / {len(neg)} neg, "
              f"midpoint {calibration.midpoint:.4f} scale {calibration.scale:.4f}")

    assert fitted.streams, "no stream produced a valid calibration"
    fitted.save(WORK / "stream_calibrations.json")
    report = {
        "git_sha": git_sha,
        "run_id": manifest["run_id"],
        "config_hash": (manifest.get("config") or {}).get("config_hash"),
        "scene": cams[0][1].parent.parent.name,
        "cameras": per_cam_counts,
        "fitted_on": fitted_on,
        "params": {"iou": IOU_MATCH, "purity": MIN_PURITY,
                   "coverage": MIN_COVERAGE, "neg_per_pos": NEG_PER_POS},
        "streams": report_streams,
    }
    (WORK / "calibration_fit_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
