"""Production-profile validation kernel — v2 DAG with the full vehicle
fusion stack on real CityFlowV2 footage.

GPU kernel (T4). Runs `athar run --profile production` (ingest ->
detect_track -> embed[transreid+clipsenet+dinov2+hsv] -> index ->
associate[weighted stream fusion 1.0/0.7/0.525] -> package) over two
same-scene CityFlowV2 cameras, then reports identity counts and the
per-term fusion evidence. This is the first end-to-end exercise of
`associate.stream_weights` with all three appearance streams live.

Inputs (kernel dataset_sources):
- mrkdagods/athar-p4-bundle              (src/ tree + yolo26m + osnet)
- mrkdagods/mtmc-veri776-pipeline-weights (vehicle transreid/clipsenet/dinov2)
- gumfreddy/mtmc-weights                  (person transreid market1501)
CityFlowV2 videos come from a mounted copy when present, else the
AIC22 Google Drive archive (same recipe as the 09s kernel).

Outputs: production_validation.json + run artifacts tarball.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
PROJECT = Path("/tmp/athar-v2")
RUNS_ROOT = Path("/tmp/runs")
CITYFLOW_DIR = Path("/tmp/cityflowv2")
GDRIVE_ID = "13wNJpS_Oaoe-7y5Dzexg_Ol7bKu1OWuC"  # AIC22_Track1_MTMC_Tracking.zip

REID_CKPTS = {
    # dataset filename -> local_path filename (weights_manifest.yaml names)
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


def fetch_cityflow_cams() -> list[tuple[str, Path]]:
    """Two same-scene cameras (cross-camera association needs overlap)."""
    search_roots = [
        CITYFLOW_DIR,
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
            # only the validation scene's members — no need for all 20GB on disk
            members = [m for m in zf.namelist() if "/validation/" in m]
            zf.extractall(str(staging), members=members or None)
        archive.unlink()
        root = staging

    # group vdo.avi by scene dir; first scene with >= 2 cams wins
    by_scene: dict[Path, list[Path]] = {}
    for vdo in sorted(root.rglob("vdo.avi")):
        by_scene.setdefault(vdo.parent.parent, []).append(vdo)
    for scene, videos in sorted(by_scene.items()):
        if len(videos) >= 2:
            picked = videos[:2]
            print(f"scene {scene.name}: using " +
                  ", ".join(v.parent.name for v in picked))
            return [(v.parent.name, v) for v in picked]
    raise RuntimeError("no CityFlowV2 scene with >= 2 cameras found")


def main() -> None:
    bundle = find_dir("GIT_SHA.txt")
    if bundle.name == "src":
        bundle = bundle.parent
    print("bundle:", bundle)
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

    import torch

    print("torch", torch.__version__, "cuda", torch.version.cuda,
          "device", torch.cuda.get_device_name(0))
    assert float((torch.ones(8, device="cuda") * 2).sum().item()) == 16.0

    # torchcodec must decode on this image (video FrameSource, no cv2 path)
    import torchcodec  # noqa: F401

    print("torchcodec import OK")

    # ---- models into the tree ----------------------------------------
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
            print("ckpt:", path.name, "->", target)
    required = {
        "vehicle_transreid_vit_base_veri776.pth",
        "clipsenet_v6_veri776_best.pth",
        "vehicle_transreid_dinov2_large_cityflowv2_final.pth",
        "person_transreid_vit_base_market1501.pth",
    }
    missing = required - placed
    assert not missing, f"missing checkpoints: {missing}"

    # ---- footage ------------------------------------------------------
    cams = fetch_cityflow_cams()

    # ---- the run ------------------------------------------------------
    cmd = [sys.executable, "-m", "athar.cli.main", "run",
           "--profile", "production", "--role", "gallery",
           "--runs-root", str(RUNS_ROOT),
           "--set", "detect_track.device=cuda",
           "--set", "embed.device=cuda"]
    for cam_name, video in cams:
        cmd += ["--video", f"{cam_name}={video}"]
    run(cmd, cwd=PROJECT, env=env)

    run_dir = next(p for p in sorted(RUNS_ROOT.iterdir())
                   if (p / "manifest.json").exists())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    print("run:", manifest["run_id"], "status:", manifest["status"])
    assert manifest["status"] == "completed", f"run not completed: {manifest['status']}"

    # ---- report -------------------------------------------------------
    summary = json.loads((run_dir / "embed_summary.json").read_text())
    traj_rel = manifest["artifacts"]["associate.trajectories"]["relpath"]
    trajectories = json.loads((run_dir / traj_rel).read_text())
    cross = [t for t in trajectories["trajectories"]
             if len({m["camera_id"] for m in t["members"]}) > 1]
    report = {
        "git_sha": git_sha,
        "config_hash": (manifest.get("config") or {}).get("config_hash"),
        "run_id": manifest["run_id"],
        "cameras": [c for c, _ in cams],
        "streams": sorted(summary["streams"]),
        "num_identities": trajectories["num_identities"],
        "num_cross_camera": len(cross),
        "cross_camera_evidence": [
            {"global_id": t["global_id"], "entity_class": t["entity_class"],
             "confidence": t.get("confidence"), "evidence": t.get("evidence")}
            for t in cross[:20]
        ],
    }
    print(json.dumps(report, indent=2))
    (WORK / "production_validation.json").write_text(json.dumps(report, indent=2))

    with tarfile.open(WORK / "production_run_artifacts.tar.gz", "w:gz") as tar:
        tar.add(run_dir / "manifest.json", arcname="manifest.json")
        tar.add(run_dir / "events.jsonl", arcname="events.jsonl")
        for record in manifest["artifacts"].values():
            relpath = record["relpath"] if isinstance(record, dict) else record
            src = run_dir / relpath
            if src.is_file() and src.stat().st_size < 200 * 1024 * 1024:
                tar.add(src, arcname=relpath)
    print("wrote", WORK / "production_run_artifacts.tar.gz")


if __name__ == "__main__":
    main()
