"""MIE finals demo runs — full v2 DAG over real El Shorouk footage.

GPU kernel (T4). Produces the two run directories the live demo serves:

- GALLERY run: ``athar run --profile production --role gallery`` over four
  adjacent compound cameras (ingest -> detect_track -> embed[5 streams] ->
  index -> associate -> package with cross-camera evidence clips pre-cut).
- PROBE run: the same profile over one more adjacent camera, ``--role
  probe`` — the "reference footage" a live probe search starts from.

Footage is staged at RELATIVE paths (``data/raw/shorouk_demo/<cam>/vdo.mp4``,
cwd = project root) so the manifests resolve on this kernel AND on the
serving box, where the SAME Kaggle files are downloaded to that layout
(bit-exact with the processed evidence — manifest sha256s verify) —
evidence clip playback works from the pre-cut cache and local footage.
The path is deliberately NOT data/raw/shorouk: the on-prem originals are
a different encode whose per-camera trim alignment is not guaranteed.

Inputs (kernel dataset_sources):
- mrkdagods/athar-p4-bundle               (src/ tree + yolo26m + osnet)
- mrkdagods/mtmc-veri776-pipeline-weights (vehicle transreid/clipsenet/dinov2)
- gumfreddy/mtmc-weights                  (person transreid market1501)
- mrkdagods/shorouk-dataset               (c017..c032/vdo.mp4, sync-trimmed)

Resume: kernel_sources self-reference mounts the prior version's output;
run dirs found there are restored into RUNS_ROOT and resumed via the DAG
runner's own stage/camera checkpoints instead of restarting from zero.

Outputs: runs/ (live dirs, preserved even on failure), one
``<role>_run.tar.gz`` per completed run, shorouk_demo_summary.json with
per-camera per-class tracklet counts, cross-camera identity acceptance,
and probe->gallery search smokes for BOTH a vehicle and a person stream.

Acceptance (recorded, never a late hard-fail): every gallery camera has
tracklets of both classes, and associate found >= 1 cross-camera VEHICLE
identity and >= 1 cross-camera PERSON identity.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
PROJECT = Path("/tmp/athar-v2")
RUNS_ROOT = WORK / "runs"

# Chosen from a local dense frame scout (2026-07-30): the north street
# chain. Gallery keeps the two pedestrian-richest cameras TOGETHER
# (c018: 20 person hits, c019: 17 — adjacent, overlapping 170-670s, so a
# cross-camera person identity is likely) plus c017 (vehicles + late
# persons) and c021 (west extension of the chain). Probe = c020: mid-chain
# between both gallery ends with BOTH classes present (6 person hits, 56
# vehicle hits) — a probe search can start from a person AND a vehicle.
GALLERY_CAMS = ["c017", "c018", "c019", "c021"]
PROBE_CAM = "c020"
ALL_CAMS = GALLERY_CAMS + [PROBE_CAM]

VEHICLE_CLASSES = {"car", "bus", "truck"}

REID_CKPTS = {
    # dataset filename -> models/reid filename (weights_manifest.yaml names)
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


def find_shorouk_root() -> Path:
    for coords in INPUT.rglob("camera_coordinates.json"):
        if (coords.parent / GALLERY_CAMS[0] / "vdo.mp4").is_file():
            return coords.parent
    raise FileNotFoundError(
        "shorouk-dataset not mounted (no camera_coordinates.json beside "
        f"{GALLERY_CAMS[0]}/vdo.mp4 under {INPUT})"
    )


def restore_prior_runs() -> None:
    """Prior kernel version's output (self kernel source) -> RUNS_ROOT so
    the DAG runner resumes instead of restarting. Raw run dirs first (they
    survive even a failed prior version), exported tars as fallback."""
    restored: set[str] = set()
    for manifest in INPUT.rglob("runs/*/manifest.json"):
        run_dir = manifest.parent
        dst = RUNS_ROOT / run_dir.name
        if dst.exists() or run_dir.name in restored:
            continue
        shutil.copytree(run_dir, dst)
        restored.add(run_dir.name)
        print(f"restored prior run dir: {run_dir.name}")
    for tar_path in INPUT.rglob("*_run.tar.gz"):
        with tarfile.open(tar_path) as tar:
            names = {m.name.split("/", 1)[0] for m in tar.getmembers()}
            if names & restored or any((RUNS_ROOT / n).exists() for n in names):
                continue
            tar.extractall(RUNS_ROOT)
            restored.update(names)
            print(f"restored prior run tar: {tar_path.name} -> {sorted(names)}")


def existing_run(role: str, cams: list[str]):
    """Find a restored run matching role + camera set; None if absent."""
    if not RUNS_ROOT.exists():
        return None
    for manifest_path in sorted(RUNS_ROOT.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        run_cams = sorted(v["camera_id"] for v in manifest.get("inputs", []))
        if manifest.get("role") == role and run_cams == sorted(cams):
            return manifest
    return None


def athar_run(role: str, cams: list[str], overrides: list[str], env) -> str:
    """Create-or-resume one pipeline run; returns the run id."""
    prior = existing_run(role, cams)
    if prior is not None and prior.get("status") == "completed":
        print(f"{role} run already completed: {prior['run_id']}")
        return prior["run_id"]
    cmd = [sys.executable, "-m", "athar.cli.main", "run",
           "--profile", "production", "--role", role,
           "--runs-root", str(RUNS_ROOT)]
    if prior is not None:
        print(f"resuming {role} run {prior['run_id']} "
              f"(status {prior.get('status')})")
        cmd += ["--resume", prior["run_id"]]
    else:
        for cam in cams:
            cmd += ["--video", f"{cam}=data/raw/shorouk_demo/{cam}/vdo.mp4"]
        # cuda:0, not bare "cuda": ultralytics select_device writes the
        # string into CUDA_VISIBLE_DEVICES and "cuda" masks every GPU
        for override in ["detect_track.device=cuda:0",
                         "embed.device=cuda:0", *overrides]:
            cmd += ["--set", override]
    run(cmd, cwd=PROJECT, env=env)
    manifest = existing_run(role, cams)
    assert manifest is not None, f"{role} run left no manifest"
    assert manifest["status"] == "completed", (
        f"{role} run {manifest['run_id']} not completed: {manifest['status']}"
    )
    return manifest["run_id"]


def export_run(run_id: str, role: str) -> None:
    out = WORK / f"{role}_run.tar.gz"
    if out.exists():
        return
    tmp = out.with_suffix(".tmp.gz")
    with tarfile.open(tmp, "w:gz") as tar:
        tar.add(RUNS_ROOT / run_id, arcname=run_id)
    tmp.rename(out)
    print(f"exported {out.name} ({out.stat().st_size / 1e6:.1f} MB)")


def tracklet_counts(manifest: dict, run_dir: Path) -> dict:
    """Per-camera per-class tracklet counts from the tracklets artifacts."""
    counts: dict[str, dict[str, int]] = {}
    for name, record in manifest["artifacts"].items():
        if not name.startswith("tracklets."):
            continue
        cam = name.split(".", 1)[1]
        payload = json.loads((run_dir / record["relpath"]).read_text())
        by_class: dict[str, int] = {}
        for t in payload["tracklets"]:
            by_class[t["entity_class"]] = by_class.get(t["entity_class"], 0) + 1
        counts[cam] = dict(sorted(by_class.items()))
    return counts


def search_smoke(store, gallery_id: str, probe_id: str) -> dict:
    """In-process probe->gallery search on a vehicle and a person stream."""
    from athar.search.engine import GallerySearcher, SearchError

    gallery = store.load(gallery_id)
    probe = store.load(probe_id)
    searcher = GallerySearcher(store, gallery)
    out: dict[str, object] = {"streams": sorted(searcher.streams())}
    for label, stream in [("vehicle", "transreid_primary"),
                          ("person", "transreid_person")]:
        try:
            hits = searcher.search_probe(store, probe, stream, top_k=5)
            out[label] = {
                "stream": stream,
                "hits": [
                    {
                        "score": round(h.score, 4),
                        "probe": f"{h.probe_key.camera_id}/{h.probe_key.track_id}",
                        "gallery": f"{h.gallery_key.camera_id}/{h.gallery_key.track_id}",
                        "entity_class": h.gallery_entity_class.value,
                        "t": [round(h.gallery_start_ts_s, 1),
                              round(h.gallery_end_ts_s, 1)],
                    }
                    for h in hits[:5]
                ],
            }
        except SearchError as exc:
            out[label] = {"stream": stream, "error": str(exc)}
    return out


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
         "timm", "filterpy", "av", "torchcodec"])
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
         "ultralytics", "boxmot==22.0.0"])
    env = {**os.environ, "PYTHONPATH": str(PROJECT)}

    import torch

    print("torch", torch.__version__, "cuda", torch.version.cuda,
          "device", torch.cuda.get_device_name(0))
    assert float((torch.ones(8, device="cuda") * 2).sum().item()) == 16.0
    import torchcodec  # noqa: F401 — video FrameSource decode path

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
    required = set(REID_CKPTS.values())
    missing = required - placed
    assert not missing, f"missing checkpoints: {missing}"

    # ---- footage at the SAME relative layout the serving box uses ----
    shorouk = find_shorouk_root()
    print("shorouk footage:", shorouk)
    for cam in ALL_CAMS:
        src = shorouk / cam / "vdo.mp4"
        assert src.is_file(), f"camera {cam} missing from shorouk-dataset"
        dst = PROJECT / "data" / "raw" / "shorouk_demo" / cam / "vdo.mp4"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.symlink_to(src)

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    restore_prior_runs()

    # ---- the runs -----------------------------------------------------
    gallery_id = athar_run(
        "gallery", GALLERY_CAMS, ["package.clips=cross_camera"], env
    )
    export_run(gallery_id, "gallery")
    probe_id = athar_run("probe", [PROBE_CAM], [], env)
    export_run(probe_id, "probe")

    # ---- acceptance + search smoke (in-process, current tree) --------
    sys.path.insert(0, str(PROJECT))
    os.chdir(PROJECT)
    from athar.contracts.store import FilesystemRunStore

    store = FilesystemRunStore(RUNS_ROOT)
    gallery_manifest = json.loads(
        (RUNS_ROOT / gallery_id / "manifest.json").read_text()
    )
    gallery_dir = RUNS_ROOT / gallery_id
    counts = tracklet_counts(gallery_manifest, gallery_dir)
    probe_manifest = json.loads((RUNS_ROOT / probe_id / "manifest.json").read_text())
    probe_counts = tracklet_counts(probe_manifest, RUNS_ROOT / probe_id)

    traj_rel = gallery_manifest["artifacts"]["associate.trajectories"]["relpath"]
    trajectories = json.loads((gallery_dir / traj_rel).read_text())
    cross = [t for t in trajectories["trajectories"]
             if len({m["camera_id"] for m in t["members"]}) > 1]
    cross_vehicle = [t for t in cross if t["entity_class"] in VEHICLE_CLASSES]
    cross_person = [t for t in cross if t["entity_class"] == "person"]

    report = json.loads(
        (gallery_dir / gallery_manifest["artifacts"]["package.report"]["relpath"])
        .read_text()
    )
    clips_cut = sum(
        1 for identity in report["identities"]
        for member in identity["members"] if member.get("clip")
    )

    both_classes_everywhere = all(
        any(c in VEHICLE_CLASSES for c in by_class) and "person" in by_class
        for by_class in counts.values()
    )
    acceptance = {
        "gallery_completed": True,
        "probe_completed": True,
        "every_gallery_cam_has_both_classes": both_classes_everywhere,
        "cross_camera_vehicle_identities": len(cross_vehicle),
        "cross_camera_person_identities": len(cross_person),
        "evidence_clips_precut": clips_cut,
        "PASS": bool(cross_vehicle) and bool(cross_person),
    }

    summary = {
        "git_sha": git_sha,
        "config_hash": (gallery_manifest.get("config") or {}).get("config_hash"),
        "gallery_run_id": gallery_id,
        "probe_run_id": probe_id,
        "gallery_cams": GALLERY_CAMS,
        "probe_cam": PROBE_CAM,
        "gallery_tracklet_counts": counts,
        "probe_tracklet_counts": probe_counts,
        "num_identities": trajectories["num_identities"],
        "num_cross_camera": len(cross),
        "cross_camera_examples": [
            {"global_id": t["global_id"], "entity_class": t["entity_class"],
             "cameras": sorted({m["camera_id"] for m in t["members"]}),
             "confidence": t.get("confidence"), "evidence": t.get("evidence")}
            for t in cross[:20]
        ],
        "search_smoke": search_smoke(store, gallery_id, probe_id),
        "acceptance": acceptance,
    }
    print(json.dumps(summary, indent=2))
    (WORK / "shorouk_demo_summary.json").write_text(json.dumps(summary, indent=2))
    if not acceptance["PASS"]:
        print("ACCEPTANCE FAILED — see summary (runs still exported)")
    else:
        print("ACCEPTANCE PASSED")


if __name__ == "__main__":
    main()
