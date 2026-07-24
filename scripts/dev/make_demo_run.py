"""DEV FIXTURE: build a demo run over real Shorouk footage for UI work.

Writes a completed-run directory (manifest, tracklets, trajectories,
package.report, thumbnails) with HAND-AUTHORED identities — no detector or
ReID ran. Its purpose is exercising the run-detail UI (cross-camera
timeline, evidence clips, map view) against evidence videos that exist on
this box, which a full pipeline run cannot provide locally (heavy compute
is Kaggle-only per workflow rules, and Shorouk footage stays on premises).

The run is unmistakably labeled: profile_name="demo-fixture" and every
artifact producer="dev-fixture". Delete the run dir to remove it.

Usage (repo root):  python scripts/dev/make_demo_run.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from athar.contracts.manifest import (  # noqa: E402
    ArtifactRecord,
    RunManifest,
    RunRole,
    RunStatus,
    VideoInput,
)
from athar.contracts.store import FilesystemRunStore  # noqa: E402
from athar.core.timebase import CameraTimeBase, SceneClock, TimeBaseSource  # noqa: E402
from athar.pipeline.stages.package import crop_with_padding  # noqa: E402

RUN_ID = "run-demo-shorouk"
SHOROUK = REPO / "data" / "raw" / "shorouk"

# (camera, track_id, entity_class, start_s, end_s) — spans in the first
# minute; identity 0 is the cross-camera person c020 -> c021.
TRACKS = {
    "c020": [(1, "person", 5.0, 20.0), (2, "car", 30.0, 48.0)],
    "c021": [(1, "person", 25.0, 40.0), (2, "person", 8.0, 14.0)],
}
TRAJECTORIES = [
    {
        "global_id": 0, "entity_class": "person",
        "members": [("c020", 1), ("c021", 1)],
        "confidence": 0.82,
        "evidence": {"appearance": 0.71, "hsv": 0.64, "spatiotemporal": 0.88},
    },
    {"global_id": 1, "entity_class": "car", "members": [("c020", 2)],
     "confidence": None, "evidence": {}},
    {"global_id": 2, "entity_class": "person", "members": [("c021", 2)],
     "confidence": None, "evidence": {}},
]


def probe_video(path: Path) -> dict:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return {
            "fps": float(stream.average_rate),
            "duration_s": float(container.duration / av.time_base),
            "width": stream.width,
            "height": stream.height,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def write_thumb(video: Path, mid_s: float, fps: float, out: Path) -> None:
    """Center-region crop of the span's middle frame (real imagery; the
    region is arbitrary — no detection ran)."""
    import cv2

    from athar.components.framesources.video import create_video_source

    source = create_video_source("thumb", video, indices=[round(mid_s * fps)])
    batch = next(source.batches(1))
    image = batch.images()[0]
    h, w = image.shape[:2]
    bbox = {"x1": w * 0.3, "y1": h * 0.25, "x2": w * 0.7, "y2": h * 0.85}
    crop = crop_with_padding(image, bbox, padding_ratio=0.0, thumb_size=256)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main() -> None:
    store = FilesystemRunStore(REPO / "data" / "runs")
    run_dir = store.run_dir(RUN_ID)
    if run_dir.exists():
        shutil.rmtree(run_dir)

    manifest = RunManifest(
        run_id=RUN_ID, role=RunRole.GALLERY, profile_name="demo-fixture"
    )
    manifest.status = RunStatus.COMPLETED
    (run_dir / "tracklets").mkdir(parents=True, exist_ok=True)

    tracklet_lookup: dict[tuple[str, int], dict] = {}
    thumb_index: dict[str, dict[str, str]] = {}
    for cam, tracks in TRACKS.items():
        video = SHOROUK / cam / "vdo.mp4"
        meta = probe_video(video)
        print(f"{cam}: {meta['duration_s']:.1f}s @ {meta['fps']:.2f}fps, hashing...")
        manifest.inputs.append(
            VideoInput(
                camera_id=cam, original_path=str(video),
                sha256=sha256_file(video), **meta,
            )
        )
        manifest.timebase.cameras[cam] = CameraTimeBase(
            camera_id=cam, fps=meta["fps"],
            source=TimeBaseSource.SYNCHRONIZED, confidence=1.0,
        )
        payload = {
            "schema_version": 1, "camera_id": cam, "observations": {},
            "tracklets": [],
        }
        thumb_index[cam] = {}
        for track_id, entity_class, start_s, end_s in tracks:
            info = {
                "key": {"run_id": RUN_ID, "camera_id": cam, "track_id": track_id},
                "entity_class": entity_class,
                "start_ts_scene_s": start_s, "end_ts_scene_s": end_s,
                "observation_count": round((end_s - start_s) * meta["fps"]),
                "mean_confidence": 0.9,
            }
            payload["tracklets"].append(info)
            tracklet_lookup[(cam, track_id)] = info
            rel = f"thumbs/{cam}/{track_id}.jpg"
            write_thumb(video, (start_s + end_s) / 2, meta["fps"], run_dir / rel)
            thumb_index[cam][str(track_id)] = rel
        (run_dir / "tracklets" / f"{cam}.json").write_text(
            json.dumps(payload, indent=2), "utf-8"
        )
        manifest.register_artifact(ArtifactRecord(
            name=f"tracklets.{cam}", relpath=f"tracklets/{cam}.json",
            schema_version=1, producer="dev-fixture", row_count=len(tracks),
        ))

    (run_dir / "thumbs" / "index.json").write_text(
        json.dumps({"schema_version": 1, "thumbnails": thumb_index}), "utf-8"
    )
    manifest.register_artifact(ArtifactRecord(
        name="package.thumbnails", relpath="thumbs/index.json",
        schema_version=1, producer="dev-fixture",
        row_count=sum(len(v) for v in thumb_index.values()),
    ))

    trajectories = {
        "schema_version": 1, "primary_stream": "none", "streams": [],
        "num_tracklets": len(tracklet_lookup),
        "num_identities": len(TRAJECTORIES),
        "trajectories": [
            {
                "global_id": t["global_id"], "entity_class": t["entity_class"],
                "confidence": t["confidence"], "evidence": t["evidence"],
                "members": [
                    {"run_id": RUN_ID, "camera_id": cam, "track_id": tid}
                    for cam, tid in t["members"]
                ],
            }
            for t in TRAJECTORIES
        ],
    }
    (run_dir / "trajectories.json").write_text(
        json.dumps(trajectories, indent=2), "utf-8"
    )
    manifest.register_artifact(ArtifactRecord(
        name="associate.trajectories", relpath="trajectories.json",
        schema_version=1, producer="dev-fixture", row_count=len(TRAJECTORIES),
    ))

    report = {
        "schema_version": 1,
        "run": {
            "run_id": RUN_ID, "role": "gallery", "profile": "demo-fixture",
            "config_hash": None, "created_at": str(manifest.created_at),
        },
        "evidence": [
            {"camera_id": v.camera_id, "original_path": v.original_path,
             "sha256": v.sha256, "duration_s": v.duration_s, "fps": v.fps}
            for v in manifest.inputs
        ],
        "identities": [
            {
                "global_id": t["global_id"], "entity_class": t["entity_class"],
                "confidence": t["confidence"], "evidence": t["evidence"],
                "cross_camera": len({cam for cam, _ in t["members"]}) > 1,
                "members": [
                    {
                        "camera_id": cam, "track_id": tid,
                        "start_ts_scene_s":
                            tracklet_lookup[(cam, tid)]["start_ts_scene_s"],
                        "end_ts_scene_s":
                            tracklet_lookup[(cam, tid)]["end_ts_scene_s"],
                        "thumbnail": thumb_index[cam].get(str(tid)),
                        "clip": None,
                    }
                    for cam, tid in t["members"]
                ],
            }
            for t in TRAJECTORIES
        ],
    }
    (run_dir / "report_inputs.json").write_text(json.dumps(report, indent=2), "utf-8")
    manifest.register_artifact(ArtifactRecord(
        name="package.report", relpath="report_inputs.json",
        schema_version=1, producer="dev-fixture", row_count=len(TRAJECTORIES),
    ))

    store.save(manifest)
    print(f"demo run written: {run_dir}")


if __name__ == "__main__":
    main()
