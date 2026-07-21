"""package stage: reviewable evidence out of a completed run (thin v1).

Produces the two artifacts every downstream reviewer (backend, report
export, case UI) starts from:

- ``package.thumbnails`` — one best-crop JPEG per tracklet
  (``thumbs/<cam>/<track_id>.jpg`` + an index JSON). "Best" = the
  highest-confidence observation (middle frame on ties) — decoded
  sparsely through the profile FrameSource, never a full pass.
- ``package.report`` — ``report_inputs.json``: the chain-of-custody
  skeleton (evidence sha256 -> config hash -> identities) with one entry
  per trajectory, per-member time spans and thumbnail paths.

Evidence CLIPS (short video segments per identity) are deliberately not
here yet — they need the transcode plumbing planned for the serving
phase; the report schema already reserves the field.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from athar.pipeline.runner import StageContext

logger = logging.getLogger(__name__)

PACKAGE_SCHEMA_VERSION = 1


def best_observation(observations: list[dict]) -> dict:
    """Highest confidence wins; the observation closest to the tracklet's
    middle breaks ties (stable for constant-confidence trackers)."""
    if not observations:
        raise ValueError("tracklet has no observations")
    mid = len(observations) // 2
    return max(
        enumerate(observations),
        key=lambda pair: (pair[1].get("confidence", 0.0), -abs(pair[0] - mid)),
    )[1]


def crop_with_padding(
    image, bbox: dict, padding_ratio: float, thumb_size: int
):
    """Padded crop resized so its long side is ``thumb_size`` (never
    upscaled)."""
    import cv2

    h, w = image.shape[:2]
    bw = bbox["x2"] - bbox["x1"]
    bh = bbox["y2"] - bbox["y1"]
    pad_x = bw * padding_ratio
    pad_y = bh * padding_ratio
    x1 = max(0, int(bbox["x1"] - pad_x))
    y1 = max(0, int(bbox["y1"] - pad_y))
    x2 = min(w, int(bbox["x2"] + pad_x))
    y2 = min(h, int(bbox["y2"] + pad_y))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    long_side = max(crop.shape[:2])
    if long_side > thumb_size:
        scale = thumb_size / long_side
        crop = cv2.resize(
            crop, (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale)))
        )
    return crop


class PackageStage:
    name = "package"

    def is_complete(self, ctx: "StageContext") -> bool:
        return "package.report" in ctx.manifest.artifacts

    def run(self, ctx: "StageContext") -> None:
        import cv2

        from athar.contracts.manifest import ArtifactRecord

        config = ctx.manifest.config
        thumb_size = int(config.get("package.thumb_size", 256))
        padding_ratio = float(config.get("package.padding_ratio", 0.15))
        decode_batch = int(config.get("package.decode_batch", 8))
        jpeg_quality = int(config.get("package.jpeg_quality", 90))

        checkpoint = ctx.load_checkpoint() or {"completed_cameras": []}
        done_cameras = set(checkpoint["completed_cameras"])
        tracklet_info: dict[tuple[str, int], dict] = {}
        thumb_index: dict[str, dict[str, str]] = {}

        total = len(ctx.manifest.inputs)
        for cam_pos, video in enumerate(ctx.manifest.inputs):
            cam = video.camera_id
            name = f"tracklets.{cam}"
            if name not in ctx.manifest.artifacts:
                continue
            payload = json.loads(
                ctx.artifact_path(name).read_text(encoding="utf-8")
            )
            for t in payload["tracklets"]:
                tracklet_info[(cam, t["key"]["track_id"])] = t

            if cam in done_cameras:
                thumb_index[cam] = self._existing_thumbs(ctx, cam, payload)
                continue

            picks: dict[int, tuple[int, dict]] = {}
            for track_id_str, observations in payload["observations"].items():
                obs = best_observation(observations)
                picks[int(track_id_str)] = (int(obs["frame_index"]), obs["bbox"])
            thumb_index[cam] = {}
            if picks:
                thumb_dir = ctx.run_dir / "thumbs" / cam
                thumb_dir.mkdir(parents=True, exist_ok=True)
                frames_needed = sorted({f for f, _ in picks.values()})
                source = ctx.registry.create(
                    "frame_source",
                    ctx.profile.frame_source.name,
                    camera_id=cam,
                    path=video.original_path,
                    indices=frames_needed,
                    **ctx.profile.frame_source.config,
                )
                decoded: dict[int, Any] = {}
                for batch in source.batches(decode_batch):
                    for i, frame_index in enumerate(batch.frame_indices):
                        decoded[int(frame_index)] = batch.images()[i]
                for track_id, (frame_index, bbox) in sorted(picks.items()):
                    image = decoded.get(frame_index)
                    if image is None:
                        continue
                    crop = crop_with_padding(image, bbox, padding_ratio, thumb_size)
                    if crop is None:
                        logger.warning(
                            "package: degenerate crop for %s/%s", cam, track_id
                        )
                        continue
                    rel = f"thumbs/{cam}/{track_id}.jpg"
                    cv2.imwrite(
                        str(ctx.run_dir / rel), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                    )
                    thumb_index[cam][str(track_id)] = rel

            done_cameras.add(cam)
            ctx.save_checkpoint({"completed_cameras": sorted(done_cameras)})
            ctx.progress(done=cam_pos + 1, total=total + 1)

        index_rel = "thumbs/index.json"
        (ctx.run_dir / "thumbs").mkdir(exist_ok=True)
        (ctx.run_dir / index_rel).write_text(
            json.dumps({"schema_version": PACKAGE_SCHEMA_VERSION, "thumbnails": thumb_index}),
            encoding="utf-8",
        )
        ctx.register_artifact(
            ArtifactRecord(
                name="package.thumbnails",
                relpath=index_rel,
                schema_version=PACKAGE_SCHEMA_VERSION,
                producer="package",
                row_count=sum(len(v) for v in thumb_index.values()),
            )
        )

        report = self._build_report(ctx, tracklet_info, thumb_index)
        report_rel = "report_inputs.json"
        (ctx.run_dir / report_rel).write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        ctx.register_artifact(
            ArtifactRecord(
                name="package.report",
                relpath=report_rel,
                schema_version=PACKAGE_SCHEMA_VERSION,
                producer="package",
                row_count=len(report["identities"]),
            )
        )
        ctx.clear_checkpoint()
        ctx.progress(done=total + 1, total=total + 1)
        logger.info(
            "package: %d thumbnails, %d identities in report",
            sum(len(v) for v in thumb_index.values()), len(report["identities"]),
        )

    def _existing_thumbs(
        self, ctx: "StageContext", cam: str, payload: dict
    ) -> dict[str, str]:
        out = {}
        for track_id_str in payload["observations"]:
            rel = f"thumbs/{cam}/{int(track_id_str)}.jpg"
            if (ctx.run_dir / rel).is_file():
                out[track_id_str] = rel
        return out

    def _build_report(
        self,
        ctx: "StageContext",
        tracklet_info: dict[tuple[str, int], dict],
        thumb_index: dict[str, dict[str, str]],
    ) -> dict:
        manifest = ctx.manifest
        identities: list[dict] = []
        if "associate.trajectories" in manifest.artifacts:
            payload = json.loads(
                ctx.artifact_path("associate.trajectories").read_text(encoding="utf-8")
            )
            for traj in payload["trajectories"]:
                members = []
                for member in traj["members"]:
                    cam = member["camera_id"]
                    track_id = int(member["track_id"])
                    info = tracklet_info.get((cam, track_id), {})
                    members.append(
                        {
                            "camera_id": cam,
                            "track_id": track_id,
                            "start_ts_scene_s": info.get("start_ts_scene_s"),
                            "end_ts_scene_s": info.get("end_ts_scene_s"),
                            "thumbnail": thumb_index.get(cam, {}).get(str(track_id)),
                            "clip": None,  # reserved: evidence clips land later
                        }
                    )
                identities.append(
                    {
                        "global_id": traj["global_id"],
                        "entity_class": traj["entity_class"],
                        "confidence": traj.get("confidence"),
                        "evidence": traj.get("evidence", {}),
                        "cross_camera": len({m["camera_id"] for m in members}) > 1,
                        "members": members,
                    }
                )
        return {
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "run": {
                "run_id": manifest.run_id,
                "role": manifest.role.value,
                "profile": manifest.profile_name,
                "config_hash": manifest.config.config_hash if manifest.config else None,
                "created_at": str(manifest.created_at),
            },
            "evidence": [
                {
                    "camera_id": v.camera_id,
                    "original_path": v.original_path,
                    "sha256": v.sha256,
                    "duration_s": v.duration_s,
                    "fps": v.fps,
                }
                for v in manifest.inputs
            ],
            "identities": identities,
        }
