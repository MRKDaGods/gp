"""embed stage: tracklets → quality-scored crops → per-branch embedding streams.

Per camera: read the ``tracklets.<cam>`` artifact, precompute each
tracklet's crop-candidate frames (the same deterministic sampling the v1
CropExtractor applies), decode ONLY those frames via the profile's
FrameSource (explicit-indices plan — torchcodec random access), then run
the v1 kernels verbatim: ``extract_crops_from_frames`` → per-branch
``embed_tracklet`` (flip-augment + softmax-quality attention pooling).

Artifacts: one ``embeddings.<cam>.<stream>`` npz per camera per stream
(keys: ``embeddings`` float32 (T, D), ``track_ids`` int64 (T,)), plus a
final ``embed.summary`` json recording every stream's dim + artifact list
(the stage-completion marker and the input map for index/associate).

Resume: camera-level checkpoint, same pattern as detect_track. Tracklet
processing is chunked (``embed.tracklet_chunk``) so at most one chunk's
union of frames is ever in memory.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from athar.components.adapters.detection import ENTITY_TO_COCO
from athar.components.protocols import ComponentKindName
from athar.contracts.manifest import ArtifactRecord
from athar.core.data_models import Tracklet as V1Tracklet
from athar.core.data_models import TrackletFrame
from athar.core.types import EntityClass

if TYPE_CHECKING:
    from athar.pipeline.runner import StageContext

logger = logging.getLogger(__name__)

EMBED_SCHEMA_VERSION = 1


def candidate_frame_ids(n_frames: int, frame_ids: list[int], samples_per_tracklet: int) -> list[int]:
    """The v1 CropExtractor candidate plan, precomputed for decode.

    Mirrors extract_crops_from_frames: n_candidates = min(n, 2*samples),
    stratified by ``int(i * n/n_candidates)``. The kernel re-derives the
    same set internally and skips frames missing from the dict, so
    supplying exactly this set is behavior-identical to supplying all.
    """
    n_candidates = min(n_frames, samples_per_tracklet * 2)
    if n_frames <= n_candidates:
        picked = range(n_frames)
    else:
        step = n_frames / n_candidates
        picked = (int(i * step) for i in range(n_candidates))
    return [frame_ids[i] for i in picked]


class EmbedStage:
    name = "embed"

    def is_complete(self, ctx: "StageContext") -> bool:
        return "embed.summary" in ctx.manifest.artifacts

    def run(self, ctx: "StageContext") -> None:
        profile = ctx.profile
        config = ctx.manifest.config
        samples = int(config.get("embed.samples_per_tracklet", 16))
        chunk_size = int(config.get("embed.tracklet_chunk", 4))
        decode_batch = int(config.get("embed.decode_batch", 16))
        device = str(config.get("embed.device", "cpu"))

        from athar.components.embedders.crop_extractor import CropExtractor

        extractor = CropExtractor(
            min_area=float(config.get("embed.min_area", 500)),
            padding_ratio=float(config.get("embed.padding_ratio", 0.1)),
            samples_per_tracklet=samples,
            min_quality=float(config.get("embed.min_quality", 0.05)),
        )

        state = ctx.load_checkpoint() or {"completed_cameras": []}
        completed = set(state["completed_cameras"])
        summary_streams: dict[str, dict[str, Any]] = {}

        # one embedder instance per branch stream, shared across cameras
        branch_embedders: list[list[Any]] = [
            [
                ctx.registry.create(
                    ComponentKindName.EMBEDDER, spec.name, device=device, **spec.config
                )
                for spec in branch.embedders
            ]
            for branch in profile.branches
        ]

        for video in ctx.manifest.inputs:
            cam = video.camera_id
            artifact_name = f"tracklets.{cam}"
            if artifact_name not in ctx.manifest.artifacts:
                logger.warning("embed: no tracklets artifact for camera %s, skipping", cam)
                continue
            for branch_index, branch in enumerate(profile.branches):
                for embedder in branch_embedders[branch_index]:
                    summary_streams.setdefault(
                        embedder.stream_name,
                        {"dim": embedder.dim, "artifacts": []},
                    )
            if cam in completed:
                # already processed before a crash; artifacts are registered
                for record_name, record in ctx.manifest.artifacts.items():
                    prefix = f"embeddings.{cam}."
                    if record_name.startswith(prefix):
                        stream = record_name[len(prefix):]
                        if stream in summary_streams:
                            summary_streams[stream]["artifacts"].append(record_name)
                continue
            ctx.cancel.raise_if_cancelled()
            self._run_camera(
                ctx, video, extractor, branch_embedders, samples,
                chunk_size, decode_batch, summary_streams,
            )
            completed.add(cam)
            ctx.save_checkpoint({"completed_cameras": sorted(completed)})

        summary = {
            "schema_version": EMBED_SCHEMA_VERSION,
            "streams": summary_streams,
        }
        relpath = "embed_summary.json"
        out = ctx.run_dir / relpath
        out.write_text(json.dumps(summary), encoding="utf-8")
        ctx.register_artifact(
            ArtifactRecord(
                name="embed.summary",
                relpath=relpath,
                schema_version=EMBED_SCHEMA_VERSION,
                producer="embed",
                row_count=len(summary_streams),
            )
        )

    def _run_camera(
        self,
        ctx: "StageContext",
        video,
        extractor,
        branch_embedders: list[list[Any]],
        samples: int,
        chunk_size: int,
        decode_batch: int,
        summary_streams: dict[str, dict[str, Any]],
    ) -> None:
        profile = ctx.profile
        cam = video.camera_id
        payload = json.loads(
            ctx.artifact_path(f"tracklets.{cam}").read_text(encoding="utf-8")
        )

        # (branch_index, v1_tracklet) pairs, only for classes some branch tracks
        work: list[tuple[int, V1Tracklet]] = []
        for dumped in payload["tracklets"]:
            entity = EntityClass(dumped["entity_class"])
            branch_index = next(
                (i for i, b in enumerate(profile.branches) if entity in b.entity_classes),
                None,
            )
            if branch_index is None:
                continue
            track_id = dumped["key"]["track_id"]
            observations = payload["observations"].get(str(track_id), [])
            if not observations:
                continue
            frames = [
                TrackletFrame(
                    frame_id=o["frame_index"],
                    timestamp=o["ts_scene_s"],
                    bbox=(o["bbox"]["x1"], o["bbox"]["y1"], o["bbox"]["x2"], o["bbox"]["y2"]),
                    confidence=o["confidence"],
                )
                for o in observations
            ]
            work.append(
                (
                    branch_index,
                    V1Tracklet(
                        track_id=track_id,
                        camera_id=cam,
                        class_id=ENTITY_TO_COCO[entity],
                        class_name=entity.value,
                        frames=frames,
                    ),
                )
            )

        rows: dict[str, list[np.ndarray]] = {}
        row_ids: dict[str, list[int]] = {}
        done = 0
        for chunk_start in range(0, len(work), chunk_size):
            ctx.cancel.raise_if_cancelled()
            chunk = work[chunk_start : chunk_start + chunk_size]
            union: set[int] = set()
            for _, tracklet in chunk:
                union.update(
                    candidate_frame_ids(
                        len(tracklet.frames),
                        [f.frame_id for f in tracklet.frames],
                        samples,
                    )
                )
            frames_in_memory: dict[int, np.ndarray] = {}
            source = ctx.registry.create(
                ComponentKindName.FRAME_SOURCE,
                profile.frame_source.name,
                camera_id=cam,
                path=video.original_path,
                indices=sorted(union),
                **{k: v for k, v in profile.frame_source.config.items()
                   if k not in ("start", "stop", "step")},
            )
            for batch in source.batches(decode_batch):
                images = batch.images()
                for pos, frame_index in enumerate(batch.frame_indices):
                    frames_in_memory[frame_index] = images[pos]

            for branch_index, tracklet in chunk:
                scored = extractor.extract_crops_from_frames(tracklet, frames_in_memory)
                if not scored:
                    logger.warning(
                        "embed: tracklet %s/%s produced no usable crops", cam, tracklet.track_id
                    )
                    continue
                for embedder in branch_embedders[branch_index]:
                    vec = embedder.embed_tracklet(scored)
                    if vec is None:
                        continue
                    rows.setdefault(embedder.stream_name, []).append(
                        np.asarray(vec, dtype=np.float32)
                    )
                    row_ids.setdefault(embedder.stream_name, []).append(tracklet.track_id)
            frames_in_memory.clear()
            done += len(chunk)
            ctx.progress(done=done, total=len(work), camera_id=cam)

        for stream_name, stream_rows in rows.items():
            relpath = f"embeddings/{cam}.{stream_name}.npz"
            out = ctx.run_dir / relpath
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                out,
                embeddings=np.stack(stream_rows),
                track_ids=np.asarray(row_ids[stream_name], dtype=np.int64),
            )
            name = f"embeddings.{cam}.{stream_name}"
            ctx.register_artifact(
                ArtifactRecord(
                    name=name,
                    relpath=relpath,
                    schema_version=EMBED_SCHEMA_VERSION,
                    producer=f"embed/{stream_name}",
                    row_count=len(stream_rows),
                )
            )
            summary_streams[stream_name]["artifacts"].append(name)
