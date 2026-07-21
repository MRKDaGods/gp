"""index stage: run-level searchable gallery from the embed streams.

Builds, from the per-camera embed artifacts:

- ``index.catalog`` — SQLite tracklet catalog (ported v1 MetadataStore):
  one row per tracklet with a run-global ``index_id``, camera, class,
  scene-time span, and the HSV histogram blob when that stream exists.
- ``index.<stream>`` — one FAISS exact-IP index per appearance stream
  (ported v1 FAISSIndex), plus a ``.rows.json`` sidecar mapping FAISS row
  position → catalog ``index_id`` (FAISS binaries don't persist id maps).

The catalog row order is deterministic: cameras in manifest-input order,
then ascending track id — so index_ids are stable across re-runs.

This is what makes a GALLERY run searchable (D5 flagship workflow): a
probe's stream vectors are queried against ``index.<stream>``; hits join
back through the catalog to TrackKeys.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import numpy as np

from athar.contracts.manifest import ArtifactRecord

if TYPE_CHECKING:
    from athar.pipeline.runner import StageContext

logger = logging.getLogger(__name__)

INDEX_SCHEMA_VERSION = 1
HSV_STREAM = "hsv"


class IndexStage:
    name = "index"

    def is_complete(self, ctx: "StageContext") -> bool:
        return "index.catalog" in ctx.manifest.artifacts

    def run(self, ctx: "StageContext") -> None:
        from athar.components.indexing.faiss_index import FAISSIndex
        from athar.components.indexing.metadata_store import MetadataStore

        summary = json.loads(
            ctx.artifact_path("embed.summary").read_text(encoding="utf-8")
        )
        streams: dict[str, dict] = summary["streams"]
        if not streams:
            raise ValueError("index: embed.summary lists no streams")

        # ---- deterministic catalog order --------------------------------
        entries: list[dict] = []  # index_id = position in this list
        for video in ctx.manifest.inputs:
            cam = video.camera_id
            name = f"tracklets.{cam}"
            if name not in ctx.manifest.artifacts:
                continue
            payload = json.loads(ctx.artifact_path(name).read_text(encoding="utf-8"))
            for t in sorted(payload["tracklets"], key=lambda t: t["key"]["track_id"]):
                entries.append(t)
        if not entries:
            raise ValueError("index: no tracklets to catalog")
        position = {
            (t["key"]["camera_id"], t["key"]["track_id"]): i for i, t in enumerate(entries)
        }

        # ---- load stream rows, joined on (camera, track_id) -------------
        stream_rows: dict[str, dict[int, np.ndarray]] = {s: {} for s in streams}
        for stream_name, info in streams.items():
            for artifact_name in info["artifacts"]:
                cam = artifact_name.split(".")[1]
                data = np.load(ctx.artifact_path(artifact_name))
                for row, track_id in zip(data["embeddings"], data["track_ids"]):
                    index_id = position.get((cam, int(track_id)))
                    if index_id is None:
                        logger.warning(
                            "index: %s row for unknown tracklet %s/%s",
                            stream_name, cam, int(track_id),
                        )
                        continue
                    stream_rows[stream_name][index_id] = np.asarray(row, dtype=np.float32)

        # ---- catalog (with HSV blobs when present) ----------------------
        from athar.components.adapters.detection import ENTITY_TO_COCO
        from athar.core.types import EntityClass

        catalog_relpath = "indexes/catalog.db"
        catalog_path = ctx.run_dir / catalog_relpath
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        if catalog_path.exists():
            catalog_path.unlink()
        store = MetadataStore(catalog_path)
        try:
            hsv = stream_rows.get(HSV_STREAM, {})
            for index_id, t in enumerate(entries):
                key = t["key"]
                store.insert_tracklet(
                    index_id=index_id,
                    track_id=key["track_id"],
                    camera_id=key["camera_id"],
                    class_id=ENTITY_TO_COCO[EntityClass(t["entity_class"])],
                    start_time=t["start_ts_scene_s"],
                    end_time=t["end_ts_scene_s"],
                    num_frames=t["observation_count"],
                    hsv_histogram=hsv.get(index_id),
                )
        finally:
            store.close()

        # ---- FAISS per appearance stream --------------------------------
        index_artifacts = []
        for stream_name, rows in stream_rows.items():
            if stream_name == HSV_STREAM or not rows:
                continue
            index_ids = sorted(rows)
            matrix = np.stack([rows[i] for i in index_ids])
            faiss_index = FAISSIndex(index_type="flat_ip")
            faiss_index.build(matrix, ids=index_ids)
            relpath = f"indexes/{stream_name}.faiss"
            faiss_index.save(ctx.run_dir / relpath)
            rows_relpath = f"indexes/{stream_name}.rows.json"
            (ctx.run_dir / rows_relpath).write_text(json.dumps(index_ids), encoding="utf-8")
            name = f"index.{stream_name}"
            ctx.register_artifact(
                ArtifactRecord(
                    name=name,
                    relpath=relpath,
                    schema_version=INDEX_SCHEMA_VERSION,
                    producer="index/faiss_flat_ip",
                    row_count=len(index_ids),
                )
            )
            ctx.register_artifact(
                ArtifactRecord(
                    name=f"index.{stream_name}.rows",
                    relpath=rows_relpath,
                    schema_version=INDEX_SCHEMA_VERSION,
                    producer="index/faiss_flat_ip",
                    row_count=len(index_ids),
                )
            )
            index_artifacts.append(name)

        ctx.register_artifact(
            ArtifactRecord(
                name="index.catalog",
                relpath=catalog_relpath,
                schema_version=INDEX_SCHEMA_VERSION,
                producer="index/metadata_store",
                row_count=len(entries),
            )
        )
        logger.info(
            "index: cataloged %d tracklets, built %d FAISS stream indexes",
            len(entries), len(index_artifacts),
        )
