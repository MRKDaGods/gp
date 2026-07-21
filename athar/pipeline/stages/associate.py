"""associate stage: cross-camera identity clustering over the gallery.

Composes the ported v1 stage-4 kernels — FAISS candidate generation,
mutual-NN filtering, SpatioTemporalValidator gating, class-adaptive
combined similarity, GraphSolver clustering — into the v2 windowed
association engine. Offline runs use ONE window covering the whole run
(council design: bit-identical semantics to v1); the window loop is the
live-readiness seam, and ``associate.window_s != 0`` is refused until the
live path lands rather than silently mis-clustering.

Pairs are generated cross-camera only (intra-camera merges are a tracker
concern) and only within the same profile branch (person↔person,
vehicle↔vehicle; cross-entity person↔vehicle links are hypothesis edges
from InteractionEventDetectors, not appearance clusters — D7).

Output: ``associate.trajectories`` — one Trajectory per identity with
per-score-term evidence means (forensic explainability, D6).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import TYPE_CHECKING, Any

import numpy as np

from athar.components.adapters.detection import COCO_TO_ENTITY
from athar.contracts.manifest import ArtifactRecord
from athar.core.ids import TrackKey
from athar.core.types import EntityClass, Trajectory

if TYPE_CHECKING:
    from athar.pipeline.runner import StageContext

logger = logging.getLogger(__name__)

ASSOCIATE_SCHEMA_VERSION = 1


def config_subtree(config, prefix: str) -> dict[str, Any]:
    """Rebuild a nested dict from a ResolvedConfig's flat dot-keys."""
    out: dict[str, Any] = {}
    for key, value in config.values.items():
        if not key.startswith(prefix + "."):
            continue
        parts = key[len(prefix) + 1 :].split(".")
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


class AssociateStage:
    name = "associate"

    def is_complete(self, ctx: "StageContext") -> bool:
        return "associate.trajectories" in ctx.manifest.artifacts

    def run(self, ctx: "StageContext") -> None:
        import faiss

        from athar.components.associators.similarity import (
            compute_combined_similarity,
            compute_hsv_similarity,
            mutual_nearest_neighbor_filter,
        )
        from athar.components.associators.graph_solver import GraphSolver
        from athar.components.associators.spatial_temporal import SpatioTemporalValidator
        from athar.components.indexing.metadata_store import MetadataStore

        config = ctx.manifest.config
        window_s = float(config.get("associate.window_s", 0.0))
        if window_s != 0.0:
            raise NotImplementedError(
                "windowed association arrives with the live path; "
                "offline runs use associate.window_s=0 (one window)"
            )
        top_k = int(config.get("associate.top_k", 20))
        mutual_top_k = int(config.get("associate.mutual_top_k", 10))

        # ---- catalog rows in index_id order ------------------------------
        catalog = MetadataStore(ctx.artifact_path("index.catalog"))
        try:
            rows = sorted(catalog.get_all(), key=lambda r: r["index_id"])
            hsv_blobs = {
                r["index_id"]: catalog.get_hsv_histogram(r["index_id"]) for r in rows
            }
        finally:
            catalog.close()
        if not rows:
            raise ValueError("associate: empty catalog")
        n = len(rows)
        camera_ids = [r["camera_id"] for r in rows]
        class_ids = [r["class_id"] for r in rows]
        start_times = [r["start_time"] for r in rows]
        end_times = [r["end_time"] for r in rows]
        num_frames = [r["num_frames"] for r in rows]

        hsv_dim = next(
            (b.shape[0] for b in hsv_blobs.values() if b is not None), 1
        )
        hsv_features = np.zeros((n, hsv_dim), dtype=np.float32)
        for index_id, blob in hsv_blobs.items():
            if blob is not None:
                hsv_features[index_id] = blob

        # branch id per row — pairs must share a branch
        branch_of: list[int | None] = []
        for cid in class_ids:
            entity = COCO_TO_ENTITY.get(cid)
            branch_of.append(
                next(
                    (
                        i
                        for i, b in enumerate(ctx.profile.branches)
                        if entity in b.entity_classes
                    ),
                    None,
                )
            )

        # ---- candidates: FAISS top-k per appearance stream ----------------
        # Each branch's appearance stream only indexes that branch's rows, so
        # sweeping every non-hsv stream gives every branch its own candidate
        # pool; with a single stream this is bit-identical to the v1 flow.
        summary = json.loads(ctx.artifact_path("embed.summary").read_text(encoding="utf-8"))
        configured = config.get("associate.primary_stream")
        streams = [configured] if configured else [
            s for s in summary["streams"] if s != "hsv"
        ]
        streams = [s for s in streams if f"index.{s}" in ctx.manifest.artifacts]
        if not streams:
            raise ValueError("associate: no appearance stream to associate on")

        candidates: dict[tuple[int, int], float] = {}
        for stream in streams:
            index = faiss.read_index(str(ctx.artifact_path(f"index.{stream}")))
            if index.ntotal == 0:
                continue
            row_map: list[int] = json.loads(
                ctx.artifact_path(f"index.{stream}.rows").read_text(encoding="utf-8")
            )
            vectors = index.reconstruct_n(0, index.ntotal)
            scores, positions = index.search(vectors, min(top_k + 1, index.ntotal))
            for qpos in range(index.ntotal):
                i = row_map[qpos]
                for score, rpos in zip(scores[qpos], positions[qpos]):
                    if rpos < 0 or rpos == qpos:
                        continue
                    j = row_map[int(rpos)]
                    if camera_ids[i] == camera_ids[j]:
                        continue  # cross-camera only
                    if branch_of[i] is None or branch_of[i] != branch_of[j]:
                        continue
                    pair = (min(i, j), max(i, j))
                    candidates[pair] = max(candidates.get(pair, -1.0), float(score))

        filtered = mutual_nearest_neighbor_filter(
            [(i, j, s) for (i, j), s in candidates.items()],
            top_k_per_query=mutual_top_k,
        )
        appearance_sim = {(i, j): s for i, j, s in filtered}
        ctx.progress(done=1, total=3)

        # ---- combined similarity + clustering (ported kernels) -----------
        st_validator = SpatioTemporalValidator(
            min_time_gap=float(config.get("associate.min_time_gap", 0.0)),
            max_time_gap=float(config.get("associate.max_time_gap", 300.0)),
            camera_transitions=config_subtree(config, "associate.camera_transitions") or None,
        )
        weights = config_subtree(config, "associate.weights")
        combined = compute_combined_similarity(
            appearance_sim=appearance_sim,
            hsv_features=hsv_features,
            start_times=start_times,
            end_times=end_times,
            camera_ids=camera_ids,
            class_ids=class_ids,
            st_validator=st_validator,
            weights=weights,
            num_frames=num_frames,
        )
        solver = GraphSolver(
            similarity_threshold=float(config.get("associate.similarity_threshold", 0.5)),
            algorithm=str(config.get("associate.algorithm", "connected_components")),
            bridge_prune_margin=float(config.get("associate.bridge_prune_margin", 0.0)),
            max_component_size=int(config.get("associate.max_component_size", 0)),
        )
        clusters = solver.solve(
            combined, n,
            camera_ids=camera_ids, start_times=start_times, end_times=end_times,
        )
        ctx.progress(done=2, total=3)

        # ---- trajectories with per-term evidence -------------------------
        trajectories: list[Trajectory] = []
        for global_id, members in enumerate(sorted(clusters, key=sorted)):
            member_list = sorted(members)
            entity = COCO_TO_ENTITY[
                Counter(class_ids[m] for m in member_list).most_common(1)[0][0]
            ]
            evidence: dict[str, float] = {}
            confidence = 1.0
            intra = [
                (i, j)
                for pos, i in enumerate(member_list)
                for j in member_list[pos + 1 :]
                if (i, j) in appearance_sim
            ]
            if intra:
                evidence["appearance"] = float(
                    np.mean([appearance_sim[p] for p in intra])
                )
                evidence["hsv"] = float(
                    np.mean(
                        [
                            compute_hsv_similarity(hsv_features[i], hsv_features[j])
                            for i, j in intra
                        ]
                    )
                )
                evidence["spatiotemporal"] = float(
                    np.mean(
                        [
                            st_validator.transition_score(
                                camera_ids[i], camera_ids[j], end_times[i], start_times[j]
                            )
                            for i, j in intra
                        ]
                    )
                )
                confidence = float(
                    np.clip(np.mean([combined.get(p, 0.0) for p in intra]), 0.0, 1.0)
                )
            trajectories.append(
                Trajectory(
                    global_id=global_id,
                    entity_class=entity,
                    members=[
                        TrackKey(
                            run_id=ctx.manifest.run_id,
                            camera_id=camera_ids[m],
                            track_id=rows[m]["track_id"],
                        )
                        for m in member_list
                    ],
                    confidence=confidence,
                    evidence=evidence,
                )
            )

        payload = {
            "schema_version": ASSOCIATE_SCHEMA_VERSION,
            "primary_stream": "+".join(streams),
            "streams": streams,
            "num_tracklets": n,
            "num_identities": len(trajectories),
            "trajectories": [t.model_dump(mode="json") for t in trajectories],
        }
        relpath = "trajectories.json"
        (ctx.run_dir / relpath).write_text(json.dumps(payload), encoding="utf-8")
        ctx.register_artifact(
            ArtifactRecord(
                name="associate.trajectories",
                relpath=relpath,
                schema_version=ASSOCIATE_SCHEMA_VERSION,
                producer=f"associate/{'+'.join(streams)}",
                row_count=len(trajectories),
            )
        )
        ctx.progress(done=3, total=3)
        multi = sum(1 for t in trajectories if len(t.members) > 1)
        logger.info(
            "associate: %d tracklets -> %d identities (%d cross-camera)",
            n, len(trajectories), multi,
        )
