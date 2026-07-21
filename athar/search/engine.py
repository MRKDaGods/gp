"""Gallery search: query preprocessed gallery runs with a probe run.

The flagship workflow's second half (D5): a probe run (reference footage,
processed through the same ingest→detect_track→embed pipeline) is searched
against a gallery run's FAISS stream indexes; hits join back through the
catalog to TrackKeys and become APPEARANCE HypothesisEdges on a case
Target — proposed, never silently fused (D7).

Compatibility is checked before any scoring: the stream must exist on both
runs with the same dim, and projected streams must share projection lineage
(``projection_fitted_on``) — cross-lineage scoring silently degrades ranks,
which is exactly the class of bug v1's global PCA pickle caused.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from athar.contracts.manifest import RunManifest, RunRole
from athar.contracts.store import FilesystemRunStore
from athar.core.ids import TrackKey
from athar.core.types import EntityClass
from athar.search.case_models import HypothesisEdge, HypothesisKind, Target

logger = logging.getLogger(__name__)


class SearchError(RuntimeError):
    pass


@dataclass(frozen=True)
class SearchHit:
    probe_key: TrackKey
    gallery_key: TrackKey
    stream: str
    score: float
    gallery_entity_class: EntityClass
    gallery_start_ts_s: float
    gallery_end_ts_s: float


class GallerySearcher:
    """Search interface over ONE indexed gallery run."""

    def __init__(self, store: FilesystemRunStore, gallery: RunManifest) -> None:
        if "index.catalog" not in gallery.artifacts:
            raise SearchError(
                f"run {gallery.run_id} is not searchable (no index.catalog artifact); "
                f"role={gallery.role.value}, status={gallery.status.value}"
            )
        if gallery.role not in (RunRole.GALLERY, RunRole.BENCHMARK):
            logger.warning(
                "searching a %s run (%s) — expected a gallery",
                gallery.role.value, gallery.run_id,
            )
        self._store = store
        self.gallery = gallery
        self._summary = json.loads(
            store.artifact_path(gallery, "embed.summary").read_text(encoding="utf-8")
        )
        self._catalog_rows = self._load_catalog()
        self._indexes: dict[str, tuple[object, list[int]]] = {}

    def _load_catalog(self) -> list[dict]:
        from athar.components.indexing.metadata_store import MetadataStore

        catalog = MetadataStore(self._store.artifact_path(self.gallery, "index.catalog"))
        try:
            return sorted(catalog.get_all(), key=lambda r: r["index_id"])
        finally:
            catalog.close()

    def streams(self) -> dict[str, int]:
        return {name: info["dim"] for name, info in self._summary["streams"].items()}

    def _stream_index(self, stream: str):
        import faiss

        if stream not in self._indexes:
            name = f"index.{stream}"
            if name not in self.gallery.artifacts:
                raise SearchError(
                    f"gallery {self.gallery.run_id} has no FAISS index for "
                    f"stream {stream!r}; available: "
                    f"{[a for a in self.gallery.artifacts if a.startswith('index.')]}"
                )
            index = faiss.read_index(str(self._store.artifact_path(self.gallery, name)))
            row_map = json.loads(
                self._store.artifact_path(self.gallery, f"index.{stream}.rows").read_text(
                    encoding="utf-8"
                )
            )
            self._indexes[stream] = (index, row_map)
        return self._indexes[stream]

    def search_vectors(
        self,
        vectors: np.ndarray,
        stream: str,
        top_k: int = 10,
        entity_classes: Optional[set[EntityClass]] = None,
    ) -> list[list[tuple[dict, float]]]:
        """Raw search: per query vector, ranked (catalog_row, score) hits."""
        from athar.components.adapters.detection import COCO_TO_ENTITY

        index, row_map = self._stream_index(stream)
        expected_dim = self._summary["streams"][stream]["dim"]
        if vectors.ndim != 2 or vectors.shape[1] != expected_dim:
            raise SearchError(
                f"query shape {vectors.shape} incompatible with stream "
                f"{stream!r} (dim {expected_dim})"
            )
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        # over-fetch when class-filtering so top_k survivable hits remain
        fetch = min(index.ntotal, top_k * (3 if entity_classes else 1))
        scores, positions = index.search(vectors, fetch)
        results: list[list[tuple[dict, float]]] = []
        for qi in range(vectors.shape[0]):
            hits: list[tuple[dict, float]] = []
            for score, pos in zip(scores[qi], positions[qi]):
                if pos < 0:
                    continue
                row = self._catalog_rows[row_map[int(pos)]]
                if entity_classes is not None:
                    entity = COCO_TO_ENTITY.get(row["class_id"])
                    if entity not in entity_classes:
                        continue
                hits.append((row, float(score)))
                if len(hits) == top_k:
                    break
            results.append(hits)
        return results

    def search_probe(
        self,
        probe_store: FilesystemRunStore,
        probe: RunManifest,
        stream: str,
        top_k: int = 10,
        min_score: float = 0.0,
        same_class_only: bool = True,
    ) -> list[SearchHit]:
        """Search every probe tracklet's stream vector against the gallery."""
        from athar.components.adapters.detection import COCO_TO_ENTITY

        probe_summary = self._check_compatible(probe_store, probe, stream)
        hits: list[SearchHit] = []
        for artifact_name in probe_summary["streams"][stream]["artifacts"]:
            cam = artifact_name.split(".")[1]
            data = np.load(probe_store.artifact_path(probe, artifact_name))
            vectors = np.asarray(data["embeddings"], dtype=np.float32)
            track_ids = data["track_ids"]
            probe_classes = self._probe_classes(probe_store, probe, cam)
            for qi, ranked in enumerate(
                self.search_vectors(vectors, stream, top_k=top_k)
            ):
                probe_key = TrackKey(
                    run_id=probe.run_id, camera_id=cam, track_id=int(track_ids[qi])
                )
                probe_class = probe_classes.get(int(track_ids[qi]))
                for row, score in ranked:
                    if score < min_score:
                        continue
                    gallery_class = COCO_TO_ENTITY[row["class_id"]]
                    if same_class_only and probe_class is not None and gallery_class != probe_class:
                        continue
                    hits.append(
                        SearchHit(
                            probe_key=probe_key,
                            gallery_key=TrackKey(
                                run_id=self.gallery.run_id,
                                camera_id=row["camera_id"],
                                track_id=row["track_id"],
                            ),
                            stream=stream,
                            score=score,
                            gallery_entity_class=gallery_class,
                            gallery_start_ts_s=row["start_time"],
                            gallery_end_ts_s=row["end_time"],
                        )
                    )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits

    def _probe_classes(
        self, probe_store: FilesystemRunStore, probe: RunManifest, cam: str
    ) -> dict[int, EntityClass]:
        name = f"tracklets.{cam}"
        if name not in probe.artifacts:
            return {}
        payload = json.loads(
            probe_store.artifact_path(probe, name).read_text(encoding="utf-8")
        )
        return {
            t["key"]["track_id"]: EntityClass(t["entity_class"])
            for t in payload["tracklets"]
        }

    def _check_compatible(
        self, probe_store: FilesystemRunStore, probe: RunManifest, stream: str
    ) -> dict:
        """Refuse incompatible probe/gallery stream pairings; returns the
        probe's embed summary."""
        gallery_streams = self._summary["streams"]
        if stream not in gallery_streams:
            raise SearchError(f"gallery {self.gallery.run_id} has no stream {stream!r}")
        if "embed.summary" not in probe.artifacts:
            raise SearchError(f"probe {probe.run_id} has no embed.summary artifact")
        probe_summary = json.loads(
            probe_store.artifact_path(probe, "embed.summary").read_text(encoding="utf-8")
        )
        if stream not in probe_summary["streams"]:
            raise SearchError(f"probe {probe.run_id} has no stream {stream!r}")
        g_dim = gallery_streams[stream]["dim"]
        p_dim = probe_summary["streams"][stream]["dim"]
        if g_dim != p_dim:
            raise SearchError(
                f"stream {stream!r} dim mismatch: gallery {g_dim} vs probe {p_dim}"
            )
        g_proj = gallery_streams[stream].get("projection_fitted_on")
        p_proj = probe_summary["streams"][stream].get("projection_fitted_on")
        if g_proj != p_proj:
            raise SearchError(
                f"stream {stream!r} projection lineage mismatch: gallery "
                f"{g_proj!r} vs probe {p_proj!r} — refusing cross-lineage scoring"
            )
        return probe_summary


def propose_appearance_hypotheses(
    target: Target,
    hits: list[SearchHit],
    max_edges: int = 50,
) -> int:
    """Attach APPEARANCE hypothesis edges for new gallery tracklets.

    Dedupes by gallery TrackKey (best score wins), skips tracklets already
    confirmed on the target or already proposed. Returns edges added.
    """
    existing = {
        (e.track_key.run_id, e.track_key.camera_id, e.track_key.track_id)
        for e in target.hypotheses
    }
    confirmed = {
        (k.run_id, k.camera_id, k.track_id) for k in target.confirmed_members
    }
    best: dict[tuple, SearchHit] = {}
    for hit in hits:
        key = (
            hit.gallery_key.run_id,
            hit.gallery_key.camera_id,
            hit.gallery_key.track_id,
        )
        if key in existing or key in confirmed:
            continue
        if key not in best or hit.score > best[key].score:
            best[key] = hit
    added = 0
    for hit in sorted(best.values(), key=lambda h: h.score, reverse=True)[:max_edges]:
        target.hypotheses.append(
            HypothesisEdge(
                kind=HypothesisKind.APPEARANCE,
                track_key=hit.gallery_key,
                raw_score=hit.score,
            )
        )
        added += 1
    return added
