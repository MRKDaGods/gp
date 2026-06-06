"""Stage-4 learned edge classifier / re-ranker.

Rescores the cross-camera ``combined_sim`` edges with a learned per-edge
``P(same-vehicle)`` model (v1: LightGBM on the ~20 engineered edge features from
``scripts/build_edge_pairs.py``). See
``docs/subagent-specs/edge-classifier-association.md`` sections 5-7.

Integration contract (pipeline.py, immediately after ``combined_sim`` is built):

    if cfg.stage4.association.edge_classifier.enabled:
        combined_sim = rescore_edges(combined_sim, ..., ec_cfg=...)

Modes (``ec_cfg.mode``):
  * ``blend``  (recommended): ``score' = (1-lambda)*combined_sim + lambda*P``,
    then an optional secondary gate drops edges with ``P < prob_threshold``.
    Keeps the FIC-calibrated cosine ordering dominant (protects conflict_free_cc
    + gallery/intra-merge thresholds) while P breaks ties / re-gates borderlines.
    ``blend_lambda == 0`` is a *provable* no-op (bit-identical to today).
  * ``replace``: ``score' = P`` (then the optional prob gate).
  * ``gate``: keep ``combined_sim`` unchanged but drop edges with
    ``P < prob_threshold`` (a pure learned veto).

The per-pair feature vector is built by the SAME ``PairFeatureBuilder`` that
``scripts/build_edge_pairs.py`` uses to produce the training table, so the
training and inference feature spaces are bit-for-bit the same (the spec's hard
"train/infer distribution match" requirement). FIC/AQE math is NOT reimplemented
here — the already-transformed pipeline arrays are passed straight in.

Leak-free eval support (``model_path`` payload):
  A pickled payload may be EITHER a single fitted estimator, OR a dict::

      {
        "feature_names": [...20 names...],   # asserted == FEATURE_NAMES order
        "models_by_train_scene": {"S02": <model>, "S01": <model>},
      }

  With ``models_by_train_scene`` present, a pair belonging to scene X is scored
  by the model whose *train scene* is NOT X (i.e. the held-out fold model). This
  is the scene-disjoint, never-train-on-the-scene-you-score protocol the 14o
  eval kernel asserts. With exactly two scenes the mapping is unambiguous
  (S01 pairs -> model_S02, S02 pairs -> model_S01); for >2 scenes a pair's model
  is the unique one whose train-scene differs (fail-loud if ambiguous).

Everything fail-loud: enabled-but-missing-model, feature-dim/name mismatch, an
unscoreable scene, or NaN/Inf features all raise immediately rather than
silently degrading the production gate.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger

# Single source of truth for the per-pair feature space + ordering.
from scripts.build_edge_pairs import (  # noqa: E402
    FEATURE_NAMES,
    PairFeatureBuilder,
    extract_scene,
)
from src.stage4_association.spatial_temporal import SpatioTemporalValidator


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
class EdgeClassifierModel:
    """Wraps either a single fitted model or a per-train-scene fold-model dict.

    ``predict_proba_for_scene(X, scene)`` returns P(same) for rows whose pair
    lives in ``scene``, automatically selecting the held-out fold model
    (train-scene != scene) when fold models are present.
    """

    def __init__(self, payload: object, model_path: Path) -> None:
        self.model_path = model_path
        self.single = None
        self.models_by_train_scene: Optional[Dict[str, object]] = None

        if isinstance(payload, dict) and "models_by_train_scene" in payload:
            feat_names = payload.get("feature_names")
            if feat_names is not None and list(feat_names) != list(FEATURE_NAMES):
                raise ValueError(
                    f"edge_classifier model feature_names mismatch in {model_path}:\n"
                    f"  model: {list(feat_names)}\n"
                    f"  code : {list(FEATURE_NAMES)}\n"
                    "Retrain the model against the current FEATURE_NAMES."
                )
            models = payload["models_by_train_scene"]
            if not isinstance(models, dict) or not models:
                raise ValueError(
                    f"edge_classifier 'models_by_train_scene' in {model_path} must be a "
                    f"non-empty dict {{train_scene: model}}; got {type(models)}"
                )
            for scene, mdl in models.items():
                if not hasattr(mdl, "predict_proba"):
                    raise ValueError(
                        f"edge_classifier fold model for train-scene {scene!r} in "
                        f"{model_path} has no predict_proba()"
                    )
            self.models_by_train_scene = {str(k): v for k, v in models.items()}
            logger.info(
                "Edge classifier: loaded fold models (leak-free), train-scenes="
                f"{sorted(self.models_by_train_scene)}"
            )
        else:
            # A bare estimator, or a dict carrying a single 'model' + metadata.
            model = payload.get("model") if isinstance(payload, dict) else payload
            if isinstance(payload, dict):
                feat_names = payload.get("feature_names")
                if feat_names is not None and list(feat_names) != list(FEATURE_NAMES):
                    raise ValueError(
                        f"edge_classifier model feature_names mismatch in {model_path}:\n"
                        f"  model: {list(feat_names)}\n  code : {list(FEATURE_NAMES)}"
                    )
            if model is None or not hasattr(model, "predict_proba"):
                raise ValueError(
                    f"edge_classifier model in {model_path} has no predict_proba() "
                    f"(got {type(model)}); expected a fitted classifier or a dict "
                    "with a 'model' / 'models_by_train_scene' key."
                )
            self.single = model
            logger.info("Edge classifier: loaded a single (non-fold) model")

    def scene_to_model(self, scene: str) -> object:
        """Return the model that must score pairs in ``scene`` (fail-loud).

        With fold models present, that is the unique model whose train-scene is
        NOT ``scene``. With a single model, it is always that model.
        """
        if self.single is not None:
            return self.single
        assert self.models_by_train_scene is not None
        candidates = [
            mdl for train_scene, mdl in self.models_by_train_scene.items()
            if train_scene != scene
        ]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise ValueError(
                f"edge_classifier: no held-out fold model to score scene {scene!r} "
                f"(only train-scenes {sorted(self.models_by_train_scene)} available; "
                "every model was trained on this scene -> would leak)."
            )
        raise ValueError(
            f"edge_classifier: ambiguous fold-model selection for scene {scene!r} — "
            f"{len(candidates)} models have train-scene != {scene!r} "
            f"(train-scenes {sorted(self.models_by_train_scene)}). Provide exactly one "
            "held-out model per evaluated scene."
        )


def load_edge_classifier(model_path: str | Path) -> EdgeClassifierModel:
    """Load the edge-classifier payload from ``model_path`` (fail-loud)."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"edge_classifier.enabled=true but model_path does not exist: {path}. "
            "Train it (scripts/build_edge_pairs.py output -> LightGBM) or set "
            "stage4.association.edge_classifier.enabled=false."
        )
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    return EdgeClassifierModel(payload, path)


# ---------------------------------------------------------------------------
# Re-scoring
# ---------------------------------------------------------------------------
def rescore_edges(
    combined_sim: Dict[Tuple[int, int], float],
    *,
    primary: np.ndarray,
    tertiary: Optional[np.ndarray],
    quaternary: Optional[np.ndarray],
    camera_ids: Sequence[str],
    class_ids: Sequence[int],
    track_ids: Sequence[int],
    start_times: Sequence[float],
    end_times: Sequence[float],
    num_frames: Sequence[int],
    mean_confs: Sequence[float],
    st_validator: SpatioTemporalValidator,
    fusion_weights: Tuple[float, float, float],
    ec_cfg,
    model: Optional[EdgeClassifierModel] = None,
    edge_probs_out: Optional[Dict[Tuple[int, int], float]] = None,
) -> Dict[Tuple[int, int], float]:
    """Rescore ``combined_sim`` edges with the learned edge classifier.

    Args:
        combined_sim: {(i, j): similarity} produced at pipeline.py:539.
        primary/tertiary/quaternary: already FIC(+AQE on primary)-transformed
            pipeline embedding arrays (DO NOT re-whiten — these match the gate).
        camera_ids/class_ids/track_ids: per-tracklet metadata (row-aligned).
        start_times/end_times/num_frames/mean_confs: per-tracklet temporal +
            quality metadata.
        st_validator: the pipeline's SpatioTemporalValidator (camera priors).
        fusion_weights: (w_primary, w_tertiary, w_quaternary) used for cos_fused.
        ec_cfg: the ``stage4.association.edge_classifier`` config block.
        model: optional pre-loaded EdgeClassifierModel (else loaded from
            ec_cfg.model_path).
        edge_probs_out: optional dict to receive {(i, j): P_same} for the
            forensic evidence trail.

    Returns:
        A NEW dict {(i, j): rescored similarity}. Edges dropped by the prob gate
        are absent from the returned dict (so the downstream graph never sees
        them). With ``blend_lambda == 0`` and ``prob_threshold <= 0`` the result
        is value-identical to the input (no-op guarantee).
    """
    mode = str(ec_cfg.get("mode", "blend")).lower()
    blend_lambda = float(ec_cfg.get("blend_lambda", 0.5))
    prob_threshold = float(ec_cfg.get("prob_threshold", 0.0))

    if mode not in {"blend", "replace", "gate"}:
        raise ValueError(f"edge_classifier.mode must be blend|replace|gate, got {mode!r}")
    if not (0.0 <= blend_lambda <= 1.0):
        raise ValueError(f"edge_classifier.blend_lambda must be in [0, 1], got {blend_lambda}")

    if not combined_sim:
        return dict(combined_sim)

    if model is None:
        model = load_edge_classifier(ec_cfg.get("model_path", ""))

    # --- Fast no-op short-circuit (provable bit-identical when lambda=0 & no gate) ---
    # blend with lambda=0 collapses to the input similarity; with prob_threshold<=0
    # no edge is dropped. Skip all model inference to guarantee zero drift.
    if mode == "blend" and blend_lambda == 0.0 and prob_threshold <= 0.0:
        logger.info(
            "Edge classifier: blend_lambda=0 & prob_threshold<=0 -> provable no-op "
            "(returning combined_sim unchanged)."
        )
        return dict(combined_sim)

    fb = PairFeatureBuilder(
        primary=primary,
        tertiary=tertiary,
        quaternary=quaternary,
        camera_ids=camera_ids,
        class_ids=class_ids,
        track_ids=track_ids,
        start_times=start_times,
        end_times=end_times,
        num_frames=num_frames,
        mean_confs=mean_confs,
        st_validator=st_validator,
        fusion_weights=fusion_weights,
    )

    cam_scene = {c: extract_scene(c) for c in set(camera_ids)}

    # Build the feature matrix for every edge, grouped by the pair's scene so we
    # can apply the correct (held-out) fold model per scene in one vectorized
    # predict per scene.
    edges = list(combined_sim.keys())
    rows: List[List[float]] = []
    pair_scenes: List[str] = []
    for (i, j) in edges:
        scene_i = cam_scene[camera_ids[i]]
        scene_j = cam_scene[camera_ids[j]]
        # Cross-camera same-scene edges only reach here; assert consistency.
        scene = scene_i or scene_j
        if scene_i and scene_j and scene_i != scene_j:
            raise ValueError(
                f"edge_classifier: cross-scene edge ({i},{j}) cam {camera_ids[i]}->"
                f"{camera_ids[j]} (scenes {scene_i}/{scene_j}) — scene blocking violated."
            )
        rows.append(fb.feature_vector(i, j))
        pair_scenes.append(scene)

    X = np.asarray(rows, dtype=np.float64)
    if X.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"edge_classifier feature-dim mismatch: built {X.shape[1]} features, "
            f"expected {len(FEATURE_NAMES)} ({FEATURE_NAMES})."
        )
    if not np.isfinite(X).all():
        bad = np.where(~np.isfinite(X))
        raise ValueError(
            f"edge_classifier: {len(bad[0])} non-finite feature value(s) — refusing to "
            f"score (first bad row={int(bad[0][0])}, col={FEATURE_NAMES[bad[1][0]]})."
        )

    # Predict P(same) per scene with the correct held-out model.
    probs = np.empty(len(edges), dtype=np.float64)
    scenes_arr = np.array(pair_scenes)
    for scene in sorted(set(pair_scenes)):
        mask = scenes_arr == scene
        mdl = model.scene_to_model(scene)
        probs[mask] = np.asarray(mdl.predict_proba(X[mask]))[:, 1]

    # Apply the chosen mode.
    out: Dict[Tuple[int, int], float] = {}
    dropped = 0
    for idx, (i, j) in enumerate(edges):
        p = float(probs[idx])
        if edge_probs_out is not None:
            edge_probs_out[(i, j)] = p
        if prob_threshold > 0.0 and p < prob_threshold:
            dropped += 1
            continue  # secondary learned gate: drop borderline edge entirely
        base = combined_sim[(i, j)]
        if mode == "gate":
            new = base
        elif mode == "replace":
            new = p
        else:  # blend
            new = (1.0 - blend_lambda) * base + blend_lambda * p
        out[(i, j)] = new

    logger.info(
        f"Edge classifier ({mode}, lambda={blend_lambda}, prob_thr={prob_threshold}): "
        f"scored {len(edges)} edges across scenes {sorted(set(pair_scenes))}; "
        f"dropped {dropped} below prob_threshold; kept {len(out)}."
    )
    return out
