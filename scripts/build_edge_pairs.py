#!/usr/bin/env python
"""Build a labeled cross-camera tracklet-pair feature table and run a
LightGBM separability probe (the DE-RISK GATE for the learned edge classifier).

This is the FIRST concrete step of the Stage-4 learned edge-classifier design
(``docs/subagent-specs/edge-classifier-association.md`` section 8). It is
**read-only** w.r.t. the pipeline: it does NOT modify
``src/stage4_association/pipeline.py`` and does NOT touch any config block.

It answers one question cheaply: *can a learned model separate the hard
cross-camera tracklet pairs better than the cosine-similarity threshold?*

Pipeline (per run dir + GT root):
  1. Load frozen 14e/K7 Stage-1 tracklets + Stage-2 per-stream embeddings.
  2. Whiten each stream with FIC (``per_camera_whiten``) and apply AQE to the
     primary stream (``average_query_expansion_batched``) -- the **exact**
     functions the live Stage-4 gate uses (imported, not reimplemented).
  3. Assign a GT ``global_id`` to every predicted tracklet by IoU majority vote
     (1-based GT frame -> 0-based internal; GT (x,y,w,h) -> (x1,y1,x2,y2)).
  4. Build cross-camera, same-class, scene-blocked pairs with section-3 features
     + a binary same-vehicle label; hard-negative-mine and subsample.
  5. Emit ``edge_pairs_S01.parquet`` / ``edge_pairs_S02.parquet`` (or ``.npz``).
  6. Print the GO/NO-GO separability report: baseline ``cos_fused`` AUC vs a
     **scene-disjoint** LightGBM held-out AUC (train S02 -> eval S01, mirror).

CRITICAL feature-space note:
  The pipeline applies FIC to *every* appearance stream, but applies AQE **only
  to the primary** stream (tertiary/quaternary are FIC-only). This script
  reproduces that exactly: ``cos_primary`` is in FIC+AQE space; ``cos_dinov2``
  and ``cos_r50ibn`` are in FIC-only space; ``cos_fused`` is the K7-weighted
  blend of those, matching ``stage4_association.pipeline`` reranking-disabled
  ``appearance_sim``. Use ``--raw-cosines`` to fall back to plain L2-normalized
  cosines (prints a loud warning that the space differs from the live gate).

Run on Kaggle (CPU) via ``notebooks/kaggle/14n_edge_pairs_probe``; it cannot be
validated locally because the frozen run + GT live on Kaggle, but a synthetic
self-test (``--self-test``) exercises every code path end to end.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Make ``src`` importable when run as a script from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse the EXACT pipeline feature-space transforms (do not reimplement the math).
from src.stage4_association.fic import per_camera_whiten  # noqa: E402
from src.stage4_association.query_expansion import (  # noqa: E402
    average_query_expansion_batched,
)
from src.stage4_association.spatial_temporal import SpatioTemporalValidator  # noqa: E402

try:
    # Reuse the pipeline's temporal-overlap helper (similarity.py:28) verbatim.
    from src.stage4_association.similarity import compute_temporal_overlap_ratio  # noqa: E402
except Exception:  # pragma: no cover - defensive
    def compute_temporal_overlap_ratio(start_i, end_i, start_j, end_j):  # type: ignore
        overlap = max(0.0, min(end_i, end_j) - max(start_i, start_j))
        if overlap <= 0:
            return 0.0
        min_dur = min(end_i - start_i, end_j - start_j)
        return min(overlap / min_dur, 1.0) if min_dur > 0 else 0.0


# K7 (vehicle_mtmc_14k_v1_k7) fusion weights. The authoritative source is
# configs/model_registry.yaml entry vehicle_mtmc_14k_v1_k7 (read at runtime by
# read_k7_weights). The values below are the documented fallback used only when
# the registry cannot be parsed: w_tertiary=0.45, w_quaternary=0.45 -> primary is
# the implicit remainder 1 - 0.45 - 0.45 = 0.10. These are the weights the K7
# Stage-4 gate blends per-stream cosines with (pipeline.py Step 3b).
K7_W_TERTIARY = 0.45
K7_W_QUATERNARY = 0.45
K7_W_PRIMARY = round(1.0 - K7_W_TERTIARY - K7_W_QUATERNARY, 6)  # 0.10

# Defaults mirror configs/datasets/cityflowv2.yaml + the 14e/K7 model_overrides.
DEFAULT_FIC_REG = 0.5
DEFAULT_FIC_MIN_SAMPLES = 5
DEFAULT_AQE_K = 2
DEFAULT_AQE_ALPHA = 5.0
DEFAULT_TOP_K = 100  # stage4.association.top_k

# Section-2 pair mining knobs.
GT_IOU_THRESH = 0.5
GT_AGREEMENT_FRAC = 0.50
HARD_NEG_COS_FUSED = 0.30
EASY_NEG_RATIO = 3.0  # easy negatives kept at ~3x positives
# Below this many held-out hard negatives, the hard-neg AUC is statistically
# meaningless (a 1-2 negative subset gives degenerate 0/1 AUCs and a spurious
# delta). When a fold has fewer, fall back to the all-rows AUC for that fold.
MIN_HARD_NEG = 10

VEHICLE_CLASS_IDS = {2, 5, 7}  # car, bus, truck (PERSON_CLASSES would be {0})

# Canonical CityFlowV2 eval cameras (used for self-test + sanity prints).
EXPECTED_CAMS = ["S01_c001", "S01_c002", "S01_c003", "S02_c006", "S02_c007", "S02_c008"]


# ---------------------------------------------------------------------------
# K7 fusion weights (authoritative: configs/model_registry.yaml)
# ---------------------------------------------------------------------------
def read_k7_weights() -> Tuple[float, float, float]:
    """Return (w_primary, w_tertiary, w_quaternary) for K7 from the model registry.

    Reads the ``vehicle_mtmc_14k_v1_k7`` entry's ``model_overrides`` in
    ``configs/model_registry.yaml`` and derives w_primary as the implicit
    remainder ``1 - w_secondary - w_tertiary - w_quaternary`` (matches the live
    Stage-4 score-fusion math in pipeline.py:497). Falls back to the documented
    constants (0.10 / 0.45 / 0.45) with a warning if the registry can't be read.
    """
    cfg_path = _REPO_ROOT / "configs" / "model_registry.yaml"
    if not cfg_path.exists():
        print(f"  WARNING: {cfg_path} not found; using fallback K7 weights "
              f"({K7_W_PRIMARY}/{K7_W_TERTIARY}/{K7_W_QUATERNARY})")
        return K7_W_PRIMARY, K7_W_TERTIARY, K7_W_QUATERNARY
    try:
        import yaml  # PyYAML ships with omegaconf, always available locally/Kaggle

        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        entry = next(m for m in data["models"] if m["id"] == "vehicle_mtmc_14k_v1_k7")
        w_sec = w_tert = w_quat = 0.0
        for ov in entry.get("model_overrides", []):
            key, _, val = str(ov).partition("=")
            key = key.strip()
            if key == "stage4.association.secondary_embeddings.weight":
                w_sec = float(val)
            elif key == "stage4.association.tertiary_embeddings.weight":
                w_tert = float(val)
            elif key == "stage4.association.quaternary_embeddings.weight":
                w_quat = float(val)
        w_pri = round(1.0 - w_sec - w_tert - w_quat, 6)
        if w_pri < -1e-9:
            raise ValueError(f"derived negative w_primary={w_pri}")
        return w_pri, w_tert, w_quat
    except Exception as exc:
        print(f"  WARNING: failed to read K7 weights from registry ({exc}); using fallback "
              f"({K7_W_PRIMARY}/{K7_W_TERTIARY}/{K7_W_QUATERNARY})")
        return K7_W_PRIMARY, K7_W_TERTIARY, K7_W_QUATERNARY


# ---------------------------------------------------------------------------
# Scene helper (mirrors pipeline._extract_scene)
# ---------------------------------------------------------------------------
def extract_scene(camera_id: str) -> str:
    """'S01_c001' -> 'S01'; cameras without an S<digits> prefix -> ''."""
    parts = camera_id.split("_")
    if len(parts) >= 2 and parts[0][:1].upper() == "S" and parts[0][1:].isdigit():
        return parts[0]
    return ""


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
@dataclass
class RunInputs:
    """Everything loaded from a frozen run directory + GT root."""

    index_map: List[dict]                 # row -> {camera_id, track_id, class_id}
    camera_ids: List[str]
    track_ids: List[int]
    class_ids: List[int]
    primary: np.ndarray                   # (N, Dp) FIC+AQE
    tertiary: Optional[np.ndarray]        # (N, Dt) FIC-only
    quaternary: Optional[np.ndarray]      # (N, Dq) FIC-only
    start_times: List[float]
    end_times: List[float]
    num_frames: List[int]
    mean_confs: List[float]
    gt_ids: List[Optional[int]]           # per-tracklet majority GT id (None = ambiguous)


def _load_npy(path: Path) -> Optional[np.ndarray]:
    return np.load(path).astype(np.float32) if path.exists() else None


def _l2norm(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, 1e-8)


def load_run(
    run_dir: Path,
    gt_root: Path,
    *,
    raw_cosines: bool,
    fic_reg: float,
    fic_min_samples: int,
    aqe_k: int,
    aqe_alpha: float,
    top_k: int,
) -> RunInputs:
    """Load tracklets + embeddings, build the pipeline feature space, assign GT ids."""
    from src.core.io_utils import load_tracklets_by_camera

    stage1_dir = run_dir / "stage1"
    stage2_dir = run_dir / "stage2"

    idx_path = stage2_dir / "embedding_index.json"
    emb_path = stage2_dir / "embeddings.npy"
    if not idx_path.exists():
        raise FileNotFoundError(f"Missing embedding index: {idx_path}")
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing primary embeddings: {emb_path}")
    if not stage1_dir.exists():
        raise FileNotFoundError(f"Missing stage1 tracklet dir: {stage1_dir}")

    index_map = json.loads(idx_path.read_text(encoding="utf-8"))
    camera_ids = [str(r["camera_id"]) for r in index_map]
    track_ids = [int(r["track_id"]) for r in index_map]
    class_ids = [int(r["class_id"]) for r in index_map]
    n = len(index_map)

    primary_raw = _load_npy(emb_path)
    if primary_raw is None or primary_raw.shape[0] != n:
        raise ValueError(
            f"Primary embeddings row mismatch: {None if primary_raw is None else primary_raw.shape} vs index {n}"
        )
    tertiary_raw = _load_npy(stage2_dir / "embeddings_tertiary.npy")
    quaternary_raw = _load_npy(stage2_dir / "embeddings_quaternary.npy")
    for name, arr in (("tertiary", tertiary_raw), ("quaternary", quaternary_raw)):
        if arr is not None and arr.shape[0] != n:
            raise ValueError(f"{name} embeddings row mismatch: {arr.shape} vs index {n}")

    # ---- Build the appearance feature space ----
    if raw_cosines:
        warnings.warn(
            "RAW-COSINE MODE: cosines are plain L2-normalized embeddings, NOT the "
            "FIC(+AQE)-whitened space the live Stage-4 gate uses. The separability "
            "signal is weakened (but not invalidated). Prefer the default reuse path.",
            stacklevel=2,
        )
        primary = _l2norm(primary_raw)
        tertiary = _l2norm(tertiary_raw) if tertiary_raw is not None else None
        quaternary = _l2norm(quaternary_raw) if quaternary_raw is not None else None
    else:
        # FIC every stream separately (matches pipeline.py:289 primary,
        # :234 tertiary, :264 quaternary -- each call independent per camera).
        primary = per_camera_whiten(
            primary_raw, camera_ids, regularisation=fic_reg, min_samples=fic_min_samples
        )
        tertiary = (
            per_camera_whiten(
                _l2norm(tertiary_raw), camera_ids, regularisation=fic_reg, min_samples=fic_min_samples
            )
            if tertiary_raw is not None
            else None
        )
        quaternary = (
            per_camera_whiten(
                _l2norm(quaternary_raw), camera_ids, regularisation=fic_reg, min_samples=fic_min_samples
            )
            if quaternary_raw is not None
            else None
        )
        # AQE on the PRIMARY ONLY (pipeline.py:387). DBA=false in K7/14e, so the
        # neighbour indices come from the pre-AQE FIC embeddings. Reproduce the
        # FAISS flat_ip top-K with a brute-force cosine argsort (identical for
        # exact inner-product search; N<=~1000 so this is trivial).
        if aqe_k and aqe_k > 0:
            indices = _bruteforce_topk_indices(primary, k=top_k)
            primary = average_query_expansion_batched(
                primary, indices, k=aqe_k, alpha=aqe_alpha
            )

    # ---- Temporal metadata from Stage-1 tracklets ----
    tracklets_by_camera = load_tracklets_by_camera(stage1_dir)
    tracklet_lookup: Dict[Tuple[str, int], "object"] = {}
    for cam, tracks in tracklets_by_camera.items():
        for t in tracks:
            tracklet_lookup[(cam, t.track_id)] = t

    start_times: List[float] = []
    end_times: List[float] = []
    num_frames: List[int] = []
    mean_confs: List[float] = []
    missing = 0
    for cam, tid in zip(camera_ids, track_ids):
        t = tracklet_lookup.get((cam, tid))
        if t is None or not t.frames:
            missing += 1
            start_times.append(0.0)
            end_times.append(0.0)
            num_frames.append(1)
            mean_confs.append(0.0)
            continue
        st = t.start_time
        et = t.end_time
        if st > et:
            st, et = et, st
        start_times.append(st)
        end_times.append(et)
        num_frames.append(t.num_frames)
        mean_confs.append(t.mean_confidence)
    if missing:
        print(f"  WARNING: {missing}/{n} index rows had no matching Stage-1 tracklet (degraded temporal feats)")

    # ---- GT-id assignment (IoU majority vote) ----
    gt_boxes = load_gt_boxes(gt_root)
    gt_ids = assign_gt_ids(
        camera_ids, track_ids, tracklet_lookup, gt_boxes, iou_thresh=GT_IOU_THRESH,
        agreement_frac=GT_AGREEMENT_FRAC,
    )
    n_assigned = sum(1 for g in gt_ids if g is not None)
    print(f"  GT-id assignment: {n_assigned}/{n} tracklets matched a GT id "
          f"({n - n_assigned} ambiguous/unmatched, excluded from pairs)")

    return RunInputs(
        index_map=index_map,
        camera_ids=camera_ids,
        track_ids=track_ids,
        class_ids=class_ids,
        primary=primary,
        tertiary=tertiary,
        quaternary=quaternary,
        start_times=start_times,
        end_times=end_times,
        num_frames=num_frames,
        mean_confs=mean_confs,
        gt_ids=gt_ids,
    )


def _bruteforce_topk_indices(emb: np.ndarray, k: int) -> np.ndarray:
    """Top-k neighbour indices per row by descending cosine (self included at col 0).

    Equivalent to a FAISS IndexFlatIP search over L2-normalized vectors. The AQE
    helper filters the self-index and out-of-range sentinels itself, so we just
    need each row's k highest-similarity column indices in sorted order.
    """
    n = emb.shape[0]
    k = min(k, n)
    sims = emb @ emb.T  # (N, N) — N is small for MTMC (<= ~1000)
    # argpartition for the top-k, then sort those k by descending similarity.
    part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    row_idx = np.arange(n)[:, None]
    part_sims = sims[row_idx, part]
    order = np.argsort(-part_sims, axis=1)
    return part[row_idx, order].astype(np.int64)


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------
def load_gt_boxes(gt_root: Path) -> Dict[str, Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]]]:
    """Parse ``<CAM>/gt/gt.txt`` -> {camera: {frame_1based: [(gid, (x1,y1,x2,y2)), ...]}}.

    GT line: frame_id(1-based), global_id, x, y, w, h, [conf, -1, -1, -1].
    Box converted (x,y,w,h) -> (x1,y1,x2,y2).
    """
    out: Dict[str, Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]]] = {}
    if not gt_root.exists():
        raise FileNotFoundError(f"GT root does not exist: {gt_root}")
    cam_dirs = sorted(p for p in gt_root.iterdir() if p.is_dir() and (p / "gt" / "gt.txt").exists())
    if not cam_dirs:
        raise FileNotFoundError(
            f"No <CAM>/gt/gt.txt found under {gt_root}. Expected layout like "
            f"{gt_root}/<CAM>/gt/gt.txt"
        )
    for cam_dir in cam_dirs:
        cam = cam_dir.name
        frame_map: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]] = defaultdict(list)
        with (cam_dir / "gt" / "gt.txt").open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.replace("\t", ",").split(",")
                if len(parts) < 6:
                    continue
                try:
                    frame_id = int(float(parts[0]))
                    gid = int(float(parts[1]))
                    x, y, w, h = (float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]))
                except ValueError:
                    continue
                frame_map[frame_id].append((gid, (x, y, x + w, y + h)))
        out[cam] = frame_map
    return out


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two (x1,y1,x2,y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def assign_gt_ids(
    camera_ids: List[str],
    track_ids: List[int],
    tracklet_lookup: Dict[Tuple[str, int], "object"],
    gt_boxes: Dict[str, Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]]],
    *,
    iou_thresh: float = GT_IOU_THRESH,
    agreement_frac: float = GT_AGREEMENT_FRAC,
) -> List[Optional[int]]:
    """Per-tracklet GT id by per-frame best-IoU match + majority vote.

    Frame convention (CLAUDE.md rule 4): internal tracklet frame_id is 0-based;
    GT is 1-based -> ``gt_frame = internal_frame + 1``. A tracklet is assigned the
    GT id agreed on by >= ``agreement_frac`` of its frames, else ``None``
    (ambiguous -> excluded from pairs).
    """
    gt_ids: List[Optional[int]] = []
    for cam, tid in zip(camera_ids, track_ids):
        t = tracklet_lookup.get((cam, tid))
        cam_gt = gt_boxes.get(cam, {})
        if t is None or not t.frames or not cam_gt:
            gt_ids.append(None)
            continue
        votes: Dict[int, int] = defaultdict(int)
        n_frames = 0
        for fr in t.frames:
            n_frames += 1
            gt_frame = fr.frame_id + 1  # 0-based -> 1-based
            candidates = cam_gt.get(gt_frame)
            if not candidates:
                continue
            best_iou = iou_thresh
            best_gid: Optional[int] = None
            for gid, gbox in candidates:
                iou = _iou(fr.bbox, gbox)
                if iou >= best_iou:
                    best_iou = iou
                    best_gid = gid
            if best_gid is not None:
                votes[best_gid] += 1
        if not votes or n_frames == 0:
            gt_ids.append(None)
            continue
        best_gid, best_count = max(votes.items(), key=lambda kv: kv[1])
        gt_ids.append(best_gid if best_count >= agreement_frac * n_frames else None)
    return gt_ids


# ---------------------------------------------------------------------------
# Feature engineering for pairs (section 3)
# ---------------------------------------------------------------------------
def _build_st_validator(camera_transitions: Optional[dict]) -> SpatioTemporalValidator:
    # Mirrors stage4.association.spatiotemporal defaults for CityFlowV2.
    return SpatioTemporalValidator(
        min_time_gap=0.0,
        max_time_gap=300.0,
        camera_transitions=camera_transitions,
    )


def _load_camera_transitions() -> Optional[dict]:
    """Read camera_transitions priors from configs/datasets/cityflowv2.yaml."""
    cfg_path = _REPO_ROOT / "configs" / "datasets" / "cityflowv2.yaml"
    if not cfg_path.exists():
        return None
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(cfg_path)
        ct = cfg.stage4.association.spatiotemporal.get("camera_transitions")
        return OmegaConf.to_container(ct, resolve=True) if ct is not None else None
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  WARNING: could not load camera_transitions ({exc}); st_score uses global prior")
        return None


def _pair_prior_times(
    st_validator: SpatioTemporalValidator, cam_a: str, cam_b: str
) -> Tuple[float, float]:
    """(pair_mean_time, pair_max_time) from the learned camera-pair prior.

    Falls back to (global mean placeholder, max_time_gap) when no prior exists.
    """
    prior = st_validator._get_pair_prior(cam_a, cam_b)
    if prior is not None:
        return (
            float(prior.get("mean_time", st_validator.min_time_gap)),
            float(prior.get("max_time", st_validator.max_time_gap)),
        )
    return (st_validator.min_time_gap, st_validator.max_time_gap)


# Ordered feature names (categorical handled separately downstream).
FEATURE_NAMES = [
    "cos_primary",
    "cos_dinov2",
    "cos_r50ibn",
    "cos_fused",
    "cos_min",
    "cos_max",
    "cos_std",
    "rank_i_of_j",
    "rank_j_of_i",
    "is_mutual_top1",
    "recip_rank_harmonic",
    "time_gap",
    "st_score",
    "temporal_overlap_ratio",
    "camera_pair_id",          # categorical (integer-coded)
    "pair_mean_time",
    "pair_max_time",
    "min_track_len",
    "len_ratio",
    "min_mean_conf",
]
CATEGORICAL_FEATURES = ["camera_pair_id"]


def build_pairs(
    run: RunInputs,
    st_validator: SpatioTemporalValidator,
    fusion_weights: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, list]:
    """Build cross-camera, same-class, scene-blocked labeled pairs with features.

    Returns a column-oriented dict (feature columns + label + provenance columns).
    Pair mining: keep all positives; keep all hard negatives (cos_fused >= 0.30);
    random-subsample the easy negative tail to ~EASY_NEG_RATIO x positives.

    ``fusion_weights`` is (w_primary, w_tertiary, w_quaternary); defaults to the
    module K7 constants. cos_fused = w_p*cos_primary + w_t*cos_dinov2 + w_q*cos_r50ibn,
    matching the live Stage-4 score fusion (pipeline.py:497-511).
    """
    w_pri, w_tert, w_quat = fusion_weights if fusion_weights is not None else (
        K7_W_PRIMARY, K7_W_TERTIARY, K7_W_QUATERNARY
    )
    n = len(run.camera_ids)
    primary, tertiary, quaternary = run.primary, run.tertiary, run.quaternary

    # Group indices by camera; precompute scene per camera.
    cam_to_idxs: Dict[str, List[int]] = defaultdict(list)
    for i, cam in enumerate(run.camera_ids):
        cam_to_idxs[cam].append(i)
    cameras = sorted(cam_to_idxs)
    cam_scene = {c: extract_scene(c) for c in cameras}

    # Per-stream full cosine matrices (rank features need full neighbourhoods).
    sim_primary = primary @ primary.T
    sim_tert = tertiary @ tertiary.T if tertiary is not None else None
    sim_quat = quaternary @ quaternary.T if quaternary is not None else None

    # Integer code for each unordered camera pair (categorical feature).
    cam_pair_code: Dict[Tuple[str, str], int] = {}

    def pair_code(ci: str, cj: str) -> int:
        key = tuple(sorted((ci, cj)))
        if key not in cam_pair_code:
            cam_pair_code[key] = len(cam_pair_code)
        return cam_pair_code[key]

    # Rank of j among i's cross-camera, same-class candidates by primary cosine.
    def cross_camera_rank(i: int, j: int) -> int:
        ci = run.camera_ids[i]
        same_class_other_cam = [
            k for k in range(n)
            if run.camera_ids[k] != ci and run.class_ids[k] == run.class_ids[i]
            and cam_scene[run.camera_ids[k]] == cam_scene[ci]
        ]
        if not same_class_other_cam:
            return n
        sims = sim_primary[i, same_class_other_cam]
        order = sorted(zip(same_class_other_cam, sims), key=lambda kv: -kv[1])
        for rank, (k, _) in enumerate(order):
            if k == j:
                return rank
        return n

    cols: Dict[str, list] = {name: [] for name in FEATURE_NAMES}
    cols.update({"label": [], "cam_i": [], "cam_j": [], "track_i": [], "track_j": [], "scene": []})

    positives: List[dict] = []
    hard_negatives: List[dict] = []
    easy_negatives: List[dict] = []

    rng = np.random.default_rng(42)

    for a_idx, cam_a in enumerate(cameras):
        scene_a = cam_scene[cam_a]
        for cam_b in cameras[a_idx + 1:]:
            scene_b = cam_scene[cam_b]
            # Scene blocking: only pair cameras within the same scene.
            if scene_a and scene_b and scene_a != scene_b:
                continue
            scene = scene_a or scene_b
            for i in cam_to_idxs[cam_a]:
                gi = run.gt_ids[i]
                if gi is None:
                    continue  # ambiguous tracklet -> excluded
                for j in cam_to_idxs[cam_b]:
                    if run.class_ids[i] != run.class_ids[j]:
                        continue
                    gj = run.gt_ids[j]
                    if gj is None:
                        continue
                    label = 1 if gi == gj else 0

                    cos_p = float(sim_primary[i, j])
                    cos_d = float(sim_tert[i, j]) if sim_tert is not None else 0.0
                    cos_r = float(sim_quat[i, j]) if sim_quat is not None else 0.0
                    cos_fused = w_pri * cos_p + w_tert * cos_d + w_quat * cos_r
                    streams = [cos_p]
                    if sim_tert is not None:
                        streams.append(cos_d)
                    if sim_quat is not None:
                        streams.append(cos_r)
                    cos_min = float(np.min(streams))
                    cos_max = float(np.max(streams))
                    cos_std = float(np.std(streams))

                    rank_j = cross_camera_rank(i, j)
                    rank_i = cross_camera_rank(j, i)
                    is_mutual_top1 = 1 if (rank_j == 0 and rank_i == 0) else 0
                    recip = 0.5 * (1.0 / (rank_j + 1.0) + 1.0 / (rank_i + 1.0))

                    si, ei = run.start_times[i], run.end_times[i]
                    sj, ej = run.start_times[j], run.end_times[j]
                    later_start = max(si, sj)
                    earlier_end = min(ei, ej)
                    time_gap = max(0.0, later_start - earlier_end)
                    if si <= sj:
                        ca, cb = cam_a, cam_b
                    else:
                        ca, cb = cam_b, cam_a
                    st_score = st_validator.transition_score(ca, cb, 0.0, time_gap)
                    t_overlap = compute_temporal_overlap_ratio(si, ei, sj, ej)
                    pair_mean_time, pair_max_time = _pair_prior_times(st_validator, cam_a, cam_b)

                    li = max(int(run.num_frames[i]), 1)
                    lj = max(int(run.num_frames[j]), 1)
                    min_track_len = float(min(li, lj))
                    len_ratio = float(min(li, lj) / max(li, lj))
                    min_mean_conf = float(min(run.mean_confs[i], run.mean_confs[j]))

                    feats = {
                        "cos_primary": cos_p,
                        "cos_dinov2": cos_d,
                        "cos_r50ibn": cos_r,
                        "cos_fused": cos_fused,
                        "cos_min": cos_min,
                        "cos_max": cos_max,
                        "cos_std": cos_std,
                        "rank_i_of_j": float(rank_i),
                        "rank_j_of_i": float(rank_j),
                        "is_mutual_top1": float(is_mutual_top1),
                        "recip_rank_harmonic": float(recip),
                        "time_gap": float(time_gap),
                        "st_score": float(st_score),
                        "temporal_overlap_ratio": float(t_overlap),
                        "camera_pair_id": pair_code(cam_a, cam_b),
                        "pair_mean_time": float(pair_mean_time),
                        "pair_max_time": float(pair_max_time),
                        "min_track_len": min_track_len,
                        "len_ratio": len_ratio,
                        "min_mean_conf": min_mean_conf,
                        "label": label,
                        "cam_i": cam_a,
                        "cam_j": cam_b,
                        "track_i": run.track_ids[i],
                        "track_j": run.track_ids[j],
                        "scene": scene,
                    }
                    if label == 1:
                        positives.append(feats)
                    elif cos_fused >= HARD_NEG_COS_FUSED:
                        hard_negatives.append(feats)
                    else:
                        easy_negatives.append(feats)

    # Subsample the easy negative tail to ~EASY_NEG_RATIO x positives.
    n_pos = len(positives)
    keep_easy = int(EASY_NEG_RATIO * max(n_pos, 1))
    if len(easy_negatives) > keep_easy:
        sel = rng.choice(len(easy_negatives), size=keep_easy, replace=False)
        easy_negatives = [easy_negatives[k] for k in sel]

    kept = positives + hard_negatives + easy_negatives
    rng.shuffle(kept)
    for row in kept:
        for name in FEATURE_NAMES:
            cols[name].append(row[name])
        for extra in ("label", "cam_i", "cam_j", "track_i", "track_j", "scene"):
            cols[extra].append(row[extra])

    print(
        f"  Pairs: {len(kept)} kept "
        f"(pos={n_pos}, hard_neg={len(hard_negatives)}, easy_neg={len(easy_negatives)})"
    )
    return cols


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_table(cols: Dict[str, list], out_path: Path) -> Path:
    """Write column dict to parquet (preferred) or .npz fallback."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd  # noqa

        df = pd.DataFrame(cols)
        try:
            df.to_parquet(out_path, index=False)
            return out_path
        except Exception as exc:
            print(f"  parquet write failed ({exc}); falling back to .npz")
    except Exception as exc:
        print(f"  pandas/pyarrow unavailable ({exc}); writing .npz")
    npz_path = out_path.with_suffix(".npz")
    np.savez_compressed(npz_path, **{k: np.array(v, dtype=object) for k, v in cols.items()})
    return npz_path


# ---------------------------------------------------------------------------
# Separability probe (the GO / NO-GO)
# ---------------------------------------------------------------------------
def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC with a no-sklearn fallback (Mann-Whitney U)."""
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(labels, scores))
    except Exception:
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        order = np.argsort(scores, kind="mergesort")
        ranks = np.empty(len(scores), dtype=np.float64)
        ranks[order] = np.arange(1, len(scores) + 1)
        # average ranks for ties
        _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, ranks)
        avg = sums / counts
        ranks = avg[inv]
        r_pos = ranks[labels == 1].sum()
        u = r_pos - len(pos) * (len(pos) + 1) / 2.0
        return float(u / (len(pos) * len(neg)))


def _train_lgbm(
    X_train: np.ndarray, y_train: np.ndarray, cat_idx: List[int]
) -> "object":
    import lightgbm as lgb

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    spw = (n_neg / max(n_pos, 1)) if n_pos else 1.0
    params = dict(
        objective="binary",
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,          # <= 64 per spec
        max_depth=4,            # <= 4 per spec (shallow)
        min_child_samples=40,   # high, to fight overfit on ~150-300 pos/fold
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=1.0,          # L1
        reg_lambda=5.0,         # L2
        scale_pos_weight=spw,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model = lgb.LGBMClassifier(**params)
    fit_kwargs = {"feature_name": list(FEATURE_NAMES)}
    if cat_idx:
        # LightGBM accepts categorical feature names; pass names to keep the
        # fit/predict feature-name space consistent (silences sklearn warning).
        fit_kwargs["categorical_feature"] = [FEATURE_NAMES[i] for i in cat_idx]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        model.fit(X_train, y_train, **fit_kwargs)
    return model


def separability_report(
    cols: Dict[str, list],
    *,
    pass_margin: float = 0.02,
    fusion_weights: Optional[Tuple[float, float, float]] = None,
) -> dict:
    """Scene-disjoint LightGBM AUC vs cos_fused-threshold baseline.

    Folds: train S02 -> eval S01 (held-out), then mirror. Reports per-fold
    held-out model AUC + baseline AUC on the SAME held-out hard-negative subset,
    the average delta, top-10 importances, and a PASS / NO-GO verdict.
    """
    w_pri, w_tert, w_quat = fusion_weights if fusion_weights is not None else (
        K7_W_PRIMARY, K7_W_TERTIARY, K7_W_QUATERNARY
    )
    scenes = np.array(cols["scene"])
    labels = np.array(cols["label"], dtype=np.int64)
    feature_matrix = np.column_stack([np.array(cols[name], dtype=np.float64) for name in FEATURE_NAMES])
    cat_idx = [FEATURE_NAMES.index(c) for c in CATEGORICAL_FEATURES]
    cos_fused = np.array(cols["cos_fused"], dtype=np.float64)

    unique_scenes = sorted(set(s for s in scenes.tolist() if s))
    print("\n" + "=" * 78)
    print("SEPARABILITY PROBE (scene-disjoint, anti-leakage)")
    print("=" * 78)
    print(f"Scenes present: {unique_scenes}  total_pairs={len(labels)}  positives={int(labels.sum())}")

    if len(unique_scenes) < 2:
        print("WARNING: < 2 scenes present — cannot run scene-disjoint CV. "
              "Reporting baseline-only AUC; verdict = INSUFFICIENT-DATA.")
        base_auc = _auc(labels, cos_fused)
        print(f"Baseline cos_fused AUC (single scene, NOT held-out): {base_auc:.4f}")
        return {"verdict": "INSUFFICIENT-DATA", "baseline_auc": base_auc, "folds": []}

    fold_rows: List[dict] = []
    model_aucs: List[float] = []
    base_aucs: List[float] = []

    for held in unique_scenes:
        train_mask = (scenes != held) & np.isin(scenes, unique_scenes)
        test_mask = scenes == held

        # Anti-leakage assertion: no held-scene camera may appear in training.
        train_cams = set(
            list(np.array(cols["cam_i"])[train_mask]) + list(np.array(cols["cam_j"])[train_mask])
        )
        test_cams = set(
            list(np.array(cols["cam_i"])[test_mask]) + list(np.array(cols["cam_j"])[test_mask])
        )
        leaked = {c for c in test_cams if extract_scene(c) == held} & train_cams
        assert not leaked, f"LEAKAGE: held scene {held} cameras {leaked} found in training set"
        assert all(extract_scene(c) != held for c in train_cams if extract_scene(c)), (
            f"LEAKAGE: training cameras contain held scene {held}: "
            f"{[c for c in train_cams if extract_scene(c) == held]}"
        )

        y_tr, y_te = labels[train_mask], labels[test_mask]
        if y_tr.sum() == 0 or y_te.sum() == 0 or (y_te == 0).sum() == 0:
            print(f"  Fold held={held}: skipped (degenerate label distribution "
                  f"train_pos={int(y_tr.sum())} test_pos={int(y_te.sum())})")
            continue

        model = _train_lgbm(feature_matrix[train_mask], y_tr, cat_idx)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            proba = model.predict_proba(feature_matrix[test_mask])[:, 1]
        m_auc = _auc(y_te, proba)

        # Baseline on the SAME held-out rows.
        b_auc = _auc(y_te, cos_fused[test_mask])

        # Hard-negative subset (the fair, hard comparison): positives + negatives
        # with cos_fused >= HARD_NEG_COS_FUSED in the held-out scene. Require a
        # minimum hard-negative count (MIN_HARD_NEG) so a 1-2 negative subset
        # can't produce a degenerate AUC and a spurious verdict.
        hn = (y_te == 1) | (cos_fused[test_mask] >= HARD_NEG_COS_FUSED)
        n_hard_neg = int((y_te[hn] == 0).sum())
        if hn.sum() > 0 and y_te[hn].sum() > 0 and n_hard_neg >= MIN_HARD_NEG:
            m_auc_hn = _auc(y_te[hn], proba[hn])
            b_auc_hn = _auc(y_te[hn], cos_fused[test_mask][hn])
        else:
            if 0 < n_hard_neg < MIN_HARD_NEG:
                print(f"    NOTE: only {n_hard_neg} hard negatives in held-out {held} "
                      f"(< {MIN_HARD_NEG}); using all-rows AUC for this fold's verdict.")
            m_auc_hn = float("nan")
            b_auc_hn = float("nan")

        model_aucs.append(m_auc_hn if not np.isnan(m_auc_hn) else m_auc)
        base_aucs.append(b_auc_hn if not np.isnan(b_auc_hn) else b_auc)

        # Top-10 importances for this fold.
        imp = model.feature_importances_
        top = sorted(zip(FEATURE_NAMES, imp), key=lambda kv: -kv[1])[:10]

        print(f"\n  Fold: train={[s for s in unique_scenes if s != held]} -> held-out={held}")
        print(f"    train n={int(train_mask.sum())} (pos={int(y_tr.sum())}) | "
              f"test n={int(test_mask.sum())} (pos={int(y_te.sum())})")
        print(f"    model AUC (all)      = {m_auc:.4f}    baseline cos_fused AUC (all)      = {b_auc:.4f}")
        print(f"    model AUC (hard-neg) = {m_auc_hn:.4f}    baseline cos_fused AUC (hard-neg) = {b_auc_hn:.4f}")
        print(f"    delta (hard-neg)     = {(m_auc_hn - b_auc_hn):+.4f}")
        print(f"    top-10 importances: {[(name, int(v)) for name, v in top]}")

        fold_rows.append({
            "held_out_scene": held,
            "train_scenes": [s for s in unique_scenes if s != held],
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "model_auc_all": m_auc,
            "baseline_auc_all": b_auc,
            "model_auc_hardneg": m_auc_hn,
            "baseline_auc_hardneg": b_auc_hn,
            "delta_hardneg": (m_auc_hn - b_auc_hn) if not np.isnan(m_auc_hn) else None,
            "top10_importances": [(name, int(v)) for name, v in top],
        })

    if not model_aucs:
        print("\nVERDICT: INSUFFICIENT-DATA (no usable fold)")
        return {"verdict": "INSUFFICIENT-DATA", "folds": fold_rows}

    mean_model = float(np.nanmean(model_aucs))
    mean_base = float(np.nanmean(base_aucs))
    delta = mean_model - mean_base
    verdict = "PASS" if delta >= pass_margin else "NO-GO (no learnable signal beyond the threshold)"

    print("\n" + "-" * 78)
    print(f"MEAN held-out model AUC (hard-neg)    = {mean_model:.4f}")
    print(f"MEAN held-out baseline cos_fused AUC  = {mean_base:.4f}")
    print(f"MEAN DELTA                            = {delta:+.4f}   (PASS margin >= +{pass_margin:.2f})")
    print(f"VERDICT: {verdict}")
    print("-" * 78)

    return {
        "verdict": verdict,
        "mean_model_auc_hardneg": mean_model,
        "mean_baseline_auc_hardneg": mean_base,
        "mean_delta": delta,
        "pass_margin": pass_margin,
        "folds": fold_rows,
        "k7_weights": {"primary": w_pri, "tertiary": w_tert, "quaternary": w_quat},
    }


# ---------------------------------------------------------------------------
# Synthetic self-test
# ---------------------------------------------------------------------------
def _run_self_test(tmp_root: Path) -> int:
    """Generate a tiny synthetic frozen run + GT and exercise the full pipeline.

    Two scenes (S01: c001/c002/c003, S02: c006/c007/c008), a handful of GT ids
    per scene, each appearing in 2-3 cameras. Embeddings are id-anchored + noise
    so positives have higher cosine than negatives -> a learnable but imperfect
    signal. This validates code paths only; it is NOT the real result.
    """
    print("=" * 78)
    print("SELF-TEST: synthetic frozen run (code-path validation only)")
    print("=" * 78)
    rng = np.random.default_rng(0)
    dim_p, dim_t, dim_q = 384, 64, 64

    cams_by_scene = {"S01": ["S01_c001", "S01_c002", "S01_c003"],
                     "S02": ["S02_c006", "S02_c007", "S02_c008"]}
    # Enough identities per scene that each held-out fold has plenty of positives
    # for a non-degenerate LightGBM (min_child_samples=40 needs a healthy count).
    n_ids_per_scene = 40

    index_map: List[dict] = []
    prim_rows: List[np.ndarray] = []
    tert_rows: List[np.ndarray] = []
    quat_rows: List[np.ndarray] = []
    tracklets_by_cam: Dict[str, list] = defaultdict(list)
    gt_lines_by_cam: Dict[str, List[str]] = defaultdict(list)

    # id anchors live in a shared space; per-stream projections give correlated cosines.
    global_anchor: Dict[int, np.ndarray] = {}

    def anchor(gid: int, dim: int, key: str) -> np.ndarray:
        # Deterministic per-(gid, stream) seed (avoid Python's hash randomization
        # so the self-test is reproducible across processes).
        key_code = {"p": 0, "t": 1, "q": 2}.get(key, 9)
        r = np.random.default_rng(gid * 10 + key_code)
        v = r.standard_normal(dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-8)

    track_counter = 0
    gid_global = 0
    for scene, cams in cams_by_scene.items():
        for _ in range(n_ids_per_scene):
            gid = gid_global
            gid_global += 1
            global_anchor[gid] = anchor(gid, dim_p, "p")
            # appear in 2-3 cameras of this scene
            k = rng.integers(2, len(cams) + 1)
            chosen = list(rng.choice(cams, size=k, replace=False))
            for ci, cam in enumerate(chosen):
                tid = track_counter
                track_counter += 1
                cls = int(rng.choice([2, 2, 2, 5, 7]))
                index_map.append({"track_id": tid, "camera_id": cam, "class_id": cls})

                # Lower noise so positives have high fused cosine and some
                # negatives land in the hard band (cos_fused >= 0.3), exercising
                # the hard-negative AUC path in the separability report.
                ap = global_anchor[gid] + 0.30 * rng.standard_normal(dim_p).astype(np.float32)
                at = anchor(gid, dim_t, "t") + 0.35 * rng.standard_normal(dim_t).astype(np.float32)
                aq = anchor(gid, dim_q, "q") + 0.35 * rng.standard_normal(dim_q).astype(np.float32)
                prim_rows.append(ap)
                tert_rows.append(at)
                quat_rows.append(aq)

                # frames + GT boxes (0-based internal; GT 1-based with x,y,w,h)
                n_fr = int(rng.integers(8, 40))
                start_f = int(rng.integers(0, 50))
                x0 = float(rng.integers(50, 800))
                y0 = float(rng.integers(50, 400))
                w0 = float(rng.integers(60, 140))
                h0 = float(rng.integers(60, 140))

                from src.core.data_models import Tracklet, TrackletFrame

                frames = []
                for f in range(n_fr):
                    fid = start_f + f
                    bx = x0 + 1.5 * f
                    by = y0 + 0.8 * f
                    frames.append(TrackletFrame(
                        frame_id=fid, timestamp=fid / 10.0,
                        bbox=(bx, by, bx + w0, by + h0), confidence=float(rng.uniform(0.4, 0.95)),
                    ))
                    # GT box (1-based frame); near-identical so IoU >= 0.5
                    gt_lines_by_cam[cam].append(
                        f"{fid + 1},{gid},{bx + 1.0:.1f},{by + 1.0:.1f},{w0:.1f},{h0:.1f},1,-1,-1,-1"
                    )
                cname = {2: "car", 5: "bus", 7: "truck"}[cls]
                tracklets_by_cam[cam].append(
                    Tracklet(track_id=tid, camera_id=cam, class_id=cls, class_name=cname, frames=frames)
                )

    # Materialize a frozen-run directory on disk.
    run_dir = tmp_root / "synthetic_run"
    (run_dir / "stage1").mkdir(parents=True, exist_ok=True)
    (run_dir / "stage2").mkdir(parents=True, exist_ok=True)
    gt_root = tmp_root / "synthetic_gt"

    from src.core.io_utils import save_tracklets_by_camera

    save_tracklets_by_camera(dict(tracklets_by_cam), run_dir / "stage1")
    np.save(run_dir / "stage2" / "embeddings.npy", _l2norm(np.array(prim_rows, dtype=np.float32)))
    np.save(run_dir / "stage2" / "embeddings_tertiary.npy", np.array(tert_rows, dtype=np.float32))
    np.save(run_dir / "stage2" / "embeddings_quaternary.npy", np.array(quat_rows, dtype=np.float32))
    (run_dir / "stage2" / "embedding_index.json").write_text(json.dumps(index_map, indent=2), encoding="utf-8")
    for cam, lines in gt_lines_by_cam.items():
        cam_gt = gt_root / cam / "gt"
        cam_gt.mkdir(parents=True, exist_ok=True)
        (cam_gt / "gt.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"  synthetic tracklets: {len(index_map)} across {len(tracklets_by_cam)} cameras")

    # Run the real pipeline functions.
    run = load_run(
        run_dir, gt_root, raw_cosines=False, fic_reg=DEFAULT_FIC_REG,
        fic_min_samples=DEFAULT_FIC_MIN_SAMPLES, aqe_k=DEFAULT_AQE_K,
        aqe_alpha=DEFAULT_AQE_ALPHA, top_k=DEFAULT_TOP_K,
    )
    fusion_weights = read_k7_weights()
    print(f"  K7 weights (registry): {fusion_weights}")
    st_validator = _build_st_validator(_load_camera_transitions())
    cols = build_pairs(run, st_validator, fusion_weights=fusion_weights)
    out = write_table(cols, tmp_root / "edge_pairs_selftest.parquet")
    print(f"  wrote {out}")
    report = separability_report(cols, fusion_weights=fusion_weights)
    ok = report.get("verdict") in {"PASS", "NO-GO (no learnable signal beyond the threshold)"}
    print(f"\nSELF-TEST {'OK' if ok else 'FAILED'} (verdict={report.get('verdict')})")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, help="Frozen run dir with stage1/ + stage2/ artifacts.")
    ap.add_argument("--gt-root", type=Path, help="GT root with <CAM>/gt/gt.txt.")
    ap.add_argument("--out-dir", type=Path, default=Path("."), help="Where to write edge_pairs_<scene>.parquet.")
    ap.add_argument("--raw-cosines", action="store_true",
                    help="Use plain L2-normalized cosines instead of FIC(+AQE) (weakens signal; prints a warning).")
    ap.add_argument("--fic-reg", type=float, default=DEFAULT_FIC_REG)
    ap.add_argument("--fic-min-samples", type=int, default=DEFAULT_FIC_MIN_SAMPLES)
    ap.add_argument("--aqe-k", type=int, default=DEFAULT_AQE_K)
    ap.add_argument("--aqe-alpha", type=float, default=DEFAULT_AQE_ALPHA)
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--pass-margin", type=float, default=0.02,
                    help="Min mean held-out AUC gain over baseline for a PASS verdict.")
    ap.add_argument("--self-test", action="store_true", help="Run synthetic end-to-end self-test and exit.")
    args = ap.parse_args(argv)

    if args.self_test:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            return _run_self_test(Path(td))

    if not args.run_dir or not args.gt_root:
        ap.error("--run-dir and --gt-root are required (or pass --self-test).")
    if not args.run_dir.exists():
        ap.error(f"--run-dir does not exist: {args.run_dir}")
    if not args.gt_root.exists():
        ap.error(f"--gt-root does not exist: {args.gt_root}")

    fusion_weights = read_k7_weights()
    print(f"Run dir : {args.run_dir}")
    print(f"GT root : {args.gt_root}")
    print(f"Feature space: {'RAW L2-cosines (DEGRADED)' if args.raw_cosines else 'FIC+AQE (matches live gate)'}")
    print(f"K7 fusion weights (from registry): "
          f"w_primary={fusion_weights[0]}, w_tertiary={fusion_weights[1]}, w_quaternary={fusion_weights[2]}")

    run = load_run(
        args.run_dir, args.gt_root, raw_cosines=args.raw_cosines, fic_reg=args.fic_reg,
        fic_min_samples=args.fic_min_samples, aqe_k=args.aqe_k, aqe_alpha=args.aqe_alpha, top_k=args.top_k,
    )
    st_validator = _build_st_validator(_load_camera_transitions())
    cols = build_pairs(run, st_validator, fusion_weights=fusion_weights)

    # Emit per-scene tables.
    scenes_present = sorted(set(s for s in cols["scene"] if s))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if scenes_present:
        for scene in scenes_present:
            sub = {k: [v for v, s in zip(vals, cols["scene"]) if s == scene] for k, vals in cols.items()}
            out = write_table(sub, args.out_dir / f"edge_pairs_{scene}.parquet")
            print(f"  wrote {out} ({len(sub['label'])} rows)")
    else:
        out = write_table(cols, args.out_dir / "edge_pairs_all.parquet")
        print(f"  wrote {out} ({len(cols['label'])} rows)")

    report = separability_report(cols, pass_margin=args.pass_margin, fusion_weights=fusion_weights)
    (args.out_dir / "separability_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote separability report: {args.out_dir / 'separability_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
