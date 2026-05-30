# CityTrack Missing-Components — Implementation Spec

> Read-only design spec. NO code changed. Companion to `docs/subagent-specs/citytrack-audit-evidence.md`.
> Three components, all behind **default-OFF** config flags, zero impact on the current best (CityFlowV2 MTMC IDF1 = **0.77936**, 14e B1).
> All line numbers verified on disk on 2026-05-30.

---

## 0. Verified integration points (cite these exactly)

| Concern | File:line | Fact |
|---|---|---|
| Stage-1 per-camera loop | `src/stage1_tracking/pipeline.py:115` (`for camera_id, cam_frames ...`) | Tracker + builder created fresh **inside** this loop |
| Detector created once | `src/stage1_tracking/pipeline.py:84` | Stateless per-frame (`detector.detect(frame)`) — safe to reuse for a 2nd pass |
| Tracker created per camera | `src/stage1_tracking/pipeline.py:124` (`tracker = TrackerWrapper(...)`) | **Stateful** — a 2nd pass needs a NEW instance |
| Builder created per camera | `src/stage1_tracking/pipeline.py:133` (`builder = TrackletBuilder(...)`) | One builder per pass |
| Per-frame update loop | `src/stage1_tracking/pipeline.py:144-171` | `frame=imread`; `dets=detector.detect`; `tracks=tracker.update(dets,frame)`; `builder.add_frame(tracks, frame_info.frame_id, frame_info.timestamp)` |
| Tracklet finalize | `src/stage1_tracking/pipeline.py:174` (`builder.finalize()`) | Returns `List[Tracklet]` |
| Builder add_frame | `src/stage1_tracking/tracklet_builder.py:218-252` | Stores `TrackletFrame(frame_id, timestamp, bbox, confidence)` keyed by `track_id` |
| TrackletFrame model | `src/core/data_models.py:44-50` | Has `frame_id, timestamp, bbox, confidence` — **per-box confidence already stored** |
| Tracklet model | `src/core/data_models.py:53-90` | `frames: List[TrackletFrame]`; has `mean_confidence`, `get_bbox_at` |
| Stage-4 entry | `src/stage4_association/pipeline.py:115` (`run_stage4`) | Receives BOTH `features: List[TrackletFeatures]` AND `tracklets_by_camera: Dict[str,List[Tracklet]]` (param at `:120`) |
| Combined-sim call site | `src/stage4_association/pipeline.py:512-523` | `combined_sim = compute_combined_similarity(...)` — the distance/sim finalization |
| Combined-sim function | `src/stage4_association/similarity.py:97` (`def compute_combined_similarity`) | Returns `Dict[(i,j), float]`; final assignment `combined[(i,j)] = score` at `similarity.py:251`, `return combined` at `:259` |
| stage1 config block | `configs/default.yaml:23` (detector `:24`, tracker `:33`, interpolation `:58`, intra_merge `:62`) | Insert `ssa:` and `bidirectional:` here |
| stage4.association block | `configs/default.yaml:175` | Insert `occlusion_aware:` here |
| cityflowv2 overrides | `configs/datasets/cityflowv2.yaml` (stage1 ~`:1`, stage4.association ~`:161-240`) | Per-dataset overrides — leave flags OFF here too |

**Key data-flow fact:** `run_stage4` already gets `tracklets_by_camera` with per-frame boxes. Occlusion can be computed **entirely inside Stage-4** from stored boxes grouped by `(camera, frame_id)` — **no data-model change, no Stage-1 carry-forward required.** `TrackletFeatures` (`data_models.py:101`) carries `track_id` + `camera_id`, so `feature_index i -> Tracklet` is a clean `(camera_id, track_id)` lookup.

---

## 1. SSA — Stationary Sensitive Association (Stage-1)

**Idea (CityTrack):** when a tracklet is stationary (center displacement < thresh over a window), the Kalman prediction drifts; freeze the highest-confidence recent detection box instead.

**Where it fits:** Our tracker (`deepocsort`) already returns observed boxes per frame; drift shows up as jittery/low-conf boxes during stops. SSA is applied as a **post-hoc box-stabilization pass on each finalized Tracklet**, NOT inside BoxMOT's KF (which we must not patch). This keeps it isolated and default-off-safe.

### Files / functions
- NEW: `src/stage1_tracking/ssa.py`
  - `def apply_ssa(tracklets: List[Tracklet], cfg: dict) -> List[Tracklet]`
  - Helpers: `_center(bbox)`, `_window_displacement(frames, i, window)`.

### Algorithm (per tracklet, per frame index `t`)
1. Compute center displacement over a backward window `W`: `disp = max ||center(t) - center(t-k)||` for `k in 1..W` (frames present in tracklet).
2. If `disp < disp_thresh` (pixels): tracklet is stationary at `t`.
3. While stationary, replace `frames[t].bbox` with the box of the **highest-confidence real detection** (confidence > 0, i.e. not interpolated) within the last `freeze_lookback` frames. Keep `frames[t].confidence` unchanged (or set to that frozen box's confidence — config `inherit_conf`).
4. Reset the "frozen anchor" whenever `disp >= disp_thresh` (motion resumes).

### Insertion point
`src/stage1_tracking/pipeline.py`, immediately after `tracklets = builder.finalize()` at `:174`, before `all_tracklets[camera_id] = tracklets` at `:175`:
```python
from src.stage1_tracking.ssa import apply_ssa  # top-of-file import
ssa_cfg = stage_cfg.get("ssa", {})
if ssa_cfg.get("enabled", False):
    tracklets = apply_ssa(tracklets, ssa_cfg)
```

### Config (default.yaml, insert after intra_merge block at :64)
```yaml
  # --- SSA: Stationary Sensitive Association (CityTrack) ---
  ssa:
    enabled: false        # default OFF — zero impact on 0.77936
    window: 10            # frames to measure displacement over
    disp_thresh: 8.0      # px; below => stationary
    freeze_lookback: 15   # frames to search back for highest-conf box
    inherit_conf: false   # if true, frozen frame takes anchor box confidence
```

### Edge cases
- **Cold start** (fewer than `window` frames seen): skip — never stationary until window filled.
- **Interpolated frames** (`confidence == 0`, `tracklet_builder.py:60`): excluded as freeze anchors but their boxes may still be overwritten.
- **Tracklet shorter than window:** no-op.
- Idempotent: re-running SSA on already-frozen tracklet yields same result.

---

## 2. BT — Bidirectional Tracking (Stage-1)

### FEASIBILITY VERDICT: **CLEAN** (true second pass is achievable, not a fake merge)
Evidence: tracker (`pipeline.py:124`) and builder (`:133`) are instantiated fresh **inside** the per-camera loop; the detector (`:84`) is created once and is stateless per frame. Therefore a genuine backward pass = a second `TrackerWrapper` + second `TrackletBuilder` iterating `reversed(cam_frames)`. Because `builder.add_frame` is passed the **absolute** `frame_info.frame_id` (not a loop index, `pipeline.py:168-170`), backward-pass tracklets carry **correct absolute frame_ids/timestamps with NO remap** needed. This is a real bidirectional tracking pass — boundary detections the forward pass missed (objects entering/exiting at sequence start, or recovered after late confirmation) get tracked from the reverse direction.

One required guard: backward-pass `track_id`s collide with forward-pass ids → offset backward ids by a large constant (`+1_000_000`) before merge.

### Files / functions
- NEW: `src/stage1_tracking/bidirectional.py`
  - `def run_backward_pass(cam_frames, detector, stage_cfg, camera_id) -> List[Tracklet]` — mirrors the forward loop over `reversed(cam_frames)` with a fresh tracker+builder; returns finalized backward tracklets (ids offset).
  - `def merge_bidirectional(fwd: List[Tracklet], bwd: List[Tracklet], features_fn, cfg) -> List[Tracklet]` — merge fwd/bwd tracklets.

### Merge logic
For each (fwd, bwd) pair sharing >= `min_shared_frames` frames:
- `iou = mean IoU over shared frames` (reuse `tracklet_builder._compute_iou` at `tracklet_builder.py:155`).
- Optional ReID cosine gate (Stage-1 has NO embeddings yet; appearance not available here). **Honest limitation:** at Stage-1 we can only merge on **IoU over shared frames** (geometric). ReID-cosine merge belongs to Stage-2+ where embeddings exist. So BT-merge here is **IoU-only**.
- If `iou >= iou_thresh`: take the **union of frames**; for overlapping frame_ids prefer the higher-confidence box. This recovers boundary frames present in only one direction.
- Unmatched backward-only tracklets that pass `min_tracklet_length` are appended (recovered tracks).

### Insertion point
`src/stage1_tracking/pipeline.py`, replacing the single forward finalize. After `tracklets = builder.finalize()` at `:174`:
```python
bt_cfg = stage_cfg.get("bidirectional", {})
if bt_cfg.get("enabled", False):
    from src.stage1_tracking.bidirectional import run_backward_pass, merge_bidirectional
    bwd = run_backward_pass(cam_frames, detector, stage_cfg, camera_id)
    tracklets = merge_bidirectional(tracklets, bwd, None, bt_cfg)
```
(SSA, if also enabled, runs AFTER the BT merge.)

### Config (default.yaml, after ssa block)
```yaml
  # --- BT: Bidirectional Tracking (CityTrack) — IoU-merge only at Stage-1 ---
  bidirectional:
    enabled: false        # default OFF
    iou_thresh: 0.5       # mean-IoU over shared frames to merge fwd/bwd
    min_shared_frames: 3
    reid_merge: false     # NOT supported at Stage-1 (no embeddings); reserved
```

### Edge cases
- **Frame-index remap:** NONE needed (absolute frame_id passed through). Document this explicitly so the coder doesn't add a spurious `len-1-i` remap.
- **track_id collision:** offset backward ids by `+1_000_000` in `run_backward_pass`.
- **Cost:** doubles Stage-1 detection+tracking time. Detections from the forward pass MAY be cached and replayed to the backward tracker (deterministic detector) — optional optimization, NOT required for correctness.
- **Determinism:** BoxMOT/DeepOCSort has internal randomness disabled by config; backward pass reproducible.

---

## 3. Occlusion-Aware Distance (Stage-4)

**Idea (CityTrack):** per-box occlusion rate `I = max IoU with other boxes in same frame`; `D_final = D x (1 + 0.1 x [occ >= 0.6])`. We work in **similarity** space (higher=closer), so the equivalent is a **similarity shrink** for occluded tracklets.

### Per-tracklet occlusion fraction (computed in Stage-4, no data-model change)
1. Group ALL boxes from `tracklets_by_camera` by `(camera_id, frame_id)`.
2. For each box: `occ_box = max IoU with every other box in same (camera, frame)`.
3. Per tracklet: `occ_frac = fraction of its real-detection frames with occ_box >= occ_box_thresh (0.6)`.
4. Tracklet is "occluded" if `occ_frac >= occ_frac_thresh`.

### Files / functions
- NEW: `src/stage4_association/occlusion.py`
  - `def compute_tracklet_occlusion(tracklets_by_camera, cfg) -> Dict[Tuple[str,int], bool]` keyed by `(camera_id, track_id)`.
  - Reuses an IoU helper (or imports `_compute_iou` from `tracklet_builder`).

### Insertion point
Two-part wiring, both default-gated:
1. In `run_stage4` (`pipeline.py`), after metadata arrays are built (`camera_ids`/`class_ids` at `pipeline.py:154-155`), compute the occluded-flag list aligned to `features` order:
```python
occ_cfg = stage_cfg.get("occlusion_aware", {})
occluded_flags = None
if occ_cfg.get("enabled", False):
    from src.stage4_association.occlusion import compute_tracklet_occlusion
    occ_map = compute_tracklet_occlusion(tracklets_by_camera, occ_cfg)
    occluded_flags = [occ_map.get((f.camera_id, f.track_id), False) for f in features]
```
2. Pass `occluded_flags` + `occ_cfg` into `compute_combined_similarity` (call site `pipeline.py:512`). Inside `similarity.py`, just before `combined[(i,j)] = score` at `:251`:
```python
if occluded_flags is not None and (occluded_flags[i] or occluded_flags[j]):
    score *= occ_penalty   # e.g. 1/(1+0.1) = 0.909  (mirrors D x 1.1)
```
Add two params to `compute_combined_similarity` signature (`similarity.py:97`): `occluded_flags: Optional[List[bool]] = None, occ_penalty: float = 1.0`. Defaults make it a strict no-op when the flag is off.

### Config (default.yaml, insert in stage4.association after `:175`, e.g. near temporal_overlap)
```yaml
    # --- Occlusion-Aware Distance (CityTrack) ---
    occlusion_aware:
      enabled: false       # default OFF — zero impact on 0.77936
      occ_box_thresh: 0.6  # IoU above which a box counts as occluded
      occ_frac_thresh: 0.3 # fraction of frames occluded => tracklet "occluded"
      penalty: 0.909       # similarity multiplier (= 1/1.1, mirrors D x (1+0.1))
```
Override path is **`stage4.association.occlusion_aware.enabled=true`** (NOT `stage4.occlusion_aware`).

### Edge cases
- **< 2 boxes in a frame:** `occ_box = 0` (no other box) — never occluded.
- **Interpolated boxes** (`confidence == 0`): excluded from occlusion counting (they are synthetic).
- **Tracklet with 0 real frames:** `occ_frac = 0`.
- **Penalty direction:** in similarity space penalty < 1.0 (shrink). Do NOT add (that would reward occlusion).

---

## 4. S02 Ablation Plan (Kaggle)

### Kernel chain (repo rule: GPU stages on Kaggle only; GTX 1050 Ti local is forbidden for stages 0/1/2)
- SSA & BT change **Stage-1 output** => must re-run Stage-1->2 (GPU) then 3->5.
  - Clone `notebooks/kaggle/10a_stages012` (GPU T4, `enable_gpu:true`) -> `10b_stage3` (CPU) -> `10c_stages45` (CPU, TrackEval).
- Occlusion changes **Stage-4 only** => can reuse baseline cached Stage-2 features.
  - Clone `notebooks/kaggle/14v_verify_b1_from_yaml` (CPU, `enable_gpu:false`; sources `14c-tta-stage2` + `10a-stages-0-2`) and just add the override.

Restrict to **S02 only** via the dataset/camera filter the verify kernels already use (S02_c006/c007/c008). The catastrophic scene is **S02_c006 (74.0% IDF1)** — SSA's primary target.

### Ablation matrix (5 runs, slugs <= 2 hyphens, auth `$env:KAGGLE_API_TOKEN`)

| Run | Slug | Kernel cloned | GPU | Override string(s) |
|---|---|---|---|---|
| R0 baseline | `citytrack-s02-base` | 10a->10b->10c | yes | (none — all flags default off) |
| R1 SSA | `citytrack-s02-ssa` | 10a->10b->10c | yes | `stage1.ssa.enabled=true` |
| R2 BT | `citytrack-s02-bt` | 10a->10b->10c | yes | `stage1.bidirectional.enabled=true` |
| R3 occ | `citytrack-s02-occ` | 14v (reuse R0 feats) | no | `stage4.association.occlusion_aware.enabled=true` |
| R4 all-on | `citytrack-s02-all` | 10a->10b->10c | yes | `stage1.ssa.enabled=true stage1.bidirectional.enabled=true stage4.association.occlusion_aware.enabled=true` |

R3 reuses R0's Stage-2 features (occlusion is Stage-4-only) -> no extra GPU.

### Metric to read
**S02 MTMC IDF1** from Stage-5 TrackEval. Lands in `10c`/`14v` output as the Stage-5 report (`data/outputs/<run>/stage5/.../*pedestrian*`-style TrackEval summary; vehicle scene = `S02` row). Compare each run's S02 MTMC IDF1 vs R0. Record per-camera, especially **S02_c006**.

### The 3 config-flag override strings
1. `stage1.ssa.enabled=true`
2. `stage1.bidirectional.enabled=true`
3. `stage4.association.occlusion_aware.enabled=true`

---

## 5. Risk + Rollback

### Most likely to regress: **BT (#2)**, then occlusion (#3)
- **BT** doubles tracks and can **merge two different vehicles** that share frames at an intersection (S02_c006 is dense) -> ID switches -> IDF1 drop. It also can resurrect low-quality boundary boxes. Highest fragmentation risk. (Note: forward-only is our current proven config.)
- **Occlusion** penalty can wrongly shrink true cross-camera matches at busy intersections (where the SAME vehicle is legitimately occluded in both views) — could suppress correct S02_c006<->c008 links (the exact hard pairs we boost elsewhere).
- **SSA** is lowest-risk: it only stabilizes boxes of already-confirmed stationary tracks; worst case neutral. But note Stage-5 `_filter_stationary` (disp=150) REMOVES parked cars — SSA-frozen stationary tracks may be MORE likely to be filtered. Watch for interaction; SSA target is *slow/stopped-at-light* vehicles, not fully parked.

### Zero-impact guarantee on 0.77936
Every hook is wrapped in `if cfg...get("enabled", False)`. With all three flags absent/false: `apply_ssa` not called, no backward pass, `occluded_flags=None` => `compute_combined_similarity` takes the existing path with default args (`occluded_flags=None, occ_penalty=1.0`) producing **byte-identical** `combined` dict. The 14e B1 config (which sets none of these keys) is therefore provably unchanged.

### pytest tests to add
- `tests/test_stage1/test_ssa.py`
  - stationary synthetic tracklet (jittery boxes) -> frozen to highest-conf box; moving tracklet -> unchanged; cold-start (< window) -> no-op; disabled-flag path returns input unchanged.
- `tests/test_stage1/test_bidirectional.py`
  - backward pass recovers a boundary detection dropped by forward; track_id offset prevents collision; absolute frame_ids preserved (NO remap); IoU-merge unions frames and prefers higher-conf box; disabled-flag => identical to forward.
- `tests/test_stage4/test_occlusion.py`
  - 2 overlapping boxes (IoU>=0.6) flagged occluded; single box/frame => not occluded; interpolated (conf=0) boxes excluded; `occluded_flags=None` => `compute_combined_similarity` output bitwise-equal to current; penalty shrinks (not grows) similarity.

### Blocker needing user decision before coding
- **BT real-vs-fake honesty:** Stage-1 BT is **IoU-only** (no embeddings exist yet at Stage-1). A ReID-cosine-gated merge would have to live in Stage-2+. Confirm whether IoU-only Stage-1 BT is acceptable, or whether BT should be deferred/re-sited to operate after Stage-2 embeddings (more faithful to CityTrack, more invasive). This is the one design fork the coder cannot resolve alone.

---

## BT — ReID-Gated Cross-Stage Design (REVISED)

> Supersedes the IoU-only Stage-1 BT in section 2. Decision: **REJECT IoU-only Stage-1 merge** (too risky at the dense S02_c006 intersection). BT now merges forward/backward tracklets using **IoU on shared frames AND ReID-embedding cosine**. Because Stage-1 has no embeddings (they are produced in Stage-2), the backward tracklets **flow through Stage-2 feature extraction** and the merge runs **at the end of Stage-2** (after embeddings exist, before Stage-3/Stage-4). RESEARCH/SPEC ONLY — no code changed. All line numbers verified on disk 2026-05-30.

### R0. Verified integration points (cite exactly)

| Concern | File:line | Fact |
|---|---|---|
| Per-camera loop | `src/stage1_tracking/pipeline.py:115` (`for camera_id, cam_frames ...`) | tracker+builder created fresh inside loop |
| Detector created once, stateless | `src/stage1_tracking/pipeline.py:85` (`detector = Detector(...)`) | `detector.detect(frame)` per frame — reusable for a 2nd pass |
| Tracker per camera (stateful) | `src/stage1_tracking/pipeline.py:124` (`tracker = TrackerWrapper(...)`) | 2nd pass needs a NEW instance |
| Builder per camera | `src/stage1_tracking/pipeline.py:133` (`builder = TrackletBuilder(...)`) | one builder per pass |
| Per-frame update | `src/stage1_tracking/pipeline.py:144-171` | `frame=imread`; `dets=detector.detect(masked)`; `tracks=tracker.update(dets, frame)`; `builder.add_frame(tracks, frame_info.frame_id, frame_info.timestamp)` |
| Absolute frame_id passed | `src/stage1_tracking/pipeline.py:164-168` | `add_frame(... frame_id=frame_info.frame_id ...)` — **absolute**, so reversed pass needs NO frame remap |
| Finalize | `src/stage1_tracking/pipeline.py:174` (`tracklets = builder.finalize()`) | returns `List[Tracklet]` |
| Store per camera | `src/stage1_tracking/pipeline.py:175` (`all_tracklets[camera_id] = tracklets`) | merge target dict |
| `Tracklet` model | `src/core/data_models.py:56` | `track_id, camera_id, class_id, class_name, frames: List[TrackletFrame]` — **NO `direction`, NO `embedding` field** |
| `TrackletFrame` model | `src/core/data_models.py:46-50` | `frame_id, timestamp, bbox, confidence` — **per-box confidence already stored** (enables higher-conf-wins on overlap) |
| `TrackletFeatures` model | `src/core/data_models.py:100` | `track_id, camera_id, class_id, embedding (PCA-whitened L2-normed), hsv_histogram, raw_embedding, multi_query_embeddings` |
| Stage-2 entry | `src/stage2_features/pipeline.py:107` (`def run_stage2(cfg, tracklets_by_camera, ...)`) | consumes Stage-1 tracklets, returns `List[TrackletFeatures]` |
| Stage-2 finalize embeddings | `src/stage2_features/pipeline.py:712` (`embeddings = l2_normalize(embeddings)`) | final primary matrix (= Stage-4 matching space) |
| Stage-2 assign per-feat | `src/stage2_features/pipeline.py:720-721` (`feat.embedding = embeddings[i]`) | per-tracklet final embedding set |
| Stage-2 SAVE block | `src/stage2_features/pipeline.py:729-760` | saves `embeddings.npy` + `index_map`, `hsv`, `embeddings_secondary.npy` (`:742`), `embeddings_tertiary.npy` (`:753`), multi-query (`:733`) |
| Stage-2 return | `src/stage2_features/pipeline.py:773` (`return all_features`) | |
| Stage-2 call site | `scripts/run_pipeline.py:154` (`features = run_stage2(...)`) | tracklets dict passed by reference |
| Stage-3 call site | `scripts/run_pipeline.py:191` | uses `features` + `tracklets_by_camera` |
| Stage-4 entry | `src/stage4_association/pipeline.py:115` (`def run_stage4`) | gets BOTH `features` + `tracklets_by_camera` |
| Stage-4 primary stack | `src/stage4_association/pipeline.py:151` (`embeddings = np.stack([f.embedding ...])`) | row order == `features` order |
| Stage-4 secondary align | `src/stage4_association/pipeline.py:183` (`if sec_raw.shape[0] == n:`) | **positional row alignment to `features` is REQUIRED** (same for tertiary/quaternary) |
| Combined-sim call | `src/stage4_association/pipeline.py:513` (`combined_sim = compute_combined_similarity(...)`) | downstream of the merged feature set |

**Key data-flow fact:** Stage-4 loads `embeddings_secondary.npy` / `embeddings_tertiary.npy` (and multi-query) from disk and asserts `shape[0] == n` where `n = len(features)` (`pipeline.py:183`). **Any merge that reduces the tracklet count MUST collapse ALL these positionally-aligned arrays identically** — otherwise the ensemble rows misalign and Stage-4 silently drops fusion (`shape mismatch` warning). This is the single non-trivial constraint and it dictates the merge location.

### R1. Backward pass (Stage-1)

When `stage1.bidirectional.enabled=true`, after the forward `builder.finalize()` at `pipeline.py:174`, run a **genuine second pass** over `reversed(cam_frames)` with a fresh tracker + builder, then carry **both** forward and backward tracklets forward into Stage-2 (do NOT merge here — no embeddings exist yet).

- NEW: `src/stage1_tracking/bidirectional.py`
  - `def run_backward_pass(cam_frames, detector, stage_cfg, camera_id, *, min_tracklet_length, min_tracklet_area, interpolate, interpolation_max_gap, intra_merge, merge_max_time_gap, merge_max_iou_distance, roi_mask) -> List[Tracklet]`
  - Mirrors `pipeline.py:124-174` exactly but iterates `reversed(cam_frames)`. A fresh `TrackerWrapper` and `TrackletBuilder` are created inside. `add_frame` still receives the **absolute** `frame_info.frame_id` and `frame_info.timestamp`, so backward tracklets carry correct absolute frame ids — **NO `len-1-i` remap** (the coder must not add one).
  - Before returning, offset every backward `track_id` by `BWD_ID_OFFSET = 1_000_000` to guarantee no collision with forward ids.

**Distinctness without a schema change:** the `+1_000_000` id offset is the canonical tag — `track_id >= 1_000_000` ⇒ backward. No new dataclass field is required. (Optional convenience: add `direction: str = "forward"` to `Tracklet` at `data_models.py:56`; default value keeps the disabled path byte-identical. Not required — recommend skipping to avoid touching the schema.)

**Insertion in `run_stage1`** — replace the single finalize at `pipeline.py:174-175`:
```python
tracklets = builder.finalize()
bt_cfg = stage_cfg.get("bidirectional", {})
if bt_cfg.get("enabled", False):
    from src.stage1_tracking.bidirectional import run_backward_pass
    bwd = run_backward_pass(cam_frames, detector, stage_cfg, camera_id, ...)
    tracklets = tracklets + bwd           # carry BOTH sets; merge happens in Stage-2
all_tracklets[camera_id] = tracklets
```
Both sets are saved by the existing `save_tracklets_by_camera` and flow unchanged into Stage-2.

### R2. Stage-2 — both sets embedded in ONE pass (CLEAN, no schema change)

`run_stage2` iterates whatever tracklets it is given (`pipeline.py:107` consumes `tracklets_by_camera`), producing one `TrackletFeatures` per tracklet. Because forward and backward tracklets are simply additional entries in each camera's list, **both sets receive embeddings in the same single Stage-2 pass** — including primary, secondary, tertiary, multi-query, camera-BN and the shared global PCA fit. No flag, no schema change needed for embedding extraction. Backward features are identifiable by `feat.track_id >= 1_000_000`.

Honest caveat: with BT on, the global PCA is fit over forward+backward rows (more, near-duplicate samples). This slightly perturbs the PCA basis vs single-pass — **only when BT is enabled** (disabled path is untouched, see R6).

### R3. Merge step (NEW module, invoked at end of Stage-2)

- NEW: `src/stage1_tracking/bidirectional_merge.py`
- Signature:
```python
def merge_bidirectional(
    features: list[TrackletFeatures],          # final, primary .embedding populated
    tracklets_by_camera: dict[str, list[Tracklet]],
    aligned_matrices: dict[str, np.ndarray | None],  # {"primary": embeddings, "hsv": hsv_matrix,
                                                      #  "secondary": sec_matrix, "tertiary": tert_matrix}
    index_map: list[dict],
    cfg: dict,                                 # stage1.bidirectional.*
) -> tuple[list[TrackletFeatures], dict, dict, list[dict]]:
    """Collapse forward/backward tracklets of the same object. Returns
    (merged_features, merged_tracklets_by_camera, merged_aligned_matrices, merged_index_map)
    with ALL rows kept in lockstep so Stage-4 positional alignment holds."""
```

**Algorithm (per camera):**
1. Split into `fwd = [t for t in cam if t.track_id < 1_000_000]` and `bwd = [t for t in cam if t.track_id >= 1_000_000]`.
2. Candidate pairs: every `(f, b)` sharing `>= min_shared_frames` frame_ids (use `Tracklet.get_bbox_at`, `data_models.py:87`, or a frame_id-keyed dict).
3. **IoU gate:** `mean IoU over shared frame_ids >= iou_thresh` (reuse `tracklet_builder._compute_iou`, `tracklet_builder.py:155`).
4. **ReID gate:** cosine between the two **final primary embeddings** (`features[i_f].embedding · features[i_b].embedding`, already L2-normed) `>= cos_thresh`. Using the final PCA-whitened embedding makes the gate operate in the exact Stage-4 matching space.
5. If BOTH gates pass → match. Resolve to a **one-to-one** assignment per camera greedily by descending (cos + iou) score (a backward tracklet matches at most one forward tracklet and vice-versa).
6. **Merge a matched pair:** keep the forward `Tracklet`; **union frames** by frame_id; for any frame_id present in both, **keep the higher-`confidence` box** (`TrackletFrame.confidence`); re-sort frames by frame_id. Recompute the merged `TrackletFeatures.embedding` as the **L2-normalized mean** of the two final primary embeddings (`embedding = l2_normalize(0.5*(e_f + e_b))`). Apply the **same mean+renormalize** to the `secondary`/`tertiary` rows, and element-wise mean to the `hsv` row.
7. **Collapse ALL aligned arrays identically:** drop the backward row from `features`, `index_map`, and every matrix in `aligned_matrices` (primary/hsv/secondary/tertiary), writing the pooled vector into the surviving forward row. This preserves positional alignment required at `stage4/pipeline.py:183`.
8. **Unmatched backward tracklets:** **KEEP them** (recommended) as standalone tracks (strip the `+1_000_000` offset by reassigning a fresh non-colliding id, e.g. `max_fwd_id + k`). Rationale: backward-only tracks recover boundary detections (objects entering/exiting at sequence start) that the forward pass missed — the entire point of BT. Gate keep with `min_tracklet_length` (already enforced upstream).

**Config (`configs/default.yaml`, under the `stage1.bidirectional` block from section 2, replacing `reid_merge`):**
```yaml
  bidirectional:
    enabled: false          # default OFF — byte-identical to current
    min_shared_frames: 3    # min overlapping frame_ids to consider a pair
    iou_thresh: 0.5         # mean IoU over shared frames
    cos_thresh: 0.5         # mean cosine on final PCA-whitened embeddings
    keep_unmatched_backward: true   # recover boundary-only tracks
    pool: mean              # pooled-embedding strategy (mean = l2norm(mean(e_f,e_b)))
```
Override path: `stage1.bidirectional.enabled=true` (NOT `stage1.bt.*`).

### R4. Exact merge insertion point (file : function : line)

**`src/stage2_features/pipeline.py` : `run_stage2` : between line 728 and line 729** — i.e. AFTER the per-feature embedding assignment loop (`:720-727`, where `feat.embedding` is final) and the secondary/tertiary matrices are finalized (`sec_matrix` ~`:741`, `tert_matrix` ~`:752` — NOTE: move the merge to AFTER those two are L2-normalized, or pass their pre-save values in), and **BEFORE the Save block (`:729` "Save outputs")**:

```python
# --- BT cross-stage merge (default OFF) ---
bt_cfg = OmegaConf.select(cfg, "stage1.bidirectional", default={}) or {}
if bt_cfg.get("enabled", False):
    from src.stage1_tracking.bidirectional_merge import merge_bidirectional
    hsv_matrix = np.stack([f.hsv_histogram for f in all_features], axis=0)
    aligned = {"primary": embeddings, "hsv": hsv_matrix,
               "secondary": sec_matrix, "tertiary": tert_matrix}
    all_features, tracklets_by_camera, aligned, index_map = merge_bidirectional(
        all_features, tracklets_by_camera, aligned, index_map, bt_cfg)
    embeddings = aligned["primary"]; hsv_matrix = aligned["hsv"]
    sec_matrix = aligned["secondary"]; tert_matrix = aligned["tertiary"]
```
Everything saved at `:729-760` then reflects the merged, collapsed, positionally-aligned set. `tracklets_by_camera` is mutated in place (same object held by `run_pipeline.py:154`), so Stage-3 (`run_pipeline.py:191`) and Stage-4 see merged single-pass tracklets with **no signature change** to `run_stage2`.

**Why end-of-Stage-2 and not a free-standing post-Stage-2 script in `run_pipeline`:** the secondary/tertiary/multi-query matrices only coexist in memory inside `run_stage2` right before the save block. A post-Stage-2 module would have to re-load and re-write four positionally-coupled npy files on disk to stay aligned (`stage4/pipeline.py:183`) — strictly more fragile. End-of-Stage-2 is the one point where all aligned arrays are in memory simultaneously.

### R5. Edge cases

- **Backward tracklet with no forward match:** KEEP (default `keep_unmatched_backward=true`); reassign a fresh id (`max_fwd_id + k`) to drop the `1e6` offset. This is the boundary-detection recovery payoff. (Set false to discard if S02 ablation shows fragmentation.)
- **Overlapping-frame box conflict:** on shared frame_ids keep the box with higher `TrackletFrame.confidence`; ties → keep forward.
- **Pooled-embedding vs 280D-PCA-whitening consistency:** the merged embedding is `l2_normalize(mean(e_f, e_b))` of two vectors **already** in the final PCA-whitened, L2-normed space — it stays in that space (consistent with how temporal/multi-query pooling already averages whitened vectors). It is NOT a re-extraction from the unioned crops (that would need a 2nd Stage-2 pass and a PCA refit). Honest limitation: acceptable because fwd/bwd embeddings of the same physical object are near-identical (cos ≳ 0.9), so the mean ≈ either vector. Apply the identical mean+renorm to secondary/tertiary so all fused spaces stay aligned and normalized.
- **Class mismatch:** only merge pairs with equal `class_id`.
- **One-to-one constraint:** greedy by descending score prevents a single forward track absorbing many backward fragments (which would itself cause ID merges).
- **Self-pairing:** never compare a forward tracklet to another forward tracklet — BT only links across directions.

### R6. Default-OFF guarantee (preserves 0.77936)

With `stage1.bidirectional.enabled=false` (the 14e B1 config sets no `stage1.bidirectional` key):
- `run_stage1`: the `if bt_cfg.get("enabled", False)` block is skipped — **no backward pass**, `tracklets` is exactly `builder.finalize()` as today.
- `run_stage2`: the merge block at `:728` is skipped — `all_features`, `embeddings`, `index_map`, `sec_matrix`, `tert_matrix` and the Save block are **byte-identical** to current.
- Stage-3/4 receive the identical single-pass set. Therefore 0.77936 (14e B1) is provably unchanged.

### R7. pytest plan (`tests/test_stage1/test_bidirectional_merge.py`)

- **Backward pass id offset + absolute frame_ids:** `run_backward_pass` returns tracklets with `track_id >= 1_000_000` and frame_ids matching the absolute input frame_ids (NO `len-1-i` remap).
- **Gate logic:** synthetic fwd/bwd of same object (high IoU on shared frames, cosine ≥ thr) → merged into ONE tracklet; union of frame_ids; overlapping frame keeps higher-confidence box.
- **ReID gate rejects cross-merge:** two different objects sharing frames at an "intersection" (high IoU, LOW cosine) → NOT merged (this is the S02_c006 safety property the design exists for).
- **Aligned-array collapse:** after merge, `len(features) == primary.shape[0] == hsv.shape[0] == secondary.shape[0] == tertiary.shape[0] == len(index_map)` (lockstep) — guards `stage4/pipeline.py:183`.
- **Unmatched backward kept:** a backward-only tracklet survives with a fresh non-colliding id when `keep_unmatched_backward=true`, and is dropped when false.
- **Pooled embedding normalized:** merged `embedding` has unit L2 norm and equals `l2_normalize(mean(e_f, e_b))`.
- **Disabled path:** `merge_bidirectional` not invoked when flag false → Stage-2 outputs identical to a no-BT run (compare saved npy bytes).

### R8. Honest verdict / decisions needed before coding

1. **Backward pass + Stage-2 embedding flow: CLEAN — no invasive schema change.** `Tracklet`/`TrackletFrame`/`TrackletFeatures` are reused as-is; direction is encoded by the `+1_000_000` id offset; per-box confidence already exists for conflict resolution.
2. **The one non-trivial pipeline reality (FLAGGED):** the multi-model ensemble saves `embeddings_secondary.npy`/`embeddings_tertiary.npy`/multi-query as **positionally row-aligned** npy files that Stage-4 re-loads and assert-matches (`stage4/pipeline.py:183`). The merge therefore CANNOT be a free-standing post-Stage-2 script without re-writing those files; it MUST run at end-of-Stage-2 where all arrays are in memory (R4). This is a localized, contained change to `run_stage2`'s save path — not a data-model change — but it does mean BT **touches Stage-2**, not just Stage-1.
3. **Pooled embedding is mean-of-final, not a re-extraction.** Acceptable (R5) but if you want maximal fidelity, the alternative is a 2nd Stage-2 crop pass over unioned frames + PCA refit — significantly more invasive and NOT recommended. Confirm mean-pooling is acceptable.
4. **Default thresholds chosen:** `min_shared_frames=3`, `iou_thresh=0.5`, `cos_thresh=0.5`. Since fwd/bwd of the same object have cos ≳ 0.9, `cos_thresh` is primarily an intersection-safety gate; recommend ablating `cos_thresh ∈ {0.5, 0.6, 0.7}` on S02_c006. **Decision needed:** confirm `keep_unmatched_backward=true` (recover boundary tracks) vs false (conservative, fewer new tracks) for the first S02 ablation.
