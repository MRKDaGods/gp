# Ground-Plane Evaluation Fix — covers both 14w (WILDTRACK person pipeline) and 14z (MVDeTr detector)

## TL;DR
Both 14w and 14z fail with the same fingerprint **Precision ≈ 95%, Recall ≈ 10%, MODA ≈ −0.01 / 0.09**. They share a common eval path. There are **two compounding bugs** in that path:

1. **GT density inflation in `load_gt_ground_positions`** (`src/stage5_evaluation/ground_plane_eval.py`, lines 152–177). The loader iterates every JSON entry without `is_in_cam` filtering or per-`personID` dedupe, so each frame contributes ~200+ GT events instead of the expected ~25 visible pedestrians per frame.
2. **Missing `frame_range` in pipeline-driven eval** (`src/stage5_evaluation/pipeline.py`, lines 216–227). `evaluate_wildtrack_ground_plane` is called without `frame_range`, so GT spans all 400 frames even when predictions only cover a subset. The 12b notebook (which historically achieved IDF1 0.947 / MODA 0.903) explicitly passes `frame_range=(pred_frame_min, pred_frame_max)`; the pipeline path does not.

Bug 1 affects **both** 14w and 14z. Bug 2 affects only 14w (14z restricts GT manually via `gt_eval = {fid: gt[fid] for fid in det_frames if fid in gt}`). Both must be fixed for both verifiers to pass.

## Symptom Recap

### 14z (MVDeTr detector verifier)
```
detections: 971 events across 40 frames (360–399, normalized = raw // 5)
gt:         400 frames (0–399)
eval gt:    40 frames after manual det-frame restriction
metrics:    precision=0.95, recall=0.097, misses=8598, MODA=−0.013
num_objects ≈ 8598 / (1 − 0.097) ≈ 9521  →  ~238 GT events per frame
```
Expected visible density on WILDTRACK is ~25/frame. 238/frame is ~10× too high → GT loader is including non-visible (off-camera) pedestrians.

### 14w (WILDTRACK person pipeline verifier)
```
IDF1=0.173 (target 0.947), MODA=0.090 (target 0.903)
Precision ≈ 95%, Recall ≈ 10%
```
14w runs `scripts/run_pipeline.py` end-to-end. Stage 5 calls `evaluate_wildtrack_ground_plane` from `src/stage5_evaluation/pipeline.py` **without** `frame_range`. This causes a second round of inflation on top of bug 1.

### Why P ≈ 95% in both cases
The few predictions that *do* match GT match within 50 cm — geometry/calibration is correct. The bug is purely on the GT-count side: a small numerator over a hugely inflated denominator gives high precision and crushed recall.

## Root Cause Files & Lines

### Bug 1 — `src/stage5_evaluation/ground_plane_eval.py`
```python
def load_gt_ground_positions(
    annotations_dir: Path,
) -> Dict[int, List[Tuple[int, float, float]]]:
    ...
    for jf in json_files:
        ...
        for p in data:
            pid = p["personID"]
            pos_id = p["positionID"]
            gx, gy = _posid_to_ground(pos_id)
            positions.append((pid, gx, gy))
        gt[frame_id] = positions
```
- No `is_in_cam` projection check against the 7 WILDTRACK camera FOVs.
- No per-`personID` dedupe.
- The canonical MVDeTr reference loader (`multiview_detector/datasets/frameDataset.py::prepare_gt`) skips pedestrians whose projected pixel is outside *every* camera's frame: `if not sum(is_in_cam(c) for c in range(num_cam)): continue`.

### Bug 2 — `src/stage5_evaluation/pipeline.py`
```python
gp_result = evaluate_wildtrack_ground_plane(
    trajectories=trajectories,
    annotations_dir=annotations_dir,
    calibrations_dir=calibrations_dir,
    conf_threshold=float(gp_eval_cfg.get("conf_threshold", 0.25)),
    match_threshold_cm=float(gp_eval_cfg.get("match_threshold_cm", 50.0)),
    nms_radius_cm=float(gp_eval_cfg.get("nms_radius_cm", 50.0)),
)
```
`frame_range` is not passed. Compare with the 12b notebook (`notebooks/kaggle/12b_wildtrack_tracking_reid/12b_wildtrack_tracking_reid.ipynb`, lines 1637 / 1958 / 1988 / 2096):
```python
evaluate_wildtrack_ground_plane(..., frame_range=(pred_frame_min, pred_frame_max))
```

## Fix (research-only; do not write code in this spec — Coder executes)

### Step 0 — Mandatory one-shot diagnostic (Kaggle CPU, <2 min)
Before patching, add a temporary debug print inside `load_gt_ground_positions` (do not commit) and rerun 14z to discriminate between the two sub-hypotheses for bug 1:
```python
if frame_id in (0, 200, 360, 399):
    pid_set = {p["personID"] for p in data}
    print(f"[gt-diag] frame={frame_id} raw_json_len={len(data)} unique_pids={len(pid_set)} positions_len={len(positions)}")
```
Decision matrix:

| raw_json_len | unique_pids | Diagnosis | Patch |
|---|---|---|---|
| ~25–35 | ~25–35 | not a GT loader bug — re-investigate motmetrics or coord transform | abort and re-spec |
| ~200–240 | ~25–35 | H2: JSON has per-view duplicates of the same `personID` | dedupe by `personID` (Step 1A) |
| ~200–240 | ~200–240 | H1: JSON lists off-camera pedestrians too | add `is_in_cam` filter (Step 1B) |
| ~25–35 | ~25–35 + inflation only inside motmetrics | unlikely — re-investigate | n/a |

### Step 1A — Patch (if Step 0 diagnoses H2)
File: `src/stage5_evaluation/ground_plane_eval.py::load_gt_ground_positions`
```python
seen = set()
positions = []
for p in data:
    pid = p["personID"]
    if pid in seen:
        continue
    seen.add(pid)
    pos_id = p["positionID"]
    gx, gy = _posid_to_ground(pos_id)
    positions.append((pid, gx, gy))
```

### Step 1B — Patch (if Step 0 diagnoses H1)
File: `src/stage5_evaluation/ground_plane_eval.py::load_gt_ground_positions`

1. Change the signature to `load_gt_ground_positions(annotations_dir: Path, calibrations: Optional[Dict[str, Dict[str, np.ndarray]]] = None, image_size: Tuple[int, int] = (1920, 1080))`.
2. If `calibrations` is `None`, log a single `logger.warning` ("GT will not be filtered to in-camera positions; recall will be under-counted") and keep current behavior for backward compatibility with any downstream caller that hasn't been updated yet.
3. If `calibrations` is provided, for each entry: compute `(gx, gy)` from `_posid_to_ground`, then for each `cam_id, cal in calibrations.items()` project the 3D point `(gx, gy, 0.0)` (and optionally a head point at z=175 cm for a slightly more permissive visibility test) to the image plane via `cv2.projectPoints` using `cal["R"]→rvec` (use `cv2.Rodrigues`), `cal["tvec"]`, `cal["K"]`. Keep the entry iff the projected pixel falls inside `(0, 0, image_size[0], image_size[1])` for **any** camera.
4. Both callers (`evaluate_wildtrack_ground_plane` in the same file, and the 14z notebook eval cell) must pass the already-loaded `cals` dict.

### Step 2 — Patch pipeline frame_range (always required for 14w)
File: `src/stage5_evaluation/pipeline.py`, around line 219:
```python
# Derive frame_range from prediction coverage so GT and pred span the same frames
pred_frames = sorted({f.frame_id for traj in trajectories for trk in traj.tracklets for f in trk.frames})
frame_range = (min(pred_frames), max(pred_frames)) if pred_frames else None

gp_result = evaluate_wildtrack_ground_plane(
    trajectories=trajectories,
    annotations_dir=annotations_dir,
    calibrations_dir=calibrations_dir,
    conf_threshold=float(gp_eval_cfg.get("conf_threshold", 0.25)),
    match_threshold_cm=float(gp_eval_cfg.get("match_threshold_cm", 50.0)),
    nms_radius_cm=float(gp_eval_cfg.get("nms_radius_cm", 50.0)),
    frame_range=frame_range,
)
```
This mirrors what 12b's notebook already does and is required even if bug 1 is fully fixed, because 14w's MVDeTr-cached detections only cover the test split (40 frames).

### Step 3 — Where the fix should NOT live
- ❌ Do **not** edit `load_mvdetr_ground_plane_detections` — coord conversion (`_grid_to_world_cm`, `_posid_to_ground`) is verified consistent with MVDeTr's `indexing='ij'` swap.
- ❌ Do **not** edit `evaluate_ground_plane` — the matcher and motmetrics call are correct.
- ❌ Do **not** apply the GT filter only inside the 14z notebook — both 14w (via pipeline) and 12b/12c need the shared loader fixed.

## Why this explains both 14w and 14z producing identical P ≈ 95% / R ≈ 10%
- Bug 1 inflates GT density per frame by ~10× in both verifiers. With pred density correct (~25/frame), `recall ≈ matched_preds / inflated_gt ≈ 25 / 238 ≈ 0.10`. ✓
- `precision = matches / num_predictions` is unaffected by GT inflation — the predictions that match are still within 50 cm of a real GT point. Hence P ≈ 95% in both. ✓
- Bug 2 additionally inflates 14w's denominator by ~10× more (eval over 400 frames instead of the ~40-frame prediction window). The fingerprint stays P ≈ 95% / R ≈ 10% because both fragments hurt recall multiplicatively in the same direction; the symptom is indistinguishable from bug 1 alone, which is why a single observation can mask both.

## Verification Plan (post-fix Kaggle pushes)

### After Step 0 (diagnostic)
Push `14z_verify_mvdetr_detector` once. Inspect kernel log for `[gt-diag]` lines. Decide H1 vs H2.

### After Step 1 + Step 2 (patch)
1. Push `14z_verify_mvdetr_detector` first (fast, no Stage 1/2/3/4 work). Expected:
   - `len(positions)` per frame drops from ~238 to ~25.
   - `num_objects + num_predictions` ≈ 25 × 40 + 24 × 40 ≈ 2000.
   - `precision ≥ 0.90`, `recall ≥ 0.90`, `MODA ∈ [0.916, 0.926]` (12a headline 0.921 ± 0.005).
2. If 14z passes, push `14w_verify_wildtrack_b1`. Expected:
   - `IDF1 ∈ [0.942, 0.952]` (target 0.947 ± 0.005).
   - `MODA ∈ [0.898, 0.908]` (target 0.903 ± 0.005).

If either still fails after Step 1 + Step 2, do not retry — re-investigate with a fresh diagnostic before pushing again (Kaggle has a 2-GPU concurrency cap and rapid re-pushes are forbidden by repo protocol).

## Risk Assessment

### Regression risk to GREEN verifiers
- **14v (CityFlow vehicle MTMC IDF1)** — `src/stage5_evaluation/ground_plane_eval.py` and `evaluate_wildtrack_ground_plane` are gated by `dataset_cfg.get("name") == "wildtrack"` or `gp_eval_cfg.get("enabled")`. CityFlow doesn't trigger this path. **No regression risk.**
- **14x (CityFlow Stage-4 siblings)** — same gating. **No regression risk.**

### Regression risk to 12a/12b headlines
- **12a (MVDeTr training)** — uses MVDeTr's internal evaluator, not our `load_gt_ground_positions`. **No regression risk.**
- **12b (WILDTRACK ReID tracking, IDF1 0.947 / MODA 0.903)** — currently passes `frame_range` explicitly and presumably matches the existing inflated GT count somehow (likely because pred coverage also spans all 400 frames at ~steady density, so the symptom partially cancels). After Step 1, num_objects will drop ~10×; matches will also drop proportionally (only previously-matched ones are kept under the in-camera filter). Net IDF1 / MODA should hold or improve marginally. **Low regression risk, but 12b should be re-verified once the patch lands.**

### Risk of choosing H1 vs H2 wrongly
Mitigated by mandatory Step 0 diagnostic. If applied to the wrong hypothesis, the patch will not move metrics — easy to detect and revert.

## Verdict
- Priority: **HIGH** (blocks all WILDTRACK verification — 14w, 14z, and any future 12b/12c reproduction).
- Effort: 1 diagnostic Kaggle push + 1 small src/ patch (Step 1 + Step 2 in the same PR) + 2 verification Kaggle pushes.
- Independence: does not touch any CityFlow code path. Vehicle pipeline (14v, 14x, all 14a–14u experiments) is unaffected.
