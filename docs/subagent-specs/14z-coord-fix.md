# 14z coord-space / matching failure — Fix Spec

## Symptom
14z verifier on Kaggle:
```
detections 971 frames 360 399        (40 normalized test frames)
gt frames 400 0 399                  (full WILDTRACK GT, after // 5)
eval gt frames 40 360 399            (intersection: 40 frames)
metrics: precision=0.95, recall=0.097, misses=8598, MODA=-0.013
```
Target: MODA ≥ 0.916 (12a Kaggle headline = 0.921 ± 0.005).

## Investigation Trace

### 14z eval cell (`notebooks/kaggle/14z_verify_mvdetr_detector/14z_verify_mvdetr_detector.ipynb`)
```python
from src.stage_wildtrack_mvdetr.pipeline import load_mvdetr_ground_plane_detections
from src.stage5_evaluation.ground_plane_eval import evaluate_ground_plane, load_gt_ground_positions

detections = load_mvdetr_ground_plane_detections(TEST_TXT, normalize_wildtrack_frames=True)
gt = load_gt_ground_positions(ANNOTATIONS_DIR)
pred = {}
for pred_id, det in enumerate(detections, start=1):
    pred.setdefault(det.frame_id, []).append((pred_id, det.x_cm, det.y_cm))
det_frames = sorted({det.frame_id for det in detections})
gt_eval = {fid: gt[fid] for fid in det_frames if fid in gt}
metrics = evaluate_ground_plane(gt_eval, pred, threshold_cm=50.0)
```

### Coord conversions on both sides

**Detections (`src/stage_wildtrack_mvdetr/pipeline.py::load_mvdetr_ground_plane_detections`):**
- test.txt format: `frame grid_x grid_y` (3 columns).
- Conversion: `_grid_to_world_cm(grid_x, grid_y) = (-300 + grid_x*2.5, -900 + grid_y*2.5)`.
- Constants: `WILDTRACK_X_MIN_CM=-300`, `WILDTRACK_Y_MIN_CM=-900`, `WILDTRACK_CELL_SIZE_CM=2.5`, `WILDTRACK_GRID_WIDTH=480`.

**GT (`src/stage5_evaluation/ground_plane_eval.py::load_gt_ground_positions`):**
- Per JSON: `for p in data: pos_id = p["positionID"]; gx, gy = _posid_to_ground(pos_id)`.
- `_posid_to_ground(pos_id) = (-300 + (pos_id % 480) * 2.5, -900 + (pos_id // 480) * 2.5)`.
- Frame ID normalization: `frame_id = wildtrack_frame // 5`.

### Upstream MVDeTr trainer (verified from `tmp_12a_artifacts/.../scripts/main.py` and GitHub `hou-yz/MVDeTr/multiview_detector/trainer.py`)
```python
xys = mvdet_decode(world_heatmap, world_offset, reduce=dataset.world_reduce)
grid_xy, scores = xys[:, :, :2], xys[:, :, 2:3]
if dataloader.dataset.base.indexing == 'xy':
    positions = grid_xy
else:                                              # WILDTRACK takes this branch (indexing='ij')
    positions = grid_xy[:, :, [1, 0]]              # SWAP cols 0/1
...
np.savetxt(res_fpath, res_list, '%d')
```
- WILDTRACK `Wildtrack` class: `indexing = 'ij'`, `worldgrid_shape = [480, 1440]`, `pos = grid_x + grid_y*480` (where "grid_x" is the i-axis ∈ [0,480) and "grid_y" is the j-axis ∈ [0,1440)).
- `mvdet_decode(reduce=4)` already upscales heatmap peaks back to the full 480×1440 grid.
- After the `[1, 0]` swap on the 'ij' branch, **test.txt column 1 = i-axis (row index, 480 range), column 2 = j-axis (col index, 1440 range)** — matching `_grid_to_world_cm(row[1], row[2])` and matching `_posid_to_ground` output. **The coordinate conversion in `load_mvdetr_ground_plane_detections` is consistent with WILDTRACK GT.**

### 12a's evaluation path (different — used in the headline 0.921)
12a does NOT call `evaluate_ground_plane(gt, pred, ...)`. It calls
`evaluate_wildtrack_ground_plane(trajectories=..., annotations_dir=..., calibrations_dir=..., match_threshold_cm=50.0)`,
which builds predictions by back-projecting bbox foot points through camera calibration and then matches against GT. **The 0.921 headline does not exercise either `load_mvdetr_ground_plane_detections` or `load_gt_ground_positions`.** 14z is the **first production caller of both** — neither has prior end-to-end verification.

### Numerical sanity check on the recall=0.097 / misses=8598 figures
- 40 test frames × ~25 visible WILDTRACK people/frame ≈ 1,000 GT events expected.
- motmetrics: `recall = (1 − misses/num_objects)` ⇒ `num_objects ≈ 8598 / (1 − 0.097) ≈ 9,521`.
- **9,521 GT events across 40 frames ≈ 238 per frame** — roughly 8–10× the WILDTRACK published density (~25/frame).
- precision=0.95 says the few preds that *do* match are within 50 cm of a GT point.

This profile is **inconsistent with a pure coord-scale or coord-axis mismatch** (those would give ~0% precision OR ~0% recall, not high precision + low recall + inflated GT count). It is consistent with **GT-side over-counting**.

## Root Cause (most likely → less likely)

### H1 (highest confidence): GT loader does not apply MVDeTr's `is_in_cam` visibility filter
MVDeTr's canonical `prepare_gt` (`multiview_detector/datasets/frameDataset.py`):
```python
for single_pedestrian in all_pedestrians:
    in_cam_range = sum(is_in_cam(cam) for cam in range(self.num_cam))
    if not in_cam_range:
        continue                                   # ← skip if not visible in ANY camera
    grid_x, grid_y = self.base.get_worldgrid_from_pos(...)
    og_gt.append(np.array([frame, grid_x, grid_y]))
```
Our `load_gt_ground_positions` skips this filter and includes EVERY `positionID` annotation in the JSON, including positions outside camera FOVs. WILDTRACK annotations carry many such positions (track of off-screen pedestrians for cross-frame consistency); these correctly inflate per-frame counts to ~200+. Predictions never see those off-screen positions, so they all become misses → recall craters.

### H2 (lower confidence): MVDeTr-style POM-grid expansion baked into JSON
If the WILDTRACK Kaggle dataset variant ships annotations that include `views` array elements as separate top-level entries (one per (person, camera) pair), `for p in data:` would iterate 7× per real person. ~30 visible × 7 ≈ 210/frame, very close to the observed 238/frame.

### H3 (low confidence): coord-axis or scale mismatch
Ruled out by the trace above — `_grid_to_world_cm` and `_posid_to_ground` agree on (origin, cell size, axis assignment), and MVDeTr's WILDTRACK 'ij'/'xy' swap is already absorbed by the trainer before test.txt is written. precision=0.95 confirms the matched subset has good geometry.

## Fix (research-only — do not write code)

### Step 0 — Diagnostic (must run first to discriminate H1 vs H2)
Add a one-shot debug print inside `load_gt_ground_positions` (temp, do not commit) and rerun 14z:
```python
print(f"frame {frame_id}: {len(positions)} GT entries, raw_json_len={len(data)}")
```
- If `raw_json_len ≈ 25–35`/frame **and** `len(positions) ≈ 25–35`/frame → not GT loader; investigate trackeval/motmetrics double-counting.
- If `raw_json_len ≈ 25–35` but `len(positions) ≈ 200`/frame → impossible with current loader (no expansion logic), would indicate a JSON-shape bug.
- If `raw_json_len ≈ 200`/frame → H2 (annotations include per-view duplicates) → fix is to dedupe by `personID`.
- If `raw_json_len ≈ 25–35` and the inflated count appears INSIDE motmetrics — H1 (annotations include off-camera pedestrians, and MOT counts each pedestrian × frame, not unique IDs) → fix is to add the `is_in_cam` visibility filter.

### Step 1 — Patch (after Step 0 confirms H1)

**File:** `src/stage5_evaluation/ground_plane_eval.py::load_gt_ground_positions`

Change signature to accept calibrations and add the visibility filter. Specifically:
1. Load camera calibrations once (re-use `_load_calibration` already in this file).
2. For each pedestrian, compute its world coord via `_posid_to_ground`, then back-project to each camera's image plane via the camera intrinsic+extrinsic matrices already loaded for `_pixel_to_ground`. If the projected pixel falls inside *any* of the 7 camera frames (1920×1080), keep the GT entry; otherwise drop it.
3. Add a new optional kwarg `calibrations: Optional[Dict[str, Dict[str, np.ndarray]]] = None` so existing tests still pass when calibrations are not provided (and emit a single-shot `logger.warning` in that case explaining that GT will not be filtered to in-camera positions).

The 14z notebook's eval cell must then pass calibrations into `load_gt_ground_positions` (it already constructs them via the `WILDTRACK_ROOT / "calibrations"` path used by `_load_calibration`).

### Step 1-alt — Patch (after Step 0 confirms H2)

**File:** `src/stage5_evaluation/ground_plane_eval.py::load_gt_ground_positions`

Replace the inner loop with a per-`personID` dedupe:
```python
seen = set()
for p in data:
    pid = p["personID"]
    if pid in seen:
        continue
    seen.add(pid)
    pos_id = p["positionID"]
    gx, gy = _posid_to_ground(pos_id)
    positions.append((pid, gx, gy))
```

### Step 2 — Where the fix should NOT live
- ❌ Do **not** modify `load_mvdetr_ground_plane_detections` — coord conversion is correct and is the same path 12b will use for tracking.
- ❌ Do **not** modify `evaluate_ground_plane` (no auto-detection of coord space; keep the API explicit).
- ❌ Do **not** apply the fix only inside the 14z notebook cell — `load_gt_ground_positions` is the canonical loader and 12b/12c will hit the same bug if it isn't fixed in `src/`.

## Reproducibility / Cross-caller Risk

- **12a** (`evaluate_wildtrack_ground_plane` path): unaffected — uses a totally separate eval path that does its own back-projection and never touches `load_gt_ground_positions`.
- **12b WILDTRACK ReID tracking notebook**: uses the same `src.stage_wildtrack_mvdetr.pipeline` detection loader plus `evaluate_ground_plane` — **will hit the same bug** the moment it tries to evaluate detection-only MODA from the same WILDTRACK GT path. Fix is upstream-shared.
- **12c if added**: same.
- **Other ground-plane eval consumers**: `evaluate_ground_plane` and `load_gt_ground_positions` are only called from `src/stage5_evaluation/pipeline.py` and the WILDTRACK notebooks. Fix is contained.

## Verification gate after fix

Re-run 14z with no other changes. Expected:
- `len(positions)` printed by `load_gt_ground_positions` for any test frame ≈ 20–35 (post-filter), not ~200.
- `metrics["recall"] >= 0.90`, `metrics["precision"] >= 0.95`, `metrics["moda"] in [0.916, 0.926]` (i.e., within ±0.005 of 0.921).
- Total `misses + matches ≈ 1,000` for 40 test frames, not ~9,500.

## Verdict

- Effort: ≈ 2–3 subagent-prompt units (Step 0 diagnostic run + analysis, then Step 1 patch + 14z re-run).
- Priority: **MEDIUM** (blocks 14z verification of the 12a MVDeTr detector headline; blocks 12b WILDTRACK detection-only sanity check; does NOT block any vehicle-pipeline work).
- Do AFTER 14w (14w is a one-line / one-PR fix; 14z requires Kaggle round-trip for the diagnostic).