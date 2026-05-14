# Kalman Filter Ground-Plane Tracker — Implementation Spec

## 1. Current Implementation Analysis

### Function: `track_ground_plane_detections()`
**Location**: `src/stage_wildtrack_mvdetr/pipeline.py` lines 178–245

**Algorithm**: Frame-by-frame Hungarian matching on Euclidean distance in world coordinates (centimeters). No motion model — only matches on the **last known position** of each track.

**Parameters (12b v9 best sweep)**:
| Parameter | Default | Best (v9 sweep) | Sweep range |
|---|---|---|---|
| `max_match_distance_cm` | 75.0 | **100.0** | [50, 75, 100, 125] |
| `max_missed_frames` | 5 | **3** | [3, 5, 8, 10] |
| `min_track_length` | 2 | **3** | [2, 3, 5] |

**Data structures**:
- Input: `List[GroundPlaneDetection]` — each has `frame_id`, `x_cm`, `y_cm`, `score`, `raw_frame_id`
- Output: `List[GroundPlaneTrack]` — each has `track_id`, `detections: List[GroundPlaneDetection]`
- Internal: `active_tracks: Dict[int, GroundPlaneTrack]`, `finished_tracks: List[GroundPlaneTrack]`

**Current algorithm step-by-step**:
1. Group detections by `frame_id`
2. For each frame (sorted):
   a. Retire stale tracks where `frame_id - last_detection.frame_id > max_missed_frames`
   b. Build cost matrix: `_distance_cm(last_det, detection)` — Euclidean L2 on `(x_cm, y_cm)`
   c. Gate: set cost to infinity if distance > `max_match_distance_cm`
   d. Solve with `scipy.optimize.linear_sum_assignment` (Hungarian)
   e. Append matched detections to tracks
   f. Create new tracks for unmatched detections
3. Filter tracks shorter than `min_track_length`

**Key weakness**: No velocity prediction. When a person walks steadily, the tracker can only match by proximity to the *last seen position*, not the *predicted position*. At 2fps with ~140cm/s walking speed, a person moves ~70cm between frames. Without prediction, the matching gate must be wide (100cm), which increases ID swap risk in crowded areas.

### Current Results (12b v9)

| Metric | Baseline (no merge) | Best (ReID merge 0.9) |
|---|---|---|
| **IDF1** | 0.9215 | **0.9279** |
| **MODA** | 0.8981 | **0.8992** |
| **Precision** | 0.9707 | 0.9707 |
| **Recall** | 0.9391 | 0.9391 |
| **ID Switches** | 12 | **11** |
| **Trajectories** | 42 | **41** |
| **MODP (cm)** | 9.81 | 9.81 |

**Error profile**: 11 ID switches with 41 trajectories. The ReID merge only fixed 1 ID switch (merge threshold=0.9). The remaining 11 ID switches are the primary target for the Kalman filter.

### Evaluation Pipeline
- **Evaluator**: `src/stage5_evaluation/ground_plane_eval.py` → `evaluate_wildtrack_ground_plane()`
- **Protocol**: Back-project tracklet foot positions to ground plane, average across cameras, DBSCAN NMS (50cm), then motmetrics with L2 matching at 50cm threshold
- **GT format**: WILDTRACK `annotations_positions/*.json` with `personID` and `positionID` (grid index → cm)
- **Match threshold**: 50cm L2 on ground plane

## 2. WILDTRACK Dataset Facts

### Coordinate System
- **Units**: Centimeters (cm)
- **Grid**: 480 × N cells, each cell = 2.5cm
- **Origin**: `x_min = -300.0 cm`, `y_min = -900.0 cm`
- **Conversion**: `x_cm = -300 + grid_x × 2.5`, `y_cm = -900 + grid_y × 2.5`
- **Physical area**: ~12m × ~36m courtyard (ETH/EPFL campus)

### Timing
- **Frame rate**: **2 fps** (WILDTRACK captures every 5th frame from original cameras)
- **Frame normalization**: Raw WILDTRACK frames are 0, 5, 10, 15, ... → normalized to 0, 1, 2, 3, ...
- **Test set**: ~400 frames (exact count depends on MVDeTr split)
- **Timestep**: `Δt = 0.5 seconds` between consecutive normalized frames

### Pedestrian Motion
- **Typical walking speed**: ~1.2–1.5 m/s = **120–150 cm/s**
- **Per-frame displacement at 2fps**: 120÷2 = **60–75 cm** per frame
- **Max reasonable displacement**: ~200 cm/frame (running, 4 m/s)
- **Standing still**: 0–10 cm/frame (detection noise)
- **MODP (localization accuracy)**: 9.81 cm — MVDeTr detections are very precise

### Detection Source
- **Detector**: MVDeTr (Multi-View Deformable Transformer)
- **Output**: `test.txt` with format `frame_id grid_x grid_y` (one line per detection)
- **Best detection performance**: MODA=92.0% (12a v11), surpasses paper's 91.5%
- **No confidence scores** in default MVDeTr output — all scores set to 1.0

## 3. Proposed Kalman Filter Tracker Design

### 3.1 State Vector

$$\mathbf{x} = \begin{bmatrix} x \\ y \\ v_x \\ v_y \end{bmatrix}$$

- `x`, `y`: ground-plane position in **centimeters** (consistent with existing pipeline)
- `v_x`, `v_y`: velocity in **cm/frame** (NOT cm/s — cleaner for the discrete-time model)
- To convert: `v_cm_per_s = v_cm_per_frame × fps = v × 2.0`

### 3.2 State Transition Model

Constant-velocity model with `Δt = 1` (one frame interval):

$$\mathbf{F} = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

For missed frames (frame gap > 1), scale the prediction: use `dt = current_frame - last_frame`:

$$\mathbf{F}(dt) = \begin{bmatrix} 1 & 0 & dt & 0 \\ 0 & 1 & 0 & dt \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

### 3.3 Measurement Model

Direct observation of position only:

$$\mathbf{H} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}$$

$$\mathbf{z} = \begin{bmatrix} x_{det} \\ y_{det} \end{bmatrix}$$

### 3.4 Process Noise Q

Model uncertainty in the constant-velocity assumption. Use the discrete white noise acceleration model:

$$\mathbf{Q} = q \cdot \begin{bmatrix} dt^3/3 & 0 & dt^2/2 & 0 \\ 0 & dt^3/3 & 0 & dt^2/2 \\ dt^2/2 & 0 & dt & 0 \\ 0 & dt^2/2 & 0 & dt \end{bmatrix}$$

**Initial value**: `q = 25.0` (cm²/frame³)

**Rationale**: At 2fps, pedestrians change speed by ~50cm/s per second → ~25cm/frame per frame. Variance = 25² / 4 ≈ 156 cm²/frame², but the matrix structure distributes this, so `q = 25.0` is a reasonable starting point. This will be swept.

### 3.5 Measurement Noise R

$$\mathbf{R} = \begin{bmatrix} r & 0 \\ 0 & r \end{bmatrix}$$

**Initial value**: `r = 100.0` (cm², i.e. σ ≈ 10cm)

**Rationale**: MODP = 9.81cm means the typical localization error is ~10cm. So `R = diag(10², 10²) = diag(100, 100)`.

### 3.6 Initial State Covariance P₀

$$\mathbf{P}_0 = \begin{bmatrix} r & 0 & 0 & 0 \\ 0 & r & 0 & 0 \\ 0 & 0 & v_{max}^2/4 & 0 \\ 0 & 0 & 0 & v_{max}^2/4 \end{bmatrix}$$

With `r = 100.0` and `v_max = 75 cm/frame` (person walking fast):
- Position uncertainty: 100 cm² (σ=10cm)
- Velocity uncertainty: 75²/4 ≈ 1406 cm²/frame² (velocity completely unknown)

### 3.7 Assignment: Hungarian on Mahalanobis Distance with Gating

**Cost metric**: Mahalanobis distance between prediction and measurement:

$$d_M = \sqrt{(\mathbf{z} - \mathbf{H}\hat{\mathbf{x}})^T \mathbf{S}^{-1} (\mathbf{z} - \mathbf{H}\hat{\mathbf{x}})}$$

where $\mathbf{S} = \mathbf{H} \mathbf{P} \mathbf{H}^T + \mathbf{R}$ is the innovation covariance.

**Gating**: Chi-squared distribution with 2 DOF (2D measurement):
- 95%: `distance_gate = 5.99`
- 99%: `distance_gate = 9.21`
- 99.5%: `distance_gate = 10.60`
- **Initial value**: `distance_gate = 9.21` (99% confidence)

**Fallback**: Also enforce a hard Euclidean gate of `max_euclidean_cm = 200.0` to prevent wild matches when P explodes during long occlusions.

### 3.8 Track Lifecycle

| Phase | Rule |
|---|---|
| **Tentative** | New track created on first unmatched detection. State: `tentative`. |
| **Confirmed** | Track with `hits >= min_hits` consecutive *or* total hits. State: `confirmed`. |
| **Coasting** | Confirmed track with no measurement for 1+ frames. Keep predicting (no update). |
| **Deleted** | Track with `misses > max_age` consecutive frames without measurement. |

**Parameters**:
- `min_hits = 3`: Require 3 detections before outputting a track (filters noise, consistent with current `min_track_length=3`)
- `max_age = 3`: Delete after 3 consecutive missed frames (consistent with current `max_missed_frames=3`)

**Only confirmed tracks are output** in the final result.

## 4. Exact Code Changes

### 4.1 New Class: `KalmanTrackState` (in `src/stage_wildtrack_mvdetr/pipeline.py`)

Add after `GroundPlaneTrack` class (around line 60):

```python
@dataclass
class KalmanTrackState:
    """Internal Kalman filter state for one ground-plane track."""
    track_id: int
    x: np.ndarray        # state [x, y, vx, vy], shape (4,)
    P: np.ndarray        # covariance, shape (4, 4)
    hits: int = 0        # total matched frames
    consecutive_misses: int = 0
    last_frame_id: int = -1
    detections: list = field(default_factory=list)  # List[GroundPlaneDetection]
    
    @property
    def is_confirmed(self) -> bool:
        return self.hits >= 3  # min_hits
```

### 4.2 New Function: `track_ground_plane_kalman()` (in `src/stage_wildtrack_mvdetr/pipeline.py`)

Add a new tracking function that replaces `track_ground_plane_detections`:

```python
def track_ground_plane_kalman(
    detections: Sequence[GroundPlaneDetection],
    max_age: int = 3,
    min_hits: int = 3,
    distance_gate: float = 9.21,
    max_euclidean_cm: float = 200.0,
    process_noise_q: float = 25.0,
    measurement_noise_r: float = 100.0,
) -> List[GroundPlaneTrack]:
```

**Implementation outline**:
1. Build `F`, `H`, `Q_base`, `R` matrices (see Section 3)
2. Group detections by frame
3. For each frame:
   a. **Predict**: For each active track, compute `x_pred = F(dt) @ x`, `P_pred = F(dt) @ P @ F(dt).T + Q(dt)`
   b. **Gate & Cost**: For each (track, detection) pair:
      - Innovation: `y = z - H @ x_pred`
      - Innovation covariance: `S = H @ P_pred @ H.T + R`
      - Mahalanobis: `d² = y.T @ S_inv @ y`
      - Gate: reject if `d² > distance_gate` or Euclidean > `max_euclidean_cm`
      - Cost = `d²` (Mahalanobis squared)
   c. **Assign**: Hungarian algorithm on gated cost matrix
   d. **Update matched tracks**: Standard Kalman update `K = P_pred @ H.T @ S_inv`, `x = x_pred + K @ y`, `P = (I - K @ H) @ P_pred`
   e. **Coast unmatched tracks**: Increment `consecutive_misses`, keep predicted state
   f. **Birth new tracks**: For unmatched detections, init `x = [x_cm, y_cm, 0, 0]`, `P = P_0`
   g. **Delete stale tracks**: Remove tracks with `consecutive_misses > max_age`
4. Return confirmed tracks as `List[GroundPlaneTrack]`

### 4.3 Changes to `__init__.py`

Export the new function:
```python
from src.stage_wildtrack_mvdetr.pipeline import track_ground_plane_kalman
```

### 4.4 12b Notebook Changes

**Cell: Parameters (line ~534)**
Add new parameters:
```python
# Kalman tracker parameters
USE_KALMAN_TRACKER = True
KALMAN_MAX_AGE = 3
KALMAN_MIN_HITS = 3
KALMAN_DISTANCE_GATE = 9.21
KALMAN_MAX_EUCLIDEAN_CM = 200.0
KALMAN_PROCESS_NOISE_Q = 25.0
KALMAN_MEASUREMENT_NOISE_R = 100.0
```

**Cell: Tracking (line ~605)**
Replace the tracking call:
```python
if USE_KALMAN_TRACKER:
    from src.stage_wildtrack_mvdetr.pipeline import track_ground_plane_kalman
    tracks = track_ground_plane_kalman(
        detections=detections,
        max_age=KALMAN_MAX_AGE,
        min_hits=KALMAN_MIN_HITS,
        distance_gate=KALMAN_DISTANCE_GATE,
        max_euclidean_cm=KALMAN_MAX_EUCLIDEAN_CM,
        process_noise_q=KALMAN_PROCESS_NOISE_Q,
        measurement_noise_r=KALMAN_MEASUREMENT_NOISE_R,
    )
else:
    tracks = track_ground_plane_detections(
        detections=detections,
        max_match_distance_cm=MAX_MATCH_DISTANCE_CM,
        max_missed_frames=MAX_MISSED_FRAMES,
        min_track_length=MIN_TRACK_LENGTH,
    )
```

**Cell: Tracking sweep (line ~1143)**
Add Kalman sweep section with the parameter grid from Section 5.

### 4.5 Test Changes

**File**: `tests/test_stage_wildtrack_mvdetr/test_pipeline.py`

Add test:
```python
def test_kalman_tracker_handles_linear_motion():
    """Kalman tracker should maintain identity through constant-velocity motion."""
    detections = [
        GroundPlaneDetection(frame_id=0, x_cm=0.0, y_cm=0.0),
        GroundPlaneDetection(frame_id=1, x_cm=60.0, y_cm=0.0),
        GroundPlaneDetection(frame_id=2, x_cm=120.0, y_cm=0.0),
        GroundPlaneDetection(frame_id=3, x_cm=180.0, y_cm=0.0),
        # Second person
        GroundPlaneDetection(frame_id=0, x_cm=500.0, y_cm=500.0),
        GroundPlaneDetection(frame_id=1, x_cm=500.0, y_cm=440.0),
        GroundPlaneDetection(frame_id=2, x_cm=500.0, y_cm=380.0),
        GroundPlaneDetection(frame_id=3, x_cm=500.0, y_cm=320.0),
    ]
    tracks = track_ground_plane_kalman(detections, min_hits=1, max_age=3)
    assert len(tracks) == 2
    # Each person should have all 4 detections in one track
    assert sorted(len(t.detections) for t in tracks) == [4, 4]


def test_kalman_tracker_survives_gap():
    """Kalman tracker should coast through a 1-frame gap using velocity prediction."""
    detections = [
        GroundPlaneDetection(frame_id=0, x_cm=0.0, y_cm=0.0),
        GroundPlaneDetection(frame_id=1, x_cm=60.0, y_cm=0.0),
        # frame 2 missed
        GroundPlaneDetection(frame_id=3, x_cm=180.0, y_cm=0.0),
    ]
    tracks = track_ground_plane_kalman(detections, min_hits=1, max_age=3)
    assert len(tracks) == 1
    assert len(tracks[0].detections) == 3
```

## 5. Parameter Sweep Values

### Phase 1: Baseline comparison (Kalman vs current)
Run Kalman with default parameters and compare to current best:
```python
KALMAN_BASELINE = {
    "max_age": 3, "min_hits": 3, "distance_gate": 9.21,
    "process_noise_q": 25.0, "measurement_noise_r": 100.0,
}
```

### Phase 2: Targeted sweeps
```python
KALMAN_SWEEP = {
    "max_age": [2, 3, 5, 8],
    "min_hits": [1, 2, 3],
    "distance_gate": [5.99, 9.21, 13.82, 16.27],  # chi2(2df): 95%, 99%, 99.9%, 99.99%
    "process_noise_q": [5.0, 10.0, 25.0, 50.0, 100.0],
    "measurement_noise_r": [25.0, 50.0, 100.0, 200.0],
}
```

**Total combinations**: 4 × 3 × 4 × 5 × 4 = 960 — too many for a full grid.

**Strategy**: Sequential 1D sweeps holding other params at baseline:
1. Sweep `process_noise_q` (5 values) → fix best
2. Sweep `measurement_noise_r` (4 values) → fix best
3. Sweep `distance_gate` (4 values) → fix best
4. Sweep `max_age` (4 values) → fix best
5. Sweep `min_hits` (3 values) → fix best
6. One final confirmation run with all best values

**Total runs**: 5 + 4 + 4 + 4 + 3 + 1 = **21 runs** (very fast at CPU speed)

### Phase 3: Joint sweep on top-2 sensitive params
Once the most sensitive 2 parameters are identified, do a focused 2D grid (5×5 = 25 runs).

## 6. Expected Output Format

The Kalman tracker MUST return `List[GroundPlaneTrack]` — the **same type** as the current tracker. This ensures:

1. **`_save_ground_plane_tracks()`** works unchanged (JSON export)
2. **`_save_ground_plane_csv()`** works unchanged (CSV export)
3. **`_tracks_to_projected_tracklets()`** works unchanged (camera projection)
4. **`evaluate_wildtrack_ground_plane()`** works unchanged (TrackEval)
5. **ReID feature extraction** works unchanged (from projected tracklets)
6. **ReID merge** works unchanged (graph-based merge on trajectories)

No changes needed downstream. The Kalman tracker is a **drop-in replacement** for `track_ground_plane_detections()`.

## 7. Expected Impact

### Why Kalman should help
- **Velocity prediction** reduces effective matching radius → fewer ID swaps in crossings
- **Coasting** during occlusion uses predicted position rather than last-seen position
- **Mahalanobis gating** adapts gate size based on track uncertainty (tight for well-tracked objects, wider for newly-born tracks)

### Quantitative expectations
- **ID switches**: 11 → ~5–8 (primary target, ~30–50% reduction)
- **IDF1**: 0.9279 → 0.93–0.94 (~0.5–1.5pp improvement)
- **MODA**: ~unchanged (tracking doesn't affect detection count)
- **Precision/Recall**: ~unchanged (min_hits=3 filters similarly to min_track_length=3)

### Risks
- If the scene has mostly standing people, the Kalman filter adds complexity but no benefit (velocity ≈ 0)
- Process noise tuning: too low → filter lags on direction changes; too high → degenerates to position-only matching
- With only 11 ID switches to fix, the ceiling is low — but even fixing 3–5 would lift IDF1 measurably

## 8. Implementation Order

1. **Add `KalmanTrackState` dataclass** to pipeline.py
2. **Implement `track_ground_plane_kalman()`** in pipeline.py
3. **Add unit tests** in test_pipeline.py
4. **Run tests locally** to verify correctness
5. **Update 12b notebook** — add toggle + parameters
6. **Add sweep cell** to 12b notebook
7. **Push to Kaggle** and run 12b v10
8. **Analyze results** — compare to v9 baseline