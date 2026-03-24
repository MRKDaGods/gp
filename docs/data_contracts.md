# Data Contracts

## Overview
This document specifies the exact data formats flowing between pipeline stages. All inter-stage communication uses file-based serialization for independent execution and checkpointing.

## Core Data Models

All defined in `src/core/data_models.py`.

### FrameInfo
```python
@dataclass
class FrameInfo:
    frame_id: int
    camera_id: str
    timestamp: float
    frame_path: str
    width: int
    height: int
```
Used by: Stage 0 → Stage 1

### Detection
```python
@dataclass
class Detection:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
```
Internal to Stage 1.

### TrackletFrame
```python
@dataclass
class TrackletFrame:
    frame_id: int
    timestamp: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
```

### Tracklet
```python
@dataclass
class Tracklet:
    track_id: int
    camera_id: str
    class_id: int
    class_name: str
    frames: list[TrackletFrame]
    # Properties: start_time, end_time, duration, num_frames, mean_confidence
```
Used by: Stage 1 → Stage 2, Stage 1 → Stage 3

### TrackletFeatures
```python
@dataclass
class TrackletFeatures:
    track_id: int
    camera_id: str
    class_id: int
  embedding: np.ndarray       # shape (384,), L2-normalized PCA output used downstream
    hsv_histogram: np.ndarray | None  # shape (num_bins,), L2-normalized
  raw_embedding: np.ndarray | None  # shape (768,), TransReID output before PCA, optional
```
Used by: Stage 2 → Stage 3

### GlobalTrajectory
```python
@dataclass
class GlobalTrajectory:
    global_id: int
    tracklets: list[Tracklet]
    # Properties: camera_sequence, time_span, total_duration, class_name, num_cameras
```
Used by: Stage 4 → Stage 5, Stage 4 → Stage 6, Stage 4 → Apps

### EvaluationResult
```python
@dataclass
class EvaluationResult:
    mota: float
    idf1: float
    hota: float
    id_switches: int
    mostly_tracked: int
    mostly_lost: int
    num_gt_ids: int
    num_pred_ids: int
    details: dict
```
Used by: Stage 5 output.

## File Formats

### Stage 0 Output
Directory: `{output_dir}/stage0/`
- `{camera_id}/frame_{NNNNNN}.jpg` — extracted frames
- `manifest.json` — list of FrameInfo dicts

manifest.json:
```json
[
  {
    "frame_id": 0,
    "camera_id": "cam01",
    "timestamp": 0.0,
    "frame_path": "stage0/cam01/frame_000000.jpg",
    "width": 1920,
    "height": 1080
  }
]
```

### Stage 1 Output
Directory: `{output_dir}/stage1/`
- `tracklets_{camera_id}.json` — tracklets per camera

tracklets_cam01.json:
```json
[
  {
    "track_id": 1,
    "camera_id": "cam01",
    "class_id": 0,
    "class_name": "person",
    "frames": [
      {"frame_id": 0, "timestamp": 0.0, "bbox": [100.5, 200.3, 150.2, 350.8], "confidence": 0.92},
      {"frame_id": 1, "timestamp": 0.1, "bbox": [102.1, 201.0, 152.0, 351.5], "confidence": 0.89}
    ]
  }
]
```

### Stage 2 Output
Directory: `{output_dir}/stage2/`
- `embeddings.npy` — shape (N, 384), all PCA-whitened tracklet embeddings stacked
- `features.json` — TrackletFeatures metadata (without numpy arrays)
- `hsv_features.npy` — shape (N, num_bins), HSV histograms
- `pca_model.pkl` — fitted PCA model reducing TransReID 768D features to 384D

### Stage 3 Output
Directory: `{output_dir}/stage3/`
- `faiss_index.bin` — serialized FAISS index
- `metadata.db` — SQLite database

SQLite schema:
```sql
CREATE TABLE tracklets (
    index_id INTEGER PRIMARY KEY,
    track_id INTEGER NOT NULL,
    camera_id TEXT NOT NULL,
    class_id INTEGER NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    num_frames INTEGER NOT NULL,
    hsv_histogram BLOB
);
CREATE INDEX idx_camera ON tracklets(camera_id);
CREATE INDEX idx_time ON tracklets(start_time, end_time);
```

### Stage 4 Output
Directory: `{output_dir}/stage4/`
- `global_trajectories.json` — list of GlobalTrajectory dicts

### Stage 5 Output
Directory: `{output_dir}/stage5/`
- `evaluation_result.json` — EvaluationResult dict
- `report.html` — styled HTML evaluation report
- `report.md` — Markdown version
- `mot_submission.txt` — MOTChallenge format
- `ablation_results.json` — if ablation was run

### Stage 6 Output
Directory: `{output_dir}/stage6/`
- `annotated_{camera_id}.mp4` — annotated video per camera
- `bev_map.png` — bird's-eye view trajectory map
- `timeline.html` — Plotly timeline visualization
- `trajectories.json` — JSON export
- `trajectories.csv` — CSV export

## Serialization Functions

All in `src/core/io_utils.py`:

| Function | Input | Output File |
|---|---|---|
| save_tracklets(tracklets, path) | List[Tracklet] | .json |
| load_tracklets(path) | .json | List[Tracklet] |
| save_tracklets_by_camera(tracklets_dict, dir) | Dict[str, List[Tracklet]] | .json per camera |
| load_tracklets_by_camera(dir) | .json files | Dict[str, List[Tracklet]] |
| save_embeddings(embeddings, path) | np.ndarray | .npy |
| load_embeddings(path) | .npy | np.ndarray |
| save_global_trajectories(trajectories, path) | List[GlobalTrajectory] | .json |
| load_global_trajectories(path) | .json | List[GlobalTrajectory] |

## Conventions

1. **Coordinate system**: All bounding boxes use `(x1, y1, x2, y2)` pixel coordinates (top-left origin)
2. **Timestamps**: Seconds from video start (float)
3. **Camera IDs**: String format, e.g., "cam01", "cam02"
4. **Class IDs**: COCO class IDs (0=person, 2=car, 5=bus, 7=truck)
5. **Embeddings**: TransReID emits 768D raw embeddings; PCA whitening reduces them to 384D and the saved vectors are always L2-normalized for cosine similarity via inner product
6. **JSON encoding**: Custom NumpyEncoder handles np.ndarray, np.float32, np.int64
