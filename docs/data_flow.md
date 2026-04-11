# MTMC Tracker — Complete Data Flow Documentation

## Overview

The MTMC Tracker is a multi-camera multi-target tracking system with a **Python/FastAPI backend** (port 8004) and **Next.js frontend** (port 3001). The backend runs a 7-stage offline pipeline and serves results via REST APIs. The frontend provides a dashboard UI for video upload, pipeline execution, timeline querying, and result visualization.

---

## 1. Application Startup

### Launcher (`start.py`)

```
start.py
├── Kill any existing processes on ports 8004 (backend) and 3001 (frontend)
├── Start backend: python -m uvicorn backend_api:app --host 0.0.0.0 --port 8004
│   └── Wait for GET /api/health → 200 (60s timeout)
├── Start frontend: npm run dev --port 3001 (in frontend/ directory)
│   └── Wait for HTTP 200 on localhost:3001 (90s timeout)
└── Monitor both processes; shutdown if either exits
```

### Backend Startup (`backend/app.py` → `@app.on_event("startup")`)

```
_on_startup()
├── _scan_startup_videos()
│   ├── Clear uploaded_videos state
│   ├── Scan uploads/ folder → register video files
│   ├── Scan data/raw/cityflowv2/ → register CityFlow videos
│   ├── Scan dataset/ folder → register dataset camera videos
│   ├── Create virtual records for CityFlow cameras with seqinfo.ini
│   └── Restore video → run_id mappings from outputs/*/probe_video_id.txt
│
└── asyncio.create_task(_background_precompute_dataset())
    ├── Check if dataset/S01/ exists
    ├── If outputs/dataset_precompute_s01/stage1/tracklets_*.json exists → skip (cached)
    └── Otherwise: run full pipeline stages 0-4 on S01 dataset
        └── Output → outputs/dataset_precompute_s01/{stage0,stage1,stage2,stage3,stage4}/
```

### In-Memory State (`backend/state.py`)

```
AppState (singleton)
├── uploaded_videos: Dict[video_id → video_record]
│   └── Video record: {id, name, filename, path, size, duration, fps, width, height, uploadedAt}
├── active_runs: Dict[run_id → run_record]
│   └── Run record: {id, stage, status, progress, message, startedAt, videoId, cameraId, ...}
├── video_to_latest_run: Dict[video_id → run_id]
│   └── Maps each video to its most recent pipeline run
└── run_id_lock: threading.Lock (for numeric ID allocation)
```

---

## 2. Directory Structure

### Key Directories

| Directory | Purpose | Read/Write |
|-----------|---------|------------|
| `uploads/` | User-uploaded video files | Write (via upload API) |
| `outputs/` | All pipeline run outputs | Write (pipeline + clip export) |
| `outputs/{run_id}/` | Single run output folder | Write |
| `outputs/dataset_precompute_s01/` | Named run for S01 dataset precompute | Write (at startup) |
| `preprocessed_datasets/dataset_precompute_s01/` | Git-tracked reference data (read-only fallback) | Read only |
| `dataset/S01/` | Raw multi-camera dataset (camera subdirectories) | Read only |
| `data/raw/cityflowv2/` | CityFlowV2 raw videos | Read only |
| `models/` | ML models (YOLO, ReID, PCA) | Read only |

### Run Folder Structure

```
outputs/{run_id}/
├── config.yaml              # Pipeline config used for this run
├── run_context.json         # Run metadata (videoId, cameraId, datasetFolder)
├── probe_video_id.txt       # Links this run to a video_id (persists across restarts)
├── pipeline.log             # Pipeline execution log
├── input/                   # Symlink or copy of input video
├── stage0/                  # Extracted frames (JPEGs)
│   └── {camera_id}/
│       ├── frame_000000.jpg
│       ├── frame_000001.jpg
│       └── ...
├── stage1/                  # Detection + tracking results
│   ├── tracklets_c001.json
│   ├── tracklets_c002.json
│   └── ...
├── stage2/                  # ReID embeddings
│   ├── embeddings.npy       # NxD float32 matrix (N frames, D embedding dim)
│   └── embedding_index.json # List of {camera_id, track_id, frame_id, class_id}
├── stage3/                  # FAISS index + SQLite metadata
│   ├── faiss_index.bin
│   └── metadata.db
├── stage4/                  # Cross-camera association
│   └── global_trajectories.json  # List of trajectory dicts with tracklets
├── selected/                # Selected tracklet clips (from user selection)
│   └── *.mp4
└── matched/                 # Matched trajectory clips + alternatives
    ├── summary.json
    ├── global_{gid}_cam_{cam}_track_{tid}.mp4  (one per matched tracklet)
    └── top5_alternatives/
        ├── alt_01_global_{gid}_cam_{cam}_track_{tid}.mp4
        ├── ...
        └── by_track/                           (per-anchor alternatives)
            └── {anchor_cam}_{anchor_tid}/
                ├── alt_01_cam_{cam}_track_{tid}.mp4
                └── ...
```

---

## 3. Pipeline Stages

### Stage 0: Frame Extraction
- **Input:** Video file(s)
- **Output:** `stage0/{camera_id}/frame_NNNNNN.jpg`
- **Process:** Extract frames at configured FPS, save as JPEG

### Stage 1: Detection + Tracking
- **Input:** Stage 0 frames
- **Output:** `stage1/tracklets_{camera_id}.json`
- **Process:** YOLO26m detection → BoT-SORT/DeepOCSORT tracking → interpolation → intra-merge
- **Data format:** List of tracklet dicts with `track_id, camera_id, class_id, frames[{frame_id, bbox, confidence}]`

### Stage 2: Feature Extraction (ReID Embeddings)
- **Input:** Stage 0 frames + Stage 1 tracklets
- **Output:** `stage2/embeddings.npy` + `stage2/embedding_index.json`
- **Process:** TransReID ViT-Base/16 CLIP extracts per-frame embeddings + HSV features + optional PCA whitening
- **Data format:** `embeddings.npy` = (N, D) float32 matrix; `embedding_index.json` = list of {camera_id, track_id, frame_id, class_id}

### Stage 3: Indexing
- **Input:** Stage 2 embeddings
- **Output:** `stage3/faiss_index.bin` + `stage3/metadata.db`
- **Process:** FAISS IndexFlatIP (cosine similarity) + SQLite metadata

### Stage 4: Cross-Camera Association
- **Input:** Stage 2 embeddings + Stage 3 index
- **Output:** `stage4/global_trajectories.json`
- **Process:** Similarity graph → connected components → global trajectory assignment

---

## 4. Core Data Flow: Timeline Query (Search by Embedding)

This is the main user-facing feature — select tracklets in the UI, find matching cross-camera trajectories using embedding similarity.

### Request Flow

```
Frontend                          Backend
   │                                 │
   │  POST /api/timeline/query       │
   │  {videoId, runId,               │
   │   selectedTrackIds: ["4","3"]}  │
   │ ─────────────────────────────►  │
   │                                 │
   │                    ┌────────────┤
   │                    │ timeline.py (router)
   │                    │   ├── Validate videoId exists in state.uploaded_videos
   │                    │   ├── Create InMemoryDatasetRepository
   │                    │   └── Call TimelineService.query_with_candidates()
   │                    │
   │                    │ TimelineService (timeline_service.py)
   │                    │   ├── Parse selectedTrackIds → set of ints {4, 3}
   │                    │   ├── Resolve probe_run_id (which run has these tracklets)
   │                    │   ├── Check stage4 trajectories exist for gallery runId
   │                    │   │   └── Load global_trajectories.json
   │                    │   │
   │                    │   ├── _run_visual_search()
   │                    │   │   ├── Load PROBE embeddings:
   │                    │   │   │   outputs/{probe_run_id}/stage2/embeddings.npy
   │                    │   │   │   outputs/{probe_run_id}/stage2/embedding_index.json
   │                    │   │   │
   │                    │   │   ├── Load GALLERY embeddings:
   │                    │   │   │   outputs/{gallery_run_id}/stage2/embeddings.npy
   │                    │   │   │   outputs/{gallery_run_id}/stage2/embedding_index.json
   │                    │   │   │
   │                    │   │   ├── PCA projection if dimensions differ
   │                    │   │   │   (probe_dim > gallery_dim → models/reid/pca_transform.pkl)
   │                    │   │   │
   │                    │   │   └── _score_trajectories()
   │                    │   │       ├── Build gallery_map: (cam, tid) → [embedding row indices]
   │                    │   │       ├── Collect probe_indices for selectedTrackIds
   │                    │   │       ├── L2-normalize probe features
   │                    │   │       │
   │                    │   │       ├── FOR EACH trajectory in global_trajectories:
   │                    │   │       │   ├── Collect gallery embedding indices for all tracklets
   │                    │   │       │   ├── Class-gate: skip cross-class matches (car≠person)
   │                    │   │       │   ├── L2-normalize gallery features
   │                    │   │       │   ├── Compute cosine similarity matrix:
   │                    │   │       │   │   sim_mat = probe_feats @ gallery_feats.T
   │                    │   │       │   ├── best_per_probe = max similarity per probe frame
   │                    │   │       │   ├── mean_best = mean(best_per_probe)
   │                    │   │       │   ├── p25_best = 25th percentile(best_per_probe)
   │                    │   │       │   │
   │                    │   │       │   ├── MATCH if:
   │                    │   │       │   │   mean_best >= 0.82 (SIMILARITY_THRESHOLD_MEAN)
   │                    │   │       │   │   AND p25_best >= 0.74 (SIMILARITY_THRESHOLD_P25)
   │                    │   │       │   │
   │                    │   │       │   └── ALL trajectories scored → ranked_candidates
   │                    │   │       │       (used for "near miss" alternatives export)
   │                    │   │       │
   │                    │   │       └── Return scored matches (sorted by score DESC)
   │                    │   │
   │                    │   ├── Fallback: exact-ID match if no visual matches
   │                    │   └── Build selected tracklet summaries
   │                    │
   │                    │ Router side-effects (timeline.py):
   │                    │   ├── _export_timeline_debug_bundle() → debug JSON
   │                    │   ├── _export_selected_clips() → selected/ folder
   │                    │   └── IF matched trajectories found:
   │                    │       └── _export_matched_clips()
   │                    │           ├── Create outputs/{probe_run_id}/matched/
   │                    │           ├── For each matched trajectory:
   │                    │           │   └── For each tracklet: export MP4 clip
   │                    │           │       (resolve frames from stage0/)
   │                    │           ├── Export top-5 alternatives from ranked_candidates
   │                    │           │   (trajectories that scored below threshold)
   │                    │           └── Write summary.json
   │                    │
   │                    └────────────┤
   │                                 │
   │  ◄─────────────────────────────  │
   │  {success, data: {              │
   │    stage4Available,             │
   │    mode: "matched",             │
   │    trajectories: [...],         │
   │    selectedTracklets: [...],    │
   │    diagnostics: {...}           │
   │  }}                             │
```

### Embedding Search Algorithm (Cosine Similarity)

```
1. PROBE EMBEDDINGS (what the user selected):
   - Filter embedding_index.json for rows where track_id ∈ selectedTrackIds
   - Extract corresponding rows from embeddings.npy → probe_feats (M × D)
   - L2-normalize: probe_feats = probe_feats / ||probe_feats||₂

2. GALLERY EMBEDDINGS (all tracklets in the dataset):
   - Build lookup: (camera_id, track_id) → [row indices in embeddings.npy]

3. FOR EACH TRAJECTORY in global_trajectories.json:
   - Collect all embedding rows for all tracklets in this trajectory → gallery_feats (K × D)
   - L2-normalize gallery_feats
   - Compute similarity matrix: sim = probe_feats @ gallery_feats.T  →  (M × K)
     This is a matrix of cosine similarities between every probe frame and every gallery frame
   - best_per_probe = max(sim, axis=1)  →  (M,)
     For each probe frame, find the best-matching gallery frame
   - mean_best = mean(best_per_probe)
   - p25_best = percentile(best_per_probe, 25)
   - MATCH if mean_best ≥ 0.82 AND p25_best ≥ 0.74

4. RESULT:
   - Matched: trajectories above both thresholds (sorted by score DESC)
   - Near-miss: all scored trajectories (used for top-5 alternatives export)
```

### Per-Anchor Alternatives (On-Demand Embedding Search)

When the user clicks a specific tracklet in the UI and asks "who else looks like this?":

```
GET /api/runs/{run_id}/matched_alternatives?anchorCameraId=c001&anchorTrackId=42

1. Build embedding bank:
   - Load stage2/embeddings.npy + embedding_index.json
   - Group by (camera_id, track_id)
   - Mean-pool all frame embeddings for each tracklet → single vector per tracklet
   - L2-normalize each vector
   → bank[(cam, tid)] = normalized_vector

2. Get anchor embedding:
   anchor_vec = bank[("c001", 42)]

3. Score all other tracklets:
   For each (cam, tid) in bank:
     score = dot(anchor_vec, bank[(cam, tid)])  # cosine similarity (both L2-normalized)

4. Sort by score DESC, take top-K

5. For each result: lazy-generate clip MP4 if not cached
   → outputs/{run_id}/matched/top5_alternatives/by_track/{anchor_cam}_{anchor_tid}/alt_*.mp4
```

---

## 5. Clip Export & Frame Resolution

### Frame Path Resolution (`_stage0_frame_path()`)

```
Fallback chain:
1. outputs/{run_id}/stage0/{camera_id}/frame_{frame_id:06d}.jpg   (primary)
2. outputs/{run_id}/stage0/{camera_id}/frame_{frame_id:06d}.png   (fallback extension)
3. outputs/dataset_precompute_s01/stage0/{camera_id}/frame_*.jpg  (precompute fallback)
4. preprocessed_datasets/dataset_precompute_s01/stage0/{camera_id}/... (read-only fallback)
```

### Clip Creation (`_export_tracklet_clip()`)

```
Input: run_id, tracklet dict (with frames[]), output MP4 path

1. Sample frames: if > 180 frames, downsample evenly
2. For each sampled frame:
   ├── Resolve frame path via _stage0_frame_path()
   ├── Read image with OpenCV
   ├── Extract bbox region with padding (0.5x bbox size)
   └── Write frame to MP4 via cv2.VideoWriter
3. Optional: re-encode with ffmpeg for browser compatibility
```

---

## 6. `outputs/dataset_precompute_s01/` — Why It Exists

This is a **named pipeline run** created automatically at backend startup. It is NOT duplication — it's the standard pipeline output for the S01 dataset:

1. **At startup**, `_background_precompute_dataset()` checks if `dataset/S01/` exists
2. If the run hasn't been done yet (no `stage1/tracklets_*.json`), it runs stages 0-4
3. Output goes to `outputs/dataset_precompute_s01/` (standard output directory)
4. This run becomes the **gallery** for timeline queries — it contains all cross-camera trajectories

The `preprocessed_datasets/dataset_precompute_s01/` folder is a **separate, read-only fallback** containing pre-computed data. It is only used if `outputs/{run_id}/stage0/` frames are missing (frame path fallback chain).

---

## 7. All Backend API Endpoints

### Health & System
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | Health check + model status |
| GET | `/` | Root API info |

### Videos
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/videos/upload` | Upload video file |
| GET | `/api/videos` | List all registered videos |
| GET | `/api/videos/{video_id}` | Get video metadata |
| DELETE | `/api/videos/{video_id}` | Delete video |
| GET | `/api/videos/stream/{video_id}` | Stream video (AVI→MP4 transcode) |

### Pipeline Execution
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/pipeline/run-stage/{stage}` | Run specific pipeline stage |
| POST | `/api/pipeline/run` | Run full pipeline |
| GET | `/api/pipeline/status/{run_id}` | Get pipeline status |
| POST | `/api/pipeline/cancel/{run_id}` | Cancel running pipeline |
| WS | `/api/ws/pipeline/{run_id}` | WebSocket for live progress |

### Tracklets & Trajectories
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/tracklets` | Get tracklets for camera/video |
| GET | `/api/trajectories/{run_id}` | Get global trajectories |
| GET | `/api/runs/{run_id}/tracklet_sequence` | Get sampled frames + bboxes for tracklet |

### Timeline Query (Core Search)
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/timeline/query` | Match selected tracklets to trajectories via cosine similarity |

### Matched Results
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/runs/{run_id}/matched_summary` | Get matched summary JSON |
| GET | `/api/runs/{run_id}/matched_clips/{filename}` | Serve matched clip MP4 |
| GET | `/api/runs/{run_id}/matched_alternatives` | Get top-K alternative tracklets |
| GET | `/api/runs/{run_id}/matched_alternatives/{clip_path}` | Serve alternative clip MP4 |

### Frames & Crops
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/frames/{video_id}/{frame_id}` | Serve single frame as JPEG |
| GET | `/api/frames/{video_id}/{frame_id}/detections` | Frame + detection overlay |
| GET | `/api/runs/{run_id}/full_frame` | Serve stage0 frame for timeline |
| GET | `/api/crops/{video_id}` | Cropped vehicle image from video |
| GET | `/api/crops/run/{run_id}` | Cropped vehicle image from stage0 |

### Detection
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/detections/{video_id}` | Get detections for video/frame |
| GET | `/api/detections/{video_id}/all` | All detections grouped by frame |

### Search
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/search/tracklet` | Search gallery for similar tracklets |

### Datasets
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/datasets` | List dataset folders |
| POST | `/api/datasets/{folder}/process` | Trigger full pipeline on dataset |

### Export & Evaluation
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/export/{run_id}` | Export trajectories (JSON/CSV/MOT) |
| GET | `/api/download/{run_id}/{filename}` | Download exported file |
| GET | `/api/evaluation/{run_id}` | Get evaluation metrics (MOTA, IDF1) |
| POST | `/api/visualization/summary/{run_id}` | Generate summary video |

### Locations & Cameras
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/locations/governorates` | Egypt governorates |
| GET | `/api/locations/cities/{id}` | Cities in governorate |
| GET | `/api/locations/zones/{id}` | Zones in city |
| GET | `/api/cameras` | Discovered cameras |

### Import
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/runs/import-kaggle` | Import Kaggle artifacts ZIP |

---

## 8. Frontend ↔ Backend Interaction Map

### Video Upload & Registration
```
User uploads video   → POST /api/videos/upload (FormData)
                     → Backend saves to uploads/, probes with ffprobe
                     → Returns video record {id, name, path, duration, fps, ...}
                     → Frontend displays in video list
```

### Pipeline Execution
```
User clicks "Run Stage 1" → POST /api/pipeline/run-stage/1 {videoId, cameraId, ...}
                           → Backend allocates numeric run_id, starts async pipeline
                           → Returns {runId, status: "running"}
                           → Frontend opens WebSocket: /api/ws/pipeline/{runId}
                              └── Receives: {stage, progress, message} events
                           → Frontend polls: GET /api/pipeline/status/{runId}
                           → When complete: status="completed", progress=100
```

### Timeline Query (Main Search Flow)
```
User selects tracklets in video → Frontend collects selectedTrackIds
                                → POST /api/timeline/query {videoId, runId, selectedTrackIds}
                                → Backend:
                                   1. Load probe + gallery embeddings from stage2/
                                   2. Compute cosine similarity across all trajectories
                                   3. Filter by thresholds (mean≥0.82, p25≥0.74)
                                   4. Export matched clips to matched/
                                   5. Export top-5 alternatives to matched/top5_alternatives/
                                → Returns {mode:"matched", trajectories:[...], diagnostics:{...}}
                                → Frontend displays matched trajectories in timeline view
```

### Viewing Matched Results
```
Frontend shows matched trajectory → GET /api/runs/{runId}/matched_summary
                                    (loads summary.json with clip manifest)
                                 → For each clip: <video src="/api/runs/{runId}/matched_clips/{filename}">
                                    (backend serves MP4 from matched/ folder)
```

### Viewing Alternatives ("Who else looks like this?")
```
User clicks anchor tracklet → GET /api/runs/{runId}/matched_alternatives?anchorCameraId=c001&anchorTrackId=42
                             → Backend:
                                1. Build embedding bank (mean-pool per tracklet, L2-normalize)
                                2. Compute cosine sim between anchor and ALL tracklets
                                3. Return top-K ranked by score
                                4. Lazy-generate clip MP4s for results
                             → Returns {alternatives: [{rank, score, camera_id, track_id, clip_path}]}
                             → Frontend shows alternative clips with scores
                             → For each clip: <video src="/api/runs/{runId}/matched_alternatives/{clip_path}">
```

### Frame Viewing in Timeline
```
Timeline needs frame image → <img src="/api/runs/{runId}/full_frame?cameraId=c001&frameId=500">
                            → Backend reads stage0/c001/frame_000500.jpg
                            → Returns JPEG

Tracklet thumbnail        → <img src="/api/crops/run/{runId}?cameraId=c001&trackId=42&frameId=500">
                            → Backend crops bbox region from stage0 frame
                            → Returns JPEG
```

---

## 9. Verified Test Results

Three fresh end-to-end test runs were executed (2026-04-11), each starting from a **clean slate** (all `matched/` and `selected/` folders deleted):

| Run | Start Time | generatedAt (summary.json) | Tests | Result |
|-----|-----------|---------------------------|-------|--------|
| 1 | 10:33:29 | 10:36:41 | 10/10 | ALL PASS |
| 2 | 10:37:42 | 10:40:29 | 10/10 | ALL PASS |
| 3 | 10:41:15 | 10:43:50 | 10/10 | ALL PASS |

**Tests verified:**
- Health check, video registration, probe video exists
- Timeline query returns 14 matched trajectories (mode=matched)
- `matched/` folder created with `summary.json` + 42 MP4 clips across 5 cameras
- `top5_alternatives/` created with 5 alternative clips
- `matched_alternatives` endpoint returns legacy alternatives (5 items)
- Per-anchor alternatives endpoint returns 5 per-anchor alternatives
- Clip serving returns valid video/mp4 binary (197,888 bytes)
