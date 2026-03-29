# Spec: 09f + 12b Parallel Kaggle Notebooks

> **Created**: 2026-03-29
> **Status**: READY FOR IMPLEMENTATION
> **Account**: ali369 (lolo)
> **Branch**: feature/people-tracking

---

## Table of Contents
1. [Part A: 09f — Vehicle ReID CityFlowV2 Fine-tuning](#part-a-09f)
2. [Part B: 12b — Person WILDTRACK Tracking + ReID Pipeline](#part-b-12b)
3. [Design Decisions](#design-decisions)
4. [Critical Rules](#critical-rules)
5. [Implementation Order](#implementation-order)

---

## Part A: 09f — Vehicle ReID CityFlowV2 Fine-tuning {#part-a-09f}

### Goal
Load the VeRi-776-pretrained ResNet101-IBN-a checkpoint from 09e (62.52% mAP on VeRi-776) and fine-tune on CityFlowV2 for vehicle ReID. This is Phase 1a of the SOTA integration plan — the critical middle step that should raise ResNet101-IBN-a from 52.77% → 70%+ mAP on CityFlowV2, enabling a meaningful 2-model ensemble with the ViT.

### Hypothesis
The ViT achieves 80.14% mAP because of 3-stage progressive specialization (CLIP → VeRi-776 → CityFlowV2). The ResNet101-IBN-a currently skips the VeRi-776 step (ImageNet → CityFlowV2 = 52.77%). Adding VeRi-776 pretraining should close most of that gap: ImageNet → VeRi-776 (62.52%) → CityFlowV2 → target 70%+ mAP.

### Kernel Metadata

```json
{
  "id": "ali369/09f-vehicle-reid-resnet101-ibn-a-cityflowv2-finetune",
  "title": "09f Vehicle ReID ResNet101-IBN-a CityFlowV2 Fine-tune (VeRi-776 Pretrained)",
  "code_file": "09f_vehicle_reid_resnet101ibn_cityflowv2.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4",
  "keywords": [],
  "dataset_sources": [
    "thanhnguyenle/data-aicity-2023-track-2",
    "mrkdagods/mtmc-weights"
  ],
  "kernel_sources": [
    "ali369/09e-vehicle-reid-resnet101-ibn-a-veri-776-pretrain"
  ],
  "competition_sources": [],
  "model_sources": []
}
```

**Dataset Sources Explained:**
- `thanhnguyenle/data-aicity-2023-track-2` — CityFlowV2 raw video + GT annotations (same as 09d)
- `mrkdagods/mtmc-weights` — Shared model weights (IBN-Net pretrained, etc.)
- **kernel_sources**: `ali369/09e-...` — 09e output contains `best_model.pth` (VeRi-776 pretrained ResNet101-IBN-a, 525MB)

### Cell-by-Cell Plan

#### Cell 0 — Markdown Title
```markdown
# 09f Vehicle ReID: ResNet101-IBN-a CityFlowV2 Fine-tune (VeRi-776 Pretrained)

**Inputs:**
- 09e kernel output: VeRi-776-pretrained ResNet101-IBN-a checkpoint (`best_model.pth`, 62.52% mAP on VeRi-776)
- CityFlowV2 dataset: `thanhnguyenle/data-aicity-2023-track-2`

**Training:**
- Fine-tune on CityFlowV2 with lower LR (pretrained), 384×384, bag-of-tricks
- ID loss (label smoothing) + Triplet loss + Circle loss, cosine scheduler

**Outputs:**
- `/kaggle/working/resnet101ibn_veri776_cityflowv2_384px_best.pth` — Best fine-tuned checkpoint
- `/kaggle/working/training_history_09f.json` — Training metrics log
```

#### Cell 1 — Logger Setup
Same pattern as 09d cell 1: Tee stdout/stderr to `/kaggle/working/debug.log`.

#### Cell 2 — GPU Compatibility + Installs
Same pattern as 09e cell 2:
- Check `nvidia-smi` for GPU type
- If P100 (sm_60): downgrade to PyTorch 2.4.1+cu124
- Install: `timm==0.9.16`, `loguru`, `omegaconf`, `scikit-learn`
- Import torch, verify CUDA

#### Cell 3 — Clone Repo
Same as 09e cell 3:
```python
!git clone --depth 1 -b feature/people-tracking https://github.com/MRKDaGods/gp.git /kaggle/working/gp
import sys; sys.path.insert(0, "/kaggle/working/gp")
```

#### Cell 4 — Install Repo Dependencies
```python
!cd /kaggle/working/gp && pip install -q -r requirements.txt
!pip install -q torchreid  # for dataset parsing if needed
```

#### Cell 5 — Config & Paths Setup
```python
# --- Data Sources ---
CITYFLOWV2_ROOT = "/kaggle/input/data-aicity-2023-track-2"
WEIGHTS_DIR = "/kaggle/input/mtmc-weights"

# --- 09e Pretrained Checkpoint ---
# Search multiple possible paths for 09e output
PRETRAINED_CANDIDATES = [
    "/kaggle/input/09e-vehicle-reid-resnet101-ibn-a-veri-776-pretrain/best_model.pth",
    "/kaggle/input/09e-vehicle-reid-resnet101-ibn-a-veri-776-pretrain/09e_vehicle_reid_resnet101ibn_veri776/best_model.pth",
    "/kaggle/input/09e-vehicle-reid-resnet101-ibn-a-veri-776-pretrain/resnet101ibn_veri776_best.pth",
    "/kaggle/input/09e-vehicle-reid-resnet101-ibn-a-veri-776-pretrain/09e_vehicle_reid_resnet101ibn_veri776/reid_veri776_resnet101_ibn_a_best.pth",
]
# Find whichever exists
PRETRAINED_PATH = None
for p in PRETRAINED_CANDIDATES:
    if os.path.exists(p):
        PRETRAINED_PATH = p
        break
assert PRETRAINED_PATH is not None, f"Cannot find 09e checkpoint! Searched: {PRETRAINED_CANDIDATES}"

# --- Training Config ---
BACKBONE = "resnet101_ibn_a"
IMG_SIZE = (384, 384)
FEAT_DIM = 2048
GEM_P = 3.0

EPOCHS = 60          # Shorter than 09d/09e (120) — pretrained converges faster
BATCH_SIZE = 32      # T4 16GB with 384×384
NUM_INSTANCES = 4    # PK sampler
LR = 7e-5            # 1/5 of base 3.5e-4 — lower for fine-tuning
WARMUP_EPOCHS = 5    # Shorter warmup for pretrained
LABEL_SMOOTHING = 0.1
TRIPLET_MARGIN = 0.3
CIRCLE_M = 0.25
CIRCLE_GAMMA = 80
RANDOM_ERASING = 0.5
COLOR_JITTER = True  # Helps domain transfer VeRi→CityFlowV2
EVAL_EVERY = 5
FP16 = True

OUTPUT_DIR = "/kaggle/working/09f_output"
```

#### Cell 6 — Extract CityFlowV2 Crops
**Directly reuse 09d cell 5 logic verbatim**: Mount CityFlowV2, read GT annotations, extract vehicle crops from video frames. Writes to `/kaggle/working/cityflowv2_reid/`.

Key parameters (same as 09d):
- Max 15 samples per track
- Min area 2000px
- Output: `cityflowv2_reid/{train,query,gallery}/XXXX_SCENE_cNNN_fFFFFFF.jpg`

#### Cell 7 — Parse Dataset Splits
**Reuse 09d cell 7 logic**: Parse cropped images into `train_data`, `query_data`, `gallery_data`. Extract `NUM_CLASSES`, `NUM_CAMERAS` from filenames.

#### Cell 8 — Build Dataloaders
**Reuse 09d cell 8 logic**: ReIDDataset + PKSampler + transforms.

Key differences from 09d:
- **Image size: (384, 384)** — not (256, 256)
- **Color jitter: enabled** — for domain transfer from VeRi-776
- **Random erasing: 0.5** — same as 09d
- **Batch size: 32** — reduced for 384px on T4 (was 64 at 256px in 09d, but 09d also used 384px... keep 32 to be safe with T4 memory)

#### Cell 9 — Build Model + Load VeRi-776 Pretrained Weights
```python
from src.training.model import ReIDModelResNet101IBN

# Build model with CityFlowV2 class count (classifier will be reinitialized)
model = ReIDModelResNet101IBN(
    num_classes=NUM_CLASSES,  # CityFlowV2 train IDs
    last_stride=1,
    pretrained=False,  # Don't load ImageNet — we're loading VeRi-776 instead
    gem_p=GEM_P,
)

# Load VeRi-776 pretrained checkpoint
ckpt = torch.load(PRETRAINED_PATH, map_location="cpu", weights_only=False)
# Handle different checkpoint formats from train_reid.py:
#   format 1: {"model": state_dict, "optimizer": ..., "epoch": ...}
#   format 2: {"state_dict": state_dict}
#   format 3: raw state_dict
if "model" in ckpt:
    pretrained_sd = ckpt["model"]
elif "state_dict" in ckpt:
    pretrained_sd = ckpt["state_dict"]
else:
    pretrained_sd = ckpt

# Strip 'module.' prefix if DataParallel
pretrained_sd = {k.replace("module.", ""): v for k, v in pretrained_sd.items()}

# Load backbone + GeM + bottleneck weights (skip classifier — different num_classes)
loaded_keys = []
skipped_keys = []
model_sd = model.state_dict()
for k, v in pretrained_sd.items():
    if k.startswith("classifier"):
        skipped_keys.append(k)  # Skip — different num_classes
        continue
    if k in model_sd and v.shape == model_sd[k].shape:
        model_sd[k] = v
        loaded_keys.append(k)
    else:
        skipped_keys.append(k)

model.load_state_dict(model_sd)
print(f"Loaded {len(loaded_keys)} params from VeRi-776 pretrain, skipped {len(skipped_keys)}")
print(f"Skipped keys: {skipped_keys}")
# Expect: loaded = backbone.* + pool.* + bottleneck.*, skipped = classifier.*

model = model.to(device)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

**CRITICAL NOTE — Code Fix Required**: The existing `ReIDModelResNet101IBN` does NOT have a `load_pretrained_reid()` method (unlike `ReIDModelBoT`). The `--pretrained-reid` flag in `train_reid.py` calls `model.load_pretrained_reid()` which would fail for `resnet101_ibn_a` backbone. For 09f, we do the loading manually in-notebook (as shown above) to avoid this bug. A follow-up PR should add `load_pretrained_reid()` to `ReIDModelResNet101IBN`.

#### Cell 10 — Loss Functions
**Reuse 09d cell 10 logic**:
- CrossEntropyLabelSmooth (ε=0.1)
- TripletLoss (margin=0.3)
- CircleLoss (m=0.25, γ=80)

All three losses combined: `L = L_id + L_tri + L_circle`

#### Cell 11 — Optimizer + Scheduler (Fine-tuning Config)
```python
# Fine-tuning: lower backbone LR, higher head LR
base_model = model.module if hasattr(model, "module") else model
params = [
    {"params": base_model.backbone.parameters(), "lr": LR * 0.1},    # 7e-6 — very gentle on pretrained backbone
    {"params": base_model.pool.parameters(), "lr": LR},               # 7e-5
    {"params": base_model.bottleneck.parameters(), "lr": LR},         # 7e-5
    {"params": base_model.classifier.parameters(), "lr": LR * 10},    # 7e-4 — fresh classifier needs faster LR
]
optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=5e-4)

# Cosine scheduler with short warmup (pretrained model)
scheduler = build_cosine_scheduler(
    optimizer,
    warmup_epochs=WARMUP_EPOCHS,  # 5 epochs
    total_epochs=EPOCHS,           # 60 epochs
    eta_min=1e-7,
)
scaler = torch.amp.GradScaler("cuda") if FP16 else None
```

**Rationale for LR choices:**
- Backbone at 0.1× base (7e-6): VeRi-776 features are already vehicle-domain-specific, only fine adjustment needed
- Classifier at 10× base (7e-4): Completely new head for CityFlowV2 IDs, needs to converge quickly
- 60 epochs total (vs 120 in 09d/09e): Pretrained backbone needs less training to converge

#### Cell 12 — Feature Extraction + Evaluation
**Reuse 09d cell 12 logic verbatim**: Horizontal flip augmentation, L2 normalize, cosine distance, mAP + CMC@1/5/10. Ignore same-camera probe-gallery pairs.

#### Cell 13 — Main Training Loop
**Reuse 09d cell 13 structure** with these modifications:
- 60 epochs (not 120)
- Evaluate every 5 epochs
- Save checkpoints to `OUTPUT_DIR/checkpoints/`
- Track best mAP, save best model
- Print per-epoch: loss, id_loss, tri_loss, circle_loss, LR, mAP, R1

#### Cell 14 — Final Validation
Load best checkpoint, run final mAP/CMC evaluation, print summary.

#### Cell 15 — Save Artifacts
```python
# Primary output: fine-tuned checkpoint for pipeline deployment
shutil.copy(
    f"{OUTPUT_DIR}/checkpoints/best_model.pth",
    "/kaggle/working/resnet101ibn_veri776_cityflowv2_384px_best.pth"
)

# Training history
with open("/kaggle/working/training_history_09f.json", "w") as f:
    json.dump(history, f, indent=2)

# Also save just the state_dict for lightweight deployment
best_ckpt = torch.load(f"{OUTPUT_DIR}/checkpoints/best_model.pth", map_location="cpu", weights_only=False)
torch.save(
    {"state_dict": best_ckpt["model"]},
    "/kaggle/working/resnet101ibn_veri776_cityflowv2_384px_deploy.pth"
)
```

#### Cell 16 — Checkpoint Validation
Verify the saved checkpoint contains expected keys: `backbone.conv1.*`, `backbone.layer1-4.*`, `pool.p`, `bottleneck.*`, `classifier.*`.

### Expected Outcomes
| Metric | 09d (no VeRi) | 09f (VeRi pretrain) | Rationale |
|--------|:-------------:|:-------------------:|-----------|
| mAP | 52.77% | **68-75%** | VeRi-776 provides vehicle-domain features |
| CMC R1 | ~78% | **85-90%** | Better feature discrimination |
| CMC R5 | ~87% | **92-96%** | — |

### Artifacts Produced
| File | Description | Size (est.) |
|------|------------|:-----------:|
| `resnet101ibn_veri776_cityflowv2_384px_best.pth` | Full checkpoint (model+optimizer+scheduler) | ~525MB |
| `resnet101ibn_veri776_cityflowv2_384px_deploy.pth` | State dict only (for pipeline deployment) | ~170MB |
| `training_history_09f.json` | Per-epoch loss/mAP/CMC | ~10KB |
| `debug.log` | Full training log | ~5MB |

### Downstream Usage
After 09f completes successfully:
1. Upload `resnet101ibn_veri776_cityflowv2_384px_deploy.pth` to Kaggle dataset `mrkdagods/mtmc-weights`
2. Update 10a notebook to load as secondary ReID model
3. Update stage2 config to enable 2-model ensemble: `stage2.reid.secondary.weights_path` + fusion weight α
4. Re-sweep association parameters with ensemble features

---

## Part B: 12b — Person WILDTRACK Tracking + ReID Pipeline {#part-b-12b}

### Goal
Take 12a's ground-plane detections (MVDeTr, 92.0% MODA), run temporal tracking in world coordinates, extract person ReID features by projecting tracked positions to camera crops, perform cross-camera ReID-free evaluation and ReID-based association, and evaluate the full person MTMC pipeline on WILDTRACK.

### Hypothesis
12a already achieves 92.0% MODA on detection — the bottleneck is now tracking + identity association. By combining MVDeTr's ground-plane detections with our existing Hungarian tracking code (`src/stage_wildtrack_mvdetr/pipeline.py`) and adding person ReID features extracted from camera crops, we can produce high-quality MTMC tracklets. The ground-plane tracking approach is fundamentally better than per-camera tracking (which produced 800+ tracklets for ~20 people).

### Kernel Metadata

```json
{
  "id": "ali369/12b-wildtrack-mvdetr-tracking-reid",
  "title": "12b WILDTRACK MVDeTr Tracking + ReID Pipeline",
  "code_file": "12b_wildtrack_mvdetr_tracking_reid.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4",
  "keywords": [],
  "dataset_sources": [
    "mrkdagods/mtmc-weights",
    "aryashah2k/large-scale-multicamera-detection-dataset"
  ],
  "kernel_sources": [
    "ali369/12a-wildtrack-mvdetr-training"
  ],
  "competition_sources": [],
  "model_sources": []
}
```

**Dataset Sources Explained:**
- `mrkdagods/mtmc-weights` — Contains `person_transreid_vit_base_market1501.pth` (pretrained person ReID model)
- `aryashah2k/large-scale-multicamera-detection-dataset` — WILDTRACK dataset (Image_subsets/, annotations_positions/, calibrations/)
- **kernel_sources**: `ali369/12a-wildtrack-mvdetr-training` — 12a output contains MVDeTr detections (`test.txt`) and trained model

### Person ReID Model Decision

**Selected: TransReID ViT-Base/16 CLIP pretrained on Market-1501**
- Already exists in `mrkdagods/mtmc-weights` as `person_transreid_vit_base_market1501.pth`
- Already used by our person pipeline (stage2 config: `stage2.reid.person`)
- 768D embeddings, clip normalization
- Market-1501 has 751 person identities — reasonable generalization to WILDTRACK's ~20 people
- **No WILDTRACK-specific fine-tuning needed for v1** — the 20 WILDTRACK identities are too few for meaningful ReID training

**Alternative considered but rejected for v1:**
- OSNet-x1.0 MSMT17: Smaller model but lower quality features
- Training a WILDTRACK-specific ReID model: Only ~20 IDs, would overfit immediately
- ResNet50-IBN-a Market-1501: Available but TransReID ViT is strictly better

### Cell-by-Cell Plan

#### Cell 0 — Markdown Title
```markdown
# 12b WILDTRACK: MVDeTr Tracking + Person ReID Pipeline

**Pipeline:**
1. Load MVDeTr ground-plane detections from 12a (test.txt)
2. Parse grid coordinates → world coordinates (cm)
3. Run Hungarian temporal tracking on ground plane
4. Project tracked positions to 7 camera views → crop person patches
5. Extract ReID features (TransReID ViT-Base/16 Market-1501)
6. Cross-camera association using ReID features (optional — ground-plane tracking may already solve identity)
7. Save tracklets + trajectories in pipeline format
8. Evaluate: ground-plane MODA/IDF1 + per-camera MTMC metrics

**Inputs:**
- 12a output: MVDeTr detections (`test.txt`), trained model (`MultiviewDetector.pth`)
- WILDTRACK: Image_subsets/, annotations_positions/, calibrations/
- Person ReID model: `person_transreid_vit_base_market1501.pth`

**Outputs:**
- Ground-plane tracklets (JSON + CSV)
- Per-camera projected tracklets (pipeline format)
- Global trajectories with ReID features
- Evaluation metrics JSON
```

#### Cell 1 — Logger Setup
Same pattern as 12a: Tee stdout/stderr to `/kaggle/working/debug.log`.

#### Cell 2 — GPU Compatibility + Installs
Same pattern as 09e/12a:
- Check GPU type, downgrade PyTorch if P100
- Install: `timm==0.9.16`, `loguru`, `omegaconf`, `motmetrics`, `scikit-learn`, `scipy`

#### Cell 3 — Clone Repo
```python
!git clone --depth 1 -b feature/people-tracking https://github.com/MRKDaGods/gp.git /kaggle/working/gp
import sys; sys.path.insert(0, "/kaggle/working/gp")
```

#### Cell 4 — Install Repo Dependencies
```python
!cd /kaggle/working/gp && pip install -q -r requirements.txt
```

#### Cell 5 — Find Data Sources + Config
```python
import os, glob
from pathlib import Path

# --- 12a MVDeTr Output ---
# The 12a kernel outputs are mounted as a kernel source
MVDETR_OUTPUT_CANDIDATES = [
    "/kaggle/input/12a-wildtrack-mvdetr-training",
]
MVDETR_OUTPUT_DIR = None
for p in MVDETR_OUTPUT_CANDIDATES:
    if os.path.isdir(p):
        MVDETR_OUTPUT_DIR = p
        break
assert MVDETR_OUTPUT_DIR is not None, "Cannot find 12a kernel output!"

# Find test.txt — MVDeTr detection output
# 12a saves to a timestamped subdirectory
ARTIFACT_SUBDIR = "aug_deform_trans_lr0.0005_baseR0.1_neck128_out0_alpha1.0_id0_drop0.0_dropcam0.0_worldRK4_10_imgRK12_10_2026-03-29_01-34-57"
TEST_TXT_CANDIDATES = [
    f"{MVDETR_OUTPUT_DIR}/{ARTIFACT_SUBDIR}/test.txt",
    f"{MVDETR_OUTPUT_DIR}/test.txt",                     # direct
]
# Also glob for any test.txt
TEST_TXT_CANDIDATES += glob.glob(f"{MVDETR_OUTPUT_DIR}/**/test.txt", recursive=True)
DETECTIONS_PATH = None
for p in TEST_TXT_CANDIDATES:
    if os.path.isfile(p):
        DETECTIONS_PATH = p
        break
assert DETECTIONS_PATH is not None, f"Cannot find test.txt in {MVDETR_OUTPUT_DIR}!"
print(f"MVDeTr detections: {DETECTIONS_PATH}")

# --- WILDTRACK Dataset ---
WILDTRACK_CANDIDATES = [
    "/kaggle/input/large-scale-multicamera-detection-dataset/Wildtrack",
    "/kaggle/input/datasets/aryashah2k/large-scale-multicamera-detection-dataset/Wildtrack",
    "/kaggle/input/large-scale-multicamera-detection-dataset",
]
WILDTRACK_ROOT = None
for p in WILDTRACK_CANDIDATES:
    if os.path.isdir(p) and os.path.isdir(os.path.join(p, "Image_subsets")):
        WILDTRACK_ROOT = p
        break
assert WILDTRACK_ROOT is not None, "Cannot find WILDTRACK dataset!"
print(f"WILDTRACK root: {WILDTRACK_ROOT}")

IMAGE_SUBSETS_DIR = os.path.join(WILDTRACK_ROOT, "Image_subsets")
ANNOTATIONS_DIR = os.path.join(WILDTRACK_ROOT, "annotations_positions")
CALIBRATIONS_DIR = os.path.join(WILDTRACK_ROOT, "calibrations")

# --- Person ReID Model ---
REID_CANDIDATES = [
    "/kaggle/input/mtmc-weights/person_transreid_vit_base_market1501.pth",
    "/kaggle/working/gp/models/reid/person_transreid_vit_base_market1501.pth",
]
REID_WEIGHTS = None
for p in REID_CANDIDATES:
    if os.path.isfile(p):
        REID_WEIGHTS = p
        break
# ReID is optional for v1 — ground-plane tracking might suffice
if REID_WEIGHTS:
    print(f"Person ReID weights: {REID_WEIGHTS}")
else:
    print("WARNING: No person ReID weights found — will skip ReID feature extraction")

# --- Output ---
OUTPUT_DIR = "/kaggle/working/12b_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Pipeline Parameters ---
MAX_MATCH_DISTANCE_CM = 75.0   # Hungarian matching threshold
MAX_MISSED_FRAMES = 5          # Frames before track is terminated
MIN_TRACK_LENGTH = 3           # Min detections to keep a track
PERSON_HEIGHT_CM = 175.0       # For camera projection
FPS = 2.0                      # WILDTRACK annotation rate
IMAGE_SIZE = (1920, 1080)      # WILDTRACK camera resolution
```

#### Cell 6 — Load + Parse MVDeTr Detections
```python
from src.stage_wildtrack_mvdetr.pipeline import (
    load_mvdetr_ground_plane_detections,
    track_ground_plane_detections,
)

# Load detections from 12a's test.txt
detections = load_mvdetr_ground_plane_detections(
    DETECTIONS_PATH,
    normalize_wildtrack_frames=True,
)
print(f"Loaded {len(detections)} ground-plane detections")

# Basic stats
frames = set(d.frame_id for d in detections)
print(f"Frames: {len(frames)} (range {min(frames)}-{max(frames)})")
avg_per_frame = len(detections) / len(frames)
print(f"Average detections per frame: {avg_per_frame:.1f}")
```

#### Cell 7 — Run Hungarian Ground-Plane Tracking
```python
# Track detections across frames using Hungarian matching on world coordinates
tracks = track_ground_plane_detections(
    detections=detections,
    max_match_distance_cm=MAX_MATCH_DISTANCE_CM,
    max_missed_frames=MAX_MISSED_FRAMES,
    min_track_length=MIN_TRACK_LENGTH,
)
print(f"Tracked {len(tracks)} ground-plane identities")

# Track statistics
track_lengths = [len(t.detections) for t in tracks]
print(f"Track lengths: min={min(track_lengths)}, max={max(track_lengths)}, "
      f"mean={sum(track_lengths)/len(track_lengths):.1f}, median={sorted(track_lengths)[len(track_lengths)//2]}")

# Save ground-plane tracks
from src.stage_wildtrack_mvdetr.pipeline import _save_ground_plane_tracks, _save_ground_plane_csv
_save_ground_plane_tracks(tracks, Path(OUTPUT_DIR) / "ground_plane_tracks.json")
_save_ground_plane_csv(tracks, Path(OUTPUT_DIR) / "ground_plane_tracks.csv")
```

#### Cell 8 — Project Tracks to Camera Views + Generate Tracklets
```python
from src.stage_wildtrack_mvdetr.pipeline import _tracks_to_projected_tracklets
from src.core.wildtrack_calibration import load_wildtrack_calibration
from src.core.io_utils import save_tracklets_by_camera, save_global_trajectories

# Load WILDTRACK camera calibrations
calibrations = load_wildtrack_calibration(CALIBRATIONS_DIR)
print(f"Loaded calibrations for {len(calibrations)} cameras: {sorted(calibrations.keys())}")

# Project ground-plane tracks to all camera views → per-camera tracklets + global trajectories
tracklets_by_camera, trajectories = _tracks_to_projected_tracklets(
    tracks=tracks,
    calibrations=calibrations,
    fps=FPS,
    image_size=IMAGE_SIZE,
)

# Stats
total_tracklets = sum(len(v) for v in tracklets_by_camera.values())
print(f"Projected to {total_tracklets} per-camera tracklets across {len(tracklets_by_camera)} cameras")
for cam_id, tracklets in sorted(tracklets_by_camera.items()):
    print(f"  {cam_id}: {len(tracklets)} tracklets")
print(f"Global trajectories: {len(trajectories)}")

# Save tracklets in pipeline format
save_tracklets_by_camera(tracklets_by_camera, Path(OUTPUT_DIR) / "tracklets")
save_global_trajectories(trajectories, Path(OUTPUT_DIR) / "global_trajectories.json")
```

#### Cell 9 — Extract Person ReID Features (GPU)
```python
# This cell is optional but highly valuable for cross-camera association quality
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T

if REID_WEIGHTS is None:
    print("Skipping ReID feature extraction — no weights available")
    REID_FEATURES = None
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build TransReID model
    from src.stage2_features.reid_models import ReIDModel
    reid_model = ReIDModel(
        model_name="transreid",
        weights_path=REID_WEIGHTS,
        embedding_dim=768,
        input_size=(256, 128),
        vit_model="vit_base_patch16_clip_224.openai",
        num_cameras=6,
        camera_bn=False,  # No camera-specific BN for WILDTRACK
        clip_normalization=True,
    )
    reid_model.model.to(device)
    reid_model.model.eval()

    # Transform for person crops
    transform = T.Compose([
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # For each trajectory, crop person patches from camera images and extract features
    # WILDTRACK image naming: Image_subsets/C{cam_idx}/00000{frame_idx*5}.png
    camera_dirs = sorted(Path(IMAGE_SUBSETS_DIR).iterdir())
    cam_name_to_dir = {}
    for d in camera_dirs:
        if d.is_dir() and d.name.startswith("C"):
            cam_name_to_dir[d.name] = d

    REID_FEATURES = {}  # global_id -> np.ndarray (768,)

    for traj in trajectories:
        all_crops = []
        for tracklet in traj.tracklets:
            cam_dir = cam_name_to_dir.get(tracklet.camera_id)
            if cam_dir is None:
                continue

            # Sample up to 8 frames per tracklet for feature averaging
            sample_frames = tracklet.frames[::max(1, len(tracklet.frames) // 8)][:8]
            for frame in sample_frames:
                # WILDTRACK frame naming: {frame_id * 5:08d}.png
                wildtrack_frame = frame.frame_id * 5
                img_path = cam_dir / f"{wildtrack_frame:08d}.png"
                if not img_path.exists():
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Crop person bbox
                x1, y1, x2, y2 = [int(c) for c in frame.bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                all_crops.append(Image.fromarray(crop))

        if not all_crops:
            continue

        # Batch extract features
        batch = torch.stack([transform(c) for c in all_crops]).to(device)
        with torch.no_grad():
            features = reid_model.model(batch)  # (N, 768)
        # Average and L2-normalize
        mean_feat = features.mean(dim=0)
        mean_feat = torch.nn.functional.normalize(mean_feat, p=2, dim=0)
        REID_FEATURES[traj.global_id] = mean_feat.cpu().numpy()

    print(f"Extracted ReID features for {len(REID_FEATURES)}/{len(trajectories)} trajectories")

    # Save features
    np.savez(
        f"{OUTPUT_DIR}/reid_features.npz",
        **{str(k): v for k, v in REID_FEATURES.items()}
    )
```

#### Cell 10 — Ground-Plane Evaluation (MODA / IDF1)
```python
from src.stage5_evaluation.ground_plane_eval import evaluate_wildtrack_ground_plane

# This evaluates on the ground-plane — the standard WILDTRACK evaluation protocol
eval_result = evaluate_wildtrack_ground_plane(
    trajectories=trajectories,
    annotations_dir=ANNOTATIONS_DIR,
    calibrations_dir=CALIBRATIONS_DIR,
    conf_threshold=0.25,
    match_threshold_cm=50.0,
    nms_radius_cm=50.0,
)

print("=" * 60)
print("WILDTRACK Ground-Plane Evaluation")
print("=" * 60)
print(f"MODA:       {eval_result.moda:.4f}")
print(f"IDF1:       {eval_result.idf1:.4f}")
print(f"Precision:  {eval_result.precision:.4f}")
print(f"Recall:     {eval_result.recall:.4f}")
print(f"ID Switches: {eval_result.id_switches}")

# Save evaluation
import json
eval_summary = {
    "moda": eval_result.moda,
    "idf1": eval_result.idf1,
    "precision": eval_result.precision,
    "recall": eval_result.recall,
    "id_switches": eval_result.id_switches,
    "num_trajectories": len(trajectories),
    "num_tracklets": sum(len(v) for v in tracklets_by_camera.values()),
    "tracking_params": {
        "max_match_distance_cm": MAX_MATCH_DISTANCE_CM,
        "max_missed_frames": MAX_MISSED_FRAMES,
        "min_track_length": MIN_TRACK_LENGTH,
    },
}
with open(f"{OUTPUT_DIR}/evaluation_summary.json", "w") as f:
    json.dump(eval_summary, f, indent=2)
```

#### Cell 11 — Tracking Parameter Sweep (Optional)
```python
# Quick sweep to find optimal tracking parameters
# This is cheap (CPU-only, just re-runs Hungarian matching on already-loaded detections)
best_idf1 = -1
best_params = {}

for max_dist in [50.0, 75.0, 100.0, 125.0]:
    for max_missed in [3, 5, 8, 10]:
        for min_len in [2, 3, 5]:
            tracks_sweep = track_ground_plane_detections(
                detections=detections,
                max_match_distance_cm=max_dist,
                max_missed_frames=max_missed,
                min_track_length=min_len,
            )
            _, trajs_sweep = _tracks_to_projected_tracklets(
                tracks_sweep, calibrations, FPS, IMAGE_SIZE
            )
            result = evaluate_wildtrack_ground_plane(
                trajs_sweep, ANNOTATIONS_DIR, CALIBRATIONS_DIR,
                conf_threshold=0.25, match_threshold_cm=50.0, nms_radius_cm=50.0,
            )
            if result.idf1 > best_idf1:
                best_idf1 = result.idf1
                best_params = {
                    "max_match_distance_cm": max_dist,
                    "max_missed_frames": max_missed,
                    "min_track_length": min_len,
                    "moda": result.moda,
                    "idf1": result.idf1,
                    "id_switches": result.id_switches,
                    "num_tracks": len(tracks_sweep),
                }

print(f"Best IDF1: {best_idf1:.4f}")
print(f"Best params: {json.dumps(best_params, indent=2)}")

# Save sweep results
with open(f"{OUTPUT_DIR}/tracking_sweep_best.json", "w") as f:
    json.dump(best_params, f, indent=2)
```

#### Cell 12 — Package Artifacts
```python
import shutil

# List all outputs
print("=" * 60)
print("12b Output Artifacts:")
print("=" * 60)
for f in sorted(Path(OUTPUT_DIR).rglob("*")):
    if f.is_file():
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.relative_to(OUTPUT_DIR)}: {size_mb:.2f} MB")

# Copy key artifacts to /kaggle/working/ for easy download
shutil.copy(f"{OUTPUT_DIR}/evaluation_summary.json", "/kaggle/working/evaluation_summary.json")
shutil.copy(f"{OUTPUT_DIR}/ground_plane_tracks.json", "/kaggle/working/ground_plane_tracks.json")
if os.path.exists(f"{OUTPUT_DIR}/tracking_sweep_best.json"):
    shutil.copy(f"{OUTPUT_DIR}/tracking_sweep_best.json", "/kaggle/working/tracking_sweep_best.json")
if os.path.exists(f"{OUTPUT_DIR}/reid_features.npz"):
    shutil.copy(f"{OUTPUT_DIR}/reid_features.npz", "/kaggle/working/reid_features.npz")
```

### Expected Outcomes

Since 12a achieves 92.0% MODA on detection, the tracking quality depends on temporal association:

| Metric | 12a (detection only) | 12b (tracking + ID) | Notes |
|--------|:-------------------:|:-------------------:|-------|
| MODA | 92.0% | **88-92%** | Tracking can drop MODA slightly (short tracks filtered) |
| IDF1 | N/A | **60-80%** | Depends on temporal consistency of Hungarian matching |
| ID switches | N/A | **<50** | Ground-plane tracking should be very consistent for ~20 people |
| Num trajectories | N/A | **18-25** | Should roughly match WILDTRACK GT (~20 people) |

### Artifacts Produced
| File | Description | Size (est.) |
|------|------------|:-----------:|
| `ground_plane_tracks.json` | Tracked identities with world coordinates | ~2MB |
| `ground_plane_tracks.csv` | Same in CSV format | ~1MB |
| `tracklets/` | Per-camera projected tracklets (pipeline format) | ~5MB |
| `global_trajectories.json` | Global trajectories for evaluation | ~3MB |
| `reid_features.npz` | Per-trajectory ReID feature vectors (768D) | ~1MB |
| `evaluation_summary.json` | MODA, IDF1, precision, recall, ID switches | ~1KB |
| `tracking_sweep_best.json` | Best tracking parameters from sweep | ~1KB |

### Downstream Usage
After 12b completes:
1. Analyze evaluation metrics — how close to WILDTRACK SOTA (IDF1~85%)?
2. If ReID features extracted, evaluate cross-camera re-identification quality
3. Feed into 12c (if needed) for Stage 4-5 association + evaluation with ReID features
4. Use tracking sweep results to update `configs/datasets/wildtrack.yaml`

---

## Design Decisions {#design-decisions}

### 09f: Fine-tuning Strategy

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Learning rate** | 7e-5 base (1/5 of original 3.5e-4) | Standard fine-tuning practice: pretrained model needs gentler updates |
| **Backbone LR** | 7e-6 (0.1× base) | VeRi-776 features are already vehicle-specific |
| **Classifier LR** | 7e-4 (10× base) | Fresh head for CityFlowV2's different ID space |
| **Epochs** | 60 (vs 120 for 09d/09e) | Pretrained model converges faster; avoid overfitting on small dataset |
| **Warmup** | 5 epochs (vs 10) | Less warmup needed when not training from scratch |
| **Scheduler** | Cosine annealing | Smoother than step decay for fine-tuning |
| **Freeze backbone?** | No | Differential LR (0.1×) is more flexible than hard freezing |
| **Image size** | 384×384 | Same as 09d; matches deployment target |
| **Losses** | ID + Triplet + Circle | Full bag-of-tricks; Circle loss helps with hard pairs across domains |
| **Color jitter** | Enabled | Helps bridge VeRi-776 → CityFlowV2 domain gap |
| **Batch size** | 32 | Conservative for T4 16GB @ 384×384; 09d used 64 but that was also 384px — monitor OOM |
| **`pretrained=False` in model init** | Yes | Don't load ImageNet → directly load VeRi-776 checkpoint instead |

### 12b: Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Tracking plane** | Ground plane (world coords) | WILDTRACK cameras all overlap; ground-plane tracking avoids 7× duplicate detections |
| **Tracking algorithm** | Hungarian matching (existing code) | Already implemented in `pipeline.py`, simple and reliable |
| **Person ReID model** | TransReID ViT-Base/16 Market-1501 | Already available on Kaggle, 768D features, proven on person ReID |
| **ReID fine-tuning?** | No (v1) | Only ~20 WILDTRACK IDs, would overfit; Market-1501 pretrain generalizes |
| **GPU needed?** | Yes (for ReID extraction) | TransReID inference on ~20 tracks × 7 cameras × 8 samples = ~1120 crops — fast on GPU |
| **Person crop method** | Project ground-plane position to camera using calibration, then estimate bbox from foot+head projection | Uses existing `_make_bbox_from_ground_point()` which projects foot (z=0) and head (z=175cm) |
| **Evaluation** | Ground-plane MODA/IDF1 (standard WILDTRACK protocol) | Uses existing `evaluate_wildtrack_ground_plane()` with 50cm L2 matching threshold |
| **Tracking sweep?** | Yes (CPU, fast) | Grid over {max_dist, max_missed, min_len} to find optimal params |

### WILDTRACK Annotation Format (for reference)
```
annotations_positions/{frame_id:08d}.json
[
  {
    "personID": 42,
    "positionID": 2591,           # grid cell index = y*480 + x
    "views": [
      {"viewNum": 0, "xmin": 123, "ymin": 45, "xmax": 189, "ymax": 290},
      ...
    ]
  },
  ...
]
```
- `positionID` → grid (x, y) → world coordinates (cm) via WILDTRACK constants
- Frame naming: WILDTRACK uses every-5th frame numbering (0, 5, 10, ..., 1795)
- Normalized frame_id = wildtrack_frame // 5 (0, 1, 2, ..., 359)

---

## Critical Rules {#critical-rules}

### Both Notebooks
1. **Frame IDs**: Internal pipeline uses 0-based frame IDs. WILDTRACK raw frames are every-5th (0, 5, 10...) — normalize by dividing by 5. MOT submission format uses 1-based (converted via `frame_id + 1`).
2. **NEVER use `replace_string_in_file` on `.ipynb`** — use `json.load() → modify → json.dump()` via Python script instead.
3. **Each line in notebook `source` arrays MUST end with `\n` EXCEPT the last line** — without this, Kaggle/papermill concatenates lines → SyntaxError.
4. **Use `ensure_ascii=True` in `json.dump`** for Windows compatibility (avoid charmap codec errors).
5. **Config override paths**: Stage 4 association params are `stage4.association.X`, NOT `stage4.X`.
6. **After any .ipynb edit, verify on-disk state**: `python -c "import json; nb=json.load(open('file.ipynb')); print(nb['cells'][0]['source'][:2])"`
7. **P100 GPU compatibility**: Check for sm_60 and downgrade to PyTorch 2.4.1+cu124 if needed.

### 09f Specific
8. **`pretrained=False` in `ReIDModelResNet101IBN.__init__`**: We're loading VeRi-776 weights, not ImageNet. Setting `pretrained=True` would first load ImageNet then overwrite with VeRi-776 — wasteful.
9. **Classifier layer is NOT loadable from 09e**: VeRi-776 has 576 classes, CityFlowV2 has different count. Must skip classifier keys when loading.
10. **Bug: `ReIDModelResNet101IBN` lacks `load_pretrained_reid()`**: The `--pretrained-reid` flag in `train_reid.py` would crash for this backbone. Handle weight loading manually in-notebook.

### 12b Specific
11. **Ground-plane evaluation uses L2 distance in cm**: Standard WILDTRACK threshold is 50cm for matching GT to predictions.
12. **MVDeTr test.txt format**: `frame_id grid_x grid_y` — where frame_id is raw WILDTRACK frame (0, 5, 10...) and grid_x/grid_y are ground-plane grid indices (converted to world cm by `_grid_to_world_cm()`).
13. **Image path convention**: `Image_subsets/C{n}/{frame_id:08d}.png` where frame_id is raw WILDTRACK frame (0, 5, 10...).
14. **Person crop quality**: Projected bboxes from ground-plane may be noisy for cameras at oblique angles. Use padding and min-area filtering.

---

## Implementation Order {#implementation-order}

Both notebooks can be **built in parallel** — they have no dependencies on each other:

```
09f ─────────────────────────────────────────→ Push → Run (12h est.) → Analyze
12b ─────────────────────────────────────────→ Push → Run (2h est.) → Analyze
```

### Step-by-step:
1. **Create directory structure**:
   - `notebooks/kaggle/09f_vehicle_reid_resnet101ibn_cityflowv2/`
   - `notebooks/kaggle/12b_wildtrack_mvdetr_tracking_reid/`

2. **Build 09f notebook**: Adapt 09d notebook, modify cells 5→cell 9 (VeRi pretrain loading), cell 11 (fine-tuning optimizer), cell 15 (artifact names)

3. **Build 12b notebook**: New notebook, reuse 12a patterns for setup cells, use existing `pipeline.py` functions for cells 6-8, add ReID extraction in cell 9

4. **Create kernel-metadata.json** for both

5. **Push both**: `kaggle kernels push -p notebooks/kaggle/09f_vehicle_reid_resnet101ibn_cityflowv2/` and same for 12b

6. **Monitor**: `python scripts/kaggle_logs.py ali369/09f-vehicle-reid-resnet101-ibn-a-cityflowv2-finetune --tail 50`

### Time Estimates on T4
- **09f**: ~8-12h (60 epochs × 384×384 × ~7.5K images, eval every 5 epochs)
- **12b**: ~1-2h (tracking is CPU, ReID extraction on ~1K crops is fast)

### What Comes Next
- **If 09f succeeds (mAP > 65%)**: Upload checkpoint → Enable ensemble in 10a → Re-sweep association → Expected +1.5-2.5pp vehicle IDF1
- **If 12b succeeds (IDF1 > 50%)**: Analyze error profile → Build 12c for improved association → Consider WILDTRACK-specific ReID training
- **If 09f mAP < 60%**: Investigate — may need more epochs, different LR, or unfreezing schedule
- **If 12b IDF1 < 40%**: Tracking params need more tuning, or ReID features are not generalizing from Market-1501