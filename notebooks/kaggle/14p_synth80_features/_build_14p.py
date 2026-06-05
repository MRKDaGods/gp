"""Builder for 14p_synth80_features.ipynb.

Clones the 14j R50-IBN extraction notebook structure and swaps ONLY the model:
load `resnext101ibn_synth80_best.pth` into an INLINED 09w `ResNeXt101IBNNeck`
(ResNeXt101-32x4d + IBN-a + GeM + BNNeck, trained @384). The crop extraction,
tracklet iteration order, dropped-index handling, per-crop hflip-TTA + L2,
softmax-quality-weighted mean (temp=3.0), and final L2-renorm are byte-identical
to 14j v4 so the downstream drift gate stays aligned.

We inline (not import) the arch because src.training.model.ReIDModelResNeXt101IBN
uses a DIFFERENT topology (resnext101_32x8d) and DIFFERENT state_dict key naming
(backbone.conv1.* / backbone.layer1.*) than the 09w checkpoint
(resnext101_32x4d, Sequential backbone -> keys backbone.0.* ... backbone.7.*).
A strict load of the 09w checkpoint requires the inlined 09w arch.

Run:  .venv/Scripts/python.exe notebooks/kaggle/14p_synth80_features/_build_14p.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF_14J = HERE.parent / "14j_r50ibn_features" / "14j_r50ibn_features.ipynb"
OUT = HERE / "14p_synth80_features.ipynb"


def md(text: str) -> dict:
    lines = text.split("\n")
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(text: str) -> dict:
    lines = text.split("\n")
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


# ---------------------------------------------------------------------------
# Cell sources
# ---------------------------------------------------------------------------

C0 = """# 14p Synth80 Feature Extraction

Kernel slug: `yahiaakhalafallah/14p-synth80-features`.

Extract ResNeXt101-IBN-a **synth80** (real CityFlow + VehicleX synthetic, mAP 47.54%) tracklet features from the SAME 929 Stage-1 tracklets used by 14h v3 / 14j v4. Outputs are written as `embeddings_quaternary.npy` (929, 2048), a byte-identical `embeddings_secondary.npy` alias, `embedding_index.json`, and `dropped_indices.json` for the downstream 14q score-fusion sweep.

The model is loaded into an INLINED copy of the 09w `ResNeXt101IBNNeck` architecture (ResNeXt101-32x4d + IBN-a + GeM + BNNeck) so the checkpoint state_dict loads strictly. Crop extraction, tracklet ordering, dropped-index handling, per-crop hflip-TTA + L2, softmax-quality-weighted mean (temperature=3.0), and final L2-renorm are byte-identical to 14j v4 so the 14q drift gate (MTMC IDF1=0.77936, id_switches=154 at w_quaternary=0.0) stays aligned. Dropped set MUST equal [280, 286, 481]."""

C1 = '''import os, sys, subprocess, shutil, json, time, tarfile, re, random
from datetime import datetime
from pathlib import Path

if shutil.which("nvidia-smi"):
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=gpu_name,compute_cap", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        gpu_name, compute_cap = result.stdout.strip().split(",", 1)
        match = re.search(r"(\\d+)\\.(\\d+)", compute_cap)
        if match:
            major, minor = match.groups()
            sm = int(major) * 10 + int(minor)
            if sm < 70:
                print(f"GPU {gpu_name.strip()} sm_{sm}: installing torch 2.4.1+cu124")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "-q",
                    "torch==2.4.1+cu124", "torchvision==0.19.1+cu124",
                    "--index-url", "https://download.pytorch.org/whl/cu124",
                ])

import numpy as np
import cv2
from PIL import Image
import torch

SEED = 20260508
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"Python : {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA   : {torch.cuda.is_available()}")
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print(f"  GPU {index}: {torch.cuda.get_device_name(index)} ({props.total_memory / 1024**3:.1f} GB)")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")'''

C2 = """## 1. Clone Repo And Install Dependencies"""

C3 = '''REPO_URL = "https://github.com/MRKDaGods/gp.git"
WORK_DIR = Path("/kaggle/working")
PROJECT = WORK_DIR / "gp"

if not PROJECT.exists():
    print(f"Cloning {REPO_URL} ...")
    subprocess.check_call(["git", "clone", "--depth", "1", "-b", "feature/pretrained-ensemble", REPO_URL, str(PROJECT)])
else:
    print("Repo already present; pulling latest ...")
    subprocess.check_call(["git", "-C", str(PROJECT), "pull", "--ff-only"])

os.chdir(str(PROJECT))
sys.path.insert(0, str(PROJECT))
print(f"Repo ready at {PROJECT}")'''

C4 = '''def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

pip("filterpy", "ftfy", "lapx", "loguru", "omegaconf", "rich", "networkx>=3.1", "click", "motmetrics", "timm")
try:
    import torchreid
    print(f"torchreid ok: {getattr(torchreid, '__version__', 'unknown')}")
except ImportError:
    pip("git+https://github.com/KaiyangZhou/deep-person-reid.git")

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"], cwd=str(PROJECT))

FAILED = []
for label, module in [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("cv2", "cv2"),
    ("omegaconf", "omegaconf"),
]:
    try:
        __import__(module)
        print(f"  OK {label}")
    except ImportError as exc:
        print(f"  MISSING {label}: {exc}")
        FAILED.append(label)
if FAILED:
    raise RuntimeError(f"Missing modules: {FAILED}")
print("Dependencies installed")'''

C5 = """## 2. Resolve Inputs"""

# Cell 6 differs from 14j: synth80 checkpoint discovery from 09w-synth80 kernel output.
C6 = '''INPUT_ROOT = Path("/kaggle/input")


def find_input_dir(slug: str, owner_slug: str | None = None, hints=()) -> Path:
    direct = INPUT_ROOT / slug
    if direct.exists():
        return direct
    if owner_slug:
        owner, _, kernel = owner_slug.partition("/")
        nested = INPUT_ROOT / "notebooks" / owner / kernel
        if nested.exists():
            return nested
    lowered_slug = slug.lower()
    lowered_hints = tuple(str(hint).lower() for hint in hints)
    for path in list(INPUT_ROOT.iterdir()) if INPUT_ROOT.exists() else []:
        if not path.is_dir():
            continue
        name = path.name.lower()
        if lowered_slug in name or (lowered_hints and all(hint in name for hint in lowered_hints)):
            return path
    return direct

# synth80 ResNeXt101-IBN-a checkpoint from the 09w-synth80 kernel output.
CHECKPOINT_INPUT = find_input_dir(
    "09w-synth80",
    "yahiaakhalafallah/09w-synth80",
    hints=("09w", "synth"),
)
SYNTH80_CHECKPOINT_NAMES = [
    "resnext101ibn_synth80_best.pth",
    "resnext101ibn_synth_best.pth",
]
CHECKPOINT_PATH = None
for candidate_name in SYNTH80_CHECKPOINT_NAMES:
    candidate_path = CHECKPOINT_INPUT / candidate_name
    if candidate_path.exists():
        CHECKPOINT_PATH = candidate_path
        break
if CHECKPOINT_PATH is None:
    matches = []
    if INPUT_ROOT.exists():
        for candidate_name in SYNTH80_CHECKPOINT_NAMES:
            matches.extend(sorted(INPUT_ROOT.rglob(candidate_name)))
    if matches:
        CHECKPOINT_PATH = matches[0]
    else:
        raise FileNotFoundError(
            f"synth80 checkpoint not found under {CHECKPOINT_INPUT}; "
            f"looked for {SYNTH80_CHECKPOINT_NAMES}"
        )
print(f"synth80 checkpoint: {CHECKPOINT_PATH} ({CHECKPOINT_PATH.stat().st_size / 1024**2:.1f} MB)")

cityflow_candidates = [
    Path("/kaggle/input/data-aicity-2023-track-2"),
    Path("/kaggle/input/datasets/thanhnguyenle/data-aicity-2023-track-2"),
]
CITYFLOW_INPUT = next((path for path in cityflow_candidates if path.exists()), None)
if CITYFLOW_INPUT is None:
    raise FileNotFoundError("CityFlowV2 dataset not found; attach thanhnguyenle/data-aicity-2023-track-2")
print(f"CityFlowV2 input: {CITYFLOW_INPUT}")

SOURCE_14H_DIR = find_input_dir(
    "14h-robust-tracklet-pooling",
    "yahiaakhalafallah/14h-robust-tracklet-pooling",
    hints=("14h", "robust", "pooling"),
)
SOURCE_10A_DIR = find_input_dir(
    "mtmc-10a-stages-0-2",
    "yahiaakhalafallah/mtmc-10a-stages-0-2",
    hints=("10a", "stages", "0", "2"),
)
print(f"14h source mount: {SOURCE_14H_DIR} exists={SOURCE_14H_DIR.exists()}")
print(f"10a source mount: {SOURCE_10A_DIR} exists={SOURCE_10A_DIR.exists()}")'''

C7 = """## 3. Prepare CityFlowV2 Videos"""

# Cell 8 identical to 14j.
C8 = '''import re as regex

for mount in ["/tmp", "/kaggle/working"]:
    total, used, free = shutil.disk_usage(mount)
    print(f"{mount:16s} {free / 1024**3:.1f} GB free / {total / 1024**3:.1f} GB total")

TMP_DATA = Path("/tmp/datasets")
TMP_DATA.mkdir(parents=True, exist_ok=True)
DATA_RAW_PARENT = PROJECT / "data" / "raw"
if not DATA_RAW_PARENT.is_symlink():
    if DATA_RAW_PARENT.exists():
        shutil.rmtree(DATA_RAW_PARENT)
    DATA_RAW_PARENT.parent.mkdir(parents=True, exist_ok=True)
    DATA_RAW_PARENT.symlink_to(TMP_DATA)

DATA_RAW = TMP_DATA / "cityflowv2"
DATA_RAW.mkdir(parents=True, exist_ok=True)
for split_dir in sorted(CITYFLOW_INPUT.iterdir()):
    if not split_dir.is_dir() or split_dir.name not in ("train", "validation", "test"):
        continue
    for scene_dir in sorted(split_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        for cam_dir in sorted(scene_dir.iterdir()):
            if not cam_dir.is_dir():
                continue
            flat_name = f"{scene_dir.name}_{cam_dir.name}"
            flat_dir = DATA_RAW / flat_name
            if not flat_dir.exists():
                flat_dir.symlink_to(cam_dir)

cam_pattern = regex.compile(r"^S\\d{2}_c\\d{3}$")
cams = sorted(path.name for path in DATA_RAW.iterdir() if path.is_dir() and cam_pattern.match(path.name))
print(f"CityFlowV2 ready: {len(cams)} cameras")
for cam in cams:
    print(f"  {cam}")'''

C9 = """## 4. Load Tracklets From 14h Or 10a"""

# Cell 10 identical to 14j.
C10 = '''def find_checkpoint_tar() -> Path:
    candidates = []
    for root in [SOURCE_14H_DIR, SOURCE_10A_DIR, INPUT_ROOT]:
        if root.exists():
            candidates.extend(sorted(root.rglob("checkpoint.tar.gz")))
    for path in candidates:
        text = str(path).lower()
        if "14h" in text or "10a" in text or "mtmc" in text:
            return path
    if candidates:
        return candidates[0]

    dl_dir = Path("/tmp/kaggle_10a_download")
    dl_dir.mkdir(parents=True, exist_ok=True)
    for candidate in ["yahiaakhalafallah/mtmc-10a-stages-0-2", "gumfreddy/mtmc-10a-stages-0-2"]:
        result = subprocess.run(
            ["kaggle", "kernels", "output", candidate, "--file-pattern", r"^checkpoint\\.tar\\.gz$", "-p", str(dl_dir)],
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        print(result.stderr)
        checkpoint_path = dl_dir / "checkpoint.tar.gz"
        if checkpoint_path.exists() and checkpoint_path.stat().st_size > 0:
            print(f"10a checkpoint downloaded from {candidate}")
            return checkpoint_path
    raise FileNotFoundError("No checkpoint.tar.gz found in 14h/10a mounts or API fallback")

checkpoint_tar = find_checkpoint_tar()
EXTRACT_DIR = Path("/tmp/source_checkpoint")
if EXTRACT_DIR.exists():
    shutil.rmtree(EXTRACT_DIR)
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Extracting {checkpoint_tar} ({checkpoint_tar.stat().st_size / 1024**2:.1f} MB)")
with tarfile.open(str(checkpoint_tar), "r:gz") as tar:
    tar.extractall(str(EXTRACT_DIR))

metadata_path = EXTRACT_DIR / "run_metadata.json"
if metadata_path.exists():
    previous_meta = json.loads(metadata_path.read_text())
    PREV_RUN_NAME = previous_meta["run_name"]
    PREV_RUN_DIR = EXTRACT_DIR / PREV_RUN_NAME
else:
    stage1_dirs = sorted(EXTRACT_DIR.rglob("stage1"))
    if not stage1_dirs:
        raise FileNotFoundError(f"No stage1 directory found after extracting {checkpoint_tar}")
    PREV_RUN_DIR = stage1_dirs[0].parent
    PREV_RUN_NAME = PREV_RUN_DIR.name

print(f"Loaded source run: {PREV_RUN_NAME}")
for stage in ["stage1", "stage2"]:
    stage_dir = PREV_RUN_DIR / stage
    print(f"  {stage}: exists={stage_dir.exists()} files={len([p for p in stage_dir.rglob('*') if p.is_file()]) if stage_dir.exists() else 0}")'''

C11 = """## 5. Extract Synth80 ResNeXt101-IBN Tracklet Features"""

# Cell 12 -- the model swap. Mirrors 14j cell 12 exactly EXCEPT the extractor class.
C12 = '''from datetime import datetime

import cv2
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models.resnet import ResNet, Bottleneck

from src.core.io_utils import load_tracklets_by_camera
from src.stage2_features.embeddings import l2_normalize
from src.stage2_features.crop_extractor import CropExtractor
from src.training.datasets import build_test_transforms

EXPECTED_TRACKLETS = 929
EXPECTED_FEATURE_DIM = 2048
EXPECTED_DROPPED_INDICES = [280, 286, 481]
# synth80 (09w) was trained AND evaluated at 384x384; we extract at its native
# training resolution for best feature quality. Dropped-index alignment is
# CropExtractor-determined (which crops survive filtering) and is INDEPENDENT of
# the model input size, so the dropped set stays [280, 286, 481] as in 14j v4.
INPUT_SIZE = (384, 384)
RUN_NAME = f"run_14p_synth80_features_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RUN_DIR = Path("/kaggle/working/outputs/14p_v1_features")
STAGE1_DIR = RUN_DIR / "stage1"
FEATURE_DIR = RUN_DIR / "stage2"
PARTIAL_DIR = FEATURE_DIR / "per_camera"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)
PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
if STAGE1_DIR.exists():
    shutil.rmtree(STAGE1_DIR)
shutil.copytree(PREV_RUN_DIR / "stage1", STAGE1_DIR)

tracklets_by_camera = load_tracklets_by_camera(STAGE1_DIR)
total_tracklets = sum(len(tracklets) for tracklets in tracklets_by_camera.values())
print(f"Tracklets: {total_tracklets} across {len(tracklets_by_camera)} cameras")
for camera_id in sorted(tracklets_by_camera):
    print(f"  {camera_id}: {len(tracklets_by_camera[camera_id])}")
if total_tracklets != EXPECTED_TRACKLETS:
    raise RuntimeError(f"Expected {EXPECTED_TRACKLETS} tracklets from 14h v3 source, found {total_tracklets}")

source_index_path = PREV_RUN_DIR / "stage2" / "embedding_index.json"
if not source_index_path.exists():
    raise FileNotFoundError(f"Missing 14h v3 embedding index: {source_index_path}")
source_index_map = json.loads(source_index_path.read_text(encoding="utf-8"))
if len(source_index_map) != EXPECTED_TRACKLETS:
    raise RuntimeError(f"Expected {EXPECTED_TRACKLETS} rows in 14h v3 embedding_index.json, found {len(source_index_map)}")

row_by_tracklet = {}
for row_index, record in enumerate(source_index_map):
    key = (str(record["camera_id"]), str(record["track_id"]))
    if key in row_by_tracklet:
        raise RuntimeError(f"Duplicate source embedding_index key: {key}")
    row_by_tracklet[key] = row_index

video_paths = {}
for cam_dir in sorted(DATA_RAW.glob("S*_c*")):
    video_path = cam_dir / "vdo.avi"
    if video_path.exists():
        video_paths[cam_dir.name] = str(video_path)
missing_videos = sorted(set(tracklets_by_camera) - set(video_paths))
if missing_videos:
    raise FileNotFoundError(f"Missing videos for cameras: {missing_videos}")


def unwrap_checkpoint_state_dict(checkpoint: object) -> dict:
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model" in checkpoint:
            return checkpoint["model"]
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


# --- INLINED 09w ResNeXt101IBNNeck architecture (cell 2 of 09w_resnext101ibn_synth) ---
# ResNeXt101-32x4d topology + IBN-a on layer1/2/3 + last_stride=1 + GeM + BNNeck.
# This MUST match the checkpoint's architecture exactly so load_state_dict(strict=True)
# succeeds. src.training.model.ReIDModelResNeXt101IBN is a DIFFERENT topology
# (resnext101_32x8d) with DIFFERENT key naming, so it is NOT used here.
class IBN_a(nn.Module):
    def __init__(self, planes):
        super().__init__()
        half = planes // 2
        self.IN = nn.InstanceNorm2d(half, affine=True)
        self.BN = nn.BatchNorm2d(planes - half)

    def forward(self, x):
        split = x.shape[1] // 2
        return torch.cat([self.IN(x[:, :split]), self.BN(x[:, split:])], dim=1)


class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), 1).pow(1.0 / self.p)


class ResNeXt101IBNNeck(nn.Module):
    def __init__(self, num_classes, gem_p=3.0, feat_dim=2048):
        super().__init__()
        base = ResNet(Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=4)
        for layer in [base.layer1, base.layer2, base.layer3]:
            for block in layer:
                if hasattr(block, "bn1"):
                    block.bn1 = IBN_a(block.bn1.num_features)
        for module in base.layer4.modules():
            if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                module.stride = (1, 1)
        self.backbone = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )
        self.pool = GeM(p=gem_p)
        self.bottleneck = nn.BatchNorm1d(feat_dim)
        self.bottleneck.bias.requires_grad_(False)
        nn.init.constant_(self.bottleneck.weight, 1.0)
        nn.init.constant_(self.bottleneck.bias, 0.0)
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)
        self.feat_dim = feat_dim

    def forward_features(self, x):
        x = self.backbone(x)
        global_feat = self.pool(x).view(x.size(0), -1)
        bn_feat = self.bottleneck(global_feat)
        return global_feat, bn_feat

    def forward(self, x):
        global_feat, bn_feat = self.forward_features(x)
        if self.training:
            logits = self.classifier(bn_feat)
            return logits, global_feat, bn_feat
        return F.normalize(bn_feat, p=2, dim=1)


class Synth80FeatureExtractor:
    """Synth80 ResNeXt101-IBN-a extractor.

    Mirrors the 14j v4 R50IBNFeatureExtractor aggregation path EXACTLY:
    BICUBIC resize to INPUT_SIZE + ImageNet norm (build_test_transforms),
    original + hflip averaged, per-crop L2-normalize, softmax-quality-weighted
    mean (temperature=3.0), then final l2_normalize. Only the backbone differs.
    """

    def __init__(self, checkpoint_path: Path, device: str):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in unwrap_checkpoint_state_dict(checkpoint).items()
        }
        if "classifier.weight" not in state_dict:
            raise KeyError("Expected classifier.weight in 09w synth80 ResNeXt101-IBN checkpoint")
        num_classes = int(state_dict["classifier.weight"].shape[0])
        self.model = ResNeXt101IBNNeck(num_classes=num_classes, gem_p=3.0, feat_dim=EXPECTED_FEATURE_DIM)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"Strict checkpoint load mismatch: missing={missing}, unexpected={unexpected}")
        self.model = self.model.to(device).eval()
        self.device = device
        self.transform = build_test_transforms(height=INPUT_SIZE[0], width=INPUT_SIZE[1])
        self.feature_dim = int(self.model.feat_dim)
        self.num_classes = num_classes
        print(json.dumps({
            "checkpoint_path": str(checkpoint_path),
            "arch": "inlined_09w_ResNeXt101IBNNeck_resnext101_32x4d",
            "strict_load": True,
            "num_classes": num_classes,
            "feature_dim": self.feature_dim,
            "eval_feature": "bn_l2norm",
            "input_size": list(INPUT_SIZE),
            "preprocess": "src.training.datasets.build_test_transforms(384,384)_imagenet_bicubic",
            "flip_eval": True,
        }, indent=2))

    def _preprocess(self, crops: list[np.ndarray]) -> torch.Tensor:
        tensors = []
        for crop in crops:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(Image.fromarray(rgb)))
        return torch.stack(tensors, dim=0)

    @torch.no_grad()
    def extract_features(self, crops: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        if not crops:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        all_features = []
        for start_idx in range(0, len(crops), batch_size):
            batch = crops[start_idx:start_idx + batch_size]
            images = self._preprocess(batch).to(self.device)
            features = self.model(images)
            flipped_features = self.model(torch.flip(images, dims=[3]))
            features = (features + flipped_features) / 2.0
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            all_features.append(features.float().cpu().numpy())
        return np.concatenate(all_features, axis=0).astype(np.float32)

    def get_tracklet_embedding_from_scored_crops(
        self,
        scored_crops,
        quality_temperature: float = 3.0,
    ) -> np.ndarray | None:
        if not scored_crops:
            return None
        crop_features = self.extract_features([sc.image for sc in scored_crops])
        if crop_features.shape[0] == 0:
            return None
        qualities = np.array([sc.quality for sc in scored_crops], dtype=np.float32)
        logits = qualities * float(quality_temperature)
        logits = logits - logits.max()
        weights = np.exp(logits).astype(np.float32)
        weights = weights / max(float(weights.sum()), 1e-8)
        pooled = (crop_features * weights[:, np.newaxis]).sum(axis=0).astype(np.float32)
        return l2_normalize(pooled[np.newaxis, :])[0]


def make_drop_record(camera_id: str, tracklet, row_index: int, reason: str) -> dict:
    frames = [tf.frame_id for tf in tracklet.frames]
    areas = [float((tf.bbox[2] - tf.bbox[0]) * (tf.bbox[3] - tf.bbox[1])) for tf in tracklet.frames]
    confidences = [float(tf.confidence) for tf in tracklet.frames]
    return {
        "index": int(row_index),
        "camera_id": camera_id,
        "track_id": int(tracklet.track_id),
        "class_id": int(tracklet.class_id),
        "reason": reason,
        "frames": int(len(tracklet.frames)),
        "first_frame_id": int(min(frames)) if frames else None,
        "last_frame_id": int(max(frames)) if frames else None,
        "min_bbox_area": float(min(areas)) if areas else None,
        "max_bbox_area": float(max(areas)) if areas else None,
        "min_confidence": float(min(confidences)) if confidences else None,
        "max_confidence": float(max(confidences)) if confidences else None,
    }


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def persist_camera_outputs(
    camera_id: str,
    row_indices: list[int],
    records: list[dict],
    camera_matrix: np.ndarray,
    camera_drops: list[dict],
    elapsed_seconds: float,
) -> None:
    camera_prefix = PARTIAL_DIR / camera_id
    np.save(camera_prefix.with_name(f"{camera_id}_embeddings_quaternary.npy"), camera_matrix.astype(np.float32))
    np.save(camera_prefix.with_name(f"{camera_id}_row_indices.npy"), np.array(row_indices, dtype=np.int64))
    write_json(camera_prefix.with_name(f"{camera_id}_embedding_index.json"), records)
    write_json(camera_prefix.with_name(f"{camera_id}_dropped_tracklets.json"), camera_drops)
    status = {
        "camera_id": camera_id,
        "rows": int(len(row_indices)),
        "extracted": int(len(row_indices) - len(camera_drops)),
        "dropped": int(len(camera_drops)),
        "elapsed_minutes": round(elapsed_seconds / 60.0, 2),
        "embedding_shape": list(camera_matrix.shape),
        "row_indices_path": str(camera_prefix.with_name(f"{camera_id}_row_indices.npy")),
        "embedding_path": str(camera_prefix.with_name(f"{camera_id}_embeddings_quaternary.npy")),
        "embedding_index_path": str(camera_prefix.with_name(f"{camera_id}_embedding_index.json")),
        "dropped_tracklets_path": str(camera_prefix.with_name(f"{camera_id}_dropped_tracklets.json")),
    }
    write_json(camera_prefix.with_name(f"{camera_id}_status.json"), status)


crop_extractor = CropExtractor(
    samples_per_tracklet=48,
    min_area=32 * 32,
    min_quality=0.05,
)
synth80_reid = Synth80FeatureExtractor(CHECKPOINT_PATH, DEVICE)
if synth80_reid.feature_dim != EXPECTED_FEATURE_DIM:
    raise RuntimeError(f"Expected feature_dim={EXPECTED_FEATURE_DIM}, got {synth80_reid.feature_dim}")

embedding_matrix = np.zeros((len(source_index_map), EXPECTED_FEATURE_DIM), dtype=np.float32)
filled_mask = np.zeros((len(source_index_map),), dtype=bool)
processed_mask = np.zeros((len(source_index_map),), dtype=bool)
per_camera = {}
dropped_by_camera = {}
dropped_tracklets = []
start = time.time()

for camera_id in sorted(tracklets_by_camera):
    camera_start = time.time()
    tracklets = tracklets_by_camera[camera_id]
    video_path = video_paths[camera_id]
    camera_row_indices = []
    camera_records = []
    camera_drops = []
    camera_embeddings = 0
    print(f"Extracting {camera_id}: {len(tracklets)} tracklets")
    for tracklet in tracklets:
        key = (str(camera_id), str(tracklet.track_id))
        if key not in row_by_tracklet:
            raise RuntimeError(f"Tracklet missing from 14h v3 embedding index: {key}")
        row_index = row_by_tracklet[key]
        source_record = dict(source_index_map[row_index])
        camera_row_indices.append(row_index)
        camera_records.append(source_record)
        processed_mask[row_index] = True

        scored_crops = crop_extractor.extract_crops(tracklet, video_path)
        if not scored_crops:
            drop_record = make_drop_record(camera_id, tracklet, row_index, "no_crops_survived_filtering")
            camera_drops.append(drop_record)
            dropped_tracklets.append(drop_record)
            print(
                f"WARNING: {camera_id} tracklet {tracklet.track_id} row {row_index}: "
                "0 crops survived filtering; writing zero-vector placeholder"
            )
            continue

        embedding = synth80_reid.get_tracklet_embedding_from_scored_crops(scored_crops, quality_temperature=3.0)
        if embedding is None:
            drop_record = make_drop_record(camera_id, tracklet, row_index, "no_embedding_from_crops")
            camera_drops.append(drop_record)
            dropped_tracklets.append(drop_record)
            print(
                f"WARNING: {camera_id} tracklet {tracklet.track_id} row {row_index}: "
                "model returned no embedding; writing zero-vector placeholder"
            )
            continue

        embedding_matrix[row_index] = embedding.astype(np.float32)
        filled_mask[row_index] = True
        camera_embeddings += 1

    camera_elapsed = time.time() - camera_start
    camera_matrix = embedding_matrix[np.array(camera_row_indices, dtype=np.int64)]
    persist_camera_outputs(
        camera_id=camera_id,
        row_indices=camera_row_indices,
        records=camera_records,
        camera_matrix=camera_matrix,
        camera_drops=camera_drops,
        elapsed_seconds=camera_elapsed,
    )
    per_camera[camera_id] = camera_embeddings
    dropped_by_camera[camera_id] = len(camera_drops)
    partial_status = {
        "run_name": RUN_NAME,
        "version": "v1",
        "completed_cameras": sorted(per_camera),
        "per_camera": per_camera,
        "dropped_by_camera": dropped_by_camera,
        "filled_rows": int(filled_mask.sum()),
        "processed_rows": int(processed_mask.sum()),
        "dropped_rows": int(len(dropped_tracklets)),
        "expected_rows": int(EXPECTED_TRACKLETS),
    }
    write_json(FEATURE_DIR / "partial_status.json", partial_status)
    print(
        f"  {camera_id}: extracted={camera_embeddings} dropped={len(camera_drops)} "
        f"saved_partial={PARTIAL_DIR / (camera_id + '_embeddings_quaternary.npy')} "
        f"elapsed={camera_elapsed / 60:.1f} min"
    )

elapsed = time.time() - start
if not processed_mask.all():
    missing_rows = np.flatnonzero(~processed_mask).astype(int).tolist()
    raise RuntimeError(f"Rows present in 14h v3 embedding index were not visited: {missing_rows[:20]}")
if int(filled_mask.sum()) + len(dropped_tracklets) != EXPECTED_TRACKLETS:
    raise RuntimeError(
        f"Internal accounting mismatch: filled={int(filled_mask.sum())} "
        f"dropped={len(dropped_tracklets)} expected={EXPECTED_TRACKLETS}"
    )

embedding_matrix = l2_normalize(embedding_matrix).astype(np.float32)
if embedding_matrix.shape != (EXPECTED_TRACKLETS, EXPECTED_FEATURE_DIM):
    raise RuntimeError(f"Unexpected embedding matrix shape: {embedding_matrix.shape}")
if not np.isfinite(embedding_matrix).all():
    raise RuntimeError("synth80 embeddings contain NaN/Inf")

dropped_indices = [
    {
        "index": int(record["index"]),
        "camera_id": record["camera_id"],
        "track_id": int(record["track_id"]),
        "reason": record["reason"],
    }
    for record in dropped_tracklets
]
dropped_indices = sorted(dropped_indices, key=lambda item: item["index"])
dropped_rows = [int(item["index"]) for item in dropped_indices]
if dropped_rows:
    print(f"WARNING: zero-filled {len(dropped_rows)} dropped tracklets at rows {dropped_rows}")
else:
    print("No dropped tracklets; all rows contain synth80 embeddings")

# HARD drift-gate alignment: the dropped set MUST equal the 14j v4 set.
if dropped_rows != EXPECTED_DROPPED_INDICES:
    raise RuntimeError(
        f"Dropped-index set {dropped_rows} != expected {EXPECTED_DROPPED_INDICES}; "
        "this breaks 14q drift-gate alignment. Halting without claiming success."
    )

np.save(FEATURE_DIR / "embeddings_quaternary.npy", embedding_matrix)
np.save(FEATURE_DIR / "embeddings_secondary.npy", embedding_matrix)
write_json(FEATURE_DIR / "embedding_index.json", source_index_map)
write_json(FEATURE_DIR / "dropped_tracklets.json", sorted(dropped_tracklets, key=lambda item: item["index"]))
write_json(FEATURE_DIR / "dropped_indices.json", {
    "count": int(len(dropped_indices)),
    "fill_value": 0.0,
    "indices": dropped_rows,
    "tracklets": dropped_indices,
})

norms = np.linalg.norm(embedding_matrix, axis=1)
non_dropped_mask = np.ones((EXPECTED_TRACKLETS,), dtype=bool)
if dropped_rows:
    non_dropped_mask[np.array(dropped_rows, dtype=np.int64)] = False
non_dropped_norms = norms[non_dropped_mask]
dropped_norms = norms[~non_dropped_mask]
summary = {
    "experiment": "14p-synth80-features",
    "kernel": "yahiaakhalafallah/14p-synth80-features",
    "version": "v1",
    "run_name": RUN_NAME,
    "source_run_name": PREV_RUN_NAME,
    "checkpoint_kernel": "yahiaakhalafallah/09w-synth80",
    "checkpoint_path": str(CHECKPOINT_PATH),
    "checkpoint_strict_load": True,
    "model_arch": "inlined_09w_ResNeXt101IBNNeck_resnext101_32x4d_ibn_gem_bnneck",
    "source_14h_mount": str(SOURCE_14H_DIR),
    "source_10a_mount": str(SOURCE_10A_DIR),
    "source_embedding_index_path": str(source_index_path),
    "embedding_path": str(FEATURE_DIR / "embeddings_quaternary.npy"),
    "secondary_alias_path": str(FEATURE_DIR / "embeddings_secondary.npy"),
    "embedding_index_path": str(FEATURE_DIR / "embedding_index.json"),
    "dropped_tracklets_path": str(FEATURE_DIR / "dropped_tracklets.json"),
    "dropped_indices_path": str(FEATURE_DIR / "dropped_indices.json"),
    "per_camera_partial_dir": str(PARTIAL_DIR),
    "shape": list(embedding_matrix.shape),
    "feature_dim": int(embedding_matrix.shape[1]),
    "dtype": str(embedding_matrix.dtype),
    "per_camera": per_camera,
    "dropped_by_camera": dropped_by_camera,
    "dropped_tracklets": sorted(dropped_tracklets, key=lambda item: item["index"]),
    "dropped_indices": dropped_rows,
    "expected_dropped_indices": EXPECTED_DROPPED_INDICES,
    "norm_min": float(norms.min()),
    "norm_max": float(norms.max()),
    "norm_mean": float(norms.mean()),
    "mean": float(embedding_matrix.mean()),
    "std": float(embedding_matrix.std()),
    "non_dropped_norm_min": float(non_dropped_norms.min()) if non_dropped_norms.size else None,
    "non_dropped_norm_max": float(non_dropped_norms.max()) if non_dropped_norms.size else None,
    "dropped_norm_min": float(dropped_norms.min()) if dropped_norms.size else None,
    "dropped_norm_max": float(dropped_norms.max()) if dropped_norms.size else None,
    "has_nan": bool(np.isnan(embedding_matrix).any()),
    "has_inf": bool(np.isinf(embedding_matrix).any()),
    "aggregation": "softmax_quality_mean_temperature_3_original_plus_hflip_per_crop_l2_zero_fill_dropped",
    "samples_per_tracklet": 48,
    "crop_min_area": 32 * 32,
    "dropout_handling": "zero_vector_placeholder_preserve_14h_embedding_index_order",
    "input_size": list(INPUT_SIZE),
    "preprocess": "build_test_transforms_384_imagenet_bicubic",
    "elapsed_minutes": round(elapsed / 60.0, 2),
}
summary_path = RUN_DIR / "14p_features_summary.json"
write_json(summary_path, summary)
write_json(Path("/kaggle/working/14p_features_summary.json"), summary)

print(json.dumps(summary, indent=2))'''

C13 = """## 6. Validate Output Contract"""

# Cell 14 mirrors 14j cell 14 but asserts the dropped set EQUALS [280, 286, 481].
C14 = '''embedding_matrix = np.load(FEATURE_DIR / "embeddings_quaternary.npy")
alias_matrix = np.load(FEATURE_DIR / "embeddings_secondary.npy")
index_map = json.loads((FEATURE_DIR / "embedding_index.json").read_text(encoding="utf-8"))
dropped_payload = json.loads((FEATURE_DIR / "dropped_indices.json").read_text(encoding="utf-8"))
dropped_indices = [int(index) for index in dropped_payload["indices"]]

assert embedding_matrix.shape == (929, 2048), embedding_matrix.shape
assert alias_matrix.shape == embedding_matrix.shape, alias_matrix.shape
assert len(index_map) == 929, len(index_map)
assert index_map == source_index_map, "embedding_index.json must compare-equal to the 14h v3 source index"
assert np.isfinite(embedding_matrix).all()
assert np.array_equal(embedding_matrix, alias_matrix)
assert dropped_indices == [280, 286, 481], f"dropped set {dropped_indices} != [280, 286, 481]"

norms = np.linalg.norm(embedding_matrix, axis=1)
non_dropped_mask = np.ones((embedding_matrix.shape[0],), dtype=bool)
if dropped_indices:
    non_dropped_mask[np.array(dropped_indices, dtype=np.int64)] = False
    assert np.allclose(embedding_matrix[np.array(dropped_indices, dtype=np.int64)], 0.0)
    assert np.allclose(norms[~non_dropped_mask], 0.0)
assert float(np.max(np.abs(norms[non_dropped_mask] - 1.0))) < 1e-4, (float(norms[non_dropped_mask].min()), float(norms[non_dropped_mask].max()))

print(json.dumps({
    "validated": True,
    "shape": list(embedding_matrix.shape),
    "index_rows": len(index_map),
    "index_matches_14h": index_map == source_index_map,
    "dropped_count": len(dropped_indices),
    "dropped_indices": dropped_indices,
    "dropped_equals_expected": dropped_indices == [280, 286, 481],
    "norm_min": float(norms.min()),
    "norm_max": float(norms.max()),
    "non_dropped_norm_min": float(norms[non_dropped_mask].min()),
    "non_dropped_norm_max": float(norms[non_dropped_mask].max()),
}, indent=2))
print("ASCII RECAP")
print("  shape          : (929, 2048)")
print(f"  dropped indices: {dropped_indices}")
print(f"  index == 14h   : {index_map == source_index_map}")
print("  contract       : embeddings_quaternary.npy + embeddings_secondary.npy alias OK")'''

cells = [
    md(C0), code(C1), md(C2), code(C3), code(C4), md(C5), code(C6),
    md(C7), code(C8), md(C9), code(C10), md(C11), code(C12), md(C13), code(C14),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=True), encoding="utf-8")
print(f"Wrote {OUT}")
