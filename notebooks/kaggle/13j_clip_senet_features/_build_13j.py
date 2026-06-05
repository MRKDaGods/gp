"""Builder for 13j_clip_senet_features.ipynb.

Clones the 14p_synth80_features notebook structure and swaps ONLY the model:
load the CityFlow-retrained CLIP-SENet checkpoint
(`vehicle_clip_senet_cityflow_ft.pth`, key `model_state`) into an INLINED copy
of the 13i/13f cell-4 CLIPSENet architecture (ResNet101IBNBranch +
TinyCLIPImageBranch + AFEMBlock + CLIPSENet). `CLIPSENet.forward` in eval mode
returns `t_bn_normalized` (2048-D L2). Image size 320x320, ImageNet norm.

Crop extraction, tracklet iteration order, dropped-index handling, per-crop
hflip-TTA + L2, softmax-quality-weighted mean (temperature=3.0), and final
L2-renorm are byte-identical to 14p (which is byte-identical to 14j v4) so the
downstream 13k drift gate stays aligned. Dropped set MUST equal [280, 286, 481].

The CLIPSENet arch is INLINED verbatim from
notebooks/kaggle/13i_clip_senet_cityflow_30ep cell-4 so that a strict
load_state_dict(strict=True) of the CityFlow checkpoint succeeds.

Run:  .venv/Scripts/python.exe notebooks/kaggle/13j_clip_senet_features/_build_13j.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "13j_clip_senet_features.ipynb"


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

C0 = """# 13j CLIP-SENet CityFlow Feature Extraction

Kernel slug: `yahiaakhalafallah/13j-clip-senet-features`.

Extract per-tracklet features from the **CityFlow-retrained CLIP-SENet**
(`vehicle_clip_senet_cityflow_ft.pth`, the 30-epoch fine-tune from
`yahiaakhalafallah/13i-clip-senet-cityflow-30ep`) for the SAME 929 Stage-1
tracklets used by 14h v3 / 14j v4 / 14p. Outputs are written as
`embeddings_clipsenet.npy` (929, 2048), a byte-identical `embeddings_quaternary.npy`
alias, `embedding_index.json`, and `dropped_indices.json` for the downstream
13k score-fusion sweep.

The model is loaded into an INLINED copy of the 13i/13f CLIPSENet architecture
(ResNet101 IBN-a appearance branch + TinyCLIP semantic branch + AFEM fusion +
BNNeck) so the CityFlow checkpoint `model_state` loads strictly. `CLIPSENet.forward`
in eval mode returns `t_bn_normalized` (2048-D L2). Crop extraction, tracklet
ordering, dropped-index handling, per-crop hflip-TTA + L2, softmax-quality-weighted
mean (temperature=3.0), and final L2-renorm are byte-identical to 14p / 14j v4 so the
13k drift gate (MTMC IDF1=0.78079, id_switches=213 at w_quaternary=0.0) stays aligned.
Dropped set MUST equal [280, 286, 481].

Image size is **320x320** with ImageNet normalization (CLIP-SENet native training
resolution). The checkpoint to extract is a CFG var (`CHECKPOINT_FILENAME`) so we
can also extract `checkpoints/epoch_18.pth` / `checkpoints/epoch_24.pth`."""

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

SEED = 20260605
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
# Clone `master`: it carries the Stage-4 quaternary_embeddings score-fusion
# support that this extraction's downstream 13k fusion sweep depends on. The
# older `feature/pretrained-ensemble` branch only has secondary/tertiary streams.
REPO_BRANCH = "master"
WORK_DIR = Path("/kaggle/working")
PROJECT = WORK_DIR / "gp"

if not PROJECT.exists():
    print(f"Cloning {REPO_URL} ({REPO_BRANCH}) ...")
    subprocess.check_call(["git", "clone", "--depth", "1", "-b", REPO_BRANCH, REPO_URL, str(PROJECT)])
else:
    print("Repo already present; pulling latest ...")
    subprocess.check_call(["git", "-C", str(PROJECT), "pull", "--ff-only"])

os.chdir(str(PROJECT))
sys.path.insert(0, str(PROJECT))
print(f"Repo ready at {PROJECT}")'''

C4 = '''def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

# CLIP-SENet needs open_clip + timm (+ pretrainedmodels for the IBN backbone).
pip("filterpy", "ftfy", "lapx", "loguru", "omegaconf", "rich", "networkx>=3.1", "click", "motmetrics", "timm")
pip("open_clip_torch==2.30.0", "pretrainedmodels")
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
    ("open_clip", "open_clip"),
    ("timm", "timm"),
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

# Cell 6: CLIP-SENet CityFlow checkpoint discovery from the 13i retrain output.
C6 = '''INPUT_ROOT = Path("/kaggle/input")

# Which CLIP-SENet checkpoint to extract. Default is the final 30-epoch fine-tune;
# set to "checkpoints/epoch_18.pth" or "checkpoints/epoch_24.pth" to extract an
# intermediate snapshot. All carry the same `model_state` key and arch.
CHECKPOINT_FILENAME = "vehicle_clip_senet_cityflow_ft.pth"


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

# CityFlow-retrained CLIP-SENet checkpoint from the 13i 30-epoch fine-tune kernel.
CHECKPOINT_INPUT = find_input_dir(
    "13i-clip-senet-cityflow-30ep",
    "yahiaakhalafallah/13i-clip-senet-cityflow-30ep",
    hints=("13i", "clip", "senet", "cityflow"),
)
print(f"13i checkpoint mount: {CHECKPOINT_INPUT} exists={CHECKPOINT_INPUT.exists()}")

# Resolve CHECKPOINT_FILENAME under the 13i mount; fall back to a recursive rglob
# anywhere under /kaggle/input. The retrain is still running when this notebook is
# built, so discovery must be robust with a clear runtime error if absent.
CHECKPOINT_PATH = None
checkpoint_basename = Path(CHECKPOINT_FILENAME).name
preferred = CHECKPOINT_INPUT / CHECKPOINT_FILENAME
if preferred.exists():
    CHECKPOINT_PATH = preferred
else:
    flat = CHECKPOINT_INPUT / checkpoint_basename
    if flat.exists():
        CHECKPOINT_PATH = flat
if CHECKPOINT_PATH is None:
    matches = []
    if INPUT_ROOT.exists():
        for candidate in sorted(INPUT_ROOT.rglob(checkpoint_basename)):
            text = str(candidate).lower()
            score = 0
            if "13i" in text or "cityflow" in text:
                score += 100
            if "clip" in text and "senet" in text:
                score += 10
            matches.append((score, candidate))
    matches.sort(key=lambda item: (-item[0], str(item[1])))
    if matches:
        CHECKPOINT_PATH = matches[0][1]
        print("Checkpoint candidates (recursive fallback):")
        for score, path in matches[:10]:
            print(f"  score={score:3d}  {path}")
if CHECKPOINT_PATH is None or not CHECKPOINT_PATH.exists():
    visible = [str(p) for p in INPUT_ROOT.rglob("*.pth")][:30] if INPUT_ROOT.exists() else []
    raise FileNotFoundError(
        f"CLIP-SENet CityFlow checkpoint '{CHECKPOINT_FILENAME}' (basename "
        f"'{checkpoint_basename}') not found under {CHECKPOINT_INPUT} or anywhere "
        f"under /kaggle/input. Attach yahiaakhalafallah/13i-clip-senet-cityflow-30ep "
        f"(the retrain output) once it completes. Visible .pth: {visible}"
    )
print(f"CLIP-SENet checkpoint: {CHECKPOINT_PATH} ({CHECKPOINT_PATH.stat().st_size / 1024**2:.1f} MB)")

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

# Cell 8 identical to 14p / 14j.
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

# Cell 10 identical to 14p / 14j.
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

C11 = """## 5. Extract CLIP-SENet Tracklet Features"""

# Cell 12 -- the model swap. Mirrors 14p cell 12 aggregation path EXACTLY,
# but inlines the 13i/13f CLIPSENet architecture verbatim (cell-4) instead of
# the 09w ResNeXt101IBNNeck. Eval forward returns t_bn_normalized (2048-D L2).
C12 = '''from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import logging
import cv2
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.core.io_utils import load_tracklets_by_camera
from src.stage2_features.embeddings import l2_normalize
from src.stage2_features.crop_extractor import CropExtractor

EXPECTED_TRACKLETS = 929
EXPECTED_FEATURE_DIM = 2048
EXPECTED_DROPPED_INDICES = [280, 286, 481]
# CLIP-SENet (13f/13i) was trained AND evaluated at 320x320 with ImageNet
# normalization. We extract at its native training resolution for best feature
# quality. Dropped-index alignment is CropExtractor-determined (which crops survive
# filtering) and is INDEPENDENT of the model input size, so the dropped set stays
# [280, 286, 481] as in 14p / 14j v4.
INPUT_SIZE = (320, 320)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RUN_NAME = f"run_13j_clip_senet_features_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RUN_DIR = Path("/kaggle/working/outputs/13j_v1_features")
STAGE1_DIR = RUN_DIR / "stage1"
FEATURE_DIR = RUN_DIR / "stage2"
PARTIAL_DIR = FEATURE_DIR / "per_camera"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)
PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
if STAGE1_DIR.exists():
    shutil.rmtree(STAGE1_DIR)
shutil.copytree(PREV_RUN_DIR / "stage1", STAGE1_DIR)

logger = logging.getLogger("clipsenet")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

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


# === INLINED 13i/13f CLIPSENet architecture (cell-4 verbatim) ==============
# ResNet101 IBN-a appearance branch + TinyCLIP semantic branch + AFEM fusion +
# BNNeck. Eval forward returns t_bn_normalized (2048-D L2). This MUST match the
# checkpoint architecture exactly so load_state_dict(strict=True) succeeds.
@dataclass(frozen=True)
class LoadedBackboneInfo:
    family: str
    model_name: str
    pretrained_tag: str | None = None


class AFEMBlock(nn.Module):
    def __init__(self, in_dim: int = 2048, out_dim: int = 2048, num_groups: int = 32, residual_mode: str = "grouped_identity"):
        super().__init__()
        if out_dim % num_groups != 0:
            raise ValueError(f"AFEM out_dim={out_dim} must be divisible by num_groups={num_groups}")
        if residual_mode not in {"grouped_identity", "sum_only"}:
            raise ValueError("residual_mode must be 'grouped_identity' or 'sum_only'")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_groups = num_groups
        self.group_dim = out_dim // num_groups
        self.residual_mode = residual_mode
        self.shared = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.group_weights = nn.Parameter(torch.randn(num_groups, self.group_dim))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        linear = self.shared[0]
        nn.init.kaiming_normal_(linear.weight, mode="fan_out")
        bn = self.shared[1]
        nn.init.ones_(bn.weight)
        nn.init.zeros_(bn.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        grouped = h.view(h.shape[0], self.num_groups, self.group_dim)
        weighted = grouped * self.group_weights.unsqueeze(0)
        enhanced = weighted.reshape(h.shape[0], self.out_dim)
        if self.residual_mode == "sum_only":
            return enhanced
        return h + enhanced


class _ResNetFeatureWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


class ResNet101IBNBranch(nn.Module):
    _IBN_MODEL = "resnet101_ibn_a"
    _FALLBACK_MODEL = "resnet101"

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.output_dim = 2048
        self.backbone: nn.Module
        self.loaded_backbone: LoadedBackboneInfo
        for loader in (
            self._load_pretrainedmodels_ibn,
            self._load_torch_hub_ibn,
            self._load_timm_ibn,
            self._load_timm_plain,
        ):
            loaded = loader(pretrained=pretrained)
            if loaded is None:
                continue
            self.backbone, self.loaded_backbone = loaded
            logger.info(
                "Appearance branch loaded via '%s' model='%s' pretrained_tag='%s'",
                self.loaded_backbone.family,
                self.loaded_backbone.model_name,
                self.loaded_backbone.pretrained_tag,
            )
            return
        raise ImportError("Unable to load appearance backbone via pretrainedmodels, torch.hub, or timm")

    def _load_pretrainedmodels_ibn(self, pretrained: bool):
        try:
            import pretrainedmodels
        except ImportError:
            logger.warning("Appearance branch loader 'pretrainedmodels' is unavailable; trying torch.hub")
            return None
        constructor = getattr(pretrainedmodels, self._IBN_MODEL, None)
        if constructor is None:
            logger.warning("Appearance branch loader 'pretrainedmodels' has no '%s' entry; trying torch.hub", self._IBN_MODEL)
            return None
        pretrained_tag = "imagenet" if pretrained else None
        try:
            raw_model = constructor(pretrained=pretrained_tag)
        except Exception as exc:
            logger.warning("Appearance branch loader 'pretrainedmodels' failed for '%s': %s", self._IBN_MODEL, exc)
            return None
        if hasattr(raw_model, "last_linear"):
            raw_model.last_linear = nn.Identity()
        backbone = _ResNetFeatureWrapper(raw_model)
        return backbone, LoadedBackboneInfo(family="pretrainedmodels", model_name=self._IBN_MODEL, pretrained_tag=pretrained_tag or "random_init")

    def _load_torch_hub_ibn(self, pretrained: bool):
        try:
            raw_model = torch.hub.load("XingangPan/IBN-Net", self._IBN_MODEL, pretrained=pretrained, trust_repo=True)
        except Exception as exc:
            logger.warning("Appearance branch loader 'torch.hub' failed for '{}': {}", self._IBN_MODEL, exc)
            return None
        if hasattr(raw_model, "fc"):
            raw_model.fc = nn.Identity()
        backbone = _ResNetFeatureWrapper(raw_model)
        return backbone, LoadedBackboneInfo(family="torch.hub", model_name=self._IBN_MODEL, pretrained_tag="official_pretrained" if pretrained else "random_init")

    def _load_timm_ibn(self, pretrained: bool):
        try:
            import timm
        except ImportError as exc:
            raise ImportError("timm is required for ResNet101IBNBranch fallbacks") from exc
        available = set(timm.list_models())
        if self._IBN_MODEL not in available:
            logger.warning("Appearance branch loader 'timm' has no '%s' entry; trying plain '%s'", self._IBN_MODEL, self._FALLBACK_MODEL)
            return None
        try:
            backbone = timm.create_model(self._IBN_MODEL, pretrained=pretrained, num_classes=0, global_pool="avg")
        except Exception as exc:
            logger.warning("Appearance branch loader 'timm' failed for '%s': %s", self._IBN_MODEL, exc)
            return None
        return backbone, LoadedBackboneInfo(family="timm", model_name=self._IBN_MODEL, pretrained_tag="timm_pretrained" if pretrained else "random_init")

    def _load_timm_plain(self, pretrained: bool):
        try:
            import timm
        except ImportError as exc:
            raise ImportError("timm is required for ResNet101IBNBranch fallbacks") from exc
        try:
            backbone = timm.create_model(self._FALLBACK_MODEL, pretrained=pretrained, num_classes=0, global_pool="avg")
        except Exception as exc:
            logger.warning("Appearance branch loader 'timm' failed for plain '%s': %s", self._FALLBACK_MODEL, exc)
            return None
        logger.warning("Appearance branch fell back to plain '%s' because no IBN-a loader succeeded", self._FALLBACK_MODEL)
        return backbone, LoadedBackboneInfo(family="timm", model_name=self._FALLBACK_MODEL, pretrained_tag="timm_pretrained" if pretrained else "random_init")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        if out.ndim != 2:
            raise RuntimeError(f"Appearance branch expected pooled 2D output, got shape {tuple(out.shape)}")
        return out


class TinyCLIPImageBranch(nn.Module):
    _OPEN_CLIP_CHAIN = (
        {"model_name": "hf-hub:wkcn/TinyCLIP-ViT-45M-32-Text-21M-LAION400M", "pretrained_tag": None},
        {"model_name": "TinyCLIP-ViT-40M-32-Text-19M", "pretrained_tag": "laion400m_e32"},
    )
    _TIMM_TINYCLIP_CHAIN = ("vit_medium_patch32_clip_224.tinyclip_laion400m",)
    _LAST_RESORT_OPEN_CLIP = ("ViT-B-32", "openai")

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.provider = ""
        self.model = None
        self.loaded_backbone: LoadedBackboneInfo | None = None
        last_error = self._try_load_open_clip(pretrained=pretrained)
        if self.model is None:
            last_error = self._try_load_timm_tinyclip(pretrained=pretrained) or last_error
        if self.model is None:
            last_error = self._try_load_open_clip_last_resort(pretrained=pretrained) or last_error
        if self.model is None or self.loaded_backbone is None:
            raise RuntimeError("Unable to load any TinyCLIP/OpenCLIP visual backbone") from last_error
        self.image_size = self._infer_image_size(self.model)

    def _try_load_open_clip(self, pretrained: bool):
        try:
            import open_clip
        except ImportError as exc:
            return exc
        last_error = None
        for candidate in self._OPEN_CLIP_CHAIN:
            model_name = candidate["model_name"]
            pretrained_tag = candidate["pretrained_tag"]
            try:
                if pretrained_tag is None:
                    model, _, _ = open_clip.create_model_and_transforms(model_name)
                else:
                    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_tag if pretrained else None)
            except Exception as exc:
                last_error = exc
                logger.warning("TinyCLIP load failed for model='%s' pretrained='%s': %s", model_name, pretrained_tag or "hf-hub-default", exc)
                continue
            self.model = model
            self.provider = "open_clip"
            self.loaded_backbone = LoadedBackboneInfo(family="semantic", model_name=model_name, pretrained_tag=pretrained_tag if pretrained else "random_init")
            self.output_dim = self._infer_open_clip_output_dim(model)
            logger.info("TinyCLIP branch loaded model='%s' via open_clip output_dim=%s", model_name, self.output_dim)
            return None
        return last_error

    def _try_load_timm_tinyclip(self, pretrained: bool):
        try:
            import timm
        except ImportError as exc:
            return exc
        last_error = None
        for model_name in self._TIMM_TINYCLIP_CHAIN:
            try:
                model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
            except Exception as exc:
                last_error = exc
                logger.warning("TinyCLIP-equivalent timm load failed for model='%s': %s", model_name, exc)
                continue
            self.model = model
            self.provider = "timm"
            self.loaded_backbone = LoadedBackboneInfo(family="semantic", model_name=model_name, pretrained_tag="timm_pretrained" if pretrained else "random_init")
            self.output_dim = self._infer_timm_output_dim(model)
            logger.info("TinyCLIP branch loaded model='%s' via timm output_dim=%s", model_name, self.output_dim)
            return None
        return last_error

    def _try_load_open_clip_last_resort(self, pretrained: bool):
        try:
            import open_clip
        except ImportError as exc:
            return exc
        model_name, pretrained_tag = self._LAST_RESORT_OPEN_CLIP
        try:
            model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_tag if pretrained else None)
        except Exception as exc:
            logger.warning("OpenCLIP last resort load failed for model='%s' pretrained='%s': %s", model_name, pretrained_tag, exc)
            return exc
        self.model = model
        self.provider = "open_clip"
        self.loaded_backbone = LoadedBackboneInfo(family="semantic", model_name=model_name, pretrained_tag=pretrained_tag if pretrained else "random_init")
        self.output_dim = self._infer_open_clip_output_dim(model)
        logger.info("TinyCLIP branch loaded model='%s' via open_clip output_dim=%s", model_name, self.output_dim)
        return None

    @staticmethod
    def _infer_open_clip_output_dim(model: nn.Module) -> int:
        visual = getattr(model, "visual", None)
        output_dim = getattr(visual, "output_dim", None)
        if isinstance(output_dim, int):
            return output_dim
        visual_proj = getattr(model, "visual_projection", None)
        if isinstance(visual_proj, torch.Tensor) and visual_proj.ndim == 2:
            return int(visual_proj.shape[-1])
        visual_proj = getattr(visual, "proj", None)
        if isinstance(visual_proj, torch.Tensor):
            if visual_proj.ndim == 1:
                return int(visual_proj.shape[0])
            if visual_proj.ndim == 2:
                return int(visual_proj.shape[-1])
        raise RuntimeError("Could not infer TinyCLIP visual output dimension")

    @staticmethod
    def _infer_timm_output_dim(model: nn.Module) -> int:
        output_dim = getattr(model, "num_features", None)
        if isinstance(output_dim, int):
            return output_dim
        raise RuntimeError("Could not infer timm TinyCLIP visual output dimension")

    @staticmethod
    def _infer_image_size(model: nn.Module):
        pretrained_cfg = getattr(model, "pretrained_cfg", None)
        if isinstance(pretrained_cfg, dict):
            input_size = pretrained_cfg.get("input_size")
            if isinstance(input_size, tuple) and len(input_size) == 3:
                return (int(input_size[-2]), int(input_size[-1]))
        visual = getattr(model, "visual", None)
        image_size = getattr(visual, "image_size", None)
        if isinstance(image_size, int):
            return (image_size, image_size)
        if isinstance(image_size, tuple) and len(image_size) == 2:
            return (int(image_size[0]), int(image_size[1]))
        return (224, 224)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[-2:]) != self.image_size:
            x = F.interpolate(x, size=self.image_size, mode="bilinear", align_corners=False)
        if self.provider == "open_clip":
            features = self.model.encode_image(x, normalize=False)
        else:
            features = self.model(x)
        if features.ndim != 2:
            raise RuntimeError(f"TinyCLIP branch expected 2D image features, got shape {tuple(features.shape)}")
        return features


class CLIPSENet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 2048,
        afem_groups: int = 32,
        feat_dim_appearance: int = 2048,
        feat_dim_semantic: int = 512,
        dropout: float = 0.0,
        appearance_pretrained: bool = True,
        semantic_pretrained: bool = True,
        residual_mode: str = "grouped_identity",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.appearance_branch = ResNet101IBNBranch(pretrained=appearance_pretrained)
        self.semantic_branch = TinyCLIPImageBranch(pretrained=semantic_pretrained)
        detected_app_dim = self.appearance_branch.output_dim
        detected_sem_dim = self.semantic_branch.output_dim
        if feat_dim_appearance != detected_app_dim:
            logger.warning("Requested feat_dim_appearance=%s but backbone reports %s. Using detected dim.", feat_dim_appearance, detected_app_dim)
        if feat_dim_semantic != detected_sem_dim:
            logger.warning("Requested feat_dim_semantic=%s but backbone reports %s. Using detected dim.", feat_dim_semantic, detected_sem_dim)
        self.feat_dim_appearance = detected_app_dim
        self.feat_dim_semantic = detected_sem_dim
        self.fusion_fc = nn.Linear(self.feat_dim_appearance + self.feat_dim_semantic, embed_dim, bias=False)
        self.afem = AFEMBlock(in_dim=embed_dim, out_dim=embed_dim, num_groups=afem_groups, residual_mode=residual_mode)
        self.bnneck = nn.BatchNorm1d(embed_dim)
        self.bnneck.bias.requires_grad_(False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(embed_dim, num_classes, bias=False)
        nn.init.kaiming_normal_(self.fusion_fc.weight, mode="fan_out")
        nn.init.normal_(self.classifier.weight, std=0.001)
        self.loaded_resnext_model = self.appearance_branch.loaded_backbone.model_name
        self.loaded_tinyclip_model = self.semantic_branch.loaded_backbone.model_name

    def forward(self, x: torch.Tensor):
        f_app = self.appearance_branch(x)
        f_sem = self.semantic_branch(x)
        t_u = self.fusion_fc(torch.cat([f_app, f_sem], dim=1))
        t_s_prime = self.afem(t_u)
        t = t_u + t_s_prime
        t_bn = self.bnneck(t)
        t_bn_normalized = F.normalize(t_bn, p=2, dim=1)
        if self.training:
            logits = self.classifier(self.dropout(t_bn))
            return t_bn_normalized, logits
        return t_bn_normalized
# === end inlined CLIPSENet ==================================================


def unwrap_checkpoint_state_dict(checkpoint: object) -> dict:
    # 13i/13f save the model under the `model_state` key (see cell 7). Support the
    # common alternatives too in case an intermediate snapshot differs.
    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            return checkpoint["model_state"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            return checkpoint["model"]
        if checkpoint and all(hasattr(value, "shape") for value in checkpoint.values()):
            return checkpoint
    raise TypeError(f"Unsupported checkpoint type/format: {type(checkpoint)!r}")


class CLIPSENetFeatureExtractor:
    """CityFlow-retrained CLIP-SENet extractor.

    Mirrors the 14p / 14j v4 aggregation path EXACTLY: BICUBIC resize to
    INPUT_SIZE + ImageNet norm, original + hflip averaged, per-crop L2-normalize,
    softmax-quality-weighted mean (temperature=3.0), then final l2_normalize.
    Only the backbone differs. Eval forward returns t_bn_normalized (2048-D L2).
    """

    def __init__(self, checkpoint_path: Path, device: str):
        import torchvision.transforms as T

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in unwrap_checkpoint_state_dict(checkpoint).items()
        }
        if "classifier.weight" not in state_dict:
            raise KeyError("Expected classifier.weight in CLIP-SENet CityFlow checkpoint")
        num_classes = int(state_dict["classifier.weight"].shape[0])
        # Build with semantic_pretrained / appearance_pretrained True so the module
        # topology matches the checkpoint; the loaded weights then overwrite them.
        self.model = CLIPSENet(num_classes=num_classes, embed_dim=EXPECTED_FEATURE_DIM, afem_groups=32)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"Strict checkpoint load mismatch: missing={list(missing)}, unexpected={list(unexpected)}")
        self.model = self.model.to(device).eval()
        self.device = device
        self.transform = T.Compose([
            T.Resize(INPUT_SIZE, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.feature_dim = EXPECTED_FEATURE_DIM
        self.num_classes = num_classes
        # Dummy forward to assert eval output is 2048-D L2-normalized.
        with torch.no_grad():
            dummy = torch.zeros(2, 3, INPUT_SIZE[0], INPUT_SIZE[1], device=device)
            out = self.model(dummy)
        if out.ndim != 2 or out.shape[1] != EXPECTED_FEATURE_DIM:
            raise RuntimeError(f"Eval forward expected (N, {EXPECTED_FEATURE_DIM}), got {tuple(out.shape)}")
        dummy_norms = out.norm(p=2, dim=1)
        # Zero input -> bnneck shifts then F.normalize; norm must be ~1 for any
        # non-degenerate output. Assert finite + unit-norm on a real-ish probe.
        probe = torch.randn(2, 3, INPUT_SIZE[0], INPUT_SIZE[1], device=device)
        with torch.no_grad():
            probe_out = self.model(probe)
        probe_norms = probe_out.norm(p=2, dim=1)
        if not torch.isfinite(probe_out).all():
            raise RuntimeError("CLIP-SENet eval output contains NaN/Inf on probe input")
        if float((probe_norms - 1.0).abs().max()) > 1e-3:
            raise RuntimeError(f"CLIP-SENet eval output is not L2-normalized: norms={probe_norms.tolist()}")
        print(json.dumps({
            "checkpoint_path": str(checkpoint_path),
            "arch": "inlined_13i_CLIPSENet_resnet101ibn_a_tinyclip_afem_bnneck",
            "strict_load": True,
            "num_classes": num_classes,
            "feature_dim": self.feature_dim,
            "eval_feature": "t_bn_normalized_l2",
            "input_size": list(INPUT_SIZE),
            "preprocess": "Resize(320,320)_BICUBIC_imagenet_norm",
            "flip_eval": True,
            "loaded_appearance_model": self.model.loaded_resnext_model,
            "loaded_semantic_model": self.model.loaded_tinyclip_model,
            "zero_input_norms": dummy_norms.tolist(),
            "probe_norms": probe_norms.tolist(),
        }, indent=2))

    def _preprocess(self, crops: list) -> torch.Tensor:
        tensors = []
        for crop in crops:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(Image.fromarray(rgb)))
        return torch.stack(tensors, dim=0)

    @torch.no_grad()
    def extract_features(self, crops: list, batch_size: int = 32) -> np.ndarray:
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

    def get_tracklet_embedding_from_scored_crops(self, scored_crops, quality_temperature: float = 3.0):
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


def persist_camera_outputs(camera_id, row_indices, records, camera_matrix, camera_drops, elapsed_seconds) -> None:
    camera_prefix = PARTIAL_DIR / camera_id
    np.save(camera_prefix.with_name(f"{camera_id}_embeddings_clipsenet.npy"), camera_matrix.astype(np.float32))
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
        "embedding_path": str(camera_prefix.with_name(f"{camera_id}_embeddings_clipsenet.npy")),
    }
    write_json(camera_prefix.with_name(f"{camera_id}_status.json"), status)


crop_extractor = CropExtractor(samples_per_tracklet=48, min_area=32 * 32, min_quality=0.05)
clipsenet_reid = CLIPSENetFeatureExtractor(CHECKPOINT_PATH, DEVICE)
if clipsenet_reid.feature_dim != EXPECTED_FEATURE_DIM:
    raise RuntimeError(f"Expected feature_dim={EXPECTED_FEATURE_DIM}, got {clipsenet_reid.feature_dim}")

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
            print(f"WARNING: {camera_id} tracklet {tracklet.track_id} row {row_index}: 0 crops survived filtering; writing zero-vector placeholder")
            continue

        embedding = clipsenet_reid.get_tracklet_embedding_from_scored_crops(scored_crops, quality_temperature=3.0)
        if embedding is None:
            drop_record = make_drop_record(camera_id, tracklet, row_index, "no_embedding_from_crops")
            camera_drops.append(drop_record)
            dropped_tracklets.append(drop_record)
            print(f"WARNING: {camera_id} tracklet {tracklet.track_id} row {row_index}: model returned no embedding; writing zero-vector placeholder")
            continue

        embedding_matrix[row_index] = embedding.astype(np.float32)
        filled_mask[row_index] = True
        camera_embeddings += 1

    camera_elapsed = time.time() - camera_start
    camera_matrix = embedding_matrix[np.array(camera_row_indices, dtype=np.int64)]
    persist_camera_outputs(camera_id, camera_row_indices, camera_records, camera_matrix, camera_drops, camera_elapsed)
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
    print(f"  {camera_id}: extracted={camera_embeddings} dropped={len(camera_drops)} elapsed={camera_elapsed / 60:.1f} min")

elapsed = time.time() - start
if not processed_mask.all():
    missing_rows = np.flatnonzero(~processed_mask).astype(int).tolist()
    raise RuntimeError(f"Rows present in 14h v3 embedding index were not visited: {missing_rows[:20]}")
if int(filled_mask.sum()) + len(dropped_tracklets) != EXPECTED_TRACKLETS:
    raise RuntimeError(f"Internal accounting mismatch: filled={int(filled_mask.sum())} dropped={len(dropped_tracklets)} expected={EXPECTED_TRACKLETS}")

embedding_matrix = l2_normalize(embedding_matrix).astype(np.float32)
if embedding_matrix.shape != (EXPECTED_TRACKLETS, EXPECTED_FEATURE_DIM):
    raise RuntimeError(f"Unexpected embedding matrix shape: {embedding_matrix.shape}")
if not np.isfinite(embedding_matrix).all():
    raise RuntimeError("CLIP-SENet embeddings contain NaN/Inf")

dropped_indices = [
    {"index": int(record["index"]), "camera_id": record["camera_id"], "track_id": int(record["track_id"]), "reason": record["reason"]}
    for record in dropped_tracklets
]
dropped_indices = sorted(dropped_indices, key=lambda item: item["index"])
dropped_rows = [int(item["index"]) for item in dropped_indices]
if dropped_rows:
    print(f"WARNING: zero-filled {len(dropped_rows)} dropped tracklets at rows {dropped_rows}")
else:
    print("No dropped tracklets; all rows contain CLIP-SENet embeddings")

# HARD drift-gate alignment: the dropped set MUST equal the 14j v4 / 14p set.
if dropped_rows != EXPECTED_DROPPED_INDICES:
    raise RuntimeError(
        f"Dropped-index set {dropped_rows} != expected {EXPECTED_DROPPED_INDICES}; "
        "this breaks 13k drift-gate alignment. Halting without claiming success."
    )

np.save(FEATURE_DIR / "embeddings_clipsenet.npy", embedding_matrix)
np.save(FEATURE_DIR / "embeddings_quaternary.npy", embedding_matrix)
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
    "experiment": "13j-clip-senet-features",
    "kernel": "yahiaakhalafallah/13j-clip-senet-features",
    "version": "v1",
    "run_name": RUN_NAME,
    "source_run_name": PREV_RUN_NAME,
    "checkpoint_kernel": "yahiaakhalafallah/13i-clip-senet-cityflow-30ep",
    "checkpoint_filename": CHECKPOINT_FILENAME,
    "checkpoint_path": str(CHECKPOINT_PATH),
    "checkpoint_strict_load": True,
    "model_arch": "inlined_13i_CLIPSENet_resnet101ibn_a_tinyclip_afem_bnneck",
    "source_14h_mount": str(SOURCE_14H_DIR),
    "source_10a_mount": str(SOURCE_10A_DIR),
    "source_embedding_index_path": str(source_index_path),
    "embedding_path": str(FEATURE_DIR / "embeddings_clipsenet.npy"),
    "quaternary_alias_path": str(FEATURE_DIR / "embeddings_quaternary.npy"),
    "embedding_index_path": str(FEATURE_DIR / "embedding_index.json"),
    "dropped_indices_path": str(FEATURE_DIR / "dropped_indices.json"),
    "shape": list(embedding_matrix.shape),
    "feature_dim": int(embedding_matrix.shape[1]),
    "dtype": str(embedding_matrix.dtype),
    "per_camera": per_camera,
    "dropped_by_camera": dropped_by_camera,
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
    "preprocess": "resize_320_bicubic_imagenet_norm",
    "eval_feature": "t_bn_normalized_l2",
    "elapsed_minutes": round(elapsed / 60.0, 2),
}
summary_path = RUN_DIR / "13j_features_summary.json"
write_json(summary_path, summary)
write_json(Path("/kaggle/working/13j_features_summary.json"), summary)

print(json.dumps(summary, indent=2))'''

C13 = """## 6. Validate Output Contract"""

# Cell 14 mirrors 14p cell 14 but for the clipsenet/quaternary alias pair.
C14 = '''embedding_matrix = np.load(FEATURE_DIR / "embeddings_clipsenet.npy")
alias_matrix = np.load(FEATURE_DIR / "embeddings_quaternary.npy")
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
print("  contract       : embeddings_clipsenet.npy + embeddings_quaternary.npy alias OK")'''

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
