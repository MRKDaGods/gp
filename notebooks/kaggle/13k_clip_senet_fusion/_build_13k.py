"""Builder for 13k_clip_senet_fusion.ipynb.

Clones the 14k_r50ibn_fusion_extended CPU Stage-3/4/5 sweep, but anchors at the
14k K7 4-way operating point and ADDS the CityFlow-retrained CLIP-SENet (from
13j) as a NEW *quaternary* score-fusion stream while keeping the full K7 base
FIXED:
  - PRIMARY    = TransReID CLIP (PCA 384D)         -- weight derived, renormalized
  - SECONDARY  = FastReID R50-IBN (2048D, 14j)     -- FIXED at K7 weight 0.45
  - TERTIARY   = DINOv2 ViT-L/14 (2048D, 14h)      -- weight renormalized
  - QUATERNARY = CLIP-SENet CityFlow (2048D, 13j)  -- the SWEPT stream

Physical slot mapping note: in 14k the R50-IBN "quaternary" stream was plumbed
through the Stage-4 *secondary_embeddings* slot (DINOv2 stayed in tertiary). This
notebook preserves that exact wiring (R50-IBN -> secondary @ 0.45, DINOv2 ->
tertiary) and uses the still-FREE *quaternary_embeddings* slot for CLIP-SENet, so
Stage 4 blends all four appearance streams simultaneously
(src/stage4_association/pipeline.py Step 3b, ~L491-514).

K0 drift gate: quaternary weight 0.0 -> (w_p=0.10, w_sec=0.45, w_tert=0.45,
thr=0.46, aqe_k=2, fic=0.5) MUST reproduce MTMC IDF1 0.78079 and id_switches=213
(the 14k K7 plateau). Fail loud otherwise.

Renormalization: R50-IBN secondary stays FIXED at 0.45; the quaternary weight is
drawn proportionally from the primary+tertiary pool (base sum 0.55) so all
weights stay >= 0 and sum to 1.0.

Run:  .venv/Scripts/python.exe notebooks/kaggle/13k_clip_senet_fusion/_build_13k.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "13k_clip_senet_fusion.ipynb"


def md(text: str) -> dict:
    lines = text.split("\n")
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(text: str) -> dict:
    lines = text.split("\n")
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


C0 = """# 13k -- CLIP-SENet Quaternary Fusion Sweep

CPU-only Stage 3-5 sweep that adds the CityFlow-retrained **CLIP-SENet** (from
`yahiaakhalafallah/13j-clip-senet-features`) as a NEW quaternary score-fusion
stream on top of the FIXED 14k K7 4-way operating point.

Fixed K7 base: PRIMARY = TransReID CLIP (PCA 384D), SECONDARY = R50-IBN (2048D,
14j) @ 0.45, TERTIARY = DINOv2 (2048D, 14h) @ 0.45, `thr=0.46`, AQE `k=2`, FIC
`reg=0.5`. The new QUATERNARY = CLIP-SENet (2048D, 13j) is swept.

**K0 drift gate**: quaternary weight `0.0` must reproduce MTMC IDF1 `0.78079`
and `id_switches=213` (the 14k K7 plateau). Fail loud otherwise.

Sweep: `w_clipsenet (quaternary) in {0.0, 0.05, 0.10, 0.15, 0.20, 0.30}` x
`thr in {0.45, 0.46, 0.47}`. R50-IBN secondary stays FIXED at 0.45; the
quaternary weight is drawn proportionally from the primary+tertiary pool (base
sum 0.55) so weights stay valid. Verdict bands: WIN >= 0.7820, MARGINAL >= 0.7810."""

C1 = '''import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np

WORK_DIR = Path("/kaggle/working")
PROJECT = WORK_DIR / "gp"
INPUT_ROOT = Path("/kaggle/input")
DATA_OUT = Path("/tmp/pipeline_outputs")
DATA_OUT.mkdir(parents=True, exist_ok=True)

print(f"Python: {sys.version.split()[0]}")
print(f"Kaggle input exists: {INPUT_ROOT.exists()}")'''

C2 = """## 1. Clone Repo And Install CPU Dependencies"""

C3 = '''REPO_URL = "https://github.com/MRKDaGods/gp.git"
# Clone `master`: it carries the Stage-4 quaternary_embeddings score-fusion
# support this sweep depends on (the new CLIP-SENet stream goes in the quaternary
# slot). The older `feature/pretrained-ensemble` branch has ONLY secondary/tertiary
# streams and would silently drop the CLIP-SENet stream, breaking the K0 gate.
REPO_BRANCH = "master"

if not PROJECT.exists():
    print(f"Cloning {REPO_URL} ({REPO_BRANCH}) ...")
    subprocess.check_call(["git", "clone", "--depth", "1", "-b", REPO_BRANCH, REPO_URL, str(PROJECT)])
else:
    print("Repo already present; pulling latest ...")
    subprocess.check_call(["git", "-C", str(PROJECT), "pull", "--ff-only"])

os.chdir(str(PROJECT))
sys.path.insert(0, str(PROJECT))

# Fail loud if the cloned branch lacks quaternary score-fusion support, rather
# than silently dropping the CLIP-SENet stream and reporting a bogus K0 reproduction.
PIPELINE_SRC = (PROJECT / "src" / "stage4_association" / "pipeline.py").read_text(encoding="utf-8")
if "quaternary_embeddings" not in PIPELINE_SRC or "quat_weight" not in PIPELINE_SRC:
    raise RuntimeError(
        f"Cloned branch '{REPO_BRANCH}' src/stage4_association/pipeline.py lacks "
        "quaternary_embeddings / quat_weight support; the CLIP-SENet quaternary "
        "stream cannot be wired. Use a branch with Stage-4 4-way fusion."
    )
print("Quaternary score-fusion support confirmed in cloned pipeline.py")


def pip(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])


pip("numpy", "scipy", "pandas", "faiss-cpu", "motmetrics", "omegaconf", "rich", "networkx>=3.1", "click", "loguru", "scikit-learn")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"], cwd=str(PROJECT))
print(f"Repo ready at {PROJECT}")'''

C4 = """## 2. Locate 14h Anchor, 14j R50-IBN Secondary, 13j CLIP-SENet Quaternary, And Ground Truth"""

# Cell 5: resolve all four sources. 14h = stage1 + primary + tertiary (DINOv2) +
# hsv. 14j = R50-IBN secondary stream. 13j = CLIP-SENet quaternary stream.
C5 = '''SOURCE_14H_OWNER_SLUG = "yahiaakhalafallah/14h-robust-tracklet-pooling"
SOURCE_14J_OWNER_SLUG = "yahiaakhalafallah/14j-r50-ibn-features"
SOURCE_13J_OWNER_SLUG = "yahiaakhalafallah/13j-clip-senet-features"
SOURCE_14H_SLUG = SOURCE_14H_OWNER_SLUG.split("/", 1)[1]
SOURCE_14J_SLUG = SOURCE_14J_OWNER_SLUG.split("/", 1)[1]
SOURCE_13J_SLUG = SOURCE_13J_OWNER_SLUG.split("/", 1)[1]
EXPECTED_CAMS = ["S01_c001", "S01_c002", "S01_c003", "S02_c006", "S02_c007", "S02_c008"]
EXPECTED_TRACKLETS = 929
EXPECTED_DROPPED_INDICES = [280, 286, 481]


def find_input_dir(slug: str, owner_slug: str, hints=()) -> Path:
    direct = INPUT_ROOT / slug
    if direct.exists():
        return direct

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


def find_14h_checkpoint() -> Path:
    source_dir = find_input_dir(SOURCE_14H_SLUG, SOURCE_14H_OWNER_SLUG, hints=("14h", "robust", "tracklet"))
    checkpoint_path = source_dir / "checkpoint.tar.gz"
    if checkpoint_path.exists():
        print(f"14h input: {source_dir}")
        return checkpoint_path

    visible = [str(path) for path in INPUT_ROOT.rglob("checkpoint.tar.gz")] if INPUT_ROOT.exists() else []
    raise FileNotFoundError(
        f"14h checkpoint.tar.gz not found for {SOURCE_14H_OWNER_SLUG}. "
        f"Visible checkpoints under /kaggle/input: {visible[:20]}"
    )


checkpoint = find_14h_checkpoint()
EXTRACT_DIR = Path("/tmp/14h_checkpoint")
if EXTRACT_DIR.exists():
    shutil.rmtree(EXTRACT_DIR)
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Extracting {checkpoint} ({checkpoint.stat().st_size / 1024**2:.1f} MB)")
with tarfile.open(str(checkpoint), "r:gz") as archive:
    archive.extractall(str(EXTRACT_DIR))

with open(EXTRACT_DIR / "run_metadata.json", encoding="utf-8") as handle:
    previous_meta = json.load(handle)
SOURCE_14H_RUN_NAME = previous_meta["run_name"]
SOURCE_14H_RUN_DIR = EXTRACT_DIR / SOURCE_14H_RUN_NAME
SOURCE_STAGE1_DIR = SOURCE_14H_RUN_DIR / "stage1"
SOURCE_STAGE2_DIR = SOURCE_14H_RUN_DIR / "stage2"
for required_path in [
    SOURCE_STAGE1_DIR,
    SOURCE_STAGE2_DIR / "embeddings.npy",
    SOURCE_STAGE2_DIR / "embeddings_tertiary.npy",
    SOURCE_STAGE2_DIR / "hsv_features.npy",
    SOURCE_STAGE2_DIR / "embedding_index.json",
]:
    if not required_path.exists():
        raise FileNotFoundError(required_path)
print(f"Loaded 14h run: {SOURCE_14H_RUN_NAME}")


def find_stream_stage2_dir(slug, owner_slug, hints, npy_name) -> Path:
    source_dir = find_input_dir(slug, owner_slug, hints=hints)
    explicit_candidates = [
        source_dir / "outputs" / "14j_v4_features" / "stage2",
        source_dir / "outputs" / "13j_v1_features" / "stage2",
        source_dir / "14j_v4_features" / "stage2",
        source_dir / "13j_v1_features" / "stage2",
        source_dir / "stage2",
    ]
    for candidate in explicit_candidates:
        if (candidate / npy_name).exists():
            print(f"{slug} {npy_name} input: {candidate}")
            return candidate

    matches = sorted(INPUT_ROOT.rglob(npy_name)) if INPUT_ROOT.exists() else []
    for match in matches:
        text = str(match).lower()
        if all(hint in text for hint in hints):
            print(f"{slug} {npy_name} input discovered: {match.parent}")
            return match.parent
    if matches:
        print(f"{slug} {npy_name} input fallback: {matches[0].parent}")
        return matches[0].parent

    visible = [str(path) for path in INPUT_ROOT.rglob("*.npy")] if INPUT_ROOT.exists() else []
    raise FileNotFoundError(
        f"{npy_name} not found for {owner_slug}. Visible npy examples: {visible[:30]}"
    )


# R50-IBN secondary stream (14j writes both embeddings_quaternary.npy and a
# byte-identical embeddings_secondary.npy alias; we use the secondary alias).
SOURCE_R50_STAGE2_DIR = find_stream_stage2_dir(
    SOURCE_14J_SLUG, SOURCE_14J_OWNER_SLUG, ("14j",), "embeddings_secondary.npy",
)
# CLIP-SENet quaternary stream (13j writes embeddings_clipsenet.npy + a
# byte-identical embeddings_quaternary.npy alias; we use the clipsenet name).
SOURCE_CLIPSENET_STAGE2_DIR = find_stream_stage2_dir(
    SOURCE_13J_SLUG, SOURCE_13J_OWNER_SLUG, ("13j",), "embeddings_clipsenet.npy",
)


def is_cityflow_gt_root(path: Path) -> bool:
    return path.exists() and all((path / cam / "gt" / "gt.txt").exists() for cam in EXPECTED_CAMS)


def find_cityflow_gt_root() -> Path:
    candidates = [
        PROJECT / "data" / "raw" / "cityflowv2",
        EXTRACT_DIR / "gt_annotations",
        Path("/kaggle/input/data-aicity-2023-track-2"),
        Path("/kaggle/input/datasets/thanhnguyenle/data-aicity-2023-track-2"),
    ]
    for candidate in candidates:
        if is_cityflow_gt_root(candidate):
            return candidate

    for gt_file in INPUT_ROOT.rglob("gt.txt") if INPUT_ROOT.exists() else []:
        if gt_file.parent.name != "gt" or gt_file.parent.parent.name not in EXPECTED_CAMS:
            continue
        candidate = gt_file.parents[2]
        if is_cityflow_gt_root(candidate):
            return candidate

    visible = [str(path) for path in INPUT_ROOT.rglob("gt.txt")] if INPUT_ROOT.exists() else []
    raise FileNotFoundError(
        "CityFlowV2 ground-truth annotations not found in normalized camera layout. "
        f"Expected all {EXPECTED_CAMS} under <root>/<cam>/gt/gt.txt. "
        f"Visible gt.txt examples: {visible[:20]}"
    )


GT_DIR = find_cityflow_gt_root()
print(f"Ground truth root: {GT_DIR}")
for cam in EXPECTED_CAMS:
    gt_file = GT_DIR / cam / "gt" / "gt.txt"
    print(f"  {cam}: {gt_file} rows={sum(1 for line in gt_file.open() if line.strip())}")'''

C6 = """## 3. Validate And Normalize Secondary (R50-IBN) And Quaternary (CLIP-SENet) Streams"""

# Cell 7: normalize BOTH the fixed R50-IBN secondary stream and the swept
# CLIP-SENet quaternary stream against the 14h index, with zero-fill drift gate.
C7 = '''STATIC_CONTRACT_K0_DISABLE_PATH = "K0_DISABLES_QUATERNARY_BY_EMPTY_PATH_AND_ZERO_WEIGHT"
STATIC_CONTRACT_STREAM_RENORM = "STREAM_L2_RENORMALIZATION_AFTER_DROPPED_ZERO_HANDLING"

source_index = json.loads((SOURCE_STAGE2_DIR / "embedding_index.json").read_text(encoding="utf-8"))
if len(source_index) != EXPECTED_TRACKLETS:
    raise RuntimeError(f"Expected {EXPECTED_TRACKLETS} embedding rows, found {len(source_index)}")


def load_and_normalize_stream(stage2_dir: Path, npy_name: str, label: str) -> tuple[np.ndarray, list]:
    stream_index = json.loads((stage2_dir / "embedding_index.json").read_text(encoding="utf-8"))
    if stream_index != source_index:
        raise RuntimeError(f"{label} embedding_index.json does not exactly match 14h v3 source ordering")
    raw = np.load(stage2_dir / npy_name).astype(np.float32)
    if raw.shape != (EXPECTED_TRACKLETS, 2048):
        raise RuntimeError(f"Unexpected {label} shape: {raw.shape}")
    if not np.isfinite(raw).all():
        raise RuntimeError(f"{label} embeddings contain NaN/Inf")

    dropped_payload_path = stage2_dir / "dropped_indices.json"
    if dropped_payload_path.exists():
        dropped_payload = json.loads(dropped_payload_path.read_text(encoding="utf-8"))
        dropped_indices = [int(index) for index in dropped_payload.get("indices", [])]
    else:
        norms_for_drop = np.linalg.norm(raw, axis=1)
        dropped_indices = [int(index) for index in np.flatnonzero(norms_for_drop < 1e-8)]
    if dropped_indices != EXPECTED_DROPPED_INDICES:
        raise RuntimeError(f"Unexpected {label} dropped indices: {dropped_indices}; expected {EXPECTED_DROPPED_INDICES}")

    stream = raw.copy()
    if dropped_indices:
        stream[np.array(dropped_indices, dtype=np.int64)] = 0.0
    norms = np.linalg.norm(stream, axis=1, keepdims=True)
    nonzero_mask = norms[:, 0] > 1e-8
    stream[nonzero_mask] = stream[nonzero_mask] / norms[nonzero_mask]
    stream[~nonzero_mask] = 0.0
    stream = stream.astype(np.float32)

    final_norms = np.linalg.norm(stream, axis=1)
    if not np.allclose(stream[np.array(dropped_indices, dtype=np.int64)], 0.0):
        raise RuntimeError(f"Dropped {label} rows are not zero after zero-fill handling")
    if float(np.max(np.abs(final_norms[nonzero_mask] - 1.0))) > 1e-4:
        raise RuntimeError(f"Non-dropped {label} rows are not unit norm after L2 renormalization")
    return stream, dropped_indices


SECONDARY_WORK_DIR = Path("/tmp/r50ibn_secondary_stage2")
SECONDARY_WORK_DIR.mkdir(parents=True, exist_ok=True)
SECONDARY_EMBEDDING_PATH = SECONDARY_WORK_DIR / "embeddings_secondary_l2.npy"
r50_secondary, r50_dropped = load_and_normalize_stream(
    SOURCE_R50_STAGE2_DIR, "embeddings_secondary.npy", "R50-IBN secondary",
)
np.save(SECONDARY_EMBEDDING_PATH, r50_secondary)

QUATERNARY_WORK_DIR = Path("/tmp/clipsenet_quaternary_stage2")
QUATERNARY_WORK_DIR.mkdir(parents=True, exist_ok=True)
QUATERNARY_EMBEDDING_PATH = QUATERNARY_WORK_DIR / "embeddings_quaternary_l2.npy"
clipsenet_quaternary, clipsenet_dropped = load_and_normalize_stream(
    SOURCE_CLIPSENET_STAGE2_DIR, "embeddings_clipsenet.npy", "CLIP-SENet quaternary",
)
np.save(QUATERNARY_EMBEDDING_PATH, clipsenet_quaternary)

print(json.dumps({
    "contract_k0_disable_path": STATIC_CONTRACT_K0_DISABLE_PATH,
    "contract_stream_renorm": STATIC_CONTRACT_STREAM_RENORM,
    "secondary_r50_path": str(SECONDARY_EMBEDDING_PATH),
    "secondary_shape": list(r50_secondary.shape),
    "secondary_dropped": r50_dropped,
    "quaternary_clipsenet_path": str(QUATERNARY_EMBEDDING_PATH),
    "quaternary_shape": list(clipsenet_quaternary.shape),
    "quaternary_dropped": clipsenet_dropped,
}, indent=2))'''

C8 = """## 4. Define K7-Anchored CLIP-SENet Quaternary Sweep"""

# Cell 9: define the sweep. K0 = quaternary weight 0.0 reproduces K7 (0.78079/213).
# R50-IBN secondary FIXED at 0.45; quaternary weight drawn proportionally from
# the primary+tertiary pool (base sum 0.55).
C9 = '''from src.core.config import load_config, save_config
from src.core.data_models import TrackletFeatures
from src.core.io_utils import load_tracklets_by_camera
from src.core.logging_utils import setup_logging
from src.stage3_indexing import run_stage3
from src.stage4_association import run_stage4
from src.stage5_evaluation import run_stage5

RUN_NAME = f"run_13k_clip_senet_fusion_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RUN_DIR = DATA_OUT / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)
setup_logging(level="INFO", log_file=RUN_DIR / "pipeline.log")
print(f"Run: {RUN_NAME}")

# 14k K7 4-way operating point (the plateau base we anchor on):
#   w_primary=0.10, w_secondary(R50-IBN)=0.45, w_tertiary(DINOv2)=0.45,
#   thr=0.46, AQE k=2, FIC reg=0.5  ->  MTMC IDF1 0.78079, id_switches 213.
BASE_PRIMARY_WEIGHT = 0.10
SECONDARY_WEIGHT = 0.45            # R50-IBN, FIXED across the whole sweep
BASE_TERTIARY_WEIGHT = 0.45        # DINOv2, renormalized as quaternary grows
PRIMARY_TERTIARY_POOL = round(BASE_PRIMARY_WEIGHT + BASE_TERTIARY_WEIGHT, 6)  # 0.55

QUATERNARY_WEIGHTS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
SIMILARITY_THRESHOLDS = [0.45, 0.46, 0.47]
ANCHOR_CONFIG = {"aqe_k": 2, "fic_regularisation": 0.5}


def renormalized_weights(w_quaternary: float) -> tuple[float, float, float]:
    """R50-IBN secondary stays 0.45; draw w_quaternary from the primary+tertiary
    pool proportionally so all weights stay non-negative and sum to 1.0."""
    if w_quaternary < 0.0 or w_quaternary > PRIMARY_TERTIARY_POOL + 1e-9:
        raise ValueError(f"w_quaternary={w_quaternary} outside [0, {PRIMARY_TERTIARY_POOL}]")
    scale = (PRIMARY_TERTIARY_POOL - w_quaternary) / PRIMARY_TERTIARY_POOL
    w_primary = round(BASE_PRIMARY_WEIGHT * scale, 6)
    w_tertiary = round(BASE_TERTIARY_WEIGHT * scale, 6)
    return w_primary, w_tertiary, SECONDARY_WEIGHT


SWEEP_CONFIGS = []
for w_quaternary in QUATERNARY_WEIGHTS:
    w_primary, w_tertiary, w_secondary = renormalized_weights(w_quaternary)
    for similarity_threshold in SIMILARITY_THRESHOLDS:
        is_k0 = (abs(w_quaternary) < 1e-9 and abs(similarity_threshold - 0.46) < 1e-9)
        SWEEP_CONFIGS.append({
            "config_id": "K0" if is_k0 else f"C{len(SWEEP_CONFIGS)}",
            "block": "drift" if is_k0 else "clipsenet_grid",
            "w_quaternary": w_quaternary,
            "w_primary": w_primary,
            "w_secondary": w_secondary,
            "w_tertiary": w_tertiary,
            "similarity_threshold": similarity_threshold,
            "notes": (
                "K0 drift gate (14k K7 plateau): K0_DISABLES_QUATERNARY_BY_EMPTY_PATH_AND_ZERO_WEIGHT"
                if is_k0 else
                f"CLIP-SENet quaternary weight {w_quaternary:.2f}; R50-IBN secondary FIXED 0.45; "
                f"renorm w_p={w_primary:.4f}, w_t={w_tertiary:.4f}"
            ),
            **ANCHOR_CONFIG,
        })

# Move the K0 drift gate to the FRONT so it runs first as the gate.
SWEEP_CONFIGS.sort(key=lambda cfg: (cfg["config_id"] != "K0", cfg["config_id"]))

if not any(cfg["config_id"] == "K0" for cfg in SWEEP_CONFIGS):
    raise RuntimeError("K0 drift gate config (w_quaternary=0.0, thr=0.46) was not generated")
if len(SWEEP_CONFIGS) != len(QUATERNARY_WEIGHTS) * len(SIMILARITY_THRESHOLDS):
    raise RuntimeError(f"Expected {len(QUATERNARY_WEIGHTS) * len(SIMILARITY_THRESHOLDS)} configs, got {len(SWEEP_CONFIGS)}")
for config in SWEEP_CONFIGS:
    total_weight = (
        float(config["w_primary"]) + float(config["w_secondary"])
        + float(config["w_tertiary"]) + float(config["w_quaternary"])
    )
    if abs(total_weight - 1.0) > 1e-6:
        raise RuntimeError(f"Weights do not sum to 1.0 for {config['config_id']}: {total_weight}")
    if float(config["w_primary"]) < -1e-9 or float(config["w_tertiary"]) < -1e-9:
        raise RuntimeError(f"Negative renormalized weight for {config['config_id']}: {config}")
    if abs(float(config["w_secondary"]) - 0.45) > 1e-9:
        raise RuntimeError(f"R50-IBN secondary weight drifted from 0.45 for {config['config_id']}")

# K0 drift-gate targets = 14k K7 plateau (MTMC IDF1 0.78079, id_switches 213).
K0_REPRO_TARGET = 0.78079
K0_REPRO_TOL = 0.001
K0_ID_SWITCH_TARGET = 213
WIN_THRESHOLD = 0.7820
MARGINAL_MIN = 0.7810
NEUTRAL_MIN = 0.7800
SOLVER = "cc"
ALGORITHM = "conflict_free_cc"
LOUVAIN_RES = 0.70
APPEARANCE_WEIGHT = 0.70
HSV_WEIGHT = 0.0
ST_WEIGHT = round(1.0 - APPEARANCE_WEIGHT - HSV_WEIGHT, 4)
BRIDGE_PRUNE = 0.0
MAX_COMP_SIZE = 12
GALLERY_THRESH = 0.48
ORPHAN_MATCH_THRESH = 0.38
INTRA_MERGE = True
INTRA_MERGE_THRESH = 0.80
INTRA_MERGE_GAP = 30
MULTI_QUERY_WEIGHT = 0.00
MTMC_ONLY = False

print(json.dumps({"config_count": len(SWEEP_CONFIGS), "configs": SWEEP_CONFIGS}, indent=2))'''

C10 = """## 5. Run Stage 3-5 Per Config"""

# Cell 11: Stage 3-5 loop. K7 base is wired as primary + secondary(R50 @ 0.45) +
# tertiary(DINOv2). The swept stream is the CLIP-SENet quaternary. K0 (quaternary
# weight 0.0, empty quaternary path) MUST reproduce 0.78079 / 213.
C11 = '''def load_metrics(report_path: Path) -> dict:
    if not report_path.exists():
        return {}
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    details = payload.get("details", {}) or {}
    error_analysis = details.get("error_analysis", {}) or {}
    return {
        "mtmc_idf1": payload.get("mtmc_idf1", details.get("mtmc_idf1", payload.get("idf1"))),
        "trackeval_idf1": payload.get("idf1"),
        "idp": payload.get("idp", details.get("idp")),
        "idr": payload.get("idr", details.get("idr")),
        "mota": payload.get("mota"),
        "hota": payload.get("hota"),
        "id_switches": payload.get("id_switches"),
        "conflations": error_analysis.get("conflated_pred"),
        "fragmentations": error_analysis.get("fragmented_gt"),
        "num_pred_ids": payload.get("num_pred_ids", error_analysis.get("total_pred")),
    }


def summarize_prediction_dir(pred_dir: Path) -> dict:
    files = sorted(pred_dir.glob("*.txt")) if pred_dir.exists() else []
    rows_by_camera = {}
    ids_by_camera = {}
    for pred_file in files:
        rows = [line.strip().split(",") for line in pred_file.open() if line.strip()]
        rows_by_camera[pred_file.stem] = len(rows)
        ids_by_camera[pred_file.stem] = len({row[1] for row in rows if len(row) > 1})
    return {
        "exists": pred_dir.exists(),
        "file_count": len(files),
        "rows_by_camera": rows_by_camera,
        "ids_by_camera": ids_by_camera,
        "total_rows": int(sum(rows_by_camera.values())),
        "total_ids_camera_sum": int(sum(ids_by_camera.values())),
    }


def copy_recovery_artifacts(config_dir: Path, config_id: str) -> Path:
    recovery_dir = Path("/kaggle/working/outputs/13k_recovery") / config_id
    if recovery_dir.exists():
        shutil.rmtree(recovery_dir)
    recovery_dir.mkdir(parents=True, exist_ok=True)
    for rel_path in [
        Path("config.yaml"),
        Path("stage4/global_trajectories.json"),
        Path("stage4/forensic_report.json"),
        Path("stage5/evaluation_report.json"),
    ]:
        source_path = config_dir / rel_path
        if source_path.exists():
            target_path = recovery_dir / rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
    pred_dir = config_dir / "stage5" / "predictions_mot"
    if pred_dir.exists():
        shutil.copytree(pred_dir, recovery_dir / "stage5" / "predictions_mot")
    return recovery_dir


def build_features(stage2_dir: Path) -> tuple[list[TrackletFeatures], dict]:
    index_map = json.loads((stage2_dir / "embedding_index.json").read_text(encoding="utf-8"))
    embeddings = np.load(stage2_dir / "embeddings.npy").astype(np.float32)
    hsv_features = np.load(stage2_dir / "hsv_features.npy").astype(np.float32)
    if embeddings.shape[0] != len(index_map) or hsv_features.shape[0] != len(index_map):
        raise ValueError(
            f"Stage2 row mismatch: embeddings={embeddings.shape}, hsv={hsv_features.shape}, index={len(index_map)}"
        )
    features = [
        TrackletFeatures(
            track_id=int(row["track_id"]),
            camera_id=str(row["camera_id"]),
            class_id=int(row["class_id"]),
            embedding=embeddings[row_index],
            hsv_histogram=hsv_features[row_index],
            raw_embedding=None,
            multi_query_embeddings=None,
        )
        for row_index, row in enumerate(index_map)
    ]
    return features, {
        "num_features": len(features),
        "primary_shape": list(embeddings.shape),
        "hsv_shape": list(hsv_features.shape),
    }


def build_overrides(config: dict, config_run_name: str, stage2_dir: Path) -> list[str]:
    w_quaternary = float(config["w_quaternary"])
    w_secondary = float(config["w_secondary"])
    w_tertiary = float(config["w_tertiary"])
    sim_thresh = float(config["similarity_threshold"])
    aqe_k = int(config["aqe_k"])
    fic_reg = float(config["fic_regularisation"])
    # R50-IBN secondary stream is ALWAYS active at the fixed K7 weight.
    secondary_path = str(SECONDARY_EMBEDDING_PATH)
    # CLIP-SENet quaternary: empty path + zero weight at K0 (drift gate).
    if w_quaternary == 0.0:
        quaternary_path = ""
        quaternary_weight = 0.0
    else:
        quaternary_path = str(QUATERNARY_EMBEDDING_PATH)
        quaternary_weight = w_quaternary
    return [
        f"project.run_name={config_run_name}",
        f"project.output_dir={DATA_OUT}",
        "stage0.cameras=[S01_c001,S01_c002,S01_c003,S02_c006,S02_c007,S02_c008]",
        f"stage4.association.query_expansion.k={aqe_k}",
        "stage4.association.query_expansion.alpha=5.0",
        "stage4.association.query_expansion.dba=false",
        f"stage4.association.graph.similarity_threshold={sim_thresh}",
        f"stage4.association.solver={SOLVER}",
        f"stage4.association.graph.algorithm={ALGORITHM}",
        f"stage4.association.graph.louvain_resolution={LOUVAIN_RES}",
        f"stage4.association.graph.bridge_prune_margin={BRIDGE_PRUNE}",
        f"stage4.association.graph.max_component_size={MAX_COMP_SIZE}",
        f"stage4.association.weights.vehicle.appearance={APPEARANCE_WEIGHT}",
        f"stage4.association.weights.vehicle.hsv={HSV_WEIGHT}",
        f"stage4.association.weights.vehicle.spatiotemporal={ST_WEIGHT}",
        "stage4.association.mutual_nn.top_k_per_query=20",
        "stage4.association.fic.enabled=true",
        f"stage4.association.fic.regularisation={fic_reg}",
        "stage4.association.reranking.enabled=false",
        "stage4.association.camera_pair_norm.enabled=false",
        "stage4.association.fac.enabled=false",
        f"stage4.association.multi_query.enabled={str(MULTI_QUERY_WEIGHT > 0.0).lower()}",
        f"stage4.association.multi_query.weight={MULTI_QUERY_WEIGHT}",
        f"stage4.association.multi_query.dir={stage2_dir}",
        "stage4.association.aflink.enabled=false",
        f"stage4.association.secondary_embeddings.path={secondary_path}",
        f"stage4.association.secondary_embeddings.weight={w_secondary}",
        f"stage4.association.tertiary_embeddings.path={stage2_dir / 'embeddings_tertiary.npy'}",
        f"stage4.association.tertiary_embeddings.weight={w_tertiary}",
        f"stage4.association.quaternary_embeddings.path={quaternary_path}",
        f"stage4.association.quaternary_embeddings.weight={quaternary_weight}",
        "stage4.association.camera_bias.enabled=false",
        "stage4.association.zone_model.enabled=false",
        "stage4.association.hierarchical.enabled=false",
        f"stage4.association.intra_camera_merge.enabled={str(INTRA_MERGE).lower()}",
        f"stage4.association.intra_camera_merge.threshold={INTRA_MERGE_THRESH}",
        f"stage4.association.intra_camera_merge.max_time_gap={INTRA_MERGE_GAP}",
        "stage4.association.gallery_expansion.enabled=true",
        f"stage4.association.gallery_expansion.threshold={GALLERY_THRESH}",
        f"stage4.association.gallery_expansion.orphan_match_threshold={ORPHAN_MATCH_THRESH}",
        "stage4.association.weights.length_weight_power=0.3",
        "stage4.association.temporal_overlap.enabled=true",
        "stage4.association.temporal_overlap.bonus=0.05",
        "stage4.association.temporal_overlap.max_mean_time=5.0",
        f"stage5.mtmc_only_submission={str(MTMC_ONLY).lower()}",
        "stage5.stationary_filter.enabled=true",
        "stage5.stationary_filter.min_displacement_px=150",
        "stage5.stationary_filter.max_mean_velocity_px=2.0",
        "stage5.min_submission_confidence=0.15",
        "stage5.cross_id_nms_iou=0.40",
        "stage5.min_trajectory_confidence=0.30",
        "stage5.min_trajectory_frames=40",
        "stage5.track_edge_trim.enabled=false",
        "stage5.track_smoothing.enabled=false",
        "stage5.gt_frame_clip=true",
        "stage5.gt_zone_filter=true",
        f"stage5.ground_truth_dir={GT_DIR}",
    ]


def run_config(config: dict) -> dict:
    config_id = config["config_id"]
    config_dir = RUN_DIR / config_id
    config_dir.mkdir(parents=True, exist_ok=True)
    tracklets_by_camera = load_tracklets_by_camera(SOURCE_STAGE1_DIR)
    features, feature_summary = build_features(SOURCE_STAGE2_DIR)
    print("\\n" + "=" * 80)
    print(
        f"Running {config_id}: w_primary={config['w_primary']:.4f}, "
        f"w_secondary={config['w_secondary']:.3f}, w_tertiary={config['w_tertiary']:.4f}, "
        f"w_quaternary={config['w_quaternary']:.3f}, sim_thresh={config['similarity_threshold']:.2f}"
    )
    print("=" * 80)

    config_run_name = f"{RUN_NAME}_{config_id}"
    cfg = load_config(
        "configs/default.yaml",
        dataset_config="configs/datasets/cityflowv2.yaml",
        overrides=build_overrides(config, config_run_name, SOURCE_STAGE2_DIR),
    )
    save_config(cfg, config_dir / "config.yaml")

    start = time.time()
    faiss_index, metadata_store = run_stage3(cfg, features, tracklets_by_camera, output_dir=config_dir / "stage3")
    stage3_min = (time.time() - start) / 60.0
    print(f"{config_id} Stage 3 complete in {stage3_min:.2f} min")

    start = time.time()
    trajectories = run_stage4(cfg, faiss_index, metadata_store, features, tracklets_by_camera, output_dir=config_dir / "stage4")
    stage4_min = (time.time() - start) / 60.0
    print(f"{config_id} Stage 4 complete in {stage4_min:.2f} min: {len(trajectories)} global trajectories")

    start = time.time()
    run_stage5(cfg, trajectories, output_dir=config_dir / "stage5")
    stage5_min = (time.time() - start) / 60.0
    print(f"{config_id} Stage 5 complete in {stage5_min:.2f} min")

    report_path = config_dir / "stage5" / "evaluation_report.json"
    metrics = load_metrics(report_path)
    prediction_summary = summarize_prediction_dir(config_dir / "stage5" / "predictions_mot")
    recovery_dir = copy_recovery_artifacts(config_dir, config_id)
    idf1_value = metrics.get("mtmc_idf1")
    if idf1_value is None:
        idf1_value = metrics.get("trackeval_idf1")
    if idf1_value is None:
        raise RuntimeError(f"IDF1 not found in {report_path}")
    if not prediction_summary["exists"] or prediction_summary["file_count"] == 0:
        raise RuntimeError(f"No MOT prediction files were written for {config_id}: {prediction_summary}")
    if idf1_value == 0.0 and prediction_summary["total_rows"] == 0:
        raise RuntimeError(f"Zero IDF1 with zero prediction rows for {config_id}: {prediction_summary}")

    row = {
        "config_id": config_id,
        "block": config["block"],
        "w_primary": float(config["w_primary"]),
        "w_secondary": float(config["w_secondary"]),
        "w_tertiary": float(config["w_tertiary"]),
        "w_quaternary": float(config["w_quaternary"]),
        "similarity_threshold": float(config["similarity_threshold"]),
        "aqe_k": int(config["aqe_k"]),
        "fic_regularisation": float(config["fic_regularisation"]),
        "notes": config["notes"],
        "mtmc_idf1": metrics.get("mtmc_idf1"),
        "idp": metrics.get("idp"),
        "idr": metrics.get("idr"),
        "id_switches": metrics.get("id_switches"),
        "fragmentations": metrics.get("fragmentations"),
        "mota": metrics.get("mota"),
        "trackeval_idf1": metrics.get("trackeval_idf1"),
        "hota": metrics.get("hota"),
        "conflations": metrics.get("conflations"),
        "num_pred_ids": metrics.get("num_pred_ids"),
        "num_trajectories": len(trajectories),
        "num_stage4_tracklets": int(sum(len(trajectory.tracklets) for trajectory in trajectories)),
        "prediction_summary": prediction_summary,
        "feature_summary": feature_summary,
        "stage_timings_min": {
            "stage3": round(stage3_min, 2),
            "stage4": round(stage4_min, 2),
            "stage5": round(stage5_min, 2),
        },
        "paths": {
            "config_dir": str(config_dir),
            "evaluation_report": str(report_path),
            "recovery_dir": str(recovery_dir),
        },
    }
    print(f"{config_id} MTMC IDF1: {idf1_value:.5f} id_switches={row['id_switches']}")
    return row


results = []
sweep_results = []
halt_reason = None
drift_detected = False
wall_start = time.time()

drift_config = next(config for config in SWEEP_CONFIGS if config["config_id"] == "K0")
drift_check_result = run_config(drift_config)
results.append(drift_check_result)
(RUN_DIR / "13k_partial_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

k0_idf1 = float(drift_check_result["mtmc_idf1"])
k0_id_switches = drift_check_result.get("id_switches")
if abs(k0_idf1 - K0_REPRO_TARGET) > K0_REPRO_TOL or k0_id_switches != K0_ID_SWITCH_TARGET:
    drift_detected = True
    halt_reason = (
        f"K0 drift gate failed: got idf1={k0_idf1:.5f}, id_switches={k0_id_switches}; "
        f"expected idf1 within {K0_REPRO_TARGET:.5f} +/- {K0_REPRO_TOL:.5f} "
        f"and id_switches={K0_ID_SWITCH_TARGET} (the 14k K7 plateau)"
    )
    print(halt_reason)
else:
    print(f"K0 drift gate passed: {k0_idf1:.5f}, id_switches={k0_id_switches}")
    for config in SWEEP_CONFIGS:
        if config["config_id"] == "K0":
            continue
        row = run_config(config)
        results.append(row)
        sweep_results.append(row)
        (RUN_DIR / "13k_partial_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

overall_best = max(results, key=lambda row: row["mtmc_idf1"] if row["mtmc_idf1"] is not None else -1.0)
best_quaternary = max(sweep_results, key=lambda row: row["mtmc_idf1"] if row["mtmc_idf1"] is not None else -1.0) if sweep_results else None

best_idf1 = float(overall_best["mtmc_idf1"])
if drift_detected:
    verdict_band = "DRIFT_FAIL"
elif best_idf1 >= WIN_THRESHOLD:
    verdict_band = "WIN"
elif best_idf1 >= MARGINAL_MIN:
    verdict_band = "MARGINAL"
elif best_idf1 >= NEUTRAL_MIN:
    verdict_band = "NEUTRAL"
else:
    verdict_band = "DEAD"

print("\\n" + "#" * 80)
print(
    f"BEST 13k CONFIG: {overall_best['config_id']} w_p={overall_best['w_primary']:.4f} "
    f"w_sec={overall_best['w_secondary']:.3f} w_t={overall_best['w_tertiary']:.4f} "
    f"w_q={overall_best['w_quaternary']:.3f} thr={overall_best['similarity_threshold']:.2f} "
    f"MTMC_IDF1={overall_best['mtmc_idf1']:.5f} id_switches={overall_best.get('id_switches')} "
    f"verdict_band={verdict_band} drift={drift_detected}"
)
if halt_reason:
    print(f"HALT: {halt_reason}")
print("#" * 80)'''

C12 = """## 6. Persist Summary"""

# Cell 13: persist the full summary dict.
C13 = '''summary_dir = Path("/kaggle/working/outputs/13k_summary")
summary_dir.mkdir(parents=True, exist_ok=True)
legacy_summary_path = summary_dir / "13k_summary.json"
required_summary_path = Path("/kaggle/working/outputs/13k_clip_senet_fusion_summary.json")
root_summary_path = Path("/kaggle/working/13k_clip_senet_fusion_summary.json")

compact_configs = [
    {
        "config_id": row.get("config_id"),
        "w_primary": row.get("w_primary"),
        "w_secondary": row.get("w_secondary"),
        "w_tertiary": row.get("w_tertiary"),
        "w_quaternary": row.get("w_quaternary"),
        "similarity_threshold": row.get("similarity_threshold"),
        "mtmc_idf1": row.get("mtmc_idf1"),
        "idp": row.get("idp"),
        "idr": row.get("idr"),
        "id_switches": row.get("id_switches"),
        "fragmentations": row.get("fragmentations"),
        "mota": row.get("mota"),
    }
    for row in results
]

top5 = sorted(
    [row for row in results if row.get("mtmc_idf1") is not None],
    key=lambda row: row["mtmc_idf1"],
    reverse=True,
)[:5]

summary = {
    "run_name": RUN_NAME,
    "source_14h_run_name": SOURCE_14H_RUN_NAME,
    "experiment": "13k-clip-senet-quaternary-fusion",
    "kernel": "yahiaakhalafallah/13k-clip-senet-fusion",
    "frame_id_convention": "0-based internal Stage 1/2 frame IDs; MOT output is converted to 1-based in Stage 5",
    "verdict_band": verdict_band,
    "drift_detected": drift_detected,
    "drift_check_config_id": "K0",
    "drift_check_result": drift_check_result,
    "drift_reproduction_target": K0_REPRO_TARGET,
    "drift_reproduction_tolerance": K0_REPRO_TOL,
    "drift_id_switch_target": K0_ID_SWITCH_TARGET,
    "configs": compact_configs,
    "planned_configs": SWEEP_CONFIGS,
    "results": results,
    "sweep_results": sweep_results,
    "overall_best": overall_best,
    "best": overall_best,
    "best_quaternary": best_quaternary,
    "top5_by_idf1": top5,
    "halt_reason": halt_reason,
    "total_config_count": len(SWEEP_CONFIGS),
    "executed_config_count": len(results),
    "feature_sources": {
        "primary_and_tertiary_kernel": SOURCE_14H_OWNER_SLUG,
        "secondary_r50_kernel": SOURCE_14J_OWNER_SLUG,
        "quaternary_clipsenet_kernel": SOURCE_13J_OWNER_SLUG,
        "checkpoint": str(checkpoint),
        "stage1_tracklets": str(SOURCE_STAGE1_DIR),
        "stage2_features": str(SOURCE_STAGE2_DIR),
        "secondary_stage2": str(SOURCE_R50_STAGE2_DIR),
        "quaternary_stage2": str(SOURCE_CLIPSENET_STAGE2_DIR),
        "secondary_normalized_path": str(SECONDARY_EMBEDDING_PATH),
        "quaternary_normalized_path": str(QUATERNARY_EMBEDDING_PATH),
    },
    "fixed_config": {
        "pca_components": 384,
        "algorithm": ALGORITHM,
        "aqe_k": 2,
        "fic_regularisation": 0.5,
        "base_primary_weight": BASE_PRIMARY_WEIGHT,
        "secondary_weight_fixed": SECONDARY_WEIGHT,
        "base_tertiary_weight": BASE_TERTIARY_WEIGHT,
        "primary_tertiary_pool": PRIMARY_TERTIARY_POOL,
        "gallery_expansion": True,
        "gallery_threshold": GALLERY_THRESH,
        "orphan_match_threshold": ORPHAN_MATCH_THRESH,
        "temporal_overlap_bonus": 0.05,
        "intra_merge": INTRA_MERGE,
        "intra_merge_threshold": INTRA_MERGE_THRESH,
        "intra_merge_gap": INTRA_MERGE_GAP,
        "mtmc_only_submission": MTMC_ONLY,
        "score_fusion_math": "cosine_sim = w_p*sim_primary + w_sec*sim_secondary(R50) + w_t*sim_tertiary(DINOv2) + w_q*sim_quaternary(CLIPSENet); dropped zero rows produce that stream's sim=0.0; K0 uses empty quaternary path and weight 0.0",
        "renormalization": "R50-IBN secondary fixed 0.45; w_quaternary drawn proportionally from the primary+tertiary pool (base sum 0.55)",
        "k0_contract": STATIC_CONTRACT_K0_DISABLE_PATH,
        "stream_renorm_contract": STATIC_CONTRACT_STREAM_RENORM,
    },
    "sweep_grid": {
        "w_quaternary": QUATERNARY_WEIGHTS,
        "similarity_threshold": SIMILARITY_THRESHOLDS,
        "grid_size": len(SWEEP_CONFIGS),
    },
    "stop_criteria": {
        "win_threshold": WIN_THRESHOLD,
        "marginal_range": [MARGINAL_MIN, WIN_THRESHOLD],
        "neutral_range": [NEUTRAL_MIN, MARGINAL_MIN],
        "dead_below": NEUTRAL_MIN,
        "drift_fail_range": [K0_REPRO_TARGET - K0_REPRO_TOL, K0_REPRO_TARGET + K0_REPRO_TOL],
        "drift_condition": f"abs(K0 - {K0_REPRO_TARGET}) > {K0_REPRO_TOL} or id_switches != {K0_ID_SWITCH_TARGET}",
    },
    "stage_timings_min": {
        "stage345_sweep": round((time.time() - wall_start) / 60.0, 2),
    },
    "paths": {
        "run_dir": str(RUN_DIR),
        "summary": str(required_summary_path),
        "legacy_summary": str(legacy_summary_path),
        "root_summary": str(root_summary_path),
    },
}

summary_payload = json.dumps(summary, indent=2)
required_summary_path.write_text(summary_payload, encoding="utf-8")
legacy_summary_path.write_text(summary_payload, encoding="utf-8")
root_summary_path.write_text(summary_payload, encoding="utf-8")
print(f"Saved summary: {required_summary_path}")
print(json.dumps(summary, indent=2))'''

cells = [
    md(C0), code(C1), md(C2), code(C3), md(C4), code(C5), md(C6), code(C7),
    md(C8), code(C9), md(C10), code(C11), md(C12), code(C13),
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
