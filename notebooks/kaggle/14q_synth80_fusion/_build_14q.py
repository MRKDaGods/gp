"""Builder for 14q_synth80_fusion.ipynb.

Clones the 14k R50-IBN fusion-extended sweep and re-points the quaternary
score-fusion stream at the 14p synth80 features. Primary + tertiary come from
the 14h v3 Stage-2 outputs (unchanged). CPU-only Stages 3-5.

Drift gate K0 (w_quaternary=0.0, empty secondary path) MUST reproduce
MTMC IDF1=0.77936 and id_switches=154. If it drifts, the sweep is halted.

Sweep (per mandate): w_quaternary in {0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45}
x similarity_threshold in {0.46,0.48,0.50} = 24 active configs, plus K0 drift
gate and K13 balance sanity probe ({0.30,0.30,0.40} @ thr=0.48) = 26 total.
For each w_q, mass is moved equally off the 14e B1 primary/tertiary pair:
  w_primary  = 0.475 - w_quaternary/2
  w_tertiary = 0.525 - w_quaternary/2
which reduces to the K0 baseline (0.475, 0.525) at w_q=0 and stays non-negative
through w_q=0.45 (w_p=0.25, w_t=0.30).

Run:  .venv/Scripts/python.exe notebooks/kaggle/14q_synth80_fusion/_build_14q.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "14q_synth80_fusion.ipynb"


def md(text: str) -> dict:
    lines = text.split("\n")
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(text: str) -> dict:
    lines = text.split("\n")
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


C0 = """# 14q -- Synth80 Fusion Sweep

CPU-only Stage 3-5 score-fusion sweep adding the **synth80** ResNeXt101-IBN-a quaternary stream (from 14p) on top of the 14e B1 / 14h v3 primary + tertiary anchor. Tests whether synth80 (our strongest CityFlow backbone, mAP 47.54%) breaks the 0.77936 MTMC IDF1 baseline.

Drift gate K0 (w_quaternary=0.0) MUST reproduce MTMC IDF1=0.77936, id_switches=154 EXACT. Sweep: w_quaternary in {0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45} x similarity_threshold in {0.46,0.48,0.50}, plus K13 balance sanity ({0.30,0.30,0.40} @ thr=0.48)."""

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

if not PROJECT.exists():
    print(f"Cloning {REPO_URL} ...")
    subprocess.check_call(["git", "clone", "--depth", "1", "-b", "feature/pretrained-ensemble", REPO_URL, str(PROJECT)])
else:
    print("Repo already present; pulling latest ...")
    subprocess.check_call(["git", "-C", str(PROJECT), "pull", "--ff-only"])

os.chdir(str(PROJECT))
sys.path.insert(0, str(PROJECT))


def pip(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])


pip("numpy", "scipy", "pandas", "faiss-cpu", "motmetrics", "omegaconf", "rich", "networkx>=3.1", "click", "loguru", "scikit-learn")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"], cwd=str(PROJECT))
print(f"Repo ready at {PROJECT}")'''

C4 = """## 2. Locate 14h Anchor, 14p Synth80 Features, And Ground Truth"""

C5 = '''SOURCE_14H_OWNER_SLUG = "yahiaakhalafallah/14h-robust-tracklet-pooling"
SOURCE_14P_OWNER_SLUG = "yahiaakhalafallah/14p-synth80-features"
SOURCE_14H_SLUG = SOURCE_14H_OWNER_SLUG.split("/", 1)[1]
SOURCE_14P_SLUG = SOURCE_14P_OWNER_SLUG.split("/", 1)[1]
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
        if lowered_slug in name or all(hint in name for hint in lowered_hints):
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


def find_quaternary_stage2_dir() -> Path:
    source_dir = find_input_dir(SOURCE_14P_SLUG, SOURCE_14P_OWNER_SLUG, hints=("14p", "synth80", "synth"))
    explicit_candidates = [
        source_dir / "outputs" / "14p_v1_features" / "stage2",
        source_dir / "14p_v1_features" / "stage2",
        source_dir / "stage2",
    ]
    for candidate in explicit_candidates:
        if (candidate / "embeddings_quaternary.npy").exists():
            print(f"14p synth80 quaternary input: {candidate}")
            return candidate

    matches = sorted(INPUT_ROOT.rglob("embeddings_quaternary.npy")) if INPUT_ROOT.exists() else []
    for match in matches:
        text = str(match).lower()
        if "14p" in text or "synth" in text:
            print(f"14p quaternary input discovered: {match.parent}")
            return match.parent
    if matches:
        print(f"14p quaternary input fallback: {matches[0].parent}")
        return matches[0].parent

    visible = [str(path) for path in INPUT_ROOT.rglob("*.npy")] if INPUT_ROOT.exists() else []
    raise FileNotFoundError(
        f"embeddings_quaternary.npy not found for {SOURCE_14P_OWNER_SLUG}. "
        f"Visible npy examples: {visible[:30]}"
    )


SOURCE_QUATERNARY_STAGE2_DIR = find_quaternary_stage2_dir()


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

C6 = """## 3. Validate And Normalize Synth80 Quaternary Stream"""

C7 = '''STATIC_CONTRACT_K0_DISABLE_PATH = "K0_DISABLES_QUATERNARY_BY_EMPTY_PATH_AND_ZERO_WEIGHT"
STATIC_CONTRACT_QUATERNARY_RENORM = "QUATERNARY_L2_RENORMALIZATION_AFTER_DROPPED_ZERO_HANDLING"

QUATERNARY_WORK_DIR = Path("/tmp/14p_quaternary_stage2")
QUATERNARY_WORK_DIR.mkdir(parents=True, exist_ok=True)
QUATERNARY_EMBEDDING_PATH = QUATERNARY_WORK_DIR / "embeddings_quaternary_l2.npy"
QUATERNARY_SECONDARY_ALIAS_PATH = QUATERNARY_WORK_DIR / "embeddings_secondary.npy"

source_index = json.loads((SOURCE_STAGE2_DIR / "embedding_index.json").read_text(encoding="utf-8"))
quaternary_index = json.loads((SOURCE_QUATERNARY_STAGE2_DIR / "embedding_index.json").read_text(encoding="utf-8"))
if source_index != quaternary_index:
    raise RuntimeError("14p synth80 embedding_index.json does not exactly match 14h v3 source ordering")
if len(source_index) != EXPECTED_TRACKLETS:
    raise RuntimeError(f"Expected {EXPECTED_TRACKLETS} embedding rows, found {len(source_index)}")

quaternary_raw = np.load(SOURCE_QUATERNARY_STAGE2_DIR / "embeddings_quaternary.npy").astype(np.float32)
if quaternary_raw.shape != (EXPECTED_TRACKLETS, 2048):
    raise RuntimeError(f"Unexpected quaternary shape: {quaternary_raw.shape}")
if not np.isfinite(quaternary_raw).all():
    raise RuntimeError("Quaternary embeddings contain NaN/Inf")

dropped_payload_path = SOURCE_QUATERNARY_STAGE2_DIR / "dropped_indices.json"
if dropped_payload_path.exists():
    dropped_payload = json.loads(dropped_payload_path.read_text(encoding="utf-8"))
    dropped_indices = [int(index) for index in dropped_payload.get("indices", [])]
else:
    norms_for_drop = np.linalg.norm(quaternary_raw, axis=1)
    dropped_indices = [int(index) for index in np.flatnonzero(norms_for_drop < 1e-8)]
if dropped_indices != EXPECTED_DROPPED_INDICES:
    raise RuntimeError(f"Unexpected dropped indices: {dropped_indices}; expected {EXPECTED_DROPPED_INDICES}")

quaternary = quaternary_raw.copy()
if dropped_indices:
    quaternary[np.array(dropped_indices, dtype=np.int64)] = 0.0

norms = np.linalg.norm(quaternary, axis=1, keepdims=True)
nonzero_mask = norms[:, 0] > 1e-8
quaternary[nonzero_mask] = quaternary[nonzero_mask] / norms[nonzero_mask]
quaternary[~nonzero_mask] = 0.0
quaternary = quaternary.astype(np.float32)

final_norms = np.linalg.norm(quaternary, axis=1)
if not np.allclose(quaternary[np.array(dropped_indices, dtype=np.int64)], 0.0):
    raise RuntimeError("Dropped quaternary rows are not zero after zero-fill handling")
if float(np.max(np.abs(final_norms[nonzero_mask] - 1.0))) > 1e-4:
    raise RuntimeError("Non-dropped quaternary rows are not unit norm after L2 renormalization")

np.save(QUATERNARY_EMBEDDING_PATH, quaternary)
np.save(QUATERNARY_SECONDARY_ALIAS_PATH, quaternary)
(QUATERNARY_WORK_DIR / "embedding_index.json").write_text(json.dumps(source_index, indent=2), encoding="utf-8")
(QUATERNARY_WORK_DIR / "dropped_indices.json").write_text(
    json.dumps({"indices": dropped_indices, "count": len(dropped_indices), "fill_value": 0.0}, indent=2),
    encoding="utf-8",
)

print(json.dumps({
    "contract_k0_disable_path": STATIC_CONTRACT_K0_DISABLE_PATH,
    "contract_quaternary_renorm": STATIC_CONTRACT_QUATERNARY_RENORM,
    "quaternary_path": str(QUATERNARY_EMBEDDING_PATH),
    "shape": list(quaternary.shape),
    "dropped_indices": dropped_indices,
    "zero_rows": [int(index) for index in np.flatnonzero(final_norms < 1e-8)],
    "unit_norm_rows": int(np.sum(np.abs(final_norms - 1.0) < 1e-4)),
    "norm_min": float(final_norms.min()),
    "norm_max": float(final_norms.max()),
}, indent=2))'''

C8 = """## 4. Define Synth80 Fusion Sweep"""

C9 = '''from src.core.config import load_config, save_config
from src.core.data_models import TrackletFeatures
from src.core.io_utils import load_tracklets_by_camera
from src.core.logging_utils import setup_logging
from src.stage3_indexing import run_stage3
from src.stage4_association import run_stage4
from src.stage5_evaluation import run_stage5

RUN_NAME = f"run_14q_synth80_fusion_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RUN_DIR = DATA_OUT / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)
setup_logging(level="INFO", log_file=RUN_DIR / "pipeline.log")
print(f"Run: {RUN_NAME}")

BASE_PRIMARY_WEIGHT = 0.475
BASE_TERTIARY_WEIGHT = 0.525
QUATERNARY_WEIGHTS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
SIMILARITY_THRESHOLDS = [0.46, 0.48, 0.50]
K13_LITERAL_WEIGHTS = {"w_primary": 0.30, "w_tertiary": 0.30, "w_quaternary": 0.40}

ANCHOR_CONFIG = {
    "aqe_k": 2,
    "fic_regularisation": 0.5,
}
# K0 drift gate: synth80 quaternary disabled (empty path + zero weight).
SWEEP_CONFIGS = [
    {
        "config_id": "K0",
        "block": "drift",
        "w_quaternary": 0.0,
        "w_primary": BASE_PRIMARY_WEIGHT,
        "w_tertiary": BASE_TERTIARY_WEIGHT,
        "similarity_threshold": 0.48,
        "notes": "K0 drift gate: K0_DISABLES_QUATERNARY_BY_EMPTY_PATH_AND_ZERO_WEIGHT",
        **ANCHOR_CONFIG,
    },
]
# Active grid: move mass equally off the 14e B1 primary/tertiary pair so that
# w_q=0 reduces to the K0 baseline (0.475, 0.525) and weights stay non-negative
# through w_q=0.45 (w_p=0.25, w_t=0.30).
for w_quaternary in QUATERNARY_WEIGHTS:
    w_primary = round(BASE_PRIMARY_WEIGHT - 0.5 * w_quaternary, 6)
    w_tertiary = round(BASE_TERTIARY_WEIGHT - 0.5 * w_quaternary, 6)
    for similarity_threshold in SIMILARITY_THRESHOLDS:
        SWEEP_CONFIGS.append({
            "config_id": f"K{len(SWEEP_CONFIGS)}",
            "block": "quaternary_grid",
            "w_quaternary": w_quaternary,
            "w_primary": w_primary,
            "w_tertiary": w_tertiary,
            "similarity_threshold": similarity_threshold,
            "notes": (
                f"synth80 quaternary weight {w_quaternary:.2f}; "
                f"non-negative table weights w_p={w_primary:.3f}, w_t={w_tertiary:.3f}"
            ),
            **ANCHOR_CONFIG,
        })
# Balance-sanity probe ("K13" in the 14k lineage). Its config_id is "KB" rather
# than "K13" because the active grid auto-numbers K1..K24 here (24 active configs),
# so a literal "K13" would COLLIDE with the active w_q=0.20/thr=0.50 config.
SWEEP_CONFIGS.append({
    "config_id": "KB",
    "block": "sanity",
    "w_quaternary": K13_LITERAL_WEIGHTS["w_quaternary"],
    "w_primary": K13_LITERAL_WEIGHTS["w_primary"],
    "w_tertiary": K13_LITERAL_WEIGHTS["w_tertiary"],
    "similarity_threshold": 0.48,
    "notes": "KB (K13-lineage) balance sanity probe uses literal weights {0.30, 0.30, 0.40}; NOT ratio-rescaled",
    **ANCHOR_CONFIG,
})
EXPECTED_CONFIG_COUNT = 1 + len(QUATERNARY_WEIGHTS) * len(SIMILARITY_THRESHOLDS) + 1
if len(SWEEP_CONFIGS) != EXPECTED_CONFIG_COUNT:
    raise RuntimeError(f"Expected {EXPECTED_CONFIG_COUNT} configs including K0+K13, got {len(SWEEP_CONFIGS)}")
for config in SWEEP_CONFIGS:
    total_weight = float(config["w_primary"]) + float(config["w_tertiary"]) + float(config["w_quaternary"])
    if abs(total_weight - 1.0) > 1e-6:
        raise RuntimeError(f"Weights do not sum to 1.0 for {config['config_id']}: {total_weight}")
    if float(config["w_primary"]) < -1e-9:
        raise RuntimeError(f"Negative primary weight for {config['config_id']}: {config['w_primary']}")
    if config["config_id"] == "KB" and (
        abs(float(config["w_primary"]) - 0.30) > 1e-9
        or abs(float(config["w_tertiary"]) - 0.30) > 1e-9
        or abs(float(config["w_quaternary"]) - 0.40) > 1e-9
    ):
        raise RuntimeError(f"KB (K13-lineage) literal weights were modified: {config}")

K0_REPRO_TARGET = 0.77936
K0_REPRO_TOL = 0.001
K0_ID_SWITCH_TARGET = 154
K13_REAL_ENSEMBLE_MIN = 0.7795
WIN_THRESHOLD = 0.7810
MARGINAL_MIN = 0.7795
NEUTRAL_MIN = 0.7785
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
    recovery_dir = Path("/kaggle/working/outputs/14q_synth80_recovery") / config_id
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
    w_tertiary = float(config["w_tertiary"])
    sim_thresh = float(config["similarity_threshold"])
    aqe_k = int(config["aqe_k"])
    fic_reg = float(config["fic_regularisation"])
    if w_quaternary == 0.0:
        secondary_path = ""
        secondary_weight = 0.0
    else:
        secondary_path = str(QUATERNARY_EMBEDDING_PATH)
        secondary_weight = w_quaternary
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
        f"stage4.association.secondary_embeddings.weight={secondary_weight}",
        f"stage4.association.tertiary_embeddings.path={stage2_dir / 'embeddings_tertiary.npy'}",
        f"stage4.association.tertiary_embeddings.weight={w_tertiary}",
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
        f"Running {config_id}: w_primary={config['w_primary']:.3f}, "
        f"w_tertiary={config['w_tertiary']:.3f}, w_quaternary={config['w_quaternary']:.3f}, "
        f"sim_thresh={config['similarity_threshold']:.2f}"
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

drift_check_result = run_config(SWEEP_CONFIGS[0])
results.append(drift_check_result)
(RUN_DIR / "14q_partial_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

k0_idf1 = float(drift_check_result["mtmc_idf1"])
k0_id_switches = drift_check_result.get("id_switches")
if abs(k0_idf1 - K0_REPRO_TARGET) > K0_REPRO_TOL or k0_id_switches != K0_ID_SWITCH_TARGET:
    drift_detected = True
    halt_reason = (
        f"K0 drift gate failed: got idf1={k0_idf1:.5f}, id_switches={k0_id_switches}; "
        f"expected idf1 within {K0_REPRO_TARGET:.5f} +/- {K0_REPRO_TOL:.5f} "
        f"and id_switches={K0_ID_SWITCH_TARGET}"
    )
    print(halt_reason)
else:
    print(f"K0 drift gate passed: {k0_idf1:.5f}, id_switches={k0_id_switches}")
    for config in SWEEP_CONFIGS[1:]:
        row = run_config(config)
        results.append(row)
        sweep_results.append(row)
        (RUN_DIR / "14q_partial_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

overall_best = max(results, key=lambda row: row["mtmc_idf1"] if row["mtmc_idf1"] is not None else -1.0)
best_quaternary = max(sweep_results, key=lambda row: row["mtmc_idf1"] if row["mtmc_idf1"] is not None else -1.0) if sweep_results else None
k13_result = next((row for row in results if row.get("config_id") == "KB"), None)
k13_matching_config = next(
    (
        row
        for row in results
        if row.get("config_id") != "KB"
        and abs(float(row.get("w_quaternary", -1.0)) - 0.40) < 1e-9
        and abs(float(row.get("similarity_threshold", -1.0)) - 0.48) < 1e-9
    ),
    None,
)
k13_sanity_check = {
    "k13_config_id": k13_result.get("config_id") if k13_result else None,
    "matching_config_id": k13_matching_config.get("config_id") if k13_matching_config else None,
    "k13_mtmc_idf1": k13_result.get("mtmc_idf1") if k13_result else None,
    "matching_mtmc_idf1": k13_matching_config.get("mtmc_idf1") if k13_matching_config else None,
    "delta_vs_matching": (
        float(k13_result["mtmc_idf1"]) - float(k13_matching_config["mtmc_idf1"])
        if k13_result and k13_matching_config and k13_result.get("mtmc_idf1") is not None and k13_matching_config.get("mtmc_idf1") is not None
        else None
    ),
    "real_ensemble_gate": K13_REAL_ENSEMBLE_MIN,
    "passes_real_ensemble_gate": (
        bool(float(k13_result["mtmc_idf1"]) >= K13_REAL_ENSEMBLE_MIN)
        if k13_result and k13_result.get("mtmc_idf1") is not None
        else False
    ),
}

best_idf1 = float(overall_best["mtmc_idf1"])
k13_idf1 = float(k13_result["mtmc_idf1"]) if k13_result and k13_result.get("mtmc_idf1") is not None else None
if drift_detected:
    verdict_band = "DRIFT_FAIL"
elif best_idf1 >= WIN_THRESHOLD and k13_idf1 is not None and k13_idf1 >= K13_REAL_ENSEMBLE_MIN:
    verdict_band = "WIN"
elif best_idf1 >= WIN_THRESHOLD:
    verdict_band = "PRIMARY-SUPPRESSION"
elif best_idf1 >= MARGINAL_MIN:
    verdict_band = "MARGINAL"
elif best_idf1 >= NEUTRAL_MIN:
    verdict_band = "NEUTRAL"
else:
    verdict_band = "DEAD"

print("\\n" + "#" * 80)
print(
    f"BEST 14q CONFIG: {overall_best['config_id']} w_p={overall_best['w_primary']:.3f} "
    f"w_t={overall_best['w_tertiary']:.3f} w_q={overall_best['w_quaternary']:.3f} "
    f"thr={overall_best['similarity_threshold']:.2f} MTMC_IDF1={overall_best['mtmc_idf1']:.5f} "
    f"id_switches={overall_best.get('id_switches')} verdict_band={verdict_band} drift={drift_detected}"
)
print(f"K13 sanity check: {json.dumps(k13_sanity_check, indent=2)}")
if halt_reason:
    print(f"HALT: {halt_reason}")
print("#" * 80)'''

C12 = """## 6. Persist Summary"""

C13 = '''summary_dir = Path("/kaggle/working/outputs/14q_v1_summary")
summary_dir.mkdir(parents=True, exist_ok=True)
legacy_summary_path = summary_dir / "14q_summary.json"
required_summary_path = Path("/kaggle/working/outputs/14q_synth80_summary.json")
root_summary_path = Path("/kaggle/working/14q_synth80_summary.json")

compact_configs = [
    {
        "config_id": row.get("config_id"),
        "w_primary": row.get("w_primary"),
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

summary = {
    "run_name": RUN_NAME,
    "source_14h_run_name": SOURCE_14H_RUN_NAME,
    "experiment": "14q-synth80-fusion",
    "kernel": "yahiaakhalafallah/14q-synth80-fusion",
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
    "k13_sanity_check": k13_sanity_check,
    "halt_reason": halt_reason,
    "total_config_count": len(SWEEP_CONFIGS),
    "executed_config_count": len(results),
    "feature_sources": {
        "primary_and_tertiary_kernel": SOURCE_14H_OWNER_SLUG,
        "quaternary_kernel": SOURCE_14P_OWNER_SLUG,
        "checkpoint": str(checkpoint),
        "stage1_tracklets": str(SOURCE_STAGE1_DIR),
        "stage2_features": str(SOURCE_STAGE2_DIR),
        "quaternary_stage2": str(SOURCE_QUATERNARY_STAGE2_DIR),
        "quaternary_normalized_path": str(QUATERNARY_EMBEDDING_PATH),
    },
    "fixed_config": {
        "pca_components": 384,
        "algorithm": ALGORITHM,
        "aqe_k": 2,
        "fic_regularisation": 0.5,
        "base_primary_weight": BASE_PRIMARY_WEIGHT,
        "base_tertiary_weight": BASE_TERTIARY_WEIGHT,
        "k13_literal_weights": K13_LITERAL_WEIGHTS,
        "gallery_expansion": True,
        "gallery_threshold": GALLERY_THRESH,
        "orphan_match_threshold": ORPHAN_MATCH_THRESH,
        "temporal_overlap_bonus": 0.05,
        "intra_merge": INTRA_MERGE,
        "intra_merge_threshold": INTRA_MERGE_THRESH,
        "intra_merge_gap": INTRA_MERGE_GAP,
        "mtmc_only_submission": MTMC_ONLY,
        "score_fusion_math": "cosine_sim = w_p * sim_primary + w_t * sim_tertiary + w_q * sim_quaternary; dropped zero rows produce sim_quaternary=0.0; K0 uses empty secondary path and weight 0.0",
        "quaternary_contract": STATIC_CONTRACT_QUATERNARY_RENORM,
        "k0_contract": STATIC_CONTRACT_K0_DISABLE_PATH,
    },
    "sweep_grid": {
        "w_quaternary": QUATERNARY_WEIGHTS,
        "similarity_threshold": SIMILARITY_THRESHOLDS,
        "grid_size_excluding_k0_and_k13": len(SWEEP_CONFIGS) - 2,
        "k13_literal_weights": K13_LITERAL_WEIGHTS,
    },
    "stop_criteria": {
        "win_threshold": WIN_THRESHOLD,
        "marginal_range": [MARGINAL_MIN, WIN_THRESHOLD],
        "neutral_range": [NEUTRAL_MIN, MARGINAL_MIN],
        "dead_below": NEUTRAL_MIN,
        "drift_fail_range": [K0_REPRO_TARGET - K0_REPRO_TOL, K0_REPRO_TARGET + K0_REPRO_TOL],
        "drift_condition": f"abs(K0 - {K0_REPRO_TARGET}) > {K0_REPRO_TOL} or id_switches != {K0_ID_SWITCH_TARGET}",
        "primary_suppression_condition": f"best >= {WIN_THRESHOLD} and K13 < {K13_REAL_ENSEMBLE_MIN}",
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
    md(C0), code(C1), md(C2), code(C3), md(C4), code(C5),
    md(C6), code(C7), md(C8), code(C9), md(C10), code(C11), md(C12), code(C13),
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
