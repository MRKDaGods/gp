"""Centralised configuration constants for the backend package.

All path constants, environment flags, and shared regex patterns live here.
Nothing in this module imports from other backend/ modules — it is a pure
dependency leaf.
"""
import os
import re
import shutil as _shutil
import sys
from pathlib import Path
from typing import List, Optional

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# Project root: backend/config.py → backend/ → project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Python executable used to spawn pipeline subprocesses.
# Prefer the project venv so all ML deps are available.
_VENV_PYTHON = _PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
_PIPELINE_PYTHON: str = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable

# ffprobe for video duration probing (optional)
_FFPROBE = _shutil.which("ffprobe")

# ── Directory constants ────────────────────────────────────────────────────
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
# Runs are read from more than one root: app-created runs land in `outputs/`,
# while Kaggle-imported / offline-pipeline runs land in `data/outputs/`. Reads
# must look in both, otherwise completed runs become invisible to the UI.
OUTPUT_DIRS = [OUTPUT_DIR, Path("data/outputs")]
TIMELINE_DEBUG_LOG = OUTPUT_DIR / "timeline_query_debug.log"
CITYFLOW_DIR = Path("data/raw/cityflowv2")
DATASET_DIR = Path("dataset")
# Source-of-truth for selectable tracking datasets: each YAML carries a
# `stage0.input_dir` plus dataset-appropriate ingestion settings.
DATASET_CONFIG_DIR = Path("configs/datasets")
# Sandbox root for the custom "browse folder" picker. The browser cannot
# escape this directory (path-traversal guarded).
DATASET_BROWSE_ROOT = Path("data/raw")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".m4v"}
DEMO_VIDEO_FALLBACK = Path("S02_c008.avi")

# ── Feature flags ──────────────────────────────────────────────────────────
ENABLE_KAGGLE_IMPORT = (
    os.getenv("MTMC_ENABLE_KAGGLE_IMPORT", "1").strip().lower()
    in {"1", "true", "yes", "on"}
)

# ── Precompute run id ──────────────────────────────────────────────────────
PRECOMPUTE_RUN_ID = "dataset_precompute_s01"

# ── Pipeline stage labels ──────────────────────────────────────────────────
_STAGE_NAMES = {
    0: "Ingestion & Pre-Processing",
    1: "Detection & Tracking (YOLOv26 + DeepOCSORT)",
    2: "Feature Extraction (ReID Embeddings)",
    3: "Indexing (FAISS + SQLite)",
    4: "Cross-Camera Association",
    5: "Evaluation",
    6: "Visualization",
}

# Regex to detect stage start markers from pipeline stdout
_STAGE_LINE_RE = re.compile(r"Stage\s+(\d)")
# Regex to detect per-camera processing lines
_CAMERA_LINE_RE = re.compile(r"Processing camera\s+([\w_]+)")
# Structured per-frame progress emitted by stage 1: "[PROGRESS] camera=X frame=i total=N"
_FRAME_LINE_RE = re.compile(r"\[PROGRESS\]\s+camera=([\w_]+)\s+frame=(\d+)\s+total=(\d+)")
# Total camera count emitted once before the stage-1 loop: "[PROGRESS] cameras_total=N"
_CAMERAS_TOTAL_RE = re.compile(r"\[PROGRESS\]\s+cameras_total=(\d+)")

# ── Timeline / ReID similarity thresholds ─────────────────────────────────
# Applied in TimelineService._score_trajectories(); both conditions must hold
# for a trajectory to be considered a visual match.
# Lowered from 0.82/0.74 to accommodate cross-run queries where probe
# embeddings (short clips, single camera) score lower than gallery self-queries.
SIMILARITY_THRESHOLD_MEAN: float = 0.72
SIMILARITY_THRESHOLD_P25: float = 0.60

# PCA model used to project probe embeddings when probe_dim > gallery_dim
PCA_MODEL_PATH = Path("models/reid/pca_transform.pkl")


# ── Run discovery across output roots ──────────────────────────────────────
def _is_safe_run_id(run_id: str) -> bool:
    """Reject run_ids that could escape an output root via path traversal."""
    return bool(run_id) and "/" not in run_id and "\\" not in run_id and ".." not in run_id


def resolve_run_dir(run_id: str) -> Optional[Path]:
    """Locate a run directory across known output roots.

    Returns the first existing match (preferring `outputs/`), or None.
    """
    if not _is_safe_run_id(run_id):
        return None
    for root in OUTPUT_DIRS:
        candidate = root / run_id
        if candidate.exists():
            return candidate
    return None


def list_run_dirs() -> List[Path]:
    """All run directories across known output roots, deduped by name.

    A directory counts as a run if it contains at least a `stage1/` folder.
    """
    seen: dict[str, Path] = {}
    for root in OUTPUT_DIRS:
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if d.is_dir() and d.name not in seen and (d / "stage1").exists():
                seen[d.name] = d
    return list(seen.values())
