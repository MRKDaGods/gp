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
TIMELINE_DEBUG_LOG = OUTPUT_DIR / "timeline_query_debug.log"
CITYFLOW_DIR = Path("data/raw/cityflowv2")
DATASET_DIR = Path("dataset")
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

# ── Timeline / ReID similarity thresholds ─────────────────────────────────
# Applied in TimelineService._score_trajectories(); both conditions must hold
# for a trajectory to be considered a visual match.
SIMILARITY_THRESHOLD_MEAN: float = 0.82
SIMILARITY_THRESHOLD_P25: float = 0.74

# PCA model used to project probe embeddings when probe_dim > gallery_dim
PCA_MODEL_PATH = Path("models/reid/pca_transform.pkl")
