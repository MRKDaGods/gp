"""MTMC Forensic Dashboard — Multi-Dataset Tracker.

Compact, multi-dataset forensic dashboard for cross-camera identity tracking.
Key features:
  - Dataset-aware: switch between WILDTRACK / CityFlowV2 / EPFL etc.
  - Compact identity grid: pick any person or vehicle, see them across all cameras
  - Cross-camera timeline: Gantt-style bars showing when/where an identity appears
  - Surveillance replay, identity corrections, audit log

Run:
    streamlit run src/apps/forensic_dashboard.py
    streamlit run src/apps/forensic_dashboard.py -- --run-dir data/outputs/run_XXX
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.apps.corrections_store import CorrectionsStore
from src.core.data_models import GlobalTrajectory, Tracklet
from src.core.io_utils import (
    load_global_trajectories,
    load_frame_manifest,
    load_evaluation_result,
    save_global_trajectories,
)

# ---------------------------------------------------------------------------
# Constants & Theme
# ---------------------------------------------------------------------------

ACCENT = "#2563eb"
DIM_FACTOR = 0.35
ACCENT_BGR = (235, 99, 37)  # BGR

DATASET_COLORS = {
    "wildtrack": "#3b82f6",
    "cityflowv2": "#f97316",
    "epfl_lab": "#22c55e",
    "epfl_terrace": "#a855f7",
    "unknown": "#6b7280",
}
DATASET_ICONS = {
    "wildtrack": "🚶", "cityflowv2": "🚗", "epfl_lab": "🏢",
    "epfl_terrace": "T", "unknown": "?",
}
CLASS_ICONS = {"person": "🚶", "car": "🚗", "bus": "🚌", "truck": "🚛"}

# 20-colour palette for identities in timeline view
ID_PALETTE = [
    "#ef4444", "#3b82f6", "#22c55e", "#f97316", "#a855f7",
    "#06b6d4", "#eab308", "#ec4899", "#14b8a6", "#f43f5e",
    "#6366f1", "#84cc16", "#e11d48", "#0284c7", "#65a30d",
    "#78716c", "#475569", "#d946ef", "#059669", "#7c3aed",
]

COMPACT_CSS = """
<style>
    .block-container { padding-top: 0.75rem; padding-bottom: 0.5rem; }
    [data-testid="stMetric"] {
        background: var(--background-color); border: 1px solid #e2e8f0;
        border-radius: 8px; padding: 8px 12px;
    }
    [data-testid="stMetricValue"] { font-size: 1.3rem; }
    [data-testid="stMetricLabel"] { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.5px; }
    .ds-badge {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-weight: 600; font-size: 0.78rem; margin-right: 4px; vertical-align: middle;
    }
    .cam-pill {
        display: inline-block; padding: 1px 7px; border-radius: 10px;
        font-size: 0.72rem; margin: 1px 2px; font-weight: 500;
    }
    .cam-on  { background: #dbeafe; color: #1d4ed8; border: 1px solid #93c5fd; }
    .cam-off { background: #f1f5f9; color: #94a3b8; border: 1px solid #e2e8f0; }
    .info-card {
        background: var(--background-color); border: 1px solid #e2e8f0;
        border-radius: 8px; padding: 12px; margin: 4px 0;
    }
    .section-hd {
        font-size: 0.92rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.5px; color: #64748b; margin: 12px 0 6px 0;
    }
</style>
"""


def ds_badge(name: str) -> str:
    c = DATASET_COLORS.get(name, DATASET_COLORS["unknown"])
    icon = DATASET_ICONS.get(name, "📁")
    return (f'<span class="ds-badge" style="background:{c}18;color:{c};'
            f'border:1px solid {c}">{icon} {name.upper()}</span>')


# ---------------------------------------------------------------------------
# Dataset / Run Discovery
# ---------------------------------------------------------------------------

OUTPUTS_DIR = Path("data/outputs")


def discover_runs() -> Dict[str, List[Path]]:
    """Scan outputs directory and group runs by dataset name."""
    by_ds: Dict[str, List[Path]] = {}
    if not OUTPUTS_DIR.exists():
        return by_ds
    for d in sorted(OUTPUTS_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        nf = d / "dataset_name.txt"
        if nf.exists():
            ds = nf.read_text().strip()
        else:
            stem = d.name.lower()
            ds = next((p for p in DATASET_COLORS if stem.startswith(p)), "unknown")
        by_ds.setdefault(ds, []).append(d)
    return by_ds


# ---------------------------------------------------------------------------
# Data Loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_trajectories(path: str) -> List[dict]:
    trajs = load_global_trajectories(path)
    result = []
    for t in trajs:
        result.append({
            "global_id": t.global_id,
            "class_name": t.class_name,
            "num_cameras": t.num_cameras,
            "total_duration": t.total_duration,
            "time_span": t.time_span,
            "camera_sequence": t.camera_sequence,
            "tracklets": [
                {
                    "track_id": tk.track_id,
                    "camera_id": tk.camera_id,
                    "class_id": tk.class_id,
                    "class_name": tk.class_name,
                    "start_time": tk.start_time,
                    "end_time": tk.end_time,
                    "num_frames": tk.num_frames,
                    "mean_confidence": tk.mean_confidence,
                    "frames": [
                        {"frame_id": f.frame_id, "timestamp": f.timestamp,
                         "bbox": list(f.bbox), "confidence": f.confidence}
                        for f in tk.frames
                    ],
                }
                for tk in t.tracklets
            ],
        })
    return result


@st.cache_data
def load_manifest(path: str) -> dict:
    frames = load_frame_manifest(path)
    if not frames:
        return {"cameras": [], "frame_counts": {}, "max_frame": 0,
                "source_w": 1920, "source_h": 1080}
    cameras = sorted({f.camera_id for f in frames})
    fc = {}
    for f in frames:
        fc[f.camera_id] = fc.get(f.camera_id, 0) + 1
    return {
        "cameras": cameras,
        "frame_counts": fc,
        "max_frame": max(f.frame_id for f in frames),
        "source_w": frames[0].width if frames else 1920,
        "source_h": frames[0].height if frames else 1080,
    }


@st.cache_data
def load_eval(path: str) -> Optional[dict]:
    try:
        r = load_evaluation_result(path)
        return {
            "mota": r.mota, "idf1": r.idf1, "hota": r.hota,
            "id_switches": r.id_switches, "details": r.details,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Frame / Crop Helpers
# ---------------------------------------------------------------------------

@st.cache_data
def read_frame_cached(stage0_dir: str, camera_id: str, frame_id: int) -> Optional[np.ndarray]:
    path = Path(stage0_dir) / camera_id / f"frame_{frame_id:06d}.jpg"
    if not path.exists():
        # Try .png fallback
        path = path.with_suffix(".png")
        if not path.exists():
            return None
    return cv2.imread(str(path))


def get_crop(stage0_dir: str, camera_id: str, frame_id: int,
             bbox: list, pad: int = 10) -> Optional[Image.Image]:
    img = read_frame_cached(stage0_dir, camera_id, frame_id)
    if img is None:
        return None
    h, w = img.shape[:2]
    x1, y1 = max(0, int(bbox[0]) - pad), max(0, int(bbox[1]) - pad)
    x2, y2 = min(w, int(bbox[2]) + pad), min(h, int(bbox[3]) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return Image.fromarray(cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))


def get_representative_crop(stage0_dir: str, tracklet: dict) -> Optional[Image.Image]:
    frames = tracklet["frames"]
    if not frames:
        return None
    mid = frames[len(frames) // 2]
    return get_crop(stage0_dir, tracklet["camera_id"], mid["frame_id"], mid["bbox"])


def render_surveillance_frame(
    stage0_dir: str, camera_id: str, frame_id: int,
    bbox: Optional[list] = None, dim: bool = False,
) -> Optional[Image.Image]:
    img = read_frame_cached(stage0_dir, camera_id, frame_id)
    if img is None:
        return None
    img = img.copy()
    if bbox is not None:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), ACCENT_BGR, 3)
        cv2.rectangle(img, (x1, y1 - 22), (x1 + 60, y1), ACCENT_BGR, -1)
        cv2.putText(img, "TARGET", (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    elif dim:
        img = (img.astype(np.float32) * DIM_FACTOR).astype(np.uint8)
    cv2.rectangle(img, (0, 0), (len(camera_id) * 12 + 14, 24), (0, 0, 0), -1)
    cv2.putText(img, camera_id, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------------------------
# Trajectory Helpers
# ---------------------------------------------------------------------------

def build_frame_lookup(traj: dict) -> Dict[int, Dict[str, dict]]:
    result: Dict[int, Dict[str, dict]] = {}
    for tk in traj["tracklets"]:
        for f in tk["frames"]:
            result.setdefault(f["frame_id"], {})[tk["camera_id"]] = f
    return result


def get_traj_by_gid(trajectories: List[dict], gid: int) -> Optional[dict]:
    return next((t for t in trajectories if t["global_id"] == gid), None)


# ---------------------------------------------------------------------------
# Session State (multi-dataset)
# ---------------------------------------------------------------------------

def _load_run_into_state(run_dir: Path, ds_name: str):
    """Load a single run's data into session state under ds_name key."""
    key = f"ds_{ds_name}"
    traj_path = run_dir / "stage4" / "global_trajectories.json"
    corrected = run_dir / "stage4" / "corrected_trajectories.json"
    load_path = corrected if corrected.exists() else traj_path

    entry: dict = {"run_dir": str(run_dir), "ds": ds_name, "trajectories": [],
                   "manifest": None, "eval": None, "stage0_dir": str(run_dir / "stage0")}
    if load_path.exists():
        entry["trajectories"] = load_trajectories(str(load_path))
    manifest_path = run_dir / "stage0" / "frames_manifest.json"
    if manifest_path.exists():
        entry["manifest"] = load_manifest(str(manifest_path))
    eval_path = run_dir / "stage5" / "evaluation_report.json"
    if eval_path.exists():
        entry["eval"] = load_eval(str(eval_path))
    entry["corrections"] = CorrectionsStore(run_dir / "corrections.db")

    st.session_state[key] = entry
    return entry


def get_active_ds() -> Optional[dict]:
    """Return the session-state dict for the currently-selected dataset."""
    ds = st.session_state.get("active_ds")
    if ds is None:
        return None
    return st.session_state.get(f"ds_{ds}")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

# ┌─────────────────────────────────────────────────────────────────────────┐
# │ OVERVIEW                                                                │
# └─────────────────────────────────────────────────────────────────────────┘

def page_overview():
    st.markdown("## Overview")

    loaded = {k.removeprefix("ds_"): v for k, v in st.session_state.items()
              if k.startswith("ds_") and isinstance(v, dict)}
    if not loaded:
        st.info("Select a dataset from the sidebar.")
        return

    for ds_name, entry in loaded.items():
        trajs = entry["trajectories"]
        ev = entry["eval"]
        color = DATASET_COLORS.get(ds_name, "#6b7280")
        st.markdown(ds_badge(ds_name) + f'  <small style="color:#94a3b8">{Path(entry["run_dir"]).name}</small>',
                    unsafe_allow_html=True)

        if not trajs:
            st.warning("No trajectories loaded.")
            st.markdown("---")
            continue

        # ── Metric pills ─────────────────────────────────────────────
        all_cams = sorted({tk["camera_id"] for t in trajs for tk in t["tracklets"]})
        multi = sum(1 for t in trajs if t["num_cameras"] > 1)
        cols = st.columns(7)
        cols[0].metric("Identities", len(trajs))
        cols[1].metric("Tracklets", sum(len(t["tracklets"]) for t in trajs))
        cols[2].metric("Multi-Cam", multi)
        cols[3].metric("Cameras", len(all_cams))
        cols[4].metric("Avg Dur", f'{np.mean([t["total_duration"] for t in trajs]):.1f}s')
        if ev and ev.get("idf1"):
            cols[5].metric("IDF1", f'{ev["idf1"]:.1%}')
            cols[6].metric("MOTA", f'{ev["mota"]:.1%}')
        else:
            cols[5].metric("IDF1", "—")
            cols[6].metric("MOTA", "—")

        # ── Camera + class compact row ────────────────────────────────
        c_left, c_right = st.columns([3, 2])
        with c_left:
            manifest = entry.get("manifest")
            cam_html = ""
            for cam in all_cams:
                fc = manifest["frame_counts"].get(cam, "?") if manifest else "?"
                tk_n = sum(1 for t in trajs for tk in t["tracklets"] if tk["camera_id"] == cam)
                cam_html += (f'<span class="cam-pill cam-on">{cam} '
                             f'<small>{fc}f / {tk_n}tk</small></span>')
            st.markdown(cam_html, unsafe_allow_html=True)

        with c_right:
            cls_counts: Dict[str, int] = {}
            for t in trajs:
                cls_counts[t["class_name"]] = cls_counts.get(t["class_name"], 0) + 1
            parts = " · ".join(f"{CLASS_ICONS.get(c, '')} {c} **{n}**" for c, n in sorted(cls_counts.items()))
            st.markdown(parts)

        # ── Per-camera eval breakdown (collapsed) ─────────────────────
        if ev and ev.get("details"):
            per_cam = ev["details"].get("per_camera", {})
            if per_cam:
                with st.expander("Per-camera metrics"):
                    rows = [{"Camera": cam, "MOTA": f'{m.get("mota",0):.3f}',
                             "IDF1": f'{m.get("idf1",0):.3f}',
                             "IDSW": int(m.get("id_switches", 0))}
                            for cam, m in sorted(per_cam.items())]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── Top multi-camera identities ───────────────────────────────
        with st.expander("Top multi-camera identities"):
            ranked = sorted(trajs, key=lambda t: (t["num_cameras"], t["total_duration"]), reverse=True)[:10]
            rows = [{
                "ID": t["global_id"],
                "Class": t["class_name"],
                "Cams": t["num_cameras"],
                "Duration": f'{t["total_duration"]:.1f}s',
                "Route": " → ".join(t["camera_sequence"]),
            } for t in ranked]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("---")


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ IDENTITY BROWSER                                                        │
# └─────────────────────────────────────────────────────────────────────────┘

def page_gallery():
    st.markdown("## Identity Browser")
    data = get_active_ds()
    if data is None:
        st.info("Select a dataset from the sidebar.")
        return

    trajs = data["trajectories"]
    stage0_dir = data["stage0_dir"]
    ds = data["ds"]

    if not trajs:
        st.warning("No trajectories loaded.")
        return

    # ── Compact filter bar ────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns([1.2, 0.8, 1, 0.7, 1.5])
    classes = sorted({t["class_name"] for t in trajs})
    with c1:
        fclass = st.selectbox("Class", ["All"] + classes, key=f"gal_cls_{ds}")
    with c2:
        fmin = st.number_input("Min cams", 1, 20, 1, key=f"gal_min_{ds}")
    with c3:
        fsort = st.selectbox("Sort", ["Cameras ↓", "Duration ↓", "Frames ↓", "ID ↑"], key=f"gal_sort_{ds}")
    with c4:
        fsize = st.selectbox("Per page", [24, 48, 96], key=f"gal_sz_{ds}")
    with c5:
        fsearch = st.text_input("Search ID", placeholder="e.g. 12", key=f"gal_q_{ds}")

    # Apply filters
    filtered = list(trajs)
    if fclass != "All":
        filtered = [t for t in filtered if t["class_name"] == fclass]
    filtered = [t for t in filtered if t["num_cameras"] >= fmin]
    if fsearch.strip():
        try:
            filtered = [t for t in filtered if t["global_id"] == int(fsearch)]
        except ValueError:
            pass
    if "Cameras" in fsort:
        filtered.sort(key=lambda t: (t["num_cameras"], t["total_duration"]), reverse=True)
    elif "Duration" in fsort:
        filtered.sort(key=lambda t: t["total_duration"], reverse=True)
    elif "Frames" in fsort:
        filtered.sort(key=lambda t: sum(tk["num_frames"] for tk in t["tracklets"]), reverse=True)
    else:
        filtered.sort(key=lambda t: t["global_id"])

    total_pages = max(1, (len(filtered) + fsize - 1) // fsize)
    pn = st.number_input("Page", 1, total_pages, 1, key=f"gal_pg_{ds}") - 1
    page_items = filtered[pn * fsize:(pn + 1) * fsize]
    st.caption(f"Showing {len(page_items)} of {len(filtered)} identities")

    # ── Card grid (5 columns) ────────────────────────────────────────
    n_cols = 5
    for row_start in range(0, len(page_items), n_cols):
        cols = st.columns(n_cols)
        for j in range(n_cols):
            idx = row_start + j
            if idx >= len(page_items):
                break
            t = page_items[idx]
            with cols[j]:
                container = st.container(border=True)
                with container:
                    # Crop thumbnail
                    if t["tracklets"]:
                        longest = max(t["tracklets"], key=lambda tk: tk["num_frames"])
                        crop = get_representative_crop(stage0_dir, longest)
                        if crop:
                            st.image(crop, use_container_width=True)

                    icon = CLASS_ICONS.get(t["class_name"], "")
                    cams_short = ", ".join(sorted({tk["camera_id"] for tk in t["tracklets"]}))
                    st.markdown(
                        f'**{icon} ID {t["global_id"]}** · {t["class_name"]}  \n'
                        f'📷{t["num_cameras"]} · ⏱{t["total_duration"]:.1f}s  \n'
                        f'<small style="color:#94a3b8">{cams_short}</small>',
                        unsafe_allow_html=True,
                    )
                    if st.button("Inspect →", key=f"ins_{ds}_{t['global_id']}"):
                        st.session_state.selected_gid = t["global_id"]
                        st.session_state.nav_to = "Inspector"
                        st.rerun()


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ CROSS-CAMERA TIMELINE                                                   │
# └─────────────────────────────────────────────────────────────────────────┘

def page_timeline():
    st.markdown("## Cross-Camera Timeline")
    data = get_active_ds()
    if data is None:
        st.info("Select a dataset.")
        return

    trajs = data["trajectories"]
    ds = data["ds"]
    if not trajs:
        st.warning("No trajectories.")
        return

    # Controls
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        only_multi = st.checkbox("Multi-camera only", True, key=f"tl_multi_{ds}")
    with c2:
        max_ids = st.slider("Max identities", 5, 60, 20, key=f"tl_max_{ds}")
    with c3:
        tl_class = st.selectbox("Class",
                                ["All"] + sorted({t["class_name"] for t in trajs}),
                                key=f"tl_cls_{ds}")

    filtered = list(trajs)
    if only_multi:
        filtered = [t for t in filtered if t["num_cameras"] > 1]
    if tl_class != "All":
        filtered = [t for t in filtered if t["class_name"] == tl_class]
    filtered = sorted(filtered, key=lambda t: t["num_cameras"], reverse=True)[:max_ids]

    if not filtered:
        st.info("No identities match filters.")
        return

    cameras = sorted({tk["camera_id"] for t in filtered for tk in t["tracklets"]})
    fig = go.Figure()
    for ti, traj in enumerate(filtered):
        color = ID_PALETTE[ti % len(ID_PALETTE)]
        first = True
        for tk in traj["tracklets"]:
            dur = max(tk["end_time"] - tk["start_time"], 0.3)
            fig.add_trace(go.Bar(
                x=[dur], y=[tk["camera_id"]], base=[tk["start_time"]],
                orientation="h",
                marker=dict(color=color, opacity=0.85),
                name=f'ID {traj["global_id"]}',
                legendgroup=str(traj["global_id"]),
                showlegend=first,
                hovertemplate=(
                    f'ID {traj["global_id"]} ({traj["class_name"]})<br>'
                    f'Camera: {tk["camera_id"]}<br>'
                    f'Time: {tk["start_time"]:.1f}s – {tk["end_time"]:.1f}s<br>'
                    f'Frames: {tk["num_frames"]}<extra></extra>'
                ),
            ))
            first = False

    fig.update_layout(
        barmode="overlay",
        height=max(280, len(cameras) * 40 + 100),
        xaxis_title="Time (seconds)",
        yaxis=dict(categoryorder="array", categoryarray=cameras, title=""),
        margin=dict(t=10, b=40, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=10), itemsizing="constant"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ SURVEILLANCE VIEW                                                       │
# └─────────────────────────────────────────────────────────────────────────┘

def page_surveillance():
    st.markdown("## Surveillance")
    data = get_active_ds()
    if data is None or data["manifest"] is None:
        st.info("No frame data available. Run the pipeline with frame extraction first.")
        return

    trajs = data["trajectories"]
    manifest = data["manifest"]
    stage0_dir = data["stage0_dir"]
    cameras = manifest["cameras"]

    if not trajs:
        st.warning("No trajectories.")
        return

    # Controls
    c_id, c_frame = st.columns([1, 3])
    with c_id:
        gid_opts = sorted(
            [t["global_id"] for t in trajs],
            key=lambda g: next(t["num_cameras"] for t in trajs if t["global_id"] == g),
            reverse=True,
        )
        labels = {
            g: f"ID {g} ({next(t['class_name'] for t in trajs if t['global_id']==g)}, "
               f"{next(t['num_cameras'] for t in trajs if t['global_id']==g)} cams)"
            for g in gid_opts
        }
        selected_gid = st.selectbox("Target", gid_opts, format_func=lambda g: labels[g])

    traj = get_traj_by_gid(trajs, selected_gid)
    if traj is None:
        return
    lookup = build_frame_lookup(traj)
    active_frames = sorted(lookup.keys())

    with c_frame:
        if active_frames:
            frame_id = st.slider("Frame", active_frames[0], active_frames[-1],
                                 active_frames[0], key="surv_frame")
        else:
            frame_id = 0

    # Status bar
    active_cams = lookup.get(frame_id, {})
    pills = "".join(
        f'<span class="cam-pill {"cam-on" if cam in active_cams else "cam-off"}">{cam}</span>'
        for cam in cameras
    )
    st.markdown(f'<div class="info-card">**ID {selected_gid}** · {traj["class_name"]} · '
                f'Frame {frame_id} · {len(active_cams)}/{len(cameras)} active {pills}</div>',
                unsafe_allow_html=True)

    # Camera grid
    n_cols = min(4, len(cameras))
    grid = st.columns(n_cols)
    for i, cam in enumerate(cameras):
        with grid[i % n_cols]:
            fd = active_cams.get(cam)
            img = render_surveillance_frame(stage0_dir, cam, frame_id,
                                            bbox=fd["bbox"] if fd else None,
                                            dim=(fd is None))
            if img is not None:
                st.image(img, use_container_width=True)
            else:
                st.caption(f"{cam}: no frame")

    # Crops of active cameras
    if active_cams:
        st.markdown('<div class="section-hd">Active Target Crops</div>', unsafe_allow_html=True)
        crop_cols = st.columns(len(active_cams))
        for j, (cam, fd) in enumerate(sorted(active_cams.items())):
            with crop_cols[j]:
                crop = get_crop(stage0_dir, cam, frame_id, fd["bbox"])
                if crop:
                    st.image(crop, caption=f"{cam} ({fd['confidence']:.2f})",
                             use_container_width=True)


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ IDENTITY INSPECTOR                                                      │
# └─────────────────────────────────────────────────────────────────────────┘

def page_inspector():
    st.markdown("## Identity Inspector")
    data = get_active_ds()
    if data is None:
        st.info("Select a dataset.")
        return

    trajs = data["trajectories"]
    stage0_dir = data["stage0_dir"]
    ds = data["ds"]
    if not trajs:
        return

    gid_opts = sorted(t["global_id"] for t in trajs)
    default = 0
    if st.session_state.get("selected_gid") in gid_opts:
        default = gid_opts.index(st.session_state.selected_gid)

    gid = st.selectbox("Identity", gid_opts, index=default, key=f"insp_gid_{ds}")
    traj = get_traj_by_gid(trajs, gid)
    if traj is None:
        return

    # Header card
    route = " → ".join(traj["camera_sequence"])
    st.markdown(
        f'<div class="info-card"><b style="font-size:1.15rem">ID {gid}</b> · '
        f'{traj["class_name"]} · {traj["num_cameras"]} cams · '
        f'{traj["total_duration"]:.1f}s · Route: {route}</div>',
        unsafe_allow_html=True,
    )

    # ── Appearance strip ──────────────────────────────────────────────
    st.markdown('<div class="section-hd">Appearance Across Cameras</div>', unsafe_allow_html=True)
    sorted_tks = sorted(traj["tracklets"], key=lambda tk: tk["start_time"])
    n = len(sorted_tks)
    cols = st.columns(min(n, 8))
    for i, tk in enumerate(sorted_tks):
        with cols[i % min(n, 8)]:
            crop = get_representative_crop(stage0_dir, tk)
            if crop:
                st.image(crop, use_container_width=True)
            st.caption(f"{tk['camera_id']} · trk {tk['track_id']}\n"
                       f"{tk['start_time']:.1f}–{tk['end_time']:.1f}s\n"
                       f"{tk['num_frames']}f · {tk['mean_confidence']:.2f}")

    # ── Timeline bar chart ────────────────────────────────────────────
    st.markdown('<div class="section-hd">Timeline</div>', unsafe_allow_html=True)
    cam_colors = {}
    fig = go.Figure()
    shown = set()
    for tk in sorted_tks:
        cam = tk["camera_id"]
        if cam not in cam_colors:
            cam_colors[cam] = ID_PALETTE[len(cam_colors) % len(ID_PALETTE)]
        fig.add_trace(go.Bar(
            x=[max(tk["end_time"] - tk["start_time"], 0.3)],
            y=[f"ID {gid}"], base=[tk["start_time"]], orientation="h",
            marker_color=cam_colors[cam],
            name=cam, legendgroup=cam, showlegend=(cam not in shown),
            hovertemplate=(f"{cam} trk {tk['track_id']}<br>"
                           f"{tk['start_time']:.1f}–{tk['end_time']:.1f}s<br>"
                           f"{tk['num_frames']} frames<extra></extra>"),
        ))
        shown.add(cam)
    fig.update_layout(barmode="stack", height=100, xaxis_title="Time (s)",
                      margin=dict(t=5, b=35, l=60, r=10))
    st.plotly_chart(fig, use_container_width=True)

    # ── Tracklet table ────────────────────────────────────────────────
    with st.expander("Tracklet Details"):
        rows = [{
            "Camera": tk["camera_id"], "Track": tk["track_id"],
            "Start": f'{tk["start_time"]:.1f}', "End": f'{tk["end_time"]:.1f}',
            "Dur": f'{tk["end_time"]-tk["start_time"]:.1f}',
            "Frames": tk["num_frames"], "Conf": f'{tk["mean_confidence"]:.3f}',
        } for tk in sorted_tks]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ CORRECTIONS                                                             │
# └─────────────────────────────────────────────────────────────────────────┘

def page_corrections():
    st.markdown("## Identity Corrections")
    data = get_active_ds()
    if data is None:
        st.info("Select a dataset.")
        return

    trajs = data["trajectories"]
    stage0_dir = data["stage0_dir"]
    store: CorrectionsStore = data["corrections"]
    ds = data["ds"]

    if not trajs:
        st.warning("No data.")
        return

    tab_reassign, tab_merge = st.tabs(["Reassign Tracklet", "Merge Identities"])

    gid_opts = sorted(t["global_id"] for t in trajs)

    # ── Reassign ──────────────────────────────────────────────────────
    with tab_reassign:
        st.caption("Move a wrongly-assigned tracklet to a different identity.")
        src_gid = st.selectbox("Source identity", gid_opts, key=f"c_src_{ds}")
        src = get_traj_by_gid(trajs, src_gid)
        if src:
            tk_labels = [
                f"{tk['camera_id']}/trk_{tk['track_id']} ({tk['num_frames']}f, "
                f"{tk['start_time']:.1f}–{tk['end_time']:.1f}s)"
                for tk in src["tracklets"]
            ]
            sel_label = st.selectbox("Tracklet", tk_labels, key=f"c_tk_{ds}")
            sel_idx = tk_labels.index(sel_label) if sel_label else 0
            sel_tk = src["tracklets"][sel_idx]

            col_crop, col_tgt = st.columns([1, 2])
            with col_crop:
                crop = get_representative_crop(stage0_dir, sel_tk)
                if crop:
                    st.image(crop, width=180)
            with col_tgt:
                tgt_opts = ["New Identity"] + [
                    f"ID {g} ({next(t['class_name'] for t in trajs if t['global_id']==g)})"
                    for g in gid_opts if g != src_gid
                ]
                tgt = st.selectbox("Reassign to", tgt_opts, key=f"c_tgt_{ds}")
                reason = st.text_input("Reason", key=f"c_rsn_{ds}")
                if st.button("Apply", type="primary", key=f"c_apply_{ds}"):
                    new_gid = (max(t["global_id"] for t in trajs) + 1
                               if tgt == "New Identity"
                               else int(tgt.split()[1]))
                    _do_reassign(trajs, src_gid, new_gid, sel_tk)
                    store.log_reassign(sel_tk["camera_id"], sel_tk["track_id"],
                                       src_gid, new_gid, reason)
                    _save_corrected(trajs, data["run_dir"])
                    load_trajectories.clear()
                    st.success(f"Tracklet → ID {new_gid}")
                    st.rerun()

    # ── Merge ─────────────────────────────────────────────────────────
    with tab_merge:
        st.caption("Merge two identities that are the same entity.")
        ca, cb = st.columns(2)
        with ca:
            gid_a = st.selectbox("Identity A", gid_opts, key=f"m_a_{ds}")
            ta = get_traj_by_gid(trajs, gid_a)
            if ta:
                crop_a = get_representative_crop(
                    stage0_dir, max(ta["tracklets"], key=lambda tk: tk["num_frames"]))
                if crop_a:
                    st.image(crop_a, width=160)
                st.caption(f"{ta['num_cameras']} cams · {ta['total_duration']:.1f}s")
        with cb:
            gid_b = st.selectbox("Identity B", [g for g in gid_opts if g != gid_a], key=f"m_b_{ds}")
            tb = get_traj_by_gid(trajs, gid_b)
            if tb:
                crop_b = get_representative_crop(
                    stage0_dir, max(tb["tracklets"], key=lambda tk: tk["num_frames"]))
                if crop_b:
                    st.image(crop_b, width=160)
                st.caption(f"{tb['num_cameras']} cams · {tb['total_duration']:.1f}s")

        reason = st.text_input("Reason", key=f"m_rsn_{ds}")
        if st.button("Merge", type="primary", key=f"m_apply_{ds}"):
            if gid_a == gid_b:
                st.error("Same identity.")
            else:
                _do_merge(trajs, gid_a, gid_b)
                store.log_merge(gid_a, gid_b, gid_a, reason)
                _save_corrected(trajs, data["run_dir"])
                load_trajectories.clear()
                st.success(f"ID {gid_b} → ID {gid_a}")
                st.rerun()


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ AUDIT LOG                                                               │
# └─────────────────────────────────────────────────────────────────────────┘

def page_audit():
    st.markdown("## Audit Log")
    data = get_active_ds()
    if data is None:
        st.info("Select a dataset.")
        return

    store: CorrectionsStore = data["corrections"]
    corrections = store.get_all(include_undone=True)
    active = [c for c in corrections if not c["undone"]]

    c1, c2, c3 = st.columns(3)
    c1.metric("Active", len(active))
    c2.metric("Reassigns", sum(1 for c in active if c["action"] == "reassign"))
    c3.metric("Merges", sum(1 for c in active if c["action"] == "merge"))

    if active and st.button("Undo Last"):
        undone = store.undo_last()
        if undone:
            st.warning(f'Undone: {undone["action"]} (ID {undone["source_gid"]} → {undone["target_gid"]})')
        st.rerun()

    if corrections:
        df = pd.DataFrame(corrections)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
        df["status"] = df.get("undone", pd.Series(dtype=int)).map({0: "Active", 1: "Undone"})
        show = [c for c in ["id", "timestamp", "action", "source_gid", "target_gid",
                             "tracklet_camera", "reason", "status"] if c in df.columns]
        st.dataframe(df[show], use_container_width=True, hide_index=True)
    else:
        st.info("No corrections recorded.")


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ DATASET COMPARISON                                                      │
# └─────────────────────────────────────────────────────────────────────────┘

def page_comparison():
    st.markdown("## Dataset Comparison")
    loaded = {k.removeprefix("ds_"): v for k, v in st.session_state.items()
              if k.startswith("ds_") and isinstance(v, dict)}
    if len(loaded) < 2:
        st.info("Load 2+ datasets (sidebar) to compare them side by side.")
        return

    # Summary table
    rows = []
    for ds, entry in loaded.items():
        trajs = entry["trajectories"]
        ev = entry.get("eval")
        row = {"Dataset": ds.upper(), "Identities": len(trajs),
               "Tracklets": sum(len(t["tracklets"]) for t in trajs),
               "Multi-Cam": sum(1 for t in trajs if t["num_cameras"] > 1),
               "Cameras": len({tk["camera_id"] for t in trajs for tk in t["tracklets"]})}
        if trajs:
            row["Avg Dur (s)"] = round(np.mean([t["total_duration"] for t in trajs]), 1)
        if ev:
            row["MOTA"] = f'{ev.get("mota", 0):.3f}'
            row["IDF1"] = f'{ev.get("idf1", 0):.3f}'
            row["HOTA"] = f'{ev.get("hota", 0):.3f}'
            row["IDSW"] = ev.get("id_switches", 0)
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Side-by-side metric chart
    metric_rows = []
    for ds, entry in loaded.items():
        ev = entry.get("eval")
        if ev and ev.get("idf1"):
            metric_rows.append({"Dataset": ds.upper(), "MOTA": ev["mota"],
                                "IDF1": ev["idf1"], "HOTA": ev.get("hota", 0)})
    if metric_rows:
        import plotly.express as px
        mdf = pd.DataFrame(metric_rows).melt(id_vars="Dataset", var_name="Metric", value_name="Score")
        ds_colors = {ds.upper(): DATASET_COLORS.get(ds, "#6b7280") for ds in loaded}
        fig = px.bar(mdf, x="Metric", y="Score", color="Dataset", barmode="group",
                     color_discrete_map=ds_colors)
        fig.update_layout(height=320, margin=dict(t=10, b=20))
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    # Camera-count distribution side by side
    st.markdown("### Cameras-per-Identity Distribution")
    cols = st.columns(len(loaded))
    for idx, (ds, entry) in enumerate(loaded.items()):
        with cols[idx]:
            trajs = entry["trajectories"]
            if not trajs:
                st.caption(f"{ds}: no data")
                continue
            cam_counts = [t["num_cameras"] for t in trajs]
            import plotly.express as px
            fig = px.histogram(x=cam_counts, nbins=max(cam_counts) if cam_counts else 5,
                               labels={"x": "Cameras", "y": "Count"},
                               color_discrete_sequence=[DATASET_COLORS.get(ds, "#6b7280")])
            fig.update_layout(height=220, margin=dict(t=10, b=30, l=30, r=10),
                              showlegend=False, title=ds.upper())
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Correction Helpers (unchanged logic)
# ---------------------------------------------------------------------------

def _do_reassign(trajs: List[dict], source_gid: int, target_gid: int, tracklet: dict):
    source = get_traj_by_gid(trajs, source_gid)
    source["tracklets"] = [
        tk for tk in source["tracklets"]
        if not (tk["camera_id"] == tracklet["camera_id"] and tk["track_id"] == tracklet["track_id"])
    ]
    if not source["tracklets"]:
        trajs[:] = [t for t in trajs if t["global_id"] != source_gid]

    target = get_traj_by_gid(trajs, target_gid)
    if target is None:
        trajs.append({
            "global_id": target_gid,
            "class_name": tracklet["class_name"],
            "num_cameras": 1,
            "total_duration": tracklet["end_time"] - tracklet["start_time"],
            "time_span": (tracklet["start_time"], tracklet["end_time"]),
            "camera_sequence": [tracklet["camera_id"]],
            "tracklets": [tracklet],
        })
    else:
        target["tracklets"].append(tracklet)
        _refresh_traj_meta(target)


def _do_merge(trajs: List[dict], gid_a: int, gid_b: int):
    traj_a = get_traj_by_gid(trajs, gid_a)
    traj_b = get_traj_by_gid(trajs, gid_b)
    if traj_a and traj_b:
        traj_a["tracklets"].extend(traj_b["tracklets"])
        _refresh_traj_meta(traj_a)
        trajs[:] = [t for t in trajs if t["global_id"] != gid_b]


def _refresh_traj_meta(traj: dict):
    tks = traj["tracklets"]
    cams = []
    for tk in sorted(tks, key=lambda x: x["start_time"]):
        if tk["camera_id"] not in cams:
            cams.append(tk["camera_id"])
    traj["camera_sequence"] = cams
    traj["num_cameras"] = len(set(cams))
    starts = [tk["start_time"] for tk in tks]
    ends = [tk["end_time"] for tk in tks]
    traj["time_span"] = (min(starts), max(ends))
    traj["total_duration"] = max(ends) - min(starts)


def _save_corrected(trajs: List[dict], run_dir: str):
    from src.core.data_models import GlobalTrajectory, Tracklet, TrackletFrame

    gt_list = []
    for t in trajs:
        tracklets = []
        for tk in t["tracklets"]:
            frames = [
                TrackletFrame(
                    frame_id=f["frame_id"], timestamp=f["timestamp"],
                    bbox=tuple(f["bbox"]), confidence=f["confidence"],
                )
                for f in tk["frames"]
            ]
            tracklets.append(Tracklet(
                track_id=tk["track_id"], camera_id=tk["camera_id"],
                class_id=tk["class_id"], class_name=tk["class_name"],
                frames=frames,
            ))
        gt_list.append(GlobalTrajectory(global_id=t["global_id"], tracklets=tracklets))

    out_path = Path(run_dir) / "stage4" / "corrected_trajectories.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_global_trajectories(gt_list, out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="MTMC Forensic Tracker",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(COMPACT_CSS, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────
    st.sidebar.markdown("### 🔍 MTMC Forensic Tracker")

    # Legacy --run-dir support
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None)
    args, _ = parser.parse_known_args()

    runs_by_ds = discover_runs()
    all_ds = sorted(runs_by_ds.keys())

    # If --run-dir was given, load it directly
    if args.run_dir:
        rd = Path(args.run_dir)
        nf = rd / "dataset_name.txt"
        ds = nf.read_text().strip() if nf.exists() else rd.name
        _load_run_into_state(rd, ds)
        st.session_state["active_ds"] = ds

    elif all_ds:
        # Dataset multi-selector
        st.sidebar.markdown("**Datasets**")
        selected_ds = st.sidebar.multiselect("Load", all_ds, default=all_ds[:2],
                                              label_visibility="collapsed")
        # Per-dataset run picker + load
        for ds in selected_ds:
            runs = runs_by_ds.get(ds, [])
            if not runs:
                continue
            labels = [r.name for r in runs]
            choice = st.sidebar.selectbox(f"{DATASET_ICONS.get(ds, '📁')} {ds}",
                                          labels, key=f"pick_{ds}")
            run_dir = runs[labels.index(choice)]
            if f"ds_{ds}" not in st.session_state:
                _load_run_into_state(run_dir, ds)

        # Unload deselected
        for k in list(st.session_state.keys()):
            if k.startswith("ds_") and k.removeprefix("ds_") not in selected_ds:
                del st.session_state[k]

        # Active dataset for single-dataset pages
        if selected_ds:
            if len(selected_ds) == 1:
                st.session_state["active_ds"] = selected_ds[0]
            else:
                active = st.sidebar.radio(
                    "Active dataset", selected_ds,
                    format_func=lambda d: f"{DATASET_ICONS.get(d, '')} {d.upper()}",
                    key="active_ds_radio",
                )
                st.session_state["active_ds"] = active
    else:
        st.sidebar.warning("No runs found in data/outputs/")
        rd_input = st.sidebar.text_input("Run directory", "data/outputs")
        rd = Path(rd_input)
        if rd.exists() and (rd / "stage4").exists():
            _load_run_into_state(rd, rd.name)
            st.session_state["active_ds"] = rd.name

    # Navigation
    st.sidebar.markdown("---")
    nav_to = st.session_state.pop("nav_to", None)
    pages = ["Overview", "Identity Browser", "Timeline",
             "Surveillance", "Inspector", "Corrections", "Audit Log", "Comparison"]
    default_idx = pages.index(nav_to) if nav_to and nav_to in pages else 0
    page = st.sidebar.radio("Navigation", pages, index=default_idx, label_visibility="collapsed")

    # Footer
    active = st.session_state.get("active_ds", "")
    loaded_count = sum(1 for k in st.session_state if k.startswith("ds_"))
    st.sidebar.caption(f"Active: {active} · {loaded_count} dataset(s) loaded")

    # ── Page Router ───────────────────────────────────────────────────
    if page == "Overview":
        page_overview()
    elif page == "Identity Browser":
        page_gallery()
    elif page == "Timeline":
        page_timeline()
    elif page == "Surveillance":
        page_surveillance()
    elif page == "Inspector":
        page_inspector()
    elif page == "Corrections":
        page_corrections()
    elif page == "Audit Log":
        page_audit()
    elif page == "Comparison":
        page_comparison()


if __name__ == "__main__":
    main()
