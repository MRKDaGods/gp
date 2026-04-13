"""Streamlit web dashboard for the MTMC tracking system.

Multi-page app with: Overview, Search, Video Playback, NL Query, 3D View, Evaluation.

Run: streamlit run src/apps/web_dashboard.py -- --run-dir data/outputs/run_XXXX
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import streamlit as st

# Ensure project root is on path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.io_utils import (
    load_evaluation_result,
    load_global_trajectories,
    load_tracklets_by_camera,
)


def _tracklet_num_frames(tracklet: Any) -> int:
    if isinstance(tracklet, dict):
        if tracklet.get("num_frames") is not None:
            return int(tracklet["num_frames"])
        return len(tracklet.get("frames") or [])

    num_frames = getattr(tracklet, "num_frames", None)
    if num_frames is not None:
        return int(num_frames)
    return len(getattr(tracklet, "frames", None) or [])


def _tracklet_start_time(tracklet: Any) -> float:
    if isinstance(tracklet, dict):
        if tracklet.get("start_time") is not None:
            return float(tracklet["start_time"])

        frames = tracklet.get("frames") or []
        if not frames:
            return 0.0

        first_frame = frames[0]
        if isinstance(first_frame, dict):
            return float(first_frame.get("timestamp", 0.0))
        return float(getattr(first_frame, "timestamp", 0.0))

    return float(getattr(tracklet, "start_time", 0.0))


def _tracklet_camera_id(tracklet: Any) -> str:
    if isinstance(tracklet, dict):
        return str(tracklet.get("camera_id", ""))
    return str(getattr(tracklet, "camera_id", ""))


def _tracklet_to_crop_payload(tracklet: Any) -> dict[str, Any] | None:
    camera_id = _tracklet_camera_id(tracklet)
    if not camera_id:
        return None

    if isinstance(tracklet, dict):
        frames = tracklet.get("frames") or []
    else:
        frames = getattr(tracklet, "frames", None) or []

    if not frames:
        return None

    payload_frames = []
    for frame in frames:
        if isinstance(frame, dict):
            frame_id = frame.get("frame_id")
            bbox = frame.get("bbox")
        else:
            frame_id = getattr(frame, "frame_id", None)
            bbox = getattr(frame, "bbox", None)

        if frame_id is None or bbox is None:
            continue

        payload_frames.append({
            "frame_id": int(frame_id),
            "bbox": list(bbox),
        })

    if not payload_frames:
        return None

    return {
        "camera_id": camera_id,
        "frames": payload_frames,
    }


def _get_tracklet_representative_crop(stage0_dir: Path, tracklet: Any):
    tracklet_payload = _tracklet_to_crop_payload(tracklet)
    if tracklet_payload is None:
        return None

    from src.apps.forensic_dashboard import get_representative_crop

    return get_representative_crop(str(stage0_dir), tracklet_payload)


def _build_camera_thumbnails(stage0_dir: Path, tracklets: list[Any]) -> list[tuple[str, Any]]:
    best_by_camera: dict[str, Any] = {}
    for tracklet in tracklets:
        camera_id = _tracklet_camera_id(tracklet)
        if not camera_id:
            continue

        best_tracklet = best_by_camera.get(camera_id)
        if best_tracklet is None or _tracklet_num_frames(tracklet) > _tracklet_num_frames(best_tracklet):
            best_by_camera[camera_id] = tracklet

    thumbnails = []
    for camera_id, tracklet in sorted(
        best_by_camera.items(),
        key=lambda item: _tracklet_start_time(item[1]),
    ):
        crop = _get_tracklet_representative_crop(stage0_dir, tracklet)
        if crop is not None:
            thumbnails.append((camera_id, crop))

    return thumbnails


def get_run_dir() -> Path:
    """Get the run directory from CLI args or sidebar input."""
    # Try CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None)
    args, _ = parser.parse_known_args()

    if args.run_dir:
        return Path(args.run_dir)

    # Fallback to sidebar input
    run_dir = st.sidebar.text_input(
        "Run directory",
        value="data/outputs",
        help="Path to a pipeline run output directory.",
    )
    return Path(run_dir)


def main():
    st.set_page_config(
        page_title="MTMC Tracker Dashboard",
        page_icon="📹",
        layout="wide",
    )

    st.sidebar.title("MTMC Tracker")
    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Search & Browse", "NL Query", "3D Visualization", "Evaluation"],
    )

    run_dir = get_run_dir()

    if page == "Overview":
        page_overview(run_dir)
    elif page == "Search & Browse":
        page_search(run_dir)
    elif page == "NL Query":
        page_nl_query(run_dir)
    elif page == "3D Visualization":
        page_3d_view(run_dir)
    elif page == "Evaluation":
        page_evaluation(run_dir)


# ---- Page: Overview ----

def page_overview(run_dir: Path):
    st.title("System Overview")

    # Load trajectories
    traj_path = run_dir / "stage4" / "global_trajectories.json"
    if not traj_path.exists():
        st.warning(f"No trajectory data found at {traj_path}. Run the pipeline first.")
        st.info("Expected directory structure: `<run_dir>/stage4/global_trajectories.json`")
        return

    trajectories = load_global_trajectories(traj_path)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Global Identities", len(trajectories))
    col2.metric("Total Tracklets", sum(len(t.tracklets) for t in trajectories))
    col3.metric(
        "Multi-Camera",
        sum(1 for t in trajectories if t.num_cameras > 1),
    )

    all_cameras = set()
    for t in trajectories:
        for tk in t.tracklets:
            all_cameras.add(tk.camera_id)
    col4.metric("Cameras", len(all_cameras))

    # Class distribution
    st.subheader("Object Class Distribution")
    class_counts = {}
    for t in trajectories:
        cls = t.class_name
        class_counts[cls] = class_counts.get(cls, 0) + 1

    import plotly.express as px

    if class_counts:
        fig = px.pie(
            names=list(class_counts.keys()),
            values=list(class_counts.values()),
            title="Trajectories by Class",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Duration histogram
    st.subheader("Trajectory Duration Distribution")
    durations = [t.total_duration for t in trajectories]
    if durations:
        fig = px.histogram(x=durations, nbins=30, labels={"x": "Duration (s)"})
        st.plotly_chart(fig, use_container_width=True)


# ---- Page: Search & Browse ----

def page_search(run_dir: Path):
    st.title("Search & Browse Trajectories")

    traj_path = run_dir / "stage4" / "global_trajectories.json"
    stage0_dir = run_dir / "stage0"
    has_stage0_frames = stage0_dir.exists()

    if not traj_path.exists():
        st.warning("No trajectory data found. Run the pipeline first.")
        return

    trajectories = load_global_trajectories(traj_path)

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_class = st.selectbox("Class", ["All"] + sorted(set(t.class_name for t in trajectories)))
    with col2:
        all_cameras = sorted(set(tk.camera_id for t in trajectories for tk in t.tracklets))
        filter_camera = st.selectbox("Camera", ["All"] + all_cameras)
    with col3:
        min_cameras = st.slider("Min cameras", 1, 10, 1)

    # Apply filters
    filtered = trajectories
    if filter_class != "All":
        filtered = [t for t in filtered if t.class_name == filter_class]
    if filter_camera != "All":
        filtered = [t for t in filtered if any(tk.camera_id == filter_camera for tk in t.tracklets)]
    filtered = [t for t in filtered if t.num_cameras >= min_cameras]

    st.info(f"Showing {len(filtered)} of {len(trajectories)} trajectories")

    # Display results
    for traj in filtered[:50]:
        tracklets = sorted(traj.tracklets, key=_tracklet_start_time)
        representative_tracklet = None
        representative_crop = None
        camera_thumbnails = []

        if tracklets and has_stage0_frames:
            representative_tracklet = max(tracklets, key=_tracklet_num_frames)
            representative_crop = _get_tracklet_representative_crop(
                stage0_dir,
                representative_tracklet,
            )
            if len(tracklets) > 1:
                camera_thumbnails = _build_camera_thumbnails(stage0_dir, tracklets)

        with st.expander(
            f"ID {traj.global_id} | {traj.class_name} | "
            f"{traj.num_cameras} cameras | {traj.total_duration:.1f}s"
        ):
            image_col, info_col = st.columns([1, 3])

            with image_col:
                if representative_crop is not None:
                    st.image(
                        representative_crop,
                        caption=f"Representative crop ({_tracklet_camera_id(representative_tracklet)})",
                        use_container_width=True,
                    )

            with info_col:
                st.write(f"**Cameras visited**: {' → '.join(traj.camera_sequence)}")
                st.write(f"**Time span**: {traj.time_span[0]:.1f}s - {traj.time_span[1]:.1f}s")

                for tk in tracklets:
                    st.write(
                        f"  - Camera `{tk.camera_id}`: "
                        f"track {tk.track_id}, "
                        f"{tk.start_time:.1f}s - {tk.end_time:.1f}s, "
                        f"{tk.num_frames} frames"
                    )

            if len(camera_thumbnails) > 1:
                st.caption("Per-camera representative crops")
                thumb_cols = st.columns(len(camera_thumbnails))
                for thumb_col, (camera_id, crop) in zip(thumb_cols, camera_thumbnails):
                    with thumb_col:
                        st.image(crop, caption=camera_id, use_container_width=True)


# ---- Page: NL Query ----

def page_nl_query(run_dir: Path):
    st.title("Natural Language Query")

    traj_path = run_dir / "stage4" / "global_trajectories.json"
    stage0_dir = run_dir / "stage0"
    has_stage0_frames = stage0_dir.exists()

    if not traj_path.exists():
        st.warning("No trajectory data found. Run the pipeline first.")
        return

    trajectories = load_global_trajectories(traj_path)

    query = st.text_input(
        "Search query",
        placeholder='e.g. "car seen on camera 1 and camera 3"',
    )

    if query:
        with st.spinner("Searching..."):
            from src.apps.nl_query import NLQueryEngine

            # Cache the engine in session state
            if "nl_engine" not in st.session_state:
                engine = NLQueryEngine()
                engine.build_index(trajectories)
                st.session_state.nl_engine = engine
            else:
                engine = st.session_state.nl_engine

            results = engine.query(query, top_k=10)

        st.subheader(f"Results for: \"{query}\"")
        for traj, score, desc in results:
            tracklets = sorted(traj.tracklets, key=_tracklet_start_time)
            representative_tracklet = None
            representative_crop = None
            camera_thumbnails = []

            if tracklets and has_stage0_frames:
                representative_tracklet = max(tracklets, key=_tracklet_num_frames)
                representative_crop = _get_tracklet_representative_crop(
                    stage0_dir,
                    representative_tracklet,
                )
                if len(tracklets) > 1:
                    camera_thumbnails = _build_camera_thumbnails(stage0_dir, tracklets)

            with st.expander(
                f"Score: {score:.3f} | ID {traj.global_id} | {traj.class_name} | "
                f"{traj.num_cameras} cameras"
            ):
                image_col, info_col = st.columns([1, 3])

                with image_col:
                    if representative_crop is not None:
                        st.image(
                            representative_crop,
                            caption=f"Representative crop ({_tracklet_camera_id(representative_tracklet)})",
                            use_container_width=True,
                        )

                with info_col:
                    st.write(f"**Description**: {desc}")
                    st.write(f"**Cameras**: {' → '.join(traj.camera_sequence)}")
                    st.write(f"**Duration**: {traj.total_duration:.1f}s")

                if len(camera_thumbnails) > 1:
                    st.caption("Per-camera representative crops")
                    thumb_cols = st.columns(len(camera_thumbnails))
                    for thumb_col, (camera_id, crop) in zip(thumb_cols, camera_thumbnails):
                        with thumb_col:
                            st.image(crop, caption=camera_id, use_container_width=True)


# ---- Page: 3D Visualization ----

def page_3d_view(run_dir: Path):
    st.title("3D Trajectory Visualization")

    traj_path = run_dir / "stage4" / "global_trajectories.json"
    if not traj_path.exists():
        st.warning("No trajectory data found. Run the pipeline first.")
        return

    trajectories = load_global_trajectories(traj_path)

    max_traj = st.slider("Max trajectories to display", 5, 100, 30)
    multi_only = st.checkbox("Multi-camera only", value=True)

    if multi_only:
        trajectories = [t for t in trajectories if t.num_cameras > 1]

    from src.apps.simulation_3d import Simulator3D

    sim = Simulator3D()
    fig = sim.render(trajectories, max_trajectories=max_traj)
    st.plotly_chart(fig, use_container_width=True)


# ---- Page: Evaluation ----

def page_evaluation(run_dir: Path):
    st.title("Evaluation Results")

    eval_path = run_dir / "stage5" / "evaluation_report.json"
    if not eval_path.exists():
        st.warning("No evaluation results found. Run stage 5 first.")
        return

    result = load_evaluation_result(eval_path)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MOTA", f"{result.mota:.4f}")
    col2.metric("IDF1", f"{result.idf1:.4f}")
    col3.metric("HOTA", f"{result.hota:.4f}")
    col4.metric("ID Switches", result.id_switches)

    col5, col6 = st.columns(2)
    col5.metric("GT Identities", result.num_gt_ids)
    col6.metric("Predicted Identities", result.num_pred_ids)

    if result.details:
        st.subheader("Detailed Metrics")
        import pandas as pd

        df = pd.DataFrame([result.details]).T
        df.columns = ["Value"]
        st.dataframe(df)

    # Check for ablation results
    ablation_path = run_dir / "stage5" / "ablation_results.csv"
    if ablation_path.exists():
        st.subheader("Ablation Study Results")
        import pandas as pd

        df = pd.read_csv(ablation_path, index_col=0)
        st.dataframe(df.style.highlight_max(axis=0, subset=["mota", "idf1", "hota"]))

        import plotly.express as px

        for metric in ["mota", "idf1", "hota"]:
            if metric in df.columns:
                fig = px.bar(df, y=metric, title=f"{metric.upper()} by Variant")
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
