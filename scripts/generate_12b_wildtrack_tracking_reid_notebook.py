from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks" / "kaggle" / "12b_wildtrack_tracking_reid"
NOTEBOOK_PATH = NOTEBOOK_DIR / "12b_wildtrack_tracking_reid.ipynb"
METADATA_PATH = NOTEBOOK_DIR / "kernel-metadata.json"


def to_source(text: str) -> list[str]:
    cleaned = dedent(text).strip("\n")
    if not cleaned:
        return []
    lines = cleaned.splitlines()
    return [f"{line}\n" for line in lines[:-1]] + [lines[-1]]


def markdown_cell(cell_id: str, text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {"language": "markdown"},
        "source": to_source(text),
    }


def code_cell(cell_id: str, text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {"language": "python"},
        "outputs": [],
        "source": to_source(text),
    }


def build_cells() -> list[dict]:
    return [
        markdown_cell(
            "12b-title",
            """
            # 12b WILDTRACK: MVDeTr Tracking + Person ReID Pipeline

            **Pipeline**
            1. Load MVDeTr ground-plane detections from 12a (`test.txt`)
            2. Parse grid coordinates into world coordinates in centimeters
            3. Run Hungarian temporal tracking on the ground plane
            4. Project tracked positions to 7 camera views and save per-camera tracklets
            5. Extract person ReID embeddings from projected camera crops
            6. Score simple cross-camera merge candidates with cosine similarity
            7. Evaluate WILDTRACK ground-plane metrics (MODA, MODP, IDF1, precision, recall)
            8. Optionally sweep tracking hyperparameters

            **Inputs**
            - 12a kernel output: `ali369/12a-wildtrack-mvdetr-training`
            - WILDTRACK dataset: `aryashah2k/large-scale-multicamera-detection-dataset`
            - Repo snapshot: `mrkdagods/mtmc-gp`
            - ReID weights: `mrkdagods/mtmc-weights`

            **Outputs**
            - `/kaggle/working/12b_output/ground_plane_tracks.json`
            - `/kaggle/working/12b_output/tracklets/`
            - `/kaggle/working/12b_output/global_trajectories.json`
            - `/kaggle/working/12b_output/reid_features.npz`
            - `/kaggle/working/12b_output/reid_merge_candidates.json`
            - `/kaggle/working/12b_output/evaluation_summary.json`
            - `/kaggle/working/12b_output/tracking_sweep_best.json`
            - `/kaggle/working/debug.log`
            """,
        ),
        code_cell(
            "12b-logger",
            """
            import atexit
            import pathlib
            import sys

            _LOG_PATH = pathlib.Path("/kaggle/working/debug.log")
            _LOG = _LOG_PATH.open("w", buffering=1)

            class _TeeWriter:
                def __init__(self, original, log_file):
                    self.original = original
                    self.log_file = log_file

                def write(self, value):
                    self.original.write(value)
                    try:
                        self.log_file.write(value)
                    except Exception:
                        pass

                def flush(self):
                    self.original.flush()
                    try:
                        self.log_file.flush()
                    except Exception:
                        pass

            sys.stdout = _TeeWriter(sys.stdout, _LOG)
            sys.stderr = _TeeWriter(sys.stderr, _LOG)

            @atexit.register
            def _shutdown_log():
                try:
                    _LOG.close()
                except Exception:
                    pass

            print("[12b] debug logging initialized")
            print(f"[12b] writing log to {_LOG_PATH}")
            """,
        ),
        code_cell(
            "12b-bootstrap",
            """
            import json
            import re
            import shutil
            import subprocess
            import sys

            def pip_install(*args):
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

            gpu_info = []
            if shutil.which("nvidia-smi"):
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=gpu_name,compute_cap", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0:
                    for raw_line in result.stdout.splitlines():
                        line = raw_line.strip()
                        if not line:
                            continue
                        name, _, capability = line.partition(",")
                        match = re.search(r"(\\d+)\\.(\\d+)", capability)
                        if match:
                            sm = int(match.group(1)) * 10 + int(match.group(2))
                            gpu_info.append({"name": name.strip(), "capability": capability.strip(), "sm": sm})

            if any(entry["sm"] < 70 for entry in gpu_info):
                pip_install(
                    "torch==2.4.1",
                    "torchvision==0.19.1",
                    "--index-url",
                    "https://download.pytorch.org/whl/cu124",
                )

            pip_install("timm==0.9.16", "loguru", "omegaconf", "motmetrics", "scikit-learn", "scipy")

            import torch

            print(
                json.dumps(
                    {
                        "python": sys.version.split()[0],
                        "torch": torch.__version__,
                        "cuda_available": torch.cuda.is_available(),
                        "device_count": torch.cuda.device_count(),
                        "gpu_info": gpu_info,
                    },
                    indent=2,
                )
            )
            """,
        ),
        code_cell(
            "12b-repo",
            """
            import os
            import subprocess
            import sys
            from pathlib import Path

            WORK_DIR = Path("/kaggle/working")
            REPO_CANDIDATES = [
                Path("/kaggle/input/mtmc-gp/gp"),
                Path("/kaggle/input/mtmc-gp"),
            ]

            PROJECT = None
            for candidate in REPO_CANDIDATES:
                if (candidate / "pyproject.toml").exists() or (candidate / "setup.py").exists():
                    PROJECT = candidate
                    break

            if PROJECT is None:
                PROJECT = WORK_DIR / "gp"
                if not PROJECT.exists():
                    subprocess.check_call(
                        [
                            "git",
                            "clone",
                            "--depth",
                            "1",
                            "--branch",
                            "feature/people-tracking",
                            "https://github.com/MRKDaGods/gp.git",
                            str(PROJECT),
                        ]
                    )
                else:
                    subprocess.check_call(["git", "-C", str(PROJECT), "pull", "--ff-only"])

            os.chdir(PROJECT)
            sys.path.insert(0, str(PROJECT))
            print(f"[12b] project root: {PROJECT}")
            """,
        ),
        code_cell(
            "12b-deps-config",
            """
            import glob
            import os
            import subprocess
            import sys
            from pathlib import Path

            def pip_install(*args):
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

            requirements_path = PROJECT / "requirements.txt"
            assert requirements_path.exists(), requirements_path
            pip_install("-r", str(requirements_path))
            pip_install("-e", str(PROJECT), "--no-deps")

            MVDETR_OUTPUT_ROOTS = [
                Path("/kaggle/input/12a-wildtrack-mvdetr-training"),
                Path("/kaggle/input/12a-wildtrack-mvdetr-training/12a_wildtrack_mvdetr"),
            ]
            MVDETR_OUTPUT_DIR = None
            for candidate in MVDETR_OUTPUT_ROOTS:
                if candidate.exists():
                    MVDETR_OUTPUT_DIR = candidate
                    break
            assert MVDETR_OUTPUT_DIR is not None, "Cannot find 12a kernel output"

            artifact_hint = "aug_deform_trans_lr0.0005_baseR0.1_neck128_out0_alpha1.0_id0_drop0.0_dropcam0.0_worldRK4_10_imgRK12_10_2026-03-29_01-34-57"
            test_candidates = [
                MVDETR_OUTPUT_DIR / artifact_hint / "test.txt",
                MVDETR_OUTPUT_DIR / "test.txt",
                MVDETR_OUTPUT_DIR / "12a_wildtrack_mvdetr" / artifact_hint / "test.txt",
                MVDETR_OUTPUT_DIR / "12a_wildtrack_mvdetr" / "test.txt",
            ]
            test_candidates.extend(Path(path) for path in glob.glob(str(MVDETR_OUTPUT_DIR / "**" / "test.txt"), recursive=True))
            DETECTIONS_PATH = next((path for path in test_candidates if path.is_file()), None)
            assert DETECTIONS_PATH is not None, f"Cannot find test.txt under {MVDETR_OUTPUT_DIR}"
            print(f"[12b] detections path: {DETECTIONS_PATH}")

            WILDTRACK_CANDIDATES = [
                Path("/kaggle/input/large-scale-multicamera-detection-dataset/Wildtrack"),
                Path("/kaggle/input/large-scale-multicamera-detection-dataset"),
                Path("/kaggle/input/datasets/aryashah2k/large-scale-multicamera-detection-dataset/Wildtrack"),
                Path("/kaggle/input/datasets/aryashah2k/large-scale-multicamera-detection-dataset"),
            ]
            WILDTRACK_ROOT = None
            for candidate in WILDTRACK_CANDIDATES:
                if (candidate / "Image_subsets").exists():
                    WILDTRACK_ROOT = candidate
                    break
            assert WILDTRACK_ROOT is not None, "Cannot find WILDTRACK dataset"
            print(f"[12b] wildtrack root: {WILDTRACK_ROOT}")

            IMAGE_SUBSETS_DIR = WILDTRACK_ROOT / "Image_subsets"
            ANNOTATIONS_DIR = WILDTRACK_ROOT / "annotations_positions"
            CALIBRATIONS_DIR = WILDTRACK_ROOT / "calibrations"
            assert IMAGE_SUBSETS_DIR.is_dir(), IMAGE_SUBSETS_DIR
            assert ANNOTATIONS_DIR.is_dir(), ANNOTATIONS_DIR
            assert CALIBRATIONS_DIR.is_dir(), CALIBRATIONS_DIR

            REID_CANDIDATES = [
                Path("/kaggle/input/mtmc-weights/person_transreid_vit_base_market1501.pth"),
                Path("/kaggle/input/mtmc-weights/models/person_transreid_vit_base_market1501.pth"),
                Path("/kaggle/input/mtmc-weights/reid/person_transreid_vit_base_market1501.pth"),
                PROJECT / "models" / "reid" / "person_transreid_vit_base_market1501.pth",
            ]
            REID_WEIGHTS = next((path for path in REID_CANDIDATES if path.is_file()), None)
            if REID_WEIGHTS is None:
                print("[12b] WARNING: person ReID weights not found; feature extraction will be skipped")
            else:
                print(f"[12b] reid weights: {REID_WEIGHTS}")

            OUTPUT_DIR = WORK_DIR / "12b_output"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            MAX_MATCH_DISTANCE_CM = 75.0
            MAX_MISSED_FRAMES = 5
            MIN_TRACK_LENGTH = 3
            FPS = 2.0
            IMAGE_SIZE = (1920, 1080)
            MATCH_THRESHOLD_CM = 50.0
            NMS_RADIUS_CM = 50.0
            REID_INPUT_SIZE = (256, 128)
            REID_BATCH_SIZE = 32
            MAX_CROPS_PER_TRACKLET = 8
            REID_MERGE_THRESHOLD = 0.75

            runtime_config = {
                "detections_path": str(DETECTIONS_PATH),
                "wildtrack_root": str(WILDTRACK_ROOT),
                "reid_weights": str(REID_WEIGHTS) if REID_WEIGHTS is not None else None,
                "output_dir": str(OUTPUT_DIR),
                "max_match_distance_cm": MAX_MATCH_DISTANCE_CM,
                "max_missed_frames": MAX_MISSED_FRAMES,
                "min_track_length": MIN_TRACK_LENGTH,
                "fps": FPS,
                "image_size": IMAGE_SIZE,
            }
            print(json.dumps(runtime_config, indent=2))
            """,
        ),
        code_cell(
            "12b-load-detections",
            """
            from src.stage_wildtrack_mvdetr.pipeline import load_mvdetr_ground_plane_detections

            detections = load_mvdetr_ground_plane_detections(
                DETECTIONS_PATH,
                normalize_wildtrack_frames=True,
            )
            assert detections, "No detections loaded from test.txt"

            frame_ids = sorted({det.frame_id for det in detections})
            raw_frame_ids = sorted({det.raw_frame_id for det in detections if det.raw_frame_id is not None})
            per_frame = len(detections) / len(frame_ids)

            print(f"[12b] detections: {len(detections)}")
            print(f"[12b] normalized frames: {len(frame_ids)} ({frame_ids[0]} -> {frame_ids[-1]})")
            if raw_frame_ids:
                print(f"[12b] raw wildtrack frames: {raw_frame_ids[0]} -> {raw_frame_ids[-1]}")
            print(f"[12b] avg detections per frame: {per_frame:.2f}")
            """,
        ),
        code_cell(
            "12b-track-project",
            """
            from pathlib import Path

            from src.core.io_utils import save_global_trajectories, save_tracklets_by_camera
            from src.core.wildtrack_calibration import load_wildtrack_calibration
            from src.stage_wildtrack_mvdetr.pipeline import (
                _save_ground_plane_csv,
                _save_ground_plane_tracks,
                _tracks_to_projected_tracklets,
                track_ground_plane_detections,
            )

            tracks = track_ground_plane_detections(
                detections=detections,
                max_match_distance_cm=MAX_MATCH_DISTANCE_CM,
                max_missed_frames=MAX_MISSED_FRAMES,
                min_track_length=MIN_TRACK_LENGTH,
            )
            assert tracks, "Tracking produced zero valid tracks"

            track_lengths = sorted(len(track.detections) for track in tracks)
            median_length = track_lengths[len(track_lengths) // 2]
            print(f"[12b] tracks: {len(tracks)}")
            print(
                f"[12b] track lengths: min={track_lengths[0]}, max={track_lengths[-1]}, "
                f"mean={sum(track_lengths) / len(track_lengths):.2f}, median={median_length}"
            )

            _save_ground_plane_tracks(tracks, OUTPUT_DIR / "ground_plane_tracks.json")
            _save_ground_plane_csv(tracks, OUTPUT_DIR / "ground_plane_tracks.csv")

            calibrations = load_wildtrack_calibration(CALIBRATIONS_DIR)
            assert calibrations, f"No calibrations loaded from {CALIBRATIONS_DIR}"
            print(f"[12b] calibration cameras: {sorted(calibrations)}")

            tracklets_by_camera, trajectories = _tracks_to_projected_tracklets(
                tracks=tracks,
                calibrations=calibrations,
                fps=FPS,
                image_size=IMAGE_SIZE,
            )

            total_tracklets = sum(len(items) for items in tracklets_by_camera.values())
            print(f"[12b] projected tracklets: {total_tracklets}")
            print(f"[12b] global trajectories: {len(trajectories)}")
            for camera_id, camera_tracklets in sorted(tracklets_by_camera.items()):
                print(f"  {camera_id}: {len(camera_tracklets)} tracklets")

            save_tracklets_by_camera(tracklets_by_camera, OUTPUT_DIR / "tracklets")
            save_global_trajectories(trajectories, OUTPUT_DIR / "global_trajectories.json")
            """,
        ),
        code_cell(
            "12b-reid-features",
            """
            import json
            from pathlib import Path

            import cv2
            import numpy as np
            from sklearn.preprocessing import normalize

            from src.stage2_features.reid_model import ReIDModel

            def sample_tracklet_frames(tracklet, max_samples=8):
                if len(tracklet.frames) <= max_samples:
                    return tracklet.frames
                indices = np.linspace(0, len(tracklet.frames) - 1, num=max_samples, dtype=int)
                return [tracklet.frames[index] for index in indices]

            camera_dirs = {
                path.name: path
                for path in sorted(IMAGE_SUBSETS_DIR.iterdir())
                if path.is_dir() and path.name.startswith("C")
            }
            print(f"[12b] image cameras: {sorted(camera_dirs)}")

            REID_FEATURES = {}
            REID_FEATURE_STATS = {}
            if REID_WEIGHTS is None:
                print("[12b] skipping ReID feature extraction")
            else:
                reid_model = ReIDModel(
                    model_name="transreid",
                    weights_path=str(REID_WEIGHTS),
                    embedding_dim=768,
                    input_size=REID_INPUT_SIZE,
                    device="cuda:0" if __import__("torch").cuda.is_available() else "cpu",
                    half=True,
                    flip_augment=True,
                    num_cameras=7,
                    vit_model="vit_base_patch16_clip_224.openai",
                    clip_normalization=True,
                )

                for trajectory in trajectories:
                    crops = []
                    crop_camera_ids = []
                    for tracklet in trajectory.tracklets:
                        camera_dir = camera_dirs.get(tracklet.camera_id)
                        if camera_dir is None:
                            continue
                        sampled_frames = sample_tracklet_frames(tracklet, MAX_CROPS_PER_TRACKLET)
                        for frame in sampled_frames:
                            wildtrack_frame = frame.frame_id * 5
                            image_path = camera_dir / f"{wildtrack_frame:08d}.png"
                            if not image_path.is_file():
                                continue
                            image = cv2.imread(str(image_path))
                            if image is None:
                                continue
                            x1, y1, x2, y2 = [int(round(value)) for value in frame.bbox]
                            pad_x = max(4, int((x2 - x1) * 0.08))
                            pad_y = max(4, int((y2 - y1) * 0.08))
                            x1 = max(0, x1 - pad_x)
                            y1 = max(0, y1 - pad_y)
                            x2 = min(image.shape[1], x2 + pad_x)
                            y2 = min(image.shape[0], y2 + pad_y)
                            if x2 <= x1 or y2 <= y1:
                                continue
                            crop = image[y1:y2, x1:x2]
                            if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 8:
                                continue
                            crops.append(crop)
                            crop_camera_ids.append(int(tracklet.camera_id[1:]) - 1)

                    if not crops:
                        continue

                    per_camera_embeddings = []
                    unique_camera_ids = sorted(set(crop_camera_ids))
                    for cam_id in unique_camera_ids:
                        camera_crops = [crop for crop, crop_cam_id in zip(crops, crop_camera_ids) if crop_cam_id == cam_id]
                        if not camera_crops:
                            continue
                        embeddings = reid_model.extract_features(
                            camera_crops,
                            batch_size=REID_BATCH_SIZE,
                            cam_id=cam_id,
                        )
                        if embeddings.size == 0:
                            continue
                        per_camera_embeddings.append(embeddings.mean(axis=0))

                    if not per_camera_embeddings:
                        continue

                    feature = np.mean(np.stack(per_camera_embeddings, axis=0), axis=0)
                    feature = normalize(feature.reshape(1, -1), norm="l2")[0].astype(np.float32)
                    REID_FEATURES[trajectory.global_id] = feature
                    REID_FEATURE_STATS[trajectory.global_id] = {
                        "num_crops": len(crops),
                        "num_tracklets": len(trajectory.tracklets),
                        "num_cameras": len({tracklet.camera_id for tracklet in trajectory.tracklets}),
                    }

                print(f"[12b] extracted ReID features: {len(REID_FEATURES)} / {len(trajectories)} trajectories")

                if REID_FEATURES:
                    np.savez(
                        OUTPUT_DIR / "reid_features.npz",
                        **{str(global_id): feature for global_id, feature in REID_FEATURES.items()},
                    )
                    with (OUTPUT_DIR / "reid_feature_stats.json").open("w", encoding="utf-8") as handle:
                        json.dump(REID_FEATURE_STATS, handle, indent=2)
            """,
        ),
        code_cell(
            "12b-reid-association",
            """
            import json
            import math

            merge_candidates = []
            if REID_FEATURES:
                trajectory_by_id = {trajectory.global_id: trajectory for trajectory in trajectories}
                feature_items = sorted(REID_FEATURES.items())
                for index, (global_id_a, feature_a) in enumerate(feature_items):
                    traj_a = trajectory_by_id[global_id_a]
                    cameras_a = {tracklet.camera_id for tracklet in traj_a.tracklets}
                    for global_id_b, feature_b in feature_items[index + 1 :]:
                        traj_b = trajectory_by_id[global_id_b]
                        cameras_b = {tracklet.camera_id for tracklet in traj_b.tracklets}
                        shared_cameras = cameras_a & cameras_b
                        score = float(np.dot(feature_a, feature_b))
                        if score < REID_MERGE_THRESHOLD:
                            continue
                        time_a = traj_a.time_span
                        time_b = traj_b.time_span
                        temporal_overlap = max(0.0, min(time_a[1], time_b[1]) - max(time_a[0], time_b[0]))
                        merge_candidates.append(
                            {
                                "global_id_a": global_id_a,
                                "global_id_b": global_id_b,
                                "cosine_similarity": round(score, 6),
                                "shared_cameras": sorted(shared_cameras),
                                "temporal_overlap_s": round(temporal_overlap, 3),
                                "camera_count_a": len(cameras_a),
                                "camera_count_b": len(cameras_b),
                            }
                        )

                merge_candidates.sort(key=lambda item: item["cosine_similarity"], reverse=True)
                with (OUTPUT_DIR / "reid_merge_candidates.json").open("w", encoding="utf-8") as handle:
                    json.dump(merge_candidates, handle, indent=2)

            print(f"[12b] merge candidates above threshold {REID_MERGE_THRESHOLD}: {len(merge_candidates)}")
            for candidate in merge_candidates[:10]:
                print(candidate)
            """,
        ),
        code_cell(
            "12b-evaluate",
            """
            import json

            from src.stage5_evaluation.ground_plane_eval import evaluate_wildtrack_ground_plane

            eval_result = evaluate_wildtrack_ground_plane(
                trajectories=trajectories,
                annotations_dir=ANNOTATIONS_DIR,
                calibrations_dir=CALIBRATIONS_DIR,
                conf_threshold=0.25,
                match_threshold_cm=MATCH_THRESHOLD_CM,
                nms_radius_cm=NMS_RADIUS_CM,
            )

            precision = float(eval_result.details.get("precision", 0.0))
            recall = float(eval_result.details.get("recall", 0.0))
            modp_cm = float(eval_result.details.get("modp_cm", 0.0))

            print("=" * 60)
            print("WILDTRACK Ground-Plane Evaluation")
            print("=" * 60)
            print(f"MODA:        {eval_result.mota:.4f}")
            print(f"MODP (cm):   {modp_cm:.4f}")
            print(f"IDF1:        {eval_result.idf1:.4f}")
            print(f"Precision:   {precision:.4f}")
            print(f"Recall:      {recall:.4f}")
            print(f"ID Switches: {eval_result.id_switches}")

            evaluation_summary = {
                "moda": float(eval_result.mota),
                "modp_cm": modp_cm,
                "idf1": float(eval_result.idf1),
                "precision": precision,
                "recall": recall,
                "id_switches": int(eval_result.id_switches),
                "num_trajectories": len(trajectories),
                "num_tracklets": sum(len(items) for items in tracklets_by_camera.values()),
                "num_reid_features": len(REID_FEATURES) if REID_FEATURES else 0,
                "num_reid_merge_candidates": len(merge_candidates),
                "tracking_params": {
                    "max_match_distance_cm": MAX_MATCH_DISTANCE_CM,
                    "max_missed_frames": MAX_MISSED_FRAMES,
                    "min_track_length": MIN_TRACK_LENGTH,
                },
            }

            with (OUTPUT_DIR / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
                json.dump(evaluation_summary, handle, indent=2)
            """,
        ),
        code_cell(
            "12b-sweep",
            """
            import json

            best_result = None
            sweep_results = []
            sweep_space = {
                "max_dist": [50.0, 75.0, 100.0, 125.0],
                "max_missed": [3, 5, 8, 10],
                "min_len": [2, 3, 5],
            }

            for max_dist in sweep_space["max_dist"]:
                for max_missed in sweep_space["max_missed"]:
                    for min_len in sweep_space["min_len"]:
                        sweep_tracks = track_ground_plane_detections(
                            detections=detections,
                            max_match_distance_cm=max_dist,
                            max_missed_frames=max_missed,
                            min_track_length=min_len,
                        )
                        _, sweep_trajectories = _tracks_to_projected_tracklets(
                            tracks=sweep_tracks,
                            calibrations=calibrations,
                            fps=FPS,
                            image_size=IMAGE_SIZE,
                        )
                        sweep_eval = evaluate_wildtrack_ground_plane(
                            trajectories=sweep_trajectories,
                            annotations_dir=ANNOTATIONS_DIR,
                            calibrations_dir=CALIBRATIONS_DIR,
                            conf_threshold=0.25,
                            match_threshold_cm=MATCH_THRESHOLD_CM,
                            nms_radius_cm=NMS_RADIUS_CM,
                        )
                        record = {
                            "max_match_distance_cm": max_dist,
                            "max_missed_frames": max_missed,
                            "min_track_length": min_len,
                            "moda": float(sweep_eval.mota),
                            "modp_cm": float(sweep_eval.details.get("modp_cm", 0.0)),
                            "idf1": float(sweep_eval.idf1),
                            "precision": float(sweep_eval.details.get("precision", 0.0)),
                            "recall": float(sweep_eval.details.get("recall", 0.0)),
                            "id_switches": int(sweep_eval.id_switches),
                            "num_tracks": len(sweep_tracks),
                        }
                        sweep_results.append(record)
                        if best_result is None or record["idf1"] > best_result["idf1"]:
                            best_result = record

            assert best_result is not None, "Tracking sweep produced no results"
            with (OUTPUT_DIR / "tracking_sweep_best.json").open("w", encoding="utf-8") as handle:
                json.dump(best_result, handle, indent=2)
            with (OUTPUT_DIR / "tracking_sweep_results.json").open("w", encoding="utf-8") as handle:
                json.dump(sweep_results, handle, indent=2)

            print(f"[12b] best sweep result: {json.dumps(best_result, indent=2)}")
            """,
        ),
        code_cell(
            "12b-package",
            """
            import json
            import shutil

            key_artifacts = [
                OUTPUT_DIR / "ground_plane_tracks.json",
                OUTPUT_DIR / "ground_plane_tracks.csv",
                OUTPUT_DIR / "global_trajectories.json",
                OUTPUT_DIR / "evaluation_summary.json",
                OUTPUT_DIR / "tracking_sweep_best.json",
                OUTPUT_DIR / "reid_merge_candidates.json",
                OUTPUT_DIR / "reid_features.npz",
            ]

            print("=" * 60)
            print("12b Output Artifacts")
            print("=" * 60)
            output_listing = []
            for path in sorted(OUTPUT_DIR.rglob("*")):
                if not path.is_file():
                    continue
                size_mb = path.stat().st_size / (1024 * 1024)
                rel_path = str(path.relative_to(OUTPUT_DIR))
                output_listing.append({"path": rel_path, "size_mb": round(size_mb, 4)})
                print(f"{rel_path}: {size_mb:.2f} MB")

            for artifact_path in key_artifacts:
                if artifact_path.is_file():
                    shutil.copy2(artifact_path, Path("/kaggle/working") / artifact_path.name)

            summary = {
                "output_dir": str(OUTPUT_DIR),
                "artifact_count": len(output_listing),
                "artifacts": output_listing,
                "copied_to_working": [path.name for path in key_artifacts if path.is_file()],
            }
            with (OUTPUT_DIR / "artifact_manifest.json").open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)
            print(json.dumps(summary, indent=2))
            """,
        ),
    ]


def build_notebook() -> dict:
    return {
        "cells": build_cells(),
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_kernel_metadata() -> dict:
    return {
        "id": "ali369/12b-wildtrack-mvdetr-tracking-reid",
        "title": "12b WILDTRACK Tracking + ReID Pipeline",
        "code_file": "12b_wildtrack_tracking_reid.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "keywords": [],
        "dataset_sources": [
            "mrkdagods/mtmc-weights",
            "aryashah2k/large-scale-multicamera-detection-dataset",
            "mrkdagods/mtmc-gp",
        ],
        "kernel_sources": [
            "ali369/12a-wildtrack-mvdetr-training",
        ],
        "competition_sources": [],
        "model_sources": [],
    }


def validate_notebook(notebook_data: dict) -> None:
    nb_node = nbformat.from_dict(notebook_data)
    nbformat.validate(nb_node)


def validate_source_lines(notebook_data: dict) -> None:
    for cell_index, cell in enumerate(notebook_data["cells"], start=1):
        source = cell.get("source", [])
        if not isinstance(source, list):
            raise TypeError(f"Cell {cell_index} source is not a list")
        if len(source) <= 1:
            continue
        if any(not line.endswith("\n") for line in source[:-1]):
            raise ValueError(f"Cell {cell_index} violates source newline rule before last line")
        if source[-1].endswith("\n"):
            raise ValueError(f"Cell {cell_index} last source line must not end with newline")


def validate_code_cells(notebook_data: dict) -> None:
    for cell_index, cell in enumerate(notebook_data["cells"], start=1):
        if cell["cell_type"] != "code":
            continue
        compile("".join(cell["source"]), f"cell_{cell_index}", "exec")


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

    notebook_data = build_notebook()
    kernel_metadata = build_kernel_metadata()

    validate_source_lines(notebook_data)
    validate_code_cells(notebook_data)
    validate_notebook(notebook_data)

    with NOTEBOOK_PATH.open("w", encoding="utf-8") as handle:
        json.dump(notebook_data, handle, ensure_ascii=True, indent=1)
        handle.write("\n")

    with METADATA_PATH.open("w", encoding="utf-8") as handle:
        json.dump(kernel_metadata, handle, ensure_ascii=True, indent=2)
        handle.write("\n")

    with NOTEBOOK_PATH.open("r", encoding="utf-8") as handle:
        reloaded_notebook = json.load(handle)
    with METADATA_PATH.open("r", encoding="utf-8") as handle:
        reloaded_metadata = json.load(handle)

    validate_source_lines(reloaded_notebook)
    validate_notebook(reloaded_notebook)

    summary = {
        "notebook": str(NOTEBOOK_PATH),
        "kernel_metadata": str(METADATA_PATH),
        "cell_count": len(reloaded_notebook["cells"]),
        "dataset_sources": reloaded_metadata["dataset_sources"],
        "kernel_sources": reloaded_metadata["kernel_sources"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()