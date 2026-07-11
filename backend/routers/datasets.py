import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException

from backend.config import (
    DATASET_BROWSE_ROOT,
    DATASET_CONFIG_DIR,
    DATASET_DIR,
    OUTPUT_DIR,
    VIDEO_EXTENSIONS,
)
from backend.dependencies import get_app_state
from backend.services.pipeline_service import (
    _execute_dataset_pipeline,
    _execute_input_dir_pipeline,
    _resolve_run_id,
    _write_run_context,
    resolve_dataset_key,
)
from backend.services.video_service import _probe_video_metadata, _register_video_path
from backend.state import AppState

router = APIRouter()

_CAMERA_COORDS_FILENAME = "camera_coordinates.json"


def _parse_coordinate_payload(data: Any) -> Dict[str, Any]:
    """Normalize a JSON object of camera id -> {lat, lng, optional label}."""
    if not isinstance(data, dict):
        return {}
    out: Dict[str, Any] = {}
    for key, raw in data.items():
        sk = str(key).strip()
        if not sk or not isinstance(raw, dict):
            continue
        try:
            lat_f = float(raw.get("lat"))
            lng_f = float(raw.get("lng"))
        except (TypeError, ValueError):
            continue
        entry: Dict[str, Any] = {"lat": lat_f, "lng": lng_f}
        label = raw.get("label")
        if isinstance(label, str) and label.strip():
            entry["label"] = label.strip()
        out[sk] = entry
    return out


def _load_camera_coordinates(dataset_path: Path) -> Dict[str, Any]:
    """Load per-camera map coordinates from dataset/<folder>/camera_coordinates.json."""
    path = dataset_path / _CAMERA_COORDS_FILENAME
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _parse_coordinate_payload(data)


def _scan_input_dir_cameras(input_dir: Path) -> Tuple[str, List[Dict[str, Any]]]:
    """Inspect a folder and report its layout + cameras."""
    if not input_dir.exists() or not input_dir.is_dir():
        return ("missing", [])

    per_camera: List[Dict[str, Any]] = []
    for child in sorted(input_dir.iterdir()):
        if child.is_dir():
            vids = [
                p for p in sorted(child.iterdir())
                if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
            ]
            if vids:
                per_camera.append({"id": child.name, "hasVideo": True, "file": vids[0].name})
    if per_camera:
        return ("per_camera", per_camera)

    flat = [
        p for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if flat:
        return ("flat", [{"id": p.stem, "hasVideo": True, "file": p.name} for p in flat])

    return ("empty", [])


def _norm(p: Path) -> str:
    return str(p).replace("\\", "/")


def _dataset_video_records(
    input_dir: Path,
    layout: str,
    cameras: List[Dict[str, Any]],
    selected: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """Build {id, cameraId, path, name} for each (selected) camera video. Ids are"""
    selected_set = set(selected) if selected else None
    records: List[Dict[str, Any]] = []
    for cam in cameras:
        if selected_set is not None and cam["id"] not in selected_set:
            continue
        vpath = (
            input_dir / cam["file"]
            if layout == "flat"
            else input_dir / cam["id"] / cam["file"]
        )
        if not vpath.exists():
            continue
        vid_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(vpath.resolve())))
        records.append({
            "id": vid_id,
            "cameraId": cam["id"],
            "path": str(vpath),
            "name": cam["id"],
        })
    return records


def _first_video_path(
    input_dir: Path, layout: str, cameras: List[Dict[str, Any]]
) -> Optional[Path]:
    """Path to the first camera's video file, for cheap metadata probing."""
    if not cameras:
        return None
    cam = cameras[0]
    if layout == "flat":
        return input_dir / cam["file"]
    if layout == "per_camera":
        return input_dir / cam["id"] / cam["file"]
    return None


@router.get("/api/datasets/available")
async def available_datasets():
    """List selectable tracking datasets, read from configs/datasets/*.yaml."""
    out: List[Dict[str, Any]] = []
    if not DATASET_CONFIG_DIR.exists():
        return {"success": True, "data": out}

    for cfg_file in sorted(DATASET_CONFIG_DIR.glob("*.yaml")):
        try:
            data = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
        except (yaml.YAMLError, OSError):
            continue
        stage0 = data.get("stage0") or {}
        input_dir = stage0.get("input_dir")
        if not input_dir:
            continue  # ReID-only configs (no video ingestion) are not selectable here

        p = Path(input_dir)
        layout, cameras = _scan_input_dir_cameras(p)
        dataset_meta = data.get("dataset") or {}

        # Probe the first camera for the *real* source fps/resolution so the UI
        # can distinguish native fps from the pipeline's frame-sampling rate.
        source_fps = None
        source_width = None
        source_height = None
        first = _first_video_path(p, layout, cameras)
        if first is not None and first.exists():
            meta = _probe_video_metadata(first)
            source_fps = meta.get("fps")
            source_width = meta.get("width")
            source_height = meta.get("height")

        out.append(
            {
                "name": cfg_file.stem,
                "configFile": _norm(cfg_file),
                "inputDir": _norm(p),
                "taskType": dataset_meta.get("task_type") or data.get("task_type"),
                "layout": layout,
                "available": layout in ("per_camera", "flat"),
                "cameraCount": len(cameras),
                "videosFound": sum(1 for c in cameras if c["hasVideo"]),
                "sourceFps": source_fps,          # native video fps (probed)
                "sampleFps": stage0.get("output_fps"),  # rate Stage 0 extracts at
                "width": source_width,
                "height": source_height,
                "cameras": cameras,
            }
        )
    return {"success": True, "data": out}


def _resolve_browse_target(rel: str) -> Path:
    """Resolve a browse path under DATASET_BROWSE_ROOT, rejecting escapes."""
    root = DATASET_BROWSE_ROOT.resolve()
    target = (DATASET_BROWSE_ROOT / (rel or "")).resolve()
    if target != root and root not in target.parents:
        raise HTTPException(status_code=400, detail="path escapes browse root")
    return target


@router.get("/api/datasets/browse")
async def browse_datasets(path: str = ""):
    """Sandboxed folder browser rooted at data/raw/ for custom dataset selection."""
    root = DATASET_BROWSE_ROOT
    if not root.exists():
        return {
            "success": True,
            "data": {
                "root": _norm(root), "path": "", "parent": None, "entries": [],
                "datasetLike": False, "layout": "missing", "cameras": [],
                "inputDir": _norm(root),
            },
        }

    target = _resolve_browse_target(path)
    if not target.exists() or not target.is_dir():
        raise HTTPException(status_code=404, detail="folder not found")

    root_resolved = root.resolve()
    rel_path = "" if target == root_resolved else _norm(target.relative_to(root_resolved))
    parent: Optional[str] = None
    if target != root_resolved:
        parent = (
            "" if target.parent == root_resolved
            else _norm(target.parent.relative_to(root_resolved))
        )

    entries: List[Dict[str, Any]] = []
    for child in sorted(target.iterdir()):
        if child.is_dir():
            entries.append({"name": child.name, "type": "dir",
                            "path": _norm(child.relative_to(root_resolved))})
        elif child.is_file() and child.suffix.lower() in VIDEO_EXTENSIONS:
            entries.append({"name": child.name, "type": "video",
                            "path": _norm(child.relative_to(root_resolved))})

    layout, cameras = _scan_input_dir_cameras(target)
    return {
        "success": True,
        "data": {
            "root": _norm(root_resolved),
            "path": rel_path,
            "parent": parent,
            "entries": entries,
            "datasetLike": layout in ("per_camera", "flat"),
            "layout": layout,
            "cameras": cameras,
            "inputDir": _norm(target),
        },
    }


@router.get("/api/datasets/videos")
async def dataset_videos(inputDir: str, state: AppState = Depends(get_app_state)):
    """Return the camera videos inside a chosen dataset/folder as gallery records."""
    resolved = Path(inputDir).resolve()
    root = DATASET_BROWSE_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise HTTPException(status_code=400, detail="inputDir must be under data/raw/")
    if not resolved.exists() or not resolved.is_dir():
        raise HTTPException(status_code=404, detail=f"inputDir not found: {inputDir}")

    layout, cameras = _scan_input_dir_cameras(resolved)
    if layout not in ("per_camera", "flat"):
        return {"success": True, "data": []}

    videos: List[Dict[str, Any]] = []
    for cam in cameras:
        if layout == "flat":
            vpath = resolved / cam["file"]
        else:
            vpath = resolved / cam["id"] / cam["file"]
        if not vpath.exists():
            continue
        _register_video_path(vpath)
        vid_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(vpath.resolve())))
        rec = state.uploaded_videos.get(vid_id)
        if rec:
            # Persist the real camera id on the stored record so run-linking and the
            # detections endpoint can resolve it (e.g. WILDTRACK "C1", which the
            # CityFlow S##_c### regex can't recover from the filename).
            rec["cameraId"] = cam["id"]
            rec["_camera_id"] = cam["id"]
            payload = dict(rec)
            payload["latestRunId"] = state.video_to_latest_run.get(vid_id)
            videos.append(payload)
    return {"success": True, "data": videos}


@router.post("/api/datasets/run")
async def run_dataset_input(
    background_tasks: BackgroundTasks,
    payload: Dict[str, Any] = Body(...),
    state: AppState = Depends(get_app_state),
):
    """Start a pipeline run against a chosen input folder (dataset or custom)."""
    input_dir = str(payload.get("inputDir") or "").strip()
    if not input_dir:
        raise HTTPException(status_code=422, detail="inputDir is required")

    resolved = Path(input_dir).resolve()
    root = DATASET_BROWSE_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise HTTPException(status_code=400, detail="inputDir must be under data/raw/")
    if not resolved.exists() or not resolved.is_dir():
        raise HTTPException(status_code=404, detail=f"inputDir not found: {input_dir}")

    layout, cameras = _scan_input_dir_cameras(resolved)
    if layout not in ("per_camera", "flat"):
        raise HTTPException(status_code=400, detail="no videos found in the selected folder")

    name = str(payload.get("name") or resolved.name)
    stages = str(payload.get("stages") or "0")
    smoke = bool(payload.get("smoke") or False)

    # Which dataset drives detection classes (vehicles vs people). Prefer an explicit
    # hint from the client, else the dataset name, else infer from the input path.
    # None -> configs/default.yaml (vehicles). Without this, a WILDTRACK run detected
    # only vehicle classes and never produced any people.
    dataset_key = resolve_dataset_key(
        str(payload.get("dataset") or payload.get("name") or ""), _norm(resolved)
    )

    # Optional subset of cameras to track. Validate against what's on disk.
    requested = payload.get("cameras") or []
    available_ids = {c["id"] for c in cameras}
    selected = [str(c) for c in requested if str(c) in available_ids]
    if requested and not selected:
        raise HTTPException(status_code=400, detail="none of the requested cameras exist")

    # Reuse an explicit runId so the per-stage flow runs each pipeline stage
    # incrementally against the SAME run dir (stages read prior stages' outputs
    # from outputs/<run_id>/stageN). Omitting runId allocates a fresh run.
    requested_run_id = payload.get("runId")
    if requested_run_id is not None and not str(requested_run_id).strip():
        requested_run_id = None
    run_id = _resolve_run_id(str(requested_run_id) if requested_run_id is not None else None)

    # Camera video records for this run (deterministic ids) - persisted in
    # run_context.json so the run can be fully rebuilt from disk after a restart.
    video_records = _dataset_video_records(resolved, layout, cameras, selected)
    state.active_runs[run_id] = {
        "id": run_id,
        "runId": run_id,
        "status": "running",
        "progress": 0,
        "message": f"Starting pipeline on {name}...",
        "startedAt": datetime.now().isoformat(),
        "datasetFolder": name,
        "inputDir": _norm(resolved),
        "dataset": dataset_key,
        "cameraCount": len(selected) if selected else len(cameras),
        "selectedCameras": selected or None,
        "stages": stages,
    }
    _write_run_context(
        run_id,
        {
            "source": "dataset-input",
            "datasetName": name,
            "dataset": dataset_key,
            "inputDir": _norm(resolved),
            "layout": layout,
            "selectedCameras": selected or None,
            "smoke": smoke,
            "videos": video_records,
        },
    )
    background_tasks.add_task(
        _execute_input_dir_pipeline,
        run_id, _norm(resolved), stages, smoke, name, selected or None, dataset_key,
    )
    return {"success": True, "data": state.active_runs[run_id]}


@router.get("/api/datasets")
async def list_datasets(state: AppState = Depends(get_app_state)):
    """List available dataset folders under dataset/ with camera info."""
    results: List[Dict[str, Any]] = []
    if not DATASET_DIR.exists():
        return {"success": True, "data": results}

    for folder in sorted(DATASET_DIR.iterdir()):
        if not folder.is_dir():
            continue
        cameras: List[Dict[str, Any]] = []
        for cam_dir in sorted(folder.iterdir()):
            if not cam_dir.is_dir():
                continue
            has_video = any((cam_dir / f"vdo{ext}").exists() for ext in VIDEO_EXTENSIONS)
            cameras.append({"id": cam_dir.name, "hasVideo": has_video})
        dataset_key = folder.name.lower()
        candidate_runs: List[tuple] = []
        if OUTPUT_DIR.exists():
            for run_dir in OUTPUT_DIR.iterdir():
                if not run_dir.is_dir():
                    continue
                run_id = run_dir.name

                matched = False
                if run_id == f"dataset_precompute_{dataset_key}":
                    matched = True

                if not matched:
                    ctx_path = run_dir / "run_context.json"
                    if ctx_path.exists():
                        try:
                            ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
                            if str(ctx.get("source", "")).startswith("dataset") and str(
                                ctx.get("datasetFolder", "")
                            ).lower() == dataset_key:
                                matched = True
                        except Exception:
                            pass

                if matched:
                    candidate_runs.append((run_dir.stat().st_mtime, run_id, run_dir))

        candidate_runs.sort(key=lambda x: x[0], reverse=True)
        latest_run_id = candidate_runs[0][1] if candidate_runs else None
        latest_run_dir = candidate_runs[0][2] if candidate_runs else None

        already_processed = False
        has_gallery = False
        if latest_run_dir is not None:
            already_processed = (latest_run_dir / "stage1").exists() and any(
                (latest_run_dir / "stage1").glob("tracklets_*.json")
            )
            has_gallery = (
                already_processed
                and (latest_run_dir / "stage2" / "embeddings.npy").exists()
                and (latest_run_dir / "stage2" / "embedding_index.json").exists()
                and (latest_run_dir / "stage4" / "global_trajectories.json").exists()
            )

        is_processing = any(
            r.get("status") == "running"
            and str(r.get("datasetFolder", "")).lower() == dataset_key
            for r in state.active_runs.values()
        )

        coord_map = _load_camera_coordinates(folder)

        results.append(
            {
                "name": folder.name,
                "path": str(folder),
                "cameras": cameras,
                "cameraCount": len(cameras),
                "videosFound": sum(1 for c in cameras if c["hasVideo"]),
                "alreadyProcessed": already_processed,
                "hasGallery": has_gallery,
                "isProcessing": is_processing,
                "runId": latest_run_id
                if (latest_run_id and (already_processed or is_processing))
                else None,
                "galleryRunId": latest_run_id if (latest_run_id and has_gallery) else None,
                "cameraCoordinates": coord_map if coord_map else None,
            }
        )

    return {"success": True, "data": results}


@router.post("/api/datasets/{folder}/process")
async def process_dataset(folder: str, background_tasks: BackgroundTasks, state: AppState = Depends(get_app_state)):
    """Trigger full pipeline (stages 0-4) on a dataset folder."""
    dataset_path = DATASET_DIR / folder
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise HTTPException(status_code=404, detail=f"Dataset folder '{folder}' not found")

    run_id = _resolve_run_id(None)

    for run in state.active_runs.values():
        if run.get("status") == "running" and str(run.get("datasetFolder", "")).lower() == folder.lower():
            return {"success": True, "data": run, "message": "Already processing"}

    state.active_runs[run_id] = {
        "id": run_id,
        "runId": run_id,
        "status": "running",
        "progress": 0,
        "message": f"Starting pipeline on {folder}...",
        "startedAt": datetime.now().isoformat(),
        "datasetFolder": folder,
        "totalStages": 5,
        "completedStages": 0,
    }

    _write_run_context(
        run_id,
        {
            "source": "dataset-process",
            "datasetFolder": folder,
            "datasetPath": str(dataset_path),
        },
    )

    background_tasks.add_task(_execute_dataset_pipeline, run_id, dataset_path, folder)
    return {"success": True, "data": state.active_runs[run_id]}


@router.put("/api/datasets/{folder}/camera-coordinates")
async def put_camera_coordinates(
    folder: str,
    coordinates: Dict[str, Any] = Body(...),
):
    """Write dataset/<folder>/camera_coordinates.json.

    Body is a JSON object mapping camera ids to {\"lat\", \"lng\", optional \"label\"}.
    """
    dataset_path = (DATASET_DIR / folder).resolve()
    try:
        dataset_path.relative_to(DATASET_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid dataset folder") from None
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise HTTPException(status_code=404, detail=f"Dataset folder '{folder}' not found")

    out = _parse_coordinate_payload(coordinates)

    target = dataset_path / _CAMERA_COORDS_FILENAME
    target.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"success": True, "data": out}
