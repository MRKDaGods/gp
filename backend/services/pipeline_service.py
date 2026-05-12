"""Pipeline subprocess orchestration, run ID allocation, and background tasks."""
import asyncio
import json
import shutil
import subprocess
import sys
import threading
import traceback as _traceback
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config import (
    _CAMERA_LINE_RE,
    _HAS_CV2,
    _PIPELINE_PYTHON,
    _PROJECT_ROOT,
    _STAGE_LINE_RE,
    _STAGE_NAMES,
    DATASET_DIR,
    OUTPUT_DIR,
    PRECOMPUTE_RUN_ID,
    UPLOAD_DIR,
    VIDEO_EXTENSIONS,
)
from backend.services.tracklet_service import _persist_probe_link
from backend.services.video_service import (
    _detect_camera_for_video,
    _extract_camera_id,
    _safe_reid_batch_size,
)
from backend.state import app_state


def _allocate_numeric_run_id() -> str:
    """Allocate the next numeric run id under outputs/ (1, 2, 3, ...)."""
    with app_state.run_id_lock:
        max_num = 0
        try:
            for child in OUTPUT_DIR.iterdir():
                if child.is_dir() and child.name.isdigit():
                    max_num = max(max_num, int(child.name))
        except Exception:
            pass

        next_num = max_num + 1
        while True:
            run_id = str(next_num)
            run_dir = OUTPUT_DIR / run_id
            try:
                run_dir.mkdir(parents=True, exist_ok=False)
                return run_id
            except FileExistsError:
                next_num += 1


def _resolve_run_id(requested_run_id: Optional[str]) -> str:
    """Resolve a run id: keep explicit id, otherwise allocate numeric id."""
    if requested_run_id is not None:
        txt = str(requested_run_id).strip()
        if txt:
            return txt
    return _allocate_numeric_run_id()


def _write_run_context(run_id: str, payload: Dict[str, Any]) -> None:
    """Persist lightweight run metadata to help auditing and dataset discovery."""
    try:
        run_dir = OUTPUT_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        context = {
            "runId": run_id,
            "createdAt": datetime.now().isoformat(),
            **payload,
        }
        (run_dir / "run_context.json").write_text(json.dumps(context, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[WARN] Failed to write run_context.json for run {run_id}: {exc}", flush=True)


def _prepare_input_for_run(run_id: str, source_video_path: Path, camera_id: str) -> Path:
    run_input_dir = OUTPUT_DIR / run_id / "input" / camera_id
    run_input_dir.mkdir(parents=True, exist_ok=True)

    target_video_path = run_input_dir / source_video_path.name
    shutil.copy2(source_video_path, target_video_path)

    return run_input_dir.parent


def _prepare_dataset_input_for_run(run_id: str, dataset_path: Path) -> Path:
    """Copy dataset input videos into outputs/{run_id}/input/ for full run reproducibility."""
    run_input_root = OUTPUT_DIR / run_id / "input"
    run_input_root.mkdir(parents=True, exist_ok=True)

    copied: List[Dict[str, str]] = []

    for child in sorted(dataset_path.iterdir()):
        if not child.is_dir():
            continue
        camera_dir = run_input_root / child.name
        camera_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(child.iterdir()):
            if not src.is_file() or src.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            dst = camera_dir / src.name
            shutil.copy2(src, dst)
            copied.append({"source": str(src), "copiedTo": str(dst.relative_to(OUTPUT_DIR / run_id).as_posix())})

    if not copied:
        misc_dir = run_input_root / "misc"
        misc_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(dataset_path.iterdir()):
            if not src.is_file() or src.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            dst = misc_dir / src.name
            shutil.copy2(src, dst)
            copied.append({"source": str(src), "copiedTo": str(dst.relative_to(OUTPUT_DIR / run_id).as_posix())})

    manifest = {
        "sourceDatasetPath": str(dataset_path),
        "copiedAt": datetime.now().isoformat(),
        "copiedVideoCount": len(copied),
        "videos": copied,
    }
    (run_input_root / "input_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return run_input_root


def _cuda_available_for_pipeline() -> bool:
    """Match subprocess torch CUDA visibility (same check as src.core.config)."""
    try:
        from src.core.config import is_torch_cuda_available
        return is_torch_cuda_available()
    except Exception:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False


def _build_pipeline_cmd(
    stages: str,
    run_id: str,
    input_dir: str,
    camera_id: Optional[str] = None,
    smoke_test: bool = False,
    use_cpu: bool = False,
    reid_model_path: Optional[str] = None,
    tracker: Optional[str] = None,
) -> list:
    """Build the subprocess command for run_pipeline.py."""
    effective_use_cpu = use_cpu or not _cuda_available_for_pipeline()
    cmd = [
        _PIPELINE_PYTHON,
        "scripts/run_pipeline.py",
        "--config",
        "configs/default.yaml",
        "--stages",
        stages,
        "--override",
        f"project.output_dir={OUTPUT_DIR.as_posix()}",
        "--override",
        f"project.run_name='{run_id}'",
        "--override",
        f"stage0.input_dir={input_dir}",
        "--override",
        "stage4.global_gallery.enabled=true",
    ]
    if camera_id:
        cmd.extend(["--override", f"stage0.cameras=[{camera_id}]"])
    if smoke_test:
        cmd.append("--smoke-test")
    if effective_use_cpu:
        cmd.extend([
            "--override", "stage1.detector.device=cpu",
            "--override", "stage1.tracker.device=cpu",
            "--override", "stage1.detector.half=false",
            "--override", "stage1.tracker.half=false",
            "--override", "stage2.reid.device=cpu",
            "--override", "stage2.reid.half=false",
            "--override", "stage2.reid.batch_size=4",
        ])
    else:
        if sys.platform == "win32":
            cmd.extend([
                "--override", "stage2.reid.half=false",
                "--override", f"stage2.reid.batch_size={_safe_reid_batch_size()}",
            ])
    if reid_model_path:
        cmd.extend([
            "--override", f"stage2.reid.vehicle.weights_path={reid_model_path}",
        ])
    if tracker:
        cmd.extend([
            "--override", f"stage1.tracker.type={tracker}",
        ])
    return cmd


async def _run_pipeline_streaming(
    run_id: str,
    cmd: list,
    stage_nums: list,
) -> Dict[str, Any]:
    """Run a pipeline subprocess using threads so it works on any asyncio event loop."""
    total_stages = max(len(stage_nums), 1)
    completed_stages = 0
    cameras_seen: list = []
    log_lines: list = []

    loop = asyncio.get_event_loop()
    line_queue: asyncio.Queue = asyncio.Queue()

    def _handle_line(line: str) -> None:
        nonlocal completed_stages
        log_lines.append(line)

        m = _STAGE_LINE_RE.search(line)
        if m:
            stage_num = int(m.group(1))
            stage_label = _STAGE_NAMES.get(stage_num, f"Stage {stage_num}")
            completed_stages += 1
            pct = min(int((completed_stages / total_stages) * 95), 95)
            cameras_seen.clear()
            if run_id in app_state.active_runs:
                app_state.active_runs[run_id]["progress"] = pct
                app_state.active_runs[run_id]["message"] = f"Running {stage_label}..."
                app_state.active_runs[run_id]["currentStageName"] = stage_label
                app_state.active_runs[run_id]["currentStageNum"] = stage_num
                app_state.active_runs[run_id]["completedStages"] = completed_stages
                app_state.active_runs[run_id]["totalStages"] = total_stages

        cm = _CAMERA_LINE_RE.search(line)
        if cm and run_id in app_state.active_runs:
            cam_id = cm.group(1)
            if cam_id not in cameras_seen:
                cameras_seen.append(cam_id)
            cam_index = cameras_seen.index(cam_id) + 1
            app_state.active_runs[run_id]["currentCamera"] = cam_id
            app_state.active_runs[run_id]["camerasProcessed"] = cam_index
            stage_name = app_state.active_runs[run_id].get("currentStageName", "Processing")
            app_state.active_runs[run_id]["message"] = (
                f"{stage_name} — camera {cam_id} ({cam_index} processed)"
            )

    def _drain_stream(stream) -> None:
        try:
            for raw_line in stream:
                line = raw_line.rstrip() if isinstance(raw_line, str) else raw_line.decode(errors="ignore").rstrip()
                loop.call_soon_threadsafe(line_queue.put_nowait, ("line", line))
        except Exception:
            pass
        finally:
            loop.call_soon_threadsafe(line_queue.put_nowait, ("done", None))

    def _run_blocking() -> int:
        proc = subprocess.Popen(
            cmd,
            cwd=str(_PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=0,
        )
        t_out = threading.Thread(target=_drain_stream, args=(proc.stdout,), daemon=True)
        t_err = threading.Thread(target=_drain_stream, args=(proc.stderr,), daemon=True)
        t_out.start()
        t_err.start()
        proc.wait()
        try:
            proc.stdout.close()
        except Exception:
            pass
        try:
            proc.stderr.close()
        except Exception:
            pass
        t_out.join(timeout=30)
        t_err.join(timeout=30)
        return proc.returncode

    future = loop.run_in_executor(None, _run_blocking)

    sentinels = 0
    while sentinels < 2:
        kind, payload = await line_queue.get()
        if kind == "done":
            sentinels += 1
        else:
            _handle_line(payload)

    returncode = await future
    run_dir = OUTPUT_DIR / run_id

    if returncode != 0:
        stderr_tail = "\n".join(log_lines[-80:])[-4000:]
        raise RuntimeError(
            f"Pipeline exited with code {returncode}.\n\n"
            f"Last log output:\n{stderr_tail}"
        )

    return {
        "runDir": str(run_dir),
        "logTail": "\n".join(log_lines[-50:]),
    }


async def _run_pipeline_stages(
    run_id: str,
    stages: str,
    video_id: str,
    camera_id: str,
    use_cpu: bool = False,
    smoke_test: bool = False,
    reid_model_path: Optional[str] = None,
    tracker: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one or more pipeline stages via subprocess with streaming progress."""
    video_meta = app_state.uploaded_videos[video_id]
    source_video_path = Path(video_meta["path"]).resolve()
    if not source_video_path.exists():
        raise FileNotFoundError(f"Video file does not exist: {source_video_path}")

    input_dir = _prepare_input_for_run(run_id, source_video_path, camera_id)

    stage_nums = [int(s.strip()) for s in stages.split(",")]

    cmd = _build_pipeline_cmd(
        stages=stages,
        run_id=run_id,
        input_dir=input_dir.as_posix(),
        camera_id=camera_id,
        smoke_test=smoke_test,
        use_cpu=use_cpu,
        reid_model_path=reid_model_path,
        tracker=tracker,
    )

    return await _run_pipeline_streaming(run_id, cmd, stage_nums)


def _materialize_import_tree(extracted_root: Path, run_dir: Path) -> None:
    """Copy extracted Kaggle artifacts into a normalized run directory."""
    stage_names = {"stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "stage6"}

    candidate_root = extracted_root
    direct_dirs = {p.name for p in extracted_root.iterdir() if p.is_dir()}
    if not stage_names.intersection(direct_dirs):
        children = [p for p in extracted_root.iterdir() if p.is_dir()]
        if len(children) == 1:
            nested_dirs = {p.name for p in children[0].iterdir() if p.is_dir()}
            if stage_names.intersection(nested_dirs):
                candidate_root = children[0]

    run_dir.mkdir(parents=True, exist_ok=True)
    for child in candidate_root.iterdir():
        destination = run_dir / child.name
        if child.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(child, destination)
        else:
            shutil.copy2(child, destination)


async def _background_precompute_dataset() -> None:
    """Run the full pipeline (stages 0-4) on the S01 dataset at startup."""
    dataset_s01 = DATASET_DIR / "S01"
    if not dataset_s01.exists():
        return

    run_dir = OUTPUT_DIR / PRECOMPUTE_RUN_ID
    if any((run_dir / "stage1").glob("tracklets_*.json")):
        for vid_id, vid_meta in list(app_state.uploaded_videos.items()):
            cam_id = _extract_camera_id(str(vid_meta.get("path", "")))
            if cam_id and cam_id.startswith("S01_"):
                app_state.video_to_latest_run[vid_id] = PRECOMPUTE_RUN_ID
        return

    try:
        cmd = [
            _PIPELINE_PYTHON,
            "scripts/run_pipeline.py",
            "--config", "configs/default.yaml",
            "--stages", "0,1,2,3,4",
            "--override", f"project.output_dir={OUTPUT_DIR.as_posix()}",
            "--override", f"project.run_name={PRECOMPUTE_RUN_ID}",
            "--override", f"stage0.input_dir={dataset_s01.as_posix()}",
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(_PROJECT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            err = stderr.decode(errors="ignore")[-2000:]
            print(f"[PRECOMPUTE] Pipeline failed: {err}")
            return

        for vid_id, vid_meta in list(app_state.uploaded_videos.items()):
            cam_id = _extract_camera_id(str(vid_meta.get("path", "")))
            if cam_id and cam_id.startswith("S01_"):
                app_state.video_to_latest_run[vid_id] = PRECOMPUTE_RUN_ID

        print(
            f"[PRECOMPUTE] S01 pipeline complete — "
            f"{len(list((run_dir / 'stage1').glob('tracklets_*.json')))} cameras processed"
        )

    except Exception as exc:
        print(f"[PRECOMPUTE] Background precompute error: {exc}")


async def execute_stage(run_id: str, stage: int, config: Dict[str, Any]):
    """Execute a real pipeline stage for a selected video."""
    try:
        if run_id not in app_state.active_runs:
            return

        video_id = config.get("videoId")
        camera_id = config.get("cameraId")
        smoke_test = bool(config.get("smokeTest", False))
        use_cpu = bool(config.get("useCpu", False))
        reid_model_path = config.get("reidModelPath")
        tracker = str(config.get("tracker") or "deepocsort")

        if not video_id or video_id not in app_state.uploaded_videos:
            raise RuntimeError(f"Stage {stage} requires a valid videoId")

        if not camera_id:
            camera_id = _detect_camera_for_video(app_state.uploaded_videos[video_id], None)

        app_state.active_runs[run_id]["cameraId"] = camera_id

        if stage == 1:
            app_state.active_runs[run_id]["message"] = f"Running detection & tracking for {camera_id}..."
            app_state.active_runs[run_id]["progress"] = 10

            run_meta = await _run_pipeline_stages(
                run_id=run_id,
                stages="0,1",
                video_id=video_id,
                camera_id=camera_id,
                use_cpu=use_cpu,
                smoke_test=smoke_test,
                tracker=tracker,
            )

            app_state.video_to_latest_run[video_id] = run_id
            _persist_probe_link(video_id, run_id)
            app_state.active_runs[run_id]["status"] = "completed"
            app_state.active_runs[run_id]["progress"] = 100
            app_state.active_runs[run_id]["message"] = "Detection & tracking complete"
            app_state.active_runs[run_id]["runDir"] = run_meta["runDir"]
            app_state.active_runs[run_id]["completedAt"] = datetime.now().isoformat()
            return

        if stage in (2, 3):
            run_dir = OUTPUT_DIR / run_id
            stage2_done = (run_dir / "stage2" / "embeddings.npy").exists()

            if stage == 3 and stage2_done:
                stages_to_run = "3"
                app_state.active_runs[run_id]["message"] = "Running indexing (embeddings already extracted)..."
            else:
                stages_to_run = "2,3"
                app_state.active_runs[run_id]["message"] = "Running feature extraction & indexing..."

            app_state.active_runs[run_id]["progress"] = 10

            run_meta = await _run_pipeline_stages(
                run_id=run_id,
                stages=stages_to_run,
                video_id=video_id,
                camera_id=camera_id,
                use_cpu=use_cpu,
                smoke_test=smoke_test,
                reid_model_path=reid_model_path,
            )

            app_state.active_runs[run_id]["status"] = "completed"
            app_state.active_runs[run_id]["progress"] = 100
            app_state.active_runs[run_id]["message"] = f"Stage {stage} complete"
            app_state.active_runs[run_id]["runDir"] = run_meta["runDir"]
            app_state.active_runs[run_id]["completedAt"] = datetime.now().isoformat()
            return

        if stage == 4:
            app_state.active_runs[run_id]["message"] = "Running cross-camera association..."
            app_state.active_runs[run_id]["progress"] = 10

            run_meta = await _run_pipeline_stages(
                run_id=run_id,
                stages="4",
                video_id=video_id,
                camera_id=camera_id,
                use_cpu=use_cpu,
                smoke_test=smoke_test,
            )

            app_state.active_runs[run_id]["status"] = "completed"
            app_state.active_runs[run_id]["progress"] = 100
            app_state.active_runs[run_id]["message"] = "Association complete"
            app_state.active_runs[run_id]["runDir"] = run_meta["runDir"]
            app_state.active_runs[run_id]["completedAt"] = datetime.now().isoformat()
            return

        stage_name = {5: "evaluation", 6: "visualization"}.get(stage, str(stage))
        app_state.active_runs[run_id]["message"] = f"Running {stage_name}..."
        app_state.active_runs[run_id]["progress"] = 10

        run_meta = await _run_pipeline_stages(
            run_id=run_id,
            stages=str(stage),
            video_id=video_id,
            camera_id=camera_id,
            use_cpu=use_cpu,
            smoke_test=smoke_test,
        )

        app_state.active_runs[run_id]["status"] = "completed"
        app_state.active_runs[run_id]["progress"] = 100
        app_state.active_runs[run_id]["completedAt"] = datetime.now().isoformat()

    except BaseException as e:
        tb = _traceback.format_exc()
        err_type = type(e).__name__
        err_msg = str(e) or f"({err_type} with no message)"
        full_error = f"{err_type}: {err_msg}"
        print(f"[PIPELINE ERROR] run={run_id} stage={stage}\n{tb}", flush=True)
        if run_id in app_state.active_runs:
            app_state.active_runs[run_id]["status"] = "error"
            app_state.active_runs[run_id]["error"] = full_error
            app_state.active_runs[run_id]["errorDetail"] = tb[-3000:]
            app_state.active_runs[run_id]["message"] = f"Stage {stage} failed — {full_error[:300]}"
        if isinstance(e, (asyncio.CancelledError, KeyboardInterrupt)):
            raise


async def execute_full_pipeline(run_id: str, config: Dict[str, Any]):
    """Execute all pipeline stages (0-4) in sequence."""
    try:
        video_id = config.get("videoId")
        camera_id = config.get("cameraId")
        smoke_test = bool(config.get("smokeTest", False))
        use_cpu = bool(config.get("useCpu", False))
        reid_model_path = config.get("reidModelPath")

        if not video_id or video_id not in app_state.uploaded_videos:
            raise RuntimeError("Full pipeline requires a valid videoId")

        if not camera_id:
            camera_id = _detect_camera_for_video(app_state.uploaded_videos[video_id], None)

        app_state.active_runs[run_id]["cameraId"] = camera_id

        app_state.active_runs[run_id]["message"] = "Running full pipeline (stages 0-4)..."
        app_state.active_runs[run_id]["progress"] = 5

        run_meta = await _run_pipeline_stages(
            run_id=run_id,
            stages="0,1,2,3,4",
            video_id=video_id,
            camera_id=camera_id,
            use_cpu=use_cpu,
            smoke_test=smoke_test,
            reid_model_path=reid_model_path,
        )

        app_state.video_to_latest_run[video_id] = run_id
        _persist_probe_link(video_id, run_id)
        app_state.active_runs[run_id]["status"] = "completed"
        app_state.active_runs[run_id]["progress"] = 100
        app_state.active_runs[run_id]["message"] = "Full pipeline complete"
        app_state.active_runs[run_id]["runDir"] = run_meta["runDir"]
        app_state.active_runs[run_id]["completedAt"] = datetime.now().isoformat()

    except BaseException as e:
        tb = _traceback.format_exc()
        err_type = type(e).__name__
        err_msg = str(e) or f"({err_type} with no message)"
        full_error = f"{err_type}: {err_msg}"
        print(f"[PIPELINE ERROR] full-pipeline run={run_id}\n{tb}", flush=True)
        if run_id in app_state.active_runs:
            app_state.active_runs[run_id]["status"] = "error"
            app_state.active_runs[run_id]["error"] = full_error
            app_state.active_runs[run_id]["errorDetail"] = tb[-3000:]
            app_state.active_runs[run_id]["message"] = f"Pipeline failed — {full_error[:300]}"
        if isinstance(e, (asyncio.CancelledError, KeyboardInterrupt)):
            raise


async def _execute_dataset_pipeline(run_id: str, dataset_path: Path, folder_name: str):
    """Background task: run stages 0-4 on a full dataset folder."""
    try:
        stage_nums = [0, 1, 2, 3, 4]
        app_state.active_runs[run_id]["message"] = "Preparing run-local dataset input copy..."
        app_state.active_runs[run_id]["progress"] = 1
        input_dir = _prepare_dataset_input_for_run(run_id, dataset_path)

        cmd = _build_pipeline_cmd(
            stages="0,1,2,3,4",
            run_id=run_id,
            input_dir=input_dir.as_posix(),
        )

        app_state.active_runs[run_id]["message"] = "Running Ingestion & Pre-Processing..."
        app_state.active_runs[run_id]["progress"] = 2

        run_meta = await _run_pipeline_streaming(run_id, cmd, stage_nums)

        scene_prefix = folder_name.upper()
        for vid_id, vid_meta in list(app_state.uploaded_videos.items()):
            cam_id = _extract_camera_id(str(vid_meta.get("path", "")))
            if cam_id and cam_id.startswith(f"{scene_prefix}_"):
                app_state.video_to_latest_run[vid_id] = run_id

        app_state.active_runs[run_id]["status"] = "completed"
        app_state.active_runs[run_id]["progress"] = 100
        app_state.active_runs[run_id]["message"] = f"Pipeline complete for {folder_name}"
        app_state.active_runs[run_id]["runDir"] = run_meta["runDir"]
        app_state.active_runs[run_id]["completedAt"] = datetime.now().isoformat()

    except Exception as e:
        if run_id in app_state.active_runs:
            app_state.active_runs[run_id]["status"] = "error"
            app_state.active_runs[run_id]["error"] = str(e)
            app_state.active_runs[run_id]["message"] = f"Error: {str(e)[:200]}"
