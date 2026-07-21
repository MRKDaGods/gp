"""Normalize-on-ingest (D11): the evidence boundary.

Every video entering ATHAR passes through here exactly once:

1. The original file is SHA-256 hashed (chain of custody) — before anything
   else touches it.
2. The container is probed (duration, fps, geometry, codec, frame count).
3. A per-camera TimeBase is declared (``ASSUMED`` unless better provenance
   is supplied) — never an implicit "cameras are in sync".
4. The manifest records inputs + timebase; downstream stages trust only the
   manifest.

Transcode-to-canonical and fisheye dewarp plug in here later as ingest
transforms (each appends to ``VideoInput.transforms``); the boundary and
records exist from day one so adding them never changes the contract.

Decoding is NOT done here — frames are decoded on demand by FrameSource
implementations (torchcodec primary, per docs/STACK.md).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from athar.contracts.manifest import RunManifest, VideoInput
from athar.core.timebase import CameraTimeBase, TimeBaseSource


class IngestError(ValueError):
    pass


@dataclass(frozen=True)
class VideoProbe:
    """Container metadata for one evidence video."""

    duration_s: Optional[float]
    fps: Optional[float]
    width: Optional[int]
    height: Optional[int]
    frame_count: Optional[int]
    codec: Optional[str]


def hash_file(path: Path | str, chunk_bytes: int = 1 << 20) -> str:
    """Streaming SHA-256 of the ORIGINAL evidence file."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_image_files(path: Path | str) -> list[Path]:
    """The frame files of an image-directory evidence source, in name order
    (the same order ImageDirFrameSource decodes them)."""
    root = Path(path)
    files = sorted(p for p in root.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise IngestError(f"no image files in directory: {root}")
    return files


def hash_image_dir(path: Path | str) -> str:
    """Manifest hash of an image-sequence directory: SHA-256 over
    (filename, content-sha256) rows in name order. Renaming, reordering,
    adding, removing, or editing any frame changes the hash."""
    digest = hashlib.sha256()
    for f in list_image_files(path):
        digest.update(f.name.encode("utf-8"))
        digest.update(bytes.fromhex(hash_file(f)))
    return digest.hexdigest()


def probe_image_dir(path: Path | str) -> VideoProbe:
    """Geometry from the first frame; an image sequence has no intrinsic
    fps, so the caller must declare a TimeBase."""
    import cv2  # noqa: PLC0415 — decode dependency

    files = list_image_files(path)
    first = cv2.imread(str(files[0]))
    if first is None:
        raise IngestError(f"cannot read first image: {files[0]}")
    height, width = first.shape[:2]
    return VideoProbe(
        duration_s=None,
        fps=None,
        width=width,
        height=height,
        frame_count=len(files),
        codec="image_dir",
    )


def probe_video(path: Path | str) -> VideoProbe:
    """Probe container metadata; PyAV preferred, OpenCV fallback.

    PyAV reads container headers (fast, accurate for VFR detection later);
    the cv2 fallback keeps probing functional in environments without av.
    """
    path = Path(path)
    if not path.is_file():
        raise IngestError(f"video not found: {path}")
    try:
        return _probe_pyav(path)
    except ImportError:
        return _probe_cv2(path)


def _probe_pyav(path: Path) -> VideoProbe:
    import av  # noqa: PLC0415 — optional dependency

    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            raise IngestError(f"no video stream in {path}")
        fps = float(stream.average_rate) if stream.average_rate else None
        duration_s = (
            float(container.duration / av.time_base) if container.duration else None
        )
        return VideoProbe(
            duration_s=duration_s,
            fps=fps,
            width=stream.codec_context.width or None,
            height=stream.codec_context.height or None,
            frame_count=stream.frames or None,
            codec=stream.codec_context.name,
        )


def _probe_cv2(path: Path) -> VideoProbe:
    import cv2  # noqa: PLC0415 — fallback path

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise IngestError(f"cannot open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or None
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        return VideoProbe(
            duration_s=(frame_count / fps) if fps and frame_count else None,
            fps=fps,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None,
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None,
            frame_count=frame_count,
            codec=None,
        )
    finally:
        cap.release()


def ingest_video(
    manifest: RunManifest,
    camera_id: str,
    path: Path | str,
    timebase: Optional[CameraTimeBase] = None,
) -> VideoInput:
    """Register one evidence video on a run: hash, probe, declare timebase.

    A camera may only be ingested once per run; the declared (or assumed)
    TimeBase must agree with the probed fps when both are present.
    """
    path = Path(path)
    if any(v.camera_id == camera_id for v in manifest.inputs):
        raise IngestError(f"camera {camera_id!r} already ingested on run {manifest.run_id}")

    if path.is_dir():
        sha256 = hash_image_dir(path)
        probe = probe_image_dir(path)
    else:
        sha256 = hash_file(path)
        probe = probe_video(path)

    if timebase is None:
        if probe.fps is None:
            raise IngestError(
                f"cannot assume a TimeBase for {camera_id!r}: probe found no fps; "
                "pass an explicit CameraTimeBase"
            )
        timebase = CameraTimeBase(
            camera_id=camera_id, fps=probe.fps, source=TimeBaseSource.ASSUMED
        )
    elif timebase.camera_id != camera_id:
        raise IngestError(
            f"timebase camera_id {timebase.camera_id!r} != ingested camera {camera_id!r}"
        )

    fps = probe.fps if probe.fps is not None else timebase.fps
    duration_s = probe.duration_s
    if duration_s is None and probe.frame_count and fps:
        duration_s = probe.frame_count / fps

    video = VideoInput(
        camera_id=camera_id,
        original_path=str(path),
        sha256=sha256,
        duration_s=duration_s,
        fps=fps,
        width=probe.width,
        height=probe.height,
    )
    manifest.inputs.append(video)
    manifest.timebase.cameras[camera_id] = timebase
    return video
