"""Evidence clips: transcode a scene-time span of an evidence video into a
browser-playable H.264 MP4.

The package stage reserved the ``clip`` field in ``report_inputs.json`` for
exactly this. Clips are cut on demand (serving-time work, not a pipeline
stage — most spans are never viewed) and cached under ``<run_dir>/clips/``
so repeated viewing never re-transcodes.

Decode/encode both go through PyAV: its wheel bundles FFmpeg with libx264,
so no system FFmpeg install is needed (air-gap friendly, same reasoning as
the torchcodec DLL trick in framesources/video.py). Decode is sequential
from the nearest keyframe before the span — frame-exact by pts, never by
container seeking arithmetic.
"""

from __future__ import annotations

import logging
import os
from fractions import Fraction
from pathlib import Path

from athar.core.timebase import CameraTimeBase

logger = logging.getLogger(__name__)

# libx264 ships inside the PyAV wheel; nvenc only exists with a capable
# driver, so it is a bonus first choice never a requirement.
ENCODER_CANDIDATES = ("libx264", "h264_nvenc")


class ClipError(Exception):
    """Transcode-side failure (no encoder, undecodable evidence, ...)."""


def scene_to_media_s(timebase: CameraTimeBase, scene_s: float) -> float:
    """Invert ``CameraTimeBase.to_scene`` for a scene-clock instant.

    to_scene: scene = local + offset + drift * local / 3600
    """
    return (scene_s - timebase.offset_s) / (1.0 + timebase.drift_s_per_hour / 3600.0)


def _open_encoder(container, template_stream, rate: Fraction):
    import av

    last_error: Exception | None = None
    for name in ENCODER_CANDIDATES:
        try:
            av.codec.Codec(name, "w")
        except av.codec.codec.UnknownCodecError:
            continue
        try:
            stream = container.add_stream(name, rate=rate)
        except av.FFmpegError as exc:  # e.g. nvenc present but no device
            last_error = exc
            continue
        # yuv420p needs even dimensions; crop a single row/column if odd.
        stream.width = template_stream.width - template_stream.width % 2
        stream.height = template_stream.height - template_stream.height % 2
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "veryfast", "crf": "23"}
        return stream
    raise ClipError(
        f"no H.264 encoder available (tried {', '.join(ENCODER_CANDIDATES)})"
        + (f": {last_error}" if last_error else "")
    )


def extract_clip(
    video_path: Path | str,
    out_path: Path | str,
    start_media_s: float,
    end_media_s: float,
) -> Path:
    """Cut ``[start_media_s, end_media_s]`` (media time) into an MP4.

    Writes to a temp file next to ``out_path`` and renames atomically, so a
    concurrent request for the same clip either sees nothing or the whole
    file.
    """
    import av

    video_path = Path(video_path)
    out_path = Path(out_path)
    if not video_path.is_file():
        raise ClipError(f"evidence video not found: {video_path}")
    if end_media_s <= start_media_s:
        raise ValueError(f"empty clip span: [{start_media_s}, {end_media_s}]")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(f".tmp-{os.getpid()}.mp4")
    frames_written = 0
    try:
        with av.open(str(video_path)) as src:
            in_stream = next((s for s in src.streams if s.type == "video"), None)
            if in_stream is None:
                raise ClipError(f"no video stream in {video_path}")
            in_stream.thread_type = "AUTO"
            rate = Fraction(in_stream.average_rate or Fraction(25, 1))
            try:  # land on the keyframe at/before the span start
                src.seek(int(start_media_s * av.time_base), backward=True)
            except av.FFmpegError:
                pass  # containers without an index decode from the top

            with av.open(
                str(tmp_path), "w", format="mp4", options={"movflags": "+faststart"}
            ) as out:
                out_stream = _open_encoder(out, in_stream, rate)
                fallback_t = start_media_s  # pts-less frames: assume nominal rate
                for frame in src.decode(in_stream):
                    t = frame.time if frame.time is not None else fallback_t
                    fallback_t = t + 1.0 / float(rate)
                    if t < start_media_s:
                        continue
                    if t > end_media_s:
                        break
                    reformatted = frame.reformat(
                        width=out_stream.width,
                        height=out_stream.height,
                        format="yuv420p",
                    )
                    reformatted.pts = frames_written
                    reformatted.time_base = Fraction(rate.denominator, rate.numerator)
                    for packet in out_stream.encode(reformatted):
                        out.mux(packet)
                    frames_written += 1
                for packet in out_stream.encode(None):
                    out.mux(packet)
    except av.FFmpegError as exc:
        tmp_path.unlink(missing_ok=True)
        raise ClipError(f"transcode failed for {video_path}: {exc}") from exc
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    if frames_written == 0:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(
            f"span [{start_media_s}, {end_media_s}] yields no frames "
            f"(video shorter than the requested start?)"
        )
    os.replace(tmp_path, out_path)
    logger.info(
        "clip: %s [%0.2fs, %0.2fs] -> %s (%d frames)",
        video_path.name, start_media_s, end_media_s, out_path.name, frames_written,
    )
    return out_path


def clip_for_span(
    run_dir: Path,
    video_path: Path | str,
    camera_id: str,
    timebase: CameraTimeBase,
    start_scene_s: float,
    end_scene_s: float,
    *,
    pad_s: float = 1.0,
    max_duration_s: float = 60.0,
) -> Path:
    """Cached scene-clock clip for one camera of a run.

    The span is padded by ``pad_s`` on both sides (context before/after the
    sighting), clamped to start >= 0, and capped at ``max_duration_s`` —
    the cap protects the server from a request spanning a whole tape.
    """
    if end_scene_s <= start_scene_s:
        raise ValueError(f"empty clip span: [{start_scene_s}, {end_scene_s}]")
    start = max(0.0, scene_to_media_s(timebase, start_scene_s) - pad_s)
    end = scene_to_media_s(timebase, end_scene_s) + pad_s
    if end - start > max_duration_s:
        raise ValueError(
            f"clip span {end - start:0.1f}s exceeds the {max_duration_s:0.0f}s cap"
        )
    name = (
        f"{camera_id}_{round(start_scene_s * 1000)}-{round(end_scene_s * 1000)}"
        f"_p{round(pad_s * 1000)}.mp4"
    )
    out_path = run_dir / "clips" / name
    if out_path.is_file():
        return out_path
    return extract_clip(video_path, out_path, start, end)
