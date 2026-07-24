"""Evidence clip transcode (athar/serving/clips.py): span mapping, frame
counts, caching, and the guard rails. Uses a tiny synthetic PyAV video —
no real footage, sub-second runtime."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

av = pytest.importorskip("av")

from athar.core.timebase import CameraTimeBase  # noqa: E402
from athar.serving.clips import (  # noqa: E402
    clip_for_span,
    extract_clip,
    scene_to_media_s,
)

FPS = 10
N_FRAMES = 40  # 4.0 s


@pytest.fixture(scope="module")
def source_video(tmp_path_factory):
    """40 frames, 64x48, 10 fps — each frame a distinct solid gray."""
    path = tmp_path_factory.mktemp("clips") / "src.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=Fraction(FPS, 1))
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(N_FRAMES):
            img = np.full((48, 64, 3), (i * 6) % 255, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(frame.reformat(format="yuv420p")):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return path


def _decode_count(path) -> int:
    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(container.streams.video[0]))


class TestSceneToMedia:
    def test_inverts_to_scene(self):
        tb = CameraTimeBase(camera_id="c", fps=25.0, offset_s=3.5, drift_s_per_hour=2.0)
        for frame_index in (0, 100, 90000):
            scene = tb.to_scene(frame_index)
            assert scene_to_media_s(tb, scene) == pytest.approx(frame_index / 25.0)

    def test_zero_offset_identity(self):
        tb = CameraTimeBase(camera_id="c", fps=10.0)
        assert scene_to_media_s(tb, 7.25) == 7.25


class TestExtractClip:
    def test_span_frame_count_and_codec(self, source_video, tmp_path):
        out = extract_clip(source_video, tmp_path / "a.mp4", 1.0, 2.5)
        assert out.is_file() and out.stat().st_size > 0
        # inclusive [1.0, 2.5] at 10 fps -> 16 frames
        assert _decode_count(out) == 16
        with av.open(str(out)) as container:
            assert container.streams.video[0].codec_context.name == "h264"

    def test_even_dimensions(self, source_video, tmp_path):
        out = extract_clip(source_video, tmp_path / "b.mp4", 0.0, 0.5)
        with av.open(str(out)) as container:
            stream = container.streams.video[0]
            assert stream.width % 2 == 0 and stream.height % 2 == 0

    def test_empty_span_rejected(self, source_video, tmp_path):
        with pytest.raises(ValueError, match="empty clip span"):
            extract_clip(source_video, tmp_path / "c.mp4", 2.0, 2.0)

    def test_span_past_end_rejected(self, source_video, tmp_path):
        with pytest.raises(ValueError, match="no frames"):
            extract_clip(source_video, tmp_path / "d.mp4", 100.0, 101.0)

    def test_missing_video(self, tmp_path):
        from athar.serving.clips import ClipError

        with pytest.raises(ClipError, match="not found"):
            extract_clip(tmp_path / "nope.mp4", tmp_path / "e.mp4", 0.0, 1.0)


class TestClipForSpan:
    def test_padding_and_offset(self, source_video, tmp_path):
        # offset 1s: scene [2.0, 2.8] -> media [1.0, 1.8], padded 0.5 -> [0.5, 2.3]
        tb = CameraTimeBase(camera_id="cam", fps=FPS, offset_s=1.0)
        out = clip_for_span(
            tmp_path, source_video, "cam", tb, 2.0, 2.8, pad_s=0.5
        )
        assert out.parent == tmp_path / "clips"
        assert _decode_count(out) == 19  # inclusive [0.5, 2.3] at 10 fps

    def test_pad_clamped_at_zero(self, source_video, tmp_path):
        tb = CameraTimeBase(camera_id="cam", fps=FPS)
        out = clip_for_span(
            tmp_path, source_video, "cam", tb, 0.1, 0.5, pad_s=5.0
        )
        assert out.is_file()  # start clamped to 0 rather than rejected

    def test_cache_reuse(self, source_video, tmp_path):
        tb = CameraTimeBase(camera_id="cam", fps=FPS)
        first = clip_for_span(tmp_path, source_video, "cam", tb, 1.0, 2.0)
        stamp = first.stat().st_mtime_ns
        second = clip_for_span(tmp_path, source_video, "cam", tb, 1.0, 2.0)
        assert second == first and second.stat().st_mtime_ns == stamp

    def test_duration_cap(self, source_video, tmp_path):
        tb = CameraTimeBase(camera_id="cam", fps=FPS)
        with pytest.raises(ValueError, match="cap"):
            clip_for_span(
                tmp_path, source_video, "cam", tb, 0.0, 30.0,
                pad_s=0.0, max_duration_s=10.0,
            )

    def test_no_tmp_left_behind_on_bad_span(self, source_video, tmp_path):
        tb = CameraTimeBase(camera_id="cam", fps=FPS)
        with pytest.raises(ValueError):
            clip_for_span(tmp_path, source_video, "cam", tb, 200.0, 201.0)
        leftovers = list((tmp_path / "clips").glob("*.tmp-*"))
        assert leftovers == []
