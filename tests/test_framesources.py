"""FrameSource contract tests.

Every video frame is a solid color encoding its own index (blue = 40 + 5*i),
so ordering, sampling plans, and frame-exactness are verifiable through lossy
codecs: decode → recover index from pixel value → must equal the claimed
original index. PNG image-dir is the lossless path.

Decoder-specific classes skip when their dependency is missing, so the suite
passes under the v1 test env (cv2 only) and widens under .venv-v2.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from athar.components.framesources import (
    DecodedFrameBatch,
    FrameSourceError,
    ImageDirFrameSource,
    OpenCVFrameSource,
    create_video_source,
    pts_deviation_s,
)
from athar.components.protocols import ComponentKindName, FrameSource
from athar.components.registry import registry

cv2 = pytest.importorskip("cv2")

FPS = 20.0
FRAMES = 32
SIZE = (64, 48)  # (w, h)


def _color_for(i: int) -> int:
    return 16 + 7 * i


_codebook: list[float] = []  # per-index decoded means, measured in the fixture


def _index_from(batch_images: np.ndarray, n: int) -> int:
    """Nearest neighbor against the MEASURED post-codec values — immune to
    uniform YUV/limited-range shifts (mp4v decodes ~4 units dark), which are
    a codec property, not a frame-ordering bug."""
    v = float(batch_images[n, :, :, 0].mean())
    return min(range(FRAMES), key=lambda i: abs(v - _codebook[i]))


@pytest.fixture(scope="module")
def coded_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("vid") / "cam01.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    assert writer.isOpened(), "cv2 mp4v writer unavailable"
    for i in range(FRAMES):
        writer.write(np.full((SIZE[1], SIZE[0], 3), _color_for(i), dtype=np.uint8))
    writer.release()

    # Measure the post-codec value of every frame with a plain sequential
    # read — the ground-truth codebook _index_from decodes against.
    _codebook.clear()
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, img = cap.read()
        if not ok:
            break
        _codebook.append(float(img[:, :, 0].mean()))
    cap.release()
    assert len(_codebook) == FRAMES
    assert _codebook == sorted(_codebook), "ramp must stay monotonic post-codec"
    return path


@pytest.fixture(scope="module")
def coded_image_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("frames")
    for i in range(FRAMES):
        cv2.imwrite(
            str(d / f"{i:06d}.png"),
            np.full((SIZE[1], SIZE[0], 3), _color_for(i), dtype=np.uint8),
        )
    return d


def _available(module: str) -> bool:
    if module == "torchcodec":
        # find_spec is not enough: torchcodec imports but fails at DLL load
        # when FFmpeg libs are absent — probe the real decoder import.
        try:
            from athar.components.framesources.video import _ensure_ffmpeg_dlls

            _ensure_ffmpeg_dlls()
            from torchcodec.decoders import VideoDecoder  # noqa: F401

            return True
        except Exception:
            return False
    return importlib.util.find_spec(module) is not None


def _video_sources():
    params = [pytest.param(OpenCVFrameSource, id="opencv")]
    if _available("av"):
        from athar.components.framesources import PyAVFrameSource

        params.append(pytest.param(PyAVFrameSource, id="pyav"))
    if _available("torchcodec"):
        from athar.components.framesources import TorchcodecFrameSource

        params.append(pytest.param(TorchcodecFrameSource, id="torchcodec"))
    return params


@pytest.fixture(params=_video_sources())
def video_source_cls(request):
    return request.param


class TestVideoSources:
    def test_full_iteration_is_frame_exact(self, video_source_cls, coded_video):
        source = video_source_cls("cam01", coded_video)
        assert isinstance(source, FrameSource)
        seen = []
        for batch in source.batches(batch_size=7):
            assert len(batch) <= 7
            imgs = batch.images()
            assert imgs.dtype == np.uint8 and imgs.shape[1:] == (SIZE[1], SIZE[0], 3)
            for n, idx in enumerate(batch.frame_indices):
                assert _index_from(imgs, n) == idx, "decoded content != claimed index"
            seen.extend(batch.frame_indices)
        assert seen == list(range(FRAMES))

    def test_sampling_preserves_original_indices(self, video_source_cls, coded_video):
        source = video_source_cls("cam01", coded_video, start=5, stop=30, step=5)
        seen = []
        for batch in source.batches(batch_size=2):
            for n, idx in enumerate(batch.frame_indices):
                assert _index_from(batch.images(), n) == idx
            seen.extend(batch.frame_indices)
        assert seen == [5, 10, 15, 20, 25]

    def test_missing_file_raises(self, video_source_cls, tmp_path):
        with pytest.raises(FrameSourceError, match="not found"):
            video_source_cls("cam01", tmp_path / "nope.mp4")

    def test_invalid_sampling_rejected(self, video_source_cls, coded_video):
        with pytest.raises(FrameSourceError, match="invalid sampling"):
            video_source_cls("cam01", coded_video, start=10, stop=5)

    def test_nominal_fps_probed(self, video_source_cls, coded_video):
        source = video_source_cls("cam01", coded_video)
        assert source.nominal_fps == pytest.approx(FPS, abs=0.5)


@pytest.mark.skipif(not _available("av"), reason="pyav not installed")
class TestPts:
    def test_pts_present_and_cfr_deviation_near_zero(self, coded_video):
        from athar.components.framesources import PyAVFrameSource

        (batch,) = list(PyAVFrameSource("cam01", coded_video).batches(batch_size=FRAMES))
        assert batch.pts_s is not None
        assert list(batch.pts_s) == sorted(batch.pts_s)
        deviation = pts_deviation_s(batch, FPS)
        assert deviation is not None and deviation < 1e-3


class TestImageDir:
    def test_lossless_roundtrip(self, coded_image_dir):
        source = ImageDirFrameSource("cam01", coded_image_dir, fps=FPS)
        batches = list(source.batches(batch_size=12))
        assert [len(b) for b in batches] == [12, 12, 8]
        for batch in batches:
            for n, idx in enumerate(batch.frame_indices):
                assert int(batch.images()[n, 0, 0, 0]) == _color_for(idx)  # exact

    def test_empty_dir_rejected(self, tmp_path):
        with pytest.raises(FrameSourceError, match="no image files"):
            ImageDirFrameSource("cam01", tmp_path)


class TestRegistryWiring:
    def test_all_names_registered(self):
        names = set(registry.names(ComponentKindName.FRAME_SOURCE))
        assert {"video", "video_torchcodec", "video_pyav", "video_opencv",
                "image_dir"} <= names

    def test_auto_source_decodes(self, coded_video):
        source = registry.create(
            ComponentKindName.FRAME_SOURCE, "video",
            camera_id="cam01", path=coded_video, stop=4,
        )
        (batch,) = list(source.batches(batch_size=8))
        assert batch.frame_indices == (0, 1, 2, 3)

    def test_auto_source_uses_best_available(self, coded_video):
        expected = "OpenCVFrameSource"
        if _available("av"):
            expected = "PyAVFrameSource"
        if _available("torchcodec"):
            expected = "TorchcodecFrameSource"
        assert type(create_video_source("cam01", coded_video)).__name__ == expected


class TestBatchValidation:
    def test_shape_mismatch_rejected(self):
        with pytest.raises(FrameSourceError, match="!= 2 indices"):
            DecodedFrameBatch(
                camera_id="c", frame_indices=(0, 1),
                _images=np.zeros((3, 4, 4, 3), dtype=np.uint8),
            )

    def test_bad_layout_rejected(self):
        with pytest.raises(FrameSourceError, match=r"\(N, H, W, 3\)"):
            DecodedFrameBatch(
                camera_id="c", frame_indices=(0,),
                _images=np.zeros((1, 4, 4), dtype=np.uint8),
            )
