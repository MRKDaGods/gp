"""Tests for the normalize-on-ingest boundary (D11)."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from athar.contracts.manifest import RunManifest, RunRole
from athar.core.ids import new_run_id
from athar.core.timebase import CameraTimeBase, TimeBaseSource
from athar.pipeline.ingest import (
    hash_image_dir,
    IngestError,
    hash_file,
    ingest_video,
    probe_video,
)

cv2 = pytest.importorskip("cv2")

FPS = 20.0
FRAMES = 40
SIZE = (64, 48)  # (w, h)


@pytest.fixture()
def tiny_video(tmp_path):
    path = tmp_path / "cam01.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE
    )
    assert writer.isOpened(), "cv2 mp4v writer unavailable"
    rng = np.random.default_rng(7)
    for _ in range(FRAMES):
        writer.write(rng.integers(0, 255, (SIZE[1], SIZE[0], 3), dtype=np.uint8))
    writer.release()
    return path


def _manifest() -> RunManifest:
    return RunManifest(run_id=new_run_id(), role=RunRole.GALLERY, profile_name="test")


class TestHashing:
    def test_matches_reference_sha256(self, tmp_path):
        f = tmp_path / "evidence.bin"
        payload = b"chain of custody" * 1000
        f.write_bytes(payload)
        assert hash_file(f) == hashlib.sha256(payload).hexdigest()


class TestProbe:
    def test_probe_reads_container_metadata(self, tiny_video):
        probe = probe_video(tiny_video)
        assert probe.fps == pytest.approx(FPS, abs=0.5)
        assert probe.width == SIZE[0]
        assert probe.height == SIZE[1]
        assert probe.frame_count == FRAMES

    def test_missing_file_is_ingest_error(self, tmp_path):
        with pytest.raises(IngestError, match="not found"):
            probe_video(tmp_path / "nope.mp4")


class TestIngest:
    def test_ingest_populates_manifest_and_timebase(self, tiny_video):
        manifest = _manifest()
        video = ingest_video(manifest, "cam01", tiny_video)

        assert video.sha256 == hash_file(tiny_video)
        assert manifest.inputs == [video]
        tb = manifest.timebase.require("cam01")
        assert tb.source is TimeBaseSource.ASSUMED
        assert tb.fps == pytest.approx(FPS, abs=0.5)

    def test_explicit_timebase_wins(self, tiny_video):
        manifest = _manifest()
        tb = CameraTimeBase(
            camera_id="cam01", fps=FPS, offset_s=12.5,
            source=TimeBaseSource.MANUAL, confidence=0.9,
        )
        ingest_video(manifest, "cam01", tiny_video, timebase=tb)
        assert manifest.timebase.require("cam01").offset_s == 12.5
        assert manifest.timebase.require("cam01").source is TimeBaseSource.MANUAL

    def test_duplicate_camera_rejected(self, tiny_video):
        manifest = _manifest()
        ingest_video(manifest, "cam01", tiny_video)
        with pytest.raises(IngestError, match="already ingested"):
            ingest_video(manifest, "cam01", tiny_video)

    def test_mismatched_timebase_camera_rejected(self, tiny_video):
        manifest = _manifest()
        tb = CameraTimeBase(camera_id="other", fps=FPS)
        with pytest.raises(IngestError, match="!= ingested camera"):
            ingest_video(manifest, "cam01", tiny_video, timebase=tb)


@pytest.fixture()
def tiny_image_dir(tmp_path):
    root = tmp_path / "cam_frames"
    root.mkdir()
    rng = np.random.default_rng(11)
    for i in range(6):
        img = rng.integers(0, 255, (SIZE[1], SIZE[0], 3), dtype=np.uint8)
        cv2.imwrite(str(root / f"{i * 5:08d}.png"), img)
    (root / "notes.txt").write_text("ignored non-image file")
    return root


class TestImageDirIngest:
    def test_requires_declared_timebase(self, tiny_image_dir):
        with pytest.raises(IngestError, match="TimeBase"):
            ingest_video(_manifest(), "cam01", tiny_image_dir)

    def test_ingests_with_manual_fps(self, tiny_image_dir):
        manifest = _manifest()
        tb = CameraTimeBase(camera_id="cam01", fps=2.0, source=TimeBaseSource.MANUAL)
        video = ingest_video(manifest, "cam01", tiny_image_dir, timebase=tb)
        assert video.fps == 2.0
        assert video.duration_s == pytest.approx(6 / 2.0)
        assert video.width == SIZE[0] and video.height == SIZE[1]
        assert manifest.timebase.require("cam01").source is TimeBaseSource.MANUAL

    def test_hash_covers_names_and_content(self, tiny_image_dir):
        base = hash_image_dir(tiny_image_dir)
        assert base == hash_image_dir(tiny_image_dir)  # deterministic

        renamed = tiny_image_dir / "00000005.png"
        moved = tiny_image_dir / "99999999.png"
        renamed.rename(moved)
        assert hash_image_dir(tiny_image_dir) != base
        moved.rename(renamed)
        assert hash_image_dir(tiny_image_dir) == base

        payload = renamed.read_bytes()
        renamed.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
        assert hash_image_dir(tiny_image_dir) != base

    def test_empty_dir_rejected(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(IngestError, match="no image files"):
            ingest_video(_manifest(), "cam01", empty)
