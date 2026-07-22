"""Run construction shared by every entry point (CLI, job worker, API).

One place builds a runnable (manifest, profile) pair — profile loading,
config layering, evidence ingest, resume validation — so a run submitted
through the API is bit-identical to one typed at the CLI. Entry points only
differ in how they report progress.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from athar.contracts.manifest import RunManifest, RunRole
from athar.contracts.store import FilesystemRunStore
from athar.profiles.base import RunProfile


class RunSetupError(RuntimeError):
    pass


def default_stages() -> list:
    """The five-stage offline DAG, with component registration side effects."""
    import athar.components.adapters  # noqa: F401 — registers detector/tracker/embedders
    import athar.components.framesources  # noqa: F401 — registers frame sources
    from athar.pipeline.stages.associate import AssociateStage
    from athar.pipeline.stages.detect_track import DetectTrackStage
    from athar.pipeline.stages.embed import EmbedStage
    from athar.pipeline.stages.index import IndexStage
    from athar.pipeline.stages.package import PackageStage

    return [DetectTrackStage(), EmbedStage(), IndexStage(), AssociateStage(),
            PackageStage()]


def create_run(
    profile_name: str,
    videos: Mapping[str, str],
    role: str = "gallery",
    fps: Optional[float] = None,
    overrides: Optional[Sequence[str]] = None,
    on_ingest: Optional[Callable[[str, object], None]] = None,
) -> tuple[RunManifest, RunProfile]:
    """New manifest with frozen config + ingested evidence.

    ``videos`` maps camera id -> video path or image-sequence directory;
    ``fps`` is required for image directories (MANUAL TimeBase, D10).
    Raises ProfileError / ConfigAuthoringError / IngestError from the
    respective subsystems and :class:`RunSetupError` for bad arguments.
    """
    from athar.contracts.authoring import parse_dotted_overrides
    from athar.contracts.config import ConfigLayer, ResolvedConfig
    from athar.core.ids import new_run_id
    from athar.pipeline.ingest import ingest_video
    from athar.profiles.builtin import load_profile

    if not videos:
        raise RunSetupError("at least one camera video is required")
    profile, defaults = load_profile(profile_name)
    layers = [(ConfigLayer.PROFILE_DEFAULT, defaults)]
    if overrides:
        layers.append((ConfigLayer.RUN_OVERRIDE, parse_dotted_overrides(list(overrides))))
    manifest = RunManifest(
        run_id=new_run_id(), role=RunRole(role), profile_name=profile.name
    )
    manifest.config = ResolvedConfig.resolve(layers)
    for cam, path in videos.items():
        timebase = None
        if Path(path).is_dir() and fps:
            from athar.core.timebase import CameraTimeBase, TimeBaseSource

            timebase = CameraTimeBase(
                camera_id=cam, fps=fps, source=TimeBaseSource.MANUAL
            )
        video = ingest_video(manifest, cam, path, timebase=timebase)
        if on_ingest is not None:
            on_ingest(cam, video)
    return manifest, profile


def resume_run(
    store: FilesystemRunStore, run_id: str, profile_name: str
) -> tuple[RunManifest, RunProfile]:
    """Load an interrupted run for resumption; the profile must match the
    one the run was created with (frozen-config guard stays intact)."""
    from athar.profiles.builtin import load_profile

    profile, _defaults = load_profile(profile_name)
    manifest = store.load(run_id)  # RunNotFound propagates
    if manifest.profile_name != profile.name:
        raise RunSetupError(
            f"run {manifest.run_id} was created with profile "
            f"{manifest.profile_name!r}, not {profile.name!r}"
        )
    return manifest, profile
