"""Filesystem run store: the ONE place that knows where runs live on disk.

Every other subsystem resolves runs through this repository. There is exactly
one run root; roles are manifest attributes; listing/filtering reads
manifests, never directory-name conventions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from athar.contracts.manifest import RunManifest, RunRole

MANIFEST_FILENAME = "manifest.json"


class RunNotFound(KeyError):
    pass


class FilesystemRunStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        return self.root / run_id

    def artifact_path(self, manifest: RunManifest, artifact_name: str) -> Path:
        record = manifest.require_artifact(artifact_name)
        return self.run_dir(manifest.run_id) / record.relpath

    def save(self, manifest: RunManifest) -> Path:
        run_dir = self.run_dir(manifest.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / MANIFEST_FILENAME
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)  # atomic on the same filesystem
        return path

    def load(self, run_id: str) -> RunManifest:
        path = self.run_dir(run_id) / MANIFEST_FILENAME
        if not path.is_file():
            raise RunNotFound(run_id)
        return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self, role: Optional[RunRole] = None) -> Iterator[RunManifest]:
        for entry in sorted(self.root.iterdir()):
            if not (entry / MANIFEST_FILENAME).is_file():
                continue
            manifest = self.load(entry.name)
            if role is None or manifest.role == role:
                yield manifest
