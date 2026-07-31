"""Ingest: the web entry point for footage.

v1's flow started at "Upload"; v2's web app previously had no way to add
footage at all (runs only existed via the CLI). This router closes that
gap: investigators upload evidence videos here (hashed on arrival for
chain-of-custody), then submit them as a pipeline job via POST /jobs.
"""

from __future__ import annotations

import hashlib
import re
import uuid

from fastapi import APIRouter, Form, HTTPException, UploadFile, status

from athar.api import audit
from athar.api.deps import DbDep, InvestigatorUser, RequireViewer, ServicesDep
from athar.api.schemas import IngestProfileOut, UploadOut
from athar.profiles.builtin import BUILTIN_PROFILES

router = APIRouter(prefix="/ingest", tags=["ingest"], dependencies=[RequireViewer])

# camera ids and batch ids become path components — slug-only, no traversal
_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

PROFILE_BLURBS = {
    "multiclass": "Baseline person + vehicle pipeline (fastest)",
    "production": "Adds CLIP-SENet + DINOv2 vehicle streams (best accuracy)",
}

_CHUNK = 4 * 1024 * 1024


@router.get("/profiles")
def list_profiles() -> list[IngestProfileOut]:
    return [
        IngestProfileOut(name=name, description=PROFILE_BLURBS.get(name, ""))
        for name in sorted(BUILTIN_PROFILES)
    ]


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_video(
    services: ServicesDep,
    db: DbDep,
    user: InvestigatorUser,
    file: UploadFile,
    camera_id: str = Form(...),
    batch_id: str = Form(default=""),
) -> UploadOut:
    """Store one evidence video under the uploads root, hashing it while
    it streams in. The returned ``path`` is what POST /jobs expects in its
    ``videos`` map."""
    if not _SLUG.match(camera_id):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "camera_id must be a slug (letters, digits, _ or -)",
        )
    if batch_id and not _SLUG.match(batch_id):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid batch_id")
    batch = batch_id or f"batch-{uuid.uuid4().hex[:8]}"

    name = file.filename or ""
    suffix = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if suffix not in ("mp4", "avi", "mkv", "mov", "ts"):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"unsupported video extension {suffix!r} (mp4/avi/mkv/mov/ts)",
        )

    dest_dir = services.settings.uploads_root / batch / camera_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"video.{suffix}"

    digest = hashlib.sha256()
    size = 0
    with open(dest, "wb") as out:
        while chunk := await file.read(_CHUNK):
            digest.update(chunk)
            out.write(chunk)
            size += len(chunk)
    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty upload")

    audit.append(
        db, user.username, "footage_uploaded",
        batch_id=batch, camera_id=camera_id, filename=file.filename,
        size_bytes=size, sha256=digest.hexdigest(),
    )
    return UploadOut(
        batch_id=batch,
        camera_id=camera_id,
        path=str(dest),
        size_bytes=size,
        sha256=digest.hexdigest(),
    )
