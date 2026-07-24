"""API tests: auth/sessions, RBAC, runs, jobs, models, search (404/409/
calibration), hash-chained audit. Uses TestClient over an app built on tmp
paths — no network, no worker subprocess, no ML models.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("pwdlib")
pytest.importorskip("sqlalchemy")
faiss = pytest.importorskip("faiss")

from fastapi.testclient import TestClient  # noqa: E402

from athar.api.app import create_app  # noqa: E402
from athar.api.db import Role  # noqa: E402
from athar.api.security import create_user  # noqa: E402
from athar.api.settings import ApiSettings  # noqa: E402
from athar.contracts.manifest import RunRole  # noqa: E402
from athar.contracts.store import FilesystemRunStore  # noqa: E402
from athar.search.calibration import ScoreCalibration, StreamCalibrations  # noqa: E402
from athar.serving.registry import (  # noqa: E402
    CheckpointRef,
    EvalReportRef,
    ModelEntry,
    ModelStage,
    ModelTask,
)

from test_search_engine import _unit, _write_run  # noqa: E402

PASSWORDS = {
    "admin": "pw-admin", "inv": "pw-inv", "inv2": "pw-inv2", "view": "pw-view",
}


@pytest.fixture()
def settings(tmp_path) -> ApiSettings:
    return ApiSettings(
        runs_root=tmp_path / "runs",
        jobs_db=tmp_path / "jobs" / "jobs.db",
        registry_db=tmp_path / "registry" / "models.db",
        app_db=tmp_path / "app" / "app.db",
        camera_locations=tmp_path / "camera_locations.json",
        cookie_secure=False,
        spawn_worker=False,
    )


@pytest.fixture()
def client(settings):
    app = create_app(settings)
    services = app.state.services
    with services.session_factory() as db:
        create_user(db, "admin", PASSWORDS["admin"], Role.ADMIN)
        create_user(db, "inv", PASSWORDS["inv"], Role.INVESTIGATOR)
        create_user(db, "inv2", PASSWORDS["inv2"], Role.INVESTIGATOR)
        create_user(db, "view", PASSWORDS["view"], Role.VIEWER)
        db.commit()
    with TestClient(app) as test_client:
        test_client.services = services  # test hook
        yield test_client


def login(client: TestClient, username: str) -> None:
    response = client.post(
        "/auth/login", json={"username": username, "password": PASSWORDS[username]}
    )
    assert response.status_code == 200, response.text


@pytest.fixture()
def store(settings) -> FilesystemRunStore:
    return FilesystemRunStore(settings.runs_root)


@pytest.fixture()
def base_vec():
    return np.random.default_rng(42).normal(size=12)


@pytest.fixture()
def gallery(store, base_vec):
    return _write_run(
        store, RunRole.GALLERY,
        {
            "g1": [(1, "car", _unit(1, base_vec, 0.05)), (2, "car", _unit(2))],
            "g2": [(1, "car", _unit(3, base_vec, 0.05))],
        },
        build_index=True,
    )


@pytest.fixture()
def probe(store, base_vec):
    manifest = _write_run(
        store, RunRole.PROBE,
        {"p1": [(5, "car", _unit(5, base_vec, 0.05))]},
        build_index=False,
    )
    store.save(manifest)  # API loads the probe from disk
    return manifest


class TestAuth:
    def test_health_open_everything_else_locked(self, client):
        assert client.get("/health").status_code == 200
        assert client.get("/runs").status_code == 401
        assert client.get("/jobs").status_code == 401
        assert client.get("/models").status_code == 401

    def test_login_me_logout(self, client):
        bad = client.post("/auth/login", json={"username": "inv", "password": "nope"})
        assert bad.status_code == 401
        login(client, "inv")
        me = client.get("/auth/me")
        assert me.status_code == 200
        assert me.json() == {
            "username": "inv", "role": "investigator",
            "created_at": me.json()["created_at"],
        }
        assert client.post("/auth/logout").status_code == 204
        assert client.get("/auth/me").status_code == 401  # session revoked

    def test_unknown_user_rejected(self, client):
        response = client.post(
            "/auth/login", json={"username": "ghost", "password": "x"}
        )
        assert response.status_code == 401


class TestRbac:
    def test_viewer_reads_but_cannot_mutate(self, client):
        login(client, "view")
        assert client.get("/runs").status_code == 200
        assert client.get("/jobs").status_code == 200
        denied = client.post("/jobs", json={"videos": {"c1": "x"}})
        assert denied.status_code == 403
        assert "investigator" in denied.json()["detail"]

    def test_investigator_cannot_admin(self, client):
        login(client, "inv")
        denied = client.post("/models/some-model/promote", json={"to": "validated"})
        assert denied.status_code == 403
        assert client.get("/audit").status_code == 403  # audit is admin-only


class TestRuns:
    def test_list_and_get(self, client, gallery):
        login(client, "view")
        listed = client.get("/runs").json()
        assert [r["run_id"] for r in listed] == [gallery.run_id]
        assert listed[0]["status"] == "completed"
        detail = client.get(f"/runs/{gallery.run_id}")
        assert detail.status_code == 200
        assert detail.json()["run_id"] == gallery.run_id
        assert client.get("/runs/run-nope").status_code == 404

    def test_events_list_and_stream(self, client, settings, gallery):
        login(client, "view")
        events_path = settings.runs_root / gallery.run_id / "events.jsonl"
        events_path.write_text(
            json.dumps({"event": "stage_started", "run_id": gallery.run_id,
                        "stage": "detect_track"}) + "\n"
            + json.dumps({"event": "run_completed", "run_id": gallery.run_id}) + "\n",
            encoding="utf-8",
        )
        listed = client.get(f"/runs/{gallery.run_id}/events").json()
        assert [e["event"] for e in listed] == ["stage_started", "run_completed"]
        with client.stream("GET", f"/runs/{gallery.run_id}/events/stream") as stream:
            body = "".join(stream.iter_text())
        assert "stage_started" in body and "run_completed" in body

    def test_artifact_download_and_404(self, client, gallery):
        login(client, "view")
        good = client.get(f"/runs/{gallery.run_id}/artifacts/embed.summary")
        assert good.status_code == 200
        assert "streams" in good.json()
        missing = client.get(f"/runs/{gallery.run_id}/artifacts/nope.artifact")
        assert missing.status_code == 404

    def test_report_404_without_package_stage(self, client, gallery):
        login(client, "view")
        for path in ("report", "report.html", "report.pdf"):
            response = client.get(f"/runs/{gallery.run_id}/{path}")
            assert response.status_code == 404
            assert "package" in response.json()["detail"]

    def _write_package_report(self, settings, store, gallery):
        from athar.contracts.manifest import ArtifactRecord

        report = {
            "schema_version": 1,
            "run": {"run_id": gallery.run_id, "role": "gallery",
                    "profile": "p", "config_hash": "ch" * 32,
                    "created_at": "2026-07-22"},
            "evidence": [{"camera_id": "g1", "original_path": "e.mp4",
                          "sha256": "ev" * 32, "duration_s": 1, "fps": 25}],
            "identities": [],
        }
        path = settings.runs_root / gallery.run_id / "report_inputs.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        manifest = store.load(gallery.run_id)
        manifest.register_artifact(ArtifactRecord(
            name="package.report", relpath="report_inputs.json",
            schema_version=1, producer="package",
        ))
        store.save(manifest)

    def test_report_html_export(self, client, settings, store, gallery):
        self._write_package_report(settings, store, gallery)
        login(client, "view")
        response = client.get(f"/runs/{gallery.run_id}/report.html?locale=en")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert "ev" * 32 in response.text  # video sha in the custody chain
        assert 'dir="ltr"' in response.text

    def test_report_pdf_export(self, client, settings, store, gallery, monkeypatch):
        self._write_package_report(settings, store, gallery)
        import athar.api.routers.runs as runs_router

        monkeypatch.setattr(
            runs_router, "html_to_pdf", lambda html: b"%PDF-1.4 fake"
        )
        login(client, "inv")
        response = client.get(f"/runs/{gallery.run_id}/report.pdf")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert response.content.startswith(b"%PDF")
        login(client, "admin")
        actions = [r["action"] for r in client.get("/audit").json()]
        assert "report_exported" in actions


def _write_media_run(settings, store, gallery):
    """Turn the gallery fixture into a media-backed run: a real (synthetic)
    evidence video for camera g1, a missing one for g2, a time base, one
    thumbnail, and a package.report with a cross-camera identity."""
    av = pytest.importorskip("av")
    from fractions import Fraction

    from athar.contracts.manifest import ArtifactRecord
    from athar.core.timebase import CameraTimeBase, SceneClock

    video = settings.runs_root.parent / "evidence_g1.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("libx264", rate=Fraction(10, 1))
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(40):
            img = np.full((48, 64, 3), (i * 6) % 255, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(frame.reformat(format="yuv420p")):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)

    manifest = store.load(gallery.run_id)
    manifest.inputs[0].original_path = str(video)
    manifest.inputs[0].duration_s = 4.0
    manifest.inputs[0].fps = 10.0
    manifest.timebase = SceneClock(cameras={
        "g1": CameraTimeBase(camera_id="g1", fps=10.0),
        "g2": CameraTimeBase(camera_id="g2", fps=10.0, offset_s=0.5),
    })

    run_dir = store.run_dir(gallery.run_id)
    thumb = run_dir / "thumbs" / "g1" / "1.jpg"
    thumb.parent.mkdir(parents=True)
    thumb.write_bytes(b"\xff\xd8\xff\xdbfakejpeg")

    report = {
        "schema_version": 1,
        "run": {"run_id": gallery.run_id, "role": "gallery", "profile": "p",
                "config_hash": "ch" * 32, "created_at": "2026-07-22"},
        "evidence": [],
        "identities": [
            {
                "global_id": 0, "entity_class": "car", "confidence": 0.8,
                "evidence": {"appearance": 0.7}, "cross_camera": True,
                "members": [
                    {"camera_id": "g1", "track_id": 1, "start_ts_scene_s": 0.5,
                     "end_ts_scene_s": 2.0, "thumbnail": "thumbs/g1/1.jpg",
                     "clip": None},
                    {"camera_id": "g2", "track_id": 1, "start_ts_scene_s": 2.5,
                     "end_ts_scene_s": 3.5, "thumbnail": None, "clip": None},
                ],
            },
        ],
    }
    (run_dir / "report_inputs.json").write_text(json.dumps(report), "utf-8")
    manifest.register_artifact(ArtifactRecord(
        name="package.report", relpath="report_inputs.json",
        schema_version=1, producer="package",
    ))
    store.save(manifest)


class TestTimelineAndClips:
    def test_timeline_404_without_report(self, client, gallery):
        login(client, "view")
        response = client.get(f"/runs/{gallery.run_id}/timeline")
        assert response.status_code == 404
        assert "package" in response.json()["detail"]

    def test_timeline(self, client, settings, store, gallery):
        _write_media_run(settings, store, gallery)
        login(client, "view")
        timeline = client.get(f"/runs/{gallery.run_id}/timeline").json()
        cameras = {c["camera_id"]: c for c in timeline["cameras"]}
        assert cameras["g1"]["video_on_disk"] is True
        assert cameras["g2"]["video_on_disk"] is False
        assert cameras["g2"]["scene_start_s"] == 0.5  # offset surfaces
        assert timeline["span_end_s"] == 4.0  # g1 coverage beats last sighting
        (identity,) = timeline["identities"]
        assert identity["cross_camera"] is True
        members = {m["camera_id"]: m for m in identity["members"]}
        assert members["g1"]["has_thumbnail"] and members["g1"]["clip_available"]
        assert not members["g2"]["has_thumbnail"]
        assert not members["g2"]["clip_available"]

    def test_thumbnail_and_404(self, client, settings, store, gallery):
        _write_media_run(settings, store, gallery)
        login(client, "view")
        ok = client.get(f"/runs/{gallery.run_id}/thumbs/g1/1")
        assert ok.status_code == 200
        assert ok.headers["content-type"] == "image/jpeg"
        assert client.get(f"/runs/{gallery.run_id}/thumbs/g1/99").status_code == 404
        assert client.get(f"/runs/{gallery.run_id}/thumbs/gX/1").status_code == 404

    def test_clip_transcode_and_audit(self, client, settings, store, gallery):
        _write_media_run(settings, store, gallery)
        login(client, "view")
        response = client.get(
            f"/runs/{gallery.run_id}/clips/g1", params={"start_s": 1.0, "end_s": 2.0}
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "video/mp4"
        assert b"ftyp" in response.content[:16]
        login(client, "admin")
        actions = [r["action"] for r in client.get("/audit").json()]
        assert "clip_exported" in actions

    def test_clip_guards(self, client, settings, store, gallery):
        _write_media_run(settings, store, gallery)
        login(client, "view")
        run = gallery.run_id
        params = {"start_s": 1.0, "end_s": 2.0}
        assert client.get(f"/runs/{run}/clips/gX", params=params).status_code == 404
        missing = client.get(f"/runs/{run}/clips/g2", params=params)
        assert missing.status_code == 404
        assert "not on disk" in missing.json()["detail"]
        empty = client.get(
            f"/runs/{run}/clips/g1", params={"start_s": 2.0, "end_s": 2.0}
        )
        assert empty.status_code == 400
        too_long = client.get(
            f"/runs/{run}/clips/g1", params={"start_s": 0.0, "end_s": 500.0}
        )
        assert too_long.status_code == 400
        assert "cap" in too_long.json()["detail"]


class TestCameraLocations:
    def test_empty_without_file(self, client):
        login(client, "view")
        assert client.get("/cameras/locations").json() == {"cameras": {}}

    def test_served_from_file(self, client, settings):
        settings.camera_locations.write_text(json.dumps({
            "c017": {"lat": 30.14, "lng": 31.62, "label": "Camera 17"},
        }), "utf-8")
        login(client, "view")
        payload = client.get("/cameras/locations").json()
        assert payload["cameras"]["c017"]["lat"] == 30.14
        assert payload["cameras"]["c017"]["label"] == "Camera 17"

    def test_malformed_file_503(self, client, settings):
        settings.camera_locations.write_text("{not json", "utf-8")
        login(client, "view")
        assert client.get("/cameras/locations").status_code == 503

    def test_requires_auth(self, client):
        assert client.get("/cameras/locations").status_code == 401


class TestJobs:
    def test_submit_get_cancel(self, client):
        login(client, "inv")
        submitted = client.post(
            "/jobs",
            json={"videos": {"c1": "evidence/c1.mp4"}, "role": "gallery",
                  "profile": "multiclass"},
        )
        assert submitted.status_code == 202
        job = submitted.json()
        assert job["status"] == "queued"
        fetched = client.get(f"/jobs/{job['job_id']}").json()
        assert fetched["payload"]["videos"] == {"c1": "evidence/c1.mp4"}
        cancelled = client.post(f"/jobs/{job['job_id']}/cancel").json()
        assert cancelled["status"] == "cancelled"  # queued -> immediate
        assert client.get("/jobs/job-nope").status_code == 404

    def test_submit_requires_videos_or_resume(self, client):
        login(client, "inv")
        response = client.post("/jobs", json={"videos": {}})
        assert response.status_code == 400

    def test_events_stream_404_before_run_attached(self, client):
        login(client, "inv")
        job = client.post("/jobs", json={"videos": {"c1": "x.mp4"}}).json()
        response = client.get(f"/jobs/{job['job_id']}/events/stream")
        assert response.status_code == 404


def _model_entry(model_id: str, task: ModelTask = ModelTask.REID_VEHICLE) -> ModelEntry:
    return ModelEntry(
        model_id=model_id, task=task, architecture="transreid",
        checkpoint=CheckpointRef(sha256="ab" * 32, size_bytes=1, filename="m.pth"),
    )


class TestModels:
    def test_lifecycle_over_http(self, client):
        services = client.services
        services.lifecycle.add(_model_entry("m1"))
        login(client, "admin")
        assert [m["model_id"] for m in client.get("/models").json()] == ["m1"]

        ungated = client.post("/models/m1/promote", json={"to": "validated"})
        assert ungated.status_code == 400
        assert "evaluation report" in ungated.json()["detail"]

        promoted = client.post(
            "/models/m1/promote",
            json={"to": "validated", "eval_run_id": "run-e1",
                  "benchmark": "veri776", "metrics": {"mAP": 0.93}},
        )
        assert promoted.status_code == 200
        assert promoted.json()["stage"] == "validated"

        history = client.get("/models/m1/history").json()
        assert [e["action"] for e in history] == ["register", "promote"]
        assert history[-1]["actor"] == "admin"
        assert client.get("/models/ghost").status_code == 404

    def test_rollback_over_http(self, client):
        services = client.services
        for mid in ("old", "new"):
            services.lifecycle.add(_model_entry(mid))
            services.lifecycle.promote(
                mid, ModelStage.VALIDATED,
                eval_report=EvalReportRef(run_id="e", benchmark="b", metrics={}),
            )
        login(client, "admin")
        for mid in ("old", "new"):
            response = client.post(
                f"/models/{mid}/promote",
                json={"to": "production", "eval_run_id": f"e-{mid}"},
            )
            assert response.status_code == 200
        rolled = client.post("/models/rollback", json={"task": "reid_vehicle"})
        assert rolled.status_code == 200
        assert rolled.json()["model_id"] == "old"
        assert rolled.json()["stage"] == "production"


class TestSearch:
    def test_happy_path_uncalibrated(self, client, gallery, probe):
        login(client, "inv")
        response = client.post(
            "/search",
            json={"gallery_run_id": gallery.run_id, "probe_run_id": probe.run_id},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["stream"] == "appearance"
        assert body["calibrated"] is False
        assert body["hits"], "lookalike gallery tracklets expected"
        top = body["hits"][0]
        assert top["score"] > 0.9
        assert top["probability"] is None  # no calibration -> no invented numbers

    def test_calibrated_probability(self, client, gallery, probe):
        client.services.calibrations = StreamCalibrations(
            streams={"appearance": ScoreCalibration(midpoint=0.5, scale=0.1)}
        )
        login(client, "inv")
        body = client.post(
            "/search",
            json={"gallery_run_id": gallery.run_id, "probe_run_id": probe.run_id},
        ).json()
        assert body["calibrated"] is True
        assert body["hits"][0]["probability"] > 0.9

    def test_unknown_runs_404(self, client, gallery):
        login(client, "inv")
        response = client.post(
            "/search",
            json={"gallery_run_id": gallery.run_id, "probe_run_id": "run-ghost"},
        )
        assert response.status_code == 404

    def test_incompatible_pair_409(self, client, settings, gallery, probe):
        # sabotage the probe's stream dim, as a run from another profile would be
        summary_path = settings.runs_root / probe.run_id / "embed_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["streams"]["appearance"]["dim"] = 4
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        login(client, "inv")
        response = client.post(
            "/search",
            json={"gallery_run_id": gallery.run_id, "probe_run_id": probe.run_id},
        )
        assert response.status_code == 409
        assert "dim mismatch" in response.json()["detail"]

    def test_viewer_cannot_search(self, client, gallery, probe):
        login(client, "view")
        response = client.post(
            "/search",
            json={"gallery_run_id": gallery.run_id, "probe_run_id": probe.run_id},
        )
        assert response.status_code == 403


class TestCases:
    def _create(self, client, title="Mall incident 14"):
        response = client.post("/cases", json={"title": title})
        assert response.status_code == 201, response.text
        return response.json()

    def test_ownership_scoping(self, client):
        login(client, "inv")
        case = self._create(client)
        assert case["owner"] == "inv"

        # another investigator: invisible in list, 404 (not 403) by id
        login(client, "inv2")
        assert client.get("/cases").json() == []
        assert client.get(f"/cases/{case['case_id']}").status_code == 404
        denied = client.patch(
            f"/cases/{case['case_id']}", json={"status": "closed"}
        )
        assert denied.status_code == 404  # existence is not confirmed

        # admin sees and can touch everything
        login(client, "admin")
        assert [c["case_id"] for c in client.get("/cases").json()] == [case["case_id"]]
        assert client.get(f"/cases/{case['case_id']}").status_code == 200

        # viewers cannot create cases at all
        login(client, "view")
        assert client.post("/cases", json={"title": "x"}).status_code == 403
        assert client.get("/cases").json() == []

    def test_workspace_flow_confirm(self, client, gallery, probe):
        login(client, "inv")
        case = self._create(client)
        case_id = case["case_id"]

        # attach both evidence runs; role is copied from the manifest
        attached = client.post(
            f"/cases/{case_id}/runs", json={"run_id": gallery.run_id}
        )
        assert attached.status_code == 201
        client.post(f"/cases/{case_id}/runs", json={"run_id": probe.run_id})
        detail = client.get(f"/cases/{case_id}").json()
        assert {r["run_id"]: r["role"] for r in detail["runs"]} == {
            gallery.run_id: "gallery", probe.run_id: "probe",
        }

        # target + hypothesis from a search-hit-shaped payload
        target = client.post(
            f"/cases/{case_id}/targets", json={"label": "Suspect A"}
        ).json()
        hyp = client.post(
            f"/cases/{case_id}/targets/{target['target_id']}/hypotheses",
            json={"run_id": gallery.run_id, "camera_id": "g1", "track_id": 1,
                  "raw_score": 0.97, "stream": "appearance"},
        )
        assert hyp.status_code == 201
        hyp_id = hyp.json()["hypothesis_id"]
        assert hyp.json()["status"] == "proposed"
        assert hyp.json()["probability"] is None  # uncalibrated: no invented numbers

        # confirm: attributed decision, member appears
        decided = client.post(
            f"/cases/{case_id}/targets/{target['target_id']}/hypotheses/{hyp_id}/decide",
            json={"status": "confirmed"},
        ).json()
        assert decided["status"] == "confirmed"
        assert decided["decided_by"] == "inv"
        assert decided["decided_at"] is not None
        detail = client.get(f"/cases/{case_id}").json()
        assert detail["targets"][0]["members"] == [
            {"run_id": gallery.run_id, "camera_id": "g1", "track_id": 1}
        ]

        # a decision is final — no re-deciding
        again = client.post(
            f"/cases/{case_id}/targets/{target['target_id']}/hypotheses/{hyp_id}/decide",
            json={"status": "rejected"},
        )
        assert again.status_code == 409

        # the whole trail is in the audit chain
        login(client, "admin")
        actions = [r["action"] for r in client.get("/audit").json()]
        for expected in ("case_created", "case_run_attached", "target_created",
                         "hypothesis_proposed", "hypothesis_decided"):
            assert expected in actions, expected
        assert client.get("/audit/verify").json()["intact"] is True

    def test_case_report_dossier(self, client, gallery, monkeypatch):
        login(client, "inv")
        case = self._create(client, title="Parking lot theft")
        case_id = case["case_id"]
        client.post(f"/cases/{case_id}/runs", json={"run_id": gallery.run_id})
        target = client.post(
            f"/cases/{case_id}/targets", json={"label": "Suspect vehicle"}
        ).json()
        hyp = client.post(
            f"/cases/{case_id}/targets/{target['target_id']}/hypotheses",
            json={"run_id": gallery.run_id, "camera_id": "g1", "track_id": 1,
                  "raw_score": 0.97, "stream": "appearance"},
        ).json()
        client.post(
            f"/cases/{case_id}/targets/{target['target_id']}/hypotheses/"
            f"{hyp['hypothesis_id']}/decide",
            json={"status": "confirmed"},
        )

        html = client.get(f"/cases/{case_id}/report.html?locale=en")
        assert html.status_code == 200
        assert html.headers["content-type"].startswith("text/html")
        text = html.text
        assert 'dir="ltr"' in text
        assert "Parking lot theft" in text
        assert gallery.run_id in text  # attached evidence with manifest facts
        assert "confirmed" in text  # the decision, not just the proposal
        assert "uncalibrated" in text  # null probability never invented
        # the audit slice made it in (actions from the case's own trail)
        for action in ("case_created", "hypothesis_decided"):
            assert action in text, action

        import athar.api.routers.cases as cases_router

        monkeypatch.setattr(
            "athar.reporting.html_to_pdf", lambda html: b"%PDF-1.4 fake"
        )
        assert cases_router  # keep the import for the monkeypatch target
        pdf = client.get(f"/cases/{case_id}/report.pdf")
        assert pdf.status_code == 200
        assert pdf.content.startswith(b"%PDF")

        # need-to-know: another investigator cannot export the dossier
        login(client, "inv2")
        assert client.get(f"/cases/{case_id}/report.html").status_code == 404

        login(client, "admin")
        actions = [r["action"] for r in client.get("/audit").json()]
        assert actions.count("case_report_exported") == 2
        assert client.get("/audit/verify").json()["intact"] is True

    def test_case_report_missing_run_manifest(self, client, settings, gallery):
        import shutil

        login(client, "inv")
        case = self._create(client, title="Ghost run case")
        case_id = case["case_id"]
        client.post(f"/cases/{case_id}/runs", json={"run_id": gallery.run_id})
        shutil.rmtree(settings.runs_root / gallery.run_id)
        html = client.get(f"/cases/{case_id}/report.html?locale=en")
        assert html.status_code == 200
        assert "run manifest not on disk" in html.text

    def test_reject_adds_no_member(self, client, gallery):
        login(client, "inv")
        case_id = self._create(client)["case_id"]
        client.post(f"/cases/{case_id}/runs", json={"run_id": gallery.run_id})
        target = client.post(
            f"/cases/{case_id}/targets", json={"label": "Suspect B"}
        ).json()
        hyp = client.post(
            f"/cases/{case_id}/targets/{target['target_id']}/hypotheses",
            json={"run_id": gallery.run_id, "camera_id": "g2", "track_id": 1,
                  "raw_score": 0.4},
        ).json()
        client.post(
            f"/cases/{case_id}/targets/{target['target_id']}/hypotheses"
            f"/{hyp['hypothesis_id']}/decide",
            json={"status": "rejected"},
        )
        detail = client.get(f"/cases/{case_id}").json()
        assert detail["targets"][0]["members"] == []
        assert detail["targets"][0]["hypotheses"][0]["status"] == "rejected"

    def test_evidence_guards(self, client, gallery, probe):
        login(client, "inv")
        case_id = self._create(client)["case_id"]

        # unknown run 404; duplicate attach 409
        ghost = client.post(f"/cases/{case_id}/runs", json={"run_id": "run-ghost"})
        assert ghost.status_code == 404
        client.post(f"/cases/{case_id}/runs", json={"run_id": gallery.run_id})
        dup = client.post(f"/cases/{case_id}/runs", json={"run_id": gallery.run_id})
        assert dup.status_code == 409

        # hypotheses may only cite runs attached to the case
        target = client.post(
            f"/cases/{case_id}/targets", json={"label": "T"}
        ).json()
        unattached = client.post(
            f"/cases/{case_id}/targets/{target['target_id']}/hypotheses",
            json={"run_id": probe.run_id, "camera_id": "p1", "track_id": 5,
                  "raw_score": 0.9},
        )
        assert unattached.status_code == 409
        assert "attach" in unattached.json()["detail"]

        # detach works and is idempotent-guarded
        gone = client.delete(f"/cases/{case_id}/runs/{gallery.run_id}")
        assert gone.status_code == 204
        assert client.get(f"/cases/{case_id}").json()["runs"] == []
        missing = client.delete(f"/cases/{case_id}/runs/{gallery.run_id}")
        assert missing.status_code == 404

    def test_update_and_close(self, client):
        login(client, "inv")
        case_id = self._create(client)["case_id"]
        updated = client.patch(
            f"/cases/{case_id}", json={"title": "Renamed", "status": "closed"}
        ).json()
        assert (updated["title"], updated["status"]) == ("Renamed", "closed")
        summary = client.get("/cases").json()[0]
        assert summary["status"] == "closed"


class TestAudit:
    def test_chain_records_and_verifies(self, client):
        login(client, "inv")
        client.post("/jobs", json={"videos": {"c1": "x.mp4"}})
        login(client, "admin")
        records = client.get("/audit").json()
        actions = [r["action"] for r in records]
        assert "login" in actions and "job_submitted" in actions
        verdict = client.get("/audit/verify").json()
        assert verdict == {"intact": True, "first_broken_seq": None}

    def test_tampering_detected(self, client):
        login(client, "admin")
        services = client.services
        from sqlalchemy import text

        with services.session_factory() as db:
            db.execute(text(
                "UPDATE audit_log SET detail = '{\"forged\": true}' WHERE seq = 1"
            ))
            db.commit()
        verdict = client.get("/audit/verify").json()
        assert verdict["intact"] is False
        assert verdict["first_broken_seq"] == 1
