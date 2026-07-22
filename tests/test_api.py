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

PASSWORDS = {"admin": "pw-admin", "inv": "pw-inv", "view": "pw-view"}


@pytest.fixture()
def settings(tmp_path) -> ApiSettings:
    return ApiSettings(
        runs_root=tmp_path / "runs",
        jobs_db=tmp_path / "jobs" / "jobs.db",
        registry_db=tmp_path / "registry" / "models.db",
        app_db=tmp_path / "app" / "app.db",
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
        response = client.get(f"/runs/{gallery.run_id}/report")
        assert response.status_code == 404
        assert "package" in response.json()["detail"]


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
