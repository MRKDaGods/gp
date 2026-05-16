from __future__ import annotations

import json
import subprocess
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import app
from backend.services.job_service import job_service


def _reset_jobs(tmp_path: Path) -> None:
    job_service.job_dir = tmp_path
    job_service.jobs.clear()
    job_service.load_jobs()


def test_eval_submit_status_and_result(monkeypatch, tmp_path: Path) -> None:
    _reset_jobs(tmp_path)

    def fake_run(command, cwd, stdout, stderr, text, check):
        output_json = Path(command[command.index("--output-json") + 1])
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps({"mAP": 89.97, "R1": 98.33}), encoding="utf-8")
        stderr.write("")
        stdout.write("mock eval complete\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("backend.services.eval_service.subprocess.run", fake_run)

    client = TestClient(app)
    submit = client.post(
        "/api/v1/eval/run",
        json={"evalType": "veri776_transreid", "configOverrides": {"batch_size": 1, "rerank": False}},
    )

    assert submit.status_code == 200
    job_id = submit.json()["jobId"]
    assert submit.json()["status"] == "queued"

    status = client.get(f"/api/v1/eval/{job_id}/status")
    assert status.status_code == 200
    assert status.json()["status"] == "completed"
    assert status.json()["progress"]["stage"] == "finished"

    result = client.get(f"/api/v1/eval/{job_id}/result")
    assert result.status_code == 200
    payload = result.json()
    assert payload["status"] == "completed"
    assert payload["result"]["summary"]["mAP"] == 89.97
    assert payload["result"]["result"]["R1"] == 98.33


def test_eval_unknown_job_404(tmp_path: Path) -> None:
    _reset_jobs(tmp_path)
    client = TestClient(app)

    status = client.get("/api/v1/eval/unknown/status")
    result = client.get("/api/v1/eval/unknown/result")

    assert status.status_code == 404
    assert result.status_code == 404


def test_eval_rejects_unknown_eval_type(tmp_path: Path) -> None:
    _reset_jobs(tmp_path)
    response = TestClient(app).post("/api/v1/eval/run", json={"evalType": "arbitrary_script", "configOverrides": {}})

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "unsupported_eval_type"


def test_model_registry_honors_dataset_filter() -> None:
    client = TestClient(app)

    cityflow = client.get("/api/models?dataset=cityflowv2")
    wildtrack = client.get("/api/models?dataset=wildtrack")

    assert cityflow.status_code == 200
    assert wildtrack.status_code == 200
    assert {entry["dataset"] for entry in cityflow.json()["data"]} == {"cityflowv2"}
    assert {entry["dataset"] for entry in wildtrack.json()["data"]} == {"wildtrack"}