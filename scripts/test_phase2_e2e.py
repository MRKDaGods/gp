"""Phase 2 end-to-end HTTP integration test harness.

This script starts the local FastAPI backend and Next.js frontend, then
exercises the Phase 2 endpoints using the same JSON shapes as the frontend.
It intentionally avoids running local GPU-heavy MTMC stages: pipeline checks
submit only CPU stage-4 shape requests without a video, then cancel the run.
"""

from __future__ import annotations

import base64
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
BACKEND_PORT = 8005
FRONTEND_PORT = 3001
BACKEND_BASE = f"http://127.0.0.1:{BACKEND_PORT}"
API_BASE = f"{BACKEND_BASE}/api"
FRONTEND_BASE = f"http://127.0.0.1:{FRONTEND_PORT}"
REQUEST_TIMEOUT_SECONDS = 30
SERVER_START_TIMEOUT_SECONDS = 180


@dataclass
class CaseResult:
    section: str
    name: str
    status: str
    detail: str


@dataclass
class ManagedProcess:
    name: str
    process: subprocess.Popen[str]
    lines: list[str]


def make_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["USE_CPU"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    if extra:
        env.update(extra)
    return env


def stream_process_output(name: str, process: subprocess.Popen[str], lines: list[str]) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        text = line.rstrip()
        if text:
            lines.append(text)
            del lines[:-300]
            print(f"[{name}] {text}", flush=True)


def start_process(name: str, command: list[str], *, cwd: Path, env: dict[str, str]) -> ManagedProcess:
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    lines: list[str] = []
    thread = threading.Thread(target=stream_process_output, args=(name, process, lines), daemon=True)
    thread.start()
    return ManagedProcess(name=name, process=process, lines=lines)


def stop_process(managed: ManagedProcess | None) -> None:
    if managed is None or managed.process.poll() is not None:
        return
    pid = managed.process.pid
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except Exception:
        managed.process.terminate()
    try:
        managed.process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        managed.process.kill()


def wait_for_http(url: str, *, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    last_error = "not attempted"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if 200 <= response.status < 500:
                    return
        except Exception as exc:  # noqa: BLE001 - startup probe keeps retrying
            last_error = str(exc)
        time.sleep(1)
    raise RuntimeError(f"Timed out waiting for {url}: {last_error}")


def request_json(method: str, path: str, payload: Any | None = None, *, timeout: int = REQUEST_TIMEOUT_SECONDS) -> tuple[int, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{API_BASE}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return response.status, json.loads(body) if body else None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed: Any = json.loads(body) if body else None
        except json.JSONDecodeError:
            parsed = body
        return exc.code, parsed


def request_text(url: str, *, timeout: int = REQUEST_TIMEOUT_SECONDS) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


def image_payloads() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def make_png(color: tuple[int, int, int]) -> str:
        image = Image.new("RGB", (64, 64), color)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    queries = [
        {"id": "query-red", "image_base64": make_png((220, 40, 40)), "metadata": {"synthetic": True}},
        {"id": "query-blue", "image_base64": make_png((40, 80, 220)), "metadata": {"synthetic": True}},
    ]
    gallery = [
        {"id": "gallery-red", "image_base64": make_png((210, 45, 45)), "metadata": {"synthetic": True}},
        {"id": "gallery-blue", "image_base64": make_png((45, 85, 210)), "metadata": {"synthetic": True}},
    ]
    return queries, gallery


def response_error_code(body: Any) -> str:
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, dict):
            return " ".join(str(value) for value in [detail.get("code"), detail.get("message")] if value)
        return str(detail or body)
    return str(body)


def has_veri776_split(root: Path) -> bool:
    return (root / "image_query").is_dir() and (root / "image_test").is_dir() and any((root / "image_query").glob("*.jpg")) and any((root / "image_test").glob("*.jpg"))


def has_cityflowv2_eval_data(root: Path) -> bool:
    expected_cameras = ["S01_c001", "S01_c002", "S01_c003", "S02_c006", "S02_c007", "S02_c008"]
    return all((root / camera / "gt.txt").is_file() and (root / camera / "vdo.avi").is_file() for camera in expected_cameras)


def load_registry() -> dict[str, Any]:
    status, body = request_json("GET", "/models?include_dead_ends=true")
    if status != 200 or not isinstance(body, dict) or not body.get("success"):
        raise RuntimeError(f"Could not load model registry over HTTP: status={status}, body={body}")
    return {str(model["id"]): model for model in body.get("data", [])}


def checkpoint_paths(model: dict[str, Any]) -> list[Path]:
    paths = []
    for checkpoint in model.get("checkpoint_refs") or []:
        local_path = checkpoint.get("local_path")
        if local_path:
            paths.append(ROOT / str(local_path))
    return paths


def primary_checkpoint_available(model: dict[str, Any]) -> tuple[bool, str]:
    paths = checkpoint_paths(model)
    if not paths:
        return False, "no checkpoint_refs in registry"
    missing = [path for path in paths if not path.exists()]
    if missing:
        return False, "missing checkpoint(s): " + ", ".join(str(path.relative_to(ROOT)) for path in missing)
    return True, "all registry checkpoints present"


def is_single_cam_response(body: Any, *, model_id: str) -> bool:
    return (
        isinstance(body, dict)
        and body.get("success") is True
        and body.get("modelId") == model_id
        and isinstance(body.get("featureDim"), int)
        and body.get("queryCount") == 2
        and body.get("galleryCount") == 2
        and isinstance(body.get("results"), list)
        and len(body["results"]) == 2
        and all(isinstance(item.get("matches"), list) for item in body["results"])
    )


def is_fusion_response(body: Any, expected_models: list[str]) -> bool:
    return (
        isinstance(body, dict)
        and body.get("success") is True
        and body.get("modelIds") == expected_models
        and body.get("queryCount") == 2
        and body.get("galleryCount") == 2
        and isinstance(body.get("weights"), list)
        and isinstance(body.get("results"), list)
        and isinstance(body.get("components"), list)
    )


def add_result(results: list[CaseResult], section: str, name: str, status: str, detail: str) -> None:
    results.append(CaseResult(section, name, status, detail.replace("\n", " ")[:600]))


def run_section_a(results: list[CaseResult], registry: dict[str, Any], queries: list[dict[str, Any]], gallery: list[dict[str, Any]]) -> list[str]:
    section = "Section A: ReID single-cam"
    requested_models = ["veri776_09v_v17_transreid", "veri776_clipsenet_v6", "cityflow_transreid"]
    available_serving_models: list[str] = []
    traversal_model_id = "veri776_09v_v17_transreid"

    for model_id in requested_models:
        model = registry.get(model_id)
        if model is None:
            add_result(results, section, model_id, "SKIP", "model_id is not present in configs/model_registry.yaml")
            continue
        if model.get("task_type") != "single_cam_reid":
            add_result(results, section, model_id, "SKIP", f"registry task_type={model.get('task_type')}; /reid/single_cam only serves single_cam_reid models")
            continue
        ok, reason = primary_checkpoint_available(model)
        if not ok:
            add_result(results, section, model_id, "SKIP", reason)
            continue

        payload = {"modelId": model_id, "queries": queries, "gallery": gallery, "topK": 2, "rerank": False, "aqeK": 0}
        status, body = request_json("POST", "/v1/reid/single_cam", payload, timeout=180)
        if status == 200 and is_single_cam_response(body, model_id=model_id):
            add_result(results, section, f"{model_id} basic", "PASS", f"200; featureDim={body.get('featureDim')}; device={body.get('device')}")
            available_serving_models.append(model_id)
        else:
            add_result(results, section, f"{model_id} basic", "FAIL", f"status={status}; body={body}")
            continue

        payload = {"modelId": model_id, "queries": queries, "gallery": gallery, "topK": 2, "rerank": True, "aqeK": 3}
        status, body = request_json("POST", "/v1/reid/single_cam", payload, timeout=180)
        if status == 200 and is_single_cam_response(body, model_id=model_id):
            add_result(results, section, f"{model_id} rerank+aqe", "PASS", "200 with rerank=true, aqeK=3")
        else:
            add_result(results, section, f"{model_id} rerank+aqe", "FAIL", f"status={status}; body={body}")

    if registry.get(traversal_model_id) is None:
        add_result(results, section, "path traversal protection", "SKIP", f"{traversal_model_id} missing from registry")
    else:
        payload = {
            "modelId": traversal_model_id,
            "queries": [{"id": "bad", "path": "../etc/passwd"}],
            "gallery": gallery[:1],
            "topK": 1,
        }
        status, body = request_json("POST", "/v1/reid/single_cam", payload)
        if status in {400, 422} and "path_traversal" in response_error_code(body):
            add_result(results, section, "path traversal protection", "PASS", f"rejected with HTTP {status}: {response_error_code(body)}")
        else:
            add_result(results, section, "path traversal protection", "FAIL", f"expected sanitized path traversal rejection; status={status}; body={body}")

    return available_serving_models


def run_section_b(results: list[CaseResult], available_models: list[str], queries: list[dict[str, Any]], gallery: list[dict[str, Any]]) -> None:
    section = "Section B: Fusion"
    if len(available_models) < 2:
        add_result(results, section, "2-model fusion", "SKIP", f"need two locally loadable serving models; available={available_models}")
        add_result(results, section, "auto-normalize weights", "SKIP", f"need two locally loadable serving models; available={available_models}")
    else:
        models = available_models[:2]
        payload = {
            "models": [{"modelId": models[0], "weight": 0.7}, {"modelId": models[1], "weight": 0.3}],
            "queries": queries,
            "gallery": gallery,
            "topK": 2,
        }
        status, body = request_json("POST", "/v1/reid/fusion", payload, timeout=240)
        if status == 200 and is_fusion_response(body, models):
            add_result(results, section, "2-model fusion", "PASS", f"200; weights={body.get('weights')}")
        else:
            add_result(results, section, "2-model fusion", "FAIL", f"status={status}; body={body}")

        payload["models"] = [{"modelId": models[0], "weight": 2}, {"modelId": models[1], "weight": 3}]
        status, body = request_json("POST", "/v1/reid/fusion", payload, timeout=240)
        weights = body.get("weights") if isinstance(body, dict) else None
        if status == 200 and weights and abs(weights[0] - 0.4) < 1e-6 and abs(weights[1] - 0.6) < 1e-6:
            add_result(results, section, "auto-normalize weights", "PASS", f"weights normalized to {weights}; warnings={body.get('warnings')}")
        else:
            add_result(results, section, "auto-normalize weights", "FAIL", f"status={status}; weights={weights}; body={body}")

    model_id = available_models[0] if available_models else "veri776_09v_v17_transreid"
    payload = {
        "models": [{"modelId": model_id, "weight": 1.0}],
        "queries": queries,
        "gallery": gallery,
        "topK": 2,
    }
    status, body = request_json("POST", "/v1/reid/fusion", payload)
    if status in {400, 422}:
        add_result(results, section, "reject single-model fusion", "PASS", f"rejected with HTTP {status}: {response_error_code(body)}")
    else:
        add_result(results, section, "reject single-model fusion", "FAIL", f"expected 4xx rejection; status={status}; body={body}")


def eval_prerequisites(eval_type: str) -> tuple[bool, str]:
    prereqs: dict[str, list[Path]] = {
        "veri776_transreid": [ROOT / "models/reid/vehicle_transreid_vit_base_veri776.pth", ROOT / "data/raw/veri776"],
        "veri776_clipsenet": [ROOT / "models/reid/clipsenet_v6_veri776_best.pth", ROOT / "data/raw/veri776"],
        "cityflow_transreid": [ROOT / "models/reid/transreid_cityflowv2_best.pth", ROOT / "data/raw/cityflowv2"],
        "veri776_14t_fusion": [
            ROOT / "models/reid/vehicle_transreid_vit_base_veri776.pth",
            ROOT / "models/reid/clipsenet_v6_veri776_best.pth",
            ROOT / "data/raw/veri776",
        ],
    }
    missing = [path for path in prereqs[eval_type] if not path.exists()]
    if missing:
        return False, "missing local artifact(s): " + ", ".join(str(path.relative_to(ROOT)) for path in missing)
    if eval_type in {"veri776_transreid", "veri776_clipsenet", "veri776_14t_fusion"} and not has_veri776_split(ROOT / "data/raw/veri776"):
        return False, "data/raw/veri776 exists but is not a complete VeRi-776 split with image_query/*.jpg and image_test/*.jpg"
    if eval_type == "cityflow_transreid" and not has_cityflowv2_eval_data(ROOT / "data/raw/cityflowv2"):
        return False, "data/raw/cityflowv2 exists but is not a complete CityFlowV2 eval tree with expected camera gt.txt/vdo.avi files"
    return True, "local checkpoint/data prerequisites present"


def poll_eval_job(job_id: str, *, timeout_seconds: int = 300) -> tuple[str, Any]:
    deadline = time.time() + timeout_seconds
    last_body: Any = None
    while time.time() < deadline:
        status, body = request_json("GET", f"/v1/eval/{urllib.parse.quote(job_id)}/status", timeout=30)
        last_body = body
        if status != 200:
            return "http_error", body
        job_status = str(body.get("status", "")).lower() if isinstance(body, dict) else ""
        if job_status in {"completed", "failed"}:
            return job_status, body
        time.sleep(2)
    return "timeout", last_body


def run_section_c(results: list[CaseResult]) -> None:
    section = "Section C: Eval endpoints"
    eval_types = ["veri776_transreid", "veri776_clipsenet", "cityflow_transreid", "veri776_14t_fusion"]
    for eval_type in eval_types:
        ok, reason = eval_prerequisites(eval_type)
        if not ok:
            add_result(results, section, eval_type, "SKIP", reason)
            continue
        config_overrides = {}
        if eval_type in {"veri776_transreid", "veri776_clipsenet", "veri776_14t_fusion"}:
            config_overrides = {"max_queries": 10, "max_gallery": 50}
        status, body = request_json("POST", "/v1/eval/run", {"evalType": eval_type, "configOverrides": config_overrides})
        if status != 200 or not isinstance(body, dict) or not body.get("jobId"):
            add_result(results, section, eval_type, "FAIL", f"submit failed: status={status}; body={body}")
            continue
        job_id = str(body["jobId"])
        final_status, final_body = poll_eval_job(job_id)
        if final_status == "completed":
            result_status, result_body = request_json("GET", f"/v1/eval/{urllib.parse.quote(job_id)}/result")
            if result_status == 200 and isinstance(result_body, dict) and result_body.get("result") is not None:
                add_result(results, section, eval_type, "PASS", f"job completed; result summary keys={list((result_body.get('result') or {}).get('summary', {}).keys())}")
            else:
                add_result(results, section, eval_type, "FAIL", f"result shape invalid: status={result_status}; body={result_body}")
        elif final_status == "failed":
            add_result(results, section, eval_type, "FAIL", f"job failed: {final_body}")
        else:
            add_result(results, section, eval_type, "FAIL", f"job did not finish cleanly: final_status={final_status}; body={final_body}")

    status, body = request_json("GET", "/v1/eval/unknown_job_id/status")
    if status == 404:
        add_result(results, section, "unknown job status 404", "PASS", "GET /api/v1/eval/unknown_job_id/status returned 404")
    else:
        add_result(results, section, "unknown job status 404", "FAIL", f"expected 404; status={status}; body={body}")


def run_section_d(results: list[CaseResult]) -> None:
    section = "Section D: Model registry dataset filtering"
    for dataset in ["cityflowv2", "wildtrack"]:
        status, body = request_json("GET", f"/models?dataset={urllib.parse.quote(dataset)}")
        data = body.get("data") if isinstance(body, dict) else None
        if status == 200 and isinstance(data, list) and all(item.get("dataset") == dataset for item in data):
            add_result(results, section, f"dataset={dataset}", "PASS", f"returned {len(data)} model(s), all dataset={dataset}")
        else:
            add_result(results, section, f"dataset={dataset}", "FAIL", f"status={status}; body={body}")

    status, body = request_json("GET", "/models?dataset=invalid")
    data = body.get("data") if isinstance(body, dict) else None
    if status == 400 or (status == 200 and data == []):
        add_result(results, section, "dataset=invalid", "PASS", f"status={status}; data={data}")
    else:
        add_result(results, section, "dataset=invalid", "FAIL", f"expected 400 or empty list; status={status}; body={body}")


def run_section_e(results: list[CaseResult]) -> None:
    section = "Section E: MTMC pipeline endpoint shape only"
    for dataset in ["cityflowv2", "wildtrack"]:
        run_id = f"phase2_e2e_{dataset}_{int(time.time())}"
        payload = {"runId": run_id, "dataset": dataset, "smokeTest": True, "useCpu": True, "config": {"dataset": dataset, "smokeTest": True}}
        status, body = request_json("POST", "/pipeline/run-stage/4", payload)
        if status == 200 and isinstance(body, dict) and body.get("success") and body.get("data", {}).get("runId") == run_id:
            cancel_status, _cancel_body = request_json("POST", f"/pipeline/cancel/{urllib.parse.quote(run_id)}", {})
            add_result(results, section, f"{dataset} stage-4 submit", "PASS", f"accepted shape-only stage-4 CPU/smoke request; cancel_status={cancel_status}; no stage 0/1/2 submitted")
        else:
            add_result(results, section, f"{dataset} stage-4 submit", "FAIL", f"submit failed: status={status}; body={body}")


def run_section_f(results: list[CaseResult]) -> None:
    section = "Section F: Frontend page render"
    expectations = {
        "/reid": ["Single-Cam ReID"],
        "/fusion": ["Fusion ReID", "Fusion"],
        "/eval": ["Eval Runner", "Eval"],
        "/": ["ReID", "Fusion", "Eval"],
    }
    for path, needles in expectations.items():
        status, body = request_text(f"{FRONTEND_BASE}{path}", timeout=60)
        if status == 200 and all(needle in body for needle in needles):
            add_result(results, section, f"GET {path}", "PASS", f"200; found {needles}")
        else:
            add_result(results, section, f"GET {path}", "FAIL", f"status={status}; missing={[needle for needle in needles if needle not in body]}; body_prefix={body[:200]!r}")


def render_report(results: list[CaseResult]) -> str:
    sections = list(dict.fromkeys(result.section for result in results))
    lines = ["# Phase 2 E2E Integration Test Report", "", f"Backend: `{BACKEND_BASE}`", f"Frontend: `{FRONTEND_BASE}`", ""]
    for section in sections:
        lines.append(f"## {section}")
        lines.append("| Case | Status | Detail |")
        lines.append("|---|---:|---|")
        for result in [item for item in results if item.section == section]:
            lines.append(f"| {result.name} | {result.status} | {result.detail.replace('|', '/')} |")
        lines.append("")

    section_summaries = []
    skip_count = sum(1 for result in results if result.status == "SKIP")
    fail_count = sum(1 for result in results if result.status == "FAIL")
    for section in sections:
        items = [item for item in results if item.section == section]
        if all(item.status == "PASS" for item in items):
            section_summaries.append("PASS")
        elif any(item.status == "FAIL" for item in items):
            section_summaries.append("FAIL")
        else:
            section_summaries.append("SKIP")
    pass_sections = section_summaries.count("PASS")
    skip_sections = section_summaries.count("SKIP")
    fail_sections = section_summaries.count("FAIL")
    lines.append("## Executive Summary")
    lines.append(
        f"{pass_sections}/{len(sections)} sections fully PASS, "
        f"{skip_sections} section(s) SKIP-only due to missing local artifacts, "
        f"{fail_sections} section(s) FAIL-with-bug. Case totals: "
        f"PASS={sum(1 for result in results if result.status == 'PASS')}, "
        f"SKIP={skip_count}, FAIL={fail_count}."
    )
    return "\n".join(lines)


def npm_command() -> str:
    return "npm.cmd" if os.name == "nt" else "npm"


def main() -> int:
    results: list[CaseResult] = []
    backend: ManagedProcess | None = None
    frontend: ManagedProcess | None = None
    try:
        backend = start_process(
            "backend",
            [sys.executable, "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", str(BACKEND_PORT)],
            cwd=ROOT,
            env=make_env(),
        )
        wait_for_http(f"{BACKEND_BASE}/api/health", timeout_seconds=SERVER_START_TIMEOUT_SECONDS)

        frontend = start_process(
            "frontend",
            [npm_command(), "--prefix", "frontend", "run", "dev", "--", "--hostname", "127.0.0.1", "--port", str(FRONTEND_PORT)],
            cwd=ROOT,
            env=make_env({"NEXT_PUBLIC_API_URL": f"{API_BASE}"}),
        )
        wait_for_http(f"{FRONTEND_BASE}/reid", timeout_seconds=SERVER_START_TIMEOUT_SECONDS)

        registry = load_registry()
        queries, gallery = image_payloads()
        available_models = run_section_a(results, registry, queries, gallery)
        run_section_b(results, available_models, queries, gallery)
        run_section_c(results)
        run_section_d(results)
        run_section_e(results)
        run_section_f(results)
    except Exception as exc:  # noqa: BLE001 - report harness failures as a matrix row
        add_result(results, "Harness", "unhandled harness error", "FAIL", repr(exc))
    finally:
        stop_process(frontend)
        stop_process(backend)

    report = render_report(results)
    print("\n" + report)
    return 1 if any(result.status == "FAIL" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())