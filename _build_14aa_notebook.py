from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "notebooks" / "kaggle" / "14aa_verify_14t_veri_fusion"
NOTEBOOK = OUT_DIR / "14aa_verify_14t_veri_fusion.ipynb"
METADATA = OUT_DIR / "kernel-metadata.json"
EXPECTED_SHA = "776a2ffc428d90fdf926e5eaf19da5310c01db4a"


def source_lines(text: str) -> list[str]:
    lines = dedent(text).strip("\n").splitlines()
    return [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines else [])


def markdown_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {"language": "markdown"}, "source": source_lines(text)}


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"language": "python"},
        "outputs": [],
        "source": source_lines(text),
    }


HEADER = """
# 14aa Verify 14t VeRi Fusion

GPU verifier for the 14t CLIP-SENet v6 x TransReID 09v v17 score-fusion WIN on VeRi-776.
Target: mAP=0.9330 +/- 0.005, R1=0.9845 +/- 0.005.
Source: docs/findings.md section 14t.
"""


SETUP = r'''
import sys, subprocess, shutil

GPU_AVAILABLE = False

# --- Guard: detect GPU BEFORE importing torch ---
# Kaggle's newest PyTorch wheels may drop P100 (sm_60) support.
# If we got a P100, downgrade to a compatible build first.
if shutil.which("nvidia-smi"):
    _nvsmi = subprocess.run(
        ["nvidia-smi", "--query-gpu=gpu_name,compute_cap", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    print(_nvsmi.stdout)
    if _nvsmi.returncode == 0 and _nvsmi.stdout.strip():
        GPU_AVAILABLE = True
        for _row in _nvsmi.stdout.strip().splitlines():
            _gpu_name, _cap = _row.split(",", 1)
            _cap = _cap.strip()
            if _cap.startswith("6."):
                print(f"GPU {_gpu_name.strip()} ({_cap}) requires compatible PyTorch; installing torch 2.4.1 cu124")
                subprocess.check_call([
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-cache-dir",
                    "--force-reinstall",
                    "torch==2.4.1",
                    "torchvision==0.19.1",
                    "torchaudio==2.4.1",
                    "--index-url",
                    "https://download.pytorch.org/whl/cu124",
                ])
                print("Compatible PyTorch installed")
                break

if not GPU_AVAILABLE:
    print("No GPU detected; verifier rows will be recorded as SKIP and no CPU fallback will run.")
'''


CLONE_INSTALL = f'''
from pathlib import Path
import json
import os
import platform
import shutil
import subprocess
import sys
import traceback

REPO_URL = "https://github.com/MRKDaGods/gp.git"
EXPECTED_MASTER_SHA_AT_BUILD = "{EXPECTED_SHA}"
WORK_DIR = Path("/kaggle/working")
PROJECT = WORK_DIR / "gp"
RESULTS_PATH = WORK_DIR / "14aa_verify_results.json"
EVAL_JSON_DIR = WORK_DIR / "14aa_eval_json"
EVAL_LOG_DIR = WORK_DIR / "14aa_eval_logs"
EVAL_JSON_DIR.mkdir(parents=True, exist_ok=True)
EVAL_LOG_DIR.mkdir(parents=True, exist_ok=True)

results = []
eval_outputs = {{}}
context = {{
    "verifier": "14aa_verify_14t_veri_fusion",
    "expected_master_sha_at_build": EXPECTED_MASTER_SHA_AT_BUILD,
    "gpu_available": bool(GPU_AVAILABLE),
    "datasets": {{}},
    "checkpoints": {{}},
}}


def run(cmd, *, cwd=None, check=True, capture=False):
    rendered = " ".join(map(str, cmd))
    print("$", rendered)
    if capture:
        return subprocess.check_output(list(map(str, cmd)), cwd=cwd, text=True)
    return subprocess.run(list(map(str, cmd)), cwd=cwd, check=check, text=True)


def pip_install(*args):
    run([sys.executable, "-m", "pip", "install", "-q", *args])


def clone_or_refresh_master():
    if not PROJECT.exists():
        run(["git", "clone", "--branch", "master", "--depth", "1", REPO_URL, str(PROJECT)])
    else:
        run(["git", "-C", PROJECT, "fetch", "origin", "master", "--depth", "1"])
        run(["git", "-C", PROJECT, "checkout", "master"])
        run(["git", "-C", PROJECT, "reset", "--hard", "origin/master"])
    os.chdir(PROJECT)
    sys.path.insert(0, str(PROJECT))
    head_sha = run(["git", "rev-parse", "HEAD"], capture=True).strip()
    context["git_sha"] = head_sha
    print("resolved git SHA:", head_sha)
    print("expected master SHA at notebook build:", EXPECTED_MASTER_SHA_AT_BUILD)
    if head_sha != EXPECTED_MASTER_SHA_AT_BUILD:
        print(f"WARN: master moved since notebook build: expected {{EXPECTED_MASTER_SHA_AT_BUILD}}, got {{head_sha}}")


clone_or_refresh_master()
print("python:", sys.version)
print("platform:", platform.platform())

pip_install("faiss-cpu", "motmetrics", "loguru", "omegaconf", "rich", "networkx>=3.1", "click", "filterpy", "ftfy", "lapx", "scikit-learn", "scipy", "pandas", "opencv-python-headless", "tqdm")
pip_install("timm==1.0.11", "open_clip_torch==2.30.0", "pretrainedmodels==0.7.4")
pip_install("--no-deps", "ultralytics", "boxmot==11.0.3")
pip_install("--no-deps", "-e", ".")

try:
    import torch
    print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available())
    GPU_AVAILABLE = bool(GPU_AVAILABLE and torch.cuda.is_available())
    context["gpu_available"] = GPU_AVAILABLE
except Exception as exc:
    print("torch import failed:", repr(exc))
    GPU_AVAILABLE = False
    context["gpu_available"] = False

for mount in [Path("/tmp"), WORK_DIR]:
    total, used, free = shutil.disk_usage(mount)
    print(f"{{mount}}: {{free / 1024**3:.1f}} GiB free / {{total / 1024**3:.1f}} GiB total")
'''


DATASET_MOUNT = r'''
INPUT_ROOT = Path("/kaggle/input")


def symlink_or_copy(src, dst):
    src = Path(src)
    dst = Path(dst)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def find_veri_root():
    known = [
        INPUT_ROOT / "veri-vehicle-re-identification-dataset" / "VeRi",
        INPUT_ROOT / "veri-vehicle-re-identification-dataset",
    ]
    for root in known:
        if (root / "image_query").is_dir() and (root / "image_test").is_dir():
            return root
    for path in INPUT_ROOT.rglob("VeRi") if INPUT_ROOT.exists() else []:
        if path.is_dir() and (path / "image_query").is_dir() and (path / "image_test").is_dir():
            return path
    for path in INPUT_ROOT.rglob("*") if INPUT_ROOT.exists() else []:
        if path.is_dir() and (path / "image_query").is_dir() and (path / "image_test").is_dir():
            return path
    raise FileNotFoundError("VeRi-776 root with image_query/image_test not found")


def dataset_roots(owner_slug):
    owner, slug = owner_slug.split("/", 1)
    candidates = [
        INPUT_ROOT / slug,
        INPUT_ROOT / owner_slug.replace("/", "-"),
        INPUT_ROOT / "datasets" / owner / slug,
        INPUT_ROOT / "notebooks" / owner / slug,
    ]
    return [path for path in candidates if path.exists()]


def find_member(owner_slugs, member):
    for owner_slug in owner_slugs:
        for root in dataset_roots(owner_slug):
            candidate = root / member
            if candidate.exists():
                print("selected", candidate)
                return candidate
    member_name = Path(member).name
    matches = []
    for path in INPUT_ROOT.rglob(member_name) if INPUT_ROOT.exists() else []:
        if str(path).replace("\\", "/").endswith(member):
            matches.append(path)
    if not matches:
        raise FileNotFoundError(f"Could not find member {member} under /kaggle/input")
    matches = sorted(matches, key=lambda path: (len(str(path)), str(path)))
    print("selected", matches[0])
    return matches[0]


def find_clipsenet_checkpoint():
    roots = dataset_roots("yahiaakhalafallah/13-clip-senet-train")
    if not roots:
        roots = [path for path in [INPUT_ROOT / "13-clip-senet-train"] if path.exists()]
    for name in ("best_mAP.pth", "best.pth"):
        for root in roots or [INPUT_ROOT]:
            matches = sorted(root.rglob(name)) if root.exists() else []
            if matches:
                print("selected", matches[0])
                return matches[0]
    raise FileNotFoundError("Could not find CLIP-SENet best_mAP.pth or best.pth")


VERI_ROOT = find_veri_root()
TRANSREID_CKPT = find_member(
    ["mrkdagods/mtmc-weights", "yahiaakhalafallah/mtmc-weights", "gumfreddy/mtmc-weights"],
    "reid/vehicle_transreid_vit_base_veri776.pth",
)
CLIPSENET_CKPT = find_clipsenet_checkpoint()

TRANSREID_LOCAL = PROJECT / "models" / "reid" / "vehicle_transreid_vit_base_veri776.pth"
CLIPSENET_LOCAL = PROJECT / "models" / "reid" / "clipsenet_v6_veri776_best.pth"
symlink_or_copy(TRANSREID_CKPT, TRANSREID_LOCAL)
symlink_or_copy(CLIPSENET_CKPT, CLIPSENET_LOCAL)

context["datasets"]["veri_root"] = str(VERI_ROOT)
context["checkpoints"] = {
    "transreid_09v": str(TRANSREID_CKPT),
    "clipsenet_v6": str(CLIPSENET_CKPT),
    "transreid_local": str(TRANSREID_LOCAL),
    "clipsenet_local": str(CLIPSENET_LOCAL),
}
print(json.dumps(context, indent=2))
'''


SUBPROCESS_HELPERS = r'''
def tail_text(path, max_chars=8000):
    path = Path(path)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


class EvalSubprocessError(RuntimeError):
    def __init__(self, label, returncode, stdout_path, stderr_path):
        self.label = label
        self.returncode = returncode
        self.stdout_path = Path(stdout_path)
        self.stderr_path = Path(stderr_path)
        stdout_tail = tail_text(self.stdout_path)
        stderr_tail = tail_text(self.stderr_path)
        message = (
            f"{label} subprocess failed with return code {returncode}. "
            f"stdout={self.stdout_path} stderr={self.stderr_path}\n"
            f"--- stderr tail ---\n{stderr_tail or '<empty>'}\n"
            f"--- stdout tail ---\n{stdout_tail or '<empty>'}"
        )
        super().__init__(message)


def run_eval_subprocess(label, cmd, *, cwd):
    safe_label = label.lower().replace(" ", "_").replace("/", "_")
    stdout_path = EVAL_LOG_DIR / f"{safe_label}.stdout.txt"
    stderr_path = EVAL_LOG_DIR / f"{safe_label}.stderr.txt"
    rendered = " ".join(map(str, cmd))
    print("$", rendered)
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        proc = subprocess.run(
            list(map(str, cmd)),
            cwd=cwd,
            text=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
    context.setdefault("eval_logs", {})[label] = {
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "returncode": int(proc.returncode),
    }
    stdout_tail = tail_text(stdout_path, max_chars=3000)
    stderr_tail = tail_text(stderr_path, max_chars=3000)
    if stdout_tail:
        print(f"--- {label} stdout tail ---")
        print(stdout_tail)
    if stderr_tail:
        print(f"--- {label} stderr tail ---")
        print(stderr_tail)
    if proc.returncode != 0:
        raise EvalSubprocessError(label, proc.returncode, stdout_path, stderr_path)
    return proc


def record_metric(label, observed, target, tolerance=0.005, required=True, error=None, extra=None):
    if observed is None:
        passed = False if required else True
        delta = None
    else:
        observed = float(observed)
        delta = observed - float(target)
        passed = abs(delta) <= float(tolerance)
    row = {
        "label": label,
        "observed": observed,
        "target": float(target),
        "tolerance": float(tolerance),
        "delta": delta,
        "status": "PASS" if passed else "FAIL",
        "required": bool(required),
    }
    if error:
        row["error"] = str(error)
    if extra:
        row.update(extra)
    results.append(row)
    print(f"{row['status']:4} {label}: observed={observed} target={target} delta={delta}")
    if error:
        print("  error:", error)
    return row


def record_skip(label, target, tolerance=0.005, required=False, reason="GPU unavailable"):
    row = {
        "label": label,
        "observed": None,
        "target": float(target),
        "tolerance": float(tolerance),
        "delta": None,
        "status": "SKIPPED",
        "required": bool(required),
        "reason": reason,
    }
    results.append(row)
    print(f"SKIP {label}: {reason}")
    return row


def record_exception(prefix, targets, exc):
    message = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    print(f"{prefix} failed: {message}")
    traceback.print_exc()
    for label, target, tolerance, required in targets:
        record_metric(label, None, target, tolerance=tolerance, required=required, error=message)


def metric_from(data, *path_options):
    for path in path_options:
        cur = data
        ok = True
        for key in path:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                ok = False
                break
        if ok and cur is not None:
            return float(cur)
    raise KeyError(f"None of the metric paths existed: {path_options}")
'''


EVAL_F = r'''
targets = [
    ("Eval F 14t fusion score mAP", 0.9330, 0.005, True),
    ("Eval F 14t fusion score R1", 0.9845, 0.005, True),
    ("Eval F drift TransReID 09v concat_patch+AQE3+rerank mAP", 0.8997, 0.01, False),
    ("Eval F drift CLIP-SENet v6 AQE10+rerank mAP", 0.9154, 0.01, False),
]

if not GPU_AVAILABLE:
    eval_outputs["eval_f_14t_fusion"] = {"skipped": True, "reason": "GPU unavailable"}
    for label, target, tolerance, _required in targets:
        record_skip(label, target, tolerance=tolerance, required=False, reason="GPU unavailable")
else:
    try:
        script = PROJECT / "scripts" / "eval" / "eval_14t_fusion_veri776.py"
        if not script.exists():
            raise FileNotFoundError("scripts/eval/eval_14t_fusion_veri776.py is missing from master")
        output_json = EVAL_JSON_DIR / "eval_f_14t_fusion.json"
        cmd = [
            sys.executable,
            str(script),
            "--transreid-checkpoint", str(TRANSREID_LOCAL),
            "--clipsenet-checkpoint", str(CLIPSENET_LOCAL),
            "--veri-root", str(VERI_ROOT),
            "--device", "cuda",
            "--w-clipsenet", "0.7",
            "--transreid-stream", "global",
            "--aqe-k", "3",
            "--rerank-k1", "80",
            "--rerank-k2", "15",
            "--rerank-lambda", "0.2",
            "--transreid-batch-size", "64",
            "--clipsenet-batch-size", "64",
            "--clipsenet-img-size", "320", "320",
            "--output-json", str(output_json),
        ]
        run_eval_subprocess("Eval F", cmd, cwd=PROJECT)
        data = json.loads(output_json.read_text(encoding="utf-8"))
        eval_outputs["eval_f_14t_fusion"] = str(output_json)
        record_metric("Eval F 14t fusion score mAP", metric_from(data, ("score_fusion", "best", "mAP")), 0.9330, tolerance=0.005, required=True)
        record_metric("Eval F 14t fusion score R1", metric_from(data, ("score_fusion", "best", "R1")), 0.9845, tolerance=0.005, required=True)
        record_metric(
            "Eval F drift TransReID 09v concat_patch+AQE3+rerank mAP",
            metric_from(data, ("drift_parents", "transreid_09v_concat_patch_aqe3_rerank", "mAP")),
            0.8997,
            tolerance=0.01,
            required=False,
        )
        record_metric(
            "Eval F drift CLIP-SENet v6 AQE10+rerank mAP",
            metric_from(data, ("drift_parents", "clipsenet_v6_aqe10_rerank_k1_50_k2_10_lambda_0_1", "mAP")),
            0.9154,
            tolerance=0.01,
            required=False,
        )
    except Exception as exc:
        record_exception("Eval F", targets, exc)
'''


SUMMARY = r'''
logs = {}
if "Eval F" in context.get("eval_logs", {}):
    logs = {
        "eval_f_stdout": context["eval_logs"]["Eval F"].get("stdout"),
        "eval_f_stderr": context["eval_logs"]["Eval F"].get("stderr"),
    }

summary = {
    "verifier": "14aa_verify_14t_veri_fusion",
    "passed": all(row["status"] == "PASS" for row in results if row.get("required", True)),
    "git_sha": context.get("git_sha"),
    "expected_master_sha_at_build": EXPECTED_MASTER_SHA_AT_BUILD,
    "gpu_available": bool(GPU_AVAILABLE),
    "metrics": results,
    "eval_outputs": eval_outputs,
    "logs": logs,
    "context": context,
}
RESULTS_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(f"wrote {RESULTS_PATH}")

print("\n14aa 14t fusion verifier summary")
print("| label | observed | target | delta | status | required |")
print("|---|---:|---:|---:|---|---|")
for row in results:
    observed = "NA" if row["observed"] is None else f"{row['observed']:.6f}"
    delta = "NA" if row["delta"] is None else f"{row['delta']:+.6f}"
    print(f"| {row['label']} | {observed} | {row['target']:.6f} | {delta} | {row['status']} | {row.get('required', True)} |")

failures = [row for row in results if row.get("required", True) and row["status"] != "PASS"]
if failures:
    raise AssertionError("14aa required metric failures:\n" + json.dumps(failures, indent=2))
print("All required 14aa 14t fusion metrics passed or were skipped because GPU was unavailable.")
'''


def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nb = {
        "cells": [
            markdown_cell(HEADER),
            code_cell(SETUP),
            code_cell(CLONE_INSTALL),
            code_cell(DATASET_MOUNT),
            code_cell(SUBPROCESS_HELPERS),
            code_cell(EVAL_F),
            code_cell(SUMMARY),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK.write_text(json.dumps(nb, indent=2, ensure_ascii=True), encoding="utf-8")
    metadata = {
        "id": "yahiaakhalafallah/14aa-verify-14t-veri-fusion",
        "title": "14aa Verify 14t VeRi Fusion",
        "code_file": "14aa_verify_14t_veri_fusion.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [
            "abhyudaya12/veri-vehicle-re-identification-dataset",
            "mrkdagods/mtmc-weights",
            "yahiaakhalafallah/mtmc-weights",
            "gumfreddy/mtmc-weights",
        ],
        "kernel_sources": ["yahiaakhalafallah/13-clip-senet-train"],
        "model_sources": [],
    }
    METADATA.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"wrote {NOTEBOOK}")
    print(f"wrote {METADATA}")


if __name__ == "__main__":
    build()