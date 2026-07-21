"""ATHAR Gate P2 — CityFlow B1 golden generation (Kaggle, CPU-only).

Reproduces the registered ``vehicle_mtmc_14e_b1`` baseline (IDF1 0.77936,
id_switches 154) from the frozen v1 tree and packages the parity goldens:

    cityflow_b1_goldens.tar.gz
      stage1/tracklets_*.json        v1 stage-1 tracklets   (from 10a)
      stage2/embeddings*.npy, ...    v1 TTA stage-2 features (from 14c)
      stage4/global_trajectories.json
      stage5/evaluation_report.json
      provenance.json                commit SHA, sources, metrics, sha256s

Adapted from the proven 14v_verify_b1_from_yaml kernel (public), which
drift-gated this exact recipe on 2026-05-15. Mounts (kernel_sources):
``14c-tta-stage2`` (TTA stage-2 features) and ``mtmc-10a-stages-0-2``
(stage-1 tracklets) — both private to the yahiaakhalafallah account, so
push this kernel FROM that account (kernel-metadata.json is set for it).

CPU kernel: stages 3-5 are FAISS + graph association + eval, no GPU.
"""

from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time

REPO_URL = "https://github.com/MRKDaGods/gp.git"
BRANCH = "verify/14v-kaggle-b1"  # frozen B1-recipe branch (no CLI overrides)
EXPECTED_SHA = "24e85f31e6663e3f4b4d6649f2b34c9ce2145f0e"
TARGET_IDF1 = 0.77936
TARGET_ID_SWITCHES = 154

WORK_DIR = Path("/kaggle/working")
PROJECT = WORK_DIR / "gp"
INPUT_ROOT = Path("/kaggle/input")

# ---------------------------------------------------------------- clone repo
if not PROJECT.exists():
    subprocess.check_call(
        ["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(PROJECT)]
    )
os.chdir(str(PROJECT))
sys.path.insert(0, str(PROJECT))
HEAD_SHA = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
print("HEAD:", HEAD_SHA, "(expected:", EXPECTED_SHA + ")")
if HEAD_SHA != EXPECTED_SHA:
    print("WARNING: branch moved since 14v was validated — goldens will record the new SHA")


def pip_install(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])


pip_install("faiss-cpu", "motmetrics", "loguru", "omegaconf", "rich", "networkx>=3.1", "click")
pip_install("filterpy", "ftfy", "lapx", "timm")
pip_install("--no-deps", "ultralytics")
pip_install("--no-deps", "boxmot==11.0.3")
pip_install("--no-deps", "-e", ".")

# --run-id shim (identical to 14v; older run_pipeline lacks the flag)
help_text = subprocess.check_output([sys.executable, "scripts/run_pipeline.py", "--help"], text=True)
if "--run-id" not in help_text:
    script_path = Path("scripts/run_pipeline.py")
    text = script_path.read_text(encoding="utf-8")
    text = text.replace(
        '@click.option("--dry-run", is_flag=True, default=False, help="Print resolved plan without running stages")\n',
        '@click.option("--dry-run", is_flag=True, default=False, help="Print resolved plan without running stages")\n'
        '@click.option("--run-id", default=None, help="Run id/name for the output directory")\n',
    )
    text = text.replace(
        "def main(config: str, dataset_config: str, stages: str, smoke_test: bool, dry_run: bool, override: tuple):",
        "def main(config: str, dataset_config: str, stages: str, smoke_test: bool, dry_run: bool, run_id: str | None, override: tuple):",
    )
    text = text.replace(
        "    if apply_cpu_when_no_cuda(cfg):\n",
        "    if run_id:\n        cfg.project.run_name = run_id\n\n    if apply_cpu_when_no_cuda(cfg):\n",
    )
    script_path.write_text(text, encoding="utf-8")
    print("Applied --run-id shim")

# ------------------------------------------------------------- CityFlow data
candidate_mounts = [
    INPUT_ROOT / "data-aicity-2023-track-2",
    INPUT_ROOT / "datasets" / "thanhnguyenle" / "data-aicity-2023-track-2",
]
CITYFLOW_INPUT = next((p for p in candidate_mounts if p.exists()), None)
if CITYFLOW_INPUT is None:
    raise FileNotFoundError("attach thanhnguyenle/data-aicity-2023-track-2")

TMP_DATA = Path("/tmp/datasets")
TMP_DATA.mkdir(parents=True, exist_ok=True)
DATA_RAW_PARENT = PROJECT / "data" / "raw"
if not DATA_RAW_PARENT.is_symlink():
    if DATA_RAW_PARENT.exists():
        shutil.rmtree(DATA_RAW_PARENT)
    DATA_RAW_PARENT.parent.mkdir(parents=True, exist_ok=True)
    DATA_RAW_PARENT.symlink_to(TMP_DATA)

DATA_RAW = TMP_DATA / "cityflowv2"
DATA_RAW.mkdir(parents=True, exist_ok=True)
for split_dir in sorted(CITYFLOW_INPUT.iterdir()):
    if not split_dir.is_dir() or split_dir.name not in ("train", "validation", "test"):
        continue
    for scene_dir in sorted(split_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        for cam_dir in sorted(scene_dir.iterdir()):
            if cam_dir.is_dir():
                flat = DATA_RAW / f"{scene_dir.name}_{cam_dir.name}"
                if not flat.exists():
                    flat.symlink_to(cam_dir)
cam_pattern = re.compile(r"^S\d{2}_c\d{3}$")
cams = sorted(p.name for p in DATA_RAW.iterdir() if p.is_dir() and cam_pattern.match(p.name))
print(f"CityFlowV2 ready: {len(cams)} cameras")

# GT lives force-committed in the repo (see 14v)
_repo_gt = PROJECT / "data" / "raw" / "cityflowv2"
if not any((_repo_gt / c / "gt" / "gt.txt").exists() for c in cams):
    print("WARNING: no GT visible — stage5 metrics will be empty")

# ------------------------------------------------- mounted kernel checkpoints
def find_input_dir(slug: str, owner_slug: str) -> Path:
    direct = INPUT_ROOT / slug
    if direct.exists():
        return direct
    owner, _, kernel = owner_slug.partition("/")
    nested = INPUT_ROOT / "notebooks" / owner / kernel
    if nested.exists():
        return nested
    for path in INPUT_ROOT.iterdir():
        if path.is_dir() and slug.lower() in path.name.lower():
            return path
    raise FileNotFoundError(f"mount for {owner_slug} not found under {INPUT_ROOT}")


def extract_checkpoint(owner_slug: str, extract_dir: Path) -> Path:
    slug = owner_slug.split("/", 1)[1]
    cp = find_input_dir(slug, owner_slug) / "checkpoint.tar.gz"
    if not cp.exists():
        matches = list(INPUT_ROOT.rglob("checkpoint.tar.gz"))
        matches = [m for m in matches if slug.lower() in str(m.parent).lower()]
        if not matches:
            raise FileNotFoundError(f"checkpoint.tar.gz for {owner_slug} not mounted")
        cp = matches[0]
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)
    print(f"Extracting {owner_slug} ({cp.stat().st_size / 1024**2:.1f} MB)")
    with tarfile.open(str(cp), "r:gz") as tar:
        tar.extractall(str(extract_dir))
    meta = json.loads((extract_dir / "run_metadata.json").read_text())
    return extract_dir / meta["run_name"]


SRC_14C = extract_checkpoint("yahiaakhalafallah/14c-tta-stage2", Path("/tmp/ckpt_14c"))
SRC_10A = extract_checkpoint("yahiaakhalafallah/mtmc-10a-stages-0-2", Path("/tmp/ckpt_10a"))

# ------------------------------------------------------------ assemble run dir
RUN_ID = "run_p2_goldens"
OUTPUT_ROOT = PROJECT / "data" / "outputs"
RUN_DIR = OUTPUT_ROOT / RUN_ID
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
if RUN_DIR.exists():
    shutil.rmtree(RUN_DIR)
RUN_DIR.mkdir(parents=True)

stage1_source = SRC_14C / "stage1"
if not list(stage1_source.glob("tracklets_*.json")):
    stage1_source = SRC_10A / "stage1"
stage2_source = SRC_14C / "stage2"
for name, src in [("stage1", stage1_source), ("stage2", stage2_source)]:
    (RUN_DIR / name).symlink_to(src, target_is_directory=True)

required = [
    RUN_DIR / "stage2" / "embeddings.npy",
    RUN_DIR / "stage2" / "embeddings_tertiary.npy",
    RUN_DIR / "stage2" / "embedding_index.json",
    RUN_DIR / "stage2" / "hsv_features.npy",
]
for p in required:
    assert p.exists(), f"missing golden input: {p}"
assert list((RUN_DIR / "stage1").glob("tracklets_*.json")), "no stage1 tracklets"

# ---------------------------------------------------- recipe sanity (B1 baked)
from omegaconf import OmegaConf  # noqa: E402

cfg = OmegaConf.load("configs/datasets/cityflowv2.yaml")
assert float(cfg.stage4.association.graph.similarity_threshold) == 0.48
assert int(cfg.stage4.association.query_expansion.k) == 2
assert float(cfg.stage4.association.fic.regularisation) == 0.5
assert float(cfg.stage4.association.tertiary_embeddings.weight) == 0.525
print("B1 recipe confirmed baked into branch config")

# --------------------------------------------------------------- run stages
t0 = time.time()
subprocess.check_call(
    [
        sys.executable, "scripts/run_pipeline.py",
        "--config", "configs/datasets/cityflowv2.yaml",
        "--stages", "3,4,5",
        "--run-id", RUN_ID,
    ]
)
print(f"stages 3-5 took {time.time() - t0:.0f}s")

# ---------------------------------------------------------------- drift gate
metrics_path = RUN_DIR / "stage5" / "evaluation_report.json"
if not metrics_path.exists():
    metrics_path = RUN_DIR / "stage5" / "metrics.json"
metrics = json.loads(metrics_path.read_text())
details = metrics.get("details", {}) or {}
idf1 = float(
    metrics.get("MTMC_IDF1") or metrics.get("mtmc_idf1")
    or details.get("mtmc_idf1") or metrics.get("idf1") or metrics["IDF1"]
)
id_switches = int(metrics.get("id_switches") or details.get("mtmc_id_switches") or metrics.get("IDS"))
drift = idf1 - TARGET_IDF1
print(f"IDF1={idf1:.5f} (target {TARGET_IDF1}, drift {drift:+.5f}) id_sw={id_switches}")
assert abs(drift) < 0.005, f"DRIFT GATE FAILED: {drift:+.5f}"
assert id_switches == TARGET_ID_SWITCHES, f"ID SWITCH CHECK FAILED: {id_switches}"

# ------------------------------------------------------------ package goldens
def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


GOLD = Path("/tmp/goldens")
if GOLD.exists():
    shutil.rmtree(GOLD)
for sub in ("stage1", "stage2", "stage4", "stage5"):
    (GOLD / sub).mkdir(parents=True)
for f in sorted((RUN_DIR / "stage1").glob("tracklets_*.json")):
    shutil.copy2(f, GOLD / "stage1" / f.name)
for name in ("embeddings.npy", "embeddings_tertiary.npy", "embedding_index.json", "hsv_features.npy"):
    shutil.copy2(RUN_DIR / "stage2" / name, GOLD / "stage2" / name)
secondary = RUN_DIR / "stage2" / "embeddings_secondary.npy"
if secondary.exists():
    shutil.copy2(secondary, GOLD / "stage2" / secondary.name)
shutil.copy2(RUN_DIR / "stage4" / "global_trajectories.json", GOLD / "stage4" / "global_trajectories.json")
shutil.copy2(metrics_path, GOLD / "stage5" / metrics_path.name)

file_hashes = {
    str(p.relative_to(GOLD)).replace("\\", "/"): sha256_of(p)
    for p in sorted(GOLD.rglob("*")) if p.is_file()
}
provenance = {
    "gate": "P2",
    "baseline_model_id": "vehicle_mtmc_14e_b1",
    "v1_commit": HEAD_SHA,
    "branch": BRANCH,
    "recipe": "baked into configs/datasets/cityflowv2.yaml on the branch (no overrides)",
    "sources": {
        "stage1": "yahiaakhalafallah/mtmc-10a-stages-0-2 (via 14c fallback order)",
        "stage2": "yahiaakhalafallah/14c-tta-stage2",
        "weights": "yahiaakhalafallah/mtmc-weights",
        "dataset": "thanhnguyenle/data-aicity-2023-track-2",
    },
    "metrics": {"mtmc_idf1": idf1, "id_switches": id_switches},
    "targets": {"mtmc_idf1": TARGET_IDF1, "id_switches": TARGET_ID_SWITCHES},
    "generated_unix": int(time.time()),
    "file_sha256": file_hashes,
}
(GOLD / "provenance.json").write_text(json.dumps(provenance, indent=2))

out_tar = WORK_DIR / "cityflow_b1_goldens.tar.gz"
with tarfile.open(str(out_tar), "w:gz") as tar:
    tar.add(str(GOLD), arcname="cityflow_b1_goldens")
print(f"goldens: {out_tar} ({out_tar.stat().st_size / 1024**2:.1f} MB)")
print(f"goldens sha256: {sha256_of(out_tar)}")
print("GATE P2 GOLDENS: OK")
