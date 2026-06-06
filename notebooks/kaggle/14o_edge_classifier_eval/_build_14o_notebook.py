#!/usr/bin/env python
"""Generator for notebooks/kaggle/14o_edge_classifier_eval/14o_edge_classifier_eval.ipynb.

Builds the self-contained, leak-free MTMC eval kernel for the Stage-4 learned
edge classifier. Run locally:  python _build_14o_notebook.py

Per CLAUDE.md rule 3: the notebook is written with json.dump(ensure_ascii=True),
each source line ends with '\n' except the last, and the on-disk round-trip is
verified at the end.

The kernel is fully self-contained (NO git push required): it clones the
paper-tests branch, inlines the (uncommitted) refactored build_edge_pairs.py +
edge_classifier.py via base64, PATCHES the cloned pipeline.py to insert the
Stage-4 hook, and asserts every step.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
OUT = HERE / "14o_edge_classifier_eval.ipynb"

# ---------------------------------------------------------------------------
# Base64 payloads (uncommitted local source inlined so no git push is needed).
# ---------------------------------------------------------------------------
BEP_PATH = REPO_ROOT / "scripts" / "build_edge_pairs.py"
EC_PATH = REPO_ROOT / "src" / "stage4_association" / "edge_classifier.py"
BEP_B64 = base64.b64encode(BEP_PATH.read_bytes()).decode("ascii")
EC_B64 = base64.b64encode(EC_PATH.read_bytes()).decode("ascii")

# The exact Stage-4 hook block to insert into the cloned pipeline.py, immediately
# after the `logger.info(f"Combined similarity pairs: ...")` anchor (pipeline.py
# :539 on paper-tests) and before the `# Step 5a:` comment. Kept byte-identical to
# the local pipeline.py edit by base64-embedding it.
HOOK_BLOCK = '''
    # Step 5-EC: Learned edge classifier / re-ranker (default OFF).
    # Rescores combined_sim with a learned per-edge P(same-vehicle) model before
    # any post-adjustment or graph solve. blend_lambda=0 + prob_threshold<=0 is a
    # provable no-op (returns combined_sim unchanged). See
    # docs/subagent-specs/edge-classifier-association.md sections 5-7.
    edge_clf_probs: Optional[Dict[Tuple[int, int], float]] = None
    ec_cfg = stage_cfg.get("edge_classifier", {})
    if ec_cfg.get("enabled", False):
        from src.stage4_association.edge_classifier import rescore_edges

        # cos_fused fusion weights mirror the score-level fusion (Step 3b):
        # tertiary stream == DINOv2, quaternary stream == R50-IBN.
        ec_fusion_weights = (
            round(1.0 - sec_weight - tert_weight - quat_weight, 6),
            tert_weight,
            quat_weight,
        )
        # mean confidence per tracklet from Stage-1 (matches build_edge_pairs).
        ec_tracklet_lookup: Dict[Tuple[str, int], Tracklet] = {}
        for _cam_id, _tracks in tracklets_by_camera.items():
            for _t in _tracks:
                ec_tracklet_lookup[(_t.camera_id, _t.track_id)] = _t
        track_ids = [f.track_id for f in features]
        mean_confs = [
            ec_tracklet_lookup[(cam, tid)].mean_confidence
            if (cam, tid) in ec_tracklet_lookup else 0.0
            for cam, tid in zip(camera_ids, track_ids)
        ]
        edge_clf_probs = {}
        n_before = len(combined_sim)
        combined_sim = rescore_edges(
            combined_sim,
            primary=embeddings,
            tertiary=tert_embeddings,
            quaternary=quat_embeddings,
            camera_ids=camera_ids,
            class_ids=class_ids,
            track_ids=track_ids,
            start_times=start_times,
            end_times=end_times,
            num_frames=num_frames,
            mean_confs=mean_confs,
            st_validator=st_validator,
            fusion_weights=ec_fusion_weights,
            ec_cfg=ec_cfg,
            edge_probs_out=edge_clf_probs,
        )
        logger.info(
            f"Edge classifier rescored {n_before} edges -> {len(combined_sim)} "
            f"kept (fusion_weights={ec_fusion_weights})."
        )
'''
HOOK_B64 = base64.b64encode(HOOK_BLOCK.encode("utf-8")).decode("ascii")


def src(text: str) -> list:
    """Split a code/markdown string into a notebook 'source' list.

    Each element ends with '\\n' except the last (CLAUDE.md rule 3).
    """
    lines = text.split("\n")
    out = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            out.append(line + "\n")
        else:
            if line != "":
                out.append(line)
    return out


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src(text),
    }


def chunk_b64(name: str, b64: str, width: int = 120) -> str:
    """Render a base64 string as a parenthesized multi-line Python literal."""
    parts = [b64[i:i + width] for i in range(0, len(b64), width)]
    body = "\n".join(f'    "{p}"' for p in parts)
    return f"{name} = (\n{body}\n)"


cells = []

# ---------------------------------------------------------------------------
cells.append(md(
    "# 14o -- Edge Classifier Eval (leak-free MTMC eval of the learned Stage-4 gate)\n"
    "\n"
    "CPU-only, self-contained MTMC eval for the Stage-4 learned edge classifier\n"
    "(`docs/subagent-specs/edge-classifier-association.md` sections 5-7). Builds on\n"
    "the 14n de-risk probe (PASSED: held-out hard-negative AUC +0.0735 over the cosine\n"
    "threshold) and answers the real question: **does the learned gate move actual\n"
    "MTMC IDF1 / id_switches off the 154 floor?**\n"
    "\n"
    "Pipeline:\n"
    "1. Clone `paper-tests` (reproduces 0.77936 + carries build_edge_pairs imports).\n"
    "2. Inline the (uncommitted) refactored `build_edge_pairs.py` + new\n"
    "   `edge_classifier.py` via base64; PATCH the cloned `pipeline.py` to insert the\n"
    "   Stage-4 hook (assert the patch applied). **No git push needed.**\n"
    "3. Assemble the 14e B1 stack (primary CLIP + DINOv2 tertiary; quaternary OFF).\n"
    "4. Build labelled pairs per scene, train TWO LightGBM fold models\n"
    "   (model_S02 on S02 pairs, model_S01 on S01 pairs).\n"
    "5. **Drift gate**: edge_classifier OFF must reproduce **0.77936 / id_switches 154**.\n"
    "6. **Leak-free eval**: apply model_S02 to S01 associations and model_S01 to S02\n"
    "   associations; sweep blend_lambda x prob_threshold; report MTMC IDF1 +\n"
    "   id_switches per config.\n"
    "7. Final verdict table + `14o_edge_classifier_summary.json`.\n"
    "\n"
    "Pre-registered bands: WIN >= 0.7820, MARGINAL >= 0.7810. KEY signal =\n"
    "id_switches moving off 154."
))

# ---------------------------------------------------------------------------
cells.append(md("## 1. Imports + paths"))
cells.append(code(
    "import base64\n"
    "import json\n"
    "import os\n"
    "import shutil\n"
    "import subprocess\n"
    "import sys\n"
    "import tarfile\n"
    "import time\n"
    "from datetime import datetime\n"
    "from pathlib import Path\n"
    "\n"
    "import numpy as np\n"
    "\n"
    "WORK_DIR = Path('/kaggle/working')\n"
    "PROJECT = WORK_DIR / 'gp'\n"
    "INPUT_ROOT = Path('/kaggle/input')\n"
    "ASSEMBLED_RUN = Path('/tmp/edge_clf_run')        # assembled stage1/stage2 run dir\n"
    "DATA_OUT = WORK_DIR / 'outputs'\n"
    "OUT_DIR = DATA_OUT / '14o_edge_classifier'\n"
    "OUT_DIR.mkdir(parents=True, exist_ok=True)\n"
    "MODELS_DIR = Path('/tmp/edge_clf_models')\n"
    "MODELS_DIR.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "print(f'Python: {sys.version.split()[0]}')\n"
    "print(f'Kaggle input exists: {INPUT_ROOT.exists()}')"
))

# ---------------------------------------------------------------------------
cells.append(md("## 2. Clone repo (paper-tests) + install CPU deps"))
cells.append(code(
    "REPO_URL = 'https://github.com/MRKDaGods/gp.git'\n"
    "REPO_BRANCH = 'paper-tests'   # reproduces 0.77936; carries scripts/build_edge_pairs.py imports\n"
    "\n"
    "if not PROJECT.exists():\n"
    "    print(f'Cloning {REPO_URL} ({REPO_BRANCH}) ...')\n"
    "    subprocess.check_call(['git', 'clone', '--depth', '1', '-b', REPO_BRANCH, REPO_URL, str(PROJECT)])\n"
    "else:\n"
    "    print('Repo present; pulling latest ...')\n"
    "    subprocess.check_call(['git', '-C', str(PROJECT), 'pull', '--ff-only'])\n"
    "\n"
    "os.chdir(str(PROJECT))\n"
    "if str(PROJECT) not in sys.path:\n"
    "    sys.path.insert(0, str(PROJECT))\n"
    "\n"
    "\n"
    "def pip(*args):\n"
    "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *args])\n"
    "\n"
    "\n"
    "# Stage 3-5 need faiss + networkx + omegaconf + loguru; the classifier needs\n"
    "# lightgbm + scikit-learn; build_edge_pairs writes parquet (pandas + pyarrow).\n"
    "pip('numpy', 'scipy', 'pandas', 'pyarrow', 'faiss-cpu', 'omegaconf', 'loguru',\n"
    "    'networkx>=3.1', 'lightgbm', 'scikit-learn', 'pyyaml')\n"
    "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.', '--no-deps'], cwd=str(PROJECT))\n"
    "print(f'Repo ready at {PROJECT}')"
))

# ---------------------------------------------------------------------------
cells.append(md(
    "## 3. Inline uncommitted source + PATCH pipeline.py\n"
    "\n"
    "`build_edge_pairs.py` (refactored to expose `PairFeatureBuilder`) and the new\n"
    "`edge_classifier.py` are uncommitted -> absent from the clone. We write them\n"
    "from base64, then insert the Stage-4 hook into the cloned `pipeline.py` so the\n"
    "kernel runs WITHOUT a git push. Every step is asserted."
))
cells.append(code(
    "# --- (a) Inline the refactored build_edge_pairs.py (PairFeatureBuilder source of truth). ---\n"
    + chunk_b64("_BEP_B64", BEP_B64) + "\n"
    "_bep_path = PROJECT / 'scripts' / 'build_edge_pairs.py'\n"
    "_bep_path.write_bytes(base64.b64decode(_BEP_B64))\n"
    "print(f'Wrote {_bep_path} ({_bep_path.stat().st_size} bytes)')\n"
    "\n"
    "# --- (b) Inline the new edge_classifier.py. ---\n"
    + chunk_b64("_EC_B64", EC_B64) + "\n"
    "_ec_path = PROJECT / 'src' / 'stage4_association' / 'edge_classifier.py'\n"
    "_ec_path.write_bytes(base64.b64decode(_EC_B64))\n"
    "print(f'Wrote {_ec_path} ({_ec_path.stat().st_size} bytes)')\n"
    "\n"
    "# --- (c) PATCH pipeline.py: insert the Stage-4 hook after the combined_sim anchor. ---\n"
    + chunk_b64("_HOOK_B64", HOOK_B64) + "\n"
    "HOOK_BLOCK = base64.b64decode(_HOOK_B64).decode('utf-8')\n"
    "_pipe_path = PROJECT / 'src' / 'stage4_association' / 'pipeline.py'\n"
    "_pipe_src = _pipe_path.read_text(encoding='utf-8')\n"
    "\n"
    "ANCHOR = '    logger.info(f\"Combined similarity pairs: {len(combined_sim)}\")\\n'\n"
    "STEP5A = '    # Step 5a: Per-camera-pair similarity normalization.'\n"
    "if 'Step 5-EC: Learned edge classifier' in _pipe_src:\n"
    "    raise RuntimeError('pipeline.py already contains the edge-classifier hook (unexpected on paper-tests).')\n"
    "if _pipe_src.count(ANCHOR) != 1:\n"
    "    raise RuntimeError(f'Expected exactly 1 combined_sim anchor, found {_pipe_src.count(ANCHOR)}.')\n"
    "if STEP5A not in _pipe_src:\n"
    "    raise RuntimeError('Could not find the Step 5a marker to anchor the hook insertion.')\n"
    "\n"
    "# Insert HOOK_BLOCK between the anchor line and the Step 5a comment.\n"
    "_target = ANCHOR + '\\n' + STEP5A\n"
    "if _pipe_src.count(_target) != 1:\n"
    "    raise RuntimeError('Anchor+Step5a target not found exactly once; pipeline.py layout changed.')\n"
    "_replacement = ANCHOR + HOOK_BLOCK + '\\n' + STEP5A\n"
    "_pipe_src_new = _pipe_src.replace(_target, _replacement, 1)\n"
    "if _pipe_src_new == _pipe_src:\n"
    "    raise RuntimeError('Patch produced no change -- hook NOT inserted.')\n"
    "_pipe_path.write_text(_pipe_src_new, encoding='utf-8')\n"
    "\n"
    "# Assert the patch applied + the module imports cleanly with the hook present.\n"
    "_check = _pipe_path.read_text(encoding='utf-8')\n"
    "assert 'Step 5-EC: Learned edge classifier' in _check, 'hook marker missing after patch'\n"
    "assert 'from src.stage4_association.edge_classifier import rescore_edges' in _check, 'hook import missing'\n"
    "\n"
    "# Force a fresh import of the patched module (clear any cached import).\n"
    "for _m in list(sys.modules):\n"
    "    if _m.startswith('src.stage4_association') or _m == 'scripts.build_edge_pairs':\n"
    "        del sys.modules[_m]\n"
    "import importlib\n"
    "import scripts.build_edge_pairs as _bep_mod\n"
    "import src.stage4_association.edge_classifier as _ec_mod\n"
    "import src.stage4_association.pipeline as _pipe_mod\n"
    "assert hasattr(_bep_mod, 'PairFeatureBuilder'), 'PairFeatureBuilder missing from inlined build_edge_pairs'\n"
    "assert hasattr(_ec_mod, 'rescore_edges'), 'rescore_edges missing from edge_classifier'\n"
    "print('PATCH OK: hook inserted, modules import cleanly.')\n"
    "print(f'  FEATURE_NAMES ({len(_bep_mod.FEATURE_NAMES)}): {_bep_mod.FEATURE_NAMES}')"
))

# ---------------------------------------------------------------------------
cells.append(md(
    "## 4. Resolve 14h anchor, 14j quaternary, GT (from 14n)\n"
    "\n"
    "kernel_sources mount under `/kaggle/input/notebooks/<owner>/<slug>/` (recurse\n"
    "search as fallback). The 14h checkpoint supplies stage1 + primary CLIP +\n"
    "DINOv2 tertiary; 14j supplies the R50-IBN quaternary (used only to BUILD the\n"
    "classifier's `cos_r50ibn` feature -- the MTMC base stack keeps quaternary OFF)."
))
cells.append(code(
    "SOURCE_14H_OWNER_SLUG = 'yahiaakhalafallah/14h-robust-tracklet-pooling'\n"
    "SOURCE_14J_OWNER_SLUG = 'yahiaakhalafallah/14j-r50-ibn-features'\n"
    "SOURCE_14H_SLUG = SOURCE_14H_OWNER_SLUG.split('/', 1)[1]\n"
    "SOURCE_14J_SLUG = SOURCE_14J_OWNER_SLUG.split('/', 1)[1]\n"
    "EXPECTED_CAMS = ['S01_c001', 'S01_c002', 'S01_c003', 'S02_c006', 'S02_c007', 'S02_c008']\n"
    "EXPECTED_TRACKLETS = 929\n"
    "\n"
    "\n"
    "def find_input_dir(slug, owner_slug, hints=()):\n"
    "    direct = INPUT_ROOT / slug\n"
    "    if direct.exists():\n"
    "        return direct\n"
    "    owner, _, kernel = owner_slug.partition('/')\n"
    "    nested = INPUT_ROOT / 'notebooks' / owner / kernel\n"
    "    if nested.exists():\n"
    "        return nested\n"
    "    lowered_slug = slug.lower()\n"
    "    lowered_hints = tuple(str(h).lower() for h in hints)\n"
    "    for path in (list(INPUT_ROOT.iterdir()) if INPUT_ROOT.exists() else []):\n"
    "        if not path.is_dir():\n"
    "            continue\n"
    "        name = path.name.lower()\n"
    "        if lowered_slug in name or all(h in name for h in lowered_hints):\n"
    "            return path\n"
    "    return direct\n"
    "\n"
    "\n"
    "def find_14h_checkpoint():\n"
    "    source_dir = find_input_dir(SOURCE_14H_SLUG, SOURCE_14H_OWNER_SLUG, hints=('14h', 'robust', 'tracklet'))\n"
    "    cp = source_dir / 'checkpoint.tar.gz'\n"
    "    if cp.exists():\n"
    "        print(f'14h input: {source_dir}')\n"
    "        return cp\n"
    "    visible = [str(p) for p in INPUT_ROOT.rglob('checkpoint.tar.gz')] if INPUT_ROOT.exists() else []\n"
    "    raise FileNotFoundError(\n"
    "        f'14h checkpoint.tar.gz not found for {SOURCE_14H_OWNER_SLUG}. Visible: {visible[:20]}')\n"
    "\n"
    "\n"
    "checkpoint = find_14h_checkpoint()\n"
    "EXTRACT_DIR = Path('/tmp/14h_checkpoint')\n"
    "if EXTRACT_DIR.exists():\n"
    "    shutil.rmtree(EXTRACT_DIR)\n"
    "EXTRACT_DIR.mkdir(parents=True, exist_ok=True)\n"
    "print(f'Extracting {checkpoint} ({checkpoint.stat().st_size / 1024**2:.1f} MB)')\n"
    "with tarfile.open(str(checkpoint), 'r:gz') as archive:\n"
    "    archive.extractall(str(EXTRACT_DIR))\n"
    "\n"
    "with open(EXTRACT_DIR / 'run_metadata.json', encoding='utf-8') as fh:\n"
    "    previous_meta = json.load(fh)\n"
    "SOURCE_14H_RUN_NAME = previous_meta['run_name']\n"
    "SOURCE_14H_RUN_DIR = EXTRACT_DIR / SOURCE_14H_RUN_NAME\n"
    "SOURCE_STAGE1_DIR = SOURCE_14H_RUN_DIR / 'stage1'\n"
    "SOURCE_STAGE2_DIR = SOURCE_14H_RUN_DIR / 'stage2'\n"
    "for required in [\n"
    "    SOURCE_STAGE1_DIR,\n"
    "    SOURCE_STAGE2_DIR / 'embeddings.npy',\n"
    "    SOURCE_STAGE2_DIR / 'embeddings_tertiary.npy',\n"
    "    SOURCE_STAGE2_DIR / 'hsv_features.npy',\n"
    "    SOURCE_STAGE2_DIR / 'embedding_index.json',\n"
    "]:\n"
    "    if not required.exists():\n"
    "        raise FileNotFoundError(required)\n"
    "print(f'Loaded 14h run: {SOURCE_14H_RUN_NAME}')"
))
cells.append(code(
    "def find_quaternary_stage2_dir():\n"
    "    source_dir = find_input_dir(SOURCE_14J_SLUG, SOURCE_14J_OWNER_SLUG, hints=('14j', 'r50', 'ibn'))\n"
    "    candidates = [\n"
    "        source_dir / 'outputs' / '14j_v4_features' / 'stage2',\n"
    "        source_dir / '14j_v4_features' / 'stage2',\n"
    "        source_dir / 'stage2',\n"
    "    ]\n"
    "    for cand in candidates:\n"
    "        if (cand / 'embeddings_quaternary.npy').exists():\n"
    "            print(f'14j quaternary input: {cand}')\n"
    "            return cand\n"
    "    matches = sorted(INPUT_ROOT.rglob('embeddings_quaternary.npy')) if INPUT_ROOT.exists() else []\n"
    "    for m in matches:\n"
    "        t = str(m).lower()\n"
    "        if '14j' in t and ('r50' in t or 'ibn' in t or 'quaternary' in t):\n"
    "            print(f'14j quaternary discovered: {m.parent}')\n"
    "            return m.parent\n"
    "    if matches:\n"
    "        print(f'14j quaternary fallback: {matches[0].parent}')\n"
    "        return matches[0].parent\n"
    "    visible = [str(p) for p in INPUT_ROOT.rglob('*.npy')] if INPUT_ROOT.exists() else []\n"
    "    raise FileNotFoundError(\n"
    "        f'embeddings_quaternary.npy not found for {SOURCE_14J_OWNER_SLUG}. Visible npy: {visible[:30]}')\n"
    "\n"
    "\n"
    "SOURCE_QUATERNARY_STAGE2_DIR = find_quaternary_stage2_dir()\n"
    "\n"
    "\n"
    "def is_cityflow_gt_root(path):\n"
    "    return path.exists() and all((path / cam / 'gt' / 'gt.txt').exists() for cam in EXPECTED_CAMS)\n"
    "\n"
    "\n"
    "def find_cityflow_gt_root():\n"
    "    candidates = [\n"
    "        PROJECT / 'data' / 'raw' / 'cityflowv2',\n"
    "        EXTRACT_DIR / 'gt_annotations',\n"
    "        Path('/kaggle/input/data-aicity-2023-track-2'),\n"
    "        Path('/kaggle/input/datasets/thanhnguyenle/data-aicity-2023-track-2'),\n"
    "    ]\n"
    "    for cand in candidates:\n"
    "        if is_cityflow_gt_root(cand):\n"
    "            return cand\n"
    "    for gt_file in (INPUT_ROOT.rglob('gt.txt') if INPUT_ROOT.exists() else []):\n"
    "        if gt_file.parent.name != 'gt' or gt_file.parent.parent.name not in EXPECTED_CAMS:\n"
    "            continue\n"
    "        cand = gt_file.parents[2]\n"
    "        if is_cityflow_gt_root(cand):\n"
    "            return cand\n"
    "    visible = [str(p) for p in INPUT_ROOT.rglob('gt.txt')] if INPUT_ROOT.exists() else []\n"
    "    raise FileNotFoundError(\n"
    "        'CityFlowV2 GT not found in <root>/<cam>/gt/gt.txt layout. '\n"
    "        f'Expected {EXPECTED_CAMS}. Visible gt.txt: {visible[:20]}')\n"
    "\n"
    "\n"
    "GT_DIR = find_cityflow_gt_root()\n"
    "print(f'Ground truth root: {GT_DIR}')"
))

# ---------------------------------------------------------------------------
cells.append(md(
    "## 5. Assemble a 2-STREAM run dir (stage1 + primary + DINOv2 tertiary)\n"
    "\n"
    "CRITICAL train/inference-match decision: the MTMC base is the **clean 2-stream\n"
    "14e B1 stack** (primary CLIP + DINOv2 tertiary; **quaternary R50-IBN OFF**). The\n"
    "live pipeline therefore never loads the quaternary stream, so its hook produces\n"
    "`cos_r50ibn = 0`. To keep the classifier's TRAINING features bit-identical to\n"
    "what the pipeline feeds it at INFERENCE (the spec's hard 'train/infer\n"
    "distribution match'), we build the labelled pairs on a **2-stream** run dir too\n"
    "(`run.quaternary = None` -> `cos_r50ibn = 0` everywhere). The 14j quaternary is\n"
    "still mounted + row-aligned-asserted (provenance), but intentionally NOT written\n"
    "into the run -- including it would silently desync train vs infer on the\n"
    "`cos_r50ibn / cos_min / cos_max / cos_std` features. `cos_std` over the two\n"
    "present streams (primary, DINOv2) still carries the stream-disagreement signal."
))
cells.append(code(
    "src_index = json.loads((SOURCE_STAGE2_DIR / 'embedding_index.json').read_text(encoding='utf-8'))\n"
    "if len(src_index) != EXPECTED_TRACKLETS:\n"
    "    raise RuntimeError(f'Expected {EXPECTED_TRACKLETS} rows, found {len(src_index)}')\n"
    "\n"
    "# Provenance only: assert the 14j quaternary aligns to the 14h ordering, then\n"
    "# DO NOT use it (2-stream base stack). This catches a stale/misaligned 14j mount.\n"
    "quat_index = json.loads((SOURCE_QUATERNARY_STAGE2_DIR / 'embedding_index.json').read_text(encoding='utf-8'))\n"
    "if src_index != quat_index:\n"
    "    raise RuntimeError('14j quaternary embedding_index.json does not match 14h ordering')\n"
    "quat_probe = np.load(SOURCE_QUATERNARY_STAGE2_DIR / 'embeddings_quaternary.npy').astype(np.float32)\n"
    "if quat_probe.shape[0] != EXPECTED_TRACKLETS:\n"
    "    raise RuntimeError(f'Unexpected quaternary shape: {quat_probe.shape}')\n"
    "print(f'14j quaternary aligned (provenance only, NOT used): {quat_probe.shape}')\n"
    "\n"
    "if ASSEMBLED_RUN.exists():\n"
    "    shutil.rmtree(ASSEMBLED_RUN)\n"
    "(ASSEMBLED_RUN / 'stage2').mkdir(parents=True, exist_ok=True)\n"
    "shutil.copytree(SOURCE_STAGE1_DIR, ASSEMBLED_RUN / 'stage1')\n"
    "# 2-stream stage2 ONLY: primary + DINOv2 tertiary + hsv + index. No quaternary.\n"
    "for fname in ['embeddings.npy', 'embeddings_tertiary.npy', 'hsv_features.npy', 'embedding_index.json']:\n"
    "    shutil.copy2(SOURCE_STAGE2_DIR / fname, ASSEMBLED_RUN / 'stage2' / fname)\n"
    "assert not (ASSEMBLED_RUN / 'stage2' / 'embeddings_quaternary.npy').exists(), \\\n"
    "    'quaternary must be ABSENT from the 2-stream training run (would desync train vs infer)'\n"
    "\n"
    "TERTIARY_PATH = ASSEMBLED_RUN / 'stage2' / 'embeddings_tertiary.npy'\n"
    "print('Assembled 2-stream run dir stage2 contents:')\n"
    "for p in sorted((ASSEMBLED_RUN / 'stage2').iterdir()):\n"
    "    print(f'  stage2/{p.name}')\n"
    "print(f'  stage1/: {len(list((ASSEMBLED_RUN / \"stage1\").glob(\"tracklets_*.json\")))} tracklet files')"
))

# ---------------------------------------------------------------------------
cells.append(md(
    "## 6. Sanity self-test the inlined build_edge_pairs.py\n"
    "\n"
    "Exercises every feature-builder code path on tiny synthetic data -- catches an\n"
    "env break before the real run."
))
cells.append(code(
    "rc = subprocess.call([sys.executable, 'scripts/build_edge_pairs.py', '--self-test'])\n"
    "print(f'self-test exit code: {rc}')\n"
    "if rc != 0:\n"
    "    raise RuntimeError('build_edge_pairs self-test failed -- fix env before the real run')"
))

# ---------------------------------------------------------------------------
cells.append(md(
    "## 7. Build labelled pairs + train TWO leak-free fold models\n"
    "\n"
    "`build_edge_pairs` produces per-pair features in the EXACT FIC+AQE space the\n"
    "live gate uses. We use the **14e B1 base-stack fusion weights**\n"
    "`(w_primary=0.475, w_tertiary=0.525, w_quaternary=0.0)` -- NOT the K7 registry\n"
    "weights -- because the MTMC base stack is 2-stream, so `cos_fused` here equals\n"
    "the appearance component the pipeline actually fuses at inference. We split by\n"
    "scene and train:\n"
    "* **model_S02** on S02 pairs only,\n"
    "* **model_S01** on S01 pairs only.\n"
    "\n"
    "At eval (Cell 9) model_S02 scores S01 associations and model_S01 scores S02\n"
    "associations -- **no tracklet is ever scored by a model trained on its own\n"
    "scene**. The fold dict is pickled to `models_by_train_scene` so\n"
    "`edge_classifier.rescore_edges` applies the held-out model per scene\n"
    "automatically (and asserts the mapping)."
))
cells.append(code(
    "import pickle\n"
    "import importlib\n"
    "import lightgbm as lgb\n"
    "\n"
    "M = importlib.import_module('scripts.build_edge_pairs')\n"
    "from src.stage4_association.spatial_temporal import SpatioTemporalValidator  # noqa\n"
    "\n"
    "# Load the 2-stream run in the live FIC+AQE feature space (primary FIC+AQE;\n"
    "# DINOv2 tertiary FIC-only; quaternary absent -> cos_r50ibn=0) -- identical\n"
    "# transforms to the 14e B1 gate. AQE k=2 matches the base stack.\n"
    "run = M.load_run(\n"
    "    ASSEMBLED_RUN, GT_DIR, raw_cosines=False, fic_reg=M.DEFAULT_FIC_REG,\n"
    "    fic_min_samples=M.DEFAULT_FIC_MIN_SAMPLES, aqe_k=M.DEFAULT_AQE_K,\n"
    "    aqe_alpha=M.DEFAULT_AQE_ALPHA, top_k=M.DEFAULT_TOP_K,\n"
    ")\n"
    "if run.quaternary is not None:\n"
    "    raise RuntimeError('Expected a 2-stream run (quaternary None); got a quaternary stream.')\n"
    "# 14e B1 base-stack fusion weights -> cos_fused == pipeline appearance fusion.\n"
    "FUSION_WEIGHTS = (0.475, 0.525, 0.0)\n"
    "print(f'14e B1 base-stack fusion weights (NOT K7): {FUSION_WEIGHTS}')\n"
    "st_validator = M._build_st_validator(M._load_camera_transitions())\n"
    "cols = M.build_pairs(run, st_validator, fusion_weights=FUSION_WEIGHTS)\n"
    "\n"
    "FEATURE_NAMES = list(M.FEATURE_NAMES)\n"
    "CAT_FEATURES = list(M.CATEGORICAL_FEATURES)\n"
    "X_all = np.column_stack([np.array(cols[name], dtype=np.float64) for name in FEATURE_NAMES])\n"
    "y_all = np.array(cols['label'], dtype=np.int64)\n"
    "scene_all = np.array(cols['scene'])\n"
    "scenes_present = sorted(set(s for s in scene_all.tolist() if s))\n"
    "print(f'Pairs total={len(y_all)} positives={int(y_all.sum())} scenes={scenes_present}')\n"
    "if set(scenes_present) != {'S01', 'S02'}:\n"
    "    raise RuntimeError(f'Expected exactly scenes S01+S02 for leak-free folds, got {scenes_present}')\n"
    "\n"
    "\n"
    "def train_fold(train_scene):\n"
    "    mask = scene_all == train_scene\n"
    "    Xtr, ytr = X_all[mask], y_all[mask]\n"
    "    if ytr.sum() == 0 or (ytr == 0).sum() == 0:\n"
    "        raise RuntimeError(f'Degenerate labels for train scene {train_scene}: pos={int(ytr.sum())}')\n"
    "    cat_idx = [FEATURE_NAMES.index(c) for c in CAT_FEATURES]\n"
    "    mdl = M._train_lgbm(Xtr, ytr, cat_idx)\n"
    "    print(f'  trained model_{train_scene}: n={len(ytr)} pos={int(ytr.sum())} neg={int((ytr==0).sum())}')\n"
    "    return mdl\n"
    "\n"
    "\n"
    "model_S02 = train_fold('S02')   # trained on S02 -> scores S01 associations\n"
    "model_S01 = train_fold('S01')   # trained on S01 -> scores S02 associations\n"
    "\n"
    "FOLD_PAYLOAD = {\n"
    "    'feature_version': 1,\n"
    "    'feature_names': FEATURE_NAMES,\n"
    "    'fusion_weights': list(FUSION_WEIGHTS),\n"
    "    'models_by_train_scene': {'S02': model_S02, 'S01': model_S01},\n"
    "    'leak_free_mapping': {'S01_pairs': 'model_S02', 'S02_pairs': 'model_S01'},\n"
    "}\n"
    "MODEL_PATH = MODELS_DIR / 'edge_clf_lgbm_folds.pkl'\n"
    "with MODEL_PATH.open('wb') as fh:\n"
    "    pickle.dump(FOLD_PAYLOAD, fh)\n"
    "print(f'Saved fold models -> {MODEL_PATH}')\n"
    "\n"
    "# Assert the leak-free routing BEFORE any eval: scene X must be scored by the\n"
    "# model whose train-scene != X.\n"
    "from src.stage4_association.edge_classifier import EdgeClassifierModel\n"
    "_ecm = EdgeClassifierModel(FOLD_PAYLOAD, MODEL_PATH)\n"
    "assert _ecm.scene_to_model('S01') is model_S02, 'LEAK: S01 must be scored by model_S02'\n"
    "assert _ecm.scene_to_model('S02') is model_S01, 'LEAK: S02 must be scored by model_S01'\n"
    "print('LEAK-FREE MAPPING ASSERTED: S01->model_S02, S02->model_S01 (never train on the scored scene).')"
))

# ---------------------------------------------------------------------------
cells.append(md(
    "## 8. Stage 3-5 driver (14e B1 base stack)\n"
    "\n"
    "Mirrors the 14k eval driver's K0 config -- the clean 2-stream 14e B1 stack\n"
    "(primary CLIP + DINOv2 tertiary, quaternary OFF, `aqe_k=2, sim_thr=0.48,\n"
    "fic=0.5`) that reproduces **0.77936 / id_switches 154**. The only added knob is\n"
    "the `edge_classifier.*` override block."
))
cells.append(code(
    "from src.core.config import load_config, save_config\n"
    "from src.core.data_models import TrackletFeatures\n"
    "from src.core.io_utils import load_tracklets_by_camera\n"
    "from src.core.logging_utils import setup_logging\n"
    "from src.stage3_indexing import run_stage3\n"
    "from src.stage4_association import run_stage4\n"
    "from src.stage5_evaluation import run_stage5\n"
    "\n"
    "RUN_NAME = f\"run_14o_edge_clf_{datetime.now().strftime('%Y%m%d_%H%M%S')}\"\n"
    "RUN_DIR = DATA_OUT / RUN_NAME\n"
    "RUN_DIR.mkdir(parents=True, exist_ok=True)\n"
    "setup_logging(level='INFO', log_file=RUN_DIR / 'pipeline.log')\n"
    "print(f'Run: {RUN_NAME}')\n"
    "\n"
    "# --- 14e B1 base stack constants (K0 config from 14k) ---\n"
    "BASE_PRIMARY_WEIGHT = 0.475\n"
    "BASE_TERTIARY_WEIGHT = 0.525\n"
    "SIM_THRESHOLD = 0.48\n"
    "AQE_K = 2\n"
    "FIC_REG = 0.5\n"
    "SOLVER = 'cc'\n"
    "ALGORITHM = 'conflict_free_cc'\n"
    "LOUVAIN_RES = 0.70\n"
    "APPEARANCE_WEIGHT = 0.70\n"
    "HSV_WEIGHT = 0.0\n"
    "ST_WEIGHT = round(1.0 - APPEARANCE_WEIGHT - HSV_WEIGHT, 4)\n"
    "BRIDGE_PRUNE = 0.0\n"
    "MAX_COMP_SIZE = 12\n"
    "GALLERY_THRESH = 0.48\n"
    "ORPHAN_MATCH_THRESH = 0.38\n"
    "INTRA_MERGE = True\n"
    "INTRA_MERGE_THRESH = 0.80\n"
    "INTRA_MERGE_GAP = 30\n"
    "MULTI_QUERY_WEIGHT = 0.0\n"
    "MTMC_ONLY = False\n"
    "\n"
    "K0_REPRO_TARGET = 0.77936\n"
    "K0_REPRO_TOL = 0.001\n"
    "K0_ID_SWITCH_TARGET = 154\n"
    "WIN_THRESHOLD = 0.7820\n"
    "MARGINAL_MIN = 0.7810\n"
    "\n"
    "\n"
    "def load_metrics(report_path):\n"
    "    if not report_path.exists():\n"
    "        return {}\n"
    "    payload = json.loads(report_path.read_text(encoding='utf-8'))\n"
    "    details = payload.get('details', {}) or {}\n"
    "    error_analysis = details.get('error_analysis', {}) or {}\n"
    "    return {\n"
    "        'mtmc_idf1': payload.get('mtmc_idf1', details.get('mtmc_idf1', payload.get('idf1'))),\n"
    "        'trackeval_idf1': payload.get('idf1'),\n"
    "        'idp': payload.get('idp', details.get('idp')),\n"
    "        'idr': payload.get('idr', details.get('idr')),\n"
    "        'mota': payload.get('mota'),\n"
    "        'hota': payload.get('hota'),\n"
    "        'id_switches': payload.get('id_switches'),\n"
    "        'conflations': error_analysis.get('conflated_pred'),\n"
    "        'fragmentations': error_analysis.get('fragmented_gt'),\n"
    "        'num_pred_ids': payload.get('num_pred_ids', error_analysis.get('total_pred')),\n"
    "    }\n"
    "\n"
    "\n"
    "def build_features(stage2_dir):\n"
    "    index_map = json.loads((stage2_dir / 'embedding_index.json').read_text(encoding='utf-8'))\n"
    "    embeddings = np.load(stage2_dir / 'embeddings.npy').astype(np.float32)\n"
    "    hsv_features = np.load(stage2_dir / 'hsv_features.npy').astype(np.float32)\n"
    "    if embeddings.shape[0] != len(index_map) or hsv_features.shape[0] != len(index_map):\n"
    "        raise ValueError(\n"
    "            f'Stage2 row mismatch: embeddings={embeddings.shape}, hsv={hsv_features.shape}, index={len(index_map)}')\n"
    "    return [\n"
    "        TrackletFeatures(\n"
    "            track_id=int(row['track_id']), camera_id=str(row['camera_id']),\n"
    "            class_id=int(row['class_id']), embedding=embeddings[row_index],\n"
    "            hsv_histogram=hsv_features[row_index], raw_embedding=None,\n"
    "            multi_query_embeddings=None,\n"
    "        )\n"
    "        for row_index, row in enumerate(index_map)\n"
    "    ]\n"
    "\n"
    "\n"
    "def build_overrides(config, config_run_name):\n"
    "    ec = config['edge_classifier']\n"
    "    return [\n"
    "        f'project.run_name={config_run_name}',\n"
    "        f'project.output_dir={DATA_OUT}',\n"
    "        'stage0.cameras=[S01_c001,S01_c002,S01_c003,S02_c006,S02_c007,S02_c008]',\n"
    "        f'stage4.association.query_expansion.k={AQE_K}',\n"
    "        'stage4.association.query_expansion.alpha=5.0',\n"
    "        'stage4.association.query_expansion.dba=false',\n"
    "        f'stage4.association.graph.similarity_threshold={SIM_THRESHOLD}',\n"
    "        f'stage4.association.solver={SOLVER}',\n"
    "        f'stage4.association.graph.algorithm={ALGORITHM}',\n"
    "        f'stage4.association.graph.louvain_resolution={LOUVAIN_RES}',\n"
    "        f'stage4.association.graph.bridge_prune_margin={BRIDGE_PRUNE}',\n"
    "        f'stage4.association.graph.max_component_size={MAX_COMP_SIZE}',\n"
    "        f'stage4.association.weights.vehicle.appearance={APPEARANCE_WEIGHT}',\n"
    "        f'stage4.association.weights.vehicle.hsv={HSV_WEIGHT}',\n"
    "        f'stage4.association.weights.vehicle.spatiotemporal={ST_WEIGHT}',\n"
    "        'stage4.association.mutual_nn.top_k_per_query=20',\n"
    "        'stage4.association.fic.enabled=true',\n"
    "        f'stage4.association.fic.regularisation={FIC_REG}',\n"
    "        'stage4.association.reranking.enabled=false',\n"
    "        'stage4.association.camera_pair_norm.enabled=false',\n"
    "        'stage4.association.fac.enabled=false',\n"
    "        f'stage4.association.multi_query.enabled={str(MULTI_QUERY_WEIGHT > 0.0).lower()}',\n"
    "        f'stage4.association.multi_query.weight={MULTI_QUERY_WEIGHT}',\n"
    "        # --- 14e B1 base stack: quaternary OFF, DINOv2 tertiary ON ---\n"
    "        'stage4.association.secondary_embeddings.path=',\n"
    "        'stage4.association.secondary_embeddings.weight=0.0',\n"
    "        f'stage4.association.tertiary_embeddings.path={TERTIARY_PATH}',\n"
    "        f'stage4.association.tertiary_embeddings.weight={BASE_TERTIARY_WEIGHT}',\n"
    "        'stage4.association.quaternary_embeddings.path=',\n"
    "        'stage4.association.quaternary_embeddings.weight=0.0',\n"
    "        'stage4.association.camera_bias.enabled=false',\n"
    "        'stage4.association.zone_model.enabled=false',\n"
    "        'stage4.association.hierarchical.enabled=false',\n"
    "        f'stage4.association.intra_camera_merge.enabled={str(INTRA_MERGE).lower()}',\n"
    "        f'stage4.association.intra_camera_merge.threshold={INTRA_MERGE_THRESH}',\n"
    "        f'stage4.association.intra_camera_merge.max_time_gap={INTRA_MERGE_GAP}',\n"
    "        'stage4.association.gallery_expansion.enabled=true',\n"
    "        f'stage4.association.gallery_expansion.threshold={GALLERY_THRESH}',\n"
    "        f'stage4.association.gallery_expansion.orphan_match_threshold={ORPHAN_MATCH_THRESH}',\n"
    "        'stage4.association.weights.length_weight_power=0.3',\n"
    "        'stage4.association.temporal_overlap.enabled=true',\n"
    "        'stage4.association.temporal_overlap.bonus=0.05',\n"
    "        'stage4.association.temporal_overlap.max_mean_time=5.0',\n"
    "        # --- edge classifier overrides ---\n"
    "        f'stage4.association.edge_classifier.enabled={str(ec[\"enabled\"]).lower()}',\n"
    "        f'stage4.association.edge_classifier.model_path={MODEL_PATH}',\n"
    "        f'stage4.association.edge_classifier.mode={ec[\"mode\"]}',\n"
    "        f'stage4.association.edge_classifier.blend_lambda={ec[\"blend_lambda\"]}',\n"
    "        f'stage4.association.edge_classifier.prob_threshold={ec[\"prob_threshold\"]}',\n"
    "        # --- stage 5 ---\n"
    "        f'stage5.mtmc_only_submission={str(MTMC_ONLY).lower()}',\n"
    "        'stage5.stationary_filter.enabled=true',\n"
    "        'stage5.stationary_filter.min_displacement_px=150',\n"
    "        'stage5.stationary_filter.max_mean_velocity_px=2.0',\n"
    "        'stage5.min_submission_confidence=0.15',\n"
    "        'stage5.cross_id_nms_iou=0.40',\n"
    "        'stage5.min_trajectory_confidence=0.30',\n"
    "        'stage5.min_trajectory_frames=40',\n"
    "        'stage5.track_edge_trim.enabled=false',\n"
    "        'stage5.track_smoothing.enabled=false',\n"
    "        'stage5.gt_frame_clip=true',\n"
    "        'stage5.gt_zone_filter=true',\n"
    "        f'stage5.ground_truth_dir={GT_DIR}',\n"
    "    ]\n"
    "\n"
    "\n"
    "def run_config(config):\n"
    "    config_id = config['config_id']\n"
    "    config_dir = RUN_DIR / config_id\n"
    "    config_dir.mkdir(parents=True, exist_ok=True)\n"
    "    tracklets_by_camera = load_tracklets_by_camera(SOURCE_STAGE1_DIR)\n"
    "    features = build_features(SOURCE_STAGE2_DIR)\n"
    "    ec = config['edge_classifier']\n"
    "    print('\\n' + '=' * 80)\n"
    "    print(f\"Running {config_id}: edge_clf enabled={ec['enabled']} mode={ec['mode']} \"\n"
    "          f\"lambda={ec['blend_lambda']} prob_thr={ec['prob_threshold']}\")\n"
    "    print('=' * 80)\n"
    "    config_run_name = f'{RUN_NAME}_{config_id}'\n"
    "    cfg = load_config(\n"
    "        'configs/default.yaml', dataset_config='configs/datasets/cityflowv2.yaml',\n"
    "        overrides=build_overrides(config, config_run_name),\n"
    "    )\n"
    "    save_config(cfg, config_dir / 'config.yaml')\n"
    "\n"
    "    faiss_index, metadata_store = run_stage3(cfg, features, tracklets_by_camera, output_dir=config_dir / 'stage3')\n"
    "    trajectories = run_stage4(cfg, faiss_index, metadata_store, features, tracklets_by_camera, output_dir=config_dir / 'stage4')\n"
    "    run_stage5(cfg, trajectories, output_dir=config_dir / 'stage5')\n"
    "\n"
    "    report_path = config_dir / 'stage5' / 'evaluation_report.json'\n"
    "    metrics = load_metrics(report_path)\n"
    "    pred_dir = config_dir / 'stage5' / 'predictions_mot'\n"
    "    pred_files = sorted(pred_dir.glob('*.txt')) if pred_dir.exists() else []\n"
    "    idf1_value = metrics.get('mtmc_idf1') or metrics.get('trackeval_idf1')\n"
    "    if idf1_value is None:\n"
    "        raise RuntimeError(f'IDF1 not found in {report_path}')\n"
    "    if not pred_files:\n"
    "        raise RuntimeError(f'No MOT prediction files written for {config_id}')\n"
    "    row = {\n"
    "        'config_id': config_id,\n"
    "        'enabled': ec['enabled'], 'mode': ec['mode'],\n"
    "        'blend_lambda': ec['blend_lambda'], 'prob_threshold': ec['prob_threshold'],\n"
    "        'mtmc_idf1': metrics.get('mtmc_idf1'), 'trackeval_idf1': metrics.get('trackeval_idf1'),\n"
    "        'idp': metrics.get('idp'), 'idr': metrics.get('idr'),\n"
    "        'id_switches': metrics.get('id_switches'),\n"
    "        'mota': metrics.get('mota'), 'hota': metrics.get('hota'),\n"
    "        'conflations': metrics.get('conflations'), 'fragmentations': metrics.get('fragmentations'),\n"
    "        'num_pred_ids': metrics.get('num_pred_ids'), 'num_trajectories': len(trajectories),\n"
    "        'notes': config.get('notes', ''),\n"
    "    }\n"
    "    print(f\"{config_id} MTMC IDF1: {float(idf1_value):.5f}  id_switches={row['id_switches']}\")\n"
    "    return row"
))

# ---------------------------------------------------------------------------
cells.append(md(
    "## 9. DRIFT GATE + leak-free sweep\n"
    "\n"
    "**Drift gate first**: edge_classifier OFF must reproduce 0.77936 / id_switches\n"
    "154 (fail loud otherwise). Then enable the classifier and sweep\n"
    "`blend_lambda in {0.0, 0.3, 0.5, 0.7}` x `prob_threshold in {0.0, 0.5, 0.6}`\n"
    "(lambda=0, prob_thr=0 is the in-pipeline no-op sanity -- must ALSO equal 154).\n"
    "Each enabled config applies model_S02 to S01 associations and model_S01 to S02\n"
    "(leak-free). KEY signal: id_switches moving off 154."
))
cells.append(code(
    "def ec_block(enabled, mode='blend', blend_lambda=0.0, prob_threshold=0.0):\n"
    "    return {'enabled': enabled, 'mode': mode, 'blend_lambda': blend_lambda, 'prob_threshold': prob_threshold}\n"
    "\n"
    "\n"
    "results = []\n"
    "halt_reason = None\n"
    "\n"
    "# --- (1) DRIFT GATE: classifier fully OFF ---\n"
    "drift_cfg = {'config_id': 'D0_off', 'notes': 'drift gate: edge_classifier disabled = clean 14e B1',\n"
    "             'edge_classifier': ec_block(False)}\n"
    "drift_row = run_config(drift_cfg)\n"
    "results.append(drift_row)\n"
    "(OUT_DIR / '14o_partial_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')\n"
    "\n"
    "d_idf1 = float(drift_row['mtmc_idf1'])\n"
    "d_idsw = drift_row.get('id_switches')\n"
    "drift_ok = abs(d_idf1 - K0_REPRO_TARGET) <= K0_REPRO_TOL and d_idsw == K0_ID_SWITCH_TARGET\n"
    "if not drift_ok:\n"
    "    halt_reason = (f'DRIFT GATE FAILED: got idf1={d_idf1:.5f}, id_switches={d_idsw}; '\n"
    "                   f'expected {K0_REPRO_TARGET:.5f} +/- {K0_REPRO_TOL} and id_switches={K0_ID_SWITCH_TARGET}')\n"
    "    print(halt_reason)\n"
    "    raise RuntimeError(halt_reason)\n"
    "print(f'DRIFT GATE PASSED: idf1={d_idf1:.5f}, id_switches={d_idsw}')\n"
    "\n"
    "# --- (2) lambda=0 no-op sanity (classifier ON, but provable in-pipeline no-op) ---\n"
    "noop_cfg = {'config_id': 'N0_lambda0', 'notes': 'classifier ON but blend_lambda=0 prob_thr=0 -> in-pipeline no-op',\n"
    "            'edge_classifier': ec_block(True, 'blend', 0.0, 0.0)}\n"
    "noop_row = run_config(noop_cfg)\n"
    "results.append(noop_row)\n"
    "(OUT_DIR / '14o_partial_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')\n"
    "noop_ok = abs(float(noop_row['mtmc_idf1']) - K0_REPRO_TARGET) <= K0_REPRO_TOL and noop_row.get('id_switches') == K0_ID_SWITCH_TARGET\n"
    "if not noop_ok:\n"
    "    raise RuntimeError(f\"lambda=0 NO-OP DRIFT: idf1={noop_row['mtmc_idf1']} id_switches={noop_row.get('id_switches')} \"\n"
    "                       f'(expected {K0_REPRO_TARGET} / {K0_ID_SWITCH_TARGET}). The hook is not a clean no-op.')\n"
    "print(f\"NO-OP SANITY PASSED: lambda=0 reproduces {noop_row['mtmc_idf1']:.5f} / id_switches {noop_row.get('id_switches')}\")\n"
    "\n"
    "# --- (3) LEAK-FREE SWEEP ---\n"
    "BLEND_LAMBDAS = [0.3, 0.5, 0.7]\n"
    "PROB_THRESHOLDS = [0.0, 0.5, 0.6]\n"
    "for bl in BLEND_LAMBDAS:\n"
    "    for pt in PROB_THRESHOLDS:\n"
    "        cid = f'E_l{int(bl*100):03d}_p{int(pt*100):03d}'\n"
    "        cfg_row = {'config_id': cid, 'notes': f'leak-free blend lambda={bl} prob_thr={pt}',\n"
    "                   'edge_classifier': ec_block(True, 'blend', bl, pt)}\n"
    "        row = run_config(cfg_row)\n"
    "        results.append(row)\n"
    "        (OUT_DIR / '14o_partial_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')\n"
    "\n"
    "print(f'\\nCompleted {len(results)} configs.')"
))

# ---------------------------------------------------------------------------
cells.append(md("## 10. Verdict table + summary JSON"))
cells.append(code(
    "BASE_IDF1 = K0_REPRO_TARGET\n"
    "BASE_IDSW = K0_ID_SWITCH_TARGET\n"
    "\n"
    "sweep_rows = [r for r in results if r['config_id'].startswith('E_')]\n"
    "best = max(sweep_rows, key=lambda r: r['mtmc_idf1'] if r['mtmc_idf1'] is not None else -1.0) if sweep_rows else None\n"
    "any_idsw_moved = any(r.get('id_switches') != BASE_IDSW for r in sweep_rows)\n"
    "best_idf1 = float(best['mtmc_idf1']) if best else -1.0\n"
    "\n"
    "if best_idf1 >= WIN_THRESHOLD:\n"
    "    verdict = 'WIN'\n"
    "elif best_idf1 >= MARGINAL_MIN:\n"
    "    verdict = 'MARGINAL'\n"
    "elif any_idsw_moved:\n"
    "    verdict = 'NO-GO (id_switches moved but IDF1 below MARGINAL band)'\n"
    "else:\n"
    "    verdict = 'NO-GO (tie at 154 -- learned gate re-learned the threshold)'\n"
    "\n"
    "print('=' * 92)\n"
    "print('14o EDGE-CLASSIFIER LEAK-FREE MTMC EVAL -- VERDICT TABLE')\n"
    "print('=' * 92)\n"
    "print(f\"{'config':<16}{'enabled':<9}{'lambda':<8}{'prob_thr':<10}{'MTMC_IDF1':<12}{'id_sw':<8}{'d_IDF1':<10}{'d_idsw':<8}\")\n"
    "print('-' * 92)\n"
    "for r in results:\n"
    "    idf1 = r['mtmc_idf1'] if r['mtmc_idf1'] is not None else float('nan')\n"
    "    idsw = r.get('id_switches')\n"
    "    d_idf1 = (idf1 - BASE_IDF1) if r['mtmc_idf1'] is not None else float('nan')\n"
    "    d_idsw = (idsw - BASE_IDSW) if isinstance(idsw, int) else None\n"
    "    print(f\"{r['config_id']:<16}{str(r['enabled']):<9}{r['blend_lambda']:<8}{r['prob_threshold']:<10}\"\n"
    "          f\"{idf1:<12.5f}{str(idsw):<8}{d_idf1:<+10.5f}{str(d_idsw):<8}\")\n"
    "print('-' * 92)\n"
    "print(f'Base (drift) MTMC IDF1 = {BASE_IDF1:.5f}  id_switches = {BASE_IDSW}')\n"
    "if best is not None:\n"
    "    print(f\"BEST sweep config = {best['config_id']}  MTMC IDF1 = {best_idf1:.5f}  \"\n"
    "          f\"id_switches = {best.get('id_switches')}  (delta IDF1 = {best_idf1 - BASE_IDF1:+.5f})\")\n"
    "print(f'id_switches EVER moved off {BASE_IDSW}: {any_idsw_moved}')\n"
    "print(f'Pre-registered bands: WIN >= {WIN_THRESHOLD}, MARGINAL >= {MARGINAL_MIN}')\n"
    "print(f'VERDICT: {verdict}')\n"
    "print('=' * 92)\n"
    "\n"
    "summary = {\n"
    "    'kernel': '14o_edge_classifier_eval',\n"
    "    'base_stack': '14e B1 (primary CLIP + DINOv2 tertiary, quaternary OFF)',\n"
    "    'fusion_weights': list(FUSION_WEIGHTS),\n"
    "    'leak_free_protocol': {\n"
    "        'description': 'two scene-disjoint LightGBM fold models; S01 associations scored by model_S02, '\n"
    "                       'S02 associations scored by model_S01 (never train on the scored scene).',\n"
    "        'S01_pairs_scored_by': 'model_S02', 'S02_pairs_scored_by': 'model_S01',\n"
    "        'train_S01_n': int((scene_all == 'S01').sum()), 'train_S02_n': int((scene_all == 'S02').sum()),\n"
    "        'train_S01_pos': int(y_all[scene_all == 'S01'].sum()), 'train_S02_pos': int(y_all[scene_all == 'S02'].sum()),\n"
    "    },\n"
    "    'drift_gate': {'target_idf1': K0_REPRO_TARGET, 'target_id_switches': K0_ID_SWITCH_TARGET,\n"
    "                   'observed_idf1': float(drift_row['mtmc_idf1']), 'observed_id_switches': drift_row.get('id_switches'),\n"
    "                   'passed': bool(drift_ok)},\n"
    "    'bands': {'win': WIN_THRESHOLD, 'marginal': MARGINAL_MIN},\n"
    "    'feature_names': FEATURE_NAMES,\n"
    "    'best_sweep': best,\n"
    "    'id_switches_moved': bool(any_idsw_moved),\n"
    "    'verdict': verdict,\n"
    "    'results': results,\n"
    "}\n"
    "summary_path = OUT_DIR / '14o_edge_classifier_summary.json'\n"
    "summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')\n"
    "print(f'Wrote {summary_path}')"
))

# ---------------------------------------------------------------------------
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with OUT.open("w", encoding="utf-8") as fh:
    json.dump(notebook, fh, ensure_ascii=True, indent=1)

# Verify on-disk round-trip (CLAUDE.md rule 3).
with OUT.open(encoding="utf-8") as fh:
    reloaded = json.load(fh)
assert len(reloaded["cells"]) == len(cells)
for ci, cell in enumerate(reloaded["cells"]):
    for li, line in enumerate(cell["source"]):
        if li < len(cell["source"]) - 1:
            assert line.endswith("\n"), f"cell {ci} line {li} missing trailing newline"
print(f"Wrote + verified {OUT} ({len(cells)} cells)")
