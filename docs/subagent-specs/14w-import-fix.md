# 14w `ImportError` on master — Fix Spec

## Symptom
Kaggle 14w verifier failed:
```
ImportError: cannot import name 'run_stage_wildtrack_mvdetr' from 'src.stage_wildtrack_mvdetr' (unknown location)
```
when `scripts/run_pipeline.py --config configs/datasets/wildtrack.yaml --stages 1,2,3,4,5` was invoked under master HEAD `e67cd24`.

## Local Repo Audit (workspace HEAD)

**`src/stage_wildtrack_mvdetr/__init__.py` already re-exports the symbol:**
```python
from .pipeline import (
    GroundPlaneDetection,
    GroundPlaneTrack,
    load_mvdetr_ground_plane_detections,
    run_stage_wildtrack_mvdetr,
    track_ground_plane_detections,
)

__all__ = [
    "GroundPlaneDetection",
    "GroundPlaneTrack",
    "load_mvdetr_ground_plane_detections",
    "run_stage_wildtrack_mvdetr",
    "track_ground_plane_detections",
]
```

**`scripts/run_pipeline.py:103`:**
```python
from src.stage_wildtrack_mvdetr import run_stage_wildtrack_mvdetr
```

**`src/stage_wildtrack_mvdetr/pipeline.py`** does define `def run_stage_wildtrack_mvdetr(...)`.

So **on the workspace HEAD, the import works**. The Coder's claim that "origin/master lacks that file" is inconsistent with what is on disk.

## Root Cause — Most Likely

One of (in order of likelihood):

1. **Stale kernel checkout.** The 14w Kaggle kernel cloned an older commit on `master` (a commit before the `__init__.py` re-exports were merged), or pinned a SHA that pre-dates PR #5. The reverted commit `7697cae` removed a runtime shim, and it is plausible the same revert (or an earlier one) also removed the legitimate `__init__.py` re-export on the remote master, even though local master still has it. **Verify with `git log --oneline origin/master -- src/stage_wildtrack_mvdetr/__init__.py` before writing any code.**
2. **`__pycache__` shadow.** A stale `src/stage_wildtrack_mvdetr/__pycache__/__init__.cpython-311.pyc` from before the re-export was added is being used by Kaggle's Python (which would still report "unknown location" because pyc-only modules without a matching .py at the same level won't resolve names). Less likely on Kaggle since the working tree is freshly cloned per run.
3. **Branch mismatch on Kaggle.** The kernel-metadata.json or a pre-run `git checkout` step in the notebook is pinning a non-master branch / SHA that genuinely lacks the export.

## Fix (research-only — do not write code)

### Step A — Confirm the remote state
Before any patch, run locally:
```powershell
git fetch origin
git log --oneline origin/master -- src/stage_wildtrack_mvdetr/__init__.py
git show origin/master:src/stage_wildtrack_mvdetr/__init__.py
```

- **If origin/master `__init__.py` already has the re-export** → the bug is on the Kaggle side (stale checkout / pinned SHA / branch mismatch). Fix: edit the 14w notebook to checkout `master` HEAD explicitly, e.g. `git fetch origin master && git checkout origin/master` before invoking `scripts/run_pipeline.py`. No source change needed.
- **If origin/master `__init__.py` is empty / missing the re-export** → push a one-line PR adding the import block above. This is the canonical fix.

### Step B — The canonical patch (only if Step A confirms remote master is missing it)

Single PR adding (or restoring) the re-export block in `src/stage_wildtrack_mvdetr/__init__.py`. No other files need to change. `pipeline.py` already defines `run_stage_wildtrack_mvdetr`; `scripts/run_pipeline.py` already imports from the package root.

### Step C — Smoke test

After the patch lands on the Kaggle-visible commit, the smoke check is:
```bash
python -c "from src.stage_wildtrack_mvdetr import run_stage_wildtrack_mvdetr; print(run_stage_wildtrack_mvdetr.__module__)"
```
Expected output:
```
src.stage_wildtrack_mvdetr.pipeline
```

## Affected Callers (Risk Audit)

`grep_search` results for `stage_wildtrack_mvdetr`:

- `scripts/run_pipeline.py:103` — `from src.stage_wildtrack_mvdetr import run_stage_wildtrack_mvdetr` (the failing path)
- `scripts/generate_12b_wildtrack_tracking_reid_notebook.py:326,352` — imports `load_mvdetr_ground_plane_detections` and other helpers via `from src.stage_wildtrack_mvdetr.pipeline import ...` (qualified path; **not affected** by missing `__init__.py` re-export)
- `notebooks/kaggle/14z_verify_mvdetr_detector/14z_verify_mvdetr_detector.ipynb:300,302` — imports `load_mvdetr_ground_plane_detections` via the `.pipeline` qualified path (**not affected**)

So the package-root import is exclusive to `scripts/run_pipeline.py`. Fix is single-file, single-PR, low-risk.

## Verdict

- Effort: ≈ 1 subagent-prompt unit (verify + one-line patch + push).
- Priority: **HIGH** (blocks all WILDTRACK pipeline runs from `scripts/run_pipeline.py` on Kaggle, including 14w).
- Do BEFORE 14z (14w is a one-line shim; 14z requires deeper diagnosis).