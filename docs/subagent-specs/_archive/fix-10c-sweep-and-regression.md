# Fix: 10c Sweep Bug + 73.6% Regression Diagnosis

## Part 1 — Root Cause: Sweep Bug

### Notebook & Cell
- **File**: `notebooks/kaggle/10c_stages45/mtmc-10c-stages-4-5-association-eval.ipynb`
- **Cell index**: 14 (fusion weight sweep loop)

### What went wrong
Cell 14 constructs a labeled `run_name` for each sweep config:
```python
for label, fw_secondary, fw_tertiary in fusion_weight_pairs:
    run_name = f"{RUN_NAME}-{label}"   # src line index 37
    cmd = [
        ...
        "--override", f"project.run_name={run_name}",   # <-- bug trigger
        ...
    ]
```

`run_pipeline.py` uses `run_name` to derive ALL stage input paths:
```python
output_base = Path(cfg.project.output_dir) / run_name
# Stage 4 loads:
#   output_base / "stage1"  -- tracklets
#   output_base / "stage2"  -- embeddings
#   output_base / "stage3"  -- FAISS index + metadata.db
```

But `DATA_OUT / f"{RUN_NAME}-{label}"` was never created. Only `DATA_OUT / RUN_NAME` exists (from 10a→10b). Stage 4 finds nothing, produces empty result, exits in ~1.1s. All 11 sweep IDF1s → `None`.

### Why other sweeps (Cells 16, 20) didn't have this bug
Cells 16 (AFLink addon) and 20 (solver comparison) define and call `_ensure_upstream_links(run_name)` which creates symlinks:
```python
def _ensure_upstream_links(run_name):
    run_dir = DATA_OUT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    for stage_sub in ("stage1", "stage2", "stage3"):
        src = DATA_OUT / RUN_NAME / stage_sub
        dst = run_dir / stage_sub
        if src.exists() and not dst.exists():
            dst.symlink_to(src)
```
Cell 14 was missing this entirely.

---

## Part 2 — Exact Fix Applied

In Cell 14, after `run_name = f"{RUN_NAME}-{label}"` (src index 37), inserted 8 lines:
```python
# Symlink stage1/stage2/stage3 from original RUN_NAME so stage4 can find inputs
_fw_run_dir = DATA_OUT / run_name
_fw_run_dir.mkdir(parents=True, exist_ok=True)
for _fw_stage in ("stage1", "stage2", "stage3"):
    _fw_src = DATA_OUT / RUN_NAME / _fw_stage
    _fw_dst = _fw_run_dir / _fw_stage
    if _fw_src.exists() and not _fw_dst.exists():
        _fw_dst.symlink_to(_fw_src)
```

Fixed notebook pushed as **yahiaakhalafallah/mtmc-10c-stages-4-5-association-eval v5**.

---

## Part 3 — Root Cause Hypotheses: 73.6% vs 77.36% Regression

### Key fact
The 73.6% came from Cell 11 (baseline run), which uses `project.run_name=RUN_NAME` directly — NOT affected by the sweep bug. It is a real measurement.

### Hypothesis A: Weaker checkpoint (most likely)
The 10a run on yahia's account may have used a different/weaker TransReID checkpoint epoch than gumfreddy's. If 10a used the last epoch instead of best-val epoch, or if the 09q training run didn't fully converge, features would be weaker. ~3.76pp drop is consistent with a moderately weaker model.

### Hypothesis B: Missing secondary embeddings
If `embeddings_secondary.npy` wasn't produced in 10a (because 09n wasn't available for yahia), `SECONDARY_EMBEDDINGS_PATH` would be None and `FUSION_WEIGHT` → 0.0 instead of 0.10. Loss of secondary fusion explains ~1pp. Not enough to explain full 3.76pp gap alone.

### Hypothesis C: Branch code divergence
`feature/pretrained-ensemble` branch may have different stage2 normalization vs what gumfreddy v61 used. PCA dims, power norm exponent, or FIC parameters could differ.

### Hypothesis D: Different 10a run configuration
Yahia's 10a may have used a different config (e.g., different backbone, smaller crop size) than gumfreddy's 10a that produced 77.36%.

---

## Part 4 — Is the Regression Fixed by the Sweep Fix?

**No.** The sweep fix only enables the 11 fusion-weight configs to actually run. It does NOT change the underlying feature quality. If the new checkpoint is weaker, all sweep results will hover around 73–75%, not 77%.

To diagnose the regression, check:
1. `run_metadata.json` from 10a: which checkpoint file was used?
2. Whether `embeddings_secondary.npy` and `embeddings_tertiary.npy` exist in the new checkpoint
3. Training loss curves / mAP from the 10a run logs

---

## Part 5 — Code Changes Needed

### Stage 4 (`src/stage4_association/pipeline.py`)
**None.** Stage 4 correctly reads from `output_base / "stageN"`. The bug was notebook-level only.

### Stage 2 (`src/stage2_features/pipeline.py`)
**None.** `w_secondary`/`w_tertiary` are Stage 4 overrides (secondary embedding fusion). Stage 2 produces the embeddings; Stage 4 fuses them. The fusion parameters are Stage 4 config, not Stage 2.

### Configs (`default.yaml`, `cityflowv2.yaml`)
**None.** `w_secondary`/`w_tertiary` are passed as CLI `--override` flags, not set in YAML configs. No YAML-level config changes needed.
