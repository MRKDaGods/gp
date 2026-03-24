# MTMC Tracker — Full Project Audit

**Date:** 2026-03-24
**Auditor:** Copilot (automated)
**Scope:** Cruft identification, cleanup planning, state-of-the-art reconciliation

---

## Table of Contents

1. [Dead Code Triage](#1-dead-code-triage)
2. [Sweep Script Audit](#2-sweep-script-audit)
3. [Notebook Lineage](#3-notebook-lineage)
4. [Duplicate gp/ Structure](#4-duplicate-gp-structure)
5. [Data Folder Cleanup](#5-data-folder-cleanup)
6. [Loose Files at Root](#6-loose-files-at-root)
7. [Current State of the Art](#7-current-state-of-the-art)
8. [Docs Freshness](#8-docs-freshness)
9. [Phase 2: Cleanup Plan](#phase-2-cleanup-plan)

---

## 1. Dead Code Triage

### Root-Level Temp Scripts (11 files)

All root `_*.py` files are one-shot notebook editors or version bumpers. None are referenced by any production code. **None are covered by .gitignore** — they're tracked in git.

| File | Purpose | Classification |
|------|---------|---------------|
| `_bump_version.py` | Bumps 10c kernel-metadata.json version string | DELETE — superseded by manual edits |
| `_check_cell9.py` | Prints cell 9 of 10c notebook for inspection | DELETE — one-shot debug tool |
| `_finalize_v71.py` | Disables scan in 10c cell 15, verifies v71 config | DELETE — v71 is long past |
| `_list_cells.py` | Lists cells in 10c notebook with first-line preview | DELETE — generic; `_list_cells.py` in scripts/ is same |
| `_prep_v67.py` | Replaces 10c scan cell with v67 algorithm sweep | DELETE — v67 is long past |
| `_prep_v68.py` | Updates 10c for v68 stage 5 filter sweep | DELETE — v68 is long past |
| `_prep_v69.py` | Updates 10c for v69 fine-scan min_traj_frames | DELETE — v69 is long past |
| `_prep_v70.py` | Updates 10c for v70 FIC+sim_thresh sweep | DELETE — v70 is long past |
| `_prep_v71.py` | Updates 10c for v71 combined wins + frames search | DELETE — v71 is long past |
| `_tmp_edit_10a.py` | Adds 384px model copy from 09b output in 10a cell 7 | DELETE — change already committed |
| `_tmp_edit.py` | Reverts v83 max_iou_distance in 10a (back to v80 best) | DELETE — revert already committed |

**Verdict: DELETE ALL 11.** All are one-shot scripts whose effects are already committed. They have no ongoing value.

### scripts/ Underscore-Prefixed Files (49 files)

Categorized by function:

#### Version Prep Scripts (edit notebooks for specific experiment versions) — DELETE ALL

| File | Version | Purpose |
|------|---------|---------|
| `_prep_v75_consolidated.py` | v75 | Consolidates all optimal params + scan untested features |
| `_prep_v76_quality.py` | v76 | quality_temperature + laplacian_min_var changes in 10a |
| `_prep_v77_tracker.py` | v77 | max_gap=50, merge_time=40 tracker tuning |
| `_prep_v78_tracker2.py` | v78 | More aggressive tracker params |
| `_prep_v79_tracker3.py` | v79 | Middle ground tracker between v77/v78 |
| `_prep_10a_v13.py` | v13/10a | Downloads 09b output, uploads weights, updates 10a |
| `_edit_10a_pca.py` | PCA test | Changes PCA 384→512 in 10a |
| `_edit_10c_v52.py` | v52 | Enables score-level fusion in 10c |
| `_edit_10c_v54.py` | v54 | Enables camera bias, zone model, hierarchical in 10c |
| `_edit_scan_cell.py` | v72 | Edits 10c scan cell for intra-merge + QE alpha sweep |
| `_edit_v73_scan.py` | v73 | 10c scan cell for appearance_weight sweep |
| `_edit_v73_scan_v2.py` | v73 | Updated version of above |
| `_edit_v74_scan.py` | v74 | 10c scan cell for CSLS/cluster_verify/temporal_split |

#### Finalize Scripts (lock in best config from a sweep) — DELETE ALL

| File | Version | Purpose |
|------|---------|---------|
| `_finalize_v72.py` | v72 | Updates intra-merge wins + disable scan for PCA 512D |
| `_finalize_v73.py` | v73 | Updates app_w=0.70 win + disable scan |

#### Fix Scripts (patch notebooks for specific issues) — DELETE ALL

| File | Purpose |
|------|---------|
| `_fix_09c_v3.py` | Unfreezes teacher backbone in 09c with AMP |
| `_fix_10c_notebook.py` | Adds bridge_prune_margin=0.0 + camera_bias=false overrides |
| `_fix_comment.py` | Fixes corrupted comment in 10c scan cell |
| `_fix_eval.py` | Fixes evaluation split + R1 bug in 09b/09c |
| `_fix_notebook_paths.py` | Migrates hardcoded mrkdagods paths for account migration |
| `_fix_v48_notebooks.py` | Removes 09c kernel source from 10a, disables FAC in 10c |
| `_fix_v49_10c.py` | Enables scan + replaces CSLS A/B with temporal_split |
| `_fix_v50_384px_camtta.py` | Adds 384px + CamTTA overrides to 10a |
| `_fix_v51_multiscale_tta.py` | Reverts 384px, adds multi-scale TTA |

#### Generator Scripts (create entire notebooks) — ARCHIVE

| File | Purpose | Notes |
|------|---------|-------|
| `_gen_09b_notebook.py` | Generates 09b_vehicle_reid_384px notebook from scratch | Historical but documents notebook structure |
| `_gen_09c_notebook.py` | Generates 09c_kd_vitl_teacher notebook from scratch | Historical, KD abandoned at 22% mAP |

#### Debug/Inspection Scripts — DELETE ALL

| File | Purpose |
|------|---------|
| `_analyze_errors.py` | Quick error pattern analysis from forensic report |
| `_check_09b_fix.py` | Checks 09b v2 notebook for is_file fix |
| `_compare_configs.py` | Compares configs with/without dataset-config |
| `_debug_09b.py` | Validates 09b notebook cells via AST parse |
| `_inspect_09b_v2.py` | Inspects 09b v2 notebook for error outputs |
| `_list_10c.py` | Lists all cells in 10c notebook |
| `_list_cells.py` | Generic: lists cells in any notebook (hardcoded 10c) |
| `_list_cells_v2.py` | Generic: lists cells with type preview (takes sys.argv) |
| `_show_cell.py` | Shows crop extraction cell from 09b source |
| `_show_nb_errors.py` | Shows cell outputs/errors from pulled notebook |
| `_show_scan.py` | Displays scan results from 10c v12 |
| `_validate_09c.py` | Validates all code cells in 09c notebook via AST |

#### Scan/Experiment Scripts (local stage4+5 sweeps) — DELETE ALL

| File | Purpose | Produced Best? |
|------|---------|----------------|
| `_scan_all_features.py` | Exhaustive Phase 2+ scan: CSLS, cluster_verify, etc. | No — all features hurt or no-op |
| `_scan_phase2.py` | Sub-cluster temporal splitting thresholds | No — zero effect |
| `_scan_supplementary.py` | FIC reg + weight configs | Partial — FIC=0.1 finding, but captured in sweep_ |
| `_scan_tier1.py` | Quick CSLS + cluster_verify test | No — CSLS catastrophic |
| `_run_baseline.py` | Reproduces 0.8297 baseline for comparison | No — just a confirmation run |

#### Pipeline/Monitor Scripts — KEEP 1, DELETE REST

| File | Purpose | Classification |
|------|---------|---------------|
| `_chain_monitor.py` | Polls 10a→10b→10c, auto-pushes next stage | DELETE — hardcoded gumfreddy slugs, stale |
| `_kaggle_pipeline.py` | Same as above, mrkdagods slugs | DELETE — superseded by `kaggle_chain.py` |
| `_wait_and_push.py` | Waits for GPU slots, pushes 09b+09c | DELETE — one-shot |
| `_test_push.py` | Minimal Kaggle push test | DELETE — one-shot |
| `_revert_pca.py` | Reverts PCA 512→384 in 10a | DELETE — change committed |
| `_patch_10c.py` | Adds --dataset-config + Phase 3 grid to 10c | DELETE — change committed |

**Summary: DELETE 47 files, ARCHIVE 2 (_gen_09b, _gen_09c to docs/archive/ or similar)**

### .gitignore Coverage Gap

Only `scripts/_tmp_*.py`, `scripts/_find_*.py`, `scripts/_print_*.py` are gitignored. The other 45+ `_*.py` files are **tracked in git**. Root `_*.py` files have **zero gitignore coverage**.

---

## 2. Sweep Script Audit

All `sweep_*.py` files are gitignored (`scripts/sweep_*.py` in .gitignore). They exist locally but won't be committed.

| Script | Purpose | Meaningful Results? |
|--------|---------|-------------------|
| `sweep_stage45.py` | Initial stage 4-5 parameter sweep | Yes — established initial best config |
| `sweep_round2.py` | CSLS, intra-cam merge, cluster verify, reranking on round 1 best | Yes — confirmed reranking hurts vehicles |
| `sweep_round3.py` – `sweep_round12.py` | Progressive parameter sweeps (10 rounds) | Diminishing returns; v47 came from this series |
| `sweep_fic_combined.py` | FIC fix with BN'd secondary embeddings | Yes — produced v47 best (IDF1=0.8297) |
| `sweep_fic_fix.py`, `sweep_fic_fix_v2.py` | FIC regularization (covariance bug fix) | Yes — critical bug fix |
| `sweep_qe_fix.py` | QE self-exclusion fix with various k values | Yes — QE k=2-3 confirmed optimal |
| `sweep_sec_bn.py` | Secondary embedding BN variants | Minor — confirmed score-level fusion at 10% |
| `sweep_sec_pca.py` | Secondary embedding PCA whitening | Minor — confirmed 384D optimal |
| `sweep_selftrain.py` | Self-training embedding refinement | No — experimental, no improvement |
| `sweep_postproc.py` – `sweep_postproc_v4.py` | Post-processing parameter sweeps | Minor — stationary filter tuning |
| `sweep_hierarchical.py` | Hierarchical clustering | No — hurt 1.0–5.1pp |

**Key finding:** `sweep_fic_combined.py` produced the v47 local best (IDF1=0.8297). The FIC covariance bug fix + concat_patch features were the breakthrough, not any single sweep.

**All sweep files are gitignored** — no action needed for git. For local cleanup, all can be deleted since results are captured in `experiment_log.md`.

---

## 3. Notebook Lineage

### Active Pipeline Chain

| Notebook | Kaggle Account | GPU | Status | Latest |
|----------|---------------|-----|--------|--------|
| **10a_stages012** | ali369 | T4 | **ACTIVE** — stages 0-2 | v84 (384px model wired) |
| **10b_stage3** | ali369 | GPU (unnecessary) | **ACTIVE** — stage 3 FAISS | Tracks 10a output |
| **10c_stages45** | ali369 | No GPU | **ACTIVE** — stages 4-5 eval | v47 (Kaggle) = 10c v44 |

### Active Training Notebooks

| Notebook | Account | Status | Notes |
|----------|---------|--------|-------|
| **09_vehicle_reid_cityflowv2** | yahiaakhalafallah | **ACTIVE** | TransReID fine-tuning on CityFlowV2; produces primary model |
| **09b_vehicle_reid_384px** | yahiaakhalafallah | **ACTIVE** | 384px fine-tune from 09 output; v84 wired into 10a |

### Historical/Abandoned Training Notebooks

| Notebook | Status | Notes |
|----------|--------|-------|
| 01_dataset_preparation | **HISTORICAL** | Dataset prep manifests; run once |
| 02_person_reid_training | **HISTORICAL** | Person ReID — pipeline is vehicle-only now |
| 03_vehicle_reid_training | **HISTORICAL** | Superseded by 09 (CityFlowV2 fine-tuning) |
| 03b_vehicle_osnet_resume | **HISTORICAL** | OSNet resume training; OSNet now secondary at 10% |
| 04_advanced_reid_training | **HISTORICAL** | TransReID training; superseded by 09 |
| 05_cityflowv2_download | **HISTORICAL** | One-shot download; has 5 tmpclaude files |
| 06_wildtrack_download | **HISTORICAL** | One-shot download; has 3 tmpclaude files |
| 07_person_reid_sota | **HISTORICAL** | Person ReID SOTA training; has output/ subdirs with results |
| 08_vehicle_reid_sota | **HISTORICAL** | Vehicle ReID SOTA on VeRi-776; superseded by 09 |
| 08b_vehicle_reid_triplet | **HISTORICAL** | Triplet loss experiment |
| 09c_kd_vitl_teacher | **ABANDONED** | Knowledge distillation — achieved only 22% mAP, abandoned |
| 10_mtmc_pipeline | **SUPERSEDED** | Old monolithic pipeline; replaced by 10a/10b/10c split |

### Reference Notebooks

| File | Status |
|------|--------|
| `reference/gp-stage-1b-detection-tracking-deepocsort.ipynb` | HISTORICAL — DeepOCSORT experiment |
| `reference/gp-stage-2.ipynb` | HISTORICAL — stage 2 reference |
| `reference/gp-stage-4.ipynb` | HISTORICAL — stage 4 reference |
| `reference/gp-stage-5-evaluation.ipynb` | HISTORICAL — stage 5 reference |

### tmpclaude Files

8 `tmpclaude-*-cwd` files scattered in notebooks/kaggle/05_cityflowv2_download/ (5 files) and 06_wildtrack_download/ (3 files). These are Claude coding agent temp files. **Safe to delete.**

---

## 4. Duplicate gp/ Structure

**The `gp/` subfolder is a stale Kaggle artifact.** When a Kaggle kernel clones the repo, it creates this mirror structure. The `.gitignore` correctly covers it:

```
# Accidentally-downloaded artifacts from kaggle kernels output
checkpoint.tar.gz
gp/
```

Contents: `models/`, `mtmc_tracker.egg-info/`, `notebooks/`, `scripts/`, `src/`, `tests/`, `pyproject.toml`, `setup.py`, `yolo26m.pt`.

**Verdict: SAFE TO DELETE locally.** It's already gitignored, so it doesn't pollute the repo. Deleting it locally recovers disk space (could be significant if models/ contains weights).

---

## 5. Data Folder Cleanup

The entire `data/` directory is gitignored. Classification:

### Required for Pipeline Runs

| Directory | Purpose | Keep? |
|-----------|---------|-------|
| `data/raw/` | CityFlowV2 videos + GT annotations | **KEEP** — pipeline input |
| `data/outputs/` | Pipeline run outputs (stage0-5 artifacts) | **KEEP** — contains current best run data |
| `data/processed/` | CityFlowV2 ReID crops, preprocessed data | **KEEP** — ReID training/eval data |
| `data/gt_upload/` | GT annotations for Kaggle upload | **KEEP** — needed for Kaggle GT dataset |

### Debug/Historical Artifacts — Safe to Delete

| Item | Purpose | Keep? |
|------|---------|-------|
| `data/09b_v2/` | Downloaded 09b v2 kernel output for debugging | DELETE |
| `data/09b_v2_nb/` | Another 09b notebook download | DELETE |
| `data/10a_debug/` | Debug artifacts from 10a runs (contains its own gp/ clone!) | DELETE |
| `data/10a_log.txt` | Kaggle 10a run log | DELETE (historical) |
| `data/10a_v2_log.txt` | Kaggle 10a v2 run log | DELETE |
| `data/10a_v3_clean.txt` | Cleaned 10a v3 log | DELETE |
| `data/10a_v3_full.txt` | Full 10a v3 log | DELETE |
| `data/10a_v3_log.txt` | 10a v3 log | DELETE |
| `data/10a_v3_poll.txt` | 10a v3 polling log | DELETE |
| `data/10a_v4_log.txt` | 10a v4 log | DELETE |
| `data/10b_log.txt` | 10b run log | DELETE |
| `data/10c_log.txt` | 10c run log | DELETE |
| `data/10c_raw.txt` | Raw 10c output | DELETE |
| `data/10c_v3_log.txt` | 10c v3 log | DELETE |

**All of data/ is gitignored** — deletions are purely local disk cleanup. The 11 log files and 3 debug dirs can be removed with no risk.

---

## 6. Loose Files at Root

| File | In .gitignore? | Needed Locally? | Action |
|------|----------------|-----------------|--------|
| `yolo26m.pt` | YES (`*.pt` + explicit `yolo26m.pt`) | Yes — YOLO detection model | Keep locally |
| `run_metadata.json` | YES (explicit entry) | No — auto-generated run metadata | Delete locally |
| `ReID_Experiments_Complete.xlsx` | **NO** | No — historical spreadsheet | **Add `*.xlsx` to .gitignore**, delete |
| `PERSON-REID.zip` | YES (`*.zip`) | No — training data archive | Delete locally |
| `VEHICLE-REID.zip` | YES (`*.zip`) | No — training data archive | Delete locally |
| `mtmc-10a-stages-0-2-tracking-reid-features.log` | YES (`*.log`) | No — stale Kaggle log | Delete locally |

### .gitignore Gap Found

`*.xlsx` is NOT in .gitignore. The `ReID_Experiments_Complete.xlsx` file could be committed accidentally.

**Action needed:** Add `*.xlsx` to .gitignore.

---

## 7. Current State of the Art — Score Reconciliation

### The Confusion

The `.github/copilot-instructions.md` claims:
- **Best local**: IDF1=0.8297, MOTA≈0.844 (v47, FIC fix + concat_patch)
- **Best Kaggle**: IDF1=0.813 (10c v4, v46 params)

**Both claims are stale/unverifiable.**

### Actual Best (from experiment_log.md)

| Metric | Value | Source | Notes |
|--------|-------|--------|-------|
| **Best Kaggle IDF1** | **0.784** (78.4%) | v80, 10c v44 (ali369 account) | min_hits=2 tweak on tracker |
| **IDF1=0.8297 claim** | **ORPHANED** | No matching experiment_log entry | Pre-dates current logging; may have been local eval with different GT or scoring |
| **IDF1=0.813 claim** | **STALE** | Was 10c v4 era; superseded by many versions | Predates v67+ sweep campaign |

### default.yaml Is STALE

The current `configs/default.yaml` does NOT reflect the best-known config. Key mismatches:

| Parameter | default.yaml | Best Known (v80) | Impact |
|-----------|-------------|-------------------|--------|
| `stage1.tracker.min_hits` | 3 | **2** | +0.2pp (v80 finding) |
| `stage2.pca.n_components` | 280 | **384** | +0.78pp (PCA experiment) |
| `stage4.association.fic.regularisation` | 0.3 | **0.1** | +0.08pp (v71) |
| `stage4.association.weights.appearance` | 0.55 | **0.70** | +0.76pp (v73) |
| `stage5.min_trajectory_frames` | ? | **40** | Critical filter param |
| `stage5.cross_id_nms_iou` | ? | **0.40** | Critical filter param |

**This is a significant finding**: the config file doesn’t match the pipeline that produced the best scores. The best configs are only encoded in the 10a/10c notebooks as overrides.

### What Needs Updating

1. `configs/default.yaml` → update to match v80 best params
2. `.github/copilot-instructions.md` → update best Kaggle to 0.784 (v80, 10c v44), note 0.8297 local is unverifiable
3. `docs/SOTA_ANALYSIS.md` → rewrite, currently says "IDF1 ~30" for our pipeline
4. `docs/BREAKTHROUGH_PLAN.md` → mark Tier 0/1 as tested, update baseline to 78.4%

---

## 8. Docs Freshness

| Document | Last Meaningful Update | Current? | Issues |
|----------|----------------------|----------|--------|
| [architecture.md](docs/architecture.md) | Mid-project | **PARTIALLY STALE** | Still mentions OSNet/ResNet50-IBN as ReID fallback; diagram shows k-reciprocal re-ranking (disabled since v25); says "0.7 Appearance + 0.1 HSV + 0.2 ST" (actual best is 0.70 app, 0.0 HSV, 0.30 ST); mentions Louvain (actual: conflict_free_cc) |
| [BREAKTHROUGH_PLAN.md](docs/BREAKTHROUGH_PLAN.md) | ~v48 era | **STALE** | Lists Tier 0/1 items that are now tested and closed (CSLS: catastrophic, intra-merge: tested, temporal_split: no effect). Kaggle baseline cited as 0.789 (now 0.784). Still useful as roadmap template. |
| [SOTA_ANALYSIS.md](docs/SOTA_ANALYSIS.md) | Early project | **VERY STALE** | Says "Our pipeline IDF1 ~30, MOTA ~-60" (pre-optimization). Still lists ResNet50-IBN as primary ReID. Gap analysis doesn't reflect current TransReID pipeline. |
| [experiment_log.md](docs/experiment_log.md) | v84 (latest) | **CURRENT** | Comprehensive and up-to-date. The authoritative reference for all experiments. |
| [data_contracts.md](docs/data_contracts.md) | Mid-project | **PARTIALLY STALE** | Embedding dim listed as 512 (OSNet) or 2048 (ResNet50); actual is 768 (TransReID) → PCA 384. |
| [dataset_guide.md](docs/dataset_guide.md) | Mid-project | **CURRENT** | Correctly describes all datasets. CityFlowV2 ReID section is accurate. |
| [setup_guide.md](docs/setup_guide.md) | Early project | **PARTIALLY STALE** | References `torchreid` in verify step; actual primary is `timm` (TransReID). Model download paths are mostly correct. |
| [kaggle_training_guide.md](docs/kaggle_training_guide.md) | Early project | **STALE** | Describes notebooks 01-04 only. Doesn't mention 07-09c series. Training configs reference OSNet/ResNet50-IBN, not TransReID. |
| [research_papers_and_metrics.md](docs/research_papers_and_metrics.md) | Mid-project | **CURRENT** | Metric definitions are timeless. Good reference document. |
| [team_workload.md](docs/team_workload.md) | Early project | **STALE** | Describes initial task allocation. Doesn't reflect actual 4-person team execution. References OSNet/ResNet50-IBN. |

---

## Phase 2: Cleanup Plan

### Priority 0: Fix Stale Config (HIGH IMPACT)

**`configs/default.yaml` does not match the best achieved config.** The best params from v80 (min_hits=2, fic_reg=0.1, appearance_weight=0.70, n_components=384, min_traj_frames=40, cross_id_nms=0.40) are only encoded as notebook overrides, not in the base config.

**Action:** Update default.yaml to reflect v80 best params.
**Risk:** LOW — pipeline reads config + overrides, but having the correct baseline prevents regression.

### Priority 1: Safe Deletions (Zero Risk)

#### 1a. Root temp scripts (11 files) — DELETE

```
_bump_version.py  _check_cell9.py  _finalize_v71.py  _list_cells.py
_prep_v67.py  _prep_v68.py  _prep_v69.py  _prep_v70.py  _prep_v71.py
_tmp_edit_10a.py  _tmp_edit.py
```

**Risk:** None. All one-shot scripts whose effects are committed. Not referenced anywhere.

#### 1b. scripts/ underscore files (47 files) — DELETE

All `scripts/_*.py` files listed in Section 1. Every one is a one-shot editor, debugger, or experiment runner.

**Risk:** None. All changes they made are already committed. Results are captured in experiment_log.md.

**Exception:** Consider ARCHIVING `_gen_09b_notebook.py` and `_gen_09c_notebook.py` if you want to preserve notebook generation templates.

#### 1c. tmpclaude files (8 files) — DELETE

```
notebooks/kaggle/05_cityflowv2_download/tmpclaude-{553f,9af3,d258,da5f,e988}-cwd
notebooks/kaggle/06_wildtrack_download/tmpclaude-{5a01,71d7,c0a0}-cwd
```

**Risk:** None. Claude agent temp files with no content.

#### 1d. Root loose files — DELETE locally

```
PERSON-REID.zip  VEHICLE-REID.zip  ReID_Experiments_Complete.xlsx
mtmc-10a-stages-0-2-tracking-reid-features.log  run_metadata.json
```

**Risk:** None. All gitignored except .xlsx (add to gitignore first).

### Priority 2: .gitignore Fixes (Zero Risk)

Add to `.gitignore`:

```gitignore
# Spreadsheets
*.xlsx

# Root-level temp scripts (one-shot notebook editors)
_*.py

# All underscore-prefixed scripts (one-shot tools)
scripts/_*.py
```

This supersedes the existing narrow rules (`scripts/_tmp_*.py`, `scripts/_find_*.py`, `scripts/_print_*.py`) with a blanket `scripts/_*.py` rule.

### Priority 3: Local-Only Cleanup (Zero Risk — data/ is gitignored)

#### 3a. data/ debug artifacts

```
data/09b_v2/
data/09b_v2_nb/
data/10a_debug/
data/10a_log.txt  data/10a_v2_log.txt  data/10a_v3_*.txt  data/10a_v4_log.txt
data/10b_log.txt  data/10c_log.txt  data/10c_raw.txt  data/10c_v3_log.txt
```

**Risk:** None. Historical debug artifacts. Pipeline runs use data/outputs/.

#### 3b. gp/ subfolder

```
gp/  (entire directory)
```

**Risk:** None. Stale Kaggle clone artifact. Already gitignored.

### Priority 4: Doc Updates (Low Risk)

| Document | Action |
|----------|--------|
| **copilot-instructions.md** | Update "Best Kaggle" from "IDF1=0.813" to "IDF1=0.784 (v80, 10c v44)". Clarify local vs Kaggle distinction. |
| **architecture.md** | Update ReID model from OSNet/ResNet50-IBN to TransReID ViT-B/16 CLIP. Update weight formula. Replace Louvain with conflict_free_cc. Note re-ranking disabled. |
| **SOTA_ANALYSIS.md** | Complete rewrite needed — says "IDF1 ~30" for our pipeline. Should reflect 78-83% range. |
| **BREAKTHROUGH_PLAN.md** | Mark Tier 0/1 items as TESTED. Update Kaggle baseline to 78.4%. Keep as-is for Tier 2-5 roadmap. |
| **data_contracts.md** | Update embedding dimensions: 768D TransReID → PCA 384D. |
| **setup_guide.md** | Update verify commands to reference timm/TransReID. |
| **kaggle_training_guide.md** | Add notebooks 07-09c. Update training configs. |
| **team_workload.md** | Retire or update with actual execution history. |

### Priority 5: Proposed Cleaner Project Structure

After cleanup:

```
configs/          # YAML configuration (unchanged)
data/             # Pipeline data (gitignored, keep outputs/ raw/ processed/ gt_upload/)
docs/             # Documentation (update stale docs)
models/           # Model weights (gitignored)
notebooks/
  kaggle/
    09_vehicle_reid_cityflowv2/    # ACTIVE training
    09b_vehicle_reid_384px/        # ACTIVE training
    10a_stages012/                 # ACTIVE pipeline
    10b_stage3/                    # ACTIVE pipeline
    10c_stages45/                  # ACTIVE pipeline
    archive/                       # Move 01-08b, 09c, 10 here
  reference/                       # Keep as-is
scripts/          # Only legitimate scripts (29 files), no _*.py
src/              # Production code (unchanged)
tests/            # Test suite (unchanged)
```

### Risk Assessment Summary

| Category | Files | Risk | Reversibility |
|----------|-------|------|--------------|
| Root _*.py deletion | 11 | **ZERO** | git restore |
| scripts/_*.py deletion | 47 | **ZERO** | git restore |
| tmpclaude deletion | 8 | **ZERO** | None needed |
| .gitignore additions | 3 rules | **ZERO** | git revert |
| data/ local cleanup | ~14 items | **ZERO** | Not in git; re-download from Kaggle if needed |
| gp/ local deletion | 1 dir | **ZERO** | Not in git; re-download from Kaggle |
| Doc updates | 7 files | **LOW** | git restore |
| Notebook archive move | 12 dirs | **LOW** | git mv back |
| Root loose file deletion | 5 files | **ZERO** | Not in git; re-download |

### Recommended Execution Order

1. **Add .gitignore rules** (Priority 2) — prevents future drift
2. **Delete root _*.py** (Priority 1a) — 11 files, immediate cleanup
3. **Delete scripts/_*.py** (Priority 1b) — 47 files, biggest win
4. **Delete tmpclaude files** (Priority 1c) — 8 files
5. **Update copilot-instructions.md** (Priority 4) — fixes stale scores
6. **Local cleanup** (Priority 3) — data/ artifacts, gp/ subfolder
7. **Doc updates** (Priority 4) — architecture.md, SOTA_ANALYSIS.md, etc.
8. **Notebook archive** (Priority 5) — optional structural improvement