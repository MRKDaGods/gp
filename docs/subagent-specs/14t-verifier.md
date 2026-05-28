# 14aa — Verify 14t VeRi-776 Fusion (CLIP-SENet × TransReID)

**Status**: PROPOSED (planner spec)  
**Author**: MTMC Planner, 2026-05-16  
**Type**: Kaggle GPU verifier kernel, single notebook, no training  
**Estimated wall time**: ~50-70 min on T4; the 14t source kernel ran in ~49.5 min on T4  
**Account**: `yahiaakhalafallah` by default; this matches the 14t, 14u, and 14y kernel owners, while `mrkdagods` has one mirror of the weights dataset

---

## 1. Goal

Slot a 14t-fusion verifier into the verification suite beside the green 14v, 14w, 14x, 14y, and 14z runs.

The verifier must reproduce the 14t WIN headline on VeRi-776 single-camera ReID:

- `mAP = 0.9330 ± 0.005`
- `R1 = 0.9845 ± 0.005`

The acceptance criterion is concrete: a new on-master CLI, `scripts/eval/eval_14t_fusion_veri776.py`, must produce a JSON whose `score_fusion.best.mAP` and `score_fusion.best.R1` fall inside those bands at the 14t WIN parameters:

- `w_clipsenet = 0.7`
- `w_transreid = 0.3`
- `transreid_stream = global` / 768-d stream
- `AQE k = 3`
- rerank `k1 = 80`, `k2 = 15`, `lambda = 0.2`

This verifier is GPU-only. Feature extraction over the full VeRi query and gallery on CPU would be many hours and should be skipped rather than attempted.

---

## 2. Reference Experiment Values

Primary source: [docs/findings.md](../findings.md#L331-L377).

The 14t section records the source kernel, configuration, runtime, and result table:

| Stream | mAP | R1 | Source |
|---|---:|---:|---|
| TransReID 09v v17 base | 0.8997 | 0.9833 | [docs/findings.md](../findings.md#L341-L349) |
| CLIP-SENet v6 post-rerank | 0.9154 | 0.9732 | [docs/findings.md](../findings.md#L341-L349) |
| **14t score-fusion best** | **0.9330** | **0.9845** | [docs/findings.md](../findings.md#L341-L349) |
| 14t concat best | 0.9319 | 0.9827 | [docs/findings.md](../findings.md#L341-L349) |

The configuration recorded in findings is the verifier's target: score-level fusion at `w_clipsenet=0.7`, `w_transreid=0.3`, the `transreid_768` global-token stream, AQE k=3, and rerank `(k1=80, k2=15, lambda=0.2)`; runtime was approximately 49.5 min on T4.

The CityFlow port failure is deliberately not part of the acceptance gate. It is a strategic warning only: [docs/findings.md](../findings.md#L367-L377) records that 14u reached only 0.77995 MTMC IDF1, +0.00059 over the 14e B1 anchor and under the noise band.

---

## 3. Kernel Structure

Notebook path:

```text
notebooks/kaggle/14aa_verify_14t_veri_fusion/14aa_verify_14t_veri_fusion.ipynb
```

Kernel metadata path:

```text
notebooks/kaggle/14aa_verify_14t_veri_fusion/kernel-metadata.json
```

`kernel-metadata.json` must use this shape:

```json
{
  "id": "yahiaakhalafallah/14aa-verify-14t-veri-fusion",
  "title": "14aa Verify 14t VeRi Fusion",
  "code_file": "14aa_verify_14t_veri_fusion.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "dataset_sources": [
    "abhyudaya12/veri-vehicle-re-identification-dataset",
    "mrkdagods/mtmc-weights",
    "yahiaakhalafallah/mtmc-weights",
    "gumfreddy/mtmc-weights"
  ],
  "kernel_sources": [
    "yahiaakhalafallah/13-clip-senet-train"
  ],
  "model_sources": []
}
```

Mirror the 14y verifier structure and error-handling pattern from `notebooks/kaggle/14y_verify_veri_reid_checkpoints/14y_verify_veri_reid_checkpoints.ipynb`. The 14y metadata pattern is in `notebooks/kaggle/14y_verify_veri_reid_checkpoints/kernel-metadata.json`.

---

## 4. Cell Skeleton

The notebook must contain exactly these cells, in this order. The Coder may add small helper definitions inside the named cells, but should not add extra conceptual cells unless needed for a Kaggle-only import workaround.

### Cell 1 — Markdown Header Cell

Purpose:

- Name the verifier.
- State the target values.
- Link the source result in [docs/findings.md](../findings.md#L331-L377).

Content outline:

```markdown
# 14aa Verify 14t VeRi Fusion

GPU verifier for the 14t CLIP-SENet v6 × TransReID 09v v17 score-fusion WIN on VeRi-776.
Target: mAP=0.9330 ± 0.005, R1=0.9845 ± 0.005.
Source: docs/findings.md §14t.
```

Lift from:

- 14y markdown header style.

### Cell 2 — Setup Cell

Purpose:

- Detect GPU availability.
- Detect compute capability.
- Apply the P100 `sm_60` PyTorch compatibility guard before imports that depend on torch.

Required logic:

- Run `subprocess.run(["nvidia-smi", "--query-gpu=gpu_name,compute_cap", "--format=csv,noheader"], ...)`.
- Parse output rows.
- If a compute capability starts with `6.`, reinstall PyTorch 2.4.1+cu124 using:

```bash
pip install --no-cache-dir --force-reinstall torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

- If no GPU is available, set a notebook-level `GPU_AVAILABLE = False` flag and let later eval rows become non-required SKIP rows.

Lift from:

- `notebooks/kaggle/14y_verify_veri_reid_checkpoints/14y_verify_veri_reid_checkpoints.ipynb`, the top cell containing `_nvsmi = subprocess.run(...)` and the 14y v5 `sm_60` guard. The same logic must be copied, not simplified.

### Cell 3 — Repo Clone + Pip Install Cell

Purpose:

- Clone master.
- Record a soft expected-SHA warning.
- Install exactly the packages needed by the eval scripts.

Required constants and commands:

```python
EXPECTED_MASTER_SHA_AT_BUILD = "<set at notebook build time>"
REPO_URL = "https://github.com/MRKDaGods/gp.git"
PROJECT = Path("/kaggle/working/gp")
```

Required behavior:

1. Clone `https://github.com/MRKDaGods/gp.git` branch `master` into `/kaggle/working/gp`.
2. Resolve actual `git rev-parse HEAD`.
3. Print both actual SHA and `EXPECTED_MASTER_SHA_AT_BUILD`.
4. If they differ, print `WARN: master moved since notebook build`, but do not raise.

Required installs:

```bash
pip install faiss-cpu motmetrics loguru omegaconf rich networkx>=3.1 click filterpy ftfy lapx scikit-learn scipy pandas opencv-python-headless tqdm
pip install timm==1.0.11 open_clip_torch==2.30.0 pretrainedmodels==0.7.4
pip install --no-deps ultralytics boxmot==11.0.3
pip install --no-deps -e .
```

Lift from:

- 14y clone/install cell, especially the cell tagged/commented `#VSC-3706f8ce`.

### Cell 4 — Dataset Mount Cell

Purpose:

- Locate VeRi-776.
- Locate the two checkpoint roles.
- Copy or symlink checkpoints into the project-local registry paths expected by the eval script.

VeRi root resolution:

- Search for `/kaggle/input/veri-vehicle-re-identification-dataset/VeRi` first.
- Accept minor slug layout variants under `/kaggle/input/**/VeRi`.
- Assert both `image_query/` and `image_test/` exist.

Checkpoint resolution:

- TransReID 09v:
  - Search in order: `mrkdagods/mtmc-weights`, `yahiaakhalafallah/mtmc-weights`, `gumfreddy/mtmc-weights`.
  - Required member: `reid/vehicle_transreid_vit_base_veri776.pth`.
  - Copy or symlink to `PROJECT / "models/reid/vehicle_transreid_vit_base_veri776.pth"`.
- CLIP-SENet v6:
  - Locate `/kaggle/input/13-clip-senet-train`.
  - Resolve `best_mAP.pth` first, then `best.pth`.
  - Copy to `PROJECT / "models/reid/clipsenet_v6_veri776_best.pth"`.

Lift from:

- 14y dataset-resolution cell and its checkpoint fall-through pattern.

### Cell 5 — Eval Subprocess Helper Cell

Purpose:

- Centralize subprocess execution.
- Capture stdout and stderr to per-label log files.
- Record PASS/FAIL/SKIP rows uniformly.

Copy these helpers from 14y verbatim:

- `EvalSubprocessError`
- `tail_text`
- `run_eval_subprocess`
- `record_metric`
- `record_exception`
- `metric_from`

This cell must preserve the 14y v3 fix pattern: failed subprocess stderr must be captured into an accessible log file and surfaced in the summary row.

### Cell 6 — Eval F: Run 14t Fusion Eval

Purpose:

- Run the new standalone fusion eval once at the 14t WIN parameters.

Required command shape:

```bash
python scripts/eval/eval_14t_fusion_veri776.py \
  --transreid-checkpoint models/reid/vehicle_transreid_vit_base_veri776.pth \
  --clipsenet-checkpoint models/reid/clipsenet_v6_veri776_best.pth \
  --veri-root <VERI_ROOT> \
  --device cuda \
  --w-clipsenet 0.7 \
  --transreid-stream global \
  --aqe-k 3 \
  --rerank-k1 80 --rerank-k2 15 --rerank-lambda 0.2 \
  --transreid-batch-size 64 \
  --clipsenet-batch-size 64 \
  --clipsenet-img-size 320 320 \
  --output-json /kaggle/working/14aa_eval_json/eval_f_14t_fusion.json
```

Required rows:

| Label | JSON path | Target | Tolerance | Required |
|---|---|---:|---:|---|
| `Eval F 14t fusion score mAP` | `score_fusion.best.mAP` | 0.9330 | 0.005 | yes |
| `Eval F 14t fusion score R1` | `score_fusion.best.R1` | 0.9845 | 0.005 | yes |
| `Eval F drift TransReID 09v concat_patch+AQE3+rerank mAP` | `drift_parents.transreid_09v_concat_patch_aqe3_rerank.mAP` | 0.8997 | 0.01 | no |
| `Eval F drift CLIP-SENet v6 AQE10+rerank mAP` | `drift_parents.clipsenet_v6_aqe10_rerank_k1_50_k2_10_lambda_0_1.mAP` | 0.9154 | 0.01 | no |

If `GPU_AVAILABLE` is false:

- Do not run the subprocess.
- Record all rows as skipped / non-required.
- Exit the notebook successfully after writing summary JSON.

### Cell 7 — Summary Cell

Purpose:

- Write `/kaggle/working/14aa_verify_results.json`.
- Raise only if required rows failed.

Required summary schema:

```json
{
  "verifier": "14aa_verify_14t_veri_fusion",
  "passed": true,
  "git_sha": "...",
  "expected_master_sha_at_build": "...",
  "gpu_available": true,
  "metrics": [],
  "eval_outputs": {
    "eval_f_14t_fusion": "/kaggle/working/14aa_eval_json/eval_f_14t_fusion.json"
  },
  "logs": {
    "eval_f_stdout": "...",
    "eval_f_stderr": "..."
  }
}
```

Lift from:

- 14y summary cell, including `passed = all(...)` over required rows and `raise AssertionError(...)` only for required-row failures.

---

## 5. Required JSON Schema for `eval_14t_fusion_veri776.py`

The verifier reads this schema from `/kaggle/working/14aa_eval_json/eval_f_14t_fusion.json`:

```json
{
  "experiment": "14t_fusion_verify",
  "wall_time_sec": 2700.0,
  "params": {
    "w_clipsenet": 0.7,
    "w_transreid": 0.3,
    "transreid_stream": "global",
    "aqe_k": 3,
    "rerank": {"k1": 80, "k2": 15, "lambda_value": 0.2}
  },
  "score_fusion": {
    "best": {"mAP": 0.933, "R1": 0.984, "R5": 0.99, "R10": 0.99},
    "all_rows": []
  },
  "drift_parents": {
    "transreid_09v_concat_patch_aqe3_rerank": {"mAP": 0.8997, "R1": 0.9815},
    "clipsenet_v6_aqe10_rerank_k1_50_k2_10_lambda_0_1": {"mAP": 0.9154, "R1": 0.9732}
  },
  "checkpoints": {
    "transreid": {"path": "...", "sha256": "..."},
    "clipsenet": {"path": "...", "sha256": "..."}
  }
}
```

The verifier must use `metric_from()` lookup paths such as:

- `metric_from(data, ("score_fusion", "best", "mAP"))`
- `metric_from(data, ("score_fusion", "best", "R1"))`
- `metric_from(data, ("drift_parents", "transreid_09v_concat_patch_aqe3_rerank", "mAP"))`
- `metric_from(data, ("drift_parents", "clipsenet_v6_aqe10_rerank_k1_50_k2_10_lambda_0_1", "mAP"))`

---

## 6. PASS/FAIL Gates

Required gates:

- `Eval F 14t fusion score mAP` must be in `[0.928, 0.938]`.
- `Eval F 14t fusion score R1` must be in `[0.9795, 0.9895]`.

Optional gates:

- Parent drift gates are warnings only.
- Use tolerance `0.01` for parent mAP drift rows.
- Do not let parent drift rows fail the kernel if the fused headline rows pass.

GPU behavior:

- If GPU is unavailable, mark all rows skipped / non-required and exit 0.
- Do not attempt a CPU fallback. The full VeRi feature extraction path on CPU is too slow for a verifier.

---

## 7. Estimated Runtime Breakdown

| Component | Estimate on T4 |
|---|---:|
| TransReID 09v feature extraction on full VeRi-776, 1,678 queries + ~11,579 gallery, with 2-view TTA | ~10-14 min |
| CLIP-SENet v6 feature extraction at 320x320, single forward | ~8-12 min |
| Score-fusion + AQE + rerank k1=80 on full query-gallery/all-pairs matrices | ~3-5 min CPU |
| Drift checks: TransReID concat+AQE+rerank and CLIP-SENet AQE10+rerank | ~10 min |
| Install, clone, checkpoint copy, JSON/log overhead | ~10-20 min |
| **Total budget** | **~50-70 min wall** |

This is comfortably under the 12h Kaggle limit.

---

## 8. Account / Quota

Push under `yahiaakhalafallah`.

Rationale:

- This account typically has 15+ h quota.
- The 14t source kernel ran under this account.
- The dataset-source whitelist should already cover `yahiaakhalafallah/13-clip-senet-train`.

Use the multi-account auth pattern from user memory:

```powershell
$env:KAGGLE_API_TOKEN = (Get-Content $HOME/.kaggle/yahiaakhalafallah_access_token -Raw).Trim()
```

Do not swap `~/.kaggle/kaggle.json` unless the local CLI unexpectedly ignores `KAGGLE_API_TOKEN`.

---

## 9. Files Coder Must Create

The verifier implementation PR must create these files:

- `scripts/eval/eval_14t_fusion_veri776.py` — new CLI; design lives in [14t-production-wiring.md](14t-production-wiring.md).
- `notebooks/kaggle/14aa_verify_14t_veri_fusion/14aa_verify_14t_veri_fusion.ipynb`.
- `notebooks/kaggle/14aa_verify_14t_veri_fusion/kernel-metadata.json`.
- `_build_14aa_notebook.py` — notebook builder following `_build_verifier_kernel.py` and `_build_14t_notebook.py` patterns.

Notebook builder requirements:

- Write JSON with `ensure_ascii=True`.
- Preserve notebook `source` array formatting: every line ends with `\n` except the final line of each cell.
- Do not edit notebooks with raw string replacement.

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| CLIP-SENet v6 kernel output retention changed. If `yahiaakhalafallah/13-clip-senet-train` was re-pushed past v6, the checkpoint may differ. | Hash-check at runtime. If SHA mismatch is discovered, keep `drift_parents` informational rather than required and report the checkpoint path/SHA in JSON. |
| TransReID checkpoint mirror drift or missing file. | Search mirrors in order: `mrkdagods`, `yahiaakhalafallah`, `gumfreddy`. Fail only if all mirrors miss `reid/vehicle_transreid_vit_base_veri776.pth`. |
| P100 `sm_60` wheel incompatibility. | Copy the 14y v5 compute-capability guard exactly. |
| New CLI does not reproduce notebook score-fusion math. | Spec 2 requires copying `score_similarity`, `score_all_similarity`, AQE on both streams, and rerank-on-fused-similarity from the 14t notebook around lines 1170-1275. |
| Parent metrics drift slightly while fused metric passes. | Parent rows are optional. The verifier's required acceptance is only the fused headline mAP/R1 band. |

End state: if the kernel writes `14aa_verify_results.json` with required rows passing, mark the 14t fusion verifier green and add it to the verification-suite inventory.