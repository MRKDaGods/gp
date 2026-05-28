---

# VeRi-776 Paper Campaign — Experiment Spec

**Date**: 2026-05-28
**Author**: MTMC Planner
**Status**: PROPOSED — ready for Coder dispatch in 3 waves
**Branch**: `feature/pipeline-model-integration`
**Paper**: `main.tex` — *"93.3% mAP on VeRi-776: Revisiting ViT Training Recipes and Score-Level Fusion for Vehicle Re-Identification"*
**Current claimed result**: 93.30% mAP / 98.45% Rank-1 (single-flip CLS-only Stream 1 @ 89.97/98.33 fused with CLIP-SENet v6, `w_1=0.3, w_2=0.7`).

---

## ⚠️ Blocking unknowns the Coder MUST resolve before Wave 2

Three of these are recipe-vs-paper mismatches discovered while drafting this spec. They affect every ablation row, so they MUST be reconciled before any GPU push.

1. **Stream-1 actual training recipe ≠ paper Table II.**
   The 89.97/98.33 checkpoint (`vehicle_transreid_vit_base_veri776.pth`, dataset `mrkdagods/mtmc-weights`) is produced by `notebooks/kaggle/08_vehicle_reid_sota/08_vehicle_reid_sota.ipynb`. Code inspection shows the actual recipe is:
   - Optimizer: AdamW, `backbone_lr=3.5e-4`, `head_lr=3.5e-3`, **LLRD decay = 0.75** (paper Table II claims **0.65**).
   - Loss: **CE-LS(ε=0.1) + TripletLoss(m=0.3) + CenterLoss(5e-4, delayed)** — paper Table II claims **CE-LS + SupCon**.
   - 140 epochs, 10-epoch warmup, batch 96 (48×2-T4 DP), P=24/K=4, 224×224, HFlip + Pad+Crop + ColorJitter + RandomErasing(p=0.5).
   - Center loss is **omitted** from paper §3.2.2 entirely.
   - Label-smoothing constant: code uses ε=0.1; metadata json line in the same kernel writes `"label_smoothing": 0.15` — internal inconsistency.

   **Implication**: the user-supplied ablation matrix's "+SupCon only" / "Full" toggles only make sense if the user accepts that the "Full" row is a *new* training run with the SupCon recipe described in the paper (not a re-load of the existing 08-kernel checkpoint). Coder must explicitly confirm with the user whether (a) the paper text gets corrected to match the 08-kernel CE+Triplet+Center recipe, or (b) a fresh SupCon "Full" training is performed and replaces the 89.97 number throughout the paper if it differs. **Do not push any training kernel until this is confirmed.**

2. **No canonical Stream-1 training notebook is recorded in the repo.** `docs/model-cards.md` line 75 states the training notebook is "NOT RECORDED IN REPO" for this checkpoint, and the 09v notebook is eval-only. The 08-kernel above is the de-facto producer. Confirm by running `kaggle kernels output yahiaakhalafallah/08-vehicle-reid-sota` (or sibling slug; query `docs/_data/kernel_inventory.json` for the exact owner/slug) and matching the produced `vehicle_transreid_vit_base_veri776.pth` SHA against the file in `mrkdagods/mtmc-weights`. Record SHA in `experiments/veri776-paper/REPORT.md`.

3. **Stream-2 (CLIP-SENet v6) checkpoint and recipe are confirmed.** Notebook `notebooks/kaggle/13_clip_senet_train/13_clip_senet_train.ipynb`, kernel `yahiaakhalafallah/13-clip-senet-train` v6, file `vehicle_clip_senet_veri776.pth` (or `best_mAP.pth`). Recipe: Adam 5e-4, cosine + 5-ep warmup, P=8/K=8, batch=128 (accum=2), 320×320, CE-LS(0.1) + SupCon(τ=0.07), 24 epochs on P100. This is the **frozen** Stream-2 across the entire campaign — do not retrain.

4. **Fusion-eval entry point is confirmed.** `notebooks/kaggle/14t_veri_fusion/14t_veri_fusion.ipynb` (kernel slug `yahiaakhalafallah/14t-veri-fusion-clip-senet-x-transreid`) is the entry point that loaded both checkpoints by path, extracted query+gallery features for both streams, ran AQE + k-reciprocal + concat-patch flip TTA, and reproduced 93.30/98.45. It discovers the TransReID checkpoint via `find_file_under_input("vehicle_transreid_vit_base_veri776.pth", preferred_slug="mtmc-weights")` and the CLIP-SENet checkpoint via `discover_clip_senet_checkpoint()` over `mrkdagods/mtmc-weights`. **This script will be reused 8 times** (5 ablations × eval + 3 seeds × eval) with the only change being which TransReID checkpoint is mounted. Each Stream-1 ablation kernel must export a `.pth` to a Kaggle dataset slot that 14t can mount as input.

5. **Cached feature matrices for retrieval panels (Part 3) — location not currently recorded.** The 14t kernel writes feature matrices to its kernel output but the exact filenames are not in `docs/findings.md`. Coder must run `kaggle kernels output yahiaakhalafallah/14t-veri-fusion-clip-senet-x-transreid -p tmp_14t_outputs/` and identify the `.npy`/`.pkl` files containing per-query and per-gallery features for both streams. Document the path in Section D.

6. **The ablation row "+AdamW only" requires special handling.** Paper Table II's claim is "LLRD=0.65, lr=3.5e-4". Code shows LLRD=0.75. To keep the +AdamW-only ablation honest, use **the actual LLRD=0.75, lr=3.5e-4** as the reference AdamW configuration. Flag the divergence from paper text in Section E.

---

## Section A — Campaign Overview

### Scope (decided by user)
- 5 isolated Stream-1 ablation kernels (Section B).
- 3 full-pipeline retrains of the "Full" Stream-1 recipe with seeds {42, 123, 456}, each evaluated through the full AQE + rerank + concat-patch-flip TTA + frozen-Stream-2 fusion pipeline (Section C).
- Total: **8 GPU kernels** + 8 CPU-bound fusion-eval kernels (the latter can be folded into the GPU kernel's tail or pushed as separate CPU notebooks).
- 1 CPU script for retrieval panels (Section D).
- Paper edits in `main.tex` (Section E).
- Aggregation artefacts (Section F).

### Account distribution and GPU-hour budget

The 08-kernel reference recipe trained in ~12h on 2×T4 (140 epochs, batch 96). All 8 training kernels are budgeted at ≤12h each per the user's hard cap.

| exp_id | Wall-time budget | Account (push to) | Concurrent slot | Rationale |
|--------|------------------|-------------------|-----------------|-----------|
| A1 Baseline (SGD+CE+Trip+DeiT) | 8–10h | gumfreddy slot 1 | A | SGD shorter; matches TransReID original |
| A2 +CLIP only | 10–12h | gumfreddy slot 2 | A | Same opt, swap init |
| A3 +SupCon only | 10–12h | mrkdagods slot 1 | B | Diff account → parallel |
| A4 +AdamW only | 10–12h | mrkdagods slot 2 | B | Same |
| A5 Full | 10–12h | ali369 slot 1 | C | Diff account → parallel |
| S1 Seed=42 (= A5 re-run) | 10–12h | ali369 slot 2 | C | Same account/wave |
| S2 Seed=123 | 10–12h | gumfreddy slot 1 (after A1) | D | After Wave 2 slot frees |
| S3 Seed=456 | 10–12h | mrkdagods slot 1 (after A3) | D | After Wave 2 slot frees |

Total GPU-hours: ~80h across 3 accounts. Wall-clock with full parallelism: ~24h (two staggered waves of 4 kernels). Kaggle 2-concurrent-GPU-per-account rule per `docs/kaggle-workflow.md` is respected; this plan never exceeds 2 concurrent kernels per account.

### Output directory convention
- Repo root: `experiments/veri776-paper/<exp_id>/`
- Sub-files per exp_id:
  - `recipe.json` — frozen hyperparam dict
  - `train_log.json` — per-epoch metrics
  - `best_mAP.pth` — final checkpoint (kept)
  - `eval_results.json` — full AQE + rerank + TTA + fusion grid
  - `kernel_metadata.json` — Kaggle slug, version, account, commit SHA
  - `feature_q.npy`, `feature_g.npy`, `q_pids.npy`, `g_pids.npy`, `q_camids.npy`, `g_camids.npy` — for downstream panels (only for A5 / seed runs)
- Roll-up: `experiments/veri776-paper/results.csv`, `results.json`, `REPORT.md` (Section F).
- Figures: `figures/paper/retrieval/panel_{1..6}.pdf`.

---

## Section B — Stream-1 Ablation Matrix (5 rows, isolated)

**Held-constant evaluation** across all 5 rows (and the 3 seed retrains): single-flip CLS-only 768D + AQE k=3 + k-reciprocal rerank `k1=80, k2=15, λ=0.2` + no concat-patch + no ten-crop. This matches paper Sec 5.2 "mAP-optimal" operating point and the held setting required for clean ablation deltas.

| exp_id | Name             | Init                                | Optimizer                          | Loss                                  | LR sched                          | Aug pipeline                                                            | Epochs | Expected mAP / R1 |
|--------|------------------|-------------------------------------|------------------------------------|---------------------------------------|-----------------------------------|--------------------------------------------------------------------------|:------:|-------------------|
| **A1** | Baseline         | DeiT-B/16 (`deit_base_distilled_patch16_224`) | **SGD** mom=0.9, lr=0.008, wd=1e-4 | **CE + Triplet(m=0.3)**               | Cosine, 5-ep warmup               | TransReID-paper aug (HFlip + Pad+Crop + RandomErasing only — no ColorJitter) | 120 | TO BE MEASURED |
| **A2** | +CLIP init only  | **CLIP ViT-B/16** (`vit_base_patch16_clip_224.openai`) | SGD mom=0.9, lr=0.008, wd=1e-4 | CE + Triplet(m=0.3)                   | Cosine, 5-ep warmup               | TransReID-paper aug (same as A1)                                         | 120 | TO BE MEASURED |
| **A3** | +SupCon loss only| DeiT-B/16                           | SGD mom=0.9, lr=0.008, wd=1e-4     | **CE-LS(ε=0.1) + SupCon(τ=0.07)**     | Cosine, 5-ep warmup               | TransReID-paper aug (same as A1)                                         | 120 | TO BE MEASURED |
| **A4** | +AdamW only      | DeiT-B/16                           | **AdamW** bb_lr=3.5e-4, head_lr=3.5e-3, LLRD=0.75 *(see flag below)*, wd=1e-2 | CE + Triplet(m=0.3) | Cosine, 10-ep warmup              | TransReID-paper aug (same as A1)                                         | 140 | TO BE MEASURED |
| **A5** | Full (paper)     | CLIP ViT-B/16                       | AdamW bb_lr=3.5e-4, head_lr=3.5e-3, LLRD=0.75, wd=1e-2 | CE-LS(ε=0.1) + SupCon(τ=0.07)         | Cosine, 10-ep warmup              | Full augmentation per paper Table I (HFlip + Pad+Crop + ColorJitter + Gauss-blur p=0.2 + RandomPerspective p=0.2 + Norm CLIP + RandomErasing p=0.5) | 140 | TO BE MEASURED (target ≥ 89.97 mAP) |

### Per-row kernel artefacts to create

| exp_id | Notebook path to CREATE                                                            | Kernel slug                                  | Target account |
|--------|------------------------------------------------------------------------------------|----------------------------------------------|----------------|
| A1     | `notebooks/kaggle/A1_veri_baseline_sgd/A1_veri_baseline_sgd.ipynb`                | `gumfreddy/a1-veri-baseline-sgd`             | gumfreddy      |
| A2     | `notebooks/kaggle/A2_veri_clip_only/A2_veri_clip_only.ipynb`                      | `gumfreddy/a2-veri-clip-only`                | gumfreddy      |
| A3     | `notebooks/kaggle/A3_veri_supcon_only/A3_veri_supcon_only.ipynb`                  | `mrkdagods/a3-veri-supcon-only`              | mrkdagods      |
| A4     | `notebooks/kaggle/A4_veri_adamw_only/A4_veri_adamw_only.ipynb`                    | `mrkdagods/a4-veri-adamw-only`               | mrkdagods      |
| A5     | `notebooks/kaggle/A5_veri_full/A5_veri_full.ipynb`                                | `ali369/a5-veri-full`                        | ali369         |

Each notebook is created by cloning `notebooks/kaggle/08_vehicle_reid_sota/08_vehicle_reid_sota.ipynb` and toggling only the four ablation switches:

- **Init switch**: replace the `timm.create_model(...)` line for the backbone string. Map: A1, A3, A4 → `deit_base_distilled_patch16_224` (NOT distilled in TransReID original — clarify with user; default to non-distilled `deit_base_patch16_224` if unclear); A2, A5 → `vit_base_patch16_clip_224.openai`.
- **Optimizer switch**: A1/A2/A3 swap the AdamW block for `torch.optim.SGD(params, lr=0.008, momentum=0.9, weight_decay=1e-4)` and `CosineAnnealingLR(T_max=EPOCHS - WARMUP)` with `WARMUP=5`. A4/A5 keep AdamW + LLRD=0.75 + 10-ep warmup.
- **Loss switch**: A1/A2/A4 keep the existing `CE(ε=0.1) + Triplet(0.3) + Center(5e-4)` *minus Center* (remove Center to keep ablation clean — flag this divergence from 08-kernel in REPORT.md). A3/A5 replace Triplet+Center with `SupCon(τ=0.07)` and use `CE-LS(ε=0.1)`. Reference implementation: `lvm` SupCon at `khosla2020supcon` — drop into `losses.py` next to the existing Triplet class.
- **Aug switch**: A1/A2/A3/A4 use the TransReID-paper aug (HFlip + Pad+Crop + RandomErasing only); A5 uses the full paper-Table-I pipeline (adds ColorJitter, Gaussian blur p=0.2, RandomPerspective p=0.2).

### Held eval (run after each training kernel completes)

For each `<exp_id>`:
1. Upload the produced `best_mAP.pth` as a Kaggle dataset under the row's owner (`<account>/veri776-paper-<exp_id>-ckpt`).
2. Push a CPU-only eval notebook `notebooks/kaggle/eval_<exp_id>/eval_<exp_id>.ipynb` (clone of `09v_veri776_eval` with the checkpoint dataset mounted), constrained to the held grid: AQE k=3 + rerank (k1=80, k2=15, λ=0.2), single-flip, CLS-only 768D. Wall-time ~30 min.
3. Persist the JSON to `experiments/veri776-paper/<exp_id>/eval_results.json`.

### Known-broken combinations — FLAG, do not run

- **SGD lr=0.008 on CLIP init** (A2): is on the edge of known-broken from `docs/experiment-log.md` 2.22 — SGD high-LR + CLIP often diverges. **Mitigation**: include a warmup of 5 epochs (already in A2 spec) and gradient clipping clip_norm=1.0. If A2 diverges, run a held-LR safe alternative at lr=0.002 (SGD-low) and label the row "A2 — adjusted LR" in the results table.
- **CircleLoss + CLIP init**: paper §3.2.2 already notes catastrophic instability — do NOT include CircleLoss in any row.
- **Long schedules >140 epochs** (e.g. 180): 08-kernel notebook records peak at 140; longer overfit. Cap all rows at 140.

---

## Section C — 3-Seed Variance (3 rows)

**Recipe**: identical to A5 ("Full"). Only the random seed differs. After training, run the full eval = AQE k=3 + rerank `k1=80,k2=15,λ=0.2` + **concat-patch flip TTA (1536D)** + score fusion with the FROZEN CLIP-SENet v6 at `w_1=0.3, w_2=0.7`. This matches paper Table III ("Full PP → Fusion" → 93.30/98.45).

| exp_id | Seed | Account    | Notebook path                                                              | Kernel slug                          | Eval entry                                                                                |
|--------|------|------------|----------------------------------------------------------------------------|--------------------------------------|-------------------------------------------------------------------------------------------|
| **S1** | 42   | ali369     | `notebooks/kaggle/S1_veri_full_seed42/S1_veri_full_seed42.ipynb`           | `ali369/s1-veri-full-seed42`         | clone of `14t_veri_fusion.ipynb` with `S1` ckpt mounted, slug `ali369/s1-veri-fusion-eval`|
| **S2** | 123  | gumfreddy  | `notebooks/kaggle/S2_veri_full_seed123/S2_veri_full_seed123.ipynb`         | `gumfreddy/s2-veri-full-seed123`     | `gumfreddy/s2-veri-fusion-eval`                                                            |
| **S3** | 456  | mrkdagods  | `notebooks/kaggle/S3_veri_full_seed456/S3_veri_full_seed456.ipynb`         | `mrkdagods/s3-veri-full-seed456`     | `mrkdagods/s3-veri-fusion-eval`                                                            |

Seed plumbing: set `torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed); random.seed(seed)` and `torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True` at the top of the training cell. Also pass `generator=torch.Generator().manual_seed(seed)` to the train DataLoader.

Report per-seed: `mAP_full_pp`, `R1_full_pp`, `mAP_fused`, `R1_fused`. Aggregate as `mean ± std` and add to paper Table III as a fourth column "Seed-variance (mean ± std)" in a new row. If std < 0.30pp mAP the seed-variance row is presented as a stability claim; if ≥ 0.30pp, paper text must temper "robustness" language.

---

## Section D — Retrieval Panels (Part 3, CPU-only)

### Script to CREATE: `scripts/paper/generate_retrieval_panels.py`

**Function**: load cached query+gallery feature matrices from the seed-S1 (or A5) run, build 6 retrieval panels (3 success, 3 failure) per the paper's qualitative analysis modes, render to PDF.

**Inputs (resolve paths at runtime)**:
- ViT-B query feats: `experiments/veri776-paper/A5/feature_q.npy` shape `(1678, 768)`
- ViT-B gallery feats: `experiments/veri776-paper/A5/feature_g.npy` shape `(11579, 768)`
- CLIP-SENet query feats: `experiments/veri776-paper/_stream2/feature_q.npy` shape `(1678, 2048)` *(extracted once from the frozen v6 ckpt and cached)*
- CLIP-SENet gallery feats: `experiments/veri776-paper/_stream2/feature_g.npy` shape `(11579, 2048)`
- Query PIDs/CAMIDs and gallery PIDs/CAMIDs: parallel `.npy` files in same dirs.
- Query/gallery image paths: `experiments/veri776-paper/_meta/query_paths.json`, `gallery_paths.json` produced by 14t. **Coder MUST verify these exist after pulling 14t kernel output** — if absent, the script must include a 30-line extraction helper that walks the VeRi-776 dataset to rebuild the parallel arrays.

**Algorithm**:
1. Compute per-stream cosine sims, AQE-expand per paper, then run k-reciprocal rerank per paper.
2. Apply fusion `S = 0.3·S_vit + 0.7·S_clip` on the final post-PP cosine matrices.
3. Compute per-query Rank-1 hit flag for: ViT-B only, CLIP-SENet only, Fusion.
4. Build deterministic candidate pools (rule below) for each of 6 categories. Use `numpy.random.default_rng(seed=20260528)` for any tie-breaking. The exact selection rule:
   - **Success #1 — fusion corrects a stream's mistake**: `vit_hit=False AND clip_hit=False AND fusion_hit=True`. Sort candidates by `(fusion_top1_sim - max(vit_top1_sim, clip_top1_sim))` descending. Take first.
   - **Success #2 — frontal↔rear viewpoint**: among `fusion_hit=True`, pick query whose GT match has the maximum `|cam_q - cam_g_top1|` (proxy for viewpoint divergence) AND `len(set([vehicle_type_q]))==1`. Take first by deterministic PID order.
   - **Success #3 — illumination shift**: among `fusion_hit=True`, use a lightweight HSV-V variance proxy from the cached query crop vs gallery crop (compute brightness mean diff > 30/255). Take first by PID order.
   - **Failure #1 — same make/model/colour confound**: pick `fusion_hit=False` where top-1 gallery crop has the same `(model_str, color_str)` metadata as the query (from VeRi-776 `train_label.xml` / `test_label.xml`).
   - **Failure #2 — extreme occlusion**: pick `fusion_hit=False` with the lowest YOLO crop area among queries (proxy for occlusion); take first by PID order.
   - **Failure #3 — low-res gallery**: among `fusion_hit=False`, pick query whose top-5 gallery crop areas are all `< 40×40 px` (read from BBox area in test set). Take first.
5. For each panel: render a 1-row 6-column figure (query + top-5). GT match shown with a green border; non-matches with red. Save as `figures/paper/retrieval/panel_{1..6}.pdf` via `matplotlib` with `bbox_inches='tight'`.
6. Emit sidecar JSON log per panel: `figures/paper/retrieval/panel_{i}.json` containing `{"panel_id": i, "category": str, "query_id": int, "query_path": str, "gallery_ids": list[int], "gallery_paths": list[str], "gt_hit_flags": list[bool], "caption_seed": str, "selection_rule": str}`.
7. Emit a master `figures/paper/retrieval/index.json` listing all 6 panels and a one-paragraph caption per panel suitable for direct LaTeX inclusion.

**Determinism**: seed=`20260528`. No network access. Pure-NumPy + PIL + matplotlib. CPU-only. Runs in `<1 min` on the local laptop.

**Coder verification**: after running, all 6 PDFs must be `> 100 KB` and `panel_*.json` must include non-empty `gallery_paths` arrays. If any panel returns `gallery_paths == []`, the metadata reconstruction step failed and Coder must repair.

---

## Section E — Paper Edits in `main.tex` (Part 4)

All edits are pure-text via `replace_string_in_file`. Each edit lists the unique surrounding context string (≥5 chars on each side) the Coder will use as `oldString`.

### E.1 — Overclaiming sweep (replace "absolute state of the art" → "state of the art in mAP")

**Reason**: fused Rank-1 (98.45) does not beat CLIP-SENet's 98.7 Rank-1. "Absolute SOTA" overclaims.

1. Abstract (currently consistent):
   - Search anchor: `establishes a new state of the art on VeRi-776 in terms of mAP`
   - Status: **OK** — abstract already correctly limits the claim. No change required, but Coder MUST verify abstract phrasing matches §5.1 and §6 (see below).

2. §5.1 Comparison with SOTA, paragraph after Table II `sota`:
   - Find: `The fusion result establishes a new absolute state of the art, with`
   - Replace with: `The fusion result establishes a new state of the art in mAP on VeRi-776, with`

3. §6 Conclusion:
   - Find: `a new state of the art on VeRi-776 in terms of mAP, exceeding the previous best published mAP`
   - Status: **OK** — already correctly bounded. Verify unchanged.

4. C2 contribution bullet (Introduction):
   - Find: `\textbf{C2---New State of the Art in mAP:}`
   - Status: **OK** — already correctly bounded.

5. Search the entire `main.tex` for the strings `absolute state of the art`, `new state of the art` (without "in mAP" qualifier), `new SOTA`. Currently only the §5.1 occurrence (#2 above) overclaims; verify no others slip in.

### E.2 — Limitation section softening once Part 1 isolated ablations land

Currently at §6 Discussion → Limitations bullet 4:
```
Per-component isolated ablations (CLIP initialisation alone, SupCon
alone, AdamW alone) require additional training runs and are deferred to
future work; the present ablations focus on cumulative post-processing steps.
```

Replace AFTER Wave 3 with the measured deltas, e.g.:
```
We further report isolated per-component ablations (Table~\ref{tab:isolated}):
the CLIP initialisation contributes $\Delta_{\text{CLIP}}$\,pp mAP, SupCon
$\Delta_{\text{SupCon}}$\,pp, and AdamW $\Delta_{\text{AdamW}}$\,pp when each is
introduced in isolation over the TransReID baseline. The combined "Full"
recipe reaches $X.XX$\,% mAP / $Y.YY$\,% Rank-1, indicating that the gains
are largely additive but not perfectly linear, with a measured interaction
residual of $\delta$\,pp.
```
Coder fills `Δ_*`, `X.XX`, `Y.YY`, `δ` from `experiments/veri776-paper/results.csv` in Wave 3.

Also confirm the limitations section already mentions:
- CityFlowV2 domain gap (✅ present, bullet 3)
- Two-stream cost (✅ present, bullet 2)
- O(N²) post-processing (✅ present, bullet 1)
No additional limitations need to be added.

### E.3 — Reference and table integrity audit

Run before Wave 3: for every `\ref{...}` and `\cite{...}` in `main.tex`, verify the label exists in `main.tex` and the bib key exists in `references.bib`. Required keys found in scan:
- Tables: `tab:augmentation, tab:hyperparams, tab:implementation, tab:sota, tab:vit_compare, tab:ablation, tab:fusion_sweep, tab:pp_contrib, tab:qual` — verify each `\label{}` resolves.
- Cites: `he2021transreid, radford2021clip, khosla2020supcon, zhong2017rerank, arandjelovic2012aqe, li2023clipreid, lu2025clipsenet, dosovitskiy2021vit, liu2016veri, liu2018provid, ye2021reidsurvey, li2024vehiclereidsurvey, zhou2018vami, meng2020pven, rao2021cal, zheng2020vehiclenet, shen2022hpgn, cybercore2021strong, ghosh2023rptm, luo2023mbr4b, wang2020fastreid, pan2018ibn, loshchilov2019adamw, oquab2023dinov2, opencodepapers2024veri, zhang2022vehiclererank, lou2019veriwild`.

If `references.bib` is missing any of these, **STOP** and ask user. Do not invent entries.

### E.4 — Numerical consistency audit

Verify that throughout `main.tex`:
1. Stream-1 mAP-optimal operating point is reported as **89.97 / 97.80** wherever cited (Table II, §5.2 line "89.97% mAP / 97.80% Rank-1", abstract, Table I sota row, conclusion).
2. Stream-1 Rank-1-optimal operating point is reported as **85.14 mAP / 98.33 Rank-1** wherever cited (§5.2 ablation context).
3. Joint operating point **89.71 mAP / 98.15 Rank-1** is mentioned only once (§5.2 paragraph after Table II) — confirm.
4. Fusion result is **93.30 / 98.45** wherever cited (abstract, C2, Table I sota row, Table III ablation row, conclusion, §5.3 fusion-sweep table row).
5. CLIP-SENet standalone Rank-1 is **98.7** (paper claim) — confirm match in §1.3 C2 and §6.

If any number is misquoted, fix it in-place with the canonical value above. Do NOT update any number based on Wave 3 results unless the corresponding measured run completed cleanly.

### E.5 — Recipe vs Table II reconciliation (depends on user decision in Blocking Unknown #1)

Two paths, mutually exclusive:

**Path α** (user picks "correct paper to match 08-kernel recipe"):
- Table II row "Loss": change `CE-LS + SupCon` → `CE-LS + Triplet + Center` (and add a row for Center weight 5e-4 with delayed start).
- Table II row "LR backbone": confirm `3.5e-4 (LLRD=0.75)` (currently paper says 0.65 — change).
- §3.2.2 add a "Center loss" paragraph documenting the third loss term.
- The C1 contribution and §5.2 narrative explaining "what drives the gain" must drop SupCon and instead credit Triplet+Center+CLIP+AdamW+LLRD.

**Path β** (user picks "retrain Full with SupCon and replace numbers"):
- After A5 + seed runs complete, replace 89.97/98.33 numbers throughout with the measured A5 result (could be slightly higher or lower). Update Tables I, III, abstract, C1, conclusion, fusion-sweep header consistently.

Coder MUST NOT proceed with E.5 until the user picks α or β.

### E.6 — Suggested 2022–2025 references to add (Coder to verify availability and add to `references.bib`)

3–5 genuinely relevant additions; Coder confirms publication metadata before inclusion:

1. **`zheng2023strongbaseline`** — Zheng et al., "Vehicle Re-Identification: An Efficient Baseline using Triplet and Classification Losses", *Pattern Recognition Letters*, 2023. Justification: directly contemporary, strong CNN baseline our recipe surpasses.
2. **`wang2024clipdriven`** — Wang et al., "CLIP-Driven Fine-Grained Text-Image Person Re-Identification", *IEEE T-MM*, 2024. Justification: closest CLIP+ReID neighbour to our recipe; cite in §2.2.
3. **`chen2024multiscalevehiclereid`** — Chen et al., "Multi-Scale Spatial Transformer for Vehicle Re-Identification", *IEEE T-ITS*, 2024. Justification: post-MBR4B vehicle method we should position against.
4. **`zhu2025vitenhanced`** — Zhu et al., "Enhanced ViT Architectures for Vehicle Re-Identification on Large-Scale Benchmarks", *IEEE T-ITS*, 2025. Justification: most direct ViT vs CLIP-ViT comparator; cite in §2.2 and Table II.
5. **`huang2023reranking`** — Huang et al., "Adaptive k-Reciprocal Re-ranking for Vehicle Re-Identification", *Neurocomputing*, 2023. Justification: supports §3.4.2 rerank discussion.

Coder action: search Google Scholar by title; if a real paper with matching scope exists, add to `references.bib` with correct metadata and cite once each in the natural section. If a title returns no real paper, **omit** rather than fabricate. Final count of added references may be 3, 4, or 5 — never invented.

---

## Section F — Output Artefacts

### F.1 — `experiments/veri776-paper/results.csv`

Columns (exact order):
```
experiment_name,seed,stream1_clip_init,stream1_supcon,stream1_adamw,stream2_fixed,aqe,rerank,tta,fusion,mAP,rank1,notes
```
One row per kernel result. Booleans as `0/1`. Free-text in `notes` for caveats. `tta` is `single_flip|concat_patch_flip|none`. `fusion` is the `w_1` value (`-` if no fusion).

Example rows the Coder will produce in Wave 3:
```
A1_baseline,42,0,0,0,clip_senet_v6,k=3,"k1=80;k2=15;lambda=0.2",single_flip,-,XX.XX,YY.YY,SGD+CE+Triplet+DeiT
A5_full,42,1,1,1,clip_senet_v6,k=3,"k1=80;k2=15;lambda=0.2",single_flip,-,XX.XX,YY.YY,Stream-1 only
A5_full_fused,42,1,1,1,clip_senet_v6,k=3,"k1=80;k2=15;lambda=0.2",concat_patch_flip,0.3,XX.XX,YY.YY,Fusion target
S1_seed42_fused,42,1,1,1,clip_senet_v6,k=3,"k1=80;k2=15;lambda=0.2",concat_patch_flip,0.3,XX.XX,YY.YY,
S2_seed123_fused,123,1,1,1,clip_senet_v6,k=3,"k1=80;k2=15;lambda=0.2",concat_patch_flip,0.3,XX.XX,YY.YY,
S3_seed456_fused,456,1,1,1,clip_senet_v6,k=3,"k1=80;k2=15;lambda=0.2",concat_patch_flip,0.3,XX.XX,YY.YY,
```

### F.2 — `experiments/veri776-paper/results.json`

Schema:
```json
{
  "campaign": "veri776-paper",
  "branch": "feature/pipeline-model-integration",
  "commit_sha": "<git rev-parse HEAD at Wave 3 aggregation>",
  "dataset": {
    "name": "veri-776",
    "kaggle_source": "abhyudaya12/veri-vehicle-re-identification-dataset",
    "split_sha": "<sha256 of sorted query+gallery filename listing>"
  },
  "experiments": [
    {
      "exp_id": "A1_baseline",
      "seed": 42,
      "recipe": { /* mirror of recipe.json */ },
      "metrics": { "mAP_held_eval": null, "R1_held_eval": null, "mAP_fused": null, "R1_fused": null },
      "kernel": { "slug": "gumfreddy/a1-veri-baseline-sgd", "version": null, "account": "gumfreddy" },
      "status": "pending|running|complete|fail",
      "wall_time_sec": null,
      "notes": ""
    }
  ],
  "seed_variance": {
    "mAP_mean": null, "mAP_std": null,
    "R1_mean": null,  "R1_std": null
  },
  "failures": [],
  "paper_table_values": {
    "tab_sota_ours_stream1": { "mAP": null, "R1": null },
    "tab_sota_ours_fusion":  { "mAP": null, "R1": null },
    "tab_isolated":          [ /* one row per A1..A5 */ ]
  }
}
```

`null` placeholders are filled by aggregation script in Wave 3.

### F.3 — `experiments/veri776-paper/REPORT.md`

Sections:
- TL;DR — A5 measured vs paper 89.97, seed-variance σ, fusion result.
- Per-ablation table with measured Δ over A1 baseline.
- Path α/β decision and resulting paper edits.
- Failures, reruns, and any deviations from this spec.
- Pointers to all artefacts (CSV, JSON, panels).

### F.4 — `figures/paper/retrieval/panel_{1..6}.pdf` + sidecar JSON (Section D).

---

## Section G — Risks and Rollback

| Risk | Trigger | Mitigation |
|------|---------|------------|
| Training kernel >12h wall-time | Any of A1–A5/S1–S3 saturates 12h | Split into (a) `train_phase` kernel saving optimizer + scheduler state at epoch 70 to `<exp_id>-resume` dataset, (b) `eval_phase` kernel resuming and finishing. Do NOT reduce epochs unless the user authorises a "short-schedule" labeled row in REPORT.md. |
| Kaggle double-push | Coder accidentally runs `kaggle kernels push` twice within seconds | After every push, `kaggle kernels status <slug>` once. If two versions are listed `running`, cancel the older one (`kaggle kernels cancel`). Cite `docs/kaggle-workflow.md` §"Push Safety Rules". |
| Missing dataset source warning | `kaggle kernels push` prints "The following are not valid dataset sources" | Cancel immediately, fix `kernel-metadata.json` `dataset_sources`, single re-push. Refuse to let the bad run consume the GPU slot. |
| SGD diverges on CLIP init (A2) | NaN loss within first 5 epochs | Switch to held-LR variant lr=0.002 + grad-clip 1.0, label row "A2 — adjusted LR" in REPORT.md and paper Table III footnote. Do not fake the unadjusted A2 number. |
| Stream-2 ckpt unavailable on a non-yahiaakhalafallah account | A5/seed eval kernels pushed under ali369 or mrkdagods cannot mount `yahiaakhalafallah/13-clip-senet-train` directly | Pre-mirror `vehicle_clip_senet_veri776.pth` to `mrkdagods/mtmc-weights` dataset (which all accounts already use) once, then point all eval kernels at `mrkdagods/mtmc-weights` for both checkpoints. |
| Path α vs β stall | User has not chosen | Coder asks via `vscode_askQuestions` once between Wave 1 and Wave 2. Do NOT default. |
| Dead-end repeats | Coder considers a variation already explored | Cross-check against `docs/dead-ends.md` before adding any "interesting" extra row. The 5 ablations are exhaustive — do not add a 6th without user approval. Avoid: ViT-L/14 scale-up (14p3, FAIL), ViT-B 256² (14q, FAIL), DINOv2 SSL (14r-probe, FAIL), CLIP-ReID 2-stage (14r-primary/recovery, FAIL/FAIL), CLIP-SENet 256² (v7, FAIL), CircleLoss + CLIP (catastrophic). |
| Disk pressure during Wave 3 aggregation | Multiple kernel outputs downloaded simultaneously | After each `kaggle kernels output`, immediately delete `last.pth` and any `*.pth` from FAILed runs per `docs/kaggle-workflow.md` "Disk Hygiene". Run `Get-ChildItem | Measure-Object -Sum Length` before/after and report reclaimed GB. |

---

## Section H — Coder Dispatch Plan (3 waves)

### Wave 1 — Paper edits + panel script (parallel, CPU-only, local)

**Touch**: `main.tex`, `references.bib`, `scripts/paper/generate_retrieval_panels.py` (new), `figures/paper/retrieval/` (new dir).
**Read**: `main.tex` (full), `references.bib` (full), `notebooks/kaggle/14t_veri_fusion/14t_veri_fusion.ipynb` (feature-extraction cells), `docs/findings.md` "Canonical VeRi-776".
**Success criterion**:
- E.1 + E.3 + E.4 all green; `main.tex` builds with `pdflatex` (or `latexmk`) with zero unresolved `\ref` and zero unresolved `\cite`.
- `scripts/paper/generate_retrieval_panels.py` exists, type-checks (`python -m py_compile`), and produces 6 PDFs + sidecar JSON given mocked feature `.npy` files of correct shape.
- E.6 references added if real, omitted if not.
- E.5 paper edits **NOT YET APPLIED** — user-decision blocker.
**Expected runtime**: 30–60 min.

### Wave 2 — Generate 8 Kaggle notebooks and push (parallel, GPU, Kaggle)

**Touch**: 8 new directories under `notebooks/kaggle/`, plus 8 eval-clone notebooks (A1..A5 held eval + S1..S3 fusion eval = 8 eval kernels total). Plus per-kernel `kernel-metadata.json`.
**Read**: `notebooks/kaggle/08_vehicle_reid_sota/08_vehicle_reid_sota.ipynb` (clone template for training), `notebooks/kaggle/09v_veri776_eval/09v-veri776-eval.ipynb` (clone template for A1..A5 held eval), `notebooks/kaggle/14t_veri_fusion/14t_veri_fusion.ipynb` (clone template for S1..S3 fusion eval), `docs/kaggle-workflow.md` (push rules).
**Push order**:
1. Resolve Blocking Unknown #1 with user (path α / β). Do not push if unresolved.
2. Push A1 (gumfreddy slot 1), A2 (gumfreddy slot 2), A3 (mrkdagods slot 1), A4 (mrkdagods slot 2) — single command each, validate metadata, poll status.
3. As each completes, upload ckpt as a Kaggle dataset (`<account>/veri776-paper-<exp_id>-ckpt`), then push the corresponding held-eval kernel.
4. Push A5 (ali369 slot 1) + S1 (ali369 slot 2) once any of A1–A4 slots free up.
5. Push S2 + S3 on whichever accounts have free slots after the first batch.
**Success criterion**: all 8 training kernels reach `complete` status; all 8 eval kernels write `eval_results.json`; per-kernel artefacts pulled into `experiments/veri776-paper/<exp_id>/`.
**Expected runtime**: 24–36h wall-clock with full parallelism.

### Wave 3 — Aggregate and finalise (CPU, local)

**Touch**: `experiments/veri776-paper/results.csv`, `results.json`, `REPORT.md`, `main.tex` (E.2 + E.5 + Table III row updates with measured numbers).
**Read**: all 8 `eval_results.json` files, all 8 `recipe.json` files, `git rev-parse HEAD`, dataset split sha-256.
**Success criterion**:
- `results.csv` and `results.json` populated; no `null` metrics for completed runs.
- `REPORT.md` contains the 5-ablation table + seed-variance line + path α/β resolution narrative.
- `main.tex` rebuilt with measured numbers; one final pass through E.4 numerical consistency; PDF builds with no `\ref`/`\cite` warnings.
- Retrieval panels (Section D) regenerated from the final S1 feature matrices and committed.
**Expected runtime**: 1–2h.

---

## Section I — Acceptance Criteria for Spec Completion

This spec is "done" when:
- Coder has the 8 training notebooks, 8 eval notebooks, 1 panel script, and the paper edit list in hand.
- All 3 blocking unknowns (Section ⚠️ items 1, 2, 5) are resolved by Coder before Wave 2 begins.
- User decision on Section E.5 path α vs β is recorded in `REPORT.md`.

No GPU work begins until Wave 1 completes and the user resolves the path α/β question.

---

## Addendum v1 — User Decisions 2026-05-28

This addendum supersedes specific scoping decisions in Sections A, B, C, D, and H. The original sections remain unchanged for historical traceability; where conflicts exist, **this addendum is authoritative**.

### AD.1 — Resolution of Blocking Unknown #1 (recipe mismatch): **Path γ (BOTH)**

The user resolves the recipe-vs-paper mismatch by running BOTH the actual deployed recipe and the paper-described recipe as two separate "Full" rows. Path α and Path β from Section E.5 are no longer mutually exclusive — they are both measured, and the paper headline is chosen empirically.

**Updated Stream-1 ablation matrix — 6 rows** (supersedes Section B's 5-row table):

| exp_id | Name                | Init        | Optimizer                              | Loss                                                  | LLRD | Aug                                 | Epochs | Purpose                                                                 |
|--------|---------------------|-------------|----------------------------------------|-------------------------------------------------------|------|-------------------------------------|:------:|-------------------------------------------------------------------------|
| **A1** | Baseline            | DeiT-B/16   | SGD mom=0.9, lr=0.008, wd=1e-4         | CE + Triplet(m=0.3)                                   | n/a  | TransReID-paper aug                 | 120    | Reference floor                                                          |
| **A2** | +CLIP only          | CLIP ViT-B/16 | SGD mom=0.9, lr=0.008, wd=1e-4       | CE + Triplet(m=0.3)                                   | n/a  | TransReID-paper aug                 | 120    | Isolate init delta                                                       |
| **A3** | +SupCon only        | **DeiT-B/16** | **SGD** mom=0.9, lr=0.008, wd=1e-4   | **CE-LS(ε=0.1) + SupCon(τ=0.07)** (Triplet REMOVED)   | n/a  | TransReID-paper aug                 | 120    | Clean isolated metric-loss probe over the SGD+DeiT baseline. Builds on A1, swapping ONLY the metric loss. SupCon is not necessarily in the final recipe — this row exists purely to quantify the SupCon-vs-Triplet delta under matched conditions. |
| **A4** | +AdamW only         | DeiT-B/16   | AdamW bb_lr=3.5e-4, head_lr=3.5e-3, wd=1e-2 | CE + Triplet(m=0.3)                              | 0.75 | TransReID-paper aug                 | 140    | Isolate optimizer delta                                                  |
| **A5α** | **Full (actual)**  | CLIP ViT-B/16 | AdamW bb_lr=3.5e-4, head_lr=3.5e-3, wd=1e-2 | **CE-LS(ε=0.1) + Triplet(m=0.3) + CenterLoss(λ=5e-4, delayed)** | **0.75** | Full paper Table I aug | 140 | The recipe that actually produced the deployed 89.97/98.33 ckpt (`vehicle_transreid_vit_base_veri776.pth`). Reproduces the de-facto Stream-1 result. **Default headline.** |
| **A5β** | **Full (paper)**    | CLIP ViT-B/16 | AdamW bb_lr=3.5e-4, head_lr=3.5e-3, wd=1e-2 | **CE-LS(ε=0.1) + SupCon(τ=0.07)**                    | **0.65** | Full paper Table I aug | 140 | The recipe described in §3.2.2 of `main.tex` as currently drafted. Tests whether the paper-as-written claim holds. |

**Headline-selection rule**: the paper's "Full Stream-1" row reports `max(A5α, A5β)` by mAP. If `|mAP(A5α) − mAP(A5β)| ≤ 0.10` (within evaluation noise per `docs/findings.md` ReID noise band), **both rows are reported honestly side-by-side** and the text discusses the equivalence rather than declaring a winner.

**Note on A3**: A3 retains DeiT init + SGD (matches A1's baseline), swapping ONLY the loss. This is intentional to keep the SupCon-vs-Triplet contrast isolated from optimizer and init effects. The original Section B A3 spec (DeiT + SGD + SupCon) is preserved; no change needed for that row.

### AD.2 — Resolution of Blocking Unknown #2 (canonical-checkpoint verification): SHA-256 gate

Before cloning the 08 notebook to construct any A*/S* notebook, the Wave-2 coder MUST execute the following gate. Failure aborts Wave 2.

```pwsh
# 1. Pull 08 kernel output to a temp directory
kaggle kernels output <08-slug> -p tmp_08_output/

# 2. Compute SHA-256 of any produced vehicle_transreid_vit_base_veri776.pth
Get-FileHash tmp_08_output/vehicle_transreid_vit_base_veri776.pth -Algorithm SHA256

# 3. Pull the same filename from mrkdagods/mtmc-weights
kaggle datasets files mrkdagods/mtmc-weights | Select-String "vehicle_transreid_vit_base_veri776.pth"
kaggle datasets download -d mrkdagods/mtmc-weights -f vehicle_transreid_vit_base_veri776.pth -p tmp_mtmc_weights/

# 4. Compute SHA-256 and compare
Get-FileHash tmp_mtmc_weights/vehicle_transreid_vit_base_veri776.pth -Algorithm SHA256
```

**Pass condition**: both SHA-256 digests are byte-identical. Record both hashes in `experiments/veri776-paper/REPORT.md` under heading `### Canonical 08-kernel SHA verification`.

**Fail condition** (SHAs differ): STOP. Do not clone the 08 notebook for any A*/S* row. Return a failure report to the orchestrator naming both SHAs, the 08 kernel slug + version inspected, and the `mrkdagods/mtmc-weights` dataset version inspected.

**Cleanup**: after the gate completes (pass OR fail), delete `tmp_08_output/` and `tmp_mtmc_weights/` per disk-hygiene rules (`/memories/orchestrator-protocol.md` § Disk Hygiene). Report reclaimed GB via `Get-ChildItem | Measure-Object -Sum Length` before/after.

### AD.3 — Resolution of Blocking Unknown #5 (panel features): inline feature dumps

The "search 14t kernel output for cached features" step (original Section ⚠️ item 5 and Section D's `_stream2/feature_q.npy` resolution path) is **DROPPED**. Replace with inline feature dumping at the tail of the A5α and A5β training kernels.

**Tail-cell contract** (appended as the FINAL cell of each of `A5α` and `A5β` training notebooks; NOT appended to A1–A4 or S1–S3):

```python
# === FINAL CELL: Feature dump for paper retrieval panels ===
# Runs only after best-mAP checkpoint has been selected and the held-eval
# pipeline (AQE k=3 + rerank k1=80,k2=15,λ=0.2 + concat-patch flip TTA) has
# emitted q/g feature matrices for THIS run's Stream-1 (post-AQE, post-rerank
# normalised features used by fusion scoring).

import json, numpy as np
from pathlib import Path

EXP_ID = "A5alpha"   # or "A5beta" — set per notebook
OUT_DIR = Path(f"/kaggle/working/features/")
(OUT_DIR / "stream1").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "stream2").mkdir(parents=True, exist_ok=True)

# Stream 1 (this run's TransReID variant) — post-PP feature matrices
np.save(OUT_DIR / "stream1" / "query.npy",   stream1_q_feats_post_pp)    # (1678, 768) or (1678, 1536) if concat-patch
np.save(OUT_DIR / "stream1" / "gallery.npy", stream1_g_feats_post_pp)    # (11579, D)

# Stream 2 (frozen CLIP-SENet v6) — extracted in-kernel using the same query/gallery order
np.save(OUT_DIR / "stream2" / "query.npy",   stream2_q_feats_post_pp)    # (1678, 2048)
np.save(OUT_DIR / "stream2" / "gallery.npy", stream2_g_feats_post_pp)    # (11579, 2048)

# Row-aligned index map for downstream panel script
index_map = {
    "exp_id": EXP_ID,
    "query":   [{"row": i, "image_path": qp, "vehicle_id": int(qid), "camera_id": int(qc)}
                for i, (qp, qid, qc) in enumerate(zip(query_paths, query_pids, query_camids))],
    "gallery": [{"row": i, "image_path": gp, "vehicle_id": int(gid), "camera_id": int(gc)}
                for i, (gp, gid, gc) in enumerate(zip(gallery_paths, gallery_pids, gallery_camids))],
    "stream1": {"dim": int(stream1_q_feats_post_pp.shape[1]), "tta": "concat_patch_flip"},
    "stream2": {"dim": int(stream2_q_feats_post_pp.shape[1]), "tta": "concat_patch_flip"},
}
with open(OUT_DIR / "index_map.json", "w") as f:
    json.dump(index_map, f, indent=2)

print(f"Dumped {EXP_ID} features:",
      stream1_q_feats_post_pp.shape, stream1_g_feats_post_pp.shape,
      stream2_q_feats_post_pp.shape, stream2_g_feats_post_pp.shape)
```

**Post-pull layout** (Wave 3 coder downloads kernel output and stages):
```
experiments/veri776-paper/A5alpha/features/
  stream1/query.npy
  stream1/gallery.npy
  stream2/query.npy
  stream2/gallery.npy
  index_map.json
experiments/veri776-paper/A5beta/features/
  ... same structure ...
```

**Section D revision**: `scripts/paper/generate_retrieval_panels.py` reads from `experiments/veri776-paper/<A5_winner>/features/` (where `<A5_winner>` = whichever of `A5alpha` / `A5beta` becomes the paper headline). It is **fully local, CPU-only, NO Kaggle dependency**. The original Section D dependence on cached 14t output is dropped.

**Seed-row scope**: S1/S2/S3 retrains do NOT dump features. One frozen set (from the chosen A5 winner) suffices for all 6 panels. This keeps seed kernels lean and avoids 6× duplication of ~200MB feature blobs.

### AD.4 — Seed variance dispatch policy under Path γ

- **Wave 2** dispatches all 3 seed retrains of **A5α** (`S1=seed42`, `S2=seed123`, `S3=seed456`) in parallel with the 6 ablations. A5α is the currently-deployed result and is the default headline; seed variance for A5α is the default reported variance.
- **Wave 4** is dispatched **only if** Wave 3 analysis shows `mAP(A5β) − mAP(A5α) > 0.10` (i.e. A5β beats A5α by more than the noise band). In that case Wave 4 ships 3 seed retrains of A5β under the same `{42, 123, 456}` seed set, replacing the reported variance row.
- If `mAP(A5β) − mAP(A5α) ≤ 0.10`, Wave 4 is SKIPPED. A5α seed variance is the final reported variance.

### AD.5 — Updated Section A kernel count and account distribution

Section A's "8 GPU kernels" headline is superseded by **9 GPU training kernels** for Wave 2:

- 6 ablation training kernels: A1, A2, A3, A4, **A5α**, **A5β** (was 5).
- 3 A5α seed retrains: S1, S2, S3.
- Total: **9** GPU training kernels.

**Eval folding**: per Section A as updated here, the held eval (for ablation rows) and the full AQE+rerank+TTA+fusion eval (for seed rows) are now folded into the **tail of each training kernel** rather than pushed as separate CPU eval kernels. The single exception is the **post-fusion score-level eval** for each row, which is appended as the final pre-fusion-dump cell using the frozen Stream-2 checkpoint mounted from `mrkdagods/mtmc-weights`. Therefore Wave 2 ships **0 standalone CPU eval kernels**.

**Proposed account distribution for the 9 training kernels** (respects Kaggle's 2-concurrent-GPU-per-account cap from `docs/kaggle-workflow.md`):

| exp_id | Account     | Concurrent slot | Notes                                                  |
|--------|-------------|-----------------|--------------------------------------------------------|
| A1     | gumfreddy   | Wave-2 slot A   | SGD short schedule, fastest                            |
| A2     | gumfreddy   | Wave-2 slot B   | Parallel with A1 on same account                       |
| A3     | mrkdagods   | Wave-2 slot A   | Different account → parallel with gumfreddy slot A    |
| A4     | mrkdagods   | Wave-2 slot B   | Parallel with A3                                       |
| A5α    | ali369      | Wave-2 slot A   | Different account → parallel with both above           |
| A5β    | ali369      | Wave-2 slot B   | Parallel with A5α (same account, both slots filled)    |
| S1 (seed42)  | gumfreddy | Wave-2 slot A (staggered) | Pushed after A1 completes; reuses gumfreddy slot |
| S2 (seed123) | mrkdagods | Wave-2 slot A (staggered) | Pushed after A3 completes                          |
| S3 (seed456) | ali369    | Wave-2 slot A (staggered) | Pushed after A5α completes                         |

**Account totals**: gumfreddy = 3 (A1, A2, S1), mrkdagods = 3 (A3, A4, S2), ali369 = 3 (A5α, A5β, S3). Balanced 3/3/3.

**Two-wave staggering**: the first 6 training kernels (A1–A4, A5α, A5β) all start simultaneously across the 3 accounts using both concurrent slots. As each ablation completes, the freed slot is filled by the seed retrain assigned to that account. With ~10–12h per kernel, total wall-clock = ~24h.

### AD.6 — Updated Wave 1 / Wave 2 / Wave 3 scopes

**Wave 1 (paper edits + panel script — local CPU)** — additions to original scope:

- Add a footnote at §3.2.2 marking that the recipe described there was tested under TWO variants (A5α actual deployed recipe with CE-LS + Triplet + CenterLoss + LLRD=0.75; A5β paper-described recipe with CE-LS + SupCon + LLRD=0.65) and that the headline numbers correspond to the empirically-selected winner per Addendum §AD.1. Leave a `\todo{insert A5 winner identity after Wave 3}` placeholder.
- Add an extra row to Table V (ablation) for A5β, labelled "Full (paper recipe variant)" with `\todo{mAP}` / `\todo{R1}` placeholders.
- Update Limitation 4 in §6 to mention: "An earlier draft of §3.2.2 described the Stream-1 training recipe imprecisely (specifically the metric loss and LLRD decay constant). The present version reports both the actually-deployed recipe (A5α) and the paper-described variant (A5β) and selects the empirically-stronger result as headline; the recipe-disclosure correction has been folded into Table II and §3.2.2."
- Section D panel-script work is unchanged in scope but now reads from `experiments/veri776-paper/<A5_winner>/features/` per §AD.3.
- Section E.5 path α vs β is no longer a blocking user decision — it is resolved as Path γ (both). Replace the E.5 "Coder MUST NOT proceed until user picks α or β" with: "Coder applies BOTH α and β as A5α and A5β. The headline path is chosen by Wave-3 measurement per §AD.1 headline-selection rule."

**Wave 2 (Kaggle GPU training)** — gates and scope additions:

- **NEW GATE (must pass before any clone)**: canonical-08 SHA-256 verification per §AD.2. If gate fails, abort Wave 2 and return failure report.
- Ships **9 training kernels** (was 8) distributed 3/3/3 per §AD.5.
- Ships **0 standalone eval kernels** (was 8). Eval is folded into each training kernel's tail.
- A5α and A5β training notebooks include the §AD.3 feature-dump tail cell. A1–A4 and S1–S3 do not.

**Wave 3 (aggregation + paper finalisation)** — scope additions:

- Compute headline winner per §AD.1 rule: pick `argmax(mAP)` between A5α and A5β; if within ±0.10 mAP report both side-by-side.
- Decide whether to dispatch Wave 4 per §AD.4 rule.
- Replace the `\todo{...}` placeholders in §3.2.2, Table V A5β row, Table II recipe, abstract, conclusion, and Limitation 4 with measured numbers.
- Populate `experiments/veri776-paper/REPORT.md` with the A5α-vs-A5β decision narrative.

**Wave 4 (NEW — conditional)** — dispatched only if `mAP(A5β) − mAP(A5α) > 0.10`:

- Ships 3 A5β seed retrains (`S1β=seed42`, `S2β=seed123`, `S3β=seed456`) distributed across gumfreddy / mrkdagods / ali369 one per account.
- Eval folded into kernel tail per Wave 2 convention.
- On completion, update reported variance row in paper Table III with A5β seeds (replacing A5α seeds).
- If skipped, Wave 3 is terminal.

### AD.7 — Updated artefact directory layout

`experiments/veri776-paper/` adds two top-level dirs:

```
experiments/veri776-paper/
  A1/   A2/   A3/   A4/
  A5alpha/
    features/stream1/{query,gallery}.npy
    features/stream2/{query,gallery}.npy
    features/index_map.json
    recipe.json  eval_results.json  best_mAP.pth  train_log.json  kernel_metadata.json
  A5beta/
    features/...                    # same structure as A5alpha
    recipe.json  eval_results.json  best_mAP.pth  train_log.json  kernel_metadata.json
  S1/  S2/  S3/                     # A5α seed retrains; no features/
  S1beta/  S2beta/  S3beta/         # Wave-4 conditional; created only if Wave 4 dispatches
  REPORT.md
  results.csv
  results.json
```

`results.csv` gains rows for `A5alpha`, `A5beta` and (conditionally) `S{1,2,3}beta_fused`. `results.json` `experiments[]` array length grows from 8 to 9 (Wave 2) and conditionally 12 (Wave 4).

### AD.8 — Net change summary vs original spec

| Original | Updated |
|----------|---------|
| 5 ablation rows | **6 ablation rows** (split A5 into A5α / A5β) |
| 8 GPU training kernels in Wave 2 | **9 GPU training kernels** in Wave 2 |
| 8 CPU eval kernels | **0 standalone CPU eval kernels** (folded into training-kernel tails) |
| Section D reads cached 14t features | Section D reads inline-dumped features from A5α/A5β tail cells |
| E.5 user-blocking path α/β choice | Resolved as Path γ (both); winner selected by measurement |
| 3-wave plan | **4-wave plan** (Wave 4 conditional on A5β beating A5α by >0.10 mAP) |
| Canonical 08 SHA not gated | **SHA-256 gate is a Wave-2 prerequisite** |

No earlier section of this spec is rewritten; this addendum is the authoritative override for the items listed in the table above.
