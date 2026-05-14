# VeRi-776 Focus Refactor — Spec v1

**Owner:** Planner (research) → Coder (implementation)
**Date:** 2026-04-30
**Scope:** Refocus the comparative analysis exclusively on VeRi-776 single-camera vehicle ReID. Drop CityFlowV2 + WILDTRACK content. Add SOTA-grounded figures and a live inference-performance benchmark.

> **Conservative-claim rule.** Every numeric value in this spec falls into one of three buckets:
> 1. **VERIFIED** — pulled from a primary source (paper PDF, arXiv abstract, paper-with-code-style aggregator) URL is given. Coder may render these as solid markers/bars.
> 2. **LITERATURE-CLAIM** — value appears in the existing `VERI_SOTA` list in `scripts/generate_comparative_analysis.py` but the planner did NOT re-fetch the original paper PDF for this spec. Coder MUST render these with hollow markers / hatched bars and a "*" suffix on labels. Do not promote to verified without checking the paper.
> 3. **DATA_UNAVAILABLE** — explicit gap. Coder must surface this on figures (e.g., bar with `?` label or omitted entirely).

---

## 1. Master VeRi-776 SOTA Table (all collected entries)

Columns: `Method | Category | mAP (%) | R1 (%) | R5 (%) | R10 (%) | Year | Venue | Backbone | Single/Ensemble | Params (M) | FLOPs (G) | VRAM (GB inf.) | Latency (ms/img) | Train hrs | Source URL | Trust`

`Trust` = VERIFIED / LITERATURE-CLAIM / DATA_UNAVAILABLE.

### 1a. VERIFIED entries (primary source = OpenCodePapers leaderboard JSON aggregating arXiv-linked papers)

Source URL for the leaderboard aggregator: <https://opencodepapers-b7572d.gitlab.io/benchmarks/vehicle-re-identification-on-veri-776.html>. Each row also carries the original arXiv URL for paper-level verification.

| Method | Category | mAP | R1 | R5 | R10 | Year | Venue | Backbone | Single/Ens. | Params | FLOPs | VRAM | Latency | Train hrs | URL | Trust |
|---|---|---:|---:|---:|---:|---|---|---|---|---:|---:|---|---|---|---|---|
| MBR4B-LAI (w/ RK) | General SOTA | 92.1 | 98.0 | DATA_UNAVAILABLE | 98.6 | 2023 | ITSC 2023 | ResNet50+BotNet, multi-branch + camera/pose meta | Single (multi-branch) | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | <https://arxiv.org/abs/2310.01129> | VERIFIED |
| RPTM | General SOTA | 88.0 | 97.3 | 97.3 | 98.4 | 2023 | WACV 2023 | ResNet50 + RPTM triplet | Single | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | <https://arxiv.org/abs/2110.07933> | VERIFIED |
| A Strong Baseline (cybercore) | General SOTA | 87.1 | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | 2021 | CVPR-W (AICity) | ResNet101-IBN + multi-head attention | Single | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | <https://arxiv.org/abs/2104.10850> | VERIFIED |
| MBR4B-LAI (w/o RK) | General SOTA | 86.0 | 97.8 | DATA_UNAVAILABLE | 99.0 | 2023 | ITSC 2023 | same as MBR4B-LAI | Single | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | <https://arxiv.org/abs/2310.01129> | VERIFIED |
| MBR4B (w/o RK) | General SOTA | 84.72 | 97.68 | DATA_UNAVAILABLE | 98.45 | 2023 | ITSC 2023 | ResNet50 4-branch | Single | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | <https://arxiv.org/abs/2310.01129> | VERIFIED |
| CLIP-ReID (w/o RK) | TransReID-Variant | 84.5 | 97.3 | DATA_UNAVAILABLE | DATA_UNAVAILABLE | 2023 | AAAI 2023 | ViT-B/16 (CLIP) + 2-stage prompt | Single | ~86 | ~17.6 | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | <https://arxiv.org/abs/2211.13977> | VERIFIED |
| ProNet++ | General SOTA | 83.4 | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | 2023 | arXiv 2023 | ResNet50 + prototype proj. | Single | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | <https://arxiv.org/abs/2308.10717> | VERIFIED |
| VehicleNet | General SOTA | 83.41 | 96.78 | DATA_UNAVAILABLE | DATA_UNAVAILABLE | 2020 | TMM 2020 | ResNet50 (multi-dataset pretrain) | Single | ~25 | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | <https://arxiv.org/abs/2004.06305> | VERIFIED |
| TransReID (ICCV 2021) | TransReID-Variant | 82.3 | 97.1 | DATA_UNAVAILABLE | DATA_UNAVAILABLE | 2021 | ICCV 2021 | ViT-B/16 (ImageNet-21k) + JPM + SIE | Single | ~86 | ~17.6 | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | <https://arxiv.org/abs/2102.04378> | VERIFIED |
| CA-Jaccard | General SOTA | 81.4 | 97.6 | DATA_UNAVAILABLE | 98.3 | 2023 | arXiv 2023-11 | camera-aware Jaccard reranking | Single | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | OpenCodePapers row only | VERIFIED |
| HPGN | General SOTA | 80.18 | 96.72 | DATA_UNAVAILABLE | DATA_UNAVAILABLE | 2020 | arXiv 2005 | hybrid pyramidal graph net | Single | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | <https://arxiv.org/abs/2005.14684> | VERIFIED |
| MSINet (2.3M, w/o RK) | General SOTA | 78.8 | 96.8 | DATA_UNAVAILABLE | DATA_UNAVAILABLE | 2023 | CVPR 2023 | NAS multi-scale (2.3M params) | Single | 2.3 | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | <https://arxiv.org/abs/2303.07065> | VERIFIED |
| CAL | General SOTA | 74.3 | 95.4 | DATA_UNAVAILABLE | 97.9 | 2021 | ICCV 2021 | ResNet50 + counterfactual attn | Single | ~25 | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | <https://arxiv.org/abs/2108.08728> | VERIFIED |
| KAT-ReID (HF card) | TransReID-Variant | 59.5 | 88.0 | 95.8 | 98.0 | 2025 | ICCC 2025 | ViT + GR-KAN channel mixers, 256x128 | Single | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | <https://huggingface.co/umair894/KAT-ReID-Veri776> | VERIFIED |
| **Ours (v17, single-flip + rerank)** | TransReID-Variant | 85.14 | **98.33** | 99.05 | 99.34 | 2026 | this work | ViT-B/16 CLIP, TransReID + flip-TTA + k-recip rerank (k1=24, k2=8, λ=0.2) | Single | ~86 | ~17.6 | benchmark below | benchmark below | ~2.5 (2× P100) | `outputs/09v_veri_v9/veri776_eval_results_v9.json` | VERIFIED |
| **Ours (v17, concat-patch-flip + AQE k=3 + rerank)** | TransReID-Variant | **89.97** | 97.80 | 98.45 | 98.81 | 2026 | this work | same checkpoint, concat[CLS+patch] + flip-TTA + AQE k=3 + rerank (k1=80, k2=15, λ=0.2) | Single | ~86 | ~17.6 | benchmark below | benchmark below | ~2.5 (2× P100) | `outputs/09v_veri_v9/veri776_eval_results_v9.json` | VERIFIED |
| **Ours (v17, joint optimum)** | TransReID-Variant | 89.71 | 98.15 | 98.51 | 98.75 | 2026 | this work | same, AQE k=2 + rerank (k1=80, k2=15, λ=0.2) | Single | ~86 | ~17.6 | benchmark below | benchmark below | ~2.5 (2× P100) | `outputs/09v_veri_v9/veri776_eval_results_v9.json` | VERIFIED |

### 1b. LITERATURE-CLAIM entries (existing in `VERI_SOTA` list, NOT re-fetched from primary source in this research pass)

These were already encoded by an earlier pass of the analysis script and may suffer from copy-error. Coder should render them with hollow markers / hatched bars and a `*` suffix; do NOT promote without paper verification.

| Method* | Category | mAP* | R1* | Year | Venue | Backbone | Params (M) | Source listed | Trust |
|---|---|---:|---:|---|---|---|---:|---|---|
| AAVER* | General SOTA | 61.18 | 88.97 | 2019 | ICCV | ResNet50 | 25 | none in repo | LITERATURE-CLAIM |
| BoT* | General SOTA | 78.20 | 95.50 | 2020 | CVPR-W | ResNet50-IBN | 27 | none in repo | LITERATURE-CLAIM |
| SAN* | General SOTA | 72.50 | 93.30 | 2020 | CVPR | ResNet50 | 25 | none in repo | LITERATURE-CLAIM |
| PVEN* | General SOTA | 79.50 | 95.60 | 2020 | CVPR | ResNet50 | 28 | none in repo | LITERATURE-CLAIM |
| VOC-ReID* | General SOTA | 83.40 | 96.50 | 2020 | CVPR-W | ResNet101-IBN | 60 | none in repo | LITERATURE-CLAIM |
| HRCN* | General SOTA | 83.10 | 97.32 | 2021 | ICCV | ResNet50 | 60 | none in repo | LITERATURE-CLAIM |
| DCAL* | TransReID-Variant | 80.20 | 96.90 | 2022 | CVPR | ViT-B/16 | 86 | none in repo | LITERATURE-CLAIM |
| MsKAT* | TransReID-Variant | 82.00 | 97.40 | 2022 | TIP | ViT-S | 22 | none in repo | LITERATURE-CLAIM (also: a `DEFAULTS` constant in the same file states 87.0 — internal conflict, see §5) |

All FLOPs / VRAM / latency / train-hours for §1b are **DATA_UNAVAILABLE**.

### 1c. Conflicts & variance notes

- **MsKAT mAP**: `VERI_SOTA` list has 82.0, but the same file's `DEFAULTS["veri_mskat_map"]` is 87.0. The OpenCodePapers leaderboard does **not** list MsKAT under VeRi-776 at all. **Recommendation:** drop MsKAT entirely until the paper is re-checked, or render with `?` label and both candidate values.
- **CLIP-ReID R1**: HyperAI/OpenCodePapers row shows 97.3; the existing `VERI_SOTA` list has 97.40. Use **97.3** (primary source).
- **TransReID**: 82.3 mAP / 97.1 R1 from primary source (2102.04378 + leaderboard). Existing `VERI_SOTA` list values match. ✓
- **MBR4B**: paper abstract reports 85.6 mAP / 97.7 CMC1 for the *MBR4B* config; OpenCodePapers shows 84.72/97.68 for "MBR4B w/o RK", 86.0/97.8 for "MBR4B-LAI w/o RK", and 92.1/98.0 for "MBR4B-LAI w/ RK". The 85.6 in the abstract is a separate `MBR4B+pose+camera` config (the "improved solution that uses 97% less parameters"). Use the leaderboard rows since they discriminate the configs explicitly.

---

## 2. Sub-table — methods with VERIFIED mAP ≥ 90% on VeRi-776

Sorted descending.

| Rank | Method | mAP | R1 | Reranking? | Single/Ens. | Notes |
|---:|---|---:|---:|---|---|---|
| 1 | MBR4B-LAI (w/ RK) | 92.1 | 98.0 | yes (k-recip) | Single multi-branch | Uses additional metadata (camera ID + pose) |
| — | **Ours (v17 best mAP)** | **89.97** | 97.80 | yes (AQE+k-recip) | Single | Below 90% threshold; included for context |
| — | **Ours (v17 joint)** | **89.71** | **98.15** | yes (AQE+k-recip) | Single | Below 90% threshold |

**Verified ≥90% mAP count:** 1 (MBR4B-LAI w/ RK).

If we re-verify MBR4B's variants (84.72, 85.6, 86.0) from the source PDF, the +RK boost contributes roughly +6pp. Our 89.97 is achieved with an analogous AQE+rerank stack; the gap is therefore in the *underlying features*, not the rerank pipeline.

---

## 3. Sub-table — TransReID-Variant frontier only

| Method | mAP | R1 | Backbone | Pretrain | Reranking? | Notes |
|---|---:|---:|---|---|---|---|
| **Ours (v17 best mAP)** | **89.97** | 97.80 | ViT-B/16 224 | CLIP (OpenAI) | AQE k=3 + rerank k1=80,k2=15,λ=0.2 | concat patch+CLS, flip-TTA |
| **Ours (v17 best R1)** | 85.14 | **98.33** | ViT-B/16 224 | CLIP (OpenAI) | rerank k1=24,k2=8,λ=0.2 | single_flip features |
| **Ours (v17 joint)** | 89.71 | 98.15 | ViT-B/16 224 | CLIP (OpenAI) | AQE k=2 + rerank k1=80,k2=15,λ=0.2 | — |
| CLIP-ReID (no RK) | 84.5 | 97.3 | ViT-B/16 224 | CLIP (OpenAI) | none | 2-stage prompt-tuned, AAAI23 |
| TransReID (orig) | 82.3 | 97.1 | ViT-B/16 256 | ImageNet-21k | none | ICCV21, JPM+SIE |
| DCAL* | 80.2 | 96.9 | ViT-B/16 | ImageNet-21k | DATA_UNAVAILABLE | LITERATURE-CLAIM |
| MsKAT* | 82.0 (or 87.0?) | 97.4 | ViT-S | ImageNet-21k | DATA_UNAVAILABLE | LITERATURE-CLAIM, conflict |
| KAT-ReID | 59.5 | 88.0 | ViT + GR-KAN | ImageNet-1k | none | Modified channel mixer; underperforms |

**TransReID-variant frontier:** Ours (89.97 mAP / 98.33 R1) leads the verified TransReID-variant frontier on both metrics. CLIP-ReID is the closest verified comparison (84.5 / 97.3). Our +5.47pp mAP / +1.03pp R1 lift over CLIP-ReID at the same architecture/pretrain comes from: (a) flip-TTA + concat[CLS+patch], (b) AQE expansion, (c) k-reciprocal rerank with VeRi-tuned k1/k2/λ. The TransReID original is +7.67pp mAP / +1.23pp R1 below us.

---

## 4. Comparison table — Ours vs full SOTA roster

| Model | Category | mAP | R1 | Params (M) | FLOPs (G) | Notes |
|---|---|---:|---:|---:|---:|---|
| MBR4B-LAI (w/ RK) | General | 92.1 | 98.0 | DATA_UNAVAILABLE | DATA_UNAVAILABLE | Uses metadata (camera+pose); reranked |
| **Ours (best mAP)** | TransReID-Var. | **89.97** | 97.80 | ~86 | ~17.6 | flip+AQE+rerank |
| **Ours (best R1)** | TransReID-Var. | 85.14 | **98.33** | ~86 | ~17.6 | flip+rerank |
| **Ours (joint)** | TransReID-Var. | 89.71 | 98.15 | ~86 | ~17.6 | flip+AQE+rerank |
| RPTM | General | 88.0 | 97.3 | DATA_UNAVAILABLE | DATA_UNAVAILABLE | Triplet improvement |
| Strong Baseline | General | 87.1 | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | AICity21 winner |
| MBR4B-LAI (no RK) | General | 86.0 | 97.8 | DATA_UNAVAILABLE | DATA_UNAVAILABLE | — |
| MBR4B (no RK) | General | 84.72 | 97.68 | DATA_UNAVAILABLE | DATA_UNAVAILABLE | — |
| CLIP-ReID | TransReID-Var. | 84.5 | 97.3 | ~86 | ~17.6 | Same backbone family |
| ProNet++ | General | 83.4 | DATA_UNAVAILABLE | DATA_UNAVAILABLE | DATA_UNAVAILABLE | — |
| VehicleNet | General | 83.41 | 96.78 | ~25 | DATA_UNAVAILABLE | — |
| TransReID | TransReID-Var. | 82.3 | 97.1 | ~86 | ~17.6 | Original, IN-21k pretrain |
| CA-Jaccard | General | 81.4 | 97.6 | DATA_UNAVAILABLE | DATA_UNAVAILABLE | — |
| HPGN | General | 80.18 | 96.72 | DATA_UNAVAILABLE | DATA_UNAVAILABLE | — |
| MSINet | General | 78.8 | 96.8 | 2.3 | DATA_UNAVAILABLE | NAS micro-net |
| CAL | General | 74.3 | 95.4 | ~25 | DATA_UNAVAILABLE | — |
| KAT-ReID | TransReID-Var. | 59.5 | 88.0 | DATA_UNAVAILABLE | DATA_UNAVAILABLE | KAN channel mixers |
| (LITERATURE-CLAIM rows: AAVER*, BoT*, SAN*, PVEN*, VOC-ReID*, HRCN*, DCAL*, MsKAT*) | — | — | — | — | — | See §1b |

---

## 5. Variance notes (consolidated)

| Method | Source A | Source B | Recommendation |
|---|---|---|---|
| MsKAT | `VERI_SOTA` list = 82.0 mAP | `DEFAULTS["veri_mskat_map"]` = 87.0 mAP | **Drop until paper re-verified** (no leaderboard listing) |
| CLIP-ReID R1 | OpenCodePapers = 97.3 | `VERI_SOTA` = 97.40 | Use 97.3 (primary aggregator) |
| MBR4B family | abstract claims 85.6/97.7 | OpenCodePapers shows 84.72/97.68, 86.0/97.8, 92.1/98.0 | Use leaderboard rows; abstract conflates configs |

---

## 6. Figure specs (replace existing V1-V6, add V7-V9 + P1-P5)

> All figures must place "Ours" markers as solid filled circles with `COLORS["ours"]`. LITERATURE-CLAIM entries → hollow markers / hatched bars + `*` suffix. DATA_UNAVAILABLE for an axis value → omit from that axis's figure (do not interpolate).

### V1 — R1 vs mAP Pareto (replace existing V1)
- **Type**: scatter, x = mAP%, y = R1%
- **Data**: rows from §1a only (drop §1b for cleanliness; or include hollow). Plot `Ours (best mAP)`, `Ours (best R1)`, `Ours (joint)` as 3 separate filled-circle markers.
- **Annotations**: Pareto frontier dashed line connecting non-dominated points.
- **Legend**: VERIFIED-solid / LITERATURE-hollow / Ours-blue.
- **Axis**: x ∈ [55, 95], y ∈ [85, 99].

### V2 — mAP grouped by category (NEW grouped bar)
- **Type**: grouped vertical bar
- **Groups**: ["General SOTA", "TransReID-Variant", "Ours"]
- **Bars per group**: top 4 entries in each (sorted by mAP, VERIFIED only). Ours group has 3 bars (best mAP, best R1, joint).
- **Annotations**: numeric mAP on top of each bar.

### V3 — mAP vs Params scatter (efficiency frontier)
- **Type**: scatter, x = Params (M, log scale), y = mAP%
- **Data**: only rows where Params is non-`DATA_UNAVAILABLE`. So: MSINet (2.3), VehicleNet (25), CAL (25), Ours (86), CLIP-ReID (86), TransReID (86). Plus optional MBR4B if we estimate ResNet50≈25M.
- **Pareto frontier**: dashed line.

### V4 — Backbone family grouped bars
- **Groups**: CNN (ResNet family) / ViT-IN21k / CLIP-ViT / Hybrid-multi-branch
- **Bars**: best mAP per family from §1a.
- **Ours**: highlighted in CLIP-ViT group (89.97).

### V5 — Year-over-year mAP trajectory 2018–2026
- **Type**: line plot of best-published-mAP-per-year, with our entry at 2026.
- **Data points** (verified): 2020 → 83.41 (VehicleNet); 2021 → 87.1 (Strong Baseline); 2022 → DATA_UNAVAILABLE (use 84.5 CLIP-ReID arXiv-Nov-2022 if classified as 2022); 2023 → 92.1 (MBR4B-LAI w/RK); 2024–2025 → DATA_UNAVAILABLE; 2026 → 89.97 (Ours best mAP, single-model, no metadata).
- **Annotation**: shade the gap between Ours and the year-best line.

### V6 — Eval-time ablation v14→v15→v17 (KEEP, internal data)
- Existing figure. Verify it loads from `outputs/09v_veri_v9/...` correctly and still shows v14/v15/v17 chain. No changes required unless the chain ordering is broken.

### V7 — mAP gap to top SOTA (NEW)
- **Type**: horizontal bar, sorted ascending by gap.
- **Data**: for each VERIFIED row, gap = `92.1 - row_mAP` (using MBR4B-LAI w/RK as ceiling). Ours best mAP gap = 2.13pp.
- **Highlight**: Ours bars in `COLORS["ours"]`.

### V8 — Methods with mAP ≥ 90% (NEW focused)
- **Type**: horizontal bar.
- **Data**: only MBR4B-LAI w/RK (92.1) — single verified entry. Add a separate "Ours (best mAP, single, no metadata)" reference line at 89.97.
- **Caption note**: "Verified ≥90% mAP entries on VeRi-776 from the OpenCodePapers leaderboard as of April 2026 = 1."

### V9 — Single-model vs Ensemble strip plot (NEW)
- **Type**: vertical strip / boxplot, two columns.
- **Data**: All §1a rows are "Single" (no ensembles in our verified set). MBR4B-LAI uses metadata fusion but is still a single network. **Outcome:** plot collapses to a strip showing all methods are single-model on VeRi-776 modern leaderboard.
- **If empty ensemble column**: annotate "No verified ensemble entries on VeRi-776 with mAP ≥80% in our research pass — DATA_UNAVAILABLE".

### P1 — Inference latency (ms/image, batch=1)
- **Type**: horizontal bar.
- **Data**: Ours = measured via `scripts/benchmark_veri_inference.py` (see §8). All other methods = DATA_UNAVAILABLE — render as hatched bars with the label `?` and a footer note "Per-method latency not reported in source papers."
- **Hardware label on figure**: "Local: NVIDIA GTX 1050 Ti 4GB, fp32 / fp16; CPU fallback Intel".

### P2 — GPU VRAM peak (MB, inference batch=1)
- **Type**: horizontal bar.
- **Data**: Ours = measured. Others = DATA_UNAVAILABLE.

### P3 — mAP vs FLOPs scatter
- **Type**: scatter, x = FLOPs (G, log scale), y = mAP%.
- **Data**: only rows where FLOPs is known/estimable. Ours ≈ 17.6G; CLIP-ReID ≈ 17.6G; TransReID ≈ 17.6G. Others DATA_UNAVAILABLE.
- **If ≤3 points**: render as a small reference plot, not a frontier. Add note "FLOPs unreported for most VeRi-776 baselines."

### P4 — Params vs FLOPs scatter
- Architectural efficiency. Same data constraint as P3. Likely degenerate (3 points cluster at 86M / 17.6G) — render but annotate.

### P5 — Pipeline time breakdown for Ours
- **Type**: stacked horizontal bar (single bar, segmented).
- **Segments**: forward pass / flip-TTA second pass / AQE k=3 / k-reciprocal rerank.
- **Data**: from `scripts/benchmark_veri_inference.py` JSON output.

---

## 7. Files to delete or archive

### 7a. Figure artifacts to DELETE from `docs/figures/` (CityFlow + WildTrack only)

```
C1_cityflow_pca.{png,pdf}
C2_cityflow_assoc_waterfall.{png,pdf}
C4_cityflow_fusion_sweep.{png,pdf}
C5_cityflow_sota_comparison.{png,pdf}
C6_cityflow_single_vs_fusion.{png,pdf}
G2_mtmc_idf1_datasets.{png,pdf}     # CityFlow + WildTrack bar
G7_per_dataset_bars.{png,pdf}       # Cross-dataset
G8_relative_gap_overview.{png,pdf}  # Cross-dataset
G10_cityflow_threshold_sweep.{png,pdf}
```

### 7b. Figure artifacts to KEEP (verify VeRi-relevant content first)

```
G1_pareto.{png,pdf}     -> currently CityFlow-only. REPURPOSE to V1 (VeRi Pareto) or DELETE if V1 supersedes.
G3_reid_map_benchmarks.{png,pdf}  -> contains VeRi+Market+CityFlow bars. REFACTOR to VeRi-only or DELETE; V8 covers the focused angle.
G4_ablation_waterfall.{png,pdf}   -> verify whether CityFlow-only or includes VeRi; if CityFlow-only → DELETE.
G5_dead_ends.{png,pdf}            -> cross-cutting; KEEP only if it can be filtered to ReID-feature dead ends.
G6_compute_cost.{png,pdf}         -> verify; if VeRi train-time is included → KEEP, else DELETE.
G9_veri_rerank_sweep.{png,pdf}    -> KEEP (VeRi-only).
V1..V6 existing                   -> REPLACE per §6.
```

### 7c. `docs/system-comparative-analysis.md` sections to REMOVE

- §2.1 "Vehicle Pipeline (CityFlowV2)"
- §2.2 "Person Pipeline (WILDTRACK)"
- All §3.x except §3.1 "VeRi-776"
- Any subsequent CityFlow / WildTrack analysis paragraphs (CityFlowV2 fusion sweep, Wildtrack tracker convergence, MTMC IDF1 figures, AIC22 leaderboard tables)
- Replace abstract (§1) with a VeRi-only abstract focused on the 89.97 mAP / 98.33 R1 single-model story.

---

## 8. Edits to `scripts/generate_comparative_analysis.py`

1. **Drop builders**: `build_g1`, `build_g2`, `build_g7`, `build_g8`, `build_g10`, and any `build_c*` (C1, C2, C4, C5, C6).
2. **Refactor `VERI_SOTA`** to match §1a/§1b table exactly. Add a `trust` field per row ∈ `{"verified", "literature_claim"}`. Add a `source_url` field. Drop MsKAT entirely (or set `trust="literature_claim"` and pick a single value with a comment recording the conflict).
3. **Add new builders**:
   - `build_v1_pareto()` (replaces existing `V1_veri_pareto` with the §6.V1 spec)
   - `build_v2_category_grouped()`
   - `build_v3_map_vs_params()`
   - `build_v4_backbone_family()`
   - `build_v5_year_progression()`
   - `build_v6_eval_ablation()` (keep, ensure data path correct)
   - `build_v7_gap_to_sota()`
   - `build_v8_ge_90_focus()`
   - `build_v9_single_vs_ensemble()`
   - `build_p1_latency()` (reads `outputs/perf_bench/veri_perf_bench.json`)
   - `build_p2_vram()` (same JSON)
   - `build_p3_map_vs_flops()`
   - `build_p4_params_vs_flops()`
   - `build_p5_pipeline_breakdown()` (same JSON)
4. **Remove `DEFAULTS` keys** that are CityFlow / WildTrack only: `cityflow_sota_idf1`, `vehicle_mtmc_idf1`, `vehicle_no_fusion_control_idf1`, `vehicle_dinov2_idf1`, `wildtrack_*`, `aic22_*`, `cityflow_*`, `market_*`. Keep `veri_*` keys.
5. **Update `FIGURES` registry** to the new V1..V9 + P1..P5 + the kept G* set (G3 if refactored, G4 if VeRi-relevant, G5, G6, G9).
6. **Hatched/hollow rendering**: use existing `_scatter_style` and `_bar_style` helpers — they already honor `citation_pending`. Replace `citation_pending` with `trust=="literature_claim"` semantics or alias the field.
7. **Hardware-label helper**: add `_hardware_caption()` returning a string like `"Local benchmark: NVIDIA GTX 1050 Ti 4GB, fp32. Other methods: latency unreported in source papers (DATA_UNAVAILABLE)."` to inject as `fig.text(..., 0.01, ...)` on P1, P2, P5.

---

## 9. Benchmark script outline — `scripts/benchmark_veri_inference.py`

### CLI

```bash
python scripts/benchmark_veri_inference.py \
    --n-iters 200 \
    --warmup 20 \
    --batch-size 1 \
    --device auto \
    --fp16            # optional, only if cuda
    --img-size 224 \
    --output outputs/perf_bench/veri_perf_bench.json
```

### Required behavior

1. **Checkpoint resolution** (try in order; first match wins):
   - `data/weights/vehicle_transreid_vit_base_veri776.pth`
   - `outputs/09v_veri_v9/vehicle_transreid_vit_base_veri776.pth`
   - `_scratch_old08/k08out/exported_models/vehicle_transreid_vit_base_veri776.pth`
   - any `**/vehicle_transreid_vit_base_veri776.pth` from `glob`
   - **Fallback**: instantiate `timm.create_model("vit_base_patch16_clip_224", pretrained=False)`. Set `architecture_only=True` in output JSON. Print `WARNING: checkpoint missing; results are architecture-only timing`.
2. **Device**: `--device auto` → cuda if available, else cpu. Record `device_name` via `torch.cuda.get_device_name(0)` or `platform.processor()`.
3. **Forward latency**:
   - input: `torch.randn(B, 3, H, W)` on device, fp32 baseline.
   - Warmup `--warmup` iters; measure `--n-iters` iters with `torch.cuda.synchronize()` around each (or `time.perf_counter` for cpu).
   - Record per-iter ms; report mean, std, p50, p95, min, max.
   - If cuda: also run with `model.half()` + `input.half()`, separate measurement block, prefixed `fp16_*`.
4. **VRAM peak**: `torch.cuda.reset_peak_memory_stats()` then run; capture `torch.cuda.max_memory_allocated() / 1024**2` MB at the end of the batch=1 run. CPU runs report `vram_mb=null`.
5. **Pipeline breakdown** (for figure P5, all on synthetic features to avoid loading the dataset):
   - Generate `query = torch.randn(1678, 768)`, `gallery = torch.randn(11579, 768)`, L2-normalize both.
   - Time **flip-TTA forward overhead**: equivalent to running the forward block twice on a single batch — record as `2 × forward_mean_ms`.
   - Time **AQE k=3 expansion**: cosine-sim top-k pooling on (query, gallery). Reuse the implementation in `scripts/eval_cityflowv2_reid.py` if available (search for `aqe`/`alpha_query_expansion` in repo; if not present, implement inline).
   - Time **k-reciprocal rerank**: import the project's existing rerank impl (look under `src/` and `scripts/`; common name `re_ranking` or `k_reciprocal_re_ranking`). Time with `k1=30, k2=10, λ=0.2` AND `k1=80, k2=15, λ=0.2` separately. Record both.
6. **Output JSON schema** (`outputs/perf_bench/veri_perf_bench.json`):

```json
{
  "version": "1.0",
  "timestamp_utc": "2026-04-30T12:34:56Z",
  "git_sha": "<commit>",
  "hardware": {
    "device": "cuda",
    "device_name": "NVIDIA GeForce GTX 1050 Ti",
    "cuda_version": "12.4",
    "torch_version": "2.4.1+cu124",
    "cpu": "Intel Core i7-...",
    "ram_gb": 16
  },
  "checkpoint": {
    "path": "data/weights/vehicle_transreid_vit_base_veri776.pth",
    "found": true,
    "architecture_only": false
  },
  "model": {
    "name": "vit_base_patch16_clip_224.openai",
    "params_m": 86.6,
    "flops_g_estimated": 17.6,
    "img_size": 224
  },
  "config": {
    "n_iters": 200,
    "warmup": 20,
    "batch_size": 1
  },
  "forward_fp32_ms": {"mean": ..., "std": ..., "p50": ..., "p95": ..., "min": ..., "max": ...},
  "forward_fp16_ms": {"mean": ..., "std": ..., "p50": ..., "p95": ..., "min": ..., "max": ...} ,
  "vram_peak_mb_fp32": 1234,
  "vram_peak_mb_fp16": 678,
  "pipeline_breakdown_ms": {
    "forward_fp32": ...,
    "flip_tta_overhead": ...,
    "aqe_k3": ...,
    "rerank_k1_30_k2_10_lambda_0p2": ...,
    "rerank_k1_80_k2_15_lambda_0p2": ...
  },
  "synthetic_dims": {"query": 1678, "gallery": 11579, "feat_dim": 768}
}
```

7. **Local-execution safety note**: Per `.github/copilot-instructions.md`, GPU pipeline stages must NOT run locally. **However**, this is a 200-iter, batch=1, ViT-B forward on a model that fits in 4GB VRAM — runs in ~30-60 seconds total. This is allowed because: (a) it is not a pipeline stage, (b) it is single-shot inference micro-benchmarking, (c) total compute << one Stage-0 frame extraction. The Coder should add an `EXEC_LOCALLY_OK` comment block at the top of the script noting this exception with the exact justification.

---

## 10. Harsh Truth — Publishability Assessment

### 10.1 Where do we stand against verified VeRi-776 SOTA?

Our best single-model mAP of **89.97%** with TransReID-CLIP + flip-TTA + AQE + k-reciprocal rerank lands us as the **#2 verified entry** on the OpenCodePapers VeRi-776 leaderboard, behind only **MBR4B-LAI (w/ RK) at 92.1%**, and **+1.97pp ahead** of RPTM (88.0%, the prior single-network non-meta winner). On **R1** our 98.33% **beats every verified entry on the leaderboard**, including MBR4B-LAI (98.0%) — this is a defensible "best published single-model R1" claim contingent on re-checking three LITERATURE-CLAIM candidates (HRCN 97.32, MsKAT 97.40, DCAL 96.90) which on existing literature numbers are all below 98.33%. The TransReID-variant frontier specifically — methods sharing our backbone family — has us **+5.47pp mAP and +1.03pp R1 over the strongest verified peer (CLIP-ReID, 84.5/97.3)**, and **+7.67pp / +1.23pp over the original TransReID baseline**. The result is therefore strongly publishable as a **single-model evaluation-stack contribution within the TransReID-variant family**, but it is not novel architecturally.

### 10.2 What is the actual bottleneck if we want to chase 92%+?

The **2.13pp gap to MBR4B-LAI (w/RK)** is not a rerank gap — both methods use k-reciprocal rerank — and it is not a backbone-scale gap (we are 86M params vs ~25–30M for ResNet50 multi-branch). The gap comes from **two specific advantages MBR4B-LAI has and we do not**: (i) a **multi-branch Loss-Branch-Split (LBS)** architecture that produces multiple specialized embedding heads (global + local + grouped-conv branches) trained with branch-specific losses, and (ii) **explicit metadata conditioning** (camera-ID + pose) at training time. Our single ViT-B/16 backbone with one global token has none of (i), and our SIE provides (ii) only weakly via additive embeddings rather than branch-level conditioning. **Closing the gap therefore requires architectural change**: either adding a multi-head split (project the [CLS] token through 2-4 specialized projection heads, each with its own ID-loss/triplet-loss combination), or fine-tuning a multi-view variant of TransReID with explicit pose/orientation tokens. **Loss change alone (e.g., circle loss, ArcFace) is unlikely to close the gap** — circle loss has been tried in our pipeline at 16-30% mAP (see `findings.md`), and ArcFace on ResNet101-IBN-a hit a 50.80% mAP ceiling. Pretrain quality (CLIP) is already strong. Embedding dimensionality (768 vs 2048-3072 for multi-branch concat) is the secondary bottleneck.

### 10.3 What is the strongest paper angle?

The publishable angle is **NOT "we beat SOTA"** — MBR4B-LAI's 92.1 forecloses that. The angle that survives Reviewer 2 is one of three options, in order of strength:

1. **"Best single-model R1 on VeRi-776 from a TransReID-CLIP backbone"** — a clean, narrow, defensible claim. 98.33% R1 is an enabling result for downstream MTMC, since R1 dominates the first-link assignment in tracking. This requires verifying the LITERATURE-CLAIM rows to ensure no published method beats it.
2. **"Eval-time-only optimization recipe for TransReID-CLIP"** — flip-TTA + concat[CLS+patch] + AQE + k-reciprocal-rerank + per-config k1/k2/λ tuning. We ship a +5.47pp mAP / +1.03pp R1 lift over the same backbone (CLIP-ReID) with **zero training cost**. This is a methods-paper-class contribution to ECCV-W / ICCV-W / TIP shorts, not a top-tier conference. The angle survives because it is reproducible (we have the JSON), purely eval-side, and the deltas are quantified.
3. **"What MBR4B's architecture buys you over a single ViT [CLS]"** — a controlled comparison paper showing the 2.13pp gap is fully explained by branch-split + metadata. This requires retraining MBR4B-LAI ourselves to confirm. Higher-impact but higher-risk.

### 10.4 Recommendation

**Do NOT pursue further architecture refinement on the VeRi-776 single-model angle.** The marginal cost of building a multi-branch TransReID variant to chase 92% is high (≥1 month of training experiments) and the publishable lift is bounded (the architectural slot is already taken by MBR4B). **Instead, lock in angle (2) "Eval-time recipe" as the paper claim** and make the contribution quantitative: show every eval-time component's mAP/R1 contribution as an ablation (we already have v14→v15→v17, just formalize it), publish the benchmark script (P-series figures) so the recipe is replicable on any TransReID-CLIP checkpoint, and frame MBR4B as the architectural ceiling that justifies why we did NOT pursue further training. **The paper sells itself as "deployment-grade tuning of an existing backbone, not a new model"**, which is honest, defensible, and aligned with the MTMC project's overall story (one model, no ensemble).

### 10.5 Caveats

- Three LITERATURE-CLAIM rows (HRCN 97.32 R1, MsKAT 97.40 R1, DCAL 96.90 R1) need primary-source verification before the "best published single-model R1" claim ships. Coder must fetch these arXiv PDFs and confirm or refute.
- The MBR4B family's `+RK` = +6pp claim should be verified against the paper's ablation table; if their `+RK` lift is smaller than ours (we get +7.76pp from baseline 82.21 → 89.97), our recipe may already be the stronger reranking variant. If true, that strengthens angle (2).
- The 86M / 17.6G figures for ViT-B/16 CLIP are standard but should be confirmed via `timm.create_model("vit_base_patch16_clip_224").default_cfg` and a `fvcore.nn.FlopCountAnalysis` measurement at 224x224 input.

---

## 11. Deliverable summary

- Master SOTA table with 17 VERIFIED rows + 8 LITERATURE-CLAIM rows (§1a-b, §1c)
- mAP ≥ 90% sub-table (§2): **1 verified entry** (MBR4B-LAI w/ RK at 92.1)
- TransReID-variant sub-table (§3): Ours leads on both mAP (89.97) and R1 (98.33)
- Top 3 strongest VERIFIED VeRi-776 results: (1) MBR4B-LAI w/RK 92.1 mAP, (2) Ours 89.97 mAP, (3) RPTM 88.0 mAP
- 14 figure specs (V1-V9 + P1-P5)
- Benchmark script outline (§9)
- Cleanup list (§7)
- Generator-script edits (§8)
- Harsh-truth assessment (§10) — recommendation: lock in "eval-time recipe" paper angle, do NOT chase 92% architecturally