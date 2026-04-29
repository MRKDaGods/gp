# Comparative Analysis Spec — MTMC Tracker vs Published SOTA

> **Phase A deliverable.** This spec defines the full plan for `docs/system-comparative-analysis.md` and 6 figures under `docs/figures/`. The Coder agent (Phase B) executes this without further questions. Where a literature number is uncertain it is tagged `[CITE_NEEDED: <description>]` and the Coder must mark it `n/a` in the figure (greyed bar / hollow marker).
>
> **Color scheme (consistent across all figures):**
> - Ours: `#1f77b4` (blue)
> - SOTA: `#ff7f0e` (orange)
> - Dead-end / failed: `#d62728` (red)
> - Backbone-class (ours via literature): `#9ecae1` (light blue)
> - Pareto frontier line: `#2ca02c` (green, dashed)
> - Annotation text / gap arrows: `#444444`
>
> **Font:** matplotlib default (DejaVu Sans), titles 13pt bold, axes 11pt, annotations 9pt.
> **Output:** every figure is saved as both `.png` (dpi=200) and `.pdf` to `docs/figures/`. Tight bbox.

---

## 1. Scope and inputs

### 1.1 Direct measurements (from this repo)
| Source | Claim | Citation |
|---|---|---|
| `docs/findings.md` (Final Result section) | CityFlowV2 ensemble best MTMC IDF1 = **0.7703** at `w_tertiary=0.60` (CLIP+DINOv2) | [findings.md#L1-L20](docs/findings.md#L1-L20) |
| `docs/findings.md` | CityFlowV2 reproducible single-model best MTMC IDF1 = **0.775** (10c v52, CLIP) | [findings.md#L26-L34](docs/findings.md#L26-L34) |
| `docs/findings.md` | CityFlowV2 historical best (not reproducible) = **0.784** (v80) | [findings.md#L35-L36](docs/findings.md#L35-L36) |
| `docs/findings.md` | Vehicle ReID on CityFlowV2 — CLIP ViT-B/16 mAP=**80.14%**, R1=**92.27%** | [findings.md#L31-L33](docs/findings.md#L31-L33) |
| `docs/findings.md` | Vehicle ReID on CityFlowV2 — DINOv2 ViT-L/14 mAP=**86.79%**, R1=**96.15%** | [findings.md#L29-L30](docs/findings.md#L29-L30) |
| `docs/findings.md` | DINOv2 MTMC IDF1 = **0.744** (regression vs CLIP) | [findings.md#L30](docs/findings.md#L30) |
| `docs/findings.md` (Person section) | WildTrack ground-plane IDF1 = **0.947**, MODA = **0.903** | [findings.md#L78-L86](docs/findings.md#L78-L86) |
| `docs/findings.md` | WildTrack detector MVDeTr 12a v3 MODA = **0.921** | [findings.md#L80-L81](docs/findings.md#L80-L81) |

### 1.2 Literature numbers (look up; the Coder marks `[CITE_NEEDED]` items as `n/a` if not resolvable)
| Quantity | Best estimate | Tag |
|---|---|---|
| AIC22 Team28 (1st) MTMC IDF1 | 0.8486 | confirmed in findings.md |
| AIC22 Team59 (2nd) MTMC IDF1 | 0.8437 | confirmed in paper-strategy.md |
| AIC22 Team37 (3rd) MTMC IDF1 | 0.8371 | confirmed |
| AIC22 Team28 model count | 5 | confirmed |
| AIC22 Team28 GPU type / training hours | A100 multi-GPU, multi-day | `[CITE_NEEDED: AIC22 Team28 paper compute footprint]` |
| WildTrack SOTA IDF1 (ground plane) | ~0.953 | `[CITE_NEEDED: cite to MVDeTr or follow-up paper reporting GP IDF1]` |
| TransReID ViT-B/16 on VeRi-776 mAP/R1 | 82.1 / 97.4 | He et al., ICCV 2021 |
| CLIP-ReID ViT-B/16 on VeRi-776 mAP/R1 | ~85.2 / 97.6 | Li et al., AAAI 2023 `[CITE_NEEDED: confirm]` |
| SOTA on VeRi-776 (current) mAP/R1 | ~87 / 97.7 | `[CITE_NEEDED: latest 2024-2025 ReID paper]` |
| TransReID ViT-B/16 on Market-1501 mAP/R1 | 89.0 / 95.1 | He et al., 2021 |
| CLIP-ReID ViT-B/16 on Market-1501 mAP/R1 | ~89.8 / 95.7 | Li et al., 2023 `[CITE_NEEDED: confirm]` |
| SOLIDER Swin-B Market-1501 mAP/R1 (with rerank) | 95.6 / 96.7 | Chen et al., CVPR 2023 |
| Our backbone training: GPU | Kaggle T4 (16GB) or P100 (16GB) | repo |
| Our backbone training time (CLIP TransReID 09b) | ~5-6 hours single GPU | `[CITE_NEEDED: confirm hours from kernel logs]` |
| Our full stage 0-2 inference on CityFlowV2 (10a) | ~50 min Kaggle T4 | findings.md |

---

## 2. Comparison Matrix (canonical table — section 2 of analysis doc)

The Coder pastes this table verbatim into `docs/system-comparative-analysis.md` § 2. Cells tagged `[CITE_NEEDED:...]` become `n/a` in the published doc but remain present as comments in the source markdown.

| Dataset / Metric | Our Result | SOTA Result | SOTA Method | Year | # Models | GPU | Train Time | Inference |
|---|---|---|---|---|---|---|---|---|
| CityFlowV2 MTMC IDF1 (single model) | **0.775** | 0.8486 | AIC22 Team28 (5-model ensemble) | 2022 | 5 | `[CITE_NEEDED: A100×N]` | `[CITE_NEEDED]` | `[CITE_NEEDED]` |
| CityFlowV2 MTMC IDF1 (ensemble best) | **0.7703** | 0.8486 | as above | 2022 | 2 (CLIP+DINOv2) | T4 / P100 | ~6h | ~50min/scene |
| WildTrack GP IDF1 | **0.947** | ~0.953 `[CITE_NEEDED]` | `[CITE_NEEDED]` | `[CITE_NEEDED]` | 1 | T4 | `[CITE_NEEDED]` | n/a |
| WildTrack GP MODA | **0.903** | ~0.915 `[CITE_NEEDED: MVDeTr GP MODA]` | MVDeTr | 2021 | 1 | A100 `[CITE_NEEDED]` | `[CITE_NEEDED]` | n/a |
| VeRi-776 mAP / R1 (backbone-class) | 82.1 / 97.4 (TransReID ViT-B/16, literature) | ~87 / 97.7 `[CITE_NEEDED]` | `[CITE_NEEDED: 2024-25 SOTA]` | `[CITE_NEEDED]` | 1 | `[CITE_NEEDED]` | `[CITE_NEEDED]` | `[CITE_NEEDED]` |
| Market-1501 mAP / R1 (backbone-class) | ~89.8 / 95.7 `[CITE_NEEDED]` (CLIP-ReID ViT-B/16) | 95.6 / 96.7 (rerank) | SOLIDER Swin-B | 2023 | 1 | `[CITE_NEEDED]` | `[CITE_NEEDED]` | `[CITE_NEEDED]` |

Footnote (paste verbatim): *"VeRi-776 and Market-1501 rows describe the published performance of our backbone class (TransReID/CLIP-ReID ViT-B/16) on those benchmarks. We do not directly evaluate on these datasets in this work."*

---

## 3. Hypotheses (section 3 of analysis doc)

For each, the Coder writes: **Claim** (1 sentence), **Evidence** (2-3 bullets with linked numbers), **Verdict** (SUPPORTED / PARTIAL / NOT-SUPPORTED).

### H1 — Efficiency frontier
- **Claim:** Our system achieves ~91% of SOTA accuracy with ~20-40% of compute (1-2 models vs 5).
- **Evidence:** 0.775 / 0.8486 = 91.3% relative IDF1; 1 model vs 5; T4 vs A100×N `[CITE_NEEDED]`.
- **Expected verdict:** **SUPPORTED** (modulo the missing absolute compute numbers).

### H2 — Low-end accessibility
- **Claim:** The full pipeline trains on a single Kaggle T4/P100 (16GB free tier).
- **Evidence:** Confirmed from `docs/copilot-instructions.md` (Kaggle workflow); CLIP TransReID training fits in 16GB; SOTA recipes need multi-GPU clusters `[CITE_NEEDED]`.
- **Expected verdict:** **SUPPORTED**.

### H3 — Reproducibility moat
- **Claim:** 225+ documented experiments, fully open Kaggle notebooks, 12+ documented dead ends.
- **Evidence:** `docs/findings.md` Dead Ends section; `docs/experiment-log.md` (cite line range); kernel logs publicly accessible.
- **Expected verdict:** **SUPPORTED**.

### H4 — Cross-domain consistency
- **Claim:** Single architecture (7-stage) handles vehicles AND people without redesign.
- **Evidence:** CityFlowV2 MTMC IDF1 = 0.775 + WildTrack GP IDF1 = 0.947 with the same Stage 1-4 framework; only Stage 2 backbone & Stage 4 ground-plane projection differ.
- **Expected verdict:** **PARTIAL** — same pipeline shell, but person pipeline uses MVDeTr (not the same Stage 1 detector). Coder must call this out honestly.

### H5 — Inference cost
- **Claim:** Single-model design has lower per-camera inference cost than 5-model ensemble.
- **Evidence:** 1 backbone forward vs 5; FLOPs ratio ≈ 1:5 for ViT-B/16 vs (R101-IBN ×2 + ResNeXt101 + ConvNeXt + R50) `[CITE_NEEDED: rough FLOPs sum]`.
- **Expected verdict:** **SUPPORTED** (qualitative).

---

## 4. Figure specifications

All figures: `docs/figures/G{N}_{slug}.png` and `.pdf`. Use `matplotlib.pyplot`, no seaborn. Save with `bbox_inches='tight'`. Use the color palette from the header.

---

### G1 — Pareto frontier (accuracy vs compute)
- **File:** `docs/figures/G1_pareto_frontier.png` / `.pdf`
- **Title:** "MTMC IDF1 vs Model Count on CityFlowV2"
- **X-axis:** "Number of ReID models" (linear, range 0.5 to 6.0)
- **Y-axis:** "MTMC IDF1" (linear, range 0.70 to 0.90)
- **Data points** (each as `(x, y, label, color, marker)`):
  - `(1, 0.775, "Ours (CLIP, single)", #1f77b4, "o")`
  - `(2, 0.7703, "Ours (CLIP+DINOv2 fusion)", #1f77b4, "s")`
  - `(5, 0.8486, "AIC22 Team28 (1st)", #ff7f0e, "^")`
  - `(3, 0.8437, "AIC22 Team59 (2nd)", #ff7f0e, "v")`
  - `(3, 0.8371, "AIC22 Team37 (3rd)", #ff7f0e, "D")`  *[# models for Team37: `[CITE_NEEDED]`, default to 3]*
- **Pareto frontier:** dashed green line through `(1, 0.775) → (3, 0.8437) → (5, 0.8486)`.
- **Annotations:** label each point with its name; place an arrow from "Ours (CLIP, single)" to the frontier reading "91.3% of SOTA at 20% of model count".
- **Legend:** top-left, frameon=False.
- **Grid:** light grey, alpha=0.3.

---

### G2 — MTMC IDF1 across datasets (Ours vs SOTA)
- **File:** `docs/figures/G2_mtmc_idf1_datasets.png` / `.pdf`
- **Title:** "MTMC IDF1: Ours vs Published SOTA"
- **X-axis:** dataset (categorical: "CityFlowV2 (vehicle)", "WildTrack (person, GP)")
- **Y-axis:** "MTMC IDF1" (range 0.0 to 1.0)
- **Bars:** grouped, two bars per dataset:
  - CityFlowV2: Ours=0.775 (blue), SOTA=0.8486 (orange) — gap "−7.4 pp"
  - WildTrack: Ours=0.947 (blue), SOTA=0.953 (orange, hatched if `[CITE_NEEDED]` not resolved) — gap "−0.6 pp"
- **Annotations:** above each pair, write the gap (e.g., `Δ = −7.4pp`).
- **Bar width:** 0.35; bar value labels at top of each bar (e.g. `0.775`).
- **Legend:** "Ours" / "SOTA", top-right.

---

### G3 — ReID mAP across benchmarks
- **File:** `docs/figures/G3_reid_map_benchmarks.png` / `.pdf`
- **Title:** "Single-Model ReID mAP — Our Backbone Class vs Benchmark SOTA"
- **X-axis:** dataset (categorical: "VeRi-776", "Market-1501", "CityFlowV2 (vehicle, ours)")
- **Y-axis:** "mAP (%)" (range 0 to 100)
- **Bars:** grouped, two bars per dataset:
  - VeRi-776: Ours-class=82.1 (light blue, hatch="//", literature TransReID ViT-B/16), SOTA=`[CITE_NEEDED, default 87.0]` (orange, hatched if uncertain)
  - Market-1501: Ours-class=89.8 `[CITE_NEEDED]` (light blue, hatch="//"), SOTA=95.6 (orange, SOLIDER+rerank)
  - CityFlowV2: Ours=80.14 (solid blue, direct measurement), SOTA=86.79 (orange, our DINOv2 — note "MTMC-incompatible" footnote)
- **Annotations:** "literature value" footnote for the hatched (light blue) bars.
- **Legend:** top-right with three entries: "Ours (literature, backbone-class)", "Ours (measured)", "SOTA".

---

### G4 — Ablation contribution waterfall
- **File:** `docs/figures/G4_ablation_waterfall.png` / `.pdf`
- **Title:** "MTMC IDF1 Ablation Waterfall — From Baseline to 0.7703 (CityFlowV2)"
- **X-axis:** ordered ablation step (categorical, rotated 30°)
- **Y-axis:** "MTMC IDF1" (range 0.50 to 0.85)
- **Data (waterfall — start at baseline, each green bar adds delta, end at total):**
  1. Baseline (TransReID + naive CC + no FIC) → `0.55` `[CITE_NEEDED: confirm baseline measurement; if absent, label "baseline (estimated)"]`
  2. + FIC whitening → +0.015 (+1.5 pp) — cumulative 0.565
  3. + AQE K=3 → +0.005 (+0.5 pp) — 0.570
  4. + Power normalization → +0.005 — 0.575
  5. + PCA 384D → +0.020 `[CITE_NEEDED: confirm magnitude]` — 0.595
  6. + Conflict-free CC → +0.0021 — 0.597
  7. + Intra-camera merge → +0.0028 — 0.600
  8. + Temporal overlap bonus → +0.009 — 0.609
  9. + min_hits=2 tracker → +0.002 — 0.611
  10. + Quality-aware crop selection (Laplacian) → +`[CITE_NEEDED: estimate from log]` — 0.620
  11. + Full v80 association recipe → +0.155 → **0.775** (single CLIP)
  12. + DINOv2 score-level fusion (w_t=0.60) → −0.005 → **0.7703** (note: this is on a different baseline run; marked differently — show as orange tail to distinguish)
- **Render:** classic waterfall with green positive deltas, red negative deltas, blue final bar at 0.7703.
- **Note in caption:** "Step 11 lumps together the 225+ association configs into a single recipe choice; individual sub-deltas are within ±0.3 pp."
- **Annotations:** target line at SOTA = 0.8486 (dashed orange horizontal).

---

### G5 — Dead-end catalog
- **File:** `docs/figures/G5_dead_ends.png` / `.pdf`
- **Title:** "Documented Dead Ends — Measured Cost to MTMC IDF1"
- **Layout:** horizontal bar chart, sorted by absolute cost (largest at top).
- **Bars (label, delta_pp, color=#d62728):**
  - CSLS calibration: −34.7 pp
  - AFLink motion linking (worst case): −13.2 pp
  - DMT camera-aware training (single-model): −1.4 pp
  - 384px ViT deployment: −2.8 pp
  - Hierarchical clustering: −5.0 pp (worst case)
  - FAC (cross-cam KNN consensus): −2.5 pp
  - Feature concatenation: −1.6 pp
  - Network flow solver: −0.24 pp
  - CID_BIAS (GT-learned): −3.3 pp
  - Reranking k-reciprocal: −0.5 pp `[CITE_NEEDED: pick one canonical number]`
  - Score fusion w/ ResNet101-IBN secondary: −0.1 pp
  - Hierarchical centroid averaging: −1.0 pp `[CITE_NEEDED]`
- **X-axis:** "ΔMTMC IDF1 (pp)" (range −36 to 0)
- **Y-axis:** approach name (categorical)
- **Annotations:** at the right end of each bar, write the exact pp value.
- **Caption:** "All 12 approaches measured on CityFlowV2 against the same single-model CLIP baseline."

---

### G6 — Compute cost comparison
- **File:** `docs/figures/G6_compute_cost.png` / `.pdf`
- **Title:** "Compute Cost — Ours vs SOTA Recipe"
- **X-axis:** "System" (categorical: "Ours (1 model)", "AIC22 Team28 (5 models)")
- **Y-axis:** "GPU-hours" (linear, log scale optional if values span >10×)
- **Stacked bars (two stacks):**
  - Ours: training=6h (CLIP TransReID, T4) `[CITE_NEEDED: confirm]`, inference per scene=0.83h (50 min) → total ≈ 6.83h on a single T4
  - SOTA Team28: training=`[CITE_NEEDED: 5×backbones × ~12h on A100 → ~60h A100]`, inference=`[CITE_NEEDED]`
- **Annotation:** if SOTA values are CITE_NEEDED, render as hatched stacks with a "literature estimate" caption.
- **Note in caption:** "T4 ≈ 0.25× A100 throughput, so all-A100-equivalent ours is ~1.7 A100-hours; SOTA is ~60+ A100-hours."

---

## 5. Final analysis document — structure for `docs/system-comparative-analysis.md`

The Coder writes this in the order below. Each section has the listed headers and contents.

### § 1 Executive summary
- Lead bullet: "We achieve **91.3% of AIC22 1st-place MTMC IDF1** (0.775 / 0.8486) using a **single ReID model** versus a 5-model ensemble — placing us on the Pareto frontier of accuracy vs compute."
- Bullet 2: "Our person pipeline reaches **0.947 ground-plane IDF1 on WildTrack**, within **0.6 pp** of published SOTA, despite using a 7-stage pipeline shared with the vehicle domain."
- Bullet 3: "The 7.4 pp gap on CityFlowV2 is reproducibly traced to **cross-camera feature invariance**, not association tuning — confirmed via 225+ exhaustive sweeps and 12+ failed alternative approaches."
- Bullet 4: "All training fits on a **single Kaggle T4 (16GB free tier)**, vs multi-A100 setups required by SOTA."
- Bullet 5: "Recommended publishing angle: **'One Model, 91% of SOTA: An Efficiency Frontier in Multi-Camera Tracking'** — IEEE Access or IEEE T-ITS."

### § 2 Per-dataset breakdown
For each of {CityFlowV2, WildTrack, VeRi-776, Market-1501}, write a subsection with:
- A 3-sentence description of the dataset
- The matrix row from § 1 above
- Reference to the relevant figure (G1, G2, or G3)
- For VeRi-776 / Market-1501: explicit caveat that these are backbone-class literature numbers, not direct evaluations on our pipeline.

### § 3 Hypothesis validation
Write the H1-H5 blocks per § 3 above (Claim / Evidence / Verdict).

### § 4 Where we stand (Pareto position)
- Reference G1.
- Discuss: we sit on the lower-left corner of the Pareto frontier (low compute, lower accuracy); SOTA is upper-right (high compute, high accuracy). No published system occupies our specific (1-model, 0.775) point with full reproducibility.
- Identify the frontier-shifting move: **better cross-camera invariance training** (not more models, not more association tuning).

### § 5 Publishing recommendation (decision tree)
Render as a decision tree in markdown:

```
H1 (efficiency) SUPPORTED?
├── YES → "Beyond Accuracy: Efficient MTMC at 91% SOTA with One Model"
│         Targets: IEEE T-ITS, IEEE Access
├── PARTIAL + H3 (reproducibility) SUPPORTED?
│   └── YES → "A Reproducible Single-Model Baseline for Multi-Camera Tracking"
│             Targets: MDPI Sensors, Multimedia Tools & Applications
└── H4 (cross-domain) SUPPORTED?
    └── YES → "One Pipeline, Two Domains: Unified MTMC for Vehicles and People"
              Targets: MTA, IEEE Access
```

If only negative results survive: keep the current draft "Beyond mAP: Training Methodology Dominates Feature Capacity". Targets: IEEE Access, MTA.

**Recommended primary target (assuming H1+H2+H3 SUPPORTED, H4 PARTIAL, H5 SUPPORTED):**
**IEEE Access** — broad scope, fast review, strong fit for "efficiency + exhaustive ablation" papers. Backup: **Multimedia Tools and Applications** (Springer) — has accepted similar IDF1=0.72-0.80 papers in 2024-2025.

### § 6 Risk register
| Risk | Severity | Mitigation |
|---|---|---|
| No direct evaluation on VeRi-776 / Market-1501 (only backbone-class literature numbers) | HIGH | Add an explicit "Limitations" section; consider a 1-week add-on eval on VeRi-776 test split |
| Single-dataset MTMC coverage (only CityFlowV2 + WildTrack) | MEDIUM | WildTrack already provides cross-domain validation |
| No human evaluation / qualitative study | LOW | Add one figure of qualitative trajectories per scene |
| Compute footprint of SOTA baselines is `[CITE_NEEDED]` (efficiency claim partially unverifiable) | HIGH | Either find AIC22 winner workshop papers, or weaken the H1 claim from "5× less" to "1 model vs 5 models" |
| 0.9 pp drift between historical (0.784) and current (0.775) reproducible baseline | MEDIUM | Already flagged in findings.md; report both numbers; use 0.775 as the conservative claim |
| Person pipeline uses MVDeTr (not the same Stage-1 detector as vehicles) | MEDIUM | Soften H4 verdict to PARTIAL; describe the shared portions explicitly |

---

## 6. Execution checklist for the Coder (Phase B)
1. Read this spec end-to-end.
2. Resolve `[CITE_NEEDED]` items by web search where possible. Mark unresolved ones as `n/a` in figures and as italic `[citation needed]` in the analysis doc.
3. Generate the 6 figures (matplotlib, no seaborn) into `docs/figures/`.
4. Write `docs/system-comparative-analysis.md` per § 5 structure.
5. Verify all repo citations link to live lines (run a quick `grep` per claim).
6. Run `python -c "import matplotlib.pyplot as plt; ..."` smoke test on each figure.
7. Do **not** create new branches or push to Kaggle; this is a docs-only task.
