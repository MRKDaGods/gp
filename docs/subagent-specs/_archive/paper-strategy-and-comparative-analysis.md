# Paper Strategy & Comparative Analysis Spec

**Status**: Planner output, ready for Coder implementation.
**Author**: MTMC Planner
**Scope**: Read-only analysis + spec. Implementation deferred to Coder.

---

## Part 1 — Paper Strategy Recommendation

### Decision context

The user is choosing between:

- **Option A — VeRi-776-only paper**: pure single-camera vehicle ReID. Target the historically claimed `R1=0.984505 / mAP=0.873314` with `qe_k=3, k1=30, k2=10, λ=0.2` on a TransReID checkpoint that was *only* fine-tuned on VeRi-776 (not the later CityFlowV2-finetuned variant).
- **Option B — Full system paper (vehicles + persons, MTMC)**: end-to-end MTMC for CityFlowV2 vehicles and WILDTRACK persons, using the cross-domain CityFlowV2-finetuned ReID we deploy. VeRi-776 becomes a supporting ReID benchmark ablation.

### Reproducibility status of the historical 98.45% claim

The historical `R1=0.984505 / mAP=0.873314` configuration is **not currently reproducible from any artifact in this repository**. Specifically:

1. The closest reproduction is **`outputs/09v_veri_v4/veri776_eval_results_v4.json`** (kernel `yahiaakhalafallah/09v-veri-776-eval-transreid-rerank` v10), which lands at **`R1 = 0.980334 / mAP = 0.8477`** with `single_flip + rerank k1=25, k2=8, λ=0.3`. The closest config to the user's claim — `qe_k=3, k1=30, k2=10, λ=0.3, AQE k=4` — gives **`R1 = 0.9797 / mAP = 0.8498`** (better mAP, slightly worse R1). No tested config reaches **R1≥0.984**.
2. `docs/subagent-specs/recreate-comparative-analysis.md` (lines 7, 110) explicitly documents that an earlier draft of `system-comparative-analysis.md` reported `mAP=87.33% / R1=98.45%` as a measured result, and that claim was **flagged as hallucinated** because no eval artifact existed. The current measured numbers (84.77% / 98.03%) replaced it.
3. The "VeRi-only" checkpoint the user references is `vehicle_transreid_vit_base_veri776.pth` in `mrkdagods/mtmc-weights`, which is the checkpoint already used in `09v_veri_v4`. It is the same weights file. The "later CityFlowV2-finetuned variant" (`transreid_cityflowv2_best.pth`) was independently evaluated cross-dataset on VeRi-776 in the same kernel and lands at only `mAP=72.73% / R1=92.49%` — clearly not a candidate for a VeRi-776 paper.
4. The historical OSNet checkpoint `vehicle_osnet_veri776.pth` (which underpinned the v80-era 0.784 MTMC IDF1) was dropped from `mrkdagods/mtmc-weights` on 2026-03-30 and is **also not recoverable**. This is documented in `docs/findings.md` and `.github/copilot-instructions.md`.

**Conclusion**: the 0.984505 / 0.873314 numbers cannot be cited as our result without further evidence. The best reproducible VeRi-776 numbers we own are **`mAP=84.77% / R1=98.03%`** (or **`mAP=84.98% / R1=97.97%`** at the AQE-rerank operating point closest to the user's described config).

### Gap-to-SOTA reality check

| Track | Best ours (reproducible) | SOTA reference | Gap |
|---|---:|---:|---:|
| VeRi-776 single-camera ReID | mAP=84.77%, R1=98.03% | mAP≈87.0%, R1≈97.7% (literature reference, marked `*` in `system-comparative-analysis.md`) | **mAP −2.2pp; R1 already at or above** |
| CityFlowV2 MTMC IDF1 | 0.7703 (CLIP+DINOv2 fusion, single ReID model) | 0.8486 (AIC22 Team28, 5-model ensemble) | **−7.83pp** |
| WILDTRACK GP IDF1 | 0.947 | ~0.953 (literature, `*`) | **−0.6pp** |

Note that on **R1**, our reproducible 98.03% is already at or above the literature reference of 97.7%. Only the mAP gap remains, and `outputs/09v_veri_v4/veri776_eval_results_v4.json` shows the AQE+rerank config reaches mAP=84.98% — that is **−2.0pp** from the 87.0% literature reference.

### Per-option assessment

**Option A — VeRi-only paper**

- *Novelty*: Limited. VeRi-776 has been studied since 2016. Modern SOTA (CLIP-ReID, TransReID variants) is dense, and a single-config `R1≈98 / mAP≈85` result without architectural novelty does not stand out. Reranking + AQE on TransReID-CLIP ViT-B/16 is not new methodology.
- *Reproducibility*: Strong as long as we cite the **measured** 84.77 / 98.03 and not the unverified 87.33 / 98.45. Targeting the unverified number invites a "where is the eval?" desk reject.
- *SOTA gap*: Already at literature R1, but mAP is −2.2pp behind. Realistic ceiling on the existing checkpoint.
- *Effort*: Low. We have the eval JSON, the rerank sweep, and the AQE sweep. Ablation tables write themselves.
- *Real-world deployability angle*: **Negative**. The VeRi-only checkpoint is not what we deploy in MTMC, and the cross-dataset evaluation in the same kernel shows it does not transfer (CityFlowV2-finetuned variant scores 72.73 mAP cross-dataset). A paper that *only* looks at VeRi-only is purely an academic ReID exercise, not a system contribution.
- *Risk*: Reviewers will ask "why not evaluate cross-domain / on a more modern dataset / on MTMC". We have those answers but the paper would not include them.

**Option B — Full system paper**

- *Novelty*: Strong. The "feature quality, not association, is the MTMC bottleneck" thesis is backed by **225+ exhaustive association configs**, a curated **dead-end catalog** (CSLS −34.7pp, AFLink −13.2pp worst, 384px −2.8pp, hierarchical clustering −1 to −5pp, DMT −1.4pp, network flow −0.24pp, CID_BIAS −1.0 to −3.3pp, FAC −2.5pp, reranking always negative, score-fusion exhausted), and the **mAP-vs-MTMC paradox** showing DINOv2 ViT-L/14 +6.65pp mAP yields **−3.1pp** standalone MTMC IDF1. This is a genuine empirical contribution that the community has not published.
- *Reproducibility*: Strong. All numbers in `findings.md`, `experiment-log.md`, and `system-comparative-analysis.md` are directly traceable to JSON outputs and Kaggle kernel versions. The cross-domain CityFlowV2-finetuned checkpoint is the deployed artifact and what a real-world user would download.
- *SOTA gap*: Honest. CityFlowV2 IDF1 = 0.7703 is **90.8% of AIC22 1st place** with **1 ReID model vs 5**, and we add a near-SOTA WILDTRACK person result (94.7% IDF1, 0.6pp gap). The framing matches the established `paper-strategy.md` thesis ("One Model, 91% of SOTA"). VeRi-776 (84.77 mAP / 98.03 R1) becomes a **clean ReID benchmark ablation** showing the underlying single-camera ReID quality is competitive — strengthening, not weakening, the system claim.
- *Effort*: Higher than Option A but most artifacts already exist. Main ask is the comparative-analysis update in Part 2 below.
- *Real-world deployability angle*: **Strong**. The cross-domain checkpoint is exactly what is deployed; the pipeline runs on a single T4/P100 (~6h training + ~50min inference) versus multi-A100 ensembles for SOTA. Efficiency story is concrete.
- *Risk*: VeRi-776 results are not headline. They are a supporting table. We must be careful to neither overclaim 87.33 / 98.45 nor underplay the legitimate 84.77 / 98.03.

### RECOMMENDATION

> **RECOMMENDATION: Option B — Full system paper (vehicles + persons MTMC, with VeRi-776 as a supporting ReID ablation).**
>
> The full system paper is the only direction with a defensible novel contribution: 225+ exhaustive association experiments, a documented dead-end catalog, the mAP-vs-MTMC paradox, and a dual-domain (vehicles non-overlapping cameras + persons overlapping cameras) evaluation under one pipeline shell, all delivered with a single ReID model at 90.8% of AIC22 1st place. Option A reduces to a polished ReID ablation on a 10-year-old benchmark with no architectural novelty, and crucially relies on a `R1≈0.984 / mAP≈0.873` claim that has already been flagged as hallucinated in `recreate-comparative-analysis.md` and is not present in any eval artifact. The reproducible measurements (`R1=98.03%, mAP=84.77%`) are strong enough to be a **supporting** ablation but not a headline. The deployable cross-domain checkpoint is also exactly what a downstream user would adopt, so Option B is honest about what we ship.

Recommended title (keep `paper-strategy.md`'s lead): *"One Model, 91% of SOTA: A Systematic Study of Feature Quality vs. Association Tuning Bottlenecks in Multi-Camera Tracking"*.

Recommended target venues, in order: **IEEE Access**, **Multimedia Tools & Applications**, **Scientific Reports**.

---

## Part 2 — Comparative Analysis Update Spec

This section is the implementation spec for updating `docs/system-comparative-analysis.md` and `scripts/generate_comparative_analysis.py`. The Coder must implement exactly what is specified here without further design decisions.

### Goals

1. Add per-dataset comparison tables and per-dataset "what we contribute" narratives for **VeRi-776**, **CityFlowV2**, and **WILDTRACK** so the document supports a paper-style discussion of each benchmark separately.
2. Add new plots saved to `docs/figures/` covering each dataset and an overview gap chart.
3. Keep existing G1–G6 figures intact (do not delete) but add new G7–G10 figures alongside them.
4. Treat all unverified numbers (`98.45 / 87.33`, OSNet v80 0.784) as **historical references only**. Never list them as our reproducible measurement.

### 2.1 Document changes (`docs/system-comparative-analysis.md`)

The current document already has solid headline tables (sections 2, 3, 4). The following changes are additive. Do not delete or rewrite existing sections except where explicitly stated.

**Insert a new top-level section 3 named `Per-Dataset Comparison`**, renumber the current sections 3 onward by +1 (current §3 "Comparison to State of the Art" becomes §4, current §4 becomes §5, etc.). Inside the new §3, add three subsections:

#### 3.1 VeRi-776 (single-camera vehicle ReID benchmark)

A table of methods on VeRi-776 with citations placeholders (`[CITE_NEEDED]`). Coder must include at least these rows; populate citation placeholders only from references already in `docs/research_papers_and_metrics.md` or `docs/SOTA_ANALYSIS.md` if present, otherwise leave as `[CITE_NEEDED]`:

| Method | Backbone | mAP (%) | R1 (%) | Source |
|---|---|---:|---:|---|
| TransReID (He et al., 2021) | ViT-B/16 | ~82.0* | ~97.0* | `[CITE_NEEDED]` |
| CLIP-ReID (Li et al., 2023) | ViT-B/16 CLIP | ~85.0* | ~97.5* | `[CITE_NEEDED]` |
| MsKAT (multi-scale knowledge transfer) | — | ~87.0* | ~97.7* | `[CITE_NEEDED]` |
| **Ours (TransReID ViT-B/16 CLIP @ 256px, baseline)** | ViT-B/16 CLIP | 81.87 | 97.50 | `outputs/09v_veri_v4/veri776_eval_results_v4.json` |
| **Ours (single_flip + rerank k1=25,k2=8,λ=0.3)** | ViT-B/16 CLIP | **84.77** | **98.03** | same |
| **Ours (AQE k=4 + rerank k1=30,k2=10,λ=0.3)** | ViT-B/16 CLIP | 84.98 | 97.97 | same |

Below the table add a 2–3 sentence narrative: our R1 (98.03%) **matches or exceeds** the literature reference (~97.7%) on the deployed checkpoint, while mAP is ~2pp behind the reported SOTA. Clarify that the historically claimed `R1=98.45 / mAP=87.33` is not reproducible on any checkpoint currently in the weights datasets and is excluded from this report (cite `recreate-comparative-analysis.md` lines 7, 110).

#### 3.2 CityFlowV2 (vehicle MTMC, AIC22 Track 1)

Reuse the existing AIC22 leaderboard table content from current §3.1 and move it here unchanged. Add an explicit new table covering **per-metric** comparison (IDF1, MOTA, HOTA, IDP, IDR) where data is available. Coder must populate from `docs/findings.md` and any `outputs/` JSON the Coder can locate by greping for `MTMC_IDF1` and `0.7703`. If MOTA/HOTA columns for AIC22 leaderboard teams are not present in any in-repo source, mark them `[CITE_NEEDED]` rather than inventing values.

| System | IDF1 | MOTA | HOTA | Models | Source |
|---|---:|---:|---:|:---:|---|
| AIC22 Team28 (1st) | 0.8486 | `[CITE_NEEDED]` | `[CITE_NEEDED]` | 5 | `paper-strategy.md` |
| AIC22 Team59 (2nd) | 0.8437 | `[CITE_NEEDED]` | `[CITE_NEEDED]` | 3 | same |
| AIC22 Team37 (3rd) | 0.8371 | `[CITE_NEEDED]` | `[CITE_NEEDED]` | — | same |
| **Ours (CLIP+DINOv2 fusion)** | **0.7703** | 0.6725 | 0.5749 | 1 (+1 score) | `findings.md` § "Final Result …" row `ter_060` |

Add a 3–4 sentence narrative emphasizing: 90.8% of 1st-place IDF1 with 1/5 the model count; gap is feature-side, not association-side; conflict-free CC and FIC whitening already make our association competitive.

#### 3.3 WILDTRACK (person MTMC, overlapping cameras)

Add a per-metric table. Mark all literature SOTA entries with `*` and `[CITE_NEEDED]` where the spec defaults are placeholders.

| System | GP IDF1 | GP MODA | Detector MODA | Source |
|---|---:|---:|---:|---|
| Literature SOTA reference | 0.953* | 0.915* | — | `[CITE_NEEDED]` |
| Hou et al. MVDet (2020) | `[CITE_NEEDED]` | ~0.88* | — | `[CITE_NEEDED]` |
| MVDeTr (Hou & Zheng, 2021) | `[CITE_NEEDED]` | ~0.92* | ~0.92* | `[CITE_NEEDED]` |
| **Ours (Kalman, 12b v1/v2/v3)** | **0.947** | **0.900** | **0.921** | `findings.md` § "Person Pipeline" |

Narrative (2–3 sentences): same pipeline shell handles persons with only the detector swapped (MVDeTr replaces YOLO26m); 0.6pp from literature with full convergence across 59 tracker configs; tracker-limited not detector-limited.

#### 3.4 Cross-Dataset Contributions (insert as new §3.4)

Bullet list of "what we contribute beyond SOTA" as keypoints usable in the paper Discussion section:

- **VeRi-776**: matched literature R1 (98.03%) on a single-config TransReID-CLIP eval; reproducible AQE/rerank ablation across ≥10 configs in one JSON.
- **CityFlowV2**: 90.8% of SOTA IDF1 with 1/5 the ReID models; **first** systematic 225+ association ablation showing the bottleneck is feature quality; documented dead-end catalog (12+ approaches with measured pp regressions).
- **WILDTRACK**: tracker-limited at 0.6pp gap with 59 tracker configs across Kalman, naive, and global-optimal solvers all converging within ±0.0004 IDF1.
- **Cross-cutting**: dual-domain evaluation (non-overlapping vehicles + overlapping persons) under one Stage-0…Stage-5 pipeline; single-T4/P100 deployable.

### 2.2 Plots to add (`docs/figures/` via `scripts/generate_comparative_analysis.py`)

Add **four new figures** alongside existing G1–G6. The Coder must add new builder functions (`build_g7`, `build_g8`, `build_g9`, `build_g10`) and add their stems to the `FIGURES` list at the top of the script. Use `matplotlib` (already imported); `seaborn` is **not** required. Save each figure as both PNG (300 DPI) and PDF using the existing `save_figure(fig, stem)` helper.

Style rules (already enforced by `set_style` and `format_axes`): keep the same color palette `COLORS`, fonts, and grid style. Asterisk-mark literature values in legends or footnotes when used.

#### G7 — `G7_per_dataset_bars.png`

Per-dataset bar chart: us vs top-3 SOTA (or top-1 SOTA where ≤3 are available). Three vertical panels (one per dataset).
- Panel A: VeRi-776, metric = mAP, bars = `[Ours 84.77, TransReID 82.0*, CLIP-ReID 85.0*, MsKAT 87.0*]`.
- Panel B: CityFlowV2, metric = MTMC IDF1, bars = `[Ours 0.7703, Team37 0.8371, Team59 0.8437, Team28 0.8486]`.
- Panel C: WILDTRACK, metric = GP IDF1, bars = `[Ours 0.947, MVDet *, MVDeTr *, Lit-SOTA 0.953*]`. Where literature value placeholder, draw bar with `hatch="//"` and color `COLORS["uncertain"]`.

Each panel labels bars with their value to 2–3 decimals. Add a footnote line via `fig.text(...)` clarifying `*` = literature value.

Implementation location: insert `build_g7()` right after `build_g6()` in `scripts/generate_comparative_analysis.py`. Signature: `def build_g7() -> None:`. Must call `save_figure(fig, "G7_per_dataset_bars")`.

Data sources for the **measured** bars:
- VeRi: `outputs/09v_veri_v4/veri776_eval_results_v4.json` (the Coder must `json.load` and read the row corresponding to single_flip + rerank k1=25, k2=8, λ=0.3; if the JSON schema is unclear, fall back to the constant `84.77` already in `DEFAULTS["veri_measured_map"]`).
- CityFlowV2: existing `DEFAULTS["vehicle_mtmc_idf1"] = 0.7703`.
- WILDTRACK: existing `DEFAULTS["wildtrack_idf1"] = 0.947`.

Add new entries to `DEFAULTS`:
```
"veri_transreid_map": 82.0,
"veri_clip_reid_map": 85.0,
"veri_mskat_map": 87.0,
"aic22_team28_idf1": 0.8486,
"aic22_team59_idf1": 0.8437,
"aic22_team37_idf1": 0.8371,
"wildtrack_mvdet_moda": 0.88,
"wildtrack_mvdetr_moda": 0.92,
```

#### G8 — `G8_relative_gap_overview.png`

Single overview chart: relative percentage of SOTA (i.e. `our_value / sota_value * 100`) for each dataset, on the same horizontal bar chart.
- Bars: `[VeRi-776 mAP 84.77/87.0=97.4%, CityFlowV2 IDF1 0.7703/0.8486=90.8%, WILDTRACK IDF1 0.947/0.953=99.4%]`.
- X-axis: 0–100% with a dashed reference line at 100%.
- Annotate each bar with the absolute pp gap (e.g. `−2.2pp mAP`, `−7.83pp IDF1`, `−0.6pp IDF1`).

Implementation: `build_g8()`, signature `def build_g8() -> None:`, calls `save_figure(fig, "G8_relative_gap_overview")`. Footnote should reiterate that VeRi and WILDTRACK SOTAs are literature values (`*`).

#### G9 — `G9_veri_rerank_sweep.png`

Ablation curve from `outputs/09v_veri_v4/veri776_eval_results_v4.json`. The Coder must `json.load` the file and parse its config sweep entries. Plot **two lines** on the same axes:
- X-axis: rerank `λ` (lambda), values from the sweep (likely 0.0, 0.1, 0.2, 0.3, 0.4, 0.5).
- Left Y-axis: mAP (%).
- Right Y-axis (twin): R1 (%).
- Mark the best operating point (`λ=0.3`, `k1=25`, `k2=8`) with a star marker.

If the JSON does not contain a clean λ sweep at fixed `(k1, k2)`, the Coder should select the closest reasonable subset and document the selection in a code comment. If parsing fails, fall back to plotting the four explicit rows already documented in `system-comparative-analysis.md` §4.2 Checkpoint A as discrete points.

Implementation: `build_g9()`, signature `def build_g9() -> None:`, calls `save_figure(fig, "G9_veri_rerank_sweep")`.

#### G10 — `G10_cityflow_threshold_sweep.png`

Ablation curve for the CityFlowV2 association similarity threshold sweep. The Coder must:
1. Run `grep_search` for `similarity_threshold` and `MTMC_IDF1` in `outputs/` and `docs/experiment-log.md`. Locate the most recent dense sweep (likely the 10c v52 or v54 family).
2. Plot MTMC IDF1 vs `sim_thresh` with the optimal point starred.
3. If no clean sweep JSON is locatable, fall back to the table values in `docs/experiment-log.md` for the latest documented sweep and document the source in the figure footnote.

Implementation: `build_g10()`, signature `def build_g10() -> None:`, calls `save_figure(fig, "G10_cityflow_threshold_sweep")`. Add `"G7_per_dataset_bars"`, `"G8_relative_gap_overview"`, `"G9_veri_rerank_sweep"`, `"G10_cityflow_threshold_sweep"` to the `FIGURES` list at top of script so the existing `verify_pngs()` smoke test exercises them.

### 2.3 Markdown integration

Insert a new §6 named `Figures` in `docs/system-comparative-analysis.md` (renumber subsequent sections accordingly) listing each figure with a 1-line caption and a markdown link, e.g.:

```markdown
- ![G7 per-dataset bars](figures/G7_per_dataset_bars.png) — Ours vs SOTA per dataset (VeRi mAP, CityFlowV2 IDF1, WILDTRACK IDF1).
- ![G8 relative gap](figures/G8_relative_gap_overview.png) — Relative percentage of SOTA per dataset.
- ![G9 VeRi rerank sweep](figures/G9_veri_rerank_sweep.png) — λ sweep on VeRi-776 rerank.
- ![G10 CityFlow threshold sweep](figures/G10_cityflow_threshold_sweep.png) — Similarity threshold sweep on CityFlowV2 Stage-4.
```

The existing `check_markdown_links` validator will catch any broken links — Coder must run the script and resolve any link issues before commit.

### 2.4 Coder checklist

1. Add new `DEFAULTS` keys listed in §2.2 G7.
2. Implement `build_g7`, `build_g8`, `build_g9`, `build_g10` in `scripts/generate_comparative_analysis.py`. Call them from `main()`.
3. Add the new stems to the module-level `FIGURES` list.
4. Re-run `python scripts/generate_comparative_analysis.py` and confirm the 6 existing PNGs still regenerate plus the 4 new PNGs are produced and `>10KB`.
5. Update `docs/system-comparative-analysis.md` per §2.1 (new §3 with 4 subsections, renumbering, and §6 Figures index).
6. Run `python scripts/generate_comparative_analysis.py` again — note that `build_markdown()` currently writes the entire markdown body. The Coder must update `build_markdown()` to include the new §3 Per-Dataset Comparison content and §6 Figures index, otherwise the script will overwrite the manual edits.
7. Verify: `check_markdown_links(...)` returns no issues, `verify_pngs(...)` passes for all 10 figures.

### 2.5 What the Coder must NOT do

- Do not introduce any new `R1=0.984 / mAP=0.873` claim. If the user later supplies the original VeRi-only checkpoint binary plus a reproducible eval kernel that demonstrates those numbers, that is a separate task.
- Do not delete existing G1–G6 figures or their builder functions.
- Do not change the existing §1 Abstract, §2 Headline Performance, or §7 Conclusion (currently §6) wording — only renumber and append.
- Do not invent MOTA/HOTA values for AIC22 leaderboard teams. Use `[CITE_NEEDED]`.

---

## Part 3 — Markdown File Inventory

### Top-level

- `README.md` — Project overview, quick-start commands, architecture diagram, training instructions. Public-facing entry point.

### Strategy / planning (active)

- `docs/paper-strategy.md` — The "One Model, 91% of SOTA" angle, target venues, ablation must-haves, dead-end summary. Canonical paper plan.
- `docs/findings.md` — Living research log: current performance numbers per dataset, dead-end catalog with measured regressions, prioritized action plan, mAP-vs-MTMC paradox analysis. **MUST be kept updated**.
- `docs/BREAKTHROUGH_PLAN.md` — Earlier breakthrough roadmap; partly superseded by `findings.md` action plan but kept for context.
- `docs/SOTA_ANALYSIS.md` — Detailed SOTA comparison and analysis of competitor methods.
- `docs/research_papers_and_metrics.md` — Reference list of cited methods and the metrics/datasets they report.

### Reference / state (active)

- `docs/system-comparative-analysis.md` — Authoritative comparison vs SOTA on CityFlowV2, WILDTRACK, and VeRi-776. **This is the file Part 2 above modifies.**
- `docs/experiment-log.md` — Chronological experiment log (preferred form). Header lists current best.
- `docs/experiment_log.md` — **Older underscored variant**, kept for historical compatibility. Coder should not write to it.
- `docs/architecture.md` — 7-stage pipeline architecture, data contracts, inter-stage dependencies.
- `docs/data_contracts.md` — Data formats passed between stages (tracklets schema, embeddings layout, etc.).
- `docs/dataset_guide.md` — Dataset acquisition, preprocessing, GT alignment for CityFlowV2 / WILDTRACK / VeRi-776 / Market-1501.
- `docs/setup_guide.md` — Local dev setup (Python 3.11.9 venv), Kaggle account setup.
- `docs/kaggle_training_guide.md` — Kaggle workflow: kernel push, log polling, account rotation, GPU session safety.
- `docs/mvdetr-integration.md` — MVDeTr-specific integration notes for the WILDTRACK person pipeline.
- `docs/paper-draft.md` — Working draft of the paper itself.
- `docs/team_workload.md` — Team contributor responsibilities/breakdown.

### Generated / specs (`docs/subagent-specs/`)

Subagent specs follow a `<topic>.md` convention; each is a self-contained Planner→Coder hand-off. Selected key entries:

- `recreate-comparative-analysis.md` — The spec that produced the current `system-comparative-analysis.md`. Documents the 87.33/98.45 hallucination removal. **Important reference for Part 2.**
- `comparative-analysis-spec.md` — Earlier spec for the comparative analysis figures.
- `figure-data-corrections.md` — Audit + corrections to figure data inputs.
- `paper-content.md`, `paper-review.md`, `paper-table-corrections.md` — Paper-writing-specific specs.
- `aic-winning-methods-analysis.md` — Analysis of AIC22 winners' methods (5-model ensembles, CID_BIAS, DMT, etc.).
- `sota-strategy-2026-04.md`, `sota-paper-research.md`, `sota-code-integration.md`, `sota-breakthrough.md`, `sota-battle-plan.md` — SOTA-targeted strategy specs (most superseded by `findings.md` dead-end conclusions).
- Numerous training/experiment specs: `09f-*`, `09b-*`, `09l-*`, `09m-*`, `09o-*`, `09r-*`, `09s-*`, `09v-*`, `10a-*`, `10c-*`, `12a-*`, `12b-*`, `dmt-*`, `eva02-*`, `ensemble-*`, `extended-*`, `kalman-*`, `multi-query-*`, `tta-*`, `vit-small-*`, `zone-*`, etc. Each captures the Planner intent for a single Kaggle kernel or stage change.
- This file (`paper-strategy-and-comparative-analysis.md`) — Combined paper-strategy decision + comparative-analysis update spec, authored by the Planner.

### Instructions (`.github/`)

- `.github/copilot-instructions.md` — Top-level project state, critical rules (GPU pipeline policy, notebook editing rules, frame ID convention, config override paths), confirmed dead ends, target performance. **Always read first.**
- `.github/instructions/context-engineering.instructions.md` — General Copilot context-engineering guidance (file paths, type annotations, COPILOT.md, etc.).
- `.github/agents/planner.agent.md` — MTMC Planner agent definition.
- `.github/agents/coder.agent.md` — MTMC Coder agent definition.
- `.github/agents/orchestrator.agent.md` — MTMC Orchestrator agent definition.
- `.github/skills/autoresearch/SKILL.md` — Autoresearch experiment-loop skill.
- `.github/skills/copilot-instructions-blueprint-generator/SKILL.md` — Skill for generating copilot-instructions blueprints.

### Memory

- `memories/session/progress.md` — Session-scoped working notes. Not a permanent doc.

### Non-markdown but referenced

- `scripts/generate_comparative_analysis.py` — Generates all `docs/figures/G*.{png,pdf}` and writes `docs/system-comparative-analysis.md` from a `build_markdown()` string template. Currently produces G1–G6. Part 2 of this spec extends it to G7–G10 and updates the markdown template. Already imports `matplotlib` and `PIL`.