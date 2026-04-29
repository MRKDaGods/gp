# 5-Hour Final Sprint — SOTA Attempt + Paper

**Created**: 2026-04-25
**Hard deadline**: T+5:00. After this, code freeze. No further experiments.
**Current best (reproducible)**: MTMC IDF1 = 0.775 (ViT-B/16 CLIP, 10c v52, gumfreddy)
**SOTA target**: 0.8486 (AIC22 1st, 5-model ensemble) — gap = 7.36pp

---

## Executive Summary

The single highest-EV action in 5 hours is **CLIP × DINOv2 score-level ensemble** (Part A). It is the only untried path with credible SOTA upside: (1) the two strongest single models we've ever trained, (2) maximally diverse pretraining (image-text contrastive vs self-distillation), (3) prior fusion failures all used either weak (R50 52.77%) or correlated (CLIP-CLIP) secondaries, and (4) DINOv2's per-camera IDF1 = 0.794 vs CLIP's strong cross-camera ceiling (0.775) is exactly the complementarity score-fusion needs. The Stage 2 pipeline already supports a `vehicle3` (tertiary) ReID model with separate PCA + camera_bn + score-level fusion in Stage 4 — we only need to enable it. Run a single 10a kernel with CLIP primary + DINOv2 tertiary on the production tracklets, then 10b, then a 10c weight sweep. In parallel the user (CPU/local) writes the paper around the now-confirmed thesis: **feature quality, not association tuning, is the MTMC bottleneck — and even within feature quality, training methodology dominates raw mAP** (the DINOv2 finding makes the paper publishable independently of the SOTA outcome).

---

## Part A — SOTA Attempt: CLIP × DINOv2 Score-Level Ensemble

### Why this and not the alternatives

| Option | EV | Why |
|---|---|---|
| **A1. CLIP primary + DINOv2 tertiary, score-level fusion (FIC per model)** | **PICK** | Two strongest models ever, maximally diverse pretraining, infra already exists |
| A2. DINOv2 intra-cam + CLIP cross-cam hybrid | Reject | Requires non-trivial new association code; AFLink-style tracklet linking is a confirmed dead end on CityFlowV2 (-3.8pp to -13.2pp); DINOv2 already runs through standard intra-merge |
| A3. New training experiment | Reject | Even fastest training (~3h) leaves no time for downstream eval; no untried recipe has clear EV |
| A4. Re-tune association | Reject | 225+ configs already within 0.3pp of optimal — confirmed exhausted |

### Hypothesis

Score-level fusion with **per-model FIC whitening** (not concat) of CLIP (cross-camera-strong) and DINOv2 (per-camera-strong, +6.65pp mAP) breaks the 0.775 ceiling. Expected: +1 to +3pp. Risk: prior CLIP+CLIP fusion lost -0.5pp because the two backbones were too correlated. CLIP and DINOv2 are NOT in the same family — one is image-text contrastive on 400M pairs, the other is self-distillation on 142M curated images. Different inductive biases on background, viewpoint, and texture.

Quantitative justification: DINOv2 per-camera IDF1 = 0.794 and CLIP per-camera IDF1 ≈ 0.77. If their cross-camera errors are uncorrelated, an FIC-whitened weighted sum should pick up the higher-precision intra-camera DINOv2 evidence without inheriting its weak cross-camera invariance, because Stage 4's exhaustive cross-camera matching is dominated by CLIP at low DINOv2 weight.

### Pipeline changes

All in `notebooks/kaggle/10a_stages012/` (CLIP-primary chain), NOT 10a_dinov2.

#### 1. `configs/datasets/cityflowv2.yaml`

Enable tertiary slot with DINOv2:

```yaml
stage2:
  reid:
    vehicle3:
      enabled: true
      save_separate: true
      model_name: "vit_large_patch14_dinov2"
      weights_path: "/kaggle/input/09s-dinov2-large-cityflowv2/best_model.pth"
      embedding_dim: 1024
      input_size: [252, 252]
      clip_normalization: false
```

(Exact `weights_path` must match the 09s v1 checkpoint output filename — verify before push. `input_size` must be 252 to match training; stride=14.)

#### 2. 10a kernel `kernel-metadata.json`

Add kernel source for the DINOv2 checkpoint:

```json
"kernel_sources": [
  "yahiaakhalafallah/09s-dinov2-large-cityflowv2",
  ...existing CLIP/checkpoint sources
]
```

#### 3. 10a notebook overrides cell

Add to the `--override` list:

```
stage2.reid.vehicle3.enabled=true
stage2.reid.vehicle3.weights_path=/kaggle/input/09s-dinov2-large-cityflowv2/best_model.pth
stage2.reid.vehicle3.input_size=[252,252]
stage2.reid.vehicle3.embedding_dim=1024
```

Also keep `vehicle2.enabled=false` (R50 secondary already proven a dead end at +0.06pp). Single tertiary, not 3-way.

#### 4. 10c kernel: weight sweep

Run a sweep over `stage4.association.tertiary_embeddings.weight` ∈ `[0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]` with `secondary_embeddings.weight=0.0`. Use the locked v52 association recipe:

```
stage4.association.graph.similarity_threshold=0.50
stage4.association.appearance_weight=0.70
stage4.association.fic.regularisation=0.50
stage4.association.query_expansion.k=3
stage4.association.gallery_expansion.threshold=0.48
stage4.association.orphan_match.threshold=0.38
```

If the best fusion weight peaks at the boundary (0.0 or 0.7), run a second pass extending the range — but only if T+3:30 budget is still available.

### Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| `vit_large_patch14_dinov2` not registered in `reid_model.py` builder | Medium | Verify with grep; if missing, copy construction code from 10a_dinov2 notebook (it works there) |
| DINOv2 ViT-L 1024D × 929 tracklets × 20 crops blows T4 memory | Medium | Halve `stage2.reid.batch_size` for vehicle3 if OOM |
| `tertiary_embeddings.path` doesn't auto-resolve to Stage 2 output | Low | Explicit override: `stage4.association.tertiary_embeddings.path=data/outputs/run_latest/stage2/embeddings_tertiary.npy` |
| Fusion regresses (correlated despite different pretraining) | Medium | Even -1pp is a publishable negative result strengthening the "training methodology > capacity" thesis |
| Kaggle GPU queue stalls | Medium | One account (yahiaakhalafallah) has 2 slots — push 10a first, hold 10b/10c until 10a output is confirmed |

---

## Part B — Paper Strategy (parallel, CPU-only)

### Recommended angle (revised given DINOv2 result)

**Title**: *"Beyond mAP: A Systematic Study of Feature Quality, Training Methodology, and Association Tuning in Multi-Camera Vehicle Tracking"*

**Core thesis** (now sharper than the previous "One Model, 91% of SOTA"):

> Cross-camera MTMC IDF1 is bottlenecked by **cross-camera feature invariance**, which is determined by **training methodology**, not raw single-camera mAP nor model capacity. We demonstrate this with two definitive controlled experiments: (1) 225+ association configs all within 0.3pp of optimal, and (2) a +6.65pp mAP DINOv2 ViT-L/14 upgrade that REGRESSED MTMC IDF1 by -3.1pp despite stronger per-camera IDF1 (0.794 vs 0.770).

This is genuinely novel — it directly contradicts the implicit assumption in nearly every ReID paper that better mAP → better tracking.

### Sections to write/update in the next 5 hours

| Priority | Section | What to write | Source material |
|---|---|---|---|
| **P0** | Abstract + Title | Replace "91% of SOTA" with "Beyond mAP" framing | This spec |
| **P0** | Introduction (1-2pp) | Position the mAP paradox as the contribution | findings.md current performance section |
| **P0** | Methods (3-4pp) | TransReID + CLIP, FIC whitening, conflict-free CC, Stage 4 details | configs/, src/stage4 |
| **P0** | Results: feature ablation table | CLIP vs DINOv2 vs (fusion if it works) — single-cam mAP, single-cam IDF1, MTMC IDF1 in three columns | findings.md |
| **P0** | Results: association exhaustion | 225+ config table summary; bar chart "all within 0.3pp" | experiment-log.md tables |
| **P1** | Discussion: why mAP misleads | Cross-camera vs intra-camera dominance analysis | findings.md DINOv2 section |
| **P1** | Dead-end catalog (1pp) | CSLS, AFLink, FAC, hierarchical, 384px, DMT, CID_BIAS, ResNet ensemble | findings.md |
| **P1** | Person pipeline section | WILDTRACK 94.7% IDF1, tracker convergence story | findings.md person section |
| **P2** | Compute efficiency comparison | 1× T4 vs 5-model A100 SOTA | Estimated GPU-hours table |
| **P2** | Qualitative figures | Cross-camera identity strips for correct + failed matches | Stage 6 viz outputs |

### Backup narrative if SOTA fusion fails

If CLIP × DINOv2 ≤ 0.775, the paper headline becomes:

> *"Even a +6.65pp mAP / +3.88pp R1 backbone upgrade does not improve MTMC IDF1, and even score-level fusion of two state-of-the-art self-supervised backbones (CLIP, DINOv2) does not break the cross-camera ceiling — establishing that single-frame ReID is fundamentally insufficient for non-overlapping MTMC."*

This is still publishable at IEEE Access / Multimedia Tools & Applications / Scientific Reports because it is a definitive negative result with a 225+ experiment ablation behind it.

### Backup narrative if SOTA fusion succeeds

If CLIP × DINOv2 ≥ 0.80, the paper headline shifts to:

> *"Two-Model Score-Level Fusion of Heterogeneous Self-Supervised Backbones Closes 70% of the Gap to a 5-Model Ensemble SOTA."*

Highest-impact venue tier becomes feasible (CVPRW AI City retroactive submission, IEEE T-ITS).

---

## Part C — Execution Timeline

GPU-bound steps run on Kaggle (yahiaakhalafallah, 2 concurrent slots). Paper work runs in parallel locally.

| T+ | GPU/Kaggle (sequential) | Local (parallel, paper) | Deliverable check |
|---|---|---|---|
| **0:00–0:30** | Verify DINOv2 backbone in `reid_model.py`; edit `cityflowv2.yaml` + 10a notebook overrides + 10a `kernel-metadata.json`; smoke-test config locally | Start abstract + intro draft | 10a kernel ready to push |
| **0:30** | **Push 10a once.** Confirm warning-free start. Begin polling. | — | 10a queued |
| **0:30–2:00** | 10a runs (~60–90 min: detection + tracking + 3 ReID extractions on T4) | Methods section, association ablation table from experiment-log.md | 10a output dataset committed |
| **2:00–2:15** | Update 10b kernel-metadata to consume new 10a output; push 10b once | Continue methods | 10b queued |
| **2:15–2:45** | 10b runs (CPU FAISS indexing, ~15–25 min) | Dead-end catalog section | 10b output ready |
| **2:45–3:00** | Update 10c kernel-metadata; embed 8-point weight sweep; push 10c once | Compute-efficiency table | 10c queued |
| **3:00–4:15** | 10c runs (~45–75 min for 8-config sweep) | Results: feature ablation table (placeholder for fusion result) | Sweep output |
| **4:15–4:30** | Parse 10c results; pick best weight; update findings.md + experiment-log.md with definitive number | Update abstract with final number | Final MTMC IDF1 known |
| **4:30–5:00** | Code freeze. Tag commit. Optional: final visualization run on best config locally | Discussion section, finalize narrative based on outcome | Paper draft + frozen results |

### Hard checkpoints

- **T+0:30**: 10a push must be warning-free. If `not valid dataset sources` warning appears, **immediately cancel** via `kaggle kernels cancel yahiaakhalafallah/mtmc-10a-stages012` and fix metadata before re-pushing.
- **T+2:00**: If 10a still running past 2:00, do not abort — let it finish but skip the paper "polish" phase.
- **T+3:00**: If 10b is not done by 3:00, run a single weight (`tertiary.weight=0.30`) on 10c instead of full sweep.
- **T+4:30**: Hard code freeze. No more pushes regardless of state.

### Concurrent slot management

Only one kernel runs at a time in this pipeline (10a → 10b → 10c is strictly sequential). The second GPU slot is held in reserve as an emergency for re-pushing if a kernel fails. **Never push 10a + 10b simultaneously.**

---

## Part D — Risk Mitigation & Decision Tree

```
End of T+4:15 — read 10c result:

├── MTMC IDF1 ≥ 0.80 (>+2.5pp lift)
│   └── PAPER: "Heterogeneous SSL Fusion Closes 70% of SOTA Gap"
│       Target: IEEE T-ITS, CVPRW AI City retro, IEEE Access
│       Action: Lead with fusion result, ablation as supporting evidence
│
├── 0.78 ≤ MTMC IDF1 < 0.80 (modest lift, +0.5–2.5pp)
│   └── PAPER: "Beyond mAP" + "first successful diverse-SSL ensemble for vehicle MTMC"
│       Target: IEEE Access, MTA, Scientific Reports
│       Action: Lead with mAP paradox, fusion as confirmation
│
├── 0.77 ≤ MTMC IDF1 < 0.78 (flat, ±0.5pp)
│   └── PAPER: "Beyond mAP" pure version (current best narrative)
│       Target: IEEE Access, MTA, Scientific Reports, Sensors
│       Action: Lead with 225+ experiments + mAP paradox + dead-end catalog
│
└── MTMC IDF1 < 0.77 (regression)
    └── PAPER: Stronger negative result — even diverse SSL fusion fails
        Target: IEEE Access, MTA, Sensors (negative results sections)
        Action: Frame as "ceiling diagnosis" — non-overlapping MTMC IDF1 is
                fundamentally bounded by single-frame appearance, motivates
                future work on temporal models, GNN association, or scene priors
```

**The paper is publishable in all four branches.** This is the key risk insurance: we are not betting publishability on the SOTA attempt.

### Hard "do not do" list during the sprint

- ❌ Do NOT enable AFLink in the 10c sweep (-3.8pp to -13.2pp on CLIP, confirmed dead end)
- ❌ Do NOT enable `mtmc_only=True` (-5pp)
- ❌ Do NOT enable `vehicle2` R50 secondary (already +0.06pp ceiling, wastes GPU time)
- ❌ Do NOT re-tune association params during the sweep — lock to v52 recipe
- ❌ Do NOT push any kernel a second time without confirming the first run is finished or cancelled
- ❌ Do NOT use the 10a_dinov2 chain — it has DINOv2 as primary, which we know fails (0.744). The whole point is CLIP primary + DINOv2 tertiary.

---

## Deliverables at T+5:00

1. **`docs/findings.md`** updated with CLIP × DINOv2 fusion result and final MTMC IDF1 number
2. **`docs/experiment-log.md`** updated with the 8-point sweep table
3. **Paper draft** (markdown or LaTeX) with abstract, intro, methods, results, discussion — final number filled in
4. **Frozen git tag** marking the final state of the codebase
5. **Decision-tree branch chosen** (per Part D) and venue list shortlisted
