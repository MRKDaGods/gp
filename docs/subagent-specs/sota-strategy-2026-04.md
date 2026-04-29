# SOTA Strategy — Brutal Assessment (2026-04-25)

> **Author**: MTMC Planner
> **Audience**: Project lead (frustrated, considering pivot)
> **Status**: Strategic checkpoint while 09r ViT-L is training

---

## 1. Honest Assessment — Is Beating SOTA Realistic?

**No. Not with the current trajectory. Probability < 5%.**

The experiment log contains one finding that overrides every optimistic narrative we have told ourselves:

> **Higher single-model mAP has repeatedly produced *worse* MTMC IDF1.**

Concrete evidence from `docs/findings.md`:

| Single-model change | mAP delta | MTMC IDF1 delta |
|---|:-:|:-:|
| 09 v2 augoverhaul (256px) | **+1.45pp** (80.14 → 81.59) | **−5.3pp** (0.775 → 0.722) |
| 09 v3 augoverhaul-EMA | +1.39pp (80.14 → 81.53) | −5.3pp (0.775 → 0.722) |
| 384px deployment (09b v2) | + (single-cam improved) | −2.8pp (0.784 → 0.756) |
| LAION-2B CLIP fusion (78.61% mAP) | — | −0.5pp (0.774 → 0.769) |

**This is not noise. Five separate experiments confirm: improving the primary feature backbone in isolation hurts MTMC.** The transfer function from single-camera retrieval quality to cross-camera identity preservation is *broken* in this pipeline at the high end. We do not understand why, and we have no diagnostic that predicts which mAP gains will transfer.

If 09r ViT-L finishes at 85% mAP, the empirical prior says it will land somewhere in **0.72–0.78 MTMC IDF1**, not 0.84+. There is no evidence in our 225+ experiment history that any single-model scaling path crosses 0.78.

### Realistic ceilings

| Strategy | Realistic MTMC IDF1 | Probability |
|---|:-:|:-:|
| Single ViT-L @ 85% mAP (09r as-deployed) | 0.74 – 0.78 | 60% in that range |
| Single ViT-L + retuned association | 0.76 – 0.79 | 40% |
| 2-model ensemble (uncorrelated architectures) | 0.78 – 0.81 | 30% if we get a truly orthogonal partner |
| 3-model diverse ensemble + box-grained matching | 0.81 – 0.84 | 10% |
| **Beating SOTA (> 0.8486)** | — | **< 5%** |

**The honest framing: we are not on a path to beat SOTA. We are on a path to publish a strong single-model + exhaustive-ablation paper, with a small chance of catching SOTA if 2–3 ensemble pieces line up.**

The AIC22 1st place result was a multi-team, multi-month effort with 5 ReID backbones + box-grained matching + a custom association pipeline. Replicating that scale of engineering with one person and ~24h Kaggle quota is not realistic on the current schedule.

---

## 2. Recommended Primary Path

### Choice: **(a) Architecturally diverse ensemble**, but with a sharp constraint.

**Rejected alternatives:**

- **(b) SAM2 foreground masking** — already tested in 10a v29 / 10c v50. Result: **−8.7pp MTMC IDF1**. Confirmed dead end. Background context carries identity signal in CityFlowV2 (road texture, scene cues correlate with camera ID and help association).
- **(c) GNN edge-classifier association** — theoretically attractive but requires significant new infrastructure (edge feature engineering, training data construction, training loop). High implementation cost, uncertain payoff. **Defer to fallback.**
- **More single-model scaling (ViT-L, ViT-H)** — empirical prior strongly negative. 09r is already running, so we will get one more data point, but we should not double down.

**Why diverse ensemble is the chosen path:**

1. It is the *only* approach with documented evidence of working at SOTA scale (every AIC top-3 used 3–5 models).
2. We already have stage 2/4 infrastructure for score-level fusion.
3. The LAION-2B fusion failure (10c v56) gave us a sharp diagnostic: **two CLIP ViT-B/16 variants are too correlated**. The constraint is now clear — we need *architecturally orthogonal* partners, not just *differently-pretrained* ones.

### The hard constraint we now know

A useful secondary needs **all three** of:

1. **mAP ≥ 70%** on CityFlowV2 (R50-IBN at 63.64% gave +0.06pp, too weak).
2. **Architecturally orthogonal** to ViT-B/16 CLIP (different inductive bias: hierarchical attention, CNN, or different pretraining corpus that is not CLIP-family).
3. **Independently trained** with a stable recipe (no CircleLoss; AdamW + Triplet + Center proven stable).

**Candidates that satisfy all three:**

| Backbone | Inductive bias | Pretrain | Why it could work |
|---|---|---|---|
| **ConvNeXt-Large** | CNN, hierarchical | ImageNet-22k | Pure CNN, very different from ViT-B CLIP feature geometry |
| **Swin-Large** | hierarchical local attention | ImageNet-22k | Window attention is orthogonal to CLIP global attention |
| **DINOv2 ViT-L/14** | self-supervised, no text supervision | LVD-142M | Pretrain corpus + objective both orthogonal to CLIP image-text |

**Strongest candidate: DINOv2 ViT-L/14.** Self-supervised features are known to capture different invariances than CLIP. DINOv2 has produced strong fine-grained retrieval results in published work without the text-alignment bias that may be the root cause of CLIP-CLIP fusion correlation.

**Second choice: ConvNeXt-Large.** Maximum architectural distance from ViT-B CLIP.

---

## 3. Concrete Next Experiment

### Notebook: `09s-dinov2-vitl14-cityflowv2`

**Account**: gumfreddy (primary; ali369/mrkdagods as fallback if quota allows)
**GPU**: Kaggle T4 / P100
**Push timing**: After 09r (ViT-L) completes and frees its GPU slot. Do not push earlier — Kaggle allows max 2 concurrent GPU sessions per account, and 09r is currently consuming one.

### Architecture & training recipe

```yaml
backbone: dinov2_vitl14  # via timm "vit_large_patch14_dinov2"
input_size: [224, 224]   # DINOv2 native resolution; do not deviate
embedding_dim: 1024      # native DINOv2 ViT-L output
projection_head:
  type: bnneck           # match TransReID convention
  bn_layers: [LayerNorm, BatchNorm1d]
  dropout: 0.1

losses:
  ce_label_smoothing: 0.05      # primary classification
  triplet:
    margin: 0.3
    miner: BatchHard
  center:
    weight: 5e-4
    delayed_start_epoch: 15      # critical — same as 09l v3
  # NO CircleLoss. NO ArcFace. Both confirmed dead ends.

optimizer: AdamW
backbone_lr: 5e-6              # lower than CLIP recipe — DINOv2 features are sensitive
head_lr: 5e-4
weight_decay: 1e-4
llrd: 0.75                     # layer-wise LR decay
warmup_epochs: 5
schedule: cosine
total_epochs: 200              # extend to 300 if mAP still climbing at 200 (09l v3 pattern)
batch_size: 64                 # P x K = 16 x 4
ema:
  enabled: true
  decay: 0.9995                # tuned for 200ep schedule (0.9999 was too high in 09 v2)

augmentations:
  # CONSERVATIVE — match 09l v3 baseline, NOT augoverhaul
  RandomHorizontalFlip: 0.5
  Pad+RandomCrop: yes
  ColorJitter: [0.2, 0.15, 0.1, 0.0]
  Normalize: imagenet
  RandomErasing: default
  # NO RandomGrayscale, NO GaussianBlur, NO RandomPerspective (augoverhaul ingredients)

eval:
  metric: mAP, R1, R5
  rerank: also_report
  on: cityflowv2_eval_split
```

### Validation gates

Before deploying as a fusion secondary, the trained model must clear:

1. **mAP ≥ 70%** on CityFlowV2 eval — minimum viable for fusion.
2. **mAP ≥ 75%** to be worth a full 10c sweep — based on 10c v60/v61 evidence that 63.64% gave only +0.06pp.
3. **Cosine correlation with primary < 0.85** on a 200-tracklet sample — measured by computing per-tracklet embedding cosine to primary ViT-B/16 CLIP features. Must be lower than the LAION-2B model's correlation with primary.

If all three gates pass → run 10c fusion sweep with `w ∈ {0.10, 0.20, 0.30, 0.40}`. Expected outcome: +0.5 to +2.0pp MTMC IDF1.

If gate 3 fails (correlation too high) → DINOv2 transferred to ReID via the same projection-head recipe converged to similar feature geometry as CLIP. Fall back to ConvNeXt-L (notebook `09t`).

---

## 4. Parallel Preparation Tasks (while 09r trains)

Tasks ordered by leverage. Do these in this order in the local repo only — no Kaggle pushes until 09r completes.

### A. Build `09s` notebook locally (do not push)

- Copy `notebooks/kaggle/09l_laion2b_clip/` as a template (it has the working stable recipe).
- Swap backbone construction to `timm.create_model("vit_large_patch14_dinov2", pretrained=True, num_classes=0)`.
- Adjust `input_size` to 224, `embedding_dim` to 1024.
- Verify the projection head matches TransReID convention (BNNeck).
- Lower `backbone_lr` from 1e-5 to 5e-6.
- Adjust EMA decay to 0.9995.
- Validate JSON structure with `python -c "import json; json.load(open('...ipynb'))"`.

### B. Add a feature-correlation diagnostic script

Create `scripts/diagnose_feature_correlation.py`:

- Loads two sets of tracklet embeddings (primary + candidate secondary).
- Computes per-tracklet mean cosine similarity between matched track IDs.
- Reports the distribution and a single scalar correlation summary.
- Cheap CPU-only run — should take < 2 min on the existing 929-tracklet outputs.

This becomes gate 3 in section 3. **Without this diagnostic we cannot tell ahead of a 10c sweep whether a fusion attempt is doomed.** This is the missing piece from the LAION-2B fusion failure — we trained a 78.6% mAP model and only learned post-hoc that it was too correlated.

### C. Pre-stage ConvNeXt-L fallback notebook (`09t`)

Same template as `09s` but with `timm.create_model("convnext_large.fb_in22k_ft_in1k_384", ...)`. Keep it locally; only push if `09s` fails gate 3 or gate 1.

### D. Document the augoverhaul → MTMC regression in `docs/findings.md` as a *first-class research finding*

This is the most surprising result in the project and we keep treating it as a one-off. The fact that **+1.45pp mAP → −5.3pp MTMC** across two independent training runs (09 v2 and 09 v3 EMA) means there is a structural bug in either:

- The augoverhaul augmentations themselves removing color/texture cues that drive cross-camera matching, OR
- A silent feature distribution shift that breaks PCA/FIC whitening downstream, OR
- A real and underexplored phenomenon: that retrieval mAP and cross-camera identity quality are partially decoupled in low-ID datasets.

A 1-2 hour root-cause investigation here has higher expected value than another training run, because it could unlock the entire single-model scaling path.

### E. Do **not** touch:

- Stage 4 association parameters. **EXHAUSTED.** 225+ configs.
- AFLink, CSLS, CID_BIAS, hierarchical clustering, network flow. **All dead ends.**
- 384px deployment of any model. **Dead end.**
- Score fusion with another CLIP-ViT-B variant. **Dead end (correlation).**
- ResNet101-IBN-a / ResNeXt fine-tuning. **Ceiling at 52.77% mAP, exhausted.**

---

## 5. Stop Conditions

### Stop the "ViT-L scaling" path if:

- 09r finishes and 10c deployment lands < 0.78 MTMC IDF1 → ViT-L does not transfer; abandon further single-model scaling. *Empirical prior says this is the most likely outcome (~60%).*
- 09r finishes < 82% mAP → did not even improve single-camera; abandon and move to ensemble work.

### Stop the "DINOv2 / ConvNeXt-L diversity" path if:

- 2 consecutive architecturally diverse secondaries both train to mAP < 70% on CityFlowV2 → the bottleneck is data scale (128 IDs), not architecture. Pivot to either VeRi-776 + CityFlowV2 multi-task training, or accept the single-model ceiling and write the paper.
- A model passes mAP ≥ 70% but fails fusion correlation gate 3 → log it, try next architecture once.
- Three diverse secondaries pass gate 1 but all give < +0.5pp in 10c fusion sweep → score-level fusion in this pipeline is structurally limited; pivot to box-grained matching instead.

### Stop the entire SOTA-chasing effort and pivot to paper if:

- We reach 2026-05-15 without a confirmed +1pp gain from any new direction.
- Kaggle quota across all three accounts drops below 12h cumulative.
- The next 3 experiments all land within ±0.3pp of 0.775 (matching the 225-config association sweep pattern — a clear signal that we are out of degrees of freedom).

The paper angle in `docs/paper-strategy.md` ("One Model, 91% of SOTA" + exhaustive ablation) is already publishable. **Do not let SOTA-chasing burn the paper window.**

---

## Summary

- **Beating SOTA: < 5% probability.** The empirical mAP-to-MTMC transfer is broken at the high end of single-model quality.
- **Highest-leverage path**: architecturally diverse ensemble, starting with DINOv2 ViT-L/14. Hard constraint: must pass a correlation diagnostic before any 10c sweep.
- **Parallel prep while 09r runs**: build `09s` notebook + correlation diagnostic + pre-stage ConvNeXt-L fallback. **Push nothing** until 09r finishes.
- **The most underrated open question**: why +1.45pp mAP → −5.3pp MTMC. A targeted root-cause investigation here has higher expected value than another training run.
- **Hard pivot trigger**: by 2026-05-15 with no +1pp gain → stop chasing SOTA, write the paper.

---

End of spec.