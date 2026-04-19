# Fix 09l LAION-2B CLIP Training — Root Cause Analysis & Plan

## Status: SPEC READY

## Root Cause Analysis

### Primary Cause: Circle Loss Numerical Instability (CONFIRMED DEAD END)

The 09l notebook uses **"Experiment B: CircleLoss Ablation"** — a training recipe that replaces Triplet Loss with Circle Loss (m=0.25, gamma=128). This is a **known catastrophic failure mode** already documented in `docs/findings.md`:

- **09 v4 (Experiment B)**: Same recipe on OpenAI CLIP → 18.45% mAP, `inf` loss every epoch
- **09d v17**: Circle Loss on ResNet101-IBN-a → 29.6% mAP
- **09f v2**: Circle Loss on ResNet → 16.2% mAP

The 09l result (20.36% mAP, `inf` loss for 120 epochs) is **identical in failure mode** to the known 09 v4 failure. Circle Loss with gamma=128 creates numerically unstable logits within `torch.amp.autocast` fp16 precision — the `gamma * ap * (sim - delta_p)` term can exceed fp16 range, producing `inf` before `logsumexp` can stabilize it.

**The LAION-2B backbone is NOT the cause** — the same failure occurs with the OpenAI CLIP backbone.

### Secondary Factor: Missing VeRi-776 Pretrained Weights

The 09l notebook initializes from CLIP only (`PRETRAINED_PATH = None`). The primary model (09b v2) uses VeRi-776 pretrained weights, which provide a much better starting point for CityFlowV2 fine-tuning. However, this is a secondary factor — even with VeRi-776 weights, Circle Loss would still produce `inf`.

### Why the Model Still Learned Somewhat (18.6% → 20.36%)

Despite `inf` reported loss, the CE loss component (computed on classifier logits, not features) likely still provided weak gradients through GradScaler's inf-handling. The scaler detects inf/nan and skips optimizer steps, but some steps may have had finite total loss from CE alone, allowing minimal learning.

## Comparison: 09l (Failed) vs 09b v2 (Primary, 80.14% mAP)

| Aspect | 09b v2 (PRIMARY) | 09l (FAILED) |
|--------|-------------------|--------------|
| **Backbone** | `vit_base_patch16_clip_224.openai` | `vit_base_patch16_clip_224.laion2b` |
| **VeRi-776 init** | Yes | No |
| **Loss stack** | CE(eps=0.05) + **Triplet(m=0.3)** + Center | CE(eps=0.05) + **Circle(m=0.25, γ=128)** + Center |
| **EMA** | Disabled | Disabled |
| **Result** | **80.14% mAP** | **20.36% mAP** (inf loss) |

The ONLY structural difference that matters is **Circle Loss vs Triplet Loss**. The backbone difference (OpenAI vs LAION-2B) is the variable we actually want to test.

## Fix Plan

### Goal
Test whether the LAION-2B CLIP backbone can match or exceed the OpenAI CLIP backbone (80.14% mAP) when trained with the **correct recipe**.

### Changes Required

1. **Replace Circle Loss with Triplet Loss** in the training config cell:
   - Remove: `circle_loss_fn = CircleLoss(m=0.25, gamma=128).to(DEVICE)`
   - Add: `triplet_loss_fn = TripletLossHardMining(margin=0.3).to(DEVICE)`
   - Note: `TripletLossHardMining` is already defined in the notebook (cell with losses)

2. **Update the training loop loss computation**:
   - Replace: `circle_loss_fn(f, pids)` → `triplet_loss_fn(f, pids)`
   - In both branches (JPM 3-output and 2-output)

3. **Update all print/logging strings**:
   - Change experiment label from "Experiment B: CircleLoss ablation" to "Experiment A: LAION-2B backbone test"
   - Update loss description to "CE(eps=0.05) + Triplet(m=0.3) + Center(5e-4)"

4. **Add VeRi-776 pretrained weight loading** (if available):
   - The notebook already has the VeRi-776 loading code (search paths defined)
   - Ensure the VeRi-776 model is attached as a Kaggle dataset source
   - If VeRi-776 weights were trained with OpenAI CLIP, they may NOT be compatible with LAION-2B CLIP (different weight distributions). In that case, train from CLIP-only init but with Triplet Loss — this should still work much better than Circle Loss
   - **Decision**: First run WITHOUT VeRi-776 (CLIP-only init + Triplet Loss) to isolate the backbone variable. If mAP < 70%, then VeRi-776 pretrain on LAION-2B is needed (requires a separate 08-equivalent notebook)

5. **Keep everything else identical to 09b v2**:
   - 256×256 resolution
   - AdamW, backbone_lr=1e-4, head_lr=1e-3, LLRD=0.75
   - 120 epochs, 10-epoch warmup, CosineAnnealingLR
   - Baseline augmentations (flip, pad+crop, weak jitter, RE)
   - EMA disabled

### Config Summary (Fixed)
```
Losses: CE(eps=0.05) + Triplet(m=0.3) + Center(5e-4, delayed@ep15)
LR: backbone=1e-4, head=1e-3, LLRD=0.75
Optimizer: AdamW(wd=5e-4)
Scheduler: CosineAnnealingLR(T_max=110) after 10-epoch warmup
Epochs: 120, eval every 20
Resolution: 256×256
EMA: disabled
Init: CLIP only (no VeRi-776, to isolate backbone variable)
```

### Expected Outcomes

| Scenario | Expected mAP | Interpretation |
|----------|-------------|----------------|
| LAION-2B + Triplet (CLIP-only init) | 65-75% | Backbone viable but needs VeRi-776 pretrain |
| LAION-2B + Triplet (CLIP-only init) | 75-80% | Backbone competitive, close to OpenAI variant |
| LAION-2B + Triplet (CLIP-only init) | >80% | Backbone superior, use as primary or ensemble |
| LAION-2B + Triplet (CLIP-only init) | <60% | Backbone fundamentally weaker, abandon |

### Risks
- VeRi-776 weights trained on OpenAI CLIP may not transfer to LAION-2B CLIP (different learned representations)
- LAION-2B was trained on noisier data than OpenAI CLIP (2B web-scraped vs 400M curated pairs) — may produce less discriminative features for fine-grained vehicle ReID
- Even with correct recipe, LAION-2B may underperform OpenAI CLIP — this is a valid experimental finding

### Measurement
- **Primary metric**: mAP and R1 on CityFlowV2 val set
- **Success threshold**: mAP ≥ 65% (ensemble-viable), ideally ≥ 75% (competitive with primary)
- **Failure threshold**: mAP < 60% (backbone not viable, abandon LAION-2B path)
- **If successful**: Deploy through 10a→10b→10c chain and measure MTMC IDF1 impact

## Decision: Re-run with Same LAION-2B Backbone

**Yes** — re-run with the same LAION-2B backbone but with the correct Triplet Loss recipe. The current failure tells us NOTHING about the backbone quality because Circle Loss destroyed training entirely. We need a clean test with the proven recipe to evaluate the backbone itself.

## Summary

The 09l failure is **100% caused by Circle Loss** (a known dead end), NOT by the LAION-2B backbone. The fix is trivial: replace `CircleLoss` with `TripletLossHardMining` in the training config and loop. This will produce a valid measurement of LAION-2B CLIP's potential as an ensemble candidate.
