# Augmentation Overhaul Ablation Plan

**Date**: 2026-04-14  
**Goal**: Isolate which change caused the -5.3pp MTMC IDF1 regression in 09 v2  
**Status**: Ready to execute  

---

## Background

The 09 v2 "augoverhaul" model changed TWO variables simultaneously:
1. **Augmentations**: Added RandomGrayscale, stronger ColorJitter (with hue), GaussianBlur, RandomPerspective, wider RandomErasing
2. **Loss function**: Replaced TripletLoss(margin=0.3) with CircleLoss(m=0.25, gamma=128)

| Condition | mAP | MTMC IDF1 |
|-----------|:---:|:---------:|
| Baseline (09 v1): Original augs + TripletLoss | 80.14% | **0.775** |
| Augoverhaul (09 v2): New augs + CircleLoss | 81.59% | **0.722** (-5.3pp) |
| Experiment B: Original augs + CircleLoss | ? | ? |
| Experiment A: Color-safe augs + TripletLoss | ? | ? |

We need experiments B and A to fill in the 2×2 matrix.

---

## Recommendation: Run Experiment B First

### Why B before A

1. **Cleanest ablation**: Only ONE variable changes vs baseline (loss function). Experiment A changes both augmentations AND reverts the loss, so it can't fully disambiguate.
2. **Minimal code changes**: Just swap `circle_loss_fn(f, pids)` → `tri_loss(f, pids)` in the training loop and instantiate TripletLossHardMining instead of CircleLoss. No augmentation code changes needed.
3. **Strong prior to test**: The analysis document notes the optimal similarity threshold shifted from 0.50 (baseline) to 0.45 (augoverhaul), suggesting CircleLoss fundamentally changed the feature space geometry for thresholded matching.
4. **Decisive decision tree**: The result of B directly determines whether to run A at all, or to run a different follow-up.

### Why NOT A first

Experiment A changes TWO things vs the augoverhaul (augmentations AND loss). If A succeeds, you know the combination works but can't attribute causation. If A fails, you still don't know which factor caused it. B is the sharper scalpel.

---

## Experiment B: Original Augmentations + CircleLoss

### Hypothesis
If CircleLoss alone causes the MTMC IDF1 regression (even at improved mAP), then running the original augmentation stack with CircleLoss will show mAP ≈ 80-81% but MTMC IDF1 significantly below 0.775. This would prove CircleLoss's ranking-oriented optimization produces features with worse margin properties for thresholded cross-camera matching.

### Code Changes (09 notebook, Cell 12 — transforms)

Revert `train_tf` to the baseline augmentation stack:

```python
# REVERT to baseline augmentations for ablation
train_tf = T.Compose([
    T.Resize((H + 16, W + 16), interpolation=T.InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(p=0.5),
    T.Pad(10),
    T.RandomCrop((H, W)),
    T.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1, hue=0.0),
    # NO RandomGrayscale
    # NO GaussianBlur
    # NO RandomPerspective
    T.ToTensor(),
    T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value="random"),
])
```

**Changes vs current (augoverhaul) notebook:**
- `ColorJitter(0.3, 0.25, 0.2, 0.05)` → `ColorJitter(0.2, 0.15, 0.1, 0.0)` (weaker, no hue)
- Remove `T.RandomGrayscale(p=0.1)` line
- Remove `T.RandomApply([T.GaussianBlur(...)], p=0.2)` line
- Remove `T.RandomPerspective(distortion_scale=0.1, p=0.2)` line
- `RandomErasing scale=(0.02, 0.4)` → `scale=(0.02, 0.33)` (narrower)

**NO changes to loss function** — keep CircleLoss(m=0.25, gamma=128) as-is in Cell 20.

### Kernel metadata

Update `kernel-metadata.json`:
```json
{
  "id": "gumfreddy/09-vehicle-reid-cityflowv2",
  "title": "09 Vehicle ReID CityFlowV2",
  ...
}
```
Push as 09 v4 (after v3 completes).

### Success Criteria

| Outcome | mAP | MTMC IDF1 | Interpretation |
|---------|:---:|:---------:|----------------|
| CircleLoss is the culprit | ≥79% | < 0.755 | CircleLoss hurts MTMC even with original augs → revert to TripletLoss, skip Experiment A as designed |
| CircleLoss is innocent | ≥79% | ≥ 0.770 | Augmentations were the problem → run Experiment A to find safe augmentation subset |
| CircleLoss helps MTMC | ≥80% | ≥ 0.775 | CircleLoss is actually beneficial; augmentations caused all the damage → focus pure augmentation ablation |
| Both are bad | < 79% | any | Something else went wrong (data order, init, etc.) → investigate before more experiments |

### Deployment Pipeline

After 09 v4 training completes:
1. **10a**: Deploy the new model at 256×256, push as 10a v21+
   - Must use `stage2.reid.vehicle.input_size=[256,256]` override
   - Copy model from 09 v4 output dataset
2. **10b**: Stage 3 indexing (CPU), same as always  
3. **10c**: Association sweep with v52-baseline thresholds (sim=0.50, appearance=0.70, fic=0.50, aqe_k=3, gallery=0.48, orphan=0.38) PLUS a secondary sweep exploring sim_thresh 0.45-0.55
   - **Important**: Do NOT enable AFLink — it's a dead end on CityFlowV2

---

## Experiment A: Color-Safe Augmentations + TripletLoss

### When to Run
- Run ONLY after Experiment B results are in
- If B shows CircleLoss is the culprit: run A as designed (color-safe augs + TripletLoss)
- If B shows CircleLoss is innocent: run A with CircleLoss instead of TripletLoss (since CircleLoss is fine)
- If B is ambiguous: run A as designed (TripletLoss is the safer choice)

### Hypothesis
Geometric augmentations (GaussianBlur, RandomPerspective, wider RandomErasing) improve robustness to occlusion, perspective changes, and noise WITHOUT harming color-based identity cues. The color-destroying augmentations (RandomGrayscale, hue jitter) are the ones that cause same-model vehicles to become indistinguishable.

### Code Changes (09 notebook, Cell 12 — transforms)

```python
# Color-SAFE augmentations only
train_tf = T.Compose([
    T.Resize((H + 16, W + 16), interpolation=T.InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(p=0.5),
    T.Pad(10),
    T.RandomCrop((H, W)),
    T.ColorJitter(brightness=0.3, contrast=0.25, saturation=0.2, hue=0.0),  # stronger BUT no hue
    # NO RandomGrayscale — destroys color identity cues
    T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.2),  # KEEP — geometric
    T.RandomPerspective(distortion_scale=0.1, p=0.2),  # KEEP — geometric
    T.ToTensor(),
    T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    T.RandomErasing(p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3), value="random"),  # KEEP wider range
])
```

**Changes vs current (augoverhaul) notebook:**
- `ColorJitter hue=0.05` → `hue=0.0` (remove hue jitter)
- Remove `T.RandomGrayscale(p=0.1)` line
- Keep GaussianBlur, RandomPerspective, wider RandomErasing

### Code Changes (09 notebook, Cell 20 — loss instantiation)

Replace CircleLoss with TripletLoss:

```python
    ce_loss = CrossEntropyLabelSmooth(num_classes, 0.05).to(DEVICE)
    tri_loss = TripletLossHardMining(margin=0.3).to(DEVICE)  # REVERT to baseline
    ctr_loss = CenterLoss(num_classes, 768).to(DEVICE)
    CENTER_WEIGHT = 5e-4
    CENTER_START = 15
```

### Code Changes (09 notebook, Cell 20 — training loop)

Replace `circle_loss_fn(f, pids)` with `tri_loss(f, pids)`:

```python
            with torch.amp.autocast("cuda"):
                out = model(imgs, cam_ids=cams)
                if len(out) == 3:
                    c, f, jc = out
                    loss = ce_loss(c, pids) + tri_loss(f, pids) + 0.5 * ce_loss(jc, pids)
                else:
                    c, f = out
                    loss = ce_loss(c, pids) + tri_loss(f, pids)
```

### Code Changes (09 notebook, Cell 24 — export metadata)

Fix the tricks array to reflect actual training:
```python
"tricks": ["SIE", "JPM", "BNNeck", "CE+LS(0.05)", "TripletLoss(m=0.3)",
            f"CenterLoss(delayed@ep{CENTER_START})", "CosLR", "RE", "CLIP-norm",
            "AugV1-colorsafe", f"EMA({EMA_DECAY})", f"LLRD({llrd_factor})"],
```

### Success Criteria

| Outcome | mAP | MTMC IDF1 | Interpretation |
|---------|:---:|:---------:|----------------|
| Best case | ≥ 80.5% | ≥ 0.778 | Color-safe augs improve both metrics → **new default recipe** |
| mAP up, MTMC same | ≥ 80.5% | 0.770-0.778 | Safe augs help mAP without hurting MTMC → cautious improvement |
| mAP up, MTMC down | ≥ 80.5% | < 0.770 | Even geometric augs hurt MTMC → augmentations are a dead end entirely |
| mAP same/down | < 80% | any | Safe augs don't help → stick with baseline |

---

## Decision Tree

```
                    Experiment B: Original augs + CircleLoss
                    ┌─────────────────┴─────────────────┐
                    │                                    │
           MTMC IDF1 < 0.755                    MTMC IDF1 ≥ 0.770
           (CircleLoss hurts)                   (CircleLoss fine)
                    │                                    │
        ┌───────────┴──────────┐              ┌──────────┴──────────┐
        │                      │              │                     │
  Run Experiment A         CONCLUSION:    Run Experiment A'      CONCLUSION:
  (color-safe augs      CircleLoss is     (color-safe augs    Augmentations
   + TripletLoss)       confirmed dead    + CircleLoss)       caused all the
        │               end for MTMC          │               damage
        │                                     │
   ┌────┴────┐                           ┌────┴────┐
   │         │                           │         │
 ≥0.778    <0.770                      ≥0.778    <0.770
   │         │                           │         │
 NEW      Augs also                   NEW      Augs also
 DEFAULT  hurt → stick               DEFAULT   hurt → stick
 RECIPE   with baseline              RECIPE    with baseline
```

### Edge Case: MTMC IDF1 in 0.755-0.770 (ambiguous zone)

If Experiment B produces MTMC IDF1 between 0.755 and 0.770, both CircleLoss and augmentations contribute to the regression. In this case:
- CircleLoss costs ~1-2pp, augmentations cost ~3-4pp
- Still run Experiment A (with TripletLoss) to confirm the augmentation contribution
- If A at 0.778+: both effects are real, deploy A's recipe
- If A at 0.770-0.778: the damage is mainly from augmentations, minor from CircleLoss

---

## Execution Checklist

### Phase 1: Experiment B (09 v4)

- [ ] Wait for 09 v3 to complete
- [ ] Edit 09 notebook: revert Cell 12 transforms to baseline
- [ ] Keep Cell 20 loss as CircleLoss (no change)
- [ ] Fix Cell 24 metadata to say "CircleLoss" not "TripletLoss" in tricks
- [ ] Update `variant` string to `"v4_circleloss_ablation"`
- [ ] Push: `kaggle kernels push -p notebooks/kaggle/09_vehicle_reid_cityflowv2/`
- [ ] Monitor: `python scripts/kaggle_logs.py 09-vehicle-reid-cityflowv2 --tail 20`
- [ ] After completion: check mAP and R1 from logs
- [ ] Push 10a with new model → 10b → 10c
- [ ] Record MTMC IDF1 result and update `docs/findings.md`

### Phase 2: Experiment A (09 v5, conditional)

- [ ] Based on Experiment B results, decide which variant of A to run (TripletLoss or CircleLoss)
- [ ] Edit 09 notebook: apply color-safe augmentations (Cell 12) + chosen loss (Cell 20)
- [ ] Push as 09 v5
- [ ] Same 10a→10b→10c deployment pipeline
- [ ] Record results and update `docs/findings.md`

---

## Cost Estimate

| Step | Time | GPU |
|------|:----:|:---:|
| 09 v4 training | ~2h | Kaggle P100 |
| 10a v21 deployment | ~30min | Kaggle P100 |
| 10b indexing | ~5min | Kaggle CPU |
| 10c association sweep | ~15min | Kaggle CPU |
| **Total per experiment** | **~3h** | |
| **Both experiments** | **~6h** | Sequential |

---

## Risk Mitigation

1. **Model file naming**: Each experiment must export with a DISTINCT filename to avoid confusion:
   - B: `vehicle_transreid_vit_base_cityflowv2_circleloss_ablation.pth`
   - A: `vehicle_transreid_vit_base_cityflowv2_colorsafe_augs.pth`

2. **EMA**: Keep EMA code in place but don't deploy the EMA model (it failed at decay=0.9999). Only evaluate the base model.

3. **Dataset naming**: Use distinct Kaggle dataset slugs:
   - B: `09-vehicle-reid-cityflowv2-circleloss-ablation`
   - A: `09-vehicle-reid-cityflowv2-colorsafe-augs`

4. **Don't change anything else**: Same optimizer (AdamW), same LR schedule (cosine, 120 epochs, warmup 10), same backbone LR (1e-4), head LR (1e-3), LLRD (0.75), weight decay (5e-4), batch size (64, PK 16×4), image size (256×256). The ONLY changes should be augmentations and/or loss.