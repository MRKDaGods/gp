# ReID Ensemble Training Plan — AIC21/22 Winning Recipe

## Executive Summary

The remaining performance gap is not in association tuning. That path is effectively exhausted: 225+ association configurations have already been tested and all cluster within roughly 0.3pp of the current optimum. The current single-model system is therefore bottlenecked by feature quality and feature diversity rather than graph logic.

The strongest existing model, the ViT-B/16 CLIP 256px backbone, already delivers 80.14% mAP on CityFlowV2 evaluation and should remain unchanged. The practical path to close the remaining 7.36pp gap to AIC22 first place is to replicate the AIC21/AIC22 winner pattern: combine multiple diverse ReID backbones and fuse them at score level rather than relying on a single embedding space.

This specification defines a concrete four-part implementation plan:

1. Keep the current ViT model unchanged as the primary anchor model.
2. Retrain ResNet101-IBN-a using the DMT-style recipe instead of the failed VeRi-776 intermediate strategy.
3. Add ResNeXt101-IBN-a as the third backbone to match the winning 3-model family used by AIC winners.
4. Implement score-level ensemble fusion in the existing stage 4 association pipeline.

The intent is not to redesign the full pipeline. The intent is to add the minimum new training and fusion infrastructure needed to reproduce the only remaining plausible improvement path.

## Section 1: Existing ViT-B/16 CLIP 256px (Model 1) — KEEP AS-IS

The current ViT-B/16 CLIP 256px model is already strong enough to serve as the anchor model for the ensemble:

- CityFlowV2 evaluation mAP: 80.14%
- Input resolution: 256x256
- Deployment status: already integrated and validated in the current pipeline
- Ensemble role: primary feature extractor and dominant similarity source

This model should not be retrained, resized, or reconfigured as part of this effort.

Reasons to keep it unchanged:

- It is already the best-performing model in the project.
- The 384px direction has already been tested and is a confirmed dead end at the MTMC level.
- Single-model DMT camera-aware variants were harmful in this project.
- Changing the anchor model would introduce unnecessary risk while the main need is diversity, not replacement.

Implementation guidance:

- Preserve the current checkpoint, backbone, preprocessing, PCA, and post-processing settings.
- Treat the ViT embedding stream as the reference stream for ensemble ablations.
- During weight sweeps, bias the initial fusion weights toward ViT because it is the strongest individual model.

No code changes are required for this model beyond allowing it to coexist with two additional models in stage 2 and stage 4.

## Section 2: ResNet101-IBN-a (Model 2) — Fix Training Recipe

### 2.1 Root Cause Analysis

The failed 09e and 09f path did not fail because ResNet101-IBN-a is inherently weak. It failed because the training recipe diverged from the winning AIC recipe in the most important place: domain adaptation.

Observed outcomes:

- Direct ImageNet to CityFlowV2 training: 52.77% mAP
- VeRi-776 pretraining then CityFlowV2 fine-tuning: 42.7% mAP

This establishes the critical diagnosis: VeRi-776 is the wrong intermediate objective for this target domain.

Why 09f failed:

- VeRi-776 pretraining is not what the AIC winners used. Their second-stage adaptation is targeted to the deployment domain, not a generic vehicle ReID benchmark.
- A supervised intermediate stage on VeRi-776 encourages the backbone to specialize to VeRi-specific appearance statistics, camera layout, and vehicle distribution.
- That specialization makes later fine-tuning less adaptable to CityFlowV2, especially when the target domain requires cross-camera invariance and dataset-specific compensation.
- The DMT recipe does not stop at supervised training. Its key contribution is pseudo-labeled domain adaptation driven by clustering on target-domain features.

The correct conclusion is that the current 52.77% direct baseline is not a failure. It is the correct starting point. The missing pieces are:

- DMT-style stage 2 pseudo-label domain adaptation
- Better alignment with the winning optimizer and scheduler recipe
- Slight architecture cleanup around pooling and loss configuration

This means the ResNet effort should be framed as a repair of training methodology, not a replacement of architecture.

### 2.2 Proposed Training Recipe

The proposed recipe has two stages. Stage 1 establishes a supervised target-domain baseline. Stage 2 performs DMT-style unsupervised domain adaptation on the full target data.

#### Stage 1 — Supervised CityFlowV2 (09d-v2)

This stage should retrain ResNet101-IBN-a from ImageNet-pretrained IBN weights directly on CityFlowV2, matching the DMT recipe as closely as practical.

Target configuration:

```yaml
model:
  backbone: resnet101_ibn_a
  pretrained: imagenet
  pretrained_path: https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_a-59ea0ac6.pth
  last_stride: 1
  pooling: gempoolP
  neck: bnneck

input:
  size_train: [384, 384]
  size_test: [384, 384]
  hflip_prob: 0.5
  pad: 10
  random_crop: true
  random_erasing_prob: 0.5

loss:
  sampler: softmax_triplet
  id_loss: cross_entropy_label_smooth
  label_smooth_epsilon: 0.1
  triplet_margin: 0.3
  id_loss_weight: 1.0
  triplet_loss_weight: 1.0

optim:
  name: adam
  base_lr: 3.0e-4
  weight_decay: 5.0e-4
  weight_decay_bias: 5.0e-4
  bias_lr_factor: 2.0

scheduler:
  name: cosine
  max_epochs: 100
  warmup_epochs: 10
  warmup_lr: 1.0e-6
  min_lr: 1.0e-5
  cooldown_epochs: 10

train:
  ims_per_batch: 64
  num_instance: 16
  fp16: true
```

Required training characteristics:

- Backbone: ResNet101-IBN-a with `last_stride=1`
- Input size: 384x384
- Pooling: GeM or `gempoolP`
- Neck: BNNeck enabled
- Loss: cross-entropy with label smoothing plus hard triplet loss
- Optimizer: Adam, not AdamW
- Scheduler: cosine with DMT-style warmup and cooldown
- Mixed precision: enabled for Kaggle efficiency

Expected outcome:

- mAP in the 50-55% range would validate parity with the current best direct baseline
- If stage 1 falls materially below 50%, stop and debug before stage 2

Implementation notes:

- Keep the stage 1 recipe simple. Do not add circle loss yet.
- Avoid introducing extra complexity such as camera-aware BN, multi-scale training, or exotic augmentations during the first reproduction.
- The goal of stage 1 is to produce a stable supervised checkpoint suitable for pseudo-label adaptation.

#### Stage 2 — Domain Adaptation (09d-stage2)

This is the essential missing piece. The notebook must implement a DMT-style unsupervised domain adaptation loop on the CityFlowV2 target domain.

Target pipeline:

1. Load the best stage 1 checkpoint.
2. Extract features for all available CityFlowV2 data using flip augmentation.
3. Apply camera compensation using the DMT `compute_P2` style FIC transform with `la=0.0005`.
4. Compute pairwise distances and cluster with DBSCAN.
5. Assign hard pseudo-labels to clustered samples.
6. Build a pseudo-labeled dataloader.
7. Train for 300 iterations per epoch.
8. Re-extract features and re-cluster every 3 epochs.
9. Repeat until stage 2 convergence.

Recommended stage 2 hyperparameters:

```yaml
stage2_uda:
  enabled: true
  dbscan:
    eps: 0.55
    min_samples: 10
    metric: precomputed
  fic:
    enabled: true
    lambda: 5.0e-4
  recluster_every_epochs: 3
  train_iters_per_epoch: 300
  flip_test: true
  smooth_track_features:
    enabled: false
```

Expected artifact:

- Final checkpoint name: `resnet101_ibn_a_2.pth`

Expected outcome:

- Improvement of roughly +5pp to +15pp mAP relative to stage 1 is the working target band
- The exact gain is uncertain, but stage 2 must beat the current 52.77% baseline to justify integration

Practical implementation requirement:

- The notebook should combine both stages in one end-to-end Kaggle workflow so stage 1 output flows directly into stage 2 without a manual handoff.

#### Optional Later Variant

Once the base DMT reproduction is working, add one controlled ablation:

- Replace cross-entropy ID loss with circle loss while keeping triplet loss unchanged

This should be treated as a second-pass experiment, not part of the first implementation. The first milestone is DMT parity, not loss-function exploration.

### 2.3 What's Different from 09e/09f

The table below should drive implementation decisions and code review. Any notebook drift back toward the 09e/09f pattern is likely to repeat the same failure.

| Aspect | Our 09e/09f | DMT Recipe | Why It Matters |
|--------|-------------|------------|----------------|
| Intermediate dataset | VeRi-776 | None beyond ImageNet before target adaptation | VeRi encourages dataset-specific shortcuts that do not transfer cleanly to CityFlowV2 |
| Domain adaptation | None | DBSCAN pseudo-label UDA | This is the main mechanism for closing the target-domain gap |
| Optimizer | AdamW | Adam | Reproducing the winning recipe reduces unnecessary confounders |
| Stage 2 exists? | No | Yes | The largest missing component in the failed training path |
| Pooling | GeM(p=3.0) | GeM or gempoolP | Small difference, but align with DMT defaults where possible |
| Data variants | None | Often multiple variants | Diversity helps ensembles, even when the backbone family overlaps |
| Camera compensation | None | FIC during clustering | Reduces camera bias in pseudo-label generation |

Immediate rule for implementation:

- Do not spend time improving VeRi-776 pretraining.
- Do not treat 09f as a base to refine.
- Start from the direct target-domain recipe and add DMT-style stage 2.

## Section 3: ResNeXt101-IBN-a (Model 3) — New Backbone

### 3.1 Why ResNeXt101-IBN-a

ResNeXt101-IBN-a is the highest-priority third backbone because it appears consistently in the winning AIC ensemble family while remaining operationally close to the ResNet101-IBN-a implementation.

Reasons to use it:

- It is used by the AIC21-MTMC winning Track 3 system.
- It is also used by the AIC22-MCVT system.
- It belongs to the same IBN backbone ecosystem, which simplifies implementation and checkpoint handling.
- It introduces meaningful feature diversity through grouped convolutions rather than duplicating the exact same representation family.

Architecture details to preserve:

- Backbone: `resnext101_ibn_a`
- Block layout: `[3, 4, 23, 3]`
- Cardinality: `32`
- Base width: `4`
- IBN configuration: `('a', 'a', 'a', None)`
- Output dimension: `2048`
- Last stride: `1`

Pretrained checkpoint source:

- https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnext101_ibn_a-6ace051d.pth

This model matters because the ensemble needs diversity, not just one stronger version of the same model. ResNeXt provides that diversity without forcing a completely different training stack.

### 3.2 Training Recipe

The training recipe should be identical to the ResNet101-IBN-a plan unless memory constraints require a smaller batch size.

Target configuration delta:

```yaml
model:
  backbone: resnext101_ibn_a
  pretrained: imagenet
  pretrained_path: https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnext101_ibn_a-6ace051d.pth
  last_stride: 1
  pooling: gempoolP
  neck: bnneck
```

Training policy:

- Use the exact same stage 1 supervised training recipe as Section 2.2.
- Use the exact same stage 2 UDA loop as Section 2.2.
- Only reduce batch size if Kaggle T4 VRAM requires it.

Expected artifact:

- Final checkpoint name: `resnext101_ibn_a_2.pth`

Recommended memory fallback:

- First try batch 64.
- If OOM occurs, reduce to batch 48.
- Preserve the rest of the recipe.

The implementation objective is not to optimize this model independently first. The objective is to create a diverse second CNN-family model that can add complementary signal to ViT and ResNet at fusion time.

### 3.3 Alternative 3rd Backbones (if ResNeXt doesn't fit in memory)

If ResNeXt101-IBN-a proves impractical on Kaggle hardware, use the following fallback order:

1. DenseNet169-IBN-a
2. SE-ResNet101-IBN-a
3. ResNeSt101

Rationale for this ranking:

- DenseNet169-IBN-a is the most architecture-diverse fallback used in DMT.
- SE-ResNet101-IBN-a keeps implementation close to the existing ResNet path while adding channel-attention variation.
- ResNeSt101 remains viable, but should be lower priority than the two options above unless pretrained weights and code support are already straightforward.

Do not choose a fallback unless ResNeXt is actually blocked by memory or implementation constraints. The primary target remains the exact winner-style three-model family.

## Section 4: Ensemble Fusion Strategy

### 4.1 Feature-Level Concatenation (Current Infrastructure)

The current stage 2 pipeline already supports ensemble-style feature concatenation through normalized embedding concatenation:

```text
norm(e1) ⊕ norm(e2)
```

This can be generalized to three models:

```text
norm(e_vit) ⊕ norm(e_resnet) ⊕ norm(e_resnext)
```

Operationally, this means:

- Each model emits its own embedding vector.
- Each embedding stream is independently normalized.
- Each stream can be reduced with its own PCA.
- The vectors are concatenated into a single higher-dimensional descriptor.

Illustrative dimensionality plan:

- ViT PCA output: 256D
- ResNet PCA output: 256D
- ResNeXt PCA output: 256D
- Final concatenated descriptor: 768D

Pros:

- Straightforward to integrate with existing embedding storage and indexing
- Keeps stage 4 mostly unchanged if treated as a single feature stream
- Easy to debug because the output is one descriptor per tracklet

Cons:

- Different backbone families may not align cleanly in a single vector space
- Higher dimension increases memory and can weaken nearest-neighbor geometry
- This departs from the AIC winner recipe, which fused scores rather than concatenated features

This path should remain available for ablation, but it should not be the default implementation target.

### 4.2 Score-Level Ensemble (DMT Approach — RECOMMENDED)

The recommended approach is score-level fusion. Each model independently produces similarities or distances, and stage 4 combines those scores after model-specific retrieval logic.

Existing project advantage:

- Stage 4 already has score fusion infrastructure through the current two-stream combination pattern:

```text
sim_combined = w1 * sim1 + w2 * sim2
```

Required extension:

```text
sim_combined = w1 * sim_vit + w2 * sim_resnet + w3 * sim_resnext
```

Recommended initial weighting:

- ViT: 0.50
- ResNet101-IBN-a: 0.25
- ResNeXt101-IBN-a: 0.25

Why score-level fusion is preferred:

- It matches the winner strategy more closely.
- Each backbone preserves its own representation geometry.
- Fusion weights can be tuned without retraining models.
- A weak or noisy model can be down-weighted rather than corrupting a shared embedding space.

Costs:

- Roughly triple feature extraction time
- Higher storage and FAISS memory usage
- More bookkeeping in stage 2, stage 3, and stage 4

Those costs are acceptable because local GPU execution is already out of scope and Kaggle execution is sequential for inference.

### 4.3 Recommended Approach

The implementation should default to score-level fusion with a clean configuration interface and leave feature concatenation as an optional fallback or ablation path.

Recommended execution flow:

1. Stage 2 extracts three embedding streams independently.
2. Stage 3 stores each stream separately and either builds separate FAISS indices or preserves embeddings for late fusion.
3. Stage 4 computes per-model similarity for each candidate pair.
4. Stage 4 combines the scores using a weighted average.
5. Ensemble weights are swept after the first full run.

First-pass weight sweep candidates:

```yaml
ensemble_weight_sweep:
  - [0.50, 0.25, 0.25]
  - [0.60, 0.20, 0.20]
  - [0.45, 0.30, 0.25]
  - [0.40, 0.30, 0.30]
  - [0.34, 0.33, 0.33]
```

Only after the three-model fusion beats the current single-model baseline should any additional rank-fusion or reranking complexity be introduced.

## Section 5: Kaggle Notebook Structure

### 5.1 New Notebooks Required

Two new training notebooks are required to produce the CNN ensemble checkpoints.

#### 1. 09g_vehicle_reid_resnet101ibn_dmt

Purpose:

- Train ResNet101-IBN-a using the two-stage DMT-style recipe.

Operational plan:

- Account: `ali369`
- GPU target: T4 16GB or P100
- Runtime estimate: 4-6 hours
- Output: `resnet101_ibn_a_2.pth`

Notebook responsibilities:

- Download or mount ImageNet-pretrained IBN weights
- Train stage 1 supervised CityFlowV2 model
- Save best checkpoint
- Run stage 2 pseudo-label adaptation using the best stage 1 checkpoint
- Save final stage 2 checkpoint and validation metrics

#### 2. 09h_vehicle_reid_resnext101ibn_dmt

Purpose:

- Train ResNeXt101-IBN-a with the same two-stage DMT-style recipe.

Operational plan:

- Account: `mrkdagods`
- GPU target: T4 or P100
- Runtime estimate: 5-7 hours
- Output: `resnext101_ibn_a_2.pth`

Notebook responsibilities:

- Mirror 09g as closely as possible
- Swap only the backbone and any batch-size fallback needed for memory
- Produce directly comparable metrics to 09g

The two notebooks should share the same structure so that later debugging and parameter synchronization are easy.

### 5.2 Modified Notebooks

Two existing pipeline notebooks must be extended after the CNN checkpoints exist.

#### 3. 10a_stages012

Current role:

- Runs stages 0, 1, and 2 and currently extracts vehicle features for the ViT model only.

Required changes:

- Load three ReID models rather than one.
- Extract embeddings independently for all three models.
- Preserve per-model embeddings through stage 2 outputs.
- Ensure the output format remains compatible with downstream stage 3 and stage 4 code.

Expected result:

- Each tracklet should carry multiple embedding streams rather than one fused vector by default.

#### 4. 10c_stages45

Current role:

- Runs association and evaluation.

Required changes:

- Extend stage 4 association to compute similarity from three embedding streams.
- Add configurable ensemble weights.
- Support weighted-average score fusion as the default fusion mode.

Expected result:

- Stage 4 can run either single-model or multi-model association from config, with identical surrounding pipeline structure.

### 5.3 Kaggle Chain Update

Current chain:

```text
10a (GPU) -> 10b (CPU) -> 10c (CPU)
```

Updated chain:

```text
09g + 09h (GPU, parallel) -> 10a (GPU, 3-model extraction) -> 10b (CPU) -> 10c (CPU, score fusion)
```

Implementation rule:

- Do not modify the high-level chain structure beyond adding the two prerequisite training notebooks and the new multi-model data flow.

This keeps operational complexity bounded while enabling the ensemble.

## Section 6: 10a Integration Details

### 6.1 Stage 2 Changes

Stage 2 should be extended to define multiple ReID models explicitly under configuration rather than implicitly through one checkpoint path.

Proposed config shape:

```yaml
stage2:
  models:
    primary:
      name: vehicle_reid_vit
      checkpoint: vit_cityflowv2_best.pth
      backbone: vit_base_patch16_224_TransReID
      input_size: [256, 256]
    secondary:
      name: vehicle_reid_resnet
      checkpoint: resnet101_ibn_a_2.pth
      backbone: resnet101_ibn_a
      input_size: [384, 384]
    tertiary:
      name: vehicle_reid_resnext
      checkpoint: resnext101_ibn_a_2.pth
      backbone: resnext101_ibn_a
      input_size: [384, 384]
```

Required processing changes:

- Instantiate one extractor per configured model.
- Run extraction independently for each model.
- Preserve each embedding stream separately in the stage 2 outputs.
- Apply model-specific PCA whitening per stream rather than one shared PCA.
- Keep flip augmentation behavior consistent across models.

Recommended output structure:

```yaml
tracklet:
  embeddings:
    vehicle_reid_vit: [...]
    vehicle_reid_resnet: [...]
    vehicle_reid_resnext: [...]
```

Implementation guidance:

- Do not concatenate by default during extraction.
- Keep the raw per-model streams available so stage 4 can choose score-level fusion.
- Preserve the current quality-aware crop selection and temporal pooling behavior unless a concrete model-specific incompatibility appears.

### 6.2 Stage 4 Changes

Stage 4 must be extended from two-stream combination logic to a general N-stream ensemble scoring path, with the immediate target of three streams.

Proposed config shape:

```yaml
stage4:
  association:
    ensemble:
      enabled: true
      weights: [0.5, 0.25, 0.25]
      fusion: weighted_average
```

Required association behavior:

- Compute similarity independently for ViT, ResNet, and ResNeXt.
- Apply the same candidate-pair filtering pipeline to each stream or ensure a shared candidate set before fusion.
- Combine scores using the configured weights.
- Preserve the ability to disable ensemble fusion and fall back to the primary model only.

Recommended pseudocode:

```python
scores = []
for model_name, weight in zip(model_names, weights):
    sim = compute_similarity(track_a.embeddings[model_name], track_b.embeddings[model_name])
    scores.append(weight * sim)

sim_combined = sum(scores)
```

Design constraints:

- Keep the fusion interface generic enough to support future fourth-model experiments.
- Ensure config validation catches mismatched model counts and weight lengths.
- Log per-model score statistics so weight sweeps are diagnosable.

## Section 7: Implementation Order & Dependencies

### Phase 1: ResNet101-IBN-a DMT Training (Week 1)

Execution sequence:

1. Create 09g notebook with stage 1 supervised training.
2. Run stage 1 on Kaggle using the `ali369` account.
3. Confirm stage 1 reaches the expected 50-55% mAP band.
4. Add stage 2 UDA loop to the same notebook.
5. Re-run end-to-end and produce `resnet101_ibn_a_2.pth`.
6. Evaluate on the CityFlowV2 evaluation split.

Go or no-go criterion:

- If stage 2 does not exceed 55% mAP, do not proceed to full integration without debugging the DMT port.

### Phase 2: ResNeXt101-IBN-a Training (Week 1-2, parallel)

Execution sequence:

1. Clone 09g into 09h.
2. Replace the backbone with `resnext101_ibn_a`.
3. Download the ResNeXt IBN pretrained weights.
4. Run stage 1 and then stage 2.
5. Evaluate on the same split and compare against ResNet101.

Primary success criterion:

- The final model should exceed 50% mAP and provide a useful complementary stream for ensemble fusion.

### Phase 3: Pipeline Integration (Week 2)

Execution sequence:

1. Add multi-model definitions to the stage 2 config.
2. Extend stage 2 extraction to emit three embedding streams.
3. Extend stage 4 association to support weighted score fusion.
4. Validate locally on a small dataset using CPU-only stages where applicable.
5. Update 10a for three-model extraction.
6. Update 10c for three-model fusion.

Acceptance criterion:

- A full dry run should complete with three embedding streams present and ensemble scoring enabled from config.

### Phase 4: Evaluation (Week 2-3)

Execution sequence:

1. Run the full Kaggle chain with all three models enabled.
2. Compare MTMC IDF1 against the current 77.5% single-model baseline.
3. Sweep ensemble weights.
4. If the ensemble clearly improves IDF1, then enable k-reciprocal reranking as a follow-up experiment.

Important policy:

- Do not enable reranking before the ensemble itself proves positive. Better features may make reranking useful again, but it should not be mixed into the first attribution test.

### Key Risk: VRAM on Kaggle T4

Practical memory expectations:

- ResNet101-IBN-a at 384x384 and batch 64 should fit within roughly 10-12GB.
- ResNeXt101-IBN-a at 384x384 and batch 64 may approach 14GB and may need batch 48.
- Stage 2 feature extraction runs in eval mode and should be manageable.
- Three-model extraction in 10a should be sequential rather than parallel, so runtime increases but peak memory remains bounded.

Mitigation plan:

- Keep one-model-at-a-time extraction.
- Drop ResNeXt batch size first if memory pressure appears.
- Avoid introducing multi-scale extraction during the first ensemble rollout.

### Key Risk: DMT Stage 2 Implementation Complexity

This is the highest implementation risk in the entire plan.

Required components that do not yet exist as a unified training loop:

- feature extraction over the full target dataset
- FIC-style camera compensation
- pairwise distance computation
- DBSCAN clustering
- pseudo-label assignment
- pseudo-labeled dataloader creation
- fixed-iteration training loop
- periodic re-clustering

Estimated implementation scope:

- Roughly 200 lines of clustering and adaptation logic, plus notebook orchestration

Mitigation plan:

- Implement ResNet101 first and validate stage 2 before duplicating the recipe to ResNeXt.
- Keep the stage 2 loop minimal and faithful to DMT before adding any project-specific enhancements.
- Log cluster counts, noise ratio, pseudo-label counts, and mAP after each re-clustering cycle.

### Success Criteria

| Metric | Threshold | Why |
|--------|:---------:|-----|
| ResNet101-IBN-a mAP (stage 2) | > 55% | Must beat the current 52.77% baseline to justify the DMT training path |
| ResNeXt101-IBN-a mAP (stage 2) | > 50% | Must be strong enough to contribute non-trivial ensemble signal |
| 2-model ensemble MTMC IDF1 | > 78.5% | Must clear the current 77.5% baseline by at least 1pp |
| 3-model ensemble MTMC IDF1 | > 80% | Marks meaningful progress toward the 84.86% SOTA target |
| Final with reranking | > 82% | Stretch goal once better multi-model features are in place |

These thresholds should be treated as decision gates, not just reporting targets. If the checkpoints do not reach these levels, the project should pause and debug before continuing to broader rollout.
