# Extended MVDeTr Training — Design Spec

**Date**: 2026-03-30
**Status**: Ready for implementation
**Target**: 12a notebook v12+

## 1. Current State

The current WILDTRACK detector training flow lives in [notebooks/kaggle/12a_wildtrack_mvdetr/12a_wildtrack_mvdetr.ipynb](../../notebooks/kaggle/12a_wildtrack_mvdetr/12a_wildtrack_mvdetr.ipynb). The training cell around notebook lines 184-211 clones MVDeTr from `https://github.com/hou-yz/MVDeTr.git`, changes into the cloned repo, and launches training with:

- `python main.py -d wildtrack`
- `--arch resnet18`
- `--world_feat deform_trans`
- `--use_mse false`
- `--epochs 10`
- `--batch_size 1`
- `--num_workers 2`
- `--lr 5e-4`
- `--world_reduce 4`
- `--world_kernel_size 10`
- `--img_reduce 12`
- `--img_kernel_size 10`
- `--dropout 0.0`
- `--dropcam 0.0`

The notebook is wired to the Kaggle-mounted WILDTRACK dataset and prepares the local repo-side dataset structure via `scripts/prepare_dataset.py` before training. After training, the notebook discovers the latest MVDeTr run directory, expects `MultiviewDetector.pth` and `test.txt`, then feeds detections into the repo-native postprocessing flow:

- Projection and world-coordinate tracking via [src/stage_wildtrack_mvdetr/pipeline.py](../../src/stage_wildtrack_mvdetr/pipeline.py)
- Ground-plane evaluation via [src/stage5_evaluation/ground_plane_eval.py](../../src/stage5_evaluation/ground_plane_eval.py)

The integration cell around notebook lines 231-248 calls `run_stage_wildtrack_mvdetr(...)`, while the evaluation cell around notebook lines 252-276 calls `evaluate_wildtrack_ground_plane(...)`.

Current postprocessing behavior is already stable:

- [src/stage_wildtrack_mvdetr/pipeline.py](../../src/stage_wildtrack_mvdetr/pipeline.py) lines 1-148 load MVDeTr `test.txt`, normalize WILDTRACK frame IDs, convert grid coordinates to world centimeters, and track detections with Hungarian matching.
- [src/stage5_evaluation/ground_plane_eval.py](../../src/stage5_evaluation/ground_plane_eval.py) lines 1-220 implement the correct WILDTRACK ground-plane protocol using calibration back-projection plus MODA/MODP-style evaluation.

Observed 12a v11 training behavior:

- Epoch 1: MODA 69.4%
- Epoch 5: MODA 89.4%
- Epoch 10: MODA 92.0%
- Runtime: about 2.5 hours total, or about 15 minutes per epoch
- Artifact outputs: `MultiviewDetector.pth`, `test.txt`, and run logs

This already exceeds the paper-reported 91.5% MODA, but the curve was still improving at epoch 10, so the current training horizon appears prematurely capped.

## 2. Goal

Increase WILDTRACK detector quality beyond the current 12a v11 result by extending training duration, adding regularization through augmentation, improving learning-rate scheduling, testing a stronger backbone, and saving the best validation checkpoint instead of relying only on the final epoch artifact.

The primary objective is to improve or at least stabilize validation MODA beyond 92.0% while keeping runtime within Kaggle's 12-hour GPU limit and preserving compatibility with the downstream WILDTRACK projection and evaluation flow.

## 3. Changes

### 3.1 Extended Training (10 → 25 Epochs)

The current notebook hard-codes `EPOCHS = 10` in the training cell at [notebooks/kaggle/12a_wildtrack_mvdetr/12a_wildtrack_mvdetr.ipynb](../../notebooks/kaggle/12a_wildtrack_mvdetr/12a_wildtrack_mvdetr.ipynb). That was sufficient to surpass the paper, but the observed curve still rose materially between epochs 5 and 10.

Proposed change:

- Increase training from 10 to 25 epochs.
- Keep all other baseline hyperparameters identical for the first extended run.
- Treat 25 epochs as the default new baseline, not as a sweep value.

Why 25 epochs:

- 25 epochs at about 15 minutes per epoch is about 6.25 hours.
- Allowing about 30 minutes for clone, environment setup, and dataset prep still keeps the run comfortably below Kaggle's 12-hour limit.
- WILDTRACK is small, roughly a few hundred annotated frames, so going far beyond 25 epochs increases overfitting risk and reduces iteration speed.

Expected behavior:

- If the epoch-10 improvement trend continues, epoch 15-20 should still yield measurable gains or at least a better best checkpoint.
- If validation saturates early, the best-checkpoint logic in Section 3.5 will prevent a longer run from hurting the final exported model.

### 3.2 Data Augmentation

The current 12a notebook does not pass any explicit augmentation flags in the training command. The spec should assume that augmentation support may already exist upstream in MVDeTr, but the implementer must verify the real CLI and dataset-path support before editing the notebook.

Required investigation before implementation:

- Inspect `main.py` in the cloned MVDeTr repo for supported train-time augmentation arguments.
- Inspect the WILDTRACK dataset loader and any `MultiviewDetector` dataset classes/transforms for augmentation hooks.
- Confirm whether augmentations are globally applied, per-camera applied, or ground-plane aware.

Recommended augmentation options, in priority order:

1. Horizontal flip augmentation.
2. Color jitter covering brightness, contrast, and saturation.
3. Random crop and/or resize augmentation if the upstream loader already supports it.

Rationale:

- Horizontal flip is a low-risk, high-value augmentation for person detection in multi-view settings as long as ground-plane labels remain consistent under the image transform path used by MVDeTr.
- Color jitter is useful for camera-to-camera illumination and white-balance variation.
- Crop/resize augmentation can improve robustness but is more likely to interfere with multi-view geometry if implemented incorrectly.

Implementation requirement:

- Do not invent new augmentation semantics in the notebook first.
- Prefer enabling existing upstream MVDeTr flags if they already exist.
- If the repo lacks exposed CLI flags, the implementation should document any required upstream patch to the dataset/transform path separately.

### 3.3 Learning Rate Schedule

The current command uses a constant learning rate of `5e-4` for 10 epochs. That is acceptable for a short run, but it is not ideal for a 25-epoch training horizon.

Required source inspection before implementation:

- Check MVDeTr `main.py` for scheduler-related arguments such as `--lr_scheduler`, `--step_size`, `--milestones`, `--gamma`, or cosine options.
- Check whether warmup already exists implicitly in the optimizer setup or training loop.

Preferred schedule options:

1. Cosine annealing across 25 epochs.
2. If cosine is unavailable, step decay with LR drops around epochs 15 and 20.
3. If supported, add 1-2 epochs of warmup to reduce early optimization instability.

Recommended policy:

- Keep base LR at `5e-4` initially.
- Use cosine annealing if the upstream code already supports it.
- Otherwise, use a simple step schedule such as:
  - epoch 1-14: `5e-4`
  - epoch 15-19: `2.5e-4`
  - epoch 20-25: `1e-4` to `1.25e-4`

Rationale:

- The current training curve suggests the model is not yet fully converged by epoch 10.
- A decayed LR over a longer run should improve late-epoch refinement and reduce the risk of bouncing around the optimum.

### 3.4 ResNet34 Backbone

The current notebook uses `--arch resnet18`. This should remain the reference baseline, but the next architecture candidate should be `resnet34`.

Why ResNet34:

- It is the same family as ResNet18, so the interface and training semantics are likely compatible if MVDeTr already uses standard torchvision-style backbones.
- It offers a straightforward capacity increase without requiring a full architecture redesign.
- It should add compute, but not enough to break the Kaggle runtime budget for 25 epochs.

Required verification before implementation:

- Confirm that the upstream MVDeTr repo accepts `--arch resnet34` in `main.py`.
- Confirm there are no backbone-specific assumptions elsewhere in the model builder.

Recommended experiment order:

1. Run `resnet18` for 25 epochs as the direct extension of the current best run.
2. Run `resnet34` for 25 epochs as the primary capacity-upgrade experiment.
3. If Kaggle time becomes tight, prioritize `resnet34` only after the `resnet18` 25-epoch baseline is reproduced.

Expectation:

- ResNet34 may improve feature quality and final validation MODA modestly.
- The main tradeoff is longer epoch time. If per-epoch runtime rises substantially, keep the best-checkpoint logic and treat ResNet34 as a higher-cost variant rather than the default.

### 3.5 Best Checkpoint Saving

The current workflow only relies on the final exported checkpoint `MultiviewDetector.pth`. That is fragile when training is extended because the best validation epoch may occur before the final epoch.

Proposed behavior:

- After each validation pass, compare the current validation MODA against the best value seen so far.
- If the new MODA exceeds `best_moda`, save `MultiviewDetector_best.pth`.
- Keep the final checkpoint as `MultiviewDetector.pth` for parity with the existing notebook flow.

Required implementation behavior:

- Save both the final and best checkpoints.
- Log the epoch index and MODA value associated with the best checkpoint.
- If the upstream MVDeTr training loop already has a checkpoint hook, extend it rather than duplicating save logic in the notebook.

Why this matters:

- A 25-epoch run increases the chance of mild overfitting after the optimum.
- The downstream 12a notebook currently selects artifacts by filename and latest run directory, so explicit best-checkpoint persistence is required if later cells are to consume the best model intentionally.

## 4. Implementation Plan

The implementation should stay localized to the 12a notebook plus, if needed, a small upstream MVDeTr patch in the cloned training code.

Step-by-step plan:

1. Inspect the current training cell in [notebooks/kaggle/12a_wildtrack_mvdetr/12a_wildtrack_mvdetr.ipynb](../../notebooks/kaggle/12a_wildtrack_mvdetr/12a_wildtrack_mvdetr.ipynb) around lines 184-211.
2. Change the notebook training configuration from `EPOCHS = 10` to `EPOCHS = 25`.
3. Before adding flags, inspect MVDeTr `main.py` and the WILDTRACK dataset loader in the cloned repo to confirm which augmentation options are actually supported.
4. If supported upstream, extend the notebook `TRAIN_ARGS` to enable horizontal flip and color jitter first, with crop/resize only if the data loader already implements it safely.
5. Inspect MVDeTr optimizer and scheduler setup, then add a cosine scheduler or a step decay schedule appropriate for 25 epochs.
6. Add a notebook-level switch or experiment variable for `arch`, defaulting to `resnet18` and allowing `resnet34` without rewriting the training cell.
7. Modify the MVDeTr training loop so validation MODA is tracked per epoch and the best checkpoint is saved as `MultiviewDetector_best.pth`.
8. Update the artifact-discovery cell in the notebook so it reports both the final checkpoint and best checkpoint when present.
9. Keep the downstream projection step unchanged. The integration path through [src/stage_wildtrack_mvdetr/pipeline.py](../../src/stage_wildtrack_mvdetr/pipeline.py) lines 1-148 and [src/stage5_evaluation/ground_plane_eval.py](../../src/stage5_evaluation/ground_plane_eval.py) lines 1-220 does not need a design change.
10. Run two experiments if budget allows:
    - `resnet18`, 25 epochs, scheduler enabled, augmentations enabled
    - `resnet34`, 25 epochs, same training recipe

Notebook cell mapping:

- Training setup and launch: notebook lines 184-211
- Artifact discovery and checkpoint selection: notebook lines immediately after the training cell
- Projection into repo-native WILDTRACK stage: notebook lines 231-248
- Ground-plane evaluation: notebook lines 252-276

## 5. Expected Impact

Primary expected gains:

- Extending from 10 to 25 epochs should improve best validation MODA by a modest but meaningful margin if the epoch-10 curve had not converged.
- A better schedule should improve late-epoch stability versus constant-LR training.
- Best-checkpoint saving should reduce regression risk even if the final epoch is slightly worse.

Reasonable quantitative expectations:

- `resnet18`, 25 epochs: likely modest gain over 92.0% MODA, with a realistic target in the low 92s to low 93s if the trend continues.
- `resnet34`, 25 epochs: potential additional improvement beyond the extended `resnet18` run, but with higher runtime uncertainty.
- Best-checkpoint saving: expected to improve result reliability even if mean final-epoch performance does not increase.

The main expected benefit is not a guaranteed large absolute jump, but a more complete optimization pass that extracts the remaining value from a setup that was still improving at epoch 10.

## 6. Risks

- Overfitting on WILDTRACK's small training set could flatten or reverse gains after the optimal epoch.
- Some augmentation types may break multiview consistency if MVDeTr's dataset path assumes geometry-preserving transforms only.
- ResNet34 may increase runtime enough to reduce experiment throughput, even if it still fits under 12 hours.
- MVDeTr may not expose scheduler or augmentation flags at the CLI level, requiring a small upstream code patch rather than a notebook-only change.
- Best-checkpoint saving requires careful alignment with the actual validation metric computation path. If MODA is logged externally rather than returned in-process, the save hook may need a deeper training-loop edit.

## 7. Measurement

The implementation should be judged with the existing WILDTRACK evaluation path, not by notebook loss alone.

Required measurements:

1. Compare epoch-10 best MODA versus epoch-25 best MODA on the same backbone.
2. Compare final checkpoint versus best checkpoint for the same run.
3. Compare `resnet18` 25-epoch versus `resnet34` 25-epoch under the same scheduler and augmentation recipe.
4. Confirm that downstream projection and evaluation remain stable using:
   - [src/stage_wildtrack_mvdetr/pipeline.py](../../src/stage_wildtrack_mvdetr/pipeline.py)
   - [src/stage5_evaluation/ground_plane_eval.py](../../src/stage5_evaluation/ground_plane_eval.py)

Success criteria:

- The notebook still completes within the Kaggle runtime limit.
- The best validation checkpoint is preserved explicitly.
- Validation MODA at best epoch meets or exceeds the current 92.0% baseline.
- Any runtime increase is justified by measurable MODA improvement.