# Phase 8 Kaggle Toggle Audit

Date: 2026-05-27

## Footer Placement

- Stage 0 Upload: no toggle. Correct; upload is a local/browser operation.
- Stage 1 Detection: has `ExecutionTargetToggle stage={1}` and sends `kaggle` in `runStage(1)` when selected.
- Stage 2 Selection: no toggle after this audit. Fixed; selection is frontend-only.
- Stage 3 Inference: keeps two execution controls for backend Stage 2 feature extraction and backend Stage 3 indexing. Both read `useStageExecutionStore` and send `kaggle` in `runStage(2)` / `runStage(3)` when selected.
- Stage 4 Timeline: has `ExecutionTargetToggle stage={4}`. Fixed in this audit so every `runStage(4)` association dispatch includes the persisted Kaggle payload when selected.
- Stage 5 Refinement: no toggle. Correct; local-only refinement UI and re-search flow.
- Stage 6 Output: no toggle. Correct; final visualization/export flow is local-only.

## Persistence And Surfaces

- `useStageExecutionStore` persists `stageExecutionTargets` under `mtmc-stage-execution-targets` and exposes `getStageExecutionTarget(stage)` / `setStageExecutionTarget(stage, target)`.
- Sidebar cloud/server indicators read the same persisted stage key via `getStageExecutionTarget(stage.id)`.
- `ExecutionTargetToggle` accepts a `stage` prop, reads/writes `useStageExecutionStore`, shows the credentials warning for Kaggle without user credentials, and opens `KaggleCredentialsModal` from its configure link.
- `runStage()` accepts the optional `kaggle` payload in `RunStageRequest`; the frontend now supplies it for Stages 1, 2 feature extraction, 3 indexing, and 4 association when their footer toggle is set to Kaggle.