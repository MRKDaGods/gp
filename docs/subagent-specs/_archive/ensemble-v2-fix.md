# Ensemble v2 Fix — Revert 10a to v80-Compatible Features

## Root Cause

10a was rebuilt with the v82 feature pipeline: 384px input, `concat_patch=true`, `power_norm=0.5`, and CamTTA enabled.
The stage 4 association settings from v80 were tuned for 256px CLS-only features, which caused a baseline regression of about 1.3pp.

## Fix: Revert 10a Features, Keep Vehicle2 Ensemble

- Revert to 256px TransReID weights: `transreid_cityflowv2_best.pth`
- Revert `input_size` to `[256,256]`
- Disable `concat_patch`
- Disable `power_norm` by setting `alpha=0.0`
- Disable CamTTA
- Disable `color_augment`
- Use the original PCA path: `pca_transform.pkl`
- Keep `vehicle2.enabled=true` for the ensemble

## Expected Outcome

- Baseline without fusion: about 0.784, matching v80
- With fusion: about 0.784 to 0.789