# Person Pipeline - WILDTRACK Reproduction

> Canonical headline: ground-plane IDF1 0.9467084639498433 from the 12b Kalman sweep. This recipe is Kaggle-first; do not run MVDeTr training or detector inference locally.

## Inputs

- Dataset: WILDTRACK, mounted in the canonical kernels as `aryashah2k/large-scale-multicamera-detection-dataset`.
- Weights dataset: `gumfreddy/mtmc-weights` for person ReID reference weights.
- Detector kernel: `gumfreddy/12a-wildtrack-mvdetr-training`.
- Tracking/eval kernel: `gumfreddy/12b-wildtrack-mvdetr-tracking-reid`, with 12a as a kernel source.

## Canonical Kaggle Chain

1. MVDeTr detector training/export: `notebooks/kaggle/12a_wildtrack_mvdetr/12a_wildtrack_mvdetr.ipynb`.
2. Ground-plane tracking + ReID merge sweep: `notebooks/kaggle/12b_wildtrack_tracking_reid/12b_wildtrack_tracking_reid.ipynb`.

## Detector Evidence

The 12a notebook trains MVDeTr ResNet18 for 25 epochs and exports `MultiviewDetector.pth` from the latest run. The downloaded kernel log verifies:

- Cell 6 (`#VSC-698ce737`) launches MVDeTr training.
- Cell 7 (`#VSC-46fae1e7`) resolves `MultiviewDetector.pth`.
- Cell 8 (`#VSC-731816ad`) copies it to `/kaggle/working/MultiviewDetector.pth`.
- Epoch-20 log line: `moda: 92.1%, modp: 81.7%, prec: 95.7%, recall: 96.4%`.
- Final loaded-model log line: `moda: 91.3%, modp: 81.8%, prec: 94.7%, recall: 96.6%`.

Because the same notebook log shows repo conversion/evaluation skipped, treat 92.1% as an epoch-line detector result and 91.3% as the verified final loaded-model line for the exported run.

## Tracking Evidence

The downloaded 12b `tracking_sweep_best.json` verifies the selected Kalman operating point:

```yaml
tracker: kalman
max_age: 2
min_hits: 2
distance_gate: 25.0
max_euclidean_cm: 200.0
q_std: 5.0
r_std: 10.0
interpolation_enabled: true
interpolation_max_gap: 1
detection_conf_threshold: 0.25
```

Verified metrics from `evaluation_summary.json`:

```text
IDF1: 0.9467084639498433
MODA: 0.9002100840336135
Precision: 0.9480249480249481
Recall: 0.957983193277311
ID switches: 5
```

The commonly cited `distance_gate=20.0`, `q_std=8.0`, `r_std=8.0` values are only the 12b baseline values; they are not the selected best.

## Push / Monitor Commands

```powershell
$env:KAGGLE_API_TOKEN = (Get-Content $HOME\.kaggle\gumfreddy_access_token -Raw).Trim()
kaggle kernels push -p notebooks/kaggle/12a_wildtrack_mvdetr
kaggle kernels push -p notebooks/kaggle/12b_wildtrack_tracking_reid
kaggle kernels status gumfreddy/12b-wildtrack-mvdetr-tracking-reid
```

Metadata-only checks used for this verification:

```powershell
kaggle datasets files gumfreddy/mtmc-weights
kaggle kernels output gumfreddy/12a-wildtrack-mvdetr-training --file-pattern '^ground_plane_eval_summary\.json$' -p <tmp>
kaggle kernels output gumfreddy/12b-wildtrack-mvdetr-tracking-reid --file-pattern '^(evaluation_summary|tracking_sweep_best|reid_merge_sweep_best)\.json$' -p <tmp>
```

No local detector inference, tracker run, or checkpoint download is required for the documentation trail. Full background is in [docs/findings.md](findings.md) and [docs/experiment-log.md](experiment-log.md).