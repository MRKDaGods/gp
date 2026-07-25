# Phase 6 — joint multi-domain vehicle ReID retrain (D4 tier 2)

ONE joint domain-balanced retrain over four vehicle ReID domains (never
sequential fine-tuning), plus a cross-domain eval matrix against the frozen
VeRi-only `transreid_primary` checkpoint. All heavy work runs on Kaggle.

## Dataset reachability (scouted + probed 2026-07-25 — three of four obtainable)

| Domain | Source on Kaggle | Status |
| --- | --- | --- |
| VeRi-776 | `abhyudaya12/veri-vehicle-re-identification-dataset` (public) | proven mount (576 train ids / 37,778 imgs — verified against the local canonical copy) |
| VeRi-Wild | `mrkdagods/veriwild-train` (private, staged by Stage A) | **STAGED**: 277,797 train imgs / 30,671 ids in two tarbins, 0 missing |
| CityFlowV2 | GDrive archive `13wNJpS_Oaoe-7y5Dzexg_Ol7bKu1OWuC`, GT crops in-kernel | proven (v2 run: 14,065 crops / 184 ids from S01/S03/S04; zero S02 identity overlap) |
| VehicleID | `maphat/vehicleid` (public upload of the official archive) | **UNUSABLE** — every member is password-encrypted (probed via `mrkdagods/vehicleid-probe`); no other Kaggle copy exists. Kernels preflight and DROP the domain gracefully (recorded in provenance); restoring it needs the official PKU password — a user action. |

So the joint run is **3-domain** (VeRi-776 + CityFlowV2 + VeRi-Wild,
~330k imgs / ~31k ids) until a usable VehicleID copy exists.

## Stages

### A. `veriwild_prep/` — stage VeRi-Wild train images (CPU kernel)
`mrkdagods/veriwild-train-prep`: gdown the 23 rar parts, extract, tar the
train images from `train_list_start0.txt` into two ~8.5GB `.tarbin` halves
(opaque files — Kaggle must not auto-explode them into 277k mounted files),
upload as private dataset `mrkdagods/veriwild-train`. Falls back to two
one-tarbin datasets if the per-dataset cap rejects the pair.

### B. `train/` — the joint training kernel (T4)
`mrkdagods/athar-joint-reid-train`: recipe = canonical A5alpha Stream-1
TransReID ViT-B/16 CLIP @224 (JPM, BNNeck v15 routing, AdamW LLRD 0.75,
CE-LS + JPM aux + hard triplet, warmup+cosine, seed 0), with three documented
deviations for the joint setting:

- **SIE off** — deployment cameras are always unseen (the dinov2 adapter
  precedent skips SIE at inference), VehicleID has no camera labels, and a
  cross-domain union camera vocab is meaningless.
- **CenterLoss off** — the 09w campaign's fp16 CenterLoss NaN trap fired at
  ~1.8k classes; the joint label space is ~45k.
- **Iteration-budgeted, domain-balanced epochs** — each epoch is 1600 batches,
  round-robin over the domains (`step % 4`), each batch a single-domain
  P=24/K=4 PK batch. Equal batch counts per domain regardless of dataset size
  is the "domain-balanced" contract; single-domain batches keep triplet
  negatives hard. 40 epochs ≈ 6.1M samples ≈ the A5alpha budget.

**CityFlow hygiene:** trains only on the TRAIN split scenes (S01/S03/S04) and
drops any identity that also appears in S02 GT — S02 is the frozen
calibration + production-validation scene and the Stage-C eval scene.

Session budget: 10.5h wall guard from process start (setup eats ~1.5h of the
12h cap); per-epoch `checkpoints/last.pth` with RNG state. To resume, attach
the prior version's output as a data source and re-run — `find_resume()`
scans `/kaggle/input/**/checkpoints/last.pth`.

Local structural smoke (CPU, synthetic tiny domains, random-init backbone):
`ATHAR_LOCAL_SMOKE=1 python athar_joint_reid_train.py` — exercises assembly,
round-robin loop, eval, resume, export.

### C. `eval/` — cross-domain eval matrix (T4)
Joint ckpt vs the frozen VeRi-only `transreid_primary`: mAP/R1 on VeRi-776
test, VeRi-Wild test-3000, VehicleID test-800, CityFlow S02 crops. Matrix
JSON frozen into the repo + guard test; joint ckpt registered as a lifecycle
candidate (promotion stays eval-gated).

## Operations

```bash
# push (token injected from ~/.kaggle/kaggle.json, never committed)
python scripts/kaggle/joint_reid/push_kernel.py veriwild_prep
python scripts/kaggle/joint_reid/push_kernel.py train NvidiaTeslaT4

# status (PYTHONUTF8=1 on Windows)
kaggle kernels status mrkdagods/veriwild-train-prep
kaggle kernels status mrkdagods/athar-joint-reid-train
```

Stage B pushes only after Stage A's dataset exists (Kaggle rejects pushes
with missing `dataset_sources`).
