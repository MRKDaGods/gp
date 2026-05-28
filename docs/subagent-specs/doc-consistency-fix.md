# Doc Consistency Fix Spec (post-PR #49)

Scope: fix the 6 documentation inconsistencies flagged by `docs/model-cards.md` (PR #49). This is a documentation-only spec. No code changes. No metric re-derivation — every canonical value below has been traced to a primary source.

Last verified: 2026-05-17. Master at `0070c54`.

Conventions:
- "metric correction": one value is wrong and must be replaced.
- "context clarification": both values are real but describe different artifacts; the fix is wording, not the number.
- "stale text deletion": a sentence was correct in the past but is no longer true.

---

## 1. `cityflow_transreid` R1: 92.41% vs 92.27%

**Classification**: metric correction (in `.github/copilot-instructions.md` only).

**Canonical truth**: **R1 = 92.41%** for the currently deployed `transreid_cityflowv2_best.pth` checkpoint produced by `gumfreddy/09-vehicle-reid-cityflowv2-augoverhaul-ema` (the AugOverhaul + EMA training run).

Primary source: [docs/findings.md#L748](docs/findings.md#L748) "Base model result: **mAP = 81.53%**, **R1 = 92.41%**" (09 v3 base, AugOverhaul + standard TripletLoss + CenterLoss recipe).

`92.27%` is NOT wrong — it is the R1 of the **prior** baseline checkpoint (09b v2 ViT-B/16 CLIP, pre-augoverhaul), see [docs/findings.md#L1094](docs/findings.md#L1094) "09b v2: mAP=80.14%, R1=92.27%". The deployed checkpoint is the augoverhaul one, not 09b v2, so the headline number for the deployed model is 92.41%.

### Occurrences

| File:line | Current text | Correctness |
|---|---|---|
| [configs/model_registry.yaml#L93](configs/model_registry.yaml#L93) | `value: 0.9241` for `cityflow_transreid` r1 | CORRECT |
| [docs/model-cards.md#L172](docs/model-cards.md#L172) | "mAP=81.53%; R1=92.41% in registry and experiment log, while some high-level docs still carry R1=92.27% as an unverified claim" | CORRECT (describes the inconsistency) |
| [docs/model-cards.md#L215](docs/model-cards.md#L215) | Verified Metrics row R1=92.41 | CORRECT |
| [docs/model-cards.md#L231](docs/model-cards.md#L231) | "Several docs disagree on R1: registry/experiment-log use 92.41%, while high-level instructions call 92.27% unverified." | CORRECT (describes the inconsistency) |
| [docs/models.md#L91](docs/models.md#L91) | `\| ViT-B/16 CLIP 256px \| CityFlowV2 \| 0.8153 / 0.9241 \| ...` | CORRECT |
| [docs/findings.md#L748](docs/findings.md#L748) | "Base model result: **mAP = 81.53%**, **R1 = 92.41%**" | CORRECT (primary source) |
| [docs/findings.md#L776](docs/findings.md#L776) | "`mAP = 81.53%`, `R1 = 92.41%`" | CORRECT |
| [docs/findings.md#L1057](docs/findings.md#L1057) | "**R1 92.41% vs 92.74%**" | CORRECT |
| [docs/experiment-log.md#L145](docs/experiment-log.md#L145) | "09 v3 augoverhaul-EMA (`mAP=81.53%`, `R1=92.41%`)" | CORRECT |
| [docs/experiment-log.md#L705](docs/experiment-log.md#L705) | "09 v3 ... **81.53%** **92.41%**" | CORRECT |
| [docs/_data/checkpoint_inventory.json#L1566](docs/_data/checkpoint_inventory.json#L1566) | "Best R1: 92.41%" | CORRECT |
| [.github/copilot-instructions.md#L101](.github/copilot-instructions.md#L101) | "TransReID ViT-B/16 CLIP 256px — mAP=81.53% (verified, AugOverhaul+EMA kernel), R1=92.27% (UNVERIFIED, claim-only) on CityFlowV2" | **WRONG** — wrong R1 and wrong "UNVERIFIED" tag |

References to the older `0.8014 / 0.9227` (e.g. [docs/findings.md#L410](docs/findings.md#L410), [docs/findings.md#L1094](docs/findings.md#L1094), [docs/experiment-log.md#L703](docs/experiment-log.md#L703), [docs/experiment-log.md#L1021](docs/experiment-log.md#L1021), [docs/paper-draft.md#L7](docs/paper-draft.md#L7), [docs/paper-draft.md#L129](docs/paper-draft.md#L129), [docs/models.md#L93](docs/models.md#L93)) are CORRECT in context — they describe the older 09b v2 baseline used historically (e.g. the paper's primary single-model row), NOT the currently deployed checkpoint. Do not touch those.

### Required edit

[.github/copilot-instructions.md#L101](.github/copilot-instructions.md#L101)

Old:
```
- **Primary model**: TransReID ViT-B/16 CLIP 256px — mAP=81.53% (verified, AugOverhaul+EMA kernel), R1=92.27% (UNVERIFIED, claim-only) on CityFlowV2
```
New:
```
- **Primary model**: TransReID ViT-B/16 CLIP 256px — mAP=81.53%, R1=92.41% (verified, AugOverhaul+EMA kernel `gumfreddy/09-vehicle-reid-cityflowv2-augoverhaul-ema`, see docs/findings.md L748) on CityFlowV2. The older 80.14% / 92.27% values refer to the prior 09b v2 baseline, not the deployed checkpoint.
```

---

## 2. DINOv2 provenance: dataset hosting and mAP/R1

**Classification**: stale text deletion in `docs/models.md` (one sentence claims hosting is unresolved when the source kernel is in fact known and accepted everywhere else).

**Canonical truth**:
- Source training/producing kernel: `yahiaakhalafallah/09s-dinov2-large-cityflowv2`
- Checkpoint filename: `vehicle_transreid_dinov2_large_cityflowv2_final.pth`
- Single-camera ReID mAP=86.79%, R1=96.15% (best epoch 115/120)
- Hosting: the file is the producing kernel's own output dataset (registry uses `kaggle_dataset: yahiaakhalafallah/09s-dinov2-large-cityflowv2` and `member: vehicle_transreid_dinov2_large_cityflowv2_final.pth`); it is not mirrored in any of the `*/mtmc-weights` aggregate datasets.

Primary sources:
- [configs/model_registry.yaml#L55-L57](configs/model_registry.yaml#L55) and [configs/model_registry.yaml#L181-L183](configs/model_registry.yaml#L181) record `kaggle_dataset: yahiaakhalafallah/09s-dinov2-large-cityflowv2`, `member: vehicle_transreid_dinov2_large_cityflowv2_final.pth`, `source_training_kernel: yahiaakhalafallah/09s-dinov2-large-cityflowv2`.
- [docs/findings.md#L408](docs/findings.md#L408): "**Best Vehicle ReID Model (09s v1 DINOv2 ViT-L/14)** | mAP=86.79%, R1=96.15%".
- [docs/findings.md#L430](docs/findings.md#L430): "**09s v1** delivered a genuine breakthrough: **DINOv2 ViT-L/14** reached **86.79% mAP / 96.15% R1** at epoch **115/120**".
- [docs/findings.md#L571](docs/findings.md#L571): "**09s v1 - DINOv2 ViT-L/14**: **BREAKTHROUGH** at **mAP=86.79% / R1=96.15%** (best epoch **115/120**)".
- [docs/audits/2026-05-15-system-audit.md#L34](docs/audits/2026-05-15-system-audit.md#L34): "86.79% mAP / 96.15% R1 single-cam".
- [docs/model-cards.md#L467](docs/model-cards.md#L467): same values, accepts kernel as resolved.
- [docs/model-cards.md#L510](docs/model-cards.md#L510): Verified Metrics row R1=96.15, mAP=86.79, kernel `yahiaakhalafallah/09s-dinov2-large-cityflowv2`.

### Occurrences

| File:line | Current text | Correctness |
|---|---|---|
| [docs/models.md#L33](docs/models.md#L33) | "Hosted: DATASET UNRESOLVED. Visible datasets across gumfreddy, mrkdagods, ali369, and yahiaakhalafallah only expose the mtmc-weights datasets, and none contains the DINOv2 tertiary checkpoint. The deployed path is a Kaggle notebook output source: /kaggle/input/09s-dinov2-large-cityflowv2/vehicle_transreid_dinov2_large_cityflowv2_final.pth." | **STALE** — registry now records the kernel output dataset as the host |
| [docs/models.md#L36](docs/models.md#L36) | "Verified metric: UNVERIFIED. The kernel log was accessible but contained no mAP/R1 lines; the output API paged through crop images before the summary JSON and did not expose it in this pass." | **STALE** — values are verified via [docs/findings.md#L408,L430,L571] from the kernel's training log |
| [docs/model-cards.md#L528](docs/model-cards.md#L528) | "The deployed tertiary checkpoint hosting remains partially unresolved in inventory docs: the producing kernel is known, but the hosted dataset was unresolved in the inventory pass." | **STALE/contradicts the registry** — model-cards itself records the hosting as resolved earlier in the section |
| [configs/model_registry.yaml#L55-L57](configs/model_registry.yaml#L55), [#L181-L183](configs/model_registry.yaml#L181) | hosting + source_training_kernel | CORRECT |
| [docs/findings.md#L408](docs/findings.md#L408), [#L430](docs/findings.md#L430), [#L571](docs/findings.md#L571) | mAP=86.79%, R1=96.15% | CORRECT (primary source) |
| [docs/model-cards.md#L467](docs/model-cards.md#L467), [#L510](docs/model-cards.md#L510) | mAP=86.79%, R1=96.15%, kernel resolved | CORRECT |

### Required edits

[docs/models.md#L33](docs/models.md#L33) — replace the line:

Old:
```
- Hosted: DATASET UNRESOLVED. Visible datasets across gumfreddy, mrkdagods, ali369, and yahiaakhalafallah only expose the mtmc-weights datasets, and none contains the DINOv2 tertiary checkpoint. The deployed path is a Kaggle notebook output source: /kaggle/input/09s-dinov2-large-cityflowv2/vehicle_transreid_dinov2_large_cityflowv2_final.pth.
```
New:
```
- Hosted: as the output of source kernel yahiaakhalafallah/09s-dinov2-large-cityflowv2 (Kaggle kernel-output dataset; not mirrored into any */mtmc-weights aggregate). Deployed path on Kaggle: /kaggle/input/09s-dinov2-large-cityflowv2/vehicle_transreid_dinov2_large_cityflowv2_final.pth. Registry record: configs/model_registry.yaml#L55-L57.
```

[docs/models.md#L36](docs/models.md#L36) — replace the line:

Old:
```
- Verified metric: UNVERIFIED. The kernel log was accessible but contained no mAP/R1 lines; the output API paged through crop images before the summary JSON and did not expose it in this pass.
```
New:
```
- Verified metric: mAP=86.79%, R1=96.15% (best epoch 115/120), recorded in docs/findings.md L408, L430, L571 from the 09s training kernel log.
```

[docs/model-cards.md#L528](docs/model-cards.md#L528) — replace the line:

Old:
```
- The deployed tertiary checkpoint hosting remains partially unresolved in inventory docs: the producing kernel is known, but the hosted dataset was unresolved in the inventory pass.
```
New:
```
- The deployed tertiary checkpoint is hosted only via the producing kernel's output dataset yahiaakhalafallah/09s-dinov2-large-cityflowv2; it is not mirrored into any */mtmc-weights aggregate dataset, so downloads must reference the producing kernel directly.
```

---

## 3. MVDeTr metric: MODA = 0.921 vs 0.913

**Classification**: context clarification. BOTH numbers are correct — they describe different artifacts. No metric needs to change; only labels.

**Canonical truth**:
- `MODA = 0.921` is the **epoch-20 training-time best** reported in the 12a training log. Registry records it `verified: false` exactly because of this provenance.
- `MODA = 0.913` is the **exported loaded-model log line** from the same `gumfreddy/12a-wildtrack-mvdetr-training` kernel — i.e. the value the exported `MultiviewDetector.pth` actually evaluates to when re-loaded.

Primary sources:
- [configs/model_registry.yaml#L266-L269](configs/model_registry.yaml#L266) `description: ... The 0.921 MODA value is retained as an unverified training-epoch claim; the exported checkpoint line in project notes is lower.` and `value: 0.921`, `verified: false`.
- [docs/models.md#L63](docs/models.md#L63) "exported loaded-model log line reports MODA=0.913, MODP=0.818, precision=0.947, recall=0.966; epoch-20 line reports MODA=0.921 but is not the final exported-checkpoint line".
- [docs/models.md#L129](docs/models.md#L129) "The epoch-20 log line claims MODA=92.1%, but the final loaded-model line for the exported run verifies MODA=91.3%".
- [docs/model-cards.md#L393](docs/model-cards.md#L393) "MODA=0.921 epoch-20 detector claim; exported loaded-model line verifies MODA=0.913".
- [docs/model-cards.md#L436-L437](docs/model-cards.md#L436) records both rows.

### Occurrences

| File:line | Current text | Status |
|---|---|---|
| [configs/model_registry.yaml#L269](configs/model_registry.yaml#L269) | `value: 0.921` (verified: false, with note "Epoch-20 detector quality claim, not the final exported-checkpoint line.") | CORRECT in context |
| [.github/copilot-instructions.md#L113](.github/copilot-instructions.md#L113) | "Detector: MVDeTr ResNet18, MODA=0.921 (12a v3, best achieved)" | LOSES CONTEXT — does not say this is the epoch-20 best, not the exported checkpoint |
| [.github/copilot-instructions.md#L177](.github/copilot-instructions.md#L177) | "Person: improved detector→better tracking: MODA 90.9→92.1% but IDF1 unchanged at 94.7%" | CORRECT in context (training-time MODA progression) |
| [docs/models.md#L63](docs/models.md#L63), [#L129](docs/models.md#L129) | Both numbers, clearly labelled | CORRECT |
| [docs/model-cards.md#L393](docs/model-cards.md#L393), [#L436-L437](docs/model-cards.md#L436), [#L452](docs/model-cards.md#L452) | Both numbers, clearly labelled | CORRECT |
| `docs/findings.md` (e.g. references to "MODA=0.921" / "best achieved") | training-time best framing | CORRECT |

### Required edit

[.github/copilot-instructions.md#L113](.github/copilot-instructions.md#L113) — clarify both numbers:

Old:
```
- **Detector**: MVDeTr ResNet18, MODA=0.921 (12a v3, best achieved)
```
New:
```
- **Detector**: MVDeTr ResNet18, MODA=0.921 (12a v3, epoch-20 training-time best); the exported checkpoint loaded-model log line verifies MODA=0.913 (see docs/models.md L63)
```

No other files need editing for this item.

---

## 4. 09v ownership: `yahiaakhalafallah` verifier vs `mrkdagods` checkpoint hosting

**Classification**: context clarification (docs/models.md conflates verifier kernel ownership with checkpoint hosting account).

**Canonical truth**:
- Verifier kernel: `yahiaakhalafallah/09v-veri-776-eval-transreid-rerank` (this is the EVAL kernel that produced the 98.33% R1 / 89.97% mAP rows). It is an eval/rerank notebook, not the original VeRi-776 training kernel.
- Checkpoint file: `vehicle_transreid_vit_base_veri776.pth`.
- Hosting: mirrored in `mrkdagods/mtmc-weights` and `gumfreddy/mtmc-weights` aggregate datasets.
- Original VeRi-776 training kernel: NOT separately recorded in this repo (model-cards.md already states "original training kernel is not separately recorded").

Primary sources:
- [configs/model_registry.yaml#L356](configs/model_registry.yaml#L356) and [#L417](configs/model_registry.yaml#L417) `source_training_kernel: yahiaakhalafallah/09v-veri-776-eval-transreid-rerank` (registry overloads `source_training_kernel` for verifier kernels — see repo memory note "checkpoint_refs.source_training_kernel is used for the kernel that trained or exported the checkpoint artifact, not eval-only notebooks", but 09v is an exception because the original training kernel was never separately recorded).
- [configs/model_registry.yaml#L394](configs/model_registry.yaml#L394), [#L403](configs/model_registry.yaml#L403), [#L427](configs/model_registry.yaml#L427), [#L430](configs/model_registry.yaml#L430) all use the same `yahiaakhalafallah/09v-veri-776-eval-transreid-rerank` slug.
- [docs/findings.md#L8](docs/findings.md#L8) "kernel `kaggle://yahiaakhalafallah/09v-veri-776-eval-transreid-rerank`".
- [docs/model-cards.md#L76-L80](docs/model-cards.md#L76) explicitly states "verifier kernel slug: yahiaakhalafallah/09v-veri-776-eval-transreid-rerank" and "author/account: yahiaakhalafallah for verifier; checkpoint hosted via mrkdagods/mtmc-weights and gumfreddy/mtmc-weights" — this is the correct attribution.

### Occurrences

| File:line | Current text | Correctness |
|---|---|---|
| [configs/model_registry.yaml#L356,L394,L403,L417,L427,L430](configs/model_registry.yaml#L356) | `yahiaakhalafallah/09v-veri-776-eval-transreid-rerank` | CORRECT |
| [docs/findings.md#L8](docs/findings.md#L8), [#L993](docs/findings.md#L993) | `yahiaakhalafallah/09v-veri-776-eval-transreid-rerank` for the verifier | CORRECT |
| [docs/model-cards.md#L67-L69](docs/model-cards.md#L67), [#L75-L80](docs/model-cards.md#L75) | yahiaakhalafallah verifier + mrkdagods/gumfreddy hosting | CORRECT |
| [docs/models.md#L92](docs/models.md#L92) | `\| ViT-B/16 CLIP 256px \| VeRi-776 \| 0.8997 / 0.9833 \| \`mrkdagods\` 09v v17 \| Yes - via 14t fusion \|` | **AMBIGUOUS** — table column header is "Source kernel", but value reads as "mrkdagods 09v v17" which mixes the hosting account with a verifier-version tag. The actual verifier kernel is `yahiaakhalafallah/09v-veri-776-eval-transreid-rerank` v17. |

### Required edit

[docs/models.md#L92](docs/models.md#L92) — disambiguate the "Source kernel" column:

Old:
```
| ViT-B/16 CLIP 256px | VeRi-776 | 0.8997 / 0.9833 | `mrkdagods` 09v v17 | Yes - via 14t fusion | Checkpoint `vehicle_transreid_vit_base_veri776.pth`; one of the two 14t VeRi-776 experts. |
```
New:
```
| ViT-B/16 CLIP 256px | VeRi-776 | 0.8997 / 0.9833 | verifier `yahiaakhalafallah/09v-veri-776-eval-transreid-rerank` v17; checkpoint hosted in `mrkdagods/mtmc-weights` + `gumfreddy/mtmc-weights`; original training kernel not separately recorded | Yes - via 14t fusion | Checkpoint `vehicle_transreid_vit_base_veri776.pth`; one of the two 14t VeRi-776 experts. |
```

(Coder note: this table row will get wide. That is acceptable — the column is already free-form in this file.)

---

## 5. CLIP-SENet batch wording

**Classification**: no fix needed. The findings.md "effective batch 128" and the notebook CFG (`batch_size=64, accum_steps=2, P_effective=16, P=8, K=8`) are arithmetically consistent: micro-batch = P*K = 8*8 = 64; effective batch = P_effective*K = 16*8 = 128 = 64 * accum_steps.

**Canonical truth (from the training notebook itself)**:
- [notebooks/kaggle/13_clip_senet_train/13_clip_senet_train.ipynb#L687-L693](notebooks/kaggle/13_clip_senet_train/13_clip_senet_train.ipynb): `'P': 8, 'P_effective': 16, 'K': 8, 'accum_steps': 2, 'batch_size': 64`.
- [notebooks/kaggle/13_clip_senet_train/13_clip_senet_train.ipynb#L930](notebooks/kaggle/13_clip_senet_train/13_clip_senet_train.ipynb): `print(f"micro-batch: P={P} K={K} = {P*K}, effective batch: {P_effective*K} via accum_steps={accum_steps}")` — i.e. micro-batch=64, effective batch=128 via accum_steps=2.

### Occurrences

| File:line | Current text | Correctness |
|---|---|---|
| [docs/model-cards.md#L119](docs/model-cards.md#L119) | "micro-batch P=8, K=8, batch_size=64, accum_steps=2; effective batch 128 with P_effective=16, K=8" | CORRECT (matches the notebook exactly) |
| `docs/findings.md` "effective batch 128" mentions | CORRECT in context |

### Required edit

None. This item is closed as a non-inconsistency. PR #49's flag wording is conservative but the values are consistent. No edits to any file.

---

## 6. Stale "Current reproducible vehicle MTMC IDF1 is 0.7703" line

**Classification**: stale text deletion in `docs/findings.md` (one specific sentence is now wrong because the headline has been promoted to 0.77936).

**Canonical truth**: as of 2026-05-07 the reproducible vehicle MTMC IDF1 headline is **0.77936** (14e B1 v1), per [docs/findings.md#L154](docs/findings.md#L154). The earlier `0.7703` (10c v15 / 10a v7 CLIP+DINOv2 score fusion, `w_tertiary=0.60`, `aqe_k=3`) is the **previous deployed baseline**, not the current headline.

### Occurrences

Every `0.7703` occurrence in `docs/findings.md` was reviewed. All but one are CORRECT in context (they describe `0.7703` historically as "previous deployed baseline", "production baseline", or "the prior best the experiment failed to clear"):

CORRECT contextual references (DO NOT EDIT):
- [docs/findings.md#L36](docs/findings.md#L36): `ter_060` row in a historical sweep table — correct
- [docs/findings.md#L39](docs/findings.md#L39): "Best operating point: w_tertiary=0.60 produced MTMC IDF1 = 0.7703" — historical narrative — correct
- [docs/findings.md#L41](docs/findings.md#L41): "the best fusion point at 0.7703 became the previous deployed best on available weights, but it ... was later superseded by the 14e TTA + AQE k=2 headline" — correct
- [docs/findings.md#L91, L98, L115, L117, L122, L128, L132, L133, L147](docs/findings.md#L91): all reference `0.7703` as "production baseline"/"previous deployed best" in 13d/13f/13h/14a/14c/14d narratives — correct
- [docs/findings.md#L154, L184, L194, L255](docs/findings.md#L154): explicitly contrast 0.7703 (previous) vs 0.77936 (current) — correct
- [docs/findings.md#L360, L425](docs/findings.md#L360): same historical baseline framing — correct

The ONLY incorrect occurrence is the bolded standalone sentence at [docs/findings.md#L428](docs/findings.md#L428), which directly contradicts the preceding sentence in the same paragraph:

[docs/findings.md#L428](docs/findings.md#L428) (current text):
```
**Updated 2026-05-07**: the new reproducible best is **0.77936** (14e B1 v1) on multi-crop TTA Stage-2 features with `aqe_k=2` instead of the long-standing production `aqe_k=3`. The 0.7703 figure below remains the previous deployed baseline. **Current reproducible vehicle MTMC IDF1 is 0.7703** from **10c v15 / 10a v7** using **CLIP+DINOv2 score-level fusion** with `w_tertiary=0.60`. The earlier **0.775 / 0.784** CLIP+OSNet-era results depended on `vehicle_osnet_veri776.pth`, a CityFlowV2-adapted OSNet checkpoint that is no longer present in the weights datasets after the **2026-03-30** regeneration of `mrkdagods/mtmc-weights`. Vehicle association remains exhausted, so future gains will need materially better features or priors rather than more stage-4 tuning.
```

The bolded "**Current reproducible vehicle MTMC IDF1 is 0.7703**" is a leftover from the pre-14e era. It is internally contradicted by the sentence two clauses earlier in the same paragraph (which already announces 0.77936 as the new best).

### Required edit

[docs/findings.md#L428](docs/findings.md#L428) — replace only the contradicting sentence; keep all surrounding context untouched.

Old (one sentence in the paragraph):
```
**Current reproducible vehicle MTMC IDF1 is 0.7703** from **10c v15 / 10a v7** using **CLIP+DINOv2 score-level fusion** with `w_tertiary=0.60`.
```
New:
```
The previous deployed baseline was **0.7703** from **10c v15 / 10a v7** using **CLIP+DINOv2 score-level fusion** with `w_tertiary=0.60`, `aqe_k=3`; that configuration is retained only as the supersession reference for the 14e B1 promotion above.
```

No other `0.7703` occurrences in `docs/findings.md` should be touched.

---

## Summary table

| # | Item | Classification | Files to edit | Lines changed |
|---|---|---|---|---|
| 1 | `cityflow_transreid` R1 92.41% vs 92.27% | metric correction | `.github/copilot-instructions.md` | 1 |
| 2 | DINOv2 provenance/hosting/metric | stale text deletion | `docs/models.md`, `docs/model-cards.md` | 3 |
| 3 | MVDeTr MODA 0.921 vs 0.913 | context clarification | `.github/copilot-instructions.md` | 1 |
| 4 | 09v ownership | context clarification | `docs/models.md` | 1 |
| 5 | CLIP-SENet batch wording | no fix needed | none | 0 |
| 6 | Stale "Current reproducible IDF1 is 0.7703" | stale text deletion | `docs/findings.md` | 1 |

Total: 7 edits across 4 files. No code changes. No metric re-derivation.

## Verification checklist for the Coder

After making the 7 edits above:

1. `grep -rn "92\.27" .github/ docs/ configs/` — every remaining hit should be in a historical 09b v2 / 80.14% / paper-draft baseline context, NOT in a "current deployed checkpoint" context.
2. `grep -rn "DATASET UNRESOLVED" docs/` — should return zero hits for the DINOv2 line in models.md.
3. `grep -rn "UNVERIFIED" docs/models.md` — the DINOv2 metric line should no longer be UNVERIFIED.
4. `grep -rn "Current reproducible vehicle MTMC IDF1 is 0\.7703" docs/findings.md` — should return zero hits.
5. `grep -rn "MODA=0\.921" .github/copilot-instructions.md` — should return one hit, with "epoch-20 training-time best" context.
6. `grep -n "mrkdagods\` 09v v17" docs/models.md` — should return zero hits; the row should now reference `yahiaakhalafallah/09v-veri-776-eval-transreid-rerank` as the verifier.

Do NOT modify any other file, any other line, or any other metric. The remaining 220+ legitimate `0.7703` historical references in findings.md must stay intact.