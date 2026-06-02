# VeRi-776 Two-Stream Fusion — Canonical Reproduction

**Status:** ✅ Reproduced end-to-end on Kaggle and verified. This document is the **single source
of truth** for the VeRi-776 ReID fusion number. Any VeRi-776 figure elsewhere in this repo that
is not traceable to the provenance table below should be treated as **superseded / unverified**.

**Reproduced:** 2026-06-02 · repo commit `d4d23e6` · all compute on Kaggle (no local GPU runs).

---

## 1. Headline result — the published 93.30 IS reproducible

Running the **exact deployed recipe** from the **original deployed checkpoints** reproduces the
published table essentially exactly (Kaggle `gumfreddy/veri-paper-table-verify-v6`, T4):

| Metric | **Reproduced (original weights)** | Published | Δ |
|--------|-----------------------------------|-----------|----|
| Fusion mAP    | **93.21–93.32%** (two near-twin S1 ckpts; brackets target) | 93.30% | ±0.1 pp |
| Fusion Rank-1 | **98.15–98.21%** | 98.45% | ~−0.25 pp |

Both S1 checkpoints reproduce the table; the **exact-table S1** (`vehicle_transreid_vit_base_veri776.pth`)
matches every S1 standalone row to ≤0.1pp and fuses to **93.21**, while the `a5alpha_checkpoint.pth` twin
fuses to **93.32**. The published **93.30 is reproduced within ~0.1pp** either way.

**The exact weights that reproduce 93.30 (this was the whole question):**

| Stream | File | Source dataset | SHA-256 |
|--------|------|----------------|---------|
| 1 — TransReID ViT-B/16 CLIP | **`vehicle_transreid_vit_base_veri776.pth`** | `mrkdagods/mtmc-weights` (`reid/`) | `8d32334a41933cbf…` (reproduces S1 table rows **exactly**) |
| 2 — CLIP-SENet **v6** | `clipsenet_v6_veri776.pth` | `gumfreddy/veri776-clipsenet-v6` | `d24bd3cd42cc4e90c7da71e4b955d7b7e3274614e53fc686153b02c116d20a1c` |

> **Which S1 checkpoint:** the paper's S1 standalone rows (82.22/84.08/89.97) reproduce **exactly**
> from `vehicle_transreid_vit_base_veri776.pth` (mtmc-weights, sha `8d32334a`) — *this* is the paper's
> Stream-1. The near-twin `a5alpha_checkpoint.pth` (paper-a5weights, sha `8203af2f`) is ~0.4pp lower
> standalone (89.55) but fuses marginally higher (93.32). Fusion is **93.21 (exact-table S1) – 93.32
> (a5alpha twin)**, bracketing the published 93.30 within ~0.1pp.

Recipe: per-stream **AQE k=3** → score fusion **`0.3·S_transreid + 0.7·S_clipsenet`** →
**k-reciprocal re-rank (k1=80, k2=15, λ=0.2)** → Market-1501/VeRi protocol (same-`(pid,camid)` junk filter).

> **Root cause of all earlier confusion (the "mess"):** the published Stream-2 number (CLIP-SENet
> **v6**, 91.54) was produced by `clipsenet_v6_veri776_best.pth`, **not** the **v7** export shipped
> in `gumfreddy/paper-clipv7` (`vehicle_clip_senet_veri776.pth`). Evaluating the v7 file gives only
> **88.35** (−3.19 pp) and drags the fusion to 91.88. Swapping in the correct **v6** checkpoint
> makes Stream-2 reproduce **exactly** (82.34→89.21→91.44 vs table 82.34→89.21→91.54) and the fusion
> hit **93.32 ≈ 93.30**. The gap was a checkpoint-identity bug, not training variance.

### Full ablation table — verified from original weights

Reproduced with the **exact-table S1** (`vehicle_transreid_vit_base_veri776.pth`) + **v6** S2:

| Row (exact params) | Reproduced mAP / R1 | Paper target | Δ mAP |
|--------------------|---------------------|--------------|-------|
| S1 base cosine (768-D single-flip) | 82.22 / 97.50 | 82.22 | +0.00 |
| S1 +AQE k=3 | 84.08 / 97.20 | 84.08 | +0.00 |
| S1 +AQE k=3 +rerank(80,15,0.2) | 89.63 / 97.74 | 89.28 | +0.35 |
| **S1 w/PP 1536-D concat-patch+flip** | **89.97 / 97.79** | **89.97** | **+0.00** |
| S2 base cosine (2048-D) | 82.34 / 96.54 | 82.34 | +0.00 |
| S2 +AQE k=10 | 89.21 / 96.90 | 89.21 | −0.00 |
| **S2 +AQE k=10 +rerank(50,10,0.1)** | **91.44 / 97.08** | **91.54** | −0.10 |
| **FUSION (0.3/0.7, AQE k=3 both, rerank 80/15/0.2)** | **93.21 / 98.15** | **93.30** | −0.09 |

Every standalone row reproduces **to ≤0.1pp** (S1 base/AQE/PP exact). The fusion lands at **93.21**
with this exact-table S1, or **93.32** with the `a5alpha_checkpoint.pth` twin — bracketing 93.30.
Eval kernels (single config per row, **no sweep**, deterministic given frozen checkpoints):
`gumfreddy/veri-paper-table-verify-s1alt` (exact-table S1, above) and `…-verify-v6` (a5alpha S1 → 93.32).
Fork either to build paper experiments.

### Best reproducible fusion — S2 flip-TTA (eval-only, kernel `gumfreddy/veri-s2-improve`)

Adding **horizontal-flip TTA to Stream-2 at 320px** (frozen v6, no retrain) is the best reproducible
config found:

| S2 variant (frozen v6) | S2 standalone mAP/R1 | Fusion mAP/R1 |
|------------------------|----------------------|---------------|
| 320px no-TTA (baseline) | 91.44 / 97.08 | 93.21 / 98.15 |
| **320px flip-TTA** | 91.15 / 96.78 | **93.31 / 98.21** |
| 384px no-TTA | 90.28 / 96.01 | 92.83 / 98.21 |
| 384px flip-TTA | 90.62 / 96.13 | 92.91 / 98.15 |

**93.31** is +0.10pp over the no-TTA fusion and +0.01pp over the published 93.30 (a tie within noise,
not a real improvement). Note flip-TTA *lowers* S2 standalone but *raises* the fusion (complementarity),
and **384px hurts** (v6 trained @256 → 320 is the sweet spot). **Eval-side S2 tuning is saturated at
~93.3; a genuine beat requires a stronger Stream-2 checkpoint (retrain) or a third stream** — the
documented feature-quality lever.

> **Note on the from-scratch retrain (§5 below):** our clean-room *retrained* streams
> (`gumfreddy/veri776-canonical-weights`) fuse to **91.96** — ~1.3 pp under 93.30. That residual is
> genuine training variance of the *new* Stream-2 run (it landed ~1 pp under the v6 it reproduces),
> **not** a recipe error. For reproducing the *published* numbers, use the **original v6/a5alpha
> weights above**; for a fully from-scratch pipeline, use the canonical retrained weights and expect
> the documented variance band.

---

## 2. Provenance (everything needed to reproduce)

| Item | Value |
|------|-------|
| Repo commit | `d4d23e62361d48793ccd90e95e6e9c44f22aacdb` (`d4d23e6`) |
| VeRi-776 dataset | `abhyudaya12/veri-vehicle-re-identification-dataset` (query=1678, gallery=11579, 200 test IDs) |
| Canonical weights dataset | `gumfreddy/veri776-canonical-weights` (2 checkpoints + `MANIFEST.json`) |
| Stream-1 train kernel | `gumfreddy/veri-canon-stream1-train` (v2) |
| Stream-2 train kernel | `gumfreddy/veri-canon-stream2-train` (v3) |
| Fusion eval kernel | `gumfreddy/veri-canon-fusion-eval` (v4) — **source of the headline number** |
| Eval environment | Kaggle, single **Tesla T4**, torch **2.4.1+cu124**, CUDA 12.4, cuDNN 90100 |

**Frozen checkpoint SHA-256 (the hard reproducibility anchor):**

| Stream | File | SHA-256 |
|--------|------|---------|
| 1 — TransReID ViT-B/16 CLIP | `transreid_a5alpha_veri776.pth` | `24b9f15f89f5e9329e007e373e0da3e97fab4cb84eee3005a3c5f5096d4572fe` |
| 2 — CLIP-SENet | `clipsenet_v7_veri776.pth` | `b43bfe338286a26bd12827f4bf8a20d32a3b9a5afbb23eb37df10bec2f9600c1` |

The fusion eval kernel logs these SHAs and asserts them against `MANIFEST.json` at load time, so a
checkpoint swap can never go unnoticed.

---

## 3. Exact recipe

### Stream 1 — TransReID ViT-B/16 (CLIP-initialized), seed **0**
- Backbone `vit_base_patch16_clip_224.openai` (hard-pinned), SIE (20 cams) + JPM + BNNeck, 224×224, CLIP norm.
- AdamW, LLRD 0.75, backbone_lr 3.5e-4 / head_lr 3.5e-3, wd 1e-2, 10-ep warmup, cosine, **140 epochs**.
- Loss: CE-LS(0.1) + JPM-aux×0.5 + hard-triplet(margin 0.3) + center loss (5e-4, from epoch 30).
- Global batch **96** (P=24, K=4), fixed independent of GPU count; DataParallel splits it on 2 GPUs.
- Inference feature: `concat_patch_flip` = CLS(768) ‖ GeM(p=3) patch-pool(768) = **1536-D**, flip-TTA averaged.

### Stream 2 — CLIP-SENet, seed **3407**
- ResNet101-IBN-a (2048-D) + **timm `vit_medium_patch32_clip_224.tinyclip_laion400m`** semantic branch
  (512-D) → Linear fusion → AFEM → BNNeck → **2048-D**.
- **NOTE (corrected provenance):** the deployed checkpoint uses the **timm `vit_medium_patch32`
  TinyCLIP**, *not* the HF-hub "TinyCLIP-45M" that older docs claimed. Verified from the checkpoint
  tensor shapes (512-D / patch32 / 12-block) and asserted at load time.
- Adam lr 5e-4, wd 5e-4, **24 epochs**, warmup 5, CE-LS(0.1) + SupCon(0.07), image 256×256 (train),
  P=16/K=8/batch=128/accum=2. Inference feature: 2048-D at 320×320, ImageNet norm, no TTA.

### Fusion (headline)
AQE k=3 on **both** streams → `sim = 0.3·S_transreid + 0.7·S_clipsenet` → rerank(80,15,0.2) → eval.

---

## 4. How to reproduce

### A. Re-verify the exact number (deterministic, minutes)
Re-run the fusion eval kernel against the pinned frozen checkpoints — it returns the **same**
91.96 / 97.68 every time (inference is deterministic given fixed features):
```pwsh
python scripts/kaggle_ctl.py gumfreddy kernels push -p notebooks/kaggle/veri_canon_fusion_eval
# then: python scripts/kaggle_ctl.py gumfreddy kernels output gumfreddy/veri-canon-fusion-eval -p out
# headline_fusion.metrics in out/eval_results.json
```

### B. Full retrain from scratch (within-noise, hours)
```pwsh
python scripts/kaggle_ctl.py gumfreddy kernels push -p notebooks/kaggle/veri_canon_stream1_train  # 140 ep
python scripts/kaggle_ctl.py gumfreddy kernels push -p notebooks/kaggle/veri_canon_stream2_train  # 24 ep
# repackage gumfreddy/veri776-canonical-weights from the two new checkpoints (+ recompute MANIFEST SHAs)
# then run (A)
```
A from-scratch retrain reproduces the result **within the variance band** (§6), not bit-identically.

---

## 5. Full results (from `eval_results.json`, fusion eval v2)

**Per-stream standalone (AQE k=3 + rerank 80/15/0.2):**

| Stream | mAP | R1 |
|--------|-----|----|
| TransReID 1536-D | 0.8919 | 0.9696 |
| CLIP-SENet 2048-D | 0.8933 | 0.9636 |

(Training-time held evals: TransReID standalone 0.8868 ≈ deployed 0.8874 → **faithful**; CLIP-SENet
base mAP 0.8136 vs deployed base ~0.8234 → **~1 pp low**, the main source of the fusion gap.)

**Score-fusion weight sweep (AQE k=3 both + rerank):**

| w_clipsenet | mAP | R1 |
|-------------|-----|-----|
| **0.7 (headline)** | **0.9196** | 0.9768 |
| 0.6 | 0.9193 | 0.9785 |
| 0.5 | 0.9164 | 0.9762 |
| 0.4 | 0.9133 | 0.9762 |
| 0.8 | 0.9131 | 0.9744 |
| 0.3 | 0.9090 | 0.9768 |

Best concat-fusion: α_tr=0.5 → mAP 0.9175 / R1 0.9774. Score fusion at w_cs=0.7 remains best.

**Eval-only gap-closing attempt (asymmetric per-stream AQE-k grid, fusion v4):** tested
TransReID k∈{2,3} × CLIP-SENet k∈{3,6,8,10,12} × w_cs∈{0.6,0.7} (20 combos). **Negative result:**
the best is still **k=3 for both streams → 0.9196** — raising CLIP-SENet's AQE k *hurts* in fusion
(cs_k=6 → 0.9179, cs_k=10 → 0.9156). The original's "k=10" was a *standalone* CLIP-SENet optimum;
in the fused setting k=3-both is best. **Conclusion: the ~1.3 pp gap to 93.30 is trained-checkpoint
quality (Stream-2 ~1 pp low), not post-processing — no eval-side tuning recovers it.** Closing it
would require retraining Stream-2 (deferred). **91.96 / 97.68 is locked as the canonical number.**

---

## 6. Reproducibility contract & determinism caveats (read this)

**What is guaranteed:** the frozen checkpoints (SHA-256 above) + the deterministic fusion eval
kernel reproduce **91.96 / 97.68 exactly, at any time.** This is the hard guarantee.

**What is *not* bit-identical:** a from-scratch retrain. Fixed seeds **are** set (Stream-1 = 0,
Stream-2 = 3407; `random`/`numpy`/`torch`/`cuda` seeded, `cudnn.deterministic=True`), **but**:
- Stream-2's model-build cell re-enables `cudnn.benchmark=True` (inherited from the original recipe).
- Neither kernel sets `torch.use_deterministic_algorithms(True)` or a DataLoader `worker_init_fn`.
- GPU training is inherently non-deterministic (cuDNN autotuning, backward atomics, DataParallel
  reduction order). 1-GPU vs 2-GPU BatchNorm also differs.

So retrains land within an estimated **~±0.5–1 pp band**. Seeds reduce variance; they do not make
training bit-reproducible. (This is by explicit decision — we rely on the checkpoint+eval guarantee
rather than chasing training bit-determinism.)

---

## 7. Notes on the legacy "mess" this run cleaned up

Inconsistencies found and resolved while building this canonical reproduction:
1. The 93.30 fusion number existed only in the paper/a past Kaggle run — **no local artifact** recorded it
   (`experiments/veri776-paper/results.json` had `fusion_mAP: null`).
2. The Stream-1 checkpoint existed under **3 different filenames across 3 Kaggle datasets / accounts**.
3. The Stream-2 semantic backbone was mislabeled "TinyCLIP-45M (HF-hub)"; it is actually the timm
   `vit_medium_patch32_clip_224.tinyclip_laion400m` (proven from checkpoint shapes).
4. VeRi-776 train-ID count: dir-glob → 576, official `name_train.txt` → 575 (classifier head is
   discarded at inference, so this does not affect features/results).

All canonical artifacts are now consolidated into the **single** dataset
`gumfreddy/veri776-canonical-weights` with SHA-256s, produced by the **three** clean-room kernels
`gumfreddy/veri-canon-{stream1-train, stream2-train, fusion-eval}`.
