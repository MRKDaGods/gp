# VeRi-776 Canonical Reproduction — Implementation Spec

**Status:** Spec only (no notebooks modified). Authoritative extraction of the four existing kernels.
**Goal:** ONE canonical, fully-reproducible pair of training kernels + one fusion-eval kernel that
reproduces the headline VeRi-776 two-stream result **~93.30% mAP / 98.45% R1** at any time.
**Headline fusion (deployed):** AQE k=3 per stream -> score fusion `0.3*S_transreid + 0.7*S_clipsenet`
-> k-reciprocal rerank (k1=80, k2=15, lambda=0.2). This lives in `paper_veri776_A5alpha.ipynb` cell 26,
NOT in the 14t sweep grid (14t is the exploration sweep; its WIN threshold is 0.9154/0.9833).

---

## 0. Source notebooks audited

| Role | Notebook | Kaggle id | Owner | GPU | machine_shape |
|------|----------|-----------|-------|-----|----------------|
| Stream 1 train | paper_veri776_A5alpha.ipynb | ali369/paper-veri776-a5alpha | ali369 | yes | NvidiaTeslaT4 (single) |
| Stream 2 train | 13_clip_senet_train_v7.ipynb | yahiaakhalafallah/13-clip-senet-train-v7 | yahiaakhalafallah | yes | (unset -> default single) |
| Feature dump | paper_features_a5alpha.ipynb | gumfreddy/paper-features-a5alpha | gumfreddy | yes | NvidiaTeslaT4 (single) |
| Fusion + eval | 14t_veri_fusion.ipynb | yahiaakhalafallah/14t-veri-fusion-clip-senet-x-transreid | yahiaakhalafallah | yes | NvidiaTeslaT4 (single) |

dataset_sources / kernel_sources per metadata:
- A5alpha: ds = [mrkdagods/mtmc-weights, abhyudaya12/veri-vehicle-re-identification-dataset]
- clip_senet_v7: ds = [abhyudaya12/veri-vehicle-re-identification-dataset]; no kernel_sources
- features_a5alpha: ds = [gumfreddy/paper-a5weights, gumfreddy/paper-clipv7, abhyudaya12/veri-vehicle-re-identification-dataset]
- 14t: ds = [mrkdagods/mtmc-weights, abhyudaya12/veri-vehicle-re-identification-dataset]; kernel_sources = [yahiaakhalafallah/13-clip-senet-train]

VeRi dataset slug is consistent across all four: **abhyudaya12/veri-vehicle-re-identification-dataset**.

---

## 1. Recipe Reference

### 1.1 STREAM 1 — TransReID ViT-B/16 CLIP (exp A5alpha)

**Backbone**
- timm model: `vit_base_patch16_clip_224.openai` (A5alpha selects it via `find_clip_vit_base()` which
  prefers an `openai` non-`ft` tag; the deployed serving loader hardcodes `vit_base_patch16_clip_224.openai`).
- Image size **224x224**, BICUBIC resize. embed_dim 768 (proj = Identity since 768==vit_dim).
- Normalization = **CLIP** mean `[0.48145466, 0.4578275, 0.40821073]`, std `[0.26862954, 0.26130258, 0.27577711]`.
- Architecture extras: SIE camera embedding (20 cameras, broadcast to ALL tokens), JPM (train-only),
  BNNeck (`bn`, bias frozen), classifier `cls_head` (bias-free). **CRITICAL:** timm `norm_pre` is applied
  before transformer blocks (CLIP pre-LayerNorm). Skipping it destroys CLIP features (root cause of v3-v5 fail).

**Optimizer / schedule (cell 15)**
- AdamW with **LLRD = 0.75** layer-wise decay (`get_llrd_param_groups`): block depth d gets
  `backbone_lr * 0.75^(num_blocks+1-d)`; embeddings (patch_embed/cls_token/pos_embed/norm_pre) lowest LR; head = head_lr.
- backbone_lr = **3.5e-4**, head_lr = **3.5e-3**, weight_decay = **1e-2**.
- Warmup = **10** epochs (linear from base_lr/WARMUP up to base_lr). After warmup: CosineAnnealingLR T_max = EPOCHS-WARMUP.
- Total epochs = **140**. AMP via `torch.amp.GradScaler("cuda")`. grad clip max_norm = **5.0** (adamw branch).
- Center loss has its OWN optimizer: SGD(lr=0.5) on `ctr_loss.parameters()`.

**Sampler / batch (cell 9)** — note these SCALE with NUM_GPUS in the old kernel:
- `BATCH = 48 * NUM_GPUS`, `P_IDS = 12 * NUM_GPUS`, `K = 4`. On 1 GPU: P=12, K=4, batch=48 (12*4==48).
  PKSampler is deterministic, seeded by `SEED + epoch`. query/gallery loaders batch_size=128, no shuffle.

**Losses (cell 10/15)** — loss mode = `triplet_center`:
- CE: CrossEntropyLabelSmooth, **epsilon = 0.1**. JPM aux CE weighted **0.5** (`ce(jc) * 0.5`).
- Triplet: hard-mining, **margin = 0.3**, on L2-normalized PRE-BN feature `g` (cdist p=2, MarginRankingLoss).
- Center loss: weight **5e-4**, **start epoch 30** (added only once `epoch >= 30`), feat_dim 768.
- SupConLoss(temp 0.07) defined but NOT used in triplet_center mode.

**Augmentations (cell 9, aug="full")**, in order:
1. Resize(224,224,BICUBIC) 2. RandomHorizontalFlip(p=0.5) 3. Pad(10) 4. RandomCrop(224,224)
5. ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1, hue=0)
6. RandomApply([GaussianBlur(kernel_size=3)], p=0.2) 7. RandomPerspective(distortion_scale=0.2, p=0.2)
8. ToTensor 9. Normalize(CLIP) 10. RandomErasing(p=0.5, value="random").
Test transform: Resize(224,224,BICUBIC)+ToTensor+Normalize(CLIP) only.

**Seed (cell 4):** SEED=0; random/np/torch/cuda seeded; `cudnn.deterministic=True`, `cudnn.benchmark=False`.

**Multi-GPU (cell 14):** `if NUM_GPUS > 1: model = nn.DataParallel(model)`. As deployed it ran on a
**single T4** (NUM_GPUS=1, DataParallel inert). NO DDP. Batch scales with NUM_GPUS in the old code.

**Dataset parse (cell 7):** regex `^(\d+)_c(\d+)`; cam = c-number minus 1 (0-based). Train PIDs relabeled 0..N-1.
num_classes = unique train PIDs (575), num_cameras = 20.

**Checkpoints saved:**
- best-by-mAP -> `/kaggle/working/vehicle_reid_sota/transreid_veri_best.pth` and
  `.../experiments/veri776-paper/A5alpha/best_mAP.pth` (raw state_dict).
- Export (cell 19) -> `/kaggle/working/exported_models/vehicle_transreid_vit_base_veri776_A5alpha.pth`
  wrapped `{"state_dict":..., "recipe":...}` + recipe.json/train_log.json/metadata.json.
- Held standalone Stream-1 eval (cell 23): single_flip, AQE k=3, AQE+rerank(80/15/0.2) -> eval_results.json.

### 1.2 STREAM 2 — CLIP-SENet v7 (13_clip_senet_train_v7)

**Architecture (cell 2; identical to the class inlined in 14t cell 14t-03):**
- Appearance branch `ResNet101IBNBranch`: `resnet101_ibn_a` via pretrainedmodels -> torch.hub(XingangPan/IBN-Net)
  -> timm; **falls back to plain resnet101** if no IBN loader works. Output 2048-d.
- Semantic branch `TinyCLIPImageBranch`: open_clip `hf-hub:wkcn/TinyCLIP-ViT-45M-32-Text-21M-LAION400M`
  -> `TinyCLIP-ViT-40M-32-Text-19M`(laion400m_e32) -> timm `vit_medium_patch32_clip_224.tinyclip_laion400m`
  -> last resort open_clip `ViT-B-32`(openai). Output dim auto-inferred (512 for TinyCLIP-45M). Interpolates
  its input to the branch native size internally.
- Fusion: `fusion_fc = Linear(2048+sem_dim -> embed_dim=2048, bias=False)`; **AFEM** (num_groups=32,
  residual_mode="grouped_identity": `t = t_u + afem(t_u)`); BNNeck (`bnneck`, bias frozen); classifier (bias-free).
- Forward returns L2-normalized `t_bn` (inference) or `(t_bn_norm, logits)` (train). **Feature dim = 2048.**

**Config (cell 3 CFG):**
- SEED = **3407**. image_size **(256, 256)** for TRAINING (extraction uses 320 — see Risk R3).
- RandomIdentitySampler: batch_p=**16**, batch_k=**8**, batch_size=**128**. accum_steps=**2** (effective 256).
- Memory fallback (cell 9): if peak alloc > 15.5 GB -> batch_p=12, batch_k=8, batch_size=96, accum_steps=3.
- epochs=**24**, warmup_epochs=**5**, eval_every=2, max_session_hours=11 (resume-aware).

**Optimizer / schedule (cell 7):**
- **Adam** (NOT AdamW), lr=**5e-4**, weight_decay=**5e-4**, no LLRD, no per-group LR.
- LambdaLR: linear warmup over 5 epochs then cosine `0.5*(1+cos(pi*progress))` to 0. AMP fp16 GradScaler.

**Losses (cell 6/train loop):** total = CrossEntropyLabelSmooth(eps=**0.1**) + SupConLoss(temp=**0.07**),
equal-weight sum, divided by accum_steps. No triplet, no center loss.

**Augmentations (cell 4):** Resize(256,256,BICUBIC)+RandomHorizontalFlip(0.5)+Pad(10)+RandomCrop(256,256)
+ToTensor+Normalize(**ImageNet**)+RandomErasing(p=0.5,value="random"). Eval: resize+totensor+normalize.

**Multi-GPU:** NONE. Single device, no DataParallel/DDP. Pins torch==2.4.1+cu124 (P100 sm_60 compat).

**Dataset parse (cell 4):** uses name_train/name_query/name_test list files if present; regex `([0-9-]+)_c([0-9]+)`,
cam 0-based, relabel train. num_classes from train (575).

**Checkpoints saved:**
- per-epoch resume `/kaggle/working/checkpoints/last.pth` (model_state, optimizer, scheduler, scaler, rng).
- best-by-mAP `/kaggle/working/checkpoints/best.pth` (same payload shape).
- final export (cell 11) `/kaggle/working/vehicle_clip_senet_veri776.pth` = **bare state_dict** (best.pth model_state)
  + `vehicle_clip_senet_veri776_metadata.json`.

### 1.3 FEATURE EXTRACTION (paper_features_a5alpha)

Imports repo `src.serving.reid_loaders` (clones MRKDaGods/gp @ branch `paper-tests`). Does NOT train.
- Stream 1 via `build_09v_model(A5ALPHA_CKPT)` -> TransReID, img 224, CLIP norm, num_cameras=20.
  `stream="concat_patch_flip"` -> CLS(768) || GeM(p=3) patch-pool(768), L2-normed -> **1536-D**;
  TTA = horizontal-flip averaged (mean of base+flip, each L2-normed, then re-L2). batch 64.
- Stream 2 via `build_clipsenet_model(CLIPSENET_CKPT)`; extract at **320x320**, ImageNet norm, NO TTA -> **2048-D**. batch 32.
- Output dtype **float16**. Files: `features/stream1/{query,gallery}.npy`, `features/stream2/{query,gallery}.npy`,
  `features/index_map.json` (per-row image_path, vehicle_id, camera_id, split + checkpoint paths).
- Discovery: Stream1 = first `a5alpha_checkpoint.pth`; Stream2 = scored search for
  `clipsenet_v6_veri776_best.pth|vehicle_clip_senet_veri776.pth|best.pth|best_mAP.pth`, prefers `13-clip-senet-train`.

### 1.4 FUSION + EVAL — exact math

**Eval protocol (standard VeRi/Market-1501, `eval_market1501`, 14t cell 14t-07):**
- argsort distmat ascending per query. Junk filter removes same-(pid AND camid):
  `remove = (g_pid==q_pid) & (g_cam==q_cam)`, keep rest -> standard cross-camera VeRi protocol.
- AP = sum(precision * relevance)/num_rel; CMC cumulative capped at 1; mAP = mean over valid queries.
- max_rank 50; R1/R5/R10 = CMC[0]/[4]/[9]. (Stream-2 train nb has its own `eval_reid` with the same junk
  filter; use the market1501 version as canonical so all streams match.)

**AQE (`average_query_expansion`):** on concat [query;gallery]: sim=X@X^T, top-k by sim (INCLUDES self),
mean-pool those k rows, L2-renormalize. iterations=1.
- Deployed paper fusion uses **k=3 for BOTH streams**. 14t main grid AQE_K=3; 14t drift block uses CLIP-SENet k=10.

**k-reciprocal rerank (`compute_reranking_torch`, Zhong et al.):**
- original_dist = `2 - 2*sim` (clamp >=0); initial_rank from topk sim; half_k1 = round(k1/2).
- reciprocal neighbor sets + 2/3-overlap expansion, Gaussian weights `exp(-original_dist)`, local QE over k2,
  Jaccard distance, final = `jaccard*(1-lambda) + original_dist*lambda`.
- **Canonical (stream1 + fusion): k1=80, k2=15, lambda=0.2.** Drift-only (stream2): k1=50, k2=10, lambda=0.1.

**Score-level fusion (HEADLINE, A5alpha cell 26 `compute_fusion`):**
1. L2-norm each stream q & g; concat -> all_tr (1536-D), all_cs (2048-D).
2. AQE each at k=3: `all_tr = aqe(all_tr, k=3)`, `all_cs = aqe(all_cs, k=3)`.
3. **sim = `0.3 * (all_tr @ all_tr^T) + 0.7 * (all_cs @ all_cs^T)`** (w_transreid=0.3, w_clipsenet=0.7).
4. original_dist = `2 - 2*sim`; rerank(80,15,0.2); eval_market1501 -> mAP/R1.
14t generalizes step 3 over w in {0.0..1.0} (`w*S_clipsenet + (1-w)*S_transreid`) and also tries concat-feature
fusion `L2([alpha*tr, (1-alpha)*cs])` for alpha in {0.3,0.5,0.7}; WIN row is the concat-AQE-k3 row.

**Result JSON (14t):** `/kaggle/working/{eval_results.json,14t_fusion_results.json,14t_summary.json,recipe.json}`;
summary verdict WIN/MARGINAL/FAIL vs {WIN: mAP>=0.9154 & R1>=0.9833}. A5alpha cells 23/26 ->
`experiments/veri776-paper/A5alpha/eval_results.json` (stream1_standalone + fusion).

---

## 2. Provenance inventory

**Stream-1 / TransReID checkpoint names referenced:**
- `vehicle_transreid_vit_base_veri776_A5alpha.pth` (A5alpha EXPORT: {state_dict,recipe})
- `transreid_veri_best.pth` / `best_mAP.pth` (A5alpha intermediate, raw state_dict)
- `a5alpha_checkpoint.pth` (REQUIRED by paper_features discovery) <- name A5alpha never actually writes
- `vehicle_transreid_vit_base_veri776.pth` (REQUIRED by 14t discovery, preferred slug mtmc-weights)

**Stream-2 / CLIP-SENet checkpoint names referenced:**
- `vehicle_clip_senet_veri776.pth` (v7 export, bare state_dict) + `best.pth`/`last.pth` (resume payloads)
- `clipsenet_v6_veri776_best.pth`, `best_mAP.pth` (alt names probed by feature/fusion discovery)

**Kaggle dataset/kernel sources:**
- abhyudaya12/veri-vehicle-re-identification-dataset (VeRi data — consistent everywhere; CANONICAL)
- mrkdagods/mtmc-weights (A5alpha + 14t weights)
- yahiaakhalafallah/13-clip-senet-train(-v7) (Stream-2 kernel output; 14t kernel_source)
- gumfreddy/paper-a5weights + gumfreddy/paper-clipv7 (paper_features weights)
- github MRKDaGods/gp (shared `src/`; A5alpha features uses branch `paper-tests`)

**INCONSISTENCIES (the "mess"):**
1. Three filenames for the SAME Stream-1 checkpoint: A5alpha writes `..._A5alpha.pth` + `best_mAP.pth`;
   paper_features wants `a5alpha_checkpoint.pth`; 14t wants `vehicle_transreid_vit_base_veri776.pth`.
2. Weight-dataset divergence: A5alpha/14t read `mrkdagods/mtmc-weights`; paper_features reads
   `gumfreddy/paper-a5weights` + `gumfreddy/paper-clipv7`. Two accounts host "the" weights.
3. Account sprawl across 4 owners (ali369, yahiaakhalafallah x2, gumfreddy) -> token hot-swap needed.
4. Stream-2 trained at 256 but evaluated/dumped at 320 (works via interpolation, undocumented).
5. AQE k for CLIP-SENet: 3 (deployed) vs 10 (14t drift). Only k=3 is the headline.
6. "v6" vs "v7" naming: training kernel is v7 while markdown/discovery still says v6.

**CANONICAL SOURCE RECOMMENDATION:**
- Stream 1: the `best_mAP.pth` from `ali369/paper-veri776-a5alpha` (raw state_dict). Re-publish ONCE as
  `transreid_a5alpha_veri776.pth`.
- Stream 2: `vehicle_clip_senet_veri776.pth` from `yahiaakhalafallah/13-clip-senet-train-v7` (best model_state).
  Re-publish as `clipsenet_v7_veri776.pth`.
- Put BOTH in ONE new dataset `<account>/veri776-canonical-weights` (exactly two files + `MANIFEST.json` with
  SHA-256). Retire mtmc-weights / paper-a5weights / paper-clipv7 as inputs to the new kernels.
- VeRi data: keep abhyudaya12/veri-vehicle-re-identification-dataset; pin its version.

---

## 3. Shared code vs inline (what to inline)

- **paper_features_a5alpha**: NOT self-contained — clones the repo and imports `src.serving.reid_loaders`
  (-> `src.stage2_features.transreid_model.build_transreid` + `scripts.eval.eval_clip_senet_veri776`).
- **14t**: mostly self-contained — inlines the whole CLIP-SENet class + all eval/AQE/rerank math, but still
  clones the repo to import `build_transreid` (cell 14t-05).
- **A5alpha train**: self-contained for training (TransReID inline cell 14), but fusion cell 26 re-clones + imports `build_transreid`.
- **13_clip_senet_train_v7**: fully self-contained (whole architecture inline cell 2).

**Must inline into the new kernels (no clone, no `src.` imports):**
1. `TransReID` + `build_transreid` loader (`src/stage2_features/transreid_model.py`, 331 lines) — including the
   state_dict remap (bottleneck->bn, classifier->cls_head, sie zero-pad, pos_embed bicubic interp) and the
   inference `_concat_patch` GeM(p=3) 1536-D path.
2. `CLIPSENet` + branches + AFEM (cell 14t-03 / 13_clip_senet cell 2, ~280 lines).
3. eval_market1501, average_query_expansion, build_rerank_state_from_similarity, compute_reranking_torch,
   VeRi parsing/transforms (already inline in 14t — copy verbatim).
All compiles from timm/open_clip/pretrainedmodels + torch; no repo needed.

---

## 4. Proposed design — 3 new self-contained kernels (2x T4 DDP)

All three: `enable_internet: true` (timm/open_clip/TinyCLIP download), fixed seeds, SHA-256 provenance logging
into `provenance.json` + stdout.

### Kernel A — `veri-canon-stream1-train` (TransReID A5alpha, 2xT4 DDP)
- metadata: enable_gpu, `machine_shape: "NvidiaTeslaT4x2"`, ds=[abhyudaya12/veri-...], no kernel/model sources.
- Recipe = 1.1 verbatim. DataParallel -> **DDP** via `mp.spawn(world_size=2)` (Kaggle notebooks can't run torchrun
  cleanly); wrap in DistributedDataParallel; DDP-aware PK sampler that shards PIDs by rank (per-rank P=6,K=4 to
  keep global P=12,K=4). All-gather features for eval on rank 0.
- DECISION: keep the ORIGINAL global batch (P=12,K=4,batch=48) split across 2 ranks (per-rank 24). Do NOT scale
  batch by GPU count (the old `*NUM_GPUS` scaling would change the recipe and the 93.30 result).
- Save canonical `transreid_a5alpha_veri776.pth` (raw state_dict, rank 0) + recipe.json + provenance.json.
- Add resume (mirror Stream-2 last.pth/rng pattern) — A5alpha has NO resume and 140 epochs may exceed one session.

### Kernel B — `veri-canon-stream2-train` (CLIP-SENet v7, 2xT4 DDP)
- metadata: same ds, `machine_shape: "NvidiaTeslaT4x2"`.
- Recipe = 1.2 verbatim (Adam 5e-4, wd 5e-4, 24 ep, warmup 5, CE-LS 0.1 + SupCon 0.07, img 256, P=16,K=8,batch=128,accum=2).
- DDP: SupCon is batch-local; naive per-rank SupCon shrinks the contrast set and shifts mAP.
  DECISION: **all-gather features+labels for the SupCon term** so the contrast set == original 128*accum; keep accum_steps.
- Pin torch/timm/open_clip/pretrainedmodels versions explicitly. On pure 2xT4 the cu124 P100 pin is optional.
- Export canonical `clipsenet_v7_veri776.pth` (bare state_dict from best mAP) + metadata + provenance.json.

### Kernel C — `veri-canon-fusion-eval` (single T4, inference only)
- metadata: enable_gpu, `machine_shape: "NvidiaTeslaT4"`,
  ds=[abhyudaya12/veri-..., <account>/veri776-canonical-weights], no kernel_sources (fully inline).
- Inline TransReID + CLIP-SENet + eval/AQE/rerank. Load the two canonical checkpoints by canonical names,
  log SHA-256, assert against MANIFEST.json.
- Extract Stream1 concat_patch_flip 1536-D (flip-avg TTA, img 224, CLIP norm); Stream2 2048-D (img 320, ImageNet, no TTA).
  Save fp16 features + index_map for retrieval panels.
- HEADLINE (reproduce 93.30/98.45): AQE k=3 both -> sim = 0.3*S_tr + 0.7*S_cs -> rerank(80,15,0.2) -> eval_market1501.
  Also emit standalone-stream rows + full w-sweep + concat grid for paper tables -> eval_results.json with seeds + SHAs.

### Common metadata template
```json
{
  "id": "<account>/veri-canon-<role>",
  "title": "VeRi Canon <Role>",
  "code_file": "<nb>.ipynb",
  "language": "python", "kernel_type": "notebook", "is_private": true,
  "enable_gpu": true, "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4x2",
  "dataset_sources": ["abhyudaya12/veri-vehicle-re-identification-dataset"],
  "competition_sources": [], "kernel_sources": [], "model_sources": []
}
```
Pin pip identically across all three: `timm==1.0.11 open_clip_torch==2.30.0 pretrainedmodels==0.7.4`
(A5alpha train used bare `timm` — pin to 1.0.11).

### Provenance logging (all three)
Log git/pip + torch/cuda/cudnn versions, GPU names, all seeds, VeRi path + file counts, SHA-256 of every
checkpoint read AND written, resolved CFG -> persist `provenance.json`.

---

## 5. Open risks

- **R1 DDP non-determinism:** even with cudnn.deterministic, NCCL all-reduce ordering + multi-worker loading
  make 1->2 GPU bit-exact unlikely. Pin GLOBAL batch composition, seed per rank, target +/- ~0.2pp mAP, document the band.
- **R2 SupCon under DDP (Stream 2):** naive per-rank SupCon shrinks contrast set and shifts mAP — all-gather
  features+labels for the SupCon term to match the single-GPU recipe.
- **R3 Stream-2 train(256) vs eval(320) mismatch:** keep both to reproduce the number, but log loudly. A clean
  redesign standardizing 320 train+eval needs a fresh sweep (out of scope).
- **R4 Dataset version pinning:** Kaggle datasets are mutable. Pin abhyudaya12/veri-... to a version and assert
  query=1678 / gallery=11579 / train~37778 + 575 train IDs at load time.
- **R5 TinyCLIP / IBN download dependence:** fallback chain can silently load a DIFFERENT backbone (plain
  resnet101, ViT-B-32) -> checkpoint mismatch. Assert `loaded_appearance_model == "resnet101_ibn_a"` and the
  expected TinyCLIP-45M tag before train/load; fail hard otherwise.
- **R6 Checkpoint format drift:** loaders accept 3+ payload shapes. Standardize on bare state_dict for both
  canonical files and assert format on load.
- **R7 Backbone tag drift:** hardcode `vit_base_patch16_clip_224.openai` (don't use dynamic `find_clip_vit_base`)
  to avoid timm picking a different openai/laion tag in a future version.
