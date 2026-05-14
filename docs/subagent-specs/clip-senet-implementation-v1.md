# CLIP-SENet Implementation Spec (v1)

**Source paper**: Lu, Fu, Chu, Wang, Xu. "CLIP-SENet: CLIP-based Semantic Enhancement Network for Vehicle Re-identification." arXiv:2502.16815v1, 24 Feb 2025. https://arxiv.org/html/2502.16815v1
**Status**: Planning only. No code yet. Authored by MTMC Planner.
**Goal**: Reproduce 92.9% mAP / 98.7% R1 on VeRi-776 and integrate as a swap-in for TransReID in our MTMC pipeline.

---

## 1. Paper Findings

### 1.1 Reported SOTA numbers (Tables I, III)
| Dataset | Metric | CLIP-SENet | Prior best | Δ |
|---|---|---|---|---|
| VeRi-776 | mAP / R1 / R5 | **92.9 / 98.7 / 99.1** | MBR 91.9 / 98.2 / 98.4 | +1.0 / +0.5 / +0.7 |
| VehicleID Small | R1 / R5 | **90.4 / 98.7** | MBR 88.3 / – | +2.1 |
| VehicleID Medium | R1 / R5 | 85.5 / 96.5 | SVRN 84.6 / – | +0.9 |
| VehicleID Large | R1 / R5 | 82.7 / 94.8 | ASSEN 82.4 / 94.3 | +0.3 / +0.5 |
| VeRi-Wild Small | mAP / R1 | 89.1 / 97.9 | ANet 86.9 / 96.5 | +2.2 / +1.4 |
| VeRi-Wild Medium | mAP / R1 | 85.2 / 97.0 | – | – |
| VeRi-Wild Large | mAP / R1 | 79.5 / 95.4 | ANet 75.9 / 92.5 | +3.6 / +2.9 |

Note: 92.9% mAP on VeRi-776 uses **re-ranking** (per § IV-B2: "we employed re-ranking technology as a post-processing step **only for the VeRi-776 dataset**"). VehicleID and VeRi-Wild numbers are without re-ranking.

### 1.2 Architecture (§ III)

Three components, all run in parallel on the same input image:

**(a) CNN Backbone** (§ III-A):
- ResNeXt101-IBN-a, ImageNet-pretrained, **final FC removed**
- GAP over the spatial map → `T_a ∈ R^{N×D}`, where `D = 2048` (ResNeXt101 stem dim)

**(b) Semantic Extraction Module / SEM** (§ III-B):
- TinyCLIP image encoder, `I(·;ε)`, weights "pretrained on LAION and YFCC-400M"
- Paper says "based on ViT-B/32" — see §6 risk register; closest published TinyCLIP variant is **ViT-45M/32 or ViT-63M/32 (auto, LAION+YFCC-400M)** from Microsoft Cream/TinyCLIP. https://github.com/microsoft/Cream/tree/main/TinyCLIP
- Output: `T_s` (CLS-token embedding, dim **512** for the standard CLIP-style image projection)
- Fusion: `T_u = FC([T_s ⊕ T_a])` — concatenate (`⊕`) along feature dim then linear-project to **2048**.

**(c) Adaptive Fine-grained Enhancement Module / AFEM** (§ III-C):
- Input: raw `T_s` from SEM
- A `f_linear` block = (Linear → BN → ReLU) produces `G+1` vectors. One branch is the residual (un-weighted); the other `G` are "grouped vectors" with per-group learnable weights `w_i` initialized from `N(0,1)`.
- Equation (4) in paper: `T_s' = f_linear(T_s) + Σ_{i=0..G} w_i ⊗ f_linear(T_s)`
- `⊗` = element-wise product, `+` = element-wise add
- Best `G = 32` (Table V; ablations show 4→32 monotone improvement, 64+ degrades)
- Output `T_s'` shape must equal `T_u` shape (2048) for the final addition.

**Final feature**: `T = T_u + T_s'` (Eq. 5)

**Training-time classifier**: `T → FC(num_classes) → ŷ` for cross-entropy. Inference uses `T` directly (no BNNeck explicitly mentioned, but standard ReID practice would insert one — flag below).

### 1.3 Losses (§ III-C)

Total: `L = L_CE + L_SupCon` (Eq. 8) — equal weights.

- **L_CE**: smoothed cross-entropy, smoothing factor `ε = 0.1` (Eq. 6). NB: paper reuses symbol `ε` for both label smoothing and TinyCLIP weights — context disambiguates.
- **L_SupCon**: Supervised Contrastive (Khosla et al., NeurIPS 2020). Eq. 7 in the paper is a simplified single-positive form — actual SupCon averages over all positives in batch; assume standard `pytorch-metric-learning` or Khosla reference impl.
- Temperature `τ`: **NOT specified in paper**. Default to `τ = 0.07` (Khosla recommendation).
- Triplet loss is explicitly **rejected** in favour of SupCon (Fig. 3 ablation).

### 1.4 Training recipe (§ IV-B1)
| Hyperparameter | VeRi-776 | VehicleID | VeRi-Wild |
|---|---|---|---|
| Image size | 320×320 | 320×320 | 320×320 |
| Optimizer | ADAM | ADAM | ADAM |
| LR schedule | Cosine annealing | WarmupMultiStepLR | WarmupMultiStepLR |
| Initial LR | 5e-4 | 3.5e-4 | 3.5e-4 |
| Batch size | 128 (P=16, K=8) | 128 | 128 |
| Epochs | 24 | 120 | 120 |
| Random seed | 3407 (global) | 3407 | 3407 |
| Hardware | 1× A40 | 1× A40 | 1× A40 |
| Augmentations | "various" — UNDERSPECIFIED | same | same |

CV (camera + viewpoint) information is added — paper labels this "CV" (Table IV column). For VeRi-776 the CV gain alone is small (+0.1 mAP), but combined with SEM+AFEM gives the final +1.5 mAP. For VehicleID single-camera, CV is not applicable. For VeRi-Wild only camera info is available. Implementation likely uses **SIE-style camera/viewpoint embedding** added to the ViT side (similar to TransReID), but **paper does not specify the exact integration point**.

### 1.5 Evaluation protocol (§ IV-B2)
- Metrics: mAP, Rank-1, Rank-5
- Re-ranking: **VeRi-776 only**. k-reciprocal (Zhong CVPR 2017) implied, hyperparams (k1, k2, λ) not given — default to k1=20, k2=6, λ=0.3.
- Test image size: not specified; assume 320×320 (matches training).
- Flip TTA: not specified.
- VeRi-776: 1,678 query / 11,579 gallery / 200 test IDs.
- VehicleID: 3 splits — Small (800), Medium (1600), Large (2400). Standard protocol picks 1 image per ID as gallery, others as query, repeated 10 times averaged.
- VeRi-Wild: Small (3000) / Medium (5000) / Large (10000) test IDs.

### 1.6 Ablation evidence (Tables IV–VI, Fig. 3)
- Baseline (ResNeXt101-IBN + CE + SupCon, no CLIP): 86.7 / 96.8 / 97.9
- + SEM: 87.3 / 97.0 / 97.6 (+0.6 mAP — modest)
- + SEM + AFEM: **91.4** / 97.6 / 98.5 (+4.7 mAP — AFEM is the key contribution)
- + SEM + AFEM + CV: 92.9 / 98.7 / 99.1 (CV adds +1.5 mAP on top)
- AFEM groups: G=4→90.2, G=8→91.1, G=16→91.7, **G=32→92.9**, G=64→90.3, G=128→90.6.
- CNN backbone: ResNet50→91.1, ResNet101→91.9, SE-ResNet101→91.0, **ResNeXt101→92.9**.
- Loss: SupCon > Triplet on both baseline and full model (Fig. 3, exact deltas not in our extract).

---

## 2. Public Code Status

**No public code released as of 2026-05-06.**
- arXiv abstract page: no "Code" link.
- GitHub search `CLIP-SENet`: 0 repository hits. https://github.com/search?q=CLIP-SENet&type=repositories
- Corresponding author Zihao Fu's GitHub (`FU032`): no public repos. https://github.com/FU032
- PapersWithCode redirects to HuggingFace; HF page returns no extractable content (likely no code linked).
- Other authors (Liping Lu, Bingrong Xu, Duanfeng Chu, Wei Wang at WHUT/SYSU) — not searched exhaustively but no obvious repo.

**Implication**: Full reimplementation required. **HIGH reproducibility risk** — common for papers without code to omit key tricks (init scales, augmentation pipeline, BNNeck placement, exact SupCon variant).

---

## 3. Architecture Spec — `src/stage2_features/clip_senet_model.py`

Add as a **new module parallel to `transreid_model.py`** — do NOT modify TransReID.

### 3.1 Module signature (pseudocode, no implementation)
```python
class AFEM(nn.Module):
    """Adaptive Fine-grained Enhancement Module.

    Input:  T_s ∈ R^{N×d_s}  (raw semantic features from TinyCLIP)
    Output: T_s' ∈ R^{N×d_out}

    Internally:
      shared_linear: Linear(d_s → d_out) + BN(d_out) + ReLU
      group_weights: nn.Parameter(torch.randn(G, d_out))  # init N(0,1)
      forward:
        h = shared_linear(T_s)          # residual branch
        weighted = sum_i (group_weights[i] * h)  # element-wise * + sum
        return h + weighted
    """
    def __init__(self, in_dim: int, out_dim: int = 2048, num_groups: int = 32): ...

class CLIPSENet(nn.Module):
    """CLIP-SENet: ResNeXt101-IBN + TinyCLIP-ViT + AFEM.

    Args:
        num_classes: int           # for training-time CE head
        cnn_backbone: str          # 'resnext101_ibn_a' (default)
        clip_variant: str          # 'tinyclip_vit_45m_32_laion400m' or similar
        clip_weights_path: str     # local path to TinyCLIP .pt (Kaggle dataset)
        afem_groups: int = 32
        feat_dim: int = 2048
        use_cv: bool = False       # camera+viewpoint side info (VeRi-776 only)
        num_cameras: int = 0
        num_viewpoints: int = 0
        label_smoothing: float = 0.1

    Modules:
        backbone:    ResNeXt101-IBN-a (timm or boxmot/IBN-Net), final FC removed
        sem:         TinyCLIP image encoder (CLS embedding, dim=512)
        fusion_fc:   Linear(2048 + 512 → 2048)        # produces T_u
        afem:        AFEM(in_dim=512, out_dim=2048, num_groups=32)
        bnneck:      nn.BatchNorm1d(2048, affine=True), bias frozen  # PROPOSED, not in paper
        classifier:  nn.Linear(2048, num_classes, bias=False)        # for CE
        (optional)   sie_camera, sie_view embeddings if use_cv

    forward(x, cam_id=None, view_id=None):
        T_a = GAP(backbone(x))                    # (N, 2048)
        T_s = sem(x)                              # (N, 512)
        T_u = fusion_fc(cat([T_s, T_a], dim=-1))  # (N, 2048)
        T_s_prime = afem(T_s)                     # (N, 2048)
        T = T_u + T_s_prime                       # (N, 2048)
        if training:
            T_bn = bnneck(T)
            logits = classifier(T_bn)
            return logits, T  # T used for SupCon
        return F.normalize(T, dim=-1)             # inference embedding
```

### 3.2 Key shape/dim decisions
- ResNeXt101-IBN-a output: 2048-dim post-GAP.
- TinyCLIP ViT-45M/32 image-projection output: 512-dim (verified via Microsoft TinyCLIP HF cards).
- Concat: 2048 + 512 = 2560 → fusion FC → 2048.
- AFEM internal `f_linear`: 512 → 2048 (so `T_s'` matches `T_u`).
- Final `T` is 2048-dim.

### 3.3 Proposed deviations from paper (flagged)
- **BNNeck**: not mentioned in paper but standard for ReID; recommend including (consistent with "Bag of Tricks", which the paper cites and uses indirectly via baseline). Disable if it hurts.
- **Camera/viewpoint embedding**: paper does not specify integration. Propose two options to A/B test in M3:
  - (CV-A) SIE-style: add learned `(num_cams + num_views, 2048)` embedding to `T` before classifier.
  - (CV-B) Append CV embedding to `T_s` before AFEM (more in spirit of "semantic enhancement").

---

## 4. Training Spec

### 4.1 Optimizer & schedule
- **Optimizer**: `torch.optim.Adam(lr=5e-4, betas=(0.9,0.999), weight_decay=1e-4)`.
  - Paper says "ADAM" — interpret literally as Adam, not AdamW. Try AdamW as an ablation only if Adam fails.
- **Scheduler (VeRi-776)**: `CosineAnnealingLR(T_max=24, eta_min=1e-7)` over 24 epochs.
- **Scheduler (VehicleID, VeRi-Wild)**: WarmupMultiStepLR — paper does not give milestones. Use TransReID/FastReID convention: 10-epoch warmup from 3.5e-5 → 3.5e-4, then ×0.1 at epochs 40 and 70 (over 120 total). Document and revisit if results miss target.

### 4.2 Batch composition
- **PK sampler**: P=16 identities × K=8 instances = batch size 128.
- Reuse the random-identity sampler from our existing `09v_*` training (TransReID baseline).

### 4.3 Augmentations (UNDERSPECIFIED — propose standard ReID stack)
1. `Resize(320, 320)` (or `Resize(int(1.125*320), int(1.125*320))` then `RandomCrop(320,320)` — pick crop variant, more standard).
2. `RandomHorizontalFlip(p=0.5)`.
3. `Pad(10) + RandomCrop(320,320)`.
4. `ToTensor` + `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])` for ResNeXt path; CLIP normalization for TinyCLIP path is `mean=[0.4815,0.4578,0.4082], std=[0.2686,0.2613,0.2758]`. **Decision**: feed two separately-normalized tensors to the two branches; avoids changing TinyCLIP statistics.
5. `RandomErasing(p=0.5, scale=(0.02,0.4), ratio=(0.3,3.3))` applied to CNN-branch tensor only (TinyCLIP semantic features benefit from clean input).

### 4.4 Losses
- `L_CE = SmoothCrossEntropy(epsilon=0.1)`
- `L_SupCon = SupConLoss(temperature=0.07)` — use Khosla 2020 reference impl or `pytorch-metric-learning` SupConLoss
- Total: `L = L_CE + L_SupCon` (equal weights)
- Apply CE to `logits` (post-BNNeck features), SupCon to `T` (pre-BNNeck, L2-normalized inside SupCon).

### 4.5 Initialization
- ResNeXt101-IBN-a: ImageNet weights from IBN-Net (https://github.com/XingangPan/IBN-Net releases or timm's `resnext101_32x8d_ibn_a`).
- TinyCLIP: official Microsoft Cream weights, host as a Kaggle dataset.
- All new FC layers: Kaiming init (per Algorithm 1 line 2).
- AFEM `group_weights`: `torch.randn(G, d_out)` (standard normal).
- Global seed: 3407 (set torch, numpy, python random).

---

## 5. Eval Spec

- Image size at eval: **320×320** (assume matches training).
- TTA: optional horizontal-flip averaging — apply only as ablation, not default (paper does not specify).
- Re-ranking: VeRi-776 only, k-reciprocal `k1=20, k2=6, λ=0.3` (assume; tune if 92.9% target missed by >0.5 mAP).
- Distance: cosine similarity on L2-normalized `T` (2048-dim).
- Compute on Kaggle T4 (~5 min for 11k gallery × 1.7k query).

---

## 6. Dataset Access Plan

### 6.1 VeRi-776
- Already on Kaggle: `abhyudaya12/veri-vehicle-re-identification-dataset` (verified, used by `09v_veri776_eval`).
- Standard split: 576 train IDs / 200 test IDs / 1678 query.

### 6.2 VehicleID
- Search Kaggle: candidate slugs (need to verify before push):
  - `philiplsm/vehicleid` (not yet verified)
  - `lkstudent/vehicleid-dataset` (not yet verified)
- If unavailable: dataset requires registration at PKU-VehicleID (https://www.pkuml.org/resources/pku-vehicleid.html). Mirror to a private Kaggle dataset.
- 13,164 train IDs / 13,164 test IDs split into 800/1600/2400.

### 6.3 VeRi-Wild
- **Defer to post-M5**. Less relevant for our MTMC/CityFlowV2 goal and adds significant compute (~10× VeRi-776).
- If pursued: dataset on Kaggle as `xxx/veri-wild` (search needed) or mirror from PKU.

### 6.4 TinyCLIP weights
- Create new Kaggle dataset `<owner>/tinyclip-vit-45m-32-laion-yfcc` mirroring `wkcn/TinyCLIP-ViT-45M-32-Text-18M-LAION400M` (HF) or the LAION+YFCC-400M auto variant. https://huggingface.co/wkcn

---

## 7. Kernel Skeleton Plans

### 7.1 `notebooks/kaggle/13_clip_senet_train_veri776/` (GPU, 2× P100)

| Cell | Type | Purpose |
|---|---|---|
| 1 | md | Title, paper ref, target metrics |
| 2 | code | `pip install` timm, open_clip_torch, pytorch-metric-learning, ftfy, regex |
| 3 | code | Mount datasets, set paths (`VERI_ROOT`, `TINYCLIP_PATH`, `IBN_WEIGHTS`) |
| 4 | code | Set seed 3407 (torch, np, random, cudnn deterministic) |
| 5 | code | Define `ResNeXt101IBN_a` loader (load IBN-Net pretrained, strip FC) |
| 6 | code | Define `load_tinyclip_image_encoder()` from local weights |
| 7 | code | Define `AFEM` module |
| 8 | code | Define `CLIPSENet` full model |
| 9 | code | Define `VeRi776Dataset` + PK-sampler |
| 10 | code | Define augmentation pipelines (CNN + CLIP normalization) |
| 11 | code | Define `SupConLoss` + `LabelSmoothCE` |
| 12 | code | Build optimizer (Adam 5e-4) + cosine scheduler (T_max=24) |
| 13 | code | Training loop (24 epochs), per-epoch query/gallery mini-eval |
| 14 | code | Save checkpoint to `/kaggle/working/vehicle_clip_senet_veri776.pth` |
| 15 | code | Print final train metrics |

Estimated wall time: ~6–8 h on 2× P100 (paper used 1× A40, ~24 h for 24 epochs likely; P100 is ~50% of A40 throughput, so DDP across 2× ≈ similar). **Risk**: may exceed Kaggle 12 h GPU budget — add resume-from-checkpoint logic.

### 7.2 `notebooks/kaggle/13e_clip_senet_eval_veri776/` (GPU, T4)

Mirror `09v_veri776_eval/`:
| Cell | Type | Purpose |
|---|---|---|
| 1 | md | Title |
| 2 | code | pip install deps |
| 3 | code | Load checkpoint + TinyCLIP weights |
| 4 | code | Build `CLIPSENet` in eval mode |
| 5 | code | Extract query + gallery features (320×320, optional flip-TTA) |
| 6 | code | Compute cosine distance matrix |
| 7 | code | k-reciprocal re-ranking (k1=20, k2=6, λ=0.3) |
| 8 | code | Compute mAP, R1, R5, R10 |
| 9 | code | Print + save `eval_results.json` |

### 7.3 (Later) `13_clip_senet_train_vehicleid/` and `13e_eval_vehicleid/`
- Same skeleton as 7.1/7.2.
- Eval: 10-trial averaged R1/R5 over the 3 gallery splits.
- 120 epochs × 13k IDs ≫ Kaggle budget — must run with resume across 2–3 sessions.

---

## 8. Milestone Plan

| ID | Goal | Acceptance criterion | Est. Kaggle GPU-hours |
|---|---|---|---|
| **M1** | Port architecture: `clip_senet_model.py` + unit tests for forward-pass shapes (random input → output shape `[N, 2048]`; gradient flow OK) | Forward + backward pass works on CPU and CUDA; shapes match spec § 3 | 0 (local CPU) |
| **M2** | Build `13_clip_senet_train_veri776` notebook; smoke-test 1 epoch on Kaggle | 1 epoch completes < 30 min, loss decreasing, eval mAP > 50% | 1 |
| **M3** | Full 24-epoch training + eval on VeRi-776 | mAP ≥ 90.0% (within 3pp of paper's 92.9%); R1 ≥ 97.5% | 8–12 |
| **M4** | (Conditional on M3 ≥ 90% mAP) VehicleID training + eval on Small split | R1 ≥ 88% on Small (paper: 90.4%) | 25–40 |
| **M5** | Stage-2 integration: add `clip_senet` choice to feature extractor; rerun MTMC pipeline on CityFlowV2 with new ReID | MTMC IDF1 ≥ 0.78 (current best 0.7703 with CLIP+DINOv2 fusion); ideally ≥ 0.80 | 5 |
| **M6** | Documentation, ablation tables, update `docs/findings.md` and `docs/experiment-log.md`; figure regeneration | All updates landed; new dead-ends/wins recorded | 0 |

---

## 9. Risk Register

| Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|
| **No public code** — undisclosed tricks (init scales, exact SupCon variant, augmentation order) | **HIGH** | Certain | Use Bag-of-Tricks defaults; ablate aggressively; flag any > 1pp gap from paper as suspect |
| **TinyCLIP variant ambiguity** ("ViT-B/32" but TinyCLIP has 22M/40M/45M/61M/63M variants) | HIGH | Certain | Try ViT-45M/32 (LAION+YFCC-400M auto) first — best size/perf tradeoff. Fallback: ViT-63M/32 |
| **AFEM equation parsing** — Eq. 4 has `Σ_{i=0..G}` over `G+1` vectors but paper text says "G grouped vectors" | MEDIUM | Likely | Implement as G groups + 1 residual; if ablation at G=32 underperforms, try alternative grouping (split feature dim into G chunks each with own scalar weight) |
| **Camera/viewpoint integration unspecified** | MEDIUM | Likely | A/B test SIE-style vs feature concat in M3 |
| **Kaggle GPU budget**: 24-epoch VeRi training + 120-epoch VehicleID may exceed 12-h limits | MEDIUM | Likely | Save checkpoints every epoch; build resume logic in cell 12 of training kernel |
| **VehicleID dataset access** | MEDIUM | Possible | Defer to M4; verify Kaggle slug or mirror privately before pushing |
| **SupCon temperature unspecified** | LOW | Certain | Default τ=0.07; small sweep [0.05, 0.07, 0.1] if M3 misses target |
| **Re-ranking hyperparams unspecified** | LOW | Certain | Default k1=20, k2=6, λ=0.3 |
| **ResNeXt101-IBN-a weights mismatch** (we previously hit this with 32x8d vs 32x32d) | MEDIUM | Possible | Verify timm/IBN-Net variant: the 32x8d is what IBN-Net officially distributes. Use `strict=True` load and abort on missing keys |
| **AFEM `T_s` dim ≠ paper's `T_u` dim** for residual sum — paper assumes same dim | LOW | Resolved | Spec § 3.2 makes both 2048 |

---

## 10. Decision Points (Stop-and-Ask Triggers)

| Trigger | Action |
|---|---|
| M3 final mAP < 85% | **STOP**. Likely a fundamental misinterpretation of AFEM or SEM fusion. Escalate to user with full ablation table before retrying |
| M3 final mAP 85–88% | **CONTINUE**, but flag as partial reproduction. Run loss/temperature/augmentation ablation before declaring done |
| M3 mAP 88–90% | **CONTINUE** to M4 with caveat noted in `findings.md` |
| M3 mAP ≥ 90% | **CONTINUE** to M4/M5; consider VeRi-Wild as stretch |
| M4 VehicleID R1 < 85% on Small | **STOP**. Likely VehicleID-specific scheduler issue (WarmupMultiStepLR milestones); ask user before extra runs |
| M5 MTMC IDF1 < 0.7703 (regression vs current best) | **STOP**. Do not deploy. Document as dead-end in `findings.md` |
| Any kernel run shows "not valid dataset sources" warning | **CANCEL immediately**, fix metadata, re-push (per `.github/copilot-instructions.md` Kaggle Push Safety Rules) |
| TinyCLIP weights cannot be obtained or hosted | **STOP**. Ask user whether to substitute OpenAI CLIP ViT-B/32 (deviates from paper but available) |

---

## 11. Open Questions for User (before M2)

1. **TinyCLIP variant**: proceed with **ViT-45M/32 (LAION+YFCC-400M auto)** as default, or test ViT-63M/32 first?
2. **VehicleID priority**: pursue M4 only if M3 succeeds, or also queue M4 in parallel?
3. **VeRi-Wild**: in-scope or defer indefinitely?
4. **Compute budget**: confirm willingness to spend ~50 GPU-hours across M2–M5.
5. **Account routing**: which Kaggle account hosts the new `13_*` kernels (currently active = gumfreddy)?

---

## 12. Citations

- Paper (HTML): https://arxiv.org/html/2502.16815v1
- Paper (PDF): https://arxiv.org/pdf/2502.16815v1
- arXiv abs page: https://arxiv.org/abs/2502.16815
- TinyCLIP code + model zoo: https://github.com/microsoft/Cream/tree/main/TinyCLIP
- TinyCLIP HF collection: https://huggingface.co/collections/wkcn/tinyclip-model-zoo-6581aa105311fe07be88cb0d
- IBN-Net: https://github.com/XingangPan/IBN-Net
- SupCon (Khosla 2020): https://arxiv.org/abs/2004.11362
- CLIP-ReID (cited baseline): Li et al., AAAI 2023
- MBR (cited prior SOTA): Almeida et al., ITSC 2023
- Author's GitHub (no public repos): https://github.com/FU032
- VeRi-776 Kaggle: `abhyudaya12/veri-vehicle-re-identification-dataset`
- Existing 09v eval kernel for skeleton reference: `notebooks/kaggle/09v_veri776_eval/`