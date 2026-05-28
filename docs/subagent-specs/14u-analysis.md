# 14u Analysis — Why CLIP-SENet × TransReID Fusion Doesn't Transfer to CityFlow MTMC

**Status**: ANALYSIS (planner postmortem; informs strategy, not implementation)  
**Author**: MTMC Planner, 2026-05-16  
**Type**: Strategic analysis, no code or experiment plan

---

## 1. Question

The 14t fusion, CLIP-SENet v6 × TransReID 09v v17, reached **mAP=0.9330 / R1=0.9845** on VeRi-776 single-camera ReID.

That was a clear WIN.

It lifted mAP by +3.33pp over the strongest single parent row.

It did so with a simple recipe.

The recipe was score-fusion.

The score-fusion weight was `w_clipsenet=0.7`.

The TransReID weight was `w_transreid=0.3`.

The TransReID stream was the 768-d global-token stream.

The post-processing was AQE k=3.

The rerank parameters were `k1=80`, `k2=15`, `lambda=0.2`.

The natural follow-up was 14u.

14u asked whether the same fusion geometry would lift CityFlowV2 MTMC IDF1.

The port was not a naive replacement-only test.

The planner spec identified that replacing the production CityFlow primary with two VeRi-only experts would be confounded and probably harmful.

The useful 14u variant was Option C.

Option C added the 14t-fused stream as a fourth score-fusion stream.

It preserved the production CityFlow CLIP-TransReID primary.

It preserved the DINOv2 tertiary stream.

It tested whether the 14t fused-and-reranked similarity carried extra cross-camera signal.

The result was negative.

The 14u sweep had 19 CPU-only configs.

The best configs were at `w_14t=0.10-0.15` and `thr=0.48-0.50`.

Best MTMC IDF1 was **0.77995**.

Best ID switches were **160**.

The 14e B1 anchor was **0.77936 / id_switches=154**.

The gain was **+0.00059 IDF1**.

That is well under the historical ~0.0024 noise band.

ID switches went up, not down.

That means the best score added more conflation rather than a cleaner association signal.

The strict 14u spec verdict was FAIL.

This document diagnoses why.

It considers four original hypothesis classes plus a fifth plateau-level synthesis.

The conclusion is that 14u failed for structural reasons, not because a small threshold was missed.

The 14t mechanism is real.

The 14t mechanism is valuable.

The 14t mechanism is also image-level and single-camera benchmark shaped.

CityFlow MTMC is tracklet-level and cross-camera shaped.

That geometry change is the center of the diagnosis.

Primary citations:

- 14t source result: [docs/findings.md](../findings.md#L331-L365)
- 14u result: [docs/findings.md](../findings.md#L367-L377)
- 14u algebraic reduction and prior evidence: [14u-cityflow-veri-fusion-port.md](14u-cityflow-veri-fusion-port.md#11-honest-framing-of-prior-evidence)

---

## 2. Hypothesis Matrix

### Summary Table

| Hypothesis | Short Name | Verdict | Rescue Potential |
|---|---|---|---|
| H1 | Pure VeRi-776 -> CityFlowV2 domain gap | HIGH | Partial only |
| H2 | Fine-tune helps but becomes redundant | HIGH | Low without new pretraining family |
| H3 | CityFlow fine-tune TransReID 09v | N/A | Already represented by production primary |
| H4 | Fusion mechanism is wrong for MTMC geometry | VERY HIGH | Requires learned association or different geometry |
| H5 | Production pipeline is already saturated multi-stream fusion | HIGH | More score streams unlikely to help |

---

### H1 — Pure VeRi-776 -> CityFlowV2 Domain Gap

This is the obvious hypothesis.

It says 14u failed because the two 14t experts were trained on VeRi-776.

CityFlowV2 is a different domain.

The cameras differ.

The scene statistics differ.

The vehicle crops differ.

The annotation setup differs.

The cross-camera target differs.

The deployment metric differs.

Evidence for H1 is strong.

13d v2 tested CLIP-SENet v6 on CityFlowV2.

CLIP-SENet v6 was a strong VeRi-776 expert.

Its VeRi-776 post-rerank mAP was 0.9154.

Its VeRi-776 R1 was 0.9732.

Despite that, standalone CLIP-SENet on CityFlow reached only 0.6855 IDF1.

The production primary was far stronger standalone.

Adding CLIP-SENet at positive weights degraded CityFlow fusion.

The degradation was monotonic enough to be diagnostic.

At `w_cs=0.2`, the score was already below control.

At `w_cs=0.6`, the drop was around -1.77pp.

At `w_cs=0.8`, the drop was around -3.68pp.

Standalone was around -8.24pp from the relevant control.

That is not tuning noise.

That is domain mismatch.

14u used CLIP-SENet v6 again.

14u also used TransReID 09v v17.

That 09v checkpoint was VeRi-776-only.

It was not the production CityFlow-trained TransReID checkpoint.

Therefore both extra experts in the 14t-derived stream were out-of-domain for CityFlow.

H1 explains why the standalone contribution should be weak.

H1 explains why the optimum weight in 14u sat near the lower boundary.

The best weights were `w_14t=0.10-0.15`.

Higher weights quickly moved backward.

That mirrors 13d.

The system only tolerated a small amount of the VeRi-derived signal.

Evidence against H1 is also important.

13h showed that domain adaptation works.

A CityFlow-fine-tuned CLIP-SENet recovered standalone IDF1 from 0.6855 to 0.7099.

That was a +2.44pp standalone lift.

So the domain gap is not absolute.

The CLIP-SENet architecture can learn useful CityFlow appearance.

But the fusion peak after fine-tuning was only 0.7691.

That was still below production 0.7703 at that time.

So H1 cannot be the whole story.

If pure domain gap were the only problem, CityFlow fine-tuning should have rescued fusion.

It did not.

H1 verdict: **HIGH**.

H1 is real.

H1 explains a large part of the failure.

H1 does not fully explain why the 14t fused-rerank geometry failed to add signal even as a low-weight fourth stream.

Rescue potential for H1: partial.

Fine-tuning can improve standalone CLIP-SENet.

Fine-tuning does not guarantee additive MTMC signal.

Any rescue must also address H2 and H4.

---

### H2 — Fine-Tune Helps CLIP-SENet but the Fine-Tuned Stream Is Redundant

H2 says the CityFlow-adapted version of CLIP-SENet becomes too correlated with the production feature stack.

This is a different claim from H1.

H1 says out-of-domain features are weak.

H2 says in-domain fine-tuned features can be non-additive.

The key evidence is 13f/13h.

CLIP-SENet v6 was fine-tuned on 666 CityFlow IDs.

Standalone IDF1 improved.

The standalone number moved from 0.6855 to 0.7099.

That is meaningful.

It confirms the architecture can adapt.

But fusion peak was `w_cs_ft=0.30 -> 0.7691`.

That was -0.12pp below production 0.7703.

This means the fine-tuned stream did not provide useful residual information.

It may have improved its own geometry.

It did not improve the ensemble geometry.

Mechanistically, this is plausible.

CLIP-SENet combines a CNN appearance branch with a TinyCLIP semantic branch.

The production primary is CLIP TransReID.

The production tertiary is DINOv2 ViT-L/14.

Both production streams already encode strong semantic and appearance features.

Both are robust to vehicle shape and color.

Both have strong pretrained visual priors.

After CityFlow fine-tuning, CLIP-SENet likely moves toward the same discriminative cues.

It learns the same fleet colors.

It learns the same common viewpoints.

It learns the same camera-specific lighting shifts.

It learns the same coarse body type separations.

The residual errors may remain the same hard cases.

The stream becomes better but not complementary.

14u reinforces H2 indirectly.

14u did not fine-tune CLIP-SENet again.

But it added a stronger fused VeRi-derived stream.

Even that produced only +0.00059.

The pattern is that score-level additions in this family have tiny returns.

H2 is also supported by 14k.

14k added an R50-IBN stream.

R50-IBN was not the same architecture as CLIP TransReID.

It was still only marginal.

The peak was 0.78079.

That was +0.00143 over 14e B1.

It missed the 0.7810 WIN bar.

So redundancy is not only a CLIP-SENet issue.

It is a score-stream addition issue.

H2 verdict: **HIGH**.

Fine-tuning can rescue standalone domain gap.

Fine-tuning does not rescue ensemble complementarity.

Rescue potential: low unless the stream comes from a genuinely different pretraining family.

A CLIP-SENet variant with non-CLIP pretraining might be more decorrelated.

A DINOv2-initialized CNN or EVA-family ViT might help more.

But that becomes a new multi-day GPU training project.

It is no longer a simple 14t port.

---

### H3 — TransReID 09v v17 Fine-Tuned on CityFlow

H3 asks the symmetric question.

Could we fix 14u by fine-tuning TransReID 09v v17 on CityFlow?

The key fact is that this already exists in substance.

The production primary is a CityFlow-trained TransReID CLIP ViT model.

It is not the 09v VeRi checkpoint.

But it is the CityFlow-domain TransReID expert in the same broad family.

Its checkpoint is represented in the production stack as `transreid_cityflowv2_best.pth`.

Its CityFlow mAP is recorded as 81.53 in the project instructions.

It dominates the association stack.

Therefore a CityFlow-fine-tuned 09v stream would collapse into the role already occupied by the primary.

If 14u replaced the 09v VeRi stream with the production primary, the formula would no longer test 14t transfer.

It would double-count the production primary.

If 14u replaced the production primary with a CityFlow-fine-tuned 09v, it would be a primary-model swap, not a 14t fusion port.

If 14u added both, the new stream would be highly correlated with the existing primary.

This is exactly the redundancy trap.

The 14u planner spec already warned about algebraic reductions.

User-proposed Option B would have used a fused primary:

`S_final = w_p * [w_clip * S_CLIPSENet + w_trans * S_TransReID09v] + w_t * S_DINOv2`.

Algebraically this is a three-way score fusion.

It drops the CityFlow primary if used as replacement.

Dropping the CityFlow primary is not viable.

Keeping it makes the TransReID part redundant.

H3 verdict: **N/A**.

The CityFlow-trained TransReID variant already exists.

It is the production primary.

There is nothing useful to rescue under the 14u question.

---

### H4 — Fusion Mechanism Is Wrong for MTMC Geometry

H4 is the strongest explanation.

It says the 14t fusion mechanism is not merely out-of-domain.

It is structurally suited to a different evaluation geometry.

VeRi-776 single-camera ReID has image-level query-gallery evaluation.

There are about 1,678 query images.

There are about 11,579 gallery images.

Each query is a single image.

Each gallery item is a single image.

The metric rewards ranking all gallery images of the same identity.

The gallery often has multiple positives for a query.

Those positives cover varied camera angles.

Those positives create useful neighborhood structure.

AQE and k-reciprocal reranking exploit that structure.

When a query image is noisy, its true neighbors can pull it into a better local manifold.

When a model misses a top-ranked positive, another model can lift it.

Score fusion improves the initial neighbor set.

AQE expands the improved neighbor set.

Rerank sharpens reciprocal consistency.

That is why 14t can gain +3.33pp mAP.

CityFlowV2 MTMC is different.

It has about 929 tracklets in the evaluated feature set.

The objects being matched are not single images.

They are tracklet-pooled features.

Each feature already summarizes multiple frames.

The production Stage 2 tracklet embedding uses softmax-quality-weighted pooling.

The 14c/14e family added multi-crop TTA smoothing.

The tracklet feature is already a bag-of-views representation.

It has less single-image noise for AQE to fix.

It also has fewer true cross-camera positives per query.

For many tracklets, there may be zero true cross-camera match in the retained candidate set.

For others, there may be one.

The neighborhood is sparse.

The k-reciprocal set is therefore less likely to be a true-positive set.

Neighbor expansion can recruit visually similar false vehicles.

This creates conflation.

14u's id_switch increase is exactly that symptom.

The best score had ID switches 160.

The anchor had ID switches 154.

The metric barely moved.

The added stream did not find cleaner true matches.

It slightly increased mistaken merges.

This aligns with earlier plateau evidence.

14e showed that AQE k=2 was optimal on TTA-smoothed tracklet features.

AQE k=3 over-smoothed.

AQE k=4 worsened further.

14f confirmed k=2 as the discrete optimum.

14g showed more TTA views did not help.

14h showed robust pooling did not help.

14i showed track-quality pre-filtering did not help.

14k showed a fourth score stream barely helped.

All of that says the tracklet-level similarity graph is saturated.

AQE/rerank are neighbor-set smoothers.

They are powerful for noisy image-level query rows.

They are weak or harmful for pre-smoothed tracklet rows with sparse positives.

H4 also explains why 14t and 14u can both be true.

14t is not a false positive result.

It solved the problem it was shaped for.

14u is not a poor implementation.

It asked the same mechanism to solve a different graph problem.

H4 verdict: **VERY HIGH**.

This is the strongest single explanation.

Rescue potential: a different fusion geometry is needed.

The better geometry would consume feature streams as evidence.

It would not just average similarity matrices.

It would learn edge probabilities.

It would model camera-pair patterns.

It would model temporal compatibility.

It would model conflict constraints.

That points toward a GNN edge classifier or another learned association model.

---

### H5 — Production CityFlow Is Already a Saturated Multi-Stream Fusion

H5 is the plateau-level synthesis.

The production CityFlow stack is not a single weak model waiting for a second expert.

It is already a multi-stream association system.

The primary is CityFlow-trained CLIP-TransReID.

The tertiary stream is DINOv2 ViT-L/14.

FIC whitening calibrates camera-specific feature statistics.

AQE k=2 smooths the graph at the current optimum.

The graph threshold is tuned around 0.48.

Conflict-free connected components enforce association consistency.

Gallery expansion and intra-camera merge add carefully tuned post-processing.

Temporal overlap bonus contributes a small but real signal.

14e B1 established the 0.77936 anchor.

14f confirmed it.

14g confirmed that expanding DINOv2 TTA views did not move it.

14h confirmed that robust pooling did not move it.

14i confirmed that track-quality filters did not move it.

14j/14k confirmed that adding R50-IBN as a fourth stream was only marginal.

14u confirms that adding a 14t-derived fused stream is also only marginal.

The residual error is not stream-count limited.

It is feature-quality and association-geometry limited.

This matters because score fusion has diminishing returns.

Every additional stream pays a calibration cost.

Every additional stream can pull false neighbors together.

If the stream is correlated with existing streams, it contributes little.

If it is out-of-domain, it contributes noise.

If it is both partially correlated and partially out-of-domain, it does exactly what 14u did.

It adds a tiny amount of signal and a tiny amount of conflation.

Net effect lands below the noise band.

H5 verdict: **HIGH**.

Rescue potential: low for more score streams.

A fifth score stream of the same type should not be expected to work.

A VeRi-only expert should not be expected to work.

A CityFlow-trained model with genuinely different pretraining might have some chance.

But even that risks the 14k plateau.

The highest-upside rescue is learned association.

---

## 3. Quantitative Summary Table

| Variant | Stream Added | Best CityFlow MTMC IDF1 | Delta vs 14e B1 | ID Switch Signal | Verdict |
|---|---|---:|---:|---:|---|
| 14e B1 anchor | Existing CLIP-TransReID primary + DINOv2 tertiary | 0.77936 | 0.00000 | 154 | WIN baseline |
| 13d v2 | CLIP-SENet v6, VeRi-only, score-fused | 0.7679 at control; 0.6855 standalone | -0.0115 at best positive usage | harmful | FAIL |
| 13h | CLIP-SENet v6 fine-tuned on CityFlow | 0.7691 at `w_cs_ft=0.30` | -0.0012 vs production 0.7703 | not enough | MARGINAL / below production |
| 14k | R50-IBN-a CityFlow-trained fourth stream | 0.78079 | +0.00143 | mixed | MARGINAL, below 0.7810 WIN bar |
| **14u** | **14t-fused CLIP-SENet × TransReID-VeRi reranked stream** | **0.77995** | **+0.00059** | **160 vs 154 anchor** | **FAIL** |

Interpretation:

- 13d proves a strong VeRi-only CLIP-SENet expert does not transfer.
- 13h proves CityFlow adaptation helps standalone but not the ensemble.
- 14k proves adding another CityFlow-trained architecture can still plateau.
- 14u proves the successful 14t fused-rerank geometry does not break the CityFlow plateau.

The table closes the simple score-fusion branch.

---

## 4. What Would Actually Work for CityFlow MTMC?

The remaining viable levers must be different in kind.

They cannot be another low-weight score stream from the same checkpoint family.

They cannot be another VeRi-only expert.

They cannot be another rerank/AQE micro-sweep.

The project instructions list two remaining untried approaches.

One is a genuinely new architecture stream.

The other is graph-based learned association.

Pseudo-label self-training is a third possible but lower-confidence route.

### Option A — Genuinely New Architecture Stream

Concrete candidate:

- EVA-02-L/14 or a similarly strong non-identical ViT family.
- CityFlow-only fine-tune.
- Use a pretraining source that is not merely another CLIP-adjacent stream.

Why it might work:

- It could provide real feature diversity.
- It could capture residual errors the CLIP TransReID + DINOv2 pair misses.
- AIC22 winners often used larger multi-model ensembles.

Why it might fail:

- DINOv2 is already a strong self-supervised ViT stream.
- EVA/DINO-style features may be correlated with the existing tertiary stream.
- Training cost is high.
- Feature calibration into the existing graph remains non-trivial.

Estimated cost:

- Multi-day GPU training.
- New `09*` family notebook.
- Full verifier loop.
- Stage 4 score-fusion sweep.

Expected lift:

- Speculative.
- Perhaps +0.5pp to +1.5pp if the stream is genuinely complementary.
- Could land marginal like 14k.

Recommendation:

- Worth one carefully scoped allocation only if learned association stalls.
- Do not start with a VeRi-only checkpoint.
- Fine-tune on CityFlow from the beginning.

### Option B — Learned Association: GNN Edge Classifier

Concrete candidate:

- Train a graph neural network or edge classifier on CityFlow training-split tracklets.
- Nodes represent tracklets.
- Edge features include primary similarity, DINOv2 similarity, FIC-adjusted similarity, temporal compatibility, camera pair, track length, confidence summaries, and conflict indicators.
- Output is a probability that two tracklets are the same vehicle.
- Replace or augment the hand-thresholded conflict-free connected-components heuristic.

Why it might work:

- It attacks the association geometry directly.
- It can learn when visually similar vehicles should not merge.
- It can learn camera-pair-specific reliability.
- It can use negative evidence, not just positive similarity.
- It can break the score-fusion plateau because it is not another score stream.

Why it might fail:

- Cross-camera positive labels may be sparse.
- Scene-specific overfit is a real risk.
- Negative sampling must be careful.
- Evaluation leakage must be avoided.
- Engineering cost is higher than another sweep.

Estimated cost:

- 1-2 weeks engineering.
- 1 day GPU training once labels/features are prepared.
- Several CPU validation sweeps.

Expected lift:

- Highest of remaining options.
- Plausibly +1.5pp to +5pp if successful.
- Could fail cleanly while still teaching useful error structure.

Recommendation:

- Prioritize this for the next CityFlow MTMC push.
- It is the only remaining lever with both novelty and a plausible path beyond 0.77936.

### Option C — Pseudo-Label Self-Training

Concrete candidate:

- Bootstrap high-confidence cross-camera pseudo-labels from 14e B1 predictions.
- Fine-tune CLIP-TransReID with cross-camera consistency.
- Keep only high-margin positive pairs.
- Use hard negatives from visually similar but conflict-incompatible tracklets.

Why it might work:

- It tightens the existing primary.
- It directly adapts to the deployed association regime.
- It may improve borderline pairs.

Why it might fail:

- Pseudo-label noise can reinforce current conflations.
- The 154 ID-switch floor can become training bias.
- Expected lift is smaller than learned association.

Estimated cost:

- 2-3 days GPU.
- Careful pseudo-label filtering.
- Full verifier loop.

Expected lift:

- Small, perhaps +0.3pp to +0.7pp.

Recommendation:

- Secondary option.
- More attractive after a GNN label-preparation pipeline exists.

---

## 5. Should We Retrain, Abandon, or Pivot?

### The 14u Question

For CLIP-SENet × TransReID-VeRi fusion on CityFlow MTMC, the answer is **ABANDON**.

Reasons:

- H1: both 14t experts are out-of-domain for CityFlow.
- H2: fine-tuned CLIP-SENet already showed redundancy with the production pair.
- H3: CityFlow TransReID is already the production primary.
- H4: the 14t AQE/rerank mechanism is image-level query-gallery geometry, not tracklet-tracklet MTMC geometry.
- H5: the production stack is already a saturated score-fusion system.

The branch has had multiple chances.

13d failed.

13f/13h were marginal and below production.

13g did not rescue the branch.

14u failed under the de-risked Option C formulation.

That is enough.

Do not retrain CLIP-SENet again for this purpose.

Do not add a fifth score stream of the same flavor.

Do not port more VeRi-only experts into CityFlow MTMC.

### CityFlow MTMC Strategy

The broader strategy should **PIVOT**.

The next CityFlow MTMC push should not be another score-level ensemble sweep.

The strongest recommendation is a GNN edge classifier.

The second recommendation is a genuinely new CityFlow-trained architecture stream.

Pseudo-label self-training is third.

Association learning has the best chance to change the failure mode.

Feature-stream additions have repeatedly failed to change the failure mode.

### 14t Itself

14t should be **KEPT**.

14t should be verified.

14t should be wired as a standalone single-camera VeRi tool.

14t should not be treated as a failed idea just because 14u failed.

The correct conclusion is narrower.

14t is a strong single-camera ReID result.

14t is not a CityFlow MTMC ingredient.

This is a valuable distinction for the paper.

---

## 6. Implications for the Paper

The 14t WIN and 14u FAIL form a useful contrast.

The contrast is publishable.

It says that two strong single-camera VeRi-776 experts can fuse to SOTA-equivalent VeRi mAP.

It also says the same fusion adds essentially zero cross-camera signal on CityFlow MTMC.

That supports the paper's central thesis.

The thesis is that MTMC is bottlenecked by feature quality at the tracklet-level cross-camera granularity.

That is distinct from per-image ReID benchmark quality.

Single-camera ReID mAP is useful.

It is not sufficient.

It is weakly predictive of MTMC IDF1 lift once the deployed stack is already strong.

The paper can use 14t as a positive result.

The paper can use 14u as a negative transfer result.

Together they show rigor.

They avoid overclaiming.

They also motivate learned association.

The six-axis plateau is the ablation backbone:

1. Stage-4 tuning.
2. Tertiary view expansion.
3. Tracklet aggregation.
4. Track-quality filtering.
5. Four-way score fusion.
6. VeRi-fusion port.

14u closes the sixth axis cleanly.

This helps frame the system honestly.

The system is not failing because nobody tried obvious fusion variants.

The system is plateaued because obvious fusion variants are exhausted.

---

## 7. Final Recommendations

1. Close 14u as DEAD END.
2. Keep 14t as a VeRi-776 single-camera WIN.
3. Build the 14aa verifier for 14t.
4. Wire 14t as a standalone eval/API surface only after verifier passes.
5. Do not wire 14t into CityFlow MTMC.
6. Do not retrain CLIP-SENet again for CityFlow score fusion.
7. Do not add a fifth score stream from the same model family.
8. Prioritize a GNN edge classifier for the next CityFlow MTMC push.
9. Consider a genuinely new CityFlow-trained architecture stream only as the second bet.
10. Use the 14t/14u contrast in the paper as evidence that single-camera ReID strength does not guarantee MTMC lift.

---

## Appendix A — Evidence Ledger

| Evidence | Observation | Interpretation |
|---|---|---|
| 14t | 0.9330 mAP / 0.9845 R1 on VeRi-776 | Same-domain image-level score fusion works |
| 13d | CLIP-SENet VeRi-only hurts CityFlow monotonically | Domain gap is severe |
| 13h | CityFlow-fine-tuned CLIP-SENet standalone improves to 0.7099 but fusion peaks below production | Adaptation helps but complementarity remains weak |
| 14e | AQE k=2 and TTA features unlock 0.77936 | Tracklet features are pre-smoothed; over-smoothing hurts |
| 14f | k=2 plateau confirmed | AQE/rerank tuning axis exhausted |
| 14g | More DINOv2 views do not help | TTA view coverage exhausted |
| 14h | Robust pooling does not help | Tracklet aggregation exhausted |
| 14i | Quality filters barely move IDF1 | Low-quality tracklets are not the main bottleneck |
| 14k | R50-IBN fourth stream reaches only 0.78079 | Score-stream count is not the bottleneck |
| 14u | 14t-fused stream reaches only 0.77995 and raises ID switches | VeRi fusion geometry does not transfer |

---

## Appendix B — Mechanism Comparison

| Dimension | VeRi-776 14t | CityFlowV2 14u |
|---|---|---|
| Unit | single image | tracklet-pooled feature |
| Query count | ~1,678 | ~929 tracklets total |
| Gallery count | ~11,579 images | same 929-node graph |
| Positives per query | often multiple gallery images | often zero or one cross-camera match |
| Noise type | single-frame viewpoint/occlusion | identity-level cross-camera ambiguity |
| AQE role | repairs noisy query neighborhoods | risks over-smoothing already pooled tracklets |
| Rerank role | strengthens reciprocal true image neighbors | can recruit visually similar false vehicles |
| Fusion role | combines complementary model rankings | adds a low-weight noisy score stream |
| Successful? | yes | no |

---

## Appendix C — Closed Branch Rules

Do not run these without a new planner-approved hypothesis:

- CLIP-SENet v6 VeRi-only into CityFlow score fusion.
- CLIP-SENet v6 CityFlow fine-tune into the current score-fusion stack.
- 14t fused VeRi stream as a CityFlow replacement primary.
- 14t fused VeRi stream as a low-weight fourth score stream.
- More AQE k=3/k=4 probes on TTA-smoothed CityFlow tracklet features.
- More rerank-on-fused-similarity probes over the same 929 tracklet graph.
- Any VeRi-only expert port that lacks CityFlow fine-tuning and a non-score-fusion mechanism.

Acceptable future work must differ in kind:

- Learned association.
- New CityFlow-trained architecture family.
- Pseudo-label self-training with strict noise controls.

---

## Appendix D — Paper-Ready Claim Draft

Potential wording:

> On VeRi-776, score-level fusion of two independently trained experts, CLIP-SENet v6 and TransReID 09v, improved mAP to 93.30% and R1 to 98.45%. However, porting the same fused similarity geometry to CityFlowV2 MTMC as a fourth association stream produced only 0.77995 IDF1 versus the 0.77936 anchor, well inside the noise band and with increased ID switches. This contrast indicates that single-camera ReID benchmark complementarity does not necessarily translate to tracklet-level cross-camera association gains.

This claim is accurate only if paired with the caveat that 14t remains valuable as a single-camera ReID result.

Do not phrase 14t as a failed fusion result.

Do not phrase 14u as disproving score fusion in all settings.

It disproves this transfer path for this production CityFlow stack.

---

## Appendix E — Detailed Hypothesis Evidence Grid

| Row | Evidence | Supports | Refutes | Weight |
|---:|---|---|---|---|
| 1 | 14t reaches 0.9330 mAP on VeRi-776 | Same-domain complementarity | CityFlow transfer | strong |
| 2 | 14t reaches 0.9845 R1 on VeRi-776 | Same-domain complementarity | CityFlow transfer | medium |
| 3 | CLIP-SENet v6 alone is strong on VeRi-776 | H1 setup | none | strong |
| 4 | TransReID 09v v17 is strong on VeRi-776 | H1 setup | none | strong |
| 5 | Both 14t parents are VeRi-trained | H1 | none | strong |
| 6 | 13d CLIP-SENet CityFlow standalone is weak | H1 | none | strong |
| 7 | 13d positive CLIP-SENet weights degrade | H1, H5 | none | strong |
| 8 | 13h fine-tuning improves standalone | partial rescue of H1 | pure H1-only explanation | medium |
| 9 | 13h fusion remains below production | H2 | pure H1-only explanation | strong |
| 10 | Production primary is already CityFlow TransReID | H3 N/A | retrain-09v rescue | strong |
| 11 | 14e AQE k=2 beats k=3 on TTA features | H4 | rerank-transfer optimism | strong |
| 12 | 14f confirms k=2 plateau | H4, H5 | rerank-transfer optimism | strong |
| 13 | 14g more DINOv2 TTA views are neutral | H5 | view coverage hypothesis | strong |
| 14 | 14h robust pooling is neutral/worse | H4, H5 | aggregation hypothesis | strong |
| 15 | 14i filters reduce ID switches without IDF1 lift | H5 | ID-switch-count-as-proxy | strong |
| 16 | 14k fourth stream is marginal | H5 | stream-count hypothesis | strong |
| 17 | 14u best is only +0.00059 | H1-H5 | transfer hypothesis | decisive |
| 18 | 14u best increases ID switches | H4 | clean-signal hypothesis | decisive |
| 19 | 14u optimum is low weight | H1 | high-confidence transfer | strong |
| 20 | Higher 14u weights regress | H1, H5 | strong additive stream | strong |

The important pattern is not one isolated negative run.

The important pattern is convergence.

Multiple experiments point to the same explanation.

Domain mismatch exists.

Fine-tuned redundancy exists.

Tracklet geometry mismatch exists.

Score-stream saturation exists.

14u sits at the intersection of all four.

That is why a micro-sweep follow-up is low value.

The failure has already been overdetermined.

---

## Appendix F — Decision Log for Future Agents

Future agents should treat the following decisions as closed unless the user explicitly reopens them.

Decision 1:

- Question: Should CLIP-SENet v6 be added to CityFlow MTMC again?
- Answer: No.
- Evidence: 13d, 13h, 14u.
- Reopen condition: a new CLIP-SENet checkpoint trained with a substantially different pretraining objective and verified standalone CityFlow strength above the production primary.

Decision 2:

- Question: Should VeRi-only TransReID 09v be added to CityFlow MTMC?
- Answer: No.
- Evidence: H3 and 14u.
- Reopen condition: none for the same checkpoint; a CityFlow-trained variant is already the production primary.

Decision 3:

- Question: Should 14t fused similarity replace the production primary?
- Answer: No.
- Evidence: 14u algebraic reduction; replacing the CityFlow primary confounds the test and weakens the recipe.
- Reopen condition: none without a new CityFlow-trained fused model.

Decision 4:

- Question: Should 14t fused similarity be added as a low-weight fourth stream?
- Answer: No.
- Evidence: 14u Option C did exactly this and failed.
- Reopen condition: none for the same feature files and same Stage 4 association stack.

Decision 5:

- Question: Should rerank-on-fused-similarity be retried on the same 929-tracklet CityFlow graph?
- Answer: No.
- Evidence: 14u and the 14e/14f AQE plateau.
- Reopen condition: a materially different graph, such as learned edge features or a larger set of high-quality tracklets.

Decision 6:

- Question: Should 14t be discarded?
- Answer: No.
- Evidence: 14t is a real VeRi-776 WIN.
- Reopen condition: only if the 14aa verifier fails to reproduce and checkpoint drift cannot be explained.

Decision 7:

- Question: Should the paper include 14u?
- Answer: Yes.
- Evidence: it cleanly supports the single-cam-vs-MTMC distinction.
- Reopen condition: only if space constraints force removal of negative results.

Decision 8:

- Question: What is the next CityFlow MTMC move?
- Answer: learned association first.
- Evidence: score-fusion axes are exhausted.
- Reopen condition: user explicitly prioritizes feature training over association research.

---

## Appendix G — Implementation Warnings for Coder Agents

This analysis is not an implementation plan.

Do not create notebooks from this file.

Do not patch Stage 2 from this file.

Do not change `src/stage2_features/reid_model.py` because of this file.

Do not change `scripts/run_pipeline.py` because of this file.

Do not add `veri776_14t_fusion` to default model selection.

Do not promote 14u settings to `configs/datasets/cityflowv2.yaml`.

Do not create a new 14u sweep unless a planner spec supersedes this analysis.

The only actionable implementation spawned by this analysis is indirect:

- Verify 14t as a standalone VeRi result.
- Wire 14t as a standalone single-camera eval script.
- Pivot CityFlow MTMC toward learned association.

If a future Coder sees `veri776_14t_fusion` in the registry, it must read it as research metadata.

It is not a production MTMC model.

It is not a Stage 2 model override.

It is not a CityFlow model recommendation.

---

## Appendix H — Compact Verdict

14t succeeded because two in-domain image-level ReID experts corrected each other's query-gallery ranking errors.

14u failed because CityFlow MTMC does not present the same problem.

CityFlow uses pre-smoothed tracklet features.

CityFlow has sparse cross-camera positives.

CityFlow's production stack already fuses strong streams.

CityFlow's residual errors are not fixed by another low-weight score matrix.

The right lesson is not that fusion is bad.

The right lesson is that benchmark-level fusion and MTMC association are different tasks.

Keep the VeRi win.

Close the CityFlow port.

Move the MTMC work to learned association.

 
---

End of analysis.