# What Actually Worked

Positive deltas only. Use this as a sanity check before proposing reinventions. See [dead-ends.md](dead-ends.md) for what failed and [findings.md](findings.md) for full analysis.

## Association / Stage 4
- **Conflict-free CC**: +0.21pp
- **Intra-merge**: +0.28pp
- **Temporal overlap bonus**: +0.9pp
- **FIC whitening**: +1 to +2pp
- **Power normalization**: +0.5pp
- **PCA 384D**: kept (no degradation, dimension reduction win)
- **AQE K=3**: production default for non-TTA features
- **AQE K=2**: discrete optimum for TTA features (14e B1 unlock, +0.77pp on TTA features vs K=3)

## Tracking / Stage 1
- **min_hits=2**: +0.2pp

## Person Pipeline
- **Kalman tuning**: +1.9pp IDF1

## TTA Family (14e plateau)
- **14e B1 v1**: NEW HEADLINE 0.77936 — multi-crop TTA Stage-2 features (14c v2) + Stage-4 fusion `w_tertiary=0.525, similarity_threshold=0.48, aqe_k=2, fic_regularisation=0.5`. +0.91pp vs prior baseline 0.7703.
- **14f confirmation** (NEUTRAL but valuable): 14e B1 = 0.77936 is a confirmed reproducible plateau (A20 drift check reproduced 0.77936 exact with id_switches=154 exact; 8 Block A configs at `aqe_k=2, w_t=0.525` tied at 0.77936). TTA × Stage-4-tuning family is EXHAUSTED at 0.77936 (not a dead end — confirmed-win plateau). k=2 is the discrete AQE optimum for TTA features.
