# ATHAR v2

ATHAR is a multi-camera forensic tracking platform for **people and vehicles**:
investigators ingest recorded CCTV footage, the pipeline detects, tracks, and
re-identifies entities across a camera network, and the app provides
reference-video search, cross-camera trajectory reconstruction, and case
reporting — built for enterprise/intelligence-grade deployments.

This branch (`athar-v2`) is a **ground-up rebuild**. The validated v1 system
(CityFlowV2 MTMC IDF1 0.779, WILDTRACK IDF1 0.946, VeRi-776 fusion mAP 93.3)
lives on branch `seif_final`; its algorithm kernels are ported here verbatim
behind new contracts, while all orchestration, state management, and the app
layer are redesigned.

**The single source of truth for plan, progress, and decisions is
[ROADMAP.md](ROADMAP.md).** Read it first.

## Layout

```
athar/            Installable Python package (pipeline, contracts, serving, api)
  core/           Identifiers, typed domain models, time base, geometry
  contracts/      Run manifest, resolved-config provenance, artifact store
  components/     Pluggable component protocols + registry (detector/tracker/…)
  profiles/       Multi-class run profiles (which components fill which slots)
  pipeline/       Stage DAG, runner, typed progress events
  serving/        Model lifecycle registry, loader, cache, device management
  search/         Case / Gallery / Probe / Target domain models, query engine
  evaluation/     Benchmark harness and parity gates
  jobs/           Durable job service (pipeline runs, eval, adaptation)
  api/            FastAPI application (thin routers over services)
  cli/            `athar` command-line entry point
configs/          Profiles, dataset descriptors, model registry (authoring format)
tests/            Unit + contract + parity tests
```

## Accessing legacy (v1) code

All v1 code remains in git on `seif_final`. To view or restore a file for
porting:

```bash
git show seif_final:src/stage4_association/similarity.py   # view
git restore --source=seif_final -- src/stage4_association/  # restore to disk
```

Untracked v1 leftovers (sweep scripts, kaggle notebooks) were preserved under
`_legacy_archive/` (ignored). Datasets and checkpoints stay in `data/` and
`models/` (ignored) and are shared by both branches.

## Development

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -e .[dev]
pytest
```
