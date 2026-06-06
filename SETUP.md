# Setup

First-time setup after cloning the repo and configuring Kaggle credentials:

```bash
pip install -r requirements.txt
python scripts/download_weights.py          # pick a model set, or choose "all"
python scripts/verify_assets.py
```

`download_weights.py` is interactive: it lists the available model sets and lets
you fetch just what you need (or everything). To grab everything plus the public
eval datasets in one shot, use `python scripts/download_assets.py --all`.

## Prerequisites

- Python 3.10 or newer. The local project venv is recommended.
- A (free) Kaggle account and API token at `~/.kaggle/kaggle.json`. See Kaggle's
  API docs: <https://www.kaggle.com/docs/api>. The weights dataset is public, but
  the Kaggle API still requires an authenticated token.
- Disk space: the full weight bundle is ~2.3 GB; individual sets are smaller (see
  the table). Keeping CityFlowV2 locally as well needs roughly 5 GB or more.

## Model Weights

All pipeline and paper checkpoints live in ONE public Kaggle dataset,
**`mrkdagods/mtmc-veri776-pipeline-weights`** (license CC BY 4.0). Each file is
SHA-256 pinned in `configs/weights_manifest.yaml`; the downloader verifies every
file after fetching.

```bash
python scripts/download_weights.py                 # interactive set picker
python scripts/download_weights.py --list          # list sets + files, no download
python scripts/download_weights.py --set all       # download everything (~2.3 GB)
python scripts/download_weights.py --set veri      # just the VeRi-776 paper streams
python scripts/download_weights.py --set veri --set person-mtmc   # combine sets
python scripts/download_weights.py --set all --dry-run            # preview only
python scripts/download_weights.py --set veri --force             # re-download
```

| Model set | Size | What it runs |
| --- | ---: | --- |
| `vehicle-mtmc-14e` | ~1.5 GB | Vehicle MTMC 14e B1 production (CityFlowV2): YOLO26m + OSNet tracker + TransReID-CLIP primary + DINOv2 tertiary |
| `vehicle-mtmc-14k` | ~1.6 GB | Vehicle MTMC 14k v1 K7 research: 14e set + FastReID R50-IBN quaternary |
| `person-mtmc` | ~47 MB | Person MTMC 12b (WILDTRACK): MVDeTr ground-plane detector |
| `veri` | ~685 MB | VeRi-776 paper two-stream fusion (93.32% mAP): TransReID-CLIP + CLIP-SENet v6 |
| `all` | ~2.3 GB | Everything above (8 files) |

The two `veri` checkpoints are the exact SHA-256-pinned files reported in the
paper (`tab:repro`): Stream 1 `8d32334a...`, Stream 2 (CLIP-SENet v6)
`d24bd3cd...`. Evaluating them reproduces the 89.97% standalone and 93.32% fusion
mAP results.

## Datasets

`scripts/download_assets.py` handles the public datasets and also delegates the
model-weight download to `download_weights.py`:

```bash
python scripts/download_assets.py --all           # all weights + public datasets
python scripts/download_assets.py --datasets      # only the public datasets
python scripts/download_assets.py --all --dry-run # preview
```

| Asset | Destination | Source |
| --- | --- | --- |
| VeRi-776 eval dataset | `data/raw/veri776/` | Kaggle dataset `abhyudaya12/veri-vehicle-re-identification-dataset` |
| CityFlowV2 dataset | `data/raw/cityflowv2/` | Manual - see below |

CityFlowV2 is not available as a complete public Kaggle dataset. Download AI City
Challenge 2022 Track 1 manually from the official site, then place it under
`data/raw/cityflowv2/`: <https://www.aicitychallenge.org/2022-data-and-evaluation/>.

## Verify

```bash
python scripts/verify_assets.py
```

`verify_assets.py` checks the local checkpoints (size + MD5) and the optional
VeRi-776 data / manual CityFlowV2 placement. The downloaders skip files that
already match their pinned checksum; if one file fails, the rest still run.
