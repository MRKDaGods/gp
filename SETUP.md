# Setup

First-time setup is three commands after cloning the repo and configuring Kaggle credentials:

```bash
pip install -r requirements.txt
python scripts/download_assets.py --all
python scripts/verify_assets.py
```

## Prerequisites

- Python 3.10 or newer. The local project venv is recommended.
- A Kaggle account and API token at `~/.kaggle/kaggle.json`. See Kaggle's API docs: <https://www.kaggle.com/docs/api>.
- Enough disk space for checkpoints and datasets. ReID checkpoints plus VeRi-776 need roughly 3 GB; keeping CityFlowV2 locally as well needs roughly 5 GB or more depending on the extracted layout.

## What The Downloader Fetches

`scripts/download_assets.py --all` pulls the public assets that can be automated:

| Asset | Destination | Source |
| --- | --- | --- |
| CLIP-SENet v6 VeRi-776 checkpoint | `models/reid/clipsenet_v6_veri776_best.pth` | Kaggle kernel `yahiaakhalafallah/13-clip-senet-train`, `checkpoints/best.pth` |
| 09v TransReID VeRi-776 checkpoint | `models/reid/vehicle_transreid_vit_base_veri776.pth` | Kaggle dataset `mrkdagods/mtmc-weights`, `reid/vehicle_transreid_vit_base_veri776.pth` |
| CityFlowV2 TransReID checkpoint | `models/reid/transreid_cityflowv2_best.pth` | Kaggle dataset `gumfreddy/mtmc-weights`, `reid/transreid_cityflowv2_best.pth` |
| MVDeTr WILDTRACK checkpoint | `models/person_detection/MultiviewDetector.pth` | Kaggle dataset `gumfreddy/12a-wildtrack-mvdetr-checkpoint`, `MultiviewDetector.pth` |

The verified MVDeTr checkpoint is 49,745,811 bytes with MD5 `18658027791f44357f07db6b9406b120`.
| VeRi-776 eval dataset | `data/raw/veri776/` | Kaggle dataset `abhyudaya12/veri-vehicle-re-identification-dataset` |

CityFlowV2 is not available as a complete public Kaggle dataset. Download AI City Challenge 2022 Track 1 manually from the official site, then place it under `data/raw/cityflowv2/`: <https://www.aicitychallenge.org/2022-data-and-evaluation/>.

## Useful Commands

```bash
# Show actions without downloading.
python scripts/download_assets.py --all --dry-run

# Fetch only model checkpoints.
python scripts/download_assets.py --reid-only
python scripts/download_assets.py --detection-only

# Re-download and replace existing files.
python scripts/download_assets.py --detection-only --force

# Verify required models, optional VeRi-776 data, and manual CityFlowV2 placement.
python scripts/verify_assets.py
```

The downloader skips files that already match the known size or MD5. If one asset fails, the remaining selected assets still run and the script prints a Markdown summary at the end.