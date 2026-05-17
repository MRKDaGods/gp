# Launching the MTMC Tracker App

## One-time setup (teammates / fresh clone)

```powershell
# 1. Clone and install Python deps
git clone <repo>
cd gp
.\.venv\Scripts\activate          # or create: python -m venv .venv
pip install -r requirements.txt

# 2. Install Node deps for frontend
cd frontend
npm install
cd ..

# 3. Authenticate with Kaggle (one-time)
# Put your kaggle.json at ~/.kaggle/kaggle.json
# Get it from https://www.kaggle.com/settings → Create New Token

# 4. Download all model checkpoints + datasets (~3-5 GB)
python scripts/download_assets.py --all

# 5. Verify everything landed
python scripts/verify_assets.py
```

## Launch backend + frontend (two terminals)

**Terminal 1 — Backend (FastAPI on port 8000):**
```powershell
.\.venv\Scripts\activate
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 — Frontend (Next.js on port 3000):**
```powershell
cd frontend
npm run dev
```

## Use the app

Open browser → http://127.0.0.1:3000

| Page | What you can do |
|---|---|
| `/` (hub) | Landing page with links + dataset switcher in header |
| `/reid` | Upload query + gallery images, pick any ReID model, see ranked matches with similarity scores. Supports rerank+AQE. |
| `/fusion` | Pick 2+ ReID models, set weights (sum-to-1), upload images, see fused ranking compared to each model alone |
| `/eval` | Submit a standalone eval, watch status, view results JSON |

Available models for ReID and fusion:
- `veri776_09v_v17_transreid` — TransReID ViT-B/16 trained on VeRi-776 (R1=98.33, mAP=89.97)
- `veri776_clipsenet_v6` — CLIP-SENet trained on VeRi-776 (cosine mAP=82.34, rerank+AQE mAP=91.54)
- `cityflow_transreid` — TransReID ViT-B/16 trained on CityFlowV2 (mAP=81.53)

Available evals:
- `veri776_transreid`
- `veri776_clipsenet`
- `cityflow_transreid`
- `veri776_14t_fusion` (TransReID × CLIP-SENet fusion, mAP=93.30)

## Optional verification

```powershell
# Run full local test suite
pytest tests/ -v

# Run full Phase 2 E2E test (spins up backend + frontend, exercises every endpoint, tears down)
python scripts/test_phase2_e2e.py
```

## CPU vs GPU notes

- All ReID single-cam + fusion endpoints work on CPU (~5s cold start, ~1s warm per query)
- For GPU acceleration, ensure `torch.cuda.is_available()`; the loader auto-detects
- **MTMC pipeline stages 0/1/2 should run on Kaggle**, not locally — see `.github/copilot-instructions.md`
- Stages 3-5 are CPU-friendly

## Troubleshooting

- `503 reid_dependency_missing` → `pip install -r requirements.txt` (torch + timm needed)
- `503 checkpoint_missing` → `python scripts/download_assets.py --all`
- `503 dataset_missing` for evals → check `data/raw/veri776/` exists
- CityFlowV2 dataset → manual download from [AIC22 official](https://www.aicitychallenge.org/), see SETUP.md
- Port 8000 / 3000 already in use → change `--port` in commands above

## More documentation

- [README.md](README.md) — project overview
- [SETUP.md](SETUP.md) — detailed setup
- [docs/findings.md](docs/findings.md) — research log
- [docs/subagent-specs/phase2-app-integration.md](docs/subagent-specs/phase2-app-integration.md) — Phase 2 design