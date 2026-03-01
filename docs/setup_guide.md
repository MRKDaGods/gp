# Setup Guide

## Prerequisites
- Python 3.10+
- CUDA-compatible GPU (4-6 GB VRAM) for local inference
- Kaggle account with GPU quota for training
- Git

## Local Environment Setup

### 1. Clone and Install
```bash
git clone <repo-url>
cd gp
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

pip install -e ".[dev]"
```

### 2. Verify Installation
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import ultralytics; print(f'Ultralytics {ultralytics.__version__}')"
python -c "from boxmot import BoTSORT; print('BoxMOT OK')"
python -c "import torchreid; print('TorchReID OK')"
python -c "import faiss; print(f'FAISS {faiss.__version__}')"
```

### 3. Download Models
```bash
# Downloads YOLO26m (auto-download by ultralytics)
# BoxMOT ReID weights (auto-download on first use)
python scripts/download_models.py
```

### 4. Prepare Datasets
Download datasets to `data/raw/` and run:
```bash
python scripts/prepare_dataset.py --dataset market1501 --root data/raw/market1501
python scripts/prepare_dataset.py --dataset veri776 --root data/raw/veri776
```

### 5. Download Trained ReID Weights from Kaggle
After running Kaggle training notebooks, download weights to:
```
models/
├── reid/
│   ├── osnet_x1_0_market1501.pth.tar
│   ├── resnet50_ibn_a_market1501.pth.tar
│   ├── osnet_x1_0_veri776.pth.tar
│   └── resnet50_ibn_a_veri776.pth.tar
```

## Running the Pipeline

### Full Pipeline
```bash
python scripts/run_pipeline.py --config configs/default.yaml --output outputs/run_001
```

### Single Stage
```bash
python scripts/run_stage.py --stage 1 --config configs/default.yaml --input outputs/run_001/stage0 --output outputs/run_001/stage1
```

### Smoke Test (tiny data subset, <30 seconds)
```bash
make smoke-test
# or
python scripts/run_pipeline.py --config configs/default.yaml --smoke-test
```

### With Dataset Config
```bash
python scripts/run_pipeline.py --config configs/default.yaml --dataset-config configs/datasets/aic2023.yaml
```

### With Overrides
```bash
python scripts/run_pipeline.py --config configs/default.yaml -o stage1.detector.confidence_threshold=0.3 -o stage4.association.reranking.enabled=false
```

## Running Tests
```bash
# All tests
pytest tests/ -v

# Specific stage
pytest tests/test_stage1/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## Makefile Commands
```bash
make install       # pip install -e .[dev]
make test          # pytest tests/ -v
make lint          # ruff check + mypy
make format        # ruff format
make smoke-test    # Quick pipeline test
make run-pipeline  # Full pipeline run
```

## Kaggle Setup
See `docs/kaggle_training_guide.md` for how to run training notebooks on Kaggle.

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in config: `stage2.reid.batch_size`
- Use OSNet (512-dim, lighter) instead of ResNet50-IBN (2048-dim)
- Reduce YOLO input size: `stage1.detector.img_size`

### BoxMOT Import Error
- Ensure `boxmot` is installed: `pip install boxmot`
- For tracker weight downloads, ensure internet connection on first run

### FAISS Issues
- Use `faiss-cpu` (not `faiss-gpu`) for compatibility
- If installation fails: `conda install -c conda-forge faiss-cpu`

### torchreid Not Found
- Install from source: `pip install git+https://github.com/KaiyangZhou/deep-person-reid.git`
