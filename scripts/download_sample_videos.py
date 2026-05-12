"""
Download sample videos from AI City Challenge dataset for MTMC demo
Compatible with trained models from Kaggle: mrkdagods/mtmc-weights
"""

import os
from pathlib import Path
import urllib.request
import json

# Sample video clips from AI City Challenge 2023 (Track 1: MTMC)
SAMPLE_VIDEOS = {
    "cityflow_sample": {
        "name": "CityFlow Sample (3 cameras)",
        "description": "Short clips from S01 cameras - compatible with trained models",
        "cameras": ["c001", "c002", "c003"],
        "duration": "30 seconds",
        "size": "~50MB",
        "download_url": "https://github.com/cfzd/AIC23_MTMC_sample/releases/download/v1.0/sample_videos.zip",
        "local_path": "data/samples/cityflow"
    },
    "veri776_sample": {
        "name": "VeRi-776 Sample",
        "description": "Vehicle ReID test images from VeRi-776 dataset",
        "cameras": ["multiple"],
        "format": "images",
        "size": "~20MB",
        "kaggle_dataset": "veri-776-vehicle-reid/veri-776",
        "local_path": "data/samples/veri776"
    }
}

def download_sample_videos():
    """Download sample videos for demo"""
    print("="*60)
    print("MTMC Tracker - Sample Video Downloader")
    print("="*60)
    print()

    samples_dir = Path("data/samples")
    samples_dir.mkdir(parents=True, exist_ok=True)

    print("Available sample datasets:")
    print()
    for key, info in SAMPLE_VIDEOS.items():
        print(f"  [{key}]")
        print(f"  Name: {info['name']}")
        print(f"  Description: {info['description']}")
        print(f"  Size: {info['size']}")
        print()

    print("="*60)
    print()
    print("IMPORTANT: These samples are compatible with models from:")
    print("https://www.kaggle.com/datasets/mrkdagods/mtmc-weights")
    print()
    print("Models trained on:")
    print("  - VeRi-776 (Vehicle ReID)")
    print("  - Market-1501 (Person ReID)")
    print("  - AI City Challenge 2023 (MTMC evaluation)")
    print()
    print("="*60)
    print()

    # Create sample metadata file
    metadata_file = samples_dir / "samples_info.json"
    with open(metadata_file, 'w') as f:
        json.dump(SAMPLE_VIDEOS, f, indent=2)

    print(f"[OK] Sample metadata saved to: {metadata_file}")
    print()
    print("DOWNLOAD OPTIONS:")
    print()
    print("Option 1: Manual Download (Recommended)")
    print("-" * 40)
    print("1. Visit AI City Challenge:")
    print("   https://www.aicitychallenge.org/2023-data-and-evaluation/")
    print()
    print("2. Download 'Track 1 - MTMC' dataset")
    print("   Extract to: data/samples/cityflow/")
    print()
    print("3. Or download VeRi-776 from Kaggle:")
    print("   kaggle datasets download -d veri-776-vehicle-reid/veri-776")
    print("   Extract to: data/samples/veri776/")
    print()

    print("Option 2: Use Kaggle API")
    print("-" * 40)
    print("# Install Kaggle CLI")
    print("pip install kaggle")
    print()
    print("# Configure credentials (kaggle.com/settings/account)")
    print("# Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<user>\\.kaggle\\ (Windows)")
    print()
    print("# Download datasets")
    print("kaggle competitions download -c aicity23-track1-mtmc")
    print("kaggle datasets download -d veri-776-vehicle-reid/veri-776")
    print()

    print("Option 3: Quick Start with Mock Data")
    print("-" * 40)
    print("The system includes mock data for UI testing.")
    print("Upload any video - detection/tracking will be simulated.")
    print("For real results, use dataset videos.")
    print()

    print("="*60)
    print()

    # Create sample directory structure
    for key, info in SAMPLE_VIDEOS.items():
        sample_path = Path(info['local_path'])
        sample_path.mkdir(parents=True, exist_ok=True)

        readme_path = sample_path / "README.txt"
        with open(readme_path, 'w') as f:
            f.write(f"{info['name']}\n")
            f.write("="*60 + "\n\n")
            f.write(f"Description: {info['description']}\n")
            f.write(f"Size: {info['size']}\n\n")
            f.write("Download Instructions:\n")
            f.write("-"*60 + "\n")
            if 'download_url' in info:
                f.write(f"URL: {info['download_url']}\n")
            if 'kaggle_dataset' in info:
                f.write(f"Kaggle: kaggle datasets download -d {info['kaggle_dataset']}\n")
            f.write("\nExtract contents to this directory.\n")

        print(f"[OK] Created directory: {sample_path}")

    print()
    print("Next Steps:")
    print("-----------")
    print("1. Download sample videos using one of the options above")
    print("2. Place videos in data/samples/cityflow/ directory")
    print("3. Upload via the web UI: http://localhost:3000")
    print("4. System will automatically process with trained models")
    print()
    print("="*60)

def create_dataset_links():
    """Create links file for easy access to datasets"""
    links_file = Path("data/DATASET_LINKS.md")
    links_file.parent.mkdir(parents=True, exist_ok=True)

    content = """# MTMC Tracker - Dataset Links

## Compatible Video Sources

### 1. AI City Challenge 2023 - Track 1: MTMC
**Primary dataset for vehicle tracking**

- **Website**: https://www.aicitychallenge.org/2023-data-and-evaluation/
- **Description**: Multi-camera vehicle tracking dataset
- **Cameras**: 40+ cameras across multiple scenarios
- **Duration**: Several hours of footage
- **Format**: MP4 video files
- **Compatible Models**: YOLOv8, OSNet ReID

**Download**:
```bash
# Requires registration at aicitychallenge.org
# Download Track 1 data
# Extract to: data/samples/cityflow/
```

### 2. VeRi-776 Dataset
**Vehicle ReID evaluation dataset**

- **Kaggle**: https://www.kaggle.com/datasets/veri-776-vehicle-reid/veri-776
- **Description**: 776 vehicles, 50,000+ images from 20 cameras
- **Format**: JPEG images
- **Compatible Models**: OSNet, ResNet50-IBN, TransReID

**Download**:
```bash
kaggle datasets download -d veri-776-vehicle-reid/veri-776
unzip veri-776.zip -d data/samples/veri776/
```

### 3. CityFlow Benchmark
**Original MTMC benchmark**

- **Website**: https://www.aicitychallenge.org/2020-data-and-evaluation/
- **Description**: 215 minutes of video, 3,290 vehicles
- **Scenarios**: S01 (urban), S02 (highway), S03-S05 (various)
- **Compatible Models**: All trained models

**Download**:
```bash
# From AI City Challenge 2020
# Extract to: data/samples/cityflow/
```

## Quick Start Sample Videos

For immediate testing, use these short clips:

### Option A: CityFlow Samples (Recommended)
```
Camera S01_c001: Highway overpass (30 sec)
Camera S01_c002: Urban intersection (30 sec)
Camera S01_c003: Side street (30 sec)
```

### Option B: Your Own Videos
Requirements:
- Resolution: 1920x1080 or similar
- FPS: 25-30 fps
- Format: MP4, AVI, MOV
- Duration: 30 seconds minimum
- Scene: Stationary camera, vehicle traffic

## Models Training Data

Models from `mrkdagods/mtmc-weights` were trained on:

1. **Detection (YOLOv8x)**
   - COCO dataset (vehicle classes)
   - AI City Challenge 2023 fine-tuning

2. **ReID (OSNet/ResNet50-IBN)**
   - Market-1501 (person)
   - VeRi-776 (vehicle)
   - AI City Challenge vehicle crops

3. **Tracking (BoxMOT)**
   - MOT17 benchmark
   - AI City Challenge sequences

## File Structure

After downloading, your structure should look like:

```
data/
├── samples/
│   ├── cityflow/
│   │   ├── S01_c001.mp4
│   │   ├── S01_c002.mp4
│   │   └── S01_c003.mp4
│   ├── veri776/
│   │   ├── image_train/
│   │   ├── image_test/
│   │   └── image_query/
│   └── samples_info.json
└── DATASET_LINKS.md (this file)
```

## Using Your Videos

1. **Upload** via web UI: http://localhost:3000
2. **Place manually** in `uploads/` directory
3. **Configure** camera IDs in `configs/default.yaml`

## Notes

- Videos from AI City Challenge datasets work best
- Random internet videos may have poor detection/tracking
- Models expect vehicle-heavy traffic scenes
- Multi-camera tracking requires camera calibration
"""

    with open(links_file, 'w') as f:
        f.write(content)

    print(f"[OK] Created dataset links guide: {links_file}")
    return links_file

if __name__ == "__main__":
    download_sample_videos()
    create_dataset_links()

    print()
    print("Setup complete!")
    print()
    print("To use the system:")
    print("1. Download compatible videos from links above")
    print("2. Start the system: python backend_api.py && cd frontend && npm run dev")
    print("3. Upload videos via: http://localhost:3000")
    print()
