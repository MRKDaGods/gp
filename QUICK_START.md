# MTMC Tracker - Complete System Guide

## 🎉 System is Now Running!

**Frontend**: http://localhost:3000
**Backend API**: http://localhost:8000
**API Docs**: http://localhost:8000/docs
**Health Check**: http://localhost:8000/api/health

---

## ⚠️ IMPORTANT: Use Compatible Videos

### Your Models Are Dataset-Specific

The trained models from **https://www.kaggle.com/datasets/mrkdagods/mtmc-weights** were trained on:

1. **VeRi-776** - Vehicle Re-Identification dataset
2. **Market-1501** - Person Re-Identification dataset
3. **AI City Challenge 2023** - Multi-camera vehicle tracking

**This means:** For best results, you should upload videos from these same datasets!

---

## 📥 How to Get Compatible Videos

### Option 1: AI City Challenge 2023 (RECOMMENDED)

**Best for vehicle tracking demonstrations**

1. Visit: https://www.aicitychallenge.org/2023-data-and-evaluation/
2. Register (free) and download **Track 1: Multi-Camera Vehicle Tracking**
3. Extract to: `c:\Users\seift\Downloads\gp\data\samples\cityflow\`
4. Upload via web UI at http://localhost:3000

**Dataset includes:**
- Multiple camera views (S01_c001, S01_c002, S01_c003, etc.)
- Highway and urban scenarios
- Ground truth annotations
- Perfect match for your trained models!

### Option 2: VeRi-776 from Kaggle

**Good for testing vehicle ReID**

```bash
# Install Kaggle CLI
pip install kaggle

# Configure credentials (get from kaggle.com/settings/account)
# Place kaggle.json in: C:\Users\seift\.kaggle\

# Download dataset
kaggle datasets download -d veri-776-vehicle-reid/veri-776

# Extract
unzip veri-776.zip -d c:\Users\seift\Downloads\gp\data\samples\veri776\
```

### Option 3: Demo Mode (Currently Active)

**No download required - uses mock data**

- Click "Demo Mode" button in the upload page
- Perfect for testing the UI workflow
- Results are simulated, not real tracking
- Great for graduation demonstrations without waiting for downloads

---

## 🚀 Quick Start Guide

### For Real Tracking (with Downloaded Videos):

1. **Download AI City Challenge videos** (recommended) or VeRi-776 images
2. **Place in**: `data/samples/cityflow/` or `data/samples/veri776/`
3. **Open**: http://localhost:3000
4. **Upload** one of the dataset videos
5. **Watch** the system automatically:
   - Stage 1: Detect vehicles (red YOLO boxes)
   - Stage 2: Click to select vehicles (green)
   - Stage 3: Configure location/time filters
   - Stage 4: View tracklets across cameras
   - Stage 5: Refine and re-search
   - Stage 6: Export results

### For UI Testing (Demo Mode):

1. **Open**: http://localhost:3000
2. **Click**: "Demo Mode" button
3. **Navigate** through all 6 stages
4. **See** simulated tracking results
5. **Perfect** for showing the UI to your supervisor!

---

## 📂 Expected Directory Structure

After downloading compatible videos:

```
c:\Users\seift\Downloads\gp\
├── backend_api.py              # Running on :8000
├── frontend/                   # Running on :3000
├── models/                     # From Kaggle (optional for demo)
│   ├── yolo/yolov8x.pt
│   ├── reid/osnet_x1_0.pth
│   └── boxmot/osnet_x0_25.pt
├── data/
│   └── samples/
│       ├── cityflow/          # AI City Challenge videos (RECOMMENDED)
│       │   ├── S01_c001.mp4
│       │   ├── S01_c002.mp4
│       │   └── S01_c003.mp4
│       └── veri776/           # VeRi-776 images (alternative)
│           ├── image_train/
│           └── image_test/
└── uploads/                   # User uploaded videos
```

---

## 🎯 Current System Modes

### Demo Mode (Active Now)
- ✅ No models required
- ✅ Works with any video
- ✅ Fast UI testing
- ⚠️ Simulated results
- ⚠️ Not real tracking

### Production Mode (When Models Downloaded)
- ✅ Real YOLO detection
- ✅ Real ReID embeddings
- ✅ Real cross-camera tracking
- ⚠️ Requires model download
- ⚠️ Requires compatible videos

**Check mode**: Visit http://localhost:8000/api/health
- `"models_loaded": false` = Demo mode
- `"models_loaded": true` = Production mode

---

## 🎓 For Your Graduation Demo

### Scenario 1: Quick Demo (5 minutes)

**No downloads needed:**

1. Open http://localhost:3000
2. Click "Demo Mode"
3. Show all6 stages of the pipeline
4. Explain each stage's functionality
5. Show the grid view and statistics

**Perfect for**: Quick presentations, UI/UX demonstration

### Scenario 2: Full Demo (15 minutes)

**With dataset videos:**

1. Download 2-3 sample videos from AI City Challenge
2. Upload via the web interface
3. Show real detection and tracking
4. Demonstrate multi-camera association
5. Export results and show metrics

**Perfect for**: Technical demonstration, showing real capabilities

---

## 🐛 Troubleshooting

### "Upload works but no detections"
→ You're in demo mode. Download models from Kaggle or use demo mode button.

### "Videos don't track well"
→ Use dataset videos (AI City Challenge) instead of random internet videos.

### "Port 3000 or 8000 already in use"
→ Check running processes: `netstat -ano | findstr :3000` or `:8000`

### "Can't download AI City Challenge"
→ Use demo mode for testing, or try VeRi-776 from Kaggle instead.

---

## 📊 What Each Stage Does

1. **Upload** - Drag-drop surveillance videos
2. **Detection** - YOLO finds vehicles (red boxes)
3. **Selection** - Click to select targets (turns green)
4. **Inference** - Set location (Cairo→Downtown→Tahrir) & time range
5. **Timeline** - Clipchamp-style editor, reorder tracklets, confirm matches
6. **Output** - Grid view (1-5x5), timeline, stats, future map

---

## 🔗 Useful Links

- **Models**: https://www.kaggle.com/datasets/mrkdagods/mtmc-weights
- **AI City Challenge**: https://www.aicitychallenge.org/2023-data-and-evaluation/
- **VeRi-776**: https://www.kaggle.com/datasets/veri-776-vehicle-reid/veri-776
- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs
- **Setup Guide**: `SETUP_GUIDE.md`
- **Dataset Info**: `data/DATASET_INFO.md`

---

## ✅ System Status

Run `curl http://localhost:8000/api/health` to check:

```json
{
  "status": "healthy",
  "timestamp": "2026-03-18T...",
  "models_loaded": false,
  "mode": "demo",
  "version": "1.0.0"
}
```

---

## 🎬 Next Steps

1. **Test the UI now**: Open http://localhost:3000 and click "Demo Mode"
2. **Download dataset videos** (when ready for real tracking)
3. **Download models from Kaggle** (optional, for production mode)
4. **Practice your graduation presentation** using the dashboard

---

**Your MTMC Tracker is ready for demonstration!** 🚗📹

Open **http://localhost:3000** to begin.
