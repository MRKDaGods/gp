# MTMC Tracker - Complete Setup Guide

## 🚀 Quick Start (3 steps)

### Step 1: Download Models from Kaggle

Download the pre-trained models from:
**https://www.kaggle.com/datasets/mrkdagods/mtmc-weights**

Extract the downloaded archive to create this structure:
```
gp/
├── models/
│   ├── yolo/
│   │   └── yolov8x.pt          # YOLOv8 detection
│   ├── reid/
│   │   ├── osnet_x1_0.pth      # OSNet ReID
│   │   ├── resnet50_ibn.pth    # ResNet50-IBN ReID
│   │   └── transreid.pth       # TransReID (optional)
│   └── boxmot/
│       └── osnet_x0_25.pt      # BoxMOT tracker ReID
```

**Alternative: Manual Download**
```bash
# Install Kaggle CLI
pip install kaggle

# Set up API credentials (get from kaggle.com/settings/account)
# Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\Users\<user>\.kaggle\ (Windows)

# Download dataset
kaggle datasets download -d mrkdagods/mtmc-weights

# Extract
unzip mtmc-weights.zip -d models/
```

### Step 2: Install Dependencies

**Backend (Python):**
```bash
pip install -r backend_requirements.txt
pip install -r requirements.txt  # Main project requirements
```

**Frontend (Node.js):**
```bash
cd frontend
npm install
```

### Step 3: Start the System

**Windows:**
```bash
start.bat
```

**Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
```

**Manual Start (if scripts don't work):**

Terminal 1 - Backend:
```bash
python backend_api.py
```

Terminal 2 - Frontend:
```bash
cd frontend
npm run dev
```

## 🌐 Access Points

- **Frontend Dashboard**: http://localhost:3000
- **Backend API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

## 📦 System Requirements

- **Python**: 3.9+ (with pip)
- **Node.js**: 18+ (with npm)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB for models + datasets
- **GPU**: Optional but recommended (CUDA-compatible for faster inference)

## 🎯 Demo Mode vs Production Mode

### Demo Mode (No Models Required)
- Uses mock data for all stages
- Perfect for UI testing and demonstrations
- No GPU required
- Enable by setting `NEXT_PUBLIC_DEMO_MODE=true` in `frontend/.env.local`

### Production Mode (Models Required)
- Full pipeline with real YOLO detection and ReID
- Requires downloaded models from Kaggle
- GPU recommended for performance
- Enable by setting `NEXT_PUBLIC_DEMO_MODE=false` in `frontend/.env.local`

## 🔧 Configuration

### Backend Config
Edit `configs/default.yaml` to configure:
- Detection thresholds
- ReID model selection
- FAISS index parameters
- Camera mappings

### Frontend Config
Edit `frontend/.env.local`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000/api
NEXT_PUBLIC_DEMO_MODE=false
NEXT_PUBLIC_WS_URL=ws://localhost:8000/api
```

## 📊 Testing the System

1. **Upload Test Video**: Use sample video from `data/samples/` or drag-drop your own
2. **Stage 1 - Detection**: Red YOLO boxes appear automatically
3. **Stage 2 - Selection**: Click boxes to select vehicles (green)
4. **Stage 3 - Inference**: Set location (Cairo→Downtown→Tahrir) and date range
5. **Stage 4 - Timeline**: View tracklets across cameras, drag to reorder
6. **Stage 5 - Refinement**: Select reference frames, re-search matches
7. **Stage 6 - Output**: View summarized video, grid, and statistics

## 🐛 Troubleshooting

### Backend won't start
- Check Python version: `python --version` (needs 3.9+)
- Install dependencies: `pip install -r backend_requirements.txt`
- Check port availability: `netstat -ano | findstr :8000` (Windows)

### Frontend won't start
- Check Node version: `node --version` (needs 18+)
- Clear cache: `cd frontend && rm -rf .next node_modules && npm install`
- Check port availability: `netstat -ano | findstr :3000` (Windows)

### Models not loading
- Verify models directory structure matches above
- Check file permissions
- Run in demo mode first to verify system works

### CORS errors
- Backend must be running before frontend
- Check CORS middleware in `backend_api.py` includes your frontend URL
- Clear browser cache

## 📚 Kaggle Models Details

The `mtmc-weights` dataset includes:

1. **YOLOv8x** (yolov8x.pt)
   - Object detection for vehicles
   - Classes: car, truck, bus, motorcycle
   - mAP@50: 0.89

2. **OSNet-x1.0** (osnet_x1_0.pth)
   - Person/Vehicle ReID embeddings
   - 512-dim features
   - Trained on Market-1501 + VeRi-776

3. **ResNet50-IBN** (resnet50_ibn.pth)
   - Alternative ReID model
   - 2048-dim features
   - Better for vehicles

4. **TransReID** (transreid.pth) - Optional
   - Transformer-based ReID
   - State-of-the-art accuracy
   - Slower inference

5. **BoxMOT ReID** (osnet_x0_25.pt)
   - Lightweight ReID for tracker
   - Real-time performance

## 🎓 Next Steps

1. **Upload Your Videos**: Place videos in `uploads/` or use web interface
2. **Configure Cameras**: Edit camera mappings in `configs/default.yaml`
3. **Run Full Pipeline**: Use the dashboard or CLI
4. **Export Results**: Download trajectories, videos, statistics
5. **Future: Add GPS Data**: Enable map-based features (see frontend Stage 6)

## 📝 API Documentation

Full API documentation available at:
http://localhost:8000/docs (Swagger UI)
http://localhost:8000/redoc (ReDoc)

## 🤝 Support

- GitHub Issues: Report bugs and feature requests
- Documentation: See `docs/` folder
- Team Workload: See `docs/team_workload.md`

---

**Ready to Demo!** 🎉

Run `start.bat` (Windows) or `./start.sh` (Linux/Mac) and open http://localhost:3000
