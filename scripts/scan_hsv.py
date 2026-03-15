"""Quick HSV weight scan: re-run Stage 4+5 on existing Stage 2 data."""
import sys, json
sys.path.insert(0, ".")
from pathlib import Path
from src.core.config import load_config
from src.core.io_utils import load_embeddings, load_hsv_features, load_tracklets_by_camera
from src.core.data_models import TrackletFeatures
from src.stage3_indexing.faiss_index import FAISSIndex
from src.stage3_indexing.metadata_store import MetadataStore
from src.stage4_association import run_stage4
from src.stage5_evaluation import run_stage5

base = Path("data/outputs/run_20260315_v2")

embeddings, index_map = load_embeddings(base / "stage2")
hsv_matrix = load_hsv_features(base / "stage2")
features = [
    TrackletFeatures(
        track_id=m["track_id"], camera_id=m["camera_id"], class_id=m["class_id"],
        embedding=embeddings[i], hsv_histogram=hsv_matrix[i],
    )
    for i, m in enumerate(index_map)
]
faiss_idx = FAISSIndex("flat_ip")
faiss_idx.load(base / "stage3" / "faiss_index.bin")
meta = MetadataStore(base / "stage3" / "metadata.db")
tracklets = load_tracklets_by_camera(base / "stage1")
print(f"Loaded: {len(features)} features, {sum(len(v) for v in tracklets.values())} tracklets")

print("HSV_W | IDF1 | MOTA | HOTA | ID-SW")
for hsv_w in [0.05, 0.10, 0.15, 0.20]:
    cfg = load_config(
        "configs/default.yaml",
        overrides=[f"stage4.association.weights.vehicle.hsv={hsv_w}", "stage5.gt_frame_clip_min_iou=0.40"],
        dataset_config="configs/datasets/cityflowv2.yaml",
    )
    scan_dir = Path(f"data/outputs/hsvscan_{hsv_w}")
    trajectories = run_stage4(cfg, faiss_idx, meta, features, tracklets, output_dir=scan_dir / "stage4")
    run_stage5(cfg, trajectories, output_dir=scan_dir / "stage5")
    rpt = json.loads((scan_dir / "stage5" / "evaluation_report.json").read_text())
    print(f"  {hsv_w:.2f}  | {rpt['idf1']:.4f} | {rpt['mota']:.4f} | {rpt['hota']:.4f} | {rpt['id_switches']}")
