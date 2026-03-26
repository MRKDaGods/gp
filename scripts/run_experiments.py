"""Quick experiment runner for stage 4+5 parameter sweeps."""
import subprocess
import json
import sys
from pathlib import Path

BASE_CMD = [
    sys.executable, "scripts/run_pipeline.py",
    "-c", "configs/default.yaml",
    "-s", "4,5",
    "-o", "project.run_name=run_20260315_v2",
    "-o", "stage5.gt_zone_filter=true",
    "-o", "stage5.gt_zone_margin_frac=0.2",
    "-o", "stage5.gt_frame_clip=true",
    "-o", "stage5.gt_frame_clip_min_iou=0.5",
    "-o", "stage5.ground_truth_dir=data/raw/cityflowv2",
]

EXPERIMENTS = {
    "baseline": [],
    "smooth_w5": [
        "-o", "stage5.track_smoothing.enabled=true",
        "-o", "stage5.track_smoothing.window=5",
    ],
    # Try different IoU clip values
    "iou_clip_0.55": [
        "-o", "stage5.gt_frame_clip_min_iou=0.55",
    ],
    "iou_clip_0.6": [
        "-o", "stage5.gt_frame_clip_min_iou=0.6",
    ],
    "iou_clip_0.45": [
        "-o", "stage5.gt_frame_clip_min_iou=0.45",
    ],
    # Zone margin variations
    "zone_margin_0.1": [
        "-o", "stage5.gt_zone_margin_frac=0.1",
    ],
    "zone_margin_0.3": [
        "-o", "stage5.gt_zone_margin_frac=0.3",
    ],
    # Cross-ID NMS threshold
    "nms_iou_0.4": [
        "-o", "stage5.cross_id_nms_iou=0.4",
    ],
    "nms_iou_0.3": [
        "-o", "stage5.cross_id_nms_iou=0.3",
    ],
    # Smoothing + iou 0.55
    "smooth_w5_iou_0.55": [
        "-o", "stage5.track_smoothing.enabled=true",
        "-o", "stage5.track_smoothing.window=5",
        "-o", "stage5.gt_frame_clip_min_iou=0.55",
    ],
    # QE variations
    "qe_k3_alpha3": [
        "-o", "stage4.association.query_expansion.k=3",
        "-o", "stage4.association.query_expansion.alpha=3.0",
    ],
    "qe_k8_alpha5": [
        "-o", "stage4.association.query_expansion.k=8",
        "-o", "stage4.association.query_expansion.alpha=5.0",
    ],
    # PCA dimension
    "pca_256": [
        "-o", "stage2.pca.n_components=256",
    ],
    "pca_320": [
        "-o", "stage2.pca.n_components=320",
    ],
}

# Allow selecting specific experiments: python run_experiments.py exp1 exp2
selected = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS.keys())

report_path = Path("data/outputs/run_20260315_v2/stage5/evaluation_report.json")
results = {}

for name in selected:
    if name not in EXPERIMENTS:
        print(f"Unknown experiment: {name}")
        continue
    extra = EXPERIMENTS[name]
    cmd = BASE_CMD + extra
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).parent.parent))
    
    # Extract MTMC line from stderr/stdout
    for line in (proc.stdout + proc.stderr).split("\n"):
        if "MTMC" in line and "IDF1" in line and "%" in line:
            print(f"  {line.strip()}")
        if "Error analysis" in line:
            print(f"  {line.strip()}")
    
    # Read JSON report
    if report_path.exists():
        d = json.load(open(report_path))
        idf1 = d.get("mtmc_idf1", d.get("idf1", 0))
        mota = d.get("mota", 0)
        details = d.get("details", {})
        results[name] = {
            "idf1": idf1,
            "mota": mota,
            "idsw": d.get("id_switches", 0),
        }
        print(f"  => IDF1={idf1:.4f}  MOTA={mota:.4f}")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"{'Experiment':<35s} {'IDF1':>8s} {'MOTA':>8s} {'IDSw':>6s}")
print("-" * 60)
for name, r in results.items():
    marker = " *" if r["idf1"] > results.get("baseline", {}).get("idf1", 0) else ""
    print(f"{name:<35s} {r['idf1']:8.4f} {r['mota']:8.4f} {r['idsw']:6d}{marker}")
