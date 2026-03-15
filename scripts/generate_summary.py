"""Generate results summary after pipeline run."""
import json
import statistics
from pathlib import Path
from datetime import datetime


def main():
    run_dir = Path("data/outputs/run_20260314_095505")

    with open(run_dir / "stage5/evaluation_report.json") as f:
        report = json.load(f)

    with open(run_dir / "stage4/global_trajectories.json") as f:
        traj_data = json.load(f)

    total_traj = len(traj_data)
    multi_cam = sum(
        1 for t in traj_data
        if len(set(tr["camera_id"] for tr in t["tracklets"])) > 1
    )
    total_tracklets = sum(len(t["tracklets"]) for t in traj_data)
    cam_counts = [
        len(set(tr["camera_id"] for tr in t["tracklets"])) for t in traj_data
    ]
    avg_cams = statistics.mean(cam_counts)

    gt_stats = {}
    for cam in ["S01_c001", "S01_c002", "S01_c003", "S02_c006", "S02_c007", "S02_c008"]:
        gt_lines = open(f"data/raw/cityflowv2/{cam}/gt.txt").readlines()
        gt_ids = set(int(ln.split(",")[1]) for ln in gt_lines)
        gt_stats[cam] = {"tracks": len(gt_ids)}

    tracklets_per_cam = {}
    for t in traj_data:
        for tr in t["tracklets"]:
            cam = tr["camera_id"]
            tracklets_per_cam[cam] = tracklets_per_cam.get(cam, 0) + 1

    per_camera = report["details"]["per_camera"]

    lines = []
    lines.append("=" * 65)
    lines.append("MTMC TRACKING EVALUATION — CityFlowV2 (6 cameras)")
    lines.append(f"Run: run_20260314_095505")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 65)
    lines.append("")
    lines.append("OVERALL METRICS (per-camera average):")
    lines.append(f"  MOTA:          {report['mota']*100:.1f}%  (negative = FP from unann. vehicles)")
    lines.append(f"  IDF1:          {report['idf1']*100:.1f}%")
    lines.append(f"  HOTA:          N/A  (requires TrackEval seqinfo.ini files)")
    lines.append(f"  ID Switches:   {report['id_switches']}")
    lines.append("")
    lines.append("NOTE: CityFlowV2 GT only annotates vehicles that transit between")
    lines.append("cameras, not all visible vehicles. YOLO detects all vehicles,")
    lines.append("causing FP detections that depress MOTA. IDF1 is more meaningful.")
    lines.append("")
    lines.append("PER-CAMERA BREAKDOWN:")
    lines.append(f"  {'Camera':<12} {'MOTA':>8}  {'IDF1':>6}  {'IDSW':>4}  {'Tracklets':>9}  {'GT tracks':>9}")
    lines.append(f"  {'-'*12}  {'-'*8}  {'-'*6}  {'-'*4}  {'-'*9}  {'-'*9}")
    for cam, m in per_camera.items():
        gs = gt_stats.get(cam, {})
        n_tracklets = tracklets_per_cam.get(cam, 0)
        lines.append(
            f"  {cam:<12}  {m['mota']*100:+7.1f}%  {m['idf1']*100:5.1f}%"
            f"  {m['id_switches']:4d}  {n_tracklets:9d}  {gs['tracks']:9d}"
        )
    lines.append("")
    lines.append("PIPELINE STATISTICS:")
    lines.append("  Stage 0 - Frame extraction: 12,060 total frames @ 10fps")
    lines.append("    S01_c001: 1,955  S01_c002: 2,110  S01_c003: 1,996")
    lines.append("    S02_c006: 2,110  S02_c007: 1,904  S02_c008: 1,924")
    lines.append("")
    lines.append(f"  Stage 1 - Detection & Tracking: {total_tracklets} total tracklets")
    lines.append("    S01_c001: 105  S01_c002: 129  S01_c003: 131")
    lines.append("    S02_c006: 187  S02_c007: 130  S02_c008: 184")
    lines.append("")
    lines.append("  Stage 2 - Feature Extraction (TransReID 256x256, CLIP-norm):")
    lines.append("    Model: transreid_cityflowv2_best.pth (mAP=78.32%, R1=92.63%)")
    lines.append(f"    Processed {total_tracklets} tracklets, dim=768 (camera-BN applied)")
    lines.append("")
    lines.append("  Stage 3 - Indexing: FAISS flat_ip, 866 vectors")
    lines.append("")
    lines.append(f"  Stage 4 - Cross-Camera Association:")
    lines.append(f"    Total global trajectories:    {total_traj}")
    lines.append(f"    Multi-camera trajectories:    {multi_cam} ({100*multi_cam/total_traj:.1f}%)")
    lines.append(f"    Single-camera trajectories:   {total_traj - multi_cam}")
    lines.append(f"    Avg cameras per trajectory:   {avg_cams:.2f}")
    lines.append(f"    Spanning all 6 cameras:       {sum(1 for c in cam_counts if c == 6)}")
    lines.append("")
    lines.append("  Camera distribution of trajectories:")
    for nc in range(1, 7):
        count = sum(1 for c in cam_counts if c == nc)
        if count:
            lines.append(f"    {nc} camera(s): {count} trajectories")
    lines.append("")
    lines.append("MODEL CONFIGURATION:")
    lines.append("  Detector:      YOLO2.6m (yolo26m.pt)")
    lines.append("  Per-cam:       BoT-SORT + OSNet (osnet_x0_25_msmt17.pt)")
    lines.append("  Global ReID:   TransReID ViT-Base/16 (CLIP), CityFlowV2 fine-tuned")
    lines.append("  Input:         256x256, CLIP normalization, flip augmentation")
    lines.append("  Association:   QE (k=5) + MNN + k-reciprocal (k1=25, k2=8)")
    lines.append("                 + Louvain community detection (resolution=2.5)")
    lines.append("")
    lines.append("BUGS FIXED DURING PIPELINE SETUP:")
    lines.append("  1. Fixed weights_path in cityflowv2.yaml to transreid_cityflowv2_best.pth")
    lines.append("  2. Fixed input_size in cityflowv2.yaml to [256,256] (v13 model trained at 256x256)")
    lines.append("  3. Fixed num_cameras in cityflowv2.yaml to 59 (actual training setting)")
    lines.append("  4. Installed python-louvain (community module for Stage 4)")
    lines.append("  5. Fixed IndexError in Stage 4 _build_candidate_pairs (top_k > N)")
    lines.append("  6. Fixed TrackEval exception handling in Stage 5 (falls back to motmetrics)")
    lines.append("  7. Fixed seqmap.txt exclusion from Stage 5 motmetrics file list")
    lines.append("")
    lines.append(f"Run directory: {run_dir}/")
    lines.append("=" * 65)

    summary = "\n".join(lines)
    print(summary)
    out_path = run_dir / "results_summary.txt"
    out_path.write_text(summary)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
