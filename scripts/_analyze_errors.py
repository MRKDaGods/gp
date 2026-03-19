"""Quick analysis of error patterns in the current best run."""
import json
from collections import Counter, defaultdict

FORENSIC = "data/outputs/kaggle_10a_v11/extracted/run_kaggle_20260318_114803/stage4/forensic_report.json"
EVAL = "data/outputs/kaggle_10a_v11/extracted/run_kaggle_20260318_114803/stage5/evaluation_report.json"

f = json.load(open(FORENSIC))
e = json.load(open(EVAL))

trajs = f["trajectories"]
cross_cam = [t for t in trajs if t["cross_camera"]]
single_cam = [t for t in trajs if not t["cross_camera"]]

print(f"Total trajectories: {len(trajs)}")
print(f"Cross-camera: {len(cross_cam)}")
print(f"Single-camera: {len(single_cam)}")

cam_dist = Counter(t["num_cameras"] for t in trajs)
print(f"\nCamera count distribution:")
for k, v in sorted(cam_dist.items()):
    print(f"  {k} cameras: {v} trajectories")

# Cross-cam trajectories by scene
s01_cross = [t for t in cross_cam if all(c.startswith("S01") for c in t["cameras_visited"])]
s02_cross = [t for t in cross_cam if all(c.startswith("S02") for c in t["cameras_visited"])]
mixed = [t for t in cross_cam if not all(c.startswith("S01") for c in t["cameras_visited"]) and not all(c.startswith("S02") for c in t["cameras_visited"])]

print(f"\nCross-camera by scene:")
print(f"  S01 only: {len(s01_cross)}")
print(f"  S02 only: {len(s02_cross)}")
print(f"  Cross-scene (S01+S02): {len(mixed)}")

# Per-camera breakdown
per_cam = defaultdict(int)
for t in trajs:
    for cam in t["cameras_visited"]:
        per_cam[cam] += 1

print(f"\nTrajectories per camera:")
for cam, count in sorted(per_cam.items()):
    single_count = sum(1 for t in single_cam if cam in t["cameras_visited"])
    cross_count = count - single_count
    print(f"  {cam}: {count} total ({cross_count} cross-cam, {single_count} single-cam)")

# Single-cam trajectories by camera
print(f"\nSingle-camera trajectories breakdown:")
single_by_cam = Counter(t["cameras_visited"][0] for t in single_cam)
for cam, count in sorted(single_by_cam.items()):
    print(f"  {cam}: {count}")

# Look at evidence quality for cross-cam trajectories
print(f"\nCross-camera trajectory quality:")
for t in sorted(cross_cam, key=lambda x: -x["num_tracklets"])[:10]:
    gid = t["global_id"]
    ncam = t["num_cameras"]
    ntrack = t["num_tracklets"]
    dur = t["total_duration_s"]
    cams = t["cameras_visited"]
    conf = t.get("confidence", "?")
    print(f"  GID {gid}: {ncam} cams, {ntrack} tracklets, {dur:.0f}s, conf={conf}, cams={cams}")

print(f"\n{'='*70}")
print(f"MTMC IDF1: {e['mtmc_idf1']:.4f}")
print(f"ID Switches: {e.get('id_switches', '?')}")
print(f"MTMC ID Switches: {e['details'].get('mtmc_id_switches', '?')}")
