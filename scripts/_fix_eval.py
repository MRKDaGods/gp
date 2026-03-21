"""Fix evaluation in 09b and 09c notebooks: scene-based split + multi-cam eval + R1 bug."""
import json

# ---- Fix 09c notebook ----
path_c = "notebooks/kaggle/09c_kd_vitl_teacher/09c_kd_vitl_teacher.ipynb"
with open(path_c, "r", encoding="utf-8") as f:
    nb = json.load(f)

CELL11_CODE = r'''TRAIN_SCENES = {"S01", "S03", "S04"}
EVAL_SCENES  = {"S02", "S05"}
import random
random.seed(42)

id_to_crops = defaultdict(list)
for path, tid, cam in all_crops:
    id_to_crops[tid].append((path, tid, cam))

# Scene-based train/val split (CityFlowV2 protocol)
train_ids, eval_ids_all = set(), set()
for tid in id_to_crops:
    scene = tid.split("_")[0]
    if scene in TRAIN_SCENES:
        train_ids.add(tid)
    elif scene in EVAL_SCENES:
        eval_ids_all.add(tid)

# Only multi-camera eval IDs for query/gallery; single-cam -> distractors
eval_id_cams = {tid: {c[2] for c in id_to_crops[tid]} for tid in eval_ids_all}
eval_ids = {tid for tid, cams in eval_id_cams.items() if len(cams) >= 2}
single_cam_eval = eval_ids_all - eval_ids

train_crops, query_crops, gallery_crops = [], [], []
for tid in sorted(train_ids):
    train_crops.extend(id_to_crops[tid])

for tid in sorted(eval_ids):
    crops = id_to_crops[tid]
    cams = sorted({c[2] for c in crops})
    q_cam = cams[0]
    q_list = [c for c in crops if c[2] == q_cam]
    g_list = [c for c in crops if c[2] != q_cam]
    query_crops.extend(q_list)
    gallery_crops.extend(g_list)

# Single-camera eval IDs as gallery distractors
for tid in sorted(single_cam_eval):
    gallery_crops.extend(id_to_crops[tid])

num_classes = len(train_ids)
id2label = {tid: i for i, tid in enumerate(sorted(train_ids))}
print(f"Train IDs  : {len(train_ids)}  (crops: {len(train_crops)})")
print(f"Eval IDs   : {len(eval_ids)} multi-cam, {len(single_cam_eval)} single-cam distractors")
print(f"Query crops: {len(query_crops)},  Gallery: {len(gallery_crops)}")
print(f"num_classes: {num_classes}")'''

lines = CELL11_CODE.split("\n")
nb["cells"][11]["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]

# Cell 17: Fix matches[order][0] -> matches[0] R1 bug
src17 = nb["cells"][17]["source"]
new_src17 = []
for line in src17:
    if "if matches[order][0]:" in line:
        line = line.replace("matches[order][0]", "matches[0]")
    new_src17.append(line)
nb["cells"][17]["source"] = new_src17

with open(path_c, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)

print("09c notebook fixed (cell 11 split + cell 17 R1 bug)")

# ---- Fix 09b notebook ----
path_b = "notebooks/kaggle/09b_vehicle_reid_384px/09b_vehicle_reid_384px.ipynb"
with open(path_b, "r", encoding="utf-8") as f:
    nb = json.load(f)

CELL11B_CODE = r'''if not all_crops:
    raise RuntimeError("No crops extracted! Check download and extraction.")

TRAIN_SCENES = {"S01", "S03", "S04"}
EVAL_SCENES  = {"S02", "S05"}

# Scene-based train/val split (CityFlowV2 protocol)
train_ids_all = sorted(tid for tid in all_crops if tid.split("_")[0] in TRAIN_SCENES)
eval_ids_all  = sorted(tid for tid in all_crops if tid.split("_")[0] in EVAL_SCENES)

# Multi-camera eval IDs for query/gallery; single-cam -> distractors
multi_cam_eval  = sorted(tid for tid in eval_ids_all if len(all_crops[tid]) >= MIN_CAMS_FOR_EVAL)
single_cam_eval = sorted(tid for tid in eval_ids_all if len(all_crops[tid]) < MIN_CAMS_FOR_EVAL)

train_ids = set(train_ids_all)
eval_ids  = set(multi_cam_eval)

cam_names = sorted({cam for cams in all_crops.values() for cam in cams})
cam2id = {c: i for i, c in enumerate(cam_names)}
num_cameras = len(cam_names)

train_pid_set = sorted(train_ids)
pid2label = {tid: i for i, tid in enumerate(train_pid_set)}
num_classes = len(train_pid_set)

rng = np.random.RandomState(SEED)

train_data, query_data, gallery_data = [], [], []

for tid in sorted(train_ids):
    label = pid2label[tid]
    for cam_name, paths in all_crops[tid].items():
        camid = cam2id[cam_name]
        for p in paths:
            train_data.append((p, label, camid))

eval_pid2label = {tid: i for i, tid in enumerate(sorted(eval_ids))}
for tid in sorted(eval_ids):
    pid = eval_pid2label[tid]
    for cam_name, paths in all_crops[tid].items():
        if not paths:
            continue
        camid = cam2id[cam_name]
        idx = rng.randint(0, len(paths))
        query_data.append((paths[idx], pid, camid))
        for i, p in enumerate(paths):
            if i != idx:
                gallery_data.append((p, pid, camid))

distractor_pid = len(eval_ids)
for tid in sorted(single_cam_eval):
    for cam_name, paths in all_crops[tid].items():
        camid = cam2id[cam_name]
        for p in paths:
            gallery_data.append((p, distractor_pid, camid))
    distractor_pid += 1

print(f"Train: {len(train_data)} images, {num_classes} IDs (scenes: S01/S03/S04)")
print(f"Eval : {len(eval_ids)} multi-cam IDs, {len(single_cam_eval)} single-cam distractors (scenes: S02/S05)")
print(f"Query: {len(query_data)}, Gallery: {len(gallery_data)}, Cameras: {num_cameras}")'''

lines = CELL11B_CODE.split("\n")
nb["cells"][11]["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]

with open(path_b, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)

print("09b notebook fixed (cell 11 scene-based split)")
print("Done!")
