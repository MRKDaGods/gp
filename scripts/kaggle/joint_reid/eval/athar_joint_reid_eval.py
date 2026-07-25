"""ATHAR Phase 6 -- cross-domain vehicle ReID eval matrix (Stage C).

Evaluates TWO checkpoints under identical deployed-mode conditions
(CLIP-norm 224 BICUBIC, flip TTA, post-BN L2 features, SIE skipped, no
rerank/AQE) on four held-out test sets:

  models:   joint  = transreid_joint4d_best.pth  (Stage B output, mounted
                     via kernel_sources athar-joint-reid-train)
            baseline = vehicle_transreid_vit_base_veri776.pth (the deployed
                     transreid_primary stream, mtmc-veri776-pipeline-weights)

  datasets: veri776        query/gallery, market protocol (cross-cam removal)
            veriwild-3000  official 3000-id split (1 query/id, gallery = the
                           rest), market protocol
            vehicleid-800  test_list_800, 10 seeded trials of 1-gallery-per-id
            cityflow-s02   GT crops of the held-out S02 scene (never trained
                           on; the frozen calibration scene); per (id, cam)
                           first crop queries, rest gallery, market protocol

Note: the baseline's VeRi-776 number here is intentionally BELOW the 93.3
paper figure -- that used SIE cam ids + AQE + rerank + fusion. The matrix
compares generalization in ATHAR's deployed single-stream configuration.

Output: /kaggle/working/cross_domain_matrix.json (frozen into the repo by
fetch_results.py).

Local structural smoke (CPU; strict-loads the REAL baseline ckpt from
models/reid/ if present -- that is the load-compat proof):
  ATHAR_LOCAL_SMOKE=1 python athar_joint_reid_eval.py
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

LOCAL_SMOKE = os.environ.get("ATHAR_LOCAL_SMOKE", "0") == "1"

GDRIVE_ID = "13wNJpS_Oaoe-7y5Dzexg_Ol7bKu1OWuC"  # AIC22_Track1_MTMC_Tracking.zip
WORK = Path("/kaggle/working" if not LOCAL_SMOKE else os.environ.get("ATHAR_SMOKE_DIR", "./_smoke_eval"))
TMP = Path("/tmp" if not LOCAL_SMOKE else str(WORK / "tmp"))
INPUT = Path("/kaggle/input")
VIT_MODEL = "vit_base_patch16_clip_224.openai"
H = W = 224
SEED = 0
VEHICLEID_TRIALS = 10

MIN_AREA, MIN_BBOX_SIDE, MAX_CROPS_PER_ID_CAM = 2000, 30, 20


def sh(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def pip_install(*pkgs: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Eval-set builders -- each returns (query_rows, gallery_rows) of
# (path, pid, cam), except vehicleid which returns the raw id->paths map.
# ---------------------------------------------------------------------------


def build_veri776() -> tuple[list, list]:
    import re

    root = None
    for p in INPUT.rglob("image_query"):
        if p.is_dir():
            root = p.parent
            break
    assert root, "VeRi-776 not mounted"
    pat = re.compile(r"^(\d+)_c(\d+)")
    out = {}
    for split in ["image_query", "image_test"]:
        rows = []
        for fname in sorted(os.listdir(root / split)):
            m = pat.match(fname)
            if fname.endswith(".jpg") and m:
                rows.append((str(root / split / fname), int(m.group(1)), int(m.group(2)) - 1))
        out[split] = rows
    print(f"[veri776] query {len(out['image_query'])} gallery {len(out['image_test'])}")
    assert len(out["image_query"]) == 1678 and len(out["image_test"]) == 11579, "bad VeRi mount"
    return out["image_query"], out["image_test"]


def build_veriwild3000() -> tuple[list, list]:
    tarbin = next(INPUT.rglob("veriwild_images.tarbin"), None)
    assert tarbin, "veriwild-test-bin not mounted"
    ex = TMP / "veriwild_test"
    if not (ex / ".done").exists():
        ex.mkdir(parents=True, exist_ok=True)
        print("[veriwild] untarring test images ...")
        r = sh(f"tar -xf {tarbin} -C {ex}")
        assert r.returncode == 0, r.stderr[-400:]
        (ex / ".done").touch()

    def parse(name):
        f = next(INPUT.rglob(name), None)
        assert f, f"{name} not mounted"
        rows = []
        for line in f.read_text().splitlines():
            parts = line.split()  # "<vehid>/<img>.jpg <vehid> <camid>"
            if len(parts) >= 3:
                rows.append((str(ex / parts[0]), int(parts[1]), int(parts[2])))
        return rows

    all_rows = parse("test_3000_id.txt")
    query = parse("test_3000_id_query.txt")
    qpaths = {p for p, _, _ in query}
    gallery = [r for r in all_rows if r[0] not in qpaths]
    print(f"[veriwild-3000] query {len(query)} gallery {len(gallery)}")
    assert len(query) == 3000
    return query, gallery


def build_vehicleid800() -> dict[int, list[str]]:
    zip_path = next(INPUT.rglob("VehicleID_V1.0.zip"), None)
    assert zip_path, "VehicleID zip not mounted"
    ex = TMP / "vehicleid"
    if not (ex / ".done").exists():
        print("[vehicleid] extracting ...")
        with zipfile.ZipFile(str(zip_path)) as zf:
            zf.extractall(str(ex))
        (ex / ".done").touch()
    split_file = next(ex.rglob("train_test_split/test_list_800.txt"), None)
    assert split_file, "test_list_800.txt not found"
    image_root = next(d for c in ["image", "images", "Image", "Images"]
                      if (d := split_file.parent.parent / c).is_dir())
    by_id = defaultdict(list)
    for line in split_file.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            by_id[int(parts[1])].append(str(image_root / f"{parts[0]}.jpg"))
    print(f"[vehicleid-800] {len(by_id)} ids, {sum(map(len, by_id.values()))} images")
    return dict(by_id)


def _load_gt_rows(gt_path: Path) -> list:
    rows = []
    for line in gt_path.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) >= 6:
            rows.append(tuple(int(float(v)) for v in parts[:6]))
    return rows


def build_cityflow_s02() -> tuple[list, list]:
    import cv2

    crop_dir = TMP / "s02_crops"
    if not (crop_dir / ".done").exists():
        staging = TMP / "_aic22_s02"
        if not any(staging.rglob("vdo.avi")):
            archive = TMP / "AIC22_Track1_MTMC_Tracking.zip"
            if not archive.exists():
                print(f"[cityflow] downloading (gdrive id={GDRIVE_ID})...")
                import gdown

                gdown.download(f"https://drive.google.com/uc?id={GDRIVE_ID}", str(archive), quiet=False)
            staging.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(str(archive)) as zf:
                members = [m for m in zf.namelist() if "validation/S02/" in m]
                assert members, "no S02 members in archive"
                zf.extractall(str(staging), members=members)
            archive.unlink(missing_ok=True)
        crop_dir.mkdir(parents=True, exist_ok=True)
        for vdo in sorted(staging.rglob("vdo.avi")):
            cam_name = vdo.parent.name
            gt = vdo.parent / "gt" / "gt.txt"
            if not gt.exists():
                continue
            id_dets = defaultdict(list)
            for frame, tid, x, y, w, h in _load_gt_rows(gt):
                id_dets[tid].append((frame, x, y, w, h))
            frame_to_dets = defaultdict(list)
            for tid, dets in id_dets.items():
                if len(dets) > MAX_CROPS_PER_ID_CAM:
                    step = len(dets) / MAX_CROPS_PER_ID_CAM
                    dets = [dets[int(i * step)] for i in range(MAX_CROPS_PER_ID_CAM)]
                for frame, x, y, w, h in dets:
                    if w * h >= MIN_AREA and w >= MIN_BBOX_SIDE and h >= MIN_BBOX_SIDE:
                        frame_to_dets[frame].append((tid, x, y, w, h))
            cap = cv2.VideoCapture(str(vdo))
            targets = sorted(frame_to_dets)
            ti, current = 0, 0
            while ti < len(targets):
                ret, img = cap.read()
                if not ret:
                    break
                current += 1
                if current < targets[ti]:
                    continue
                while ti < len(targets) and targets[ti] < current:
                    ti += 1
                if ti >= len(targets) or current != targets[ti]:
                    continue
                ih, iw = img.shape[:2]
                for tid, x, y, w, h in frame_to_dets[current]:
                    x1, y1, x2, y2 = max(0, x), max(0, y), min(iw, x + w), min(ih, y + h)
                    if x2 - x1 >= MIN_BBOX_SIDE and y2 - y1 >= MIN_BBOX_SIDE:
                        cv2.imwrite(str(crop_dir / f"{tid:04d}_{cam_name}_f{current:06d}.jpg"),
                                    img[y1:y2, x1:x2])
                ti += 1
            cap.release()
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        (crop_dir / ".done").touch()

    # per (id, cam): first crop -> query, rest -> gallery (cross-cam protocol)
    by_id_cam = defaultdict(list)
    for f in sorted(crop_dir.glob("*.jpg")):
        parts = f.stem.split("_")  # 0042_c006_f000123
        by_id_cam[(int(parts[0]), parts[1])].append(str(f))
    cam2id = {c: i for i, c in enumerate(sorted({c for _, c in by_id_cam}))}
    query, gallery = [], []
    for (tid, cam), paths in sorted(by_id_cam.items()):
        query.append((paths[0], tid, cam2id[cam]))
        gallery.extend((p, tid, cam2id[cam]) for p in paths[1:])
    print(f"[cityflow-s02] query {len(query)} gallery {len(gallery)} "
          f"({len({t for t, _ in by_id_cam})} ids, {len(cam2id)} cams)")
    return query, gallery


# ---------------------------------------------------------------------------
def main() -> None:
    if not LOCAL_SMOKE:
        pip_install("--upgrade", "torch==2.4.1+cu124", "torchvision==0.19.1+cu124",
                    "--index-url", "https://download.pytorch.org/whl/cu124")
        pip_install("timm==1.0.11", "gdown", "opencv-python-headless")

    import numpy as np
    import timm
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as T
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    WORK.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    print(f"device={device} local_smoke={LOCAL_SMOKE}")

    class TransReID(nn.Module):
        """Canon A5alpha arch; SIE parameters kept for strict load, always
        skipped in forward (deployed mode: cameras unseen)."""

        def __init__(self, num_classes, num_cameras, jpm, vit_model, pretrained):
            super().__init__()
            self.vit = timm.create_model(vit_model, pretrained=pretrained, num_classes=0)
            dim = self.vit.embed_dim
            if num_cameras:
                self.sie_embed = nn.Parameter(torch.zeros(num_cameras, 1, dim))
            self.bn = nn.BatchNorm1d(dim)
            self.bn.bias.requires_grad_(False)
            self.cls_head = nn.Linear(dim, num_classes, bias=False)
            if jpm:
                self.bn_jpm = nn.BatchNorm1d(dim)
                self.bn_jpm.bias.requires_grad_(False)
                self.jpm_cls = nn.Linear(dim, num_classes, bias=False)

        def forward(self, x):
            x = self.vit.patch_embed(x)
            x = self.vit._pos_embed(x)
            if hasattr(self.vit, "patch_drop"):
                x = self.vit.patch_drop(x)
            if hasattr(self.vit, "norm_pre"):
                x = self.vit.norm_pre(x)
            for blk in self.vit.blocks:
                x = blk(x)
            x = self.vit.norm(x)
            return F.normalize(self.bn(x[:, 0]), p=2, dim=1)

    def build_model(ckpt_path: Path):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        num_classes = sd["cls_head.weight"].shape[0]
        num_cameras = sd["sie_embed"].shape[0] if "sie_embed" in sd else 0
        jpm = "jpm_cls.weight" in sd
        model = TransReID(num_classes, num_cameras, jpm, VIT_MODEL, pretrained=False)
        model.load_state_dict(sd, strict=True)
        model.eval().to(device)
        print(f"loaded {ckpt_path.name}: classes={num_classes} cams={num_cameras} jpm={jpm} (STRICT)")
        return model

    CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
    tf = T.Compose([
        T.Resize((H, W), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(), T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])

    class Rows(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            path, pid, cam = self.rows[i]
            return tf(Image.open(path).convert("RGB")), pid, cam

    @torch.no_grad()
    def extract(model, rows):
        loader = DataLoader(Rows(rows), batch_size=128, num_workers=0 if LOCAL_SMOKE else 2,
                            pin_memory=not LOCAL_SMOKE)
        feats, pids, cams = [], [], []
        for imgs, pid, cam in loader:
            imgs = imgs.to(device)
            with torch.amp.autocast(device, enabled=device == "cuda"):
                f = model(imgs) + model(torch.flip(imgs, [3]))
            feats.append(F.normalize(f.float(), p=2, dim=1).cpu().numpy())
            pids.append(pid.numpy())
            cams.append(cam.numpy())
        return np.concatenate(feats), np.concatenate(pids), np.concatenate(cams)

    def market_eval(qf, qp, qc, gf, gp, gc, max_rank=10):
        distmat = 1.0 - qf @ gf.T
        indices = np.argsort(distmat, axis=1)
        matches = (gp[indices] == qp[:, None]).astype(np.int32)
        all_cmc, all_ap = [], []
        for qi in range(distmat.shape[0]):
            order = indices[qi]
            remove = (gp[order] == qp[qi]) & (gc[order] == qc[qi])
            raw = matches[qi][~remove]
            if raw.sum() == 0:
                continue
            cmc = raw.cumsum()
            cmc[cmc > 1] = 1
            all_cmc.append(cmc[:max_rank])
            prec = raw.cumsum() / (np.arange(len(raw)) + 1.0)
            all_ap.append((prec * raw).sum() / raw.sum())
        if not all_ap:
            return {"mAP": 0.0, "R1": 0.0, "R5": 0.0, "valid_queries": 0}
        cmc = np.array(all_cmc).mean(0)
        return {"mAP": float(np.mean(all_ap)), "R1": float(cmc[0]),
                "R5": float(cmc[min(4, len(cmc) - 1)]), "valid_queries": len(all_ap)}

    # -- locate checkpoints ---------------------------------------------------
    if LOCAL_SMOKE:
        baseline_path = Path("models/reid/vehicle_transreid_vit_base_veri776.pth")
        joint_path = None
    else:
        baseline_path = next(INPUT.rglob("vehicle_transreid_vit_base_veri776.pth"), None)
        joint_path = next(INPUT.rglob("transreid_joint4d_best.pth"), None)
        assert baseline_path, "baseline ckpt not mounted (attach mrkdagods/mtmc-veri776-pipeline-weights)"
        assert joint_path, "joint ckpt not mounted (attach kernel athar-joint-reid-train as data source)"

    models = {}
    if baseline_path and baseline_path.exists():
        models["baseline_veri776"] = build_model(baseline_path)
    if joint_path:
        models["joint4d"] = build_model(joint_path)
    assert models, "no checkpoints found"

    # -- build eval sets ------------------------------------------------------
    if LOCAL_SMOKE:
        import numpy as _np
        from PIL import Image as _Image

        rng = _np.random.default_rng(0)
        droot = TMP / "smoke_imgs"
        droot.mkdir(parents=True, exist_ok=True)

        def synth_rows(n, pids, cams):
            rows = []
            for i in range(n):
                p = droot / f"{len(list(droot.iterdir()))}_{i}.jpg"
                _Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=_np.uint8)).save(p)
                rows.append((str(p), pids[i % len(pids)], cams[i % len(cams)]))
            return rows

        eval_sets = {"veri776": (synth_rows(6, [0, 1, 2], [0]), synth_rows(12, [0, 1, 2], [1]))}
        vehicleid = {i: [p for p, _, _ in synth_rows(3, [i], [0])] for i in range(4)}
    else:
        eval_sets = {
            "veri776": build_veri776(),
            "cityflow_s02": build_cityflow_s02(),
            "veriwild_3000": build_veriwild3000(),
        }
        vehicleid = build_vehicleid800()

    # -- the matrix -----------------------------------------------------------
    t0 = time.time()
    matrix = {}
    for model_name, model in models.items():
        matrix[model_name] = {}
        for ds_name, (query, gallery) in eval_sets.items():
            qf, qp, qc = extract(model, query)
            gf, gp, gc = extract(model, gallery)
            res = market_eval(qf, qp, qc, gf, gp, gc)
            res.update({"n_query": len(query), "n_gallery": len(gallery)})
            matrix[model_name][ds_name] = res
            print(f"{model_name} x {ds_name}: {json.dumps(res)}", flush=True)

        # VehicleID: 10 seeded trials, 1 random gallery image per id
        import numpy as _np

        all_rows = [(p, pid, 0) for pid, paths in sorted(vehicleid.items()) for p in paths]
        feats, pids_a, _ = extract(model, all_rows)
        by_pid = defaultdict(list)
        for i, (_, pid, _) in enumerate(all_rows):
            by_pid[pid].append(i)
        rng = _np.random.default_rng(SEED)
        trials = []
        for _trial in range(VEHICLEID_TRIALS):
            g_idx = [idxs[rng.integers(len(idxs))] for idxs in by_pid.values()]
            g_set = set(g_idx)
            q_idx = [i for i in range(len(all_rows)) if i not in g_set]
            trials.append(market_eval(
                feats[q_idx], pids_a[q_idx], _np.zeros(len(q_idx), dtype=int),
                feats[g_idx], pids_a[g_idx], _np.ones(len(g_idx), dtype=int),
            ))
        matrix[model_name]["vehicleid_800"] = {
            "mAP": float(_np.mean([t["mAP"] for t in trials])),
            "R1": float(_np.mean([t["R1"] for t in trials])),
            "R5": float(_np.mean([t["R5"] for t in trials])),
            "trials": VEHICLEID_TRIALS,
            "n_images": len(all_rows), "n_ids": len(by_pid),
        }
        print(f"{model_name} x vehicleid_800: {json.dumps(matrix[model_name]['vehicleid_800'])}",
              flush=True)

    payload = {
        "protocol": {
            "mode": "deployed single-stream: CLIP-norm 224 BICUBIC, flip TTA, "
                    "post-BN L2 features, SIE skipped, no rerank/AQE",
            "note": "baseline veri776 is intentionally below the 93.3 paper "
                    "figure (paper used SIE cams + AQE + rerank + fusion)",
            "vehicleid_trials": VEHICLEID_TRIALS,
            "seed": SEED,
        },
        "checkpoints": {
            name: {"path": str(p), "sha256": sha256(p)}
            for name, p in [("baseline_veri776", baseline_path), ("joint4d", joint_path)]
            if p and Path(p).exists()
        },
        "matrix": matrix,
        "elapsed_hours": round((time.time() - t0) / 3600, 3),
    }
    (WORK / "cross_domain_matrix.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
