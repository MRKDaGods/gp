from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks" / "kaggle" / "09f_vehicle_reid_resnet101ibn_cityflowv2"
NOTEBOOK_PATH = NOTEBOOK_DIR / "09f_vehicle_reid_resnet101ibn_cityflowv2.ipynb"
METADATA_PATH = NOTEBOOK_DIR / "kernel-metadata.json"


def to_source(text: str) -> list[str]:
    cleaned = dedent(text).strip("\n")
    if not cleaned:
        return []
    lines = cleaned.splitlines()
    return [f"{line}\n" for line in lines[:-1]] + [lines[-1]]


def markdown_cell(cell_id: str, text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {"language": "markdown"},
        "source": to_source(text),
    }


def code_cell(cell_id: str, text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {"language": "python"},
        "outputs": [],
        "source": to_source(text),
    }


def build_cells() -> list[dict]:
    return [
        markdown_cell(
            "09f-title",
            """
            # 09f Vehicle ReID: ResNet101-IBN-a CityFlowV2 Fine-tune (VeRi-776 Pretrained)

            **Inputs:**
            - 09e kernel output: VeRi-776-pretrained ResNet101-IBN-a checkpoint (`best_model.pth`, 62.52% mAP on VeRi-776)
            - CityFlowV2 dataset: `thanhnguyenle/data-aicity-2023-track-2`
            - Shared weights: `mrkdagods/mtmc-weights`
            - Project repo snapshot: `mrkdagods/mtmc-gp`

            **Training:**
            - Fine-tune on CityFlowV2 with differential learning rates, 384x384 inputs, cosine schedule
            - ID loss + Triplet loss + Circle loss, mixed precision, evaluation every 5 epochs

            **Outputs:**
            - `/kaggle/working/resnet101ibn_veri776_cityflowv2_384px_best.pth`
            - `/kaggle/working/resnet101ibn_veri776_cityflowv2_384px_deploy.pth`
            - `/kaggle/working/training_history_09f.json`
            - `/kaggle/working/metrics_09f.json`
            - `/kaggle/working/debug.log`
            """,
        ),
        code_cell(
            "09f-logger",
            """
            import atexit
            import pathlib
            import sys

            _LOG_PATH = pathlib.Path("/kaggle/working/debug.log")
            _LOG = _LOG_PATH.open("w", buffering=1)

            class _TeeWriter:
                def __init__(self, original, log_file):
                    self.original = original
                    self.log_file = log_file

                def write(self, value):
                    self.original.write(value)
                    try:
                        self.log_file.write(value)
                    except Exception:
                        pass

                def flush(self):
                    self.original.flush()
                    try:
                        self.log_file.flush()
                    except Exception:
                        pass

            sys.stdout = _TeeWriter(sys.stdout, _LOG)
            sys.stderr = _TeeWriter(sys.stderr, _LOG)

            @atexit.register
            def _shutdown_log():
                try:
                    _LOG.close()
                except Exception:
                    pass

            print("[09f] debug logging initialized")
            print(f"[09f] writing log to {_LOG_PATH}")
            """,
        ),
        code_cell(
            "09f-bootstrap",
            """
            import json
            import re
            import shutil
            import subprocess
            import sys

            def pip_install(*args):
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

            gpu_info = []
            if shutil.which("nvidia-smi"):
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=gpu_name,compute_cap", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0:
                    for raw_line in result.stdout.splitlines():
                        line = raw_line.strip()
                        if not line:
                            continue
                        name, _, capability = line.partition(",")
                        match = re.search(r"(\\d+)\\.(\\d+)", capability)
                        if match:
                            sm = int(match.group(1)) * 10 + int(match.group(2))
                            gpu_info.append({"name": name.strip(), "capability": capability.strip(), "sm": sm})

            if any(entry["sm"] < 70 for entry in gpu_info):
                pip_install(
                    "torch==2.4.1",
                    "torchvision==0.19.1",
                    "--index-url",
                    "https://download.pytorch.org/whl/cu124",
                )

            pip_install("timm==0.9.16", "loguru", "omegaconf", "scikit-learn")

            import torch

            print(
                json.dumps(
                    {
                        "python": sys.version.split()[0],
                        "torch": torch.__version__,
                        "cuda_available": torch.cuda.is_available(),
                        "device_count": torch.cuda.device_count(),
                        "gpu_info": gpu_info,
                    },
                    indent=2,
                )
            )
            """,
        ),
        code_cell(
            "09f-repo",
            """
            import os
            import subprocess
            import sys
            from pathlib import Path

            WORK_DIR = Path("/kaggle/working")
            REPO_CANDIDATES = [
                Path("/kaggle/input/mtmc-gp/gp"),
                Path("/kaggle/input/mtmc-gp"),
            ]

            PROJECT = None
            for candidate in REPO_CANDIDATES:
                if (candidate / "pyproject.toml").exists() or (candidate / "setup.py").exists():
                    PROJECT = candidate
                    break

            if PROJECT is None:
                PROJECT = WORK_DIR / "gp"
                if not PROJECT.exists():
                    subprocess.check_call(
                        [
                            "git",
                            "clone",
                            "--depth",
                            "1",
                            "--branch",
                            "feature/people-tracking",
                            "https://github.com/MRKDaGods/gp.git",
                            str(PROJECT),
                        ]
                    )
                else:
                    subprocess.check_call(["git", "-C", str(PROJECT), "pull", "--ff-only"])

            os.chdir(PROJECT)
            sys.path.insert(0, str(PROJECT))
            print(f"[09f] project root: {PROJECT}")
            """,
        ),
        code_cell(
            "09f-deps",
            """
            import subprocess
            import sys

            def pip_install(*args):
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

            requirements_path = PROJECT / "requirements.txt"
            assert requirements_path.exists(), requirements_path
            pip_install("-r", str(requirements_path))
            pip_install("-e", str(PROJECT), "--no-deps")

            try:
                import torchreid  # noqa: F401
            except ImportError:
                pip_install("git+https://github.com/KaiyangZhou/deep-person-reid.git")

            import cv2  # noqa: F401
            import numpy  # noqa: F401
            import PIL  # noqa: F401
            import timm  # noqa: F401

            print("[09f] repository dependencies installed")
            """,
        ),
        code_cell(
            "09f-config",
            """
            import json
            import os
            from pathlib import Path

            CITYFLOW_CANDIDATES = [
                Path("/kaggle/input/mtmc-cityflowv2-frames"),
                Path("/kaggle/input/data-aicity-2023-track-2"),
                Path("/kaggle/input/datasets/thanhnguyenle/data-aicity-2023-track-2"),
            ]
            CITYFLOW_ROOT = next((path for path in CITYFLOW_CANDIDATES if path.exists()), None)
            assert CITYFLOW_ROOT is not None, f"CityFlowV2 dataset not found. Tried: {CITYFLOW_CANDIDATES}"

            RAW_TRAIN_CANDIDATES = [
                CITYFLOW_ROOT / "AIC22_Track1_MTMC_Tracking" / "train",
                CITYFLOW_ROOT / "train",
            ]
            RAW_TRAIN_ROOT = next((path for path in RAW_TRAIN_CANDIDATES if path.exists()), None)
            assert RAW_TRAIN_ROOT is not None, f"CityFlowV2 train split not found. Tried: {RAW_TRAIN_CANDIDATES}"

            WEIGHTS_CANDIDATES = [
                Path("/kaggle/input/mtmc-weights"),
                Path("/kaggle/input/datasets/mrkdagods/mtmc-weights"),
            ]
            WEIGHTS_DIR = next((path for path in WEIGHTS_CANDIDATES if path.exists()), None)
            assert WEIGHTS_DIR is not None, f"mtmc-weights dataset not found. Tried: {WEIGHTS_CANDIDATES}"

            PRETRAINED_CANDIDATES = [
                Path("/kaggle/input/09e-vehicle-reid-resnet101-ibn-a-veri-776-pretrain/best_model.pth"),
                Path("/kaggle/input/09e-vehicle-reid-resnet101-ibn-a-veri-776-pretrain/09e_vehicle_reid_resnet101ibn_veri776/best_model.pth"),
                Path("/kaggle/input/09e-vehicle-reid-resnet101-ibn-a-veri-776-pretrain/resnet101ibn_veri776_best.pth"),
                Path("/kaggle/input/09e-vehicle-reid-resnet101-ibn-a-veri-776-pretrain/09e_vehicle_reid_resnet101ibn_veri776/reid_veri776_resnet101_ibn_a_best.pth"),
            ]
            PRETRAINED_PATH = next((path for path in PRETRAINED_CANDIDATES if path.exists()), None)
            assert PRETRAINED_PATH is not None, f"Cannot find 09e checkpoint. Tried: {PRETRAINED_CANDIDATES}"

            OUTPUT_DIR = Path("/kaggle/working/09f_output")
            CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
            REID_ROOT = Path("/kaggle/working/cityflowv2_reid")
            TMP_CROP_ROOT = Path("/kaggle/working/cityflowv2_crops")
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

            BACKBONE = "resnet101_ibn_a"
            IMG_SIZE = (384, 384)
            FEAT_DIM = 2048
            GEM_P = 3.0
            EPOCHS = 60
            BATCH_SIZE = 32
            NUM_INSTANCES = 4
            NUM_WORKERS = 4
            LR = 7e-5
            WARMUP_EPOCHS = 5
            WEIGHT_DECAY = 5e-4
            LABEL_SMOOTHING = 0.1
            TRIPLET_MARGIN = 0.3
            CIRCLE_M = 0.25
            CIRCLE_GAMMA = 80
            CIRCLE_WEIGHT = 1.0
            RANDOM_ERASING = 0.5
            COLOR_JITTER = True
            EVAL_EVERY = 5
            FP16 = True

            print(
                json.dumps(
                    {
                        "cityflow_root": str(CITYFLOW_ROOT),
                        "raw_train_root": str(RAW_TRAIN_ROOT),
                        "weights_dir": str(WEIGHTS_DIR),
                        "pretrained_path": str(PRETRAINED_PATH),
                        "output_dir": str(OUTPUT_DIR),
                        "reid_root": str(REID_ROOT),
                        "img_size": IMG_SIZE,
                        "epochs": EPOCHS,
                        "batch_size": BATCH_SIZE,
                        "lr": LR,
                    },
                    indent=2,
                )
            )
            """,
        ),
        code_cell(
            "09f-extract",
            """
            import gc
            import json
            import shutil
            from collections import defaultdict
            from pathlib import Path

            import cv2
            import numpy as np

            MAX_SAMPLES_PER_TRACK = 15
            MIN_BOX_AREA = 2000
            MIN_BOX_SIDE = 30

            if TMP_CROP_ROOT.exists():
                shutil.rmtree(TMP_CROP_ROOT)
            if REID_ROOT.exists():
                shutil.rmtree(REID_ROOT)
            TMP_CROP_ROOT.mkdir(parents=True, exist_ok=True)
            for split_name in ["train", "query", "gallery"]:
                (REID_ROOT / split_name).mkdir(parents=True, exist_ok=True)

            camera_dirs = sorted(path for path in RAW_TRAIN_ROOT.glob("S*/c*") if path.is_dir())
            assert camera_dirs, f"No camera directories found under {RAW_TRAIN_ROOT}"

            all_crops = {}
            total_crop_count = 0

            for cam_dir in camera_dirs:
                scene = cam_dir.parent.name
                cam = cam_dir.name
                cam_id = f"{scene}_{cam}"
                gt_file = cam_dir / "gt" / "gt.txt"
                video_candidates = [cam_dir / "vdo.avi", cam_dir / "vdo.mp4"]
                video_file = next((path for path in video_candidates if path.exists()), None)

                if not gt_file.exists() or video_file is None:
                    print(f"[09f] skip {cam_id}: missing gt or video")
                    continue

                detections = defaultdict(list)
                with gt_file.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        parts = line.strip().split(",")
                        if len(parts) < 6:
                            continue
                        frame_id = int(float(parts[0]))
                        track_id = int(float(parts[1]))
                        x = int(float(parts[2]))
                        y = int(float(parts[3]))
                        w = int(float(parts[4]))
                        h = int(float(parts[5]))
                        if w * h < MIN_BOX_AREA or w < MIN_BOX_SIDE or h < MIN_BOX_SIDE:
                            continue
                        detections[track_id].append((frame_id, x, y, w, h))

                sampled_by_frame = defaultdict(list)
                for track_id, dets in detections.items():
                    dets.sort(key=lambda item: item[0])
                    if len(dets) <= MAX_SAMPLES_PER_TRACK:
                        sampled = dets
                    else:
                        sampled_indices = np.linspace(0, len(dets) - 1, num=MAX_SAMPLES_PER_TRACK, dtype=int)
                        sampled = [dets[index] for index in sampled_indices]
                    for frame_id, x, y, w, h in sampled:
                        sampled_by_frame[frame_id].append((track_id, x, y, w, h))

                cap = cv2.VideoCapture(str(video_file))
                if not cap.isOpened():
                    print(f"[09f] skip {cam_id}: failed to open {video_file}")
                    continue

                camera_crop_count = 0
                for frame_id in sorted(sampled_by_frame.keys()):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    frame_height, frame_width = frame.shape[:2]
                    for track_id, x, y, w, h in sampled_by_frame[frame_id]:
                        x1 = max(0, x)
                        y1 = max(0, y)
                        x2 = min(frame_width, x + w)
                        y2 = min(frame_height, y + h)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue
                        file_name = f"{track_id:04d}_{scene}_{cam}_f{frame_id:06d}.jpg"
                        crop_path = TMP_CROP_ROOT / file_name
                        if not cv2.imwrite(str(crop_path), crop):
                            continue
                        all_crops.setdefault(track_id, {}).setdefault(cam_id, []).append((str(crop_path), frame_id))
                        camera_crop_count += 1

                cap.release()
                total_crop_count += camera_crop_count
                print(f"[09f] {cam_id}: {len(detections)} vehicles, {camera_crop_count} crops")
                gc.collect()

            multi_cam_ids = [track_id for track_id, cams in all_crops.items() if len(cams) >= 2]
            single_cam_ids = [track_id for track_id, cams in all_crops.items() if len(cams) == 1]
            rng = np.random.default_rng(42)
            rng.shuffle(multi_cam_ids)

            n_train = int(len(multi_cam_ids) * 0.7)
            train_ids = set(multi_cam_ids[:n_train])
            eval_ids = set(multi_cam_ids[n_train:])

            split_counts = {"train": 0, "query": 0, "gallery": 0}

            for track_id in sorted(train_ids):
                for _, items in sorted(all_crops[track_id].items()):
                    for crop_path, _ in sorted(items, key=lambda item: item[1]):
                        shutil.copy2(crop_path, REID_ROOT / "train" / Path(crop_path).name)
                        split_counts["train"] += 1

            for track_id in sorted(eval_ids):
                for _, items in sorted(all_crops[track_id].items()):
                    ordered_items = sorted(items, key=lambda item: item[1])
                    if not ordered_items:
                        continue
                    query_source, _ = ordered_items[0]
                    shutil.copy2(query_source, REID_ROOT / "query" / Path(query_source).name)
                    split_counts["query"] += 1
                    for gallery_source, _ in ordered_items[1:]:
                        shutil.copy2(gallery_source, REID_ROOT / "gallery" / Path(gallery_source).name)
                        split_counts["gallery"] += 1

            for track_id in sorted(single_cam_ids):
                for _, items in sorted(all_crops[track_id].items()):
                    for crop_path, _ in sorted(items, key=lambda item: item[1]):
                        shutil.copy2(crop_path, REID_ROOT / "gallery" / Path(crop_path).name)
                        split_counts["gallery"] += 1

            SPLIT_STATS = {
                "total_vehicle_ids": len(all_crops),
                "total_crops": total_crop_count,
                "train_ids": len(train_ids),
                "eval_ids": len(eval_ids),
                "single_cam_ids": len(single_cam_ids),
                **split_counts,
            }

            with (OUTPUT_DIR / "split_stats_09f.json").open("w", encoding="utf-8") as handle:
                json.dump(SPLIT_STATS, handle, indent=2)

            print(json.dumps(SPLIT_STATS, indent=2))
            """,
        ),
        code_cell(
            "09f-parse",
            """
            import json

            from src.training.datasets import parse_cityflowv2

            train_data, query_data, gallery_data = parse_cityflowv2(str(REID_ROOT))
            NUM_CLASSES = len({pid for _, pid, _ in train_data})
            NUM_CAMERAS = len({cam for _, _, cam in train_data})

            print(
                json.dumps(
                    {
                        "train_images": len(train_data),
                        "query_images": len(query_data),
                        "gallery_images": len(gallery_data),
                        "num_classes": NUM_CLASSES,
                        "num_cameras": NUM_CAMERAS,
                    },
                    indent=2,
                )
            )
            """,
        ),
        code_cell(
            "09f-loaders",
            """
            from src.training.datasets import build_dataloader

            train_loader, query_loader, gallery_loader, loader_classes, loader_cameras = build_dataloader(
                dataset_name="cityflowv2",
                root=str(REID_ROOT),
                height=IMG_SIZE[0],
                width=IMG_SIZE[1],
                batch_size=BATCH_SIZE,
                num_instances=NUM_INSTANCES,
                num_workers=NUM_WORKERS,
                random_erasing_prob=RANDOM_ERASING,
                color_jitter=COLOR_JITTER,
            )
            assert loader_classes == NUM_CLASSES, (loader_classes, NUM_CLASSES)
            assert loader_cameras == NUM_CAMERAS, (loader_cameras, NUM_CAMERAS)

            print(
                {
                    "train_batches": len(train_loader),
                    "query_batches": len(query_loader),
                    "gallery_batches": len(gallery_loader),
                    "batch_size": BATCH_SIZE,
                }
            )
            """,
        ),
        code_cell(
            "09f-model",
            """
            import json

            import torch

            from src.training.model import ReIDModelResNet101IBN

            device = "cuda:0" if torch.cuda.is_available() else "cpu"

            model = ReIDModelResNet101IBN(
                num_classes=NUM_CLASSES,
                last_stride=1,
                pretrained=False,
                gem_p=GEM_P,
            )

            checkpoint = torch.load(PRETRAINED_PATH, map_location="cpu", weights_only=False)
            if "model" in checkpoint:
                pretrained_state = checkpoint["model"]
            elif "state_dict" in checkpoint:
                pretrained_state = checkpoint["state_dict"]
            else:
                pretrained_state = checkpoint

            pretrained_state = {key.replace("module.", "", 1): value for key, value in pretrained_state.items()}

            loaded_keys = []
            skipped_keys = []
            model_state = model.state_dict()
            for key, value in pretrained_state.items():
                if key.startswith("classifier"):
                    skipped_keys.append(key)
                    continue
                if key in model_state and tuple(value.shape) == tuple(model_state[key].shape):
                    model_state[key] = value
                    loaded_keys.append(key)
                else:
                    skipped_keys.append(key)

            model.load_state_dict(model_state)
            model = model.to(device)
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)

            LOAD_SUMMARY = {
                "device": device,
                "loaded_param_count": len(loaded_keys),
                "skipped_param_count": len(skipped_keys),
                "skipped_preview": skipped_keys[:10],
            }
            print(json.dumps(LOAD_SUMMARY, indent=2))
            """,
        ),
        code_cell(
            "09f-losses",
            """
            import json

            from src.training.losses import CircleLoss, CrossEntropyLabelSmooth, TripletLoss

            LOSS_RECIPE = "id+triplet+circle"
            id_loss_fn = CrossEntropyLabelSmooth(NUM_CLASSES, epsilon=LABEL_SMOOTHING)
            triplet_loss_fn = TripletLoss(margin=TRIPLET_MARGIN)
            circle_loss_fn = CircleLoss(m=CIRCLE_M, gamma=CIRCLE_GAMMA)
            center_loss_fn = None

            print(
                json.dumps(
                    {
                        "loss_recipe": LOSS_RECIPE,
                        "label_smoothing": LABEL_SMOOTHING,
                        "triplet_margin": TRIPLET_MARGIN,
                        "circle_m": CIRCLE_M,
                        "circle_gamma": CIRCLE_GAMMA,
                    },
                    indent=2,
                )
            )
            """,
        ),
        code_cell(
            "09f-optim",
            """
            import json

            import torch

            from src.training.train_reid import build_cosine_scheduler

            base_model = model.module if hasattr(model, "module") else model
            optimizer = torch.optim.AdamW(
                [
                    {"params": base_model.backbone.parameters(), "lr": LR * 0.1},
                    {"params": base_model.pool.parameters(), "lr": LR},
                    {"params": base_model.bottleneck.parameters(), "lr": LR},
                    {"params": base_model.classifier.parameters(), "lr": LR * 10},
                ],
                lr=LR,
                weight_decay=WEIGHT_DECAY,
            )
            scheduler = build_cosine_scheduler(
                optimizer,
                warmup_epochs=WARMUP_EPOCHS,
                total_epochs=EPOCHS,
                eta_min=1e-7,
            )
            scaler = torch.amp.GradScaler("cuda") if FP16 and device.startswith("cuda") else None

            print(
                json.dumps(
                    {
                        "param_group_lrs": [group["lr"] for group in optimizer.param_groups],
                        "warmup_epochs": WARMUP_EPOCHS,
                        "epochs": EPOCHS,
                        "fp16": scaler is not None,
                    },
                    indent=2,
                )
            )
            """,
        ),
        code_cell(
            "09f-eval",
            """
            import json

            from src.training.evaluate_reid import evaluate_reid

            def evaluate_model(current_model):
                mAP, cmc, _, _ = evaluate_reid(
                    current_model,
                    query_loader,
                    gallery_loader,
                    device=device,
                    rerank=False,
                )
                return {
                    "mAP": float(mAP),
                    "rank1": float(cmc[0]) if len(cmc) > 0 else 0.0,
                    "rank5": float(cmc[4]) if len(cmc) > 4 else 0.0,
                    "rank10": float(cmc[9]) if len(cmc) > 9 else 0.0,
                }

            BASELINE_METRICS = evaluate_model(model)
            print(json.dumps(BASELINE_METRICS, indent=2))
            """,
        ),
        code_cell(
            "09f-train",
            """
            import json
            import time

            import torch

            from src.training.train_reid import train_one_epoch

            history = []
            best_mAP = -1.0
            best_checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
            last_checkpoint_path = CHECKPOINT_DIR / "last_model.pth"
            history_path = Path("/kaggle/working/training_history_09f.json")

            for epoch in range(EPOCHS):
                epoch_start = time.time()
                train_metrics = train_one_epoch(
                    model,
                    train_loader,
                    id_loss_fn,
                    triplet_loss_fn,
                    center_loss_fn,
                    optimizer,
                    None,
                    scaler,
                    device,
                    epoch,
                    circle_loss_fn=circle_loss_fn,
                    circle_weight=CIRCLE_WEIGHT,
                )
                scheduler.step()

                record = {
                    "epoch": epoch + 1,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "elapsed_sec": round(time.time() - epoch_start, 2),
                    **{key: float(value) for key, value in train_metrics.items()},
                }

                checkpoint_payload = {
                    "epoch": epoch,
                    "model": (model.module if hasattr(model, "module") else model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_mAP": best_mAP,
                    "config": {
                        "img_size": IMG_SIZE,
                        "epochs": EPOCHS,
                        "batch_size": BATCH_SIZE,
                        "lr": LR,
                    },
                }
                torch.save(checkpoint_payload, last_checkpoint_path)

                if (epoch + 1) % EVAL_EVERY == 0 or epoch == EPOCHS - 1:
                    eval_metrics = evaluate_model(model)
                    record.update(eval_metrics)
                    record["eval"] = True

                    if eval_metrics["mAP"] > best_mAP:
                        best_mAP = eval_metrics["mAP"]
                        checkpoint_payload["best_mAP"] = best_mAP
                        torch.save(checkpoint_payload, best_checkpoint_path)
                        torch.save(checkpoint_payload, CHECKPOINT_DIR / f"epoch_{epoch + 1:03d}.pth")
                else:
                    record["eval"] = False

                history.append(record)
                with history_path.open("w", encoding="utf-8") as handle:
                    json.dump(history, handle, indent=2)

                print(
                    json.dumps(
                        {
                            "epoch": record["epoch"],
                            "loss": round(record["loss"], 4),
                            "id_loss": round(record["id_loss"], 4),
                            "tri_loss": round(record["tri_loss"], 4),
                            "circle_loss": round(record["circle_loss"], 4),
                            "lr": record["lr"],
                            "mAP": record.get("mAP"),
                            "rank1": record.get("rank1"),
                        }
                    )
                )

            assert best_checkpoint_path.exists(), f"Best checkpoint missing: {best_checkpoint_path}"
            print(f"[09f] best checkpoint: {best_checkpoint_path}")
            """,
        ),
        code_cell(
            "09f-final-eval",
            """
            import json

            import torch

            best_checkpoint = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
            target_model = model.module if hasattr(model, "module") else model
            target_model.load_state_dict(best_checkpoint["model"])

            FINAL_METRICS = evaluate_model(model)
            FINAL_METRICS["best_epoch"] = int(best_checkpoint["epoch"]) + 1
            FINAL_METRICS["best_mAP_checkpoint"] = float(best_checkpoint.get("best_mAP", FINAL_METRICS["mAP"]))
            print(json.dumps(FINAL_METRICS, indent=2))
            """,
        ),
        code_cell(
            "09f-artifacts",
            """
            import json
            import shutil

            import torch

            BEST_CHECKPOINT_COPY = Path("/kaggle/working/resnet101ibn_veri776_cityflowv2_384px_best.pth")
            DEPLOY_CHECKPOINT = Path("/kaggle/working/resnet101ibn_veri776_cityflowv2_384px_deploy.pth")
            HISTORY_PATH = Path("/kaggle/working/training_history_09f.json")
            METRICS_PATH = Path("/kaggle/working/metrics_09f.json")

            shutil.copy2(best_checkpoint_path, BEST_CHECKPOINT_COPY)
            torch.save({"state_dict": best_checkpoint["model"]}, DEPLOY_CHECKPOINT)

            metrics_payload = {
                "pretrained_path": str(PRETRAINED_PATH),
                "loss_recipe": LOSS_RECIPE,
                "split_stats": SPLIT_STATS,
                "load_summary": LOAD_SUMMARY,
                "baseline_metrics": BASELINE_METRICS,
                "final_metrics": FINAL_METRICS,
            }
            with METRICS_PATH.open("w", encoding="utf-8") as handle:
                json.dump(metrics_payload, handle, indent=2)

            print(
                json.dumps(
                    {
                        "best_checkpoint": str(BEST_CHECKPOINT_COPY),
                        "deploy_checkpoint": str(DEPLOY_CHECKPOINT),
                        "training_history": str(HISTORY_PATH),
                        "metrics": str(METRICS_PATH),
                        "debug_log": "/kaggle/working/debug.log",
                    },
                    indent=2,
                )
            )
            """,
        ),
        code_cell(
            "09f-validate-ckpt",
            """
            import json

            import torch

            deploy_payload = torch.load(DEPLOY_CHECKPOINT, map_location="cpu", weights_only=False)
            state_dict = deploy_payload["state_dict"]

            required_prefixes = [
                "backbone.conv1",
                "backbone.layer1",
                "backbone.layer2",
                "backbone.layer3",
                "backbone.layer4",
                "pool.p",
                "bottleneck",
                "classifier",
            ]
            prefix_hits = {
                prefix: any(key.startswith(prefix) for key in state_dict)
                for prefix in required_prefixes
            }
            assert all(prefix_hits.values()), prefix_hits

            print(
                json.dumps(
                    {
                        "parameter_tensors": len(state_dict),
                        "prefix_hits": prefix_hits,
                        "sample_keys": list(state_dict.keys())[:12],
                    },
                    indent=2,
                )
            )
            """,
        ),
    ]


def build_notebook() -> dict:
    return {
        "cells": build_cells(),
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_kernel_metadata() -> dict:
    return {
        "id": "ali369/09f-vehicle-reid-resnet101-ibn-a-cityflowv2-finetune",
        "title": "09f Vehicle ReID ResNet101-IBN-a CityFlowV2 Fine-tune (VeRi-776 Pretrained)",
        "code_file": "09f_vehicle_reid_resnet101ibn_cityflowv2.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "keywords": [],
        "dataset_sources": [
            "thanhnguyenle/data-aicity-2023-track-2",
            "mrkdagods/mtmc-weights",
            "mrkdagods/mtmc-gp",
        ],
        "kernel_sources": [
            "ali369/09e-vehicle-reid-resnet101-ibn-a-veri-776-pretrain",
        ],
        "competition_sources": [],
        "model_sources": [],
    }


def validate_notebook(notebook_data: dict) -> None:
    nb_node = nbformat.from_dict(notebook_data)
    nbformat.validate(nb_node)


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

    notebook_data = build_notebook()
    kernel_metadata = build_kernel_metadata()
    validate_notebook(notebook_data)

    with NOTEBOOK_PATH.open("w", encoding="utf-8") as handle:
        json.dump(notebook_data, handle, ensure_ascii=True, indent=1)
        handle.write("\n")

    with METADATA_PATH.open("w", encoding="utf-8") as handle:
        json.dump(kernel_metadata, handle, ensure_ascii=True, indent=2)
        handle.write("\n")

    with NOTEBOOK_PATH.open("r", encoding="utf-8") as handle:
        reloaded_notebook = json.load(handle)
    validate_notebook(reloaded_notebook)

    with METADATA_PATH.open("r", encoding="utf-8") as handle:
        reloaded_metadata = json.load(handle)

    summary = {
        "notebook": str(NOTEBOOK_PATH),
        "kernel_metadata": str(METADATA_PATH),
        "cell_count": len(reloaded_notebook["cells"]),
        "dataset_sources": reloaded_metadata["dataset_sources"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()