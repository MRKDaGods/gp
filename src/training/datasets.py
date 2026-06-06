"""ReID dataset loaders for Market-1501, VeRi-776, and CityFlowV2.

Handles the standard train/query/gallery splits and provides
PyTorch Dataset and DataLoader with identity-balanced sampling
(PK sampler: P identities x K instances per identity per batch).
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler

import torchvision.transforms as T

from src.training.seed import seed_worker, make_generator


# Dataset parsers


def parse_market1501(root: str) -> Tuple[List, List, List]:
    """Parse Market-1501 dataset.

    Filename format: XXXX_cYsZ_NNNNNN_NN.jpg
      XXXX = person_id (-1 = junk, 0 = distractor)
      Y = camera_id (1-6)
    """
    train, query, gallery = [], [], []

    for split_name, split_list in [
        ("bounding_box_train", train),
        ("query", query),
        ("bounding_box_test", gallery),
    ]:
        split_dir = os.path.join(root, split_name)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Market-1501 split not found: {split_dir}")

        for fname in sorted(os.listdir(split_dir)):
            if not fname.endswith(".jpg"):
                continue
            pid = int(fname.split("_")[0])
            if pid < 0:  # junk images
                continue
            cam = int(fname.split("_")[1][1]) - 1  # 0-indexed
            img_path = os.path.join(split_dir, fname)
            split_list.append((img_path, pid, cam))

    # Re-label train pids to 0..N-1
    train_pids = sorted(set(pid for _, pid, _ in train))
    pid2label = {pid: label for label, pid in enumerate(train_pids)}
    train = [(path, pid2label[pid], cam) for path, pid, cam in train]

    return train, query, gallery


def parse_veri776(root: str) -> Tuple[List, List, List]:
    """Parse VeRi-776 dataset.

    Filename format: XXXX_cYYY_NNNNNNNN_N.jpg
      XXXX = vehicle_id
      YYY = camera_id
    """
    train, query, gallery = [], [], []

    for split_name, split_list in [
        ("image_train", train),
        ("image_query", query),
        ("image_test", gallery),
    ]:
        split_dir = os.path.join(root, split_name)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"VeRi-776 split not found: {split_dir}")

        for fname in sorted(os.listdir(split_dir)):
            if not fname.endswith(".jpg"):
                continue
            parts = fname.split("_")
            pid = int(parts[0])
            cam = int(parts[1][1:]) - 1  # strip 'c', 0-index
            img_path = os.path.join(split_dir, fname)
            split_list.append((img_path, pid, cam))

    # Re-label train pids
    train_pids = sorted(set(pid for _, pid, _ in train))
    pid2label = {pid: label for label, pid in enumerate(train_pids)}
    train = [(path, pid2label[pid], cam) for path, pid, cam in train]

    return train, query, gallery


def parse_cityflowv2(root: str) -> Tuple[List, List, List]:
    """Parse CityFlowV2 ReID crops.

    Expected structure (created by scripts/extract_cityflowv2_crops.py):
        root/
          train/   XXXX_SCENE_cNNN_fFFFFFF.jpg
          query/   XXXX_SCENE_cNNN_fFFFFFF.jpg
          gallery/ XXXX_SCENE_cNNN_fFFFFFF.jpg

    Filename: {vehicle_id:04d}_{scene}_{camera}_f{frame:06d}.jpg
    Camera ID is extracted as scene_camera (e.g. S01_c001).
    """
    train, query, gallery = [], [], []

    for split_name, split_list in [
        ("train", train),
        ("query", query),
        ("gallery", gallery),
    ]:
        split_dir = os.path.join(root, split_name)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"CityFlowV2 ReID split not found: {split_dir}")

        for fname in sorted(os.listdir(split_dir)):
            if not fname.endswith(".jpg"):
                continue
            # Format: 0042_S01_c001_f000123.jpg
            parts = fname.split("_")
            if len(parts) < 4:
                continue
            pid = int(parts[0])
            cam_name = parts[1] + "_" + parts[2]  # e.g. S01_c001
            img_path = os.path.join(split_dir, fname)
            split_list.append((img_path, pid, cam_name))

    # Map camera names to integer IDs
    all_cams = sorted({cam for _, _, cam in train + query + gallery})
    cam2id = {c: i for i, c in enumerate(all_cams)}
    train = [(p, pid, cam2id[c]) for p, pid, c in train]
    query = [(p, pid, cam2id[c]) for p, pid, c in query]
    gallery = [(p, pid, cam2id[c]) for p, pid, c in gallery]

    # Re-label train pids to 0..N-1
    train_pids = sorted(set(pid for _, pid, _ in train))
    pid2label = {pid: label for label, pid in enumerate(train_pids)}
    train = [(path, pid2label[pid], cam) for path, pid, cam in train]

    return train, query, gallery


DATASET_PARSERS = {
    "market1501": parse_market1501,
    "veri776": parse_veri776,
    "cityflowv2": parse_cityflowv2,
}


# Transforms


def build_train_transforms(
    height: int = 256,
    width: int = 128,
    random_erasing_prob: float = 0.5,
    color_jitter: bool = False,
) -> T.Compose:
    """Build training augmentation pipeline (BoT recipe)."""
    transforms_list = [
        T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
    ]
    if color_jitter:
        transforms_list.append(
            T.ColorJitter(
                brightness=0.2,
                contrast=0.15,
                saturation=0.1,
                hue=0.05,
            )
        )
    transforms_list.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=random_erasing_prob, value="random"),
    ])
    return T.Compose(transforms_list)


def build_test_transforms(height: int = 256, width: int = 128) -> T.Compose:
    """Build test/eval transforms."""
    return T.Compose([
        T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# Dataset


class ReIDDataset(Dataset):
    """Generic ReID dataset."""

    def __init__(
        self,
        data: List[Tuple[str, int, int]],
        transform: Optional[T.Compose] = None,
    ):
        self.data = data
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        img_path, pid, cam = self.data[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, cam, img_path


# PK Sampler


class PKSampler(Sampler):
    """PK sampler: P identities x K instances per identity per batch.

    Used for triplet loss training to ensure meaningful positive/negative
    pairs exist in each batch.
    """

    def __init__(
        self,
        data_source: List[Tuple[str, int, int]],
        p: int = 16,
        k: int = 4,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.p = p
        self.k = k
        # Dedicated RNG so identity sampling is reproducible (and independent of
        # the global numpy state). When seed is None, falls back to fresh entropy.
        self.rng = np.random.default_rng(seed)

        # Build pid -> indices mapping
        self.pid_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, (_, pid, _) in enumerate(data_source):
            self.pid_to_indices[pid].append(idx)

        self.pids = list(self.pid_to_indices.keys())
        self.batch_size = p * k

    def __iter__(self):
        """Yield indices for PK-balanced batches."""
        self.rng.shuffle(self.pids)
        batch = []
        for pid in self.pids:
            indices = self.pid_to_indices[pid]
            if len(indices) < self.k:
                # Over-sample if not enough instances
                selected = self.rng.choice(indices, size=self.k, replace=True).tolist()
            else:
                selected = self.rng.choice(indices, size=self.k, replace=False).tolist()
            batch.extend(selected)

            if len(batch) >= self.batch_size:
                yield from batch[: self.batch_size]
                batch = batch[self.batch_size :]

        # Remaining partial batch
        if batch:
            yield from batch

    def __len__(self) -> int:
        return len(self.pids) * self.k


def build_dataloader(
    dataset_name: str,
    root: str,
    height: int = 256,
    width: int = 128,
    batch_size: int = 64,
    num_instances: int = 4,
    num_workers: int = 4,
    random_erasing_prob: float = 0.5,
    color_jitter: bool = False,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
    """Build train/query/gallery dataloaders.

    Returns:
        (train_loader, query_loader, gallery_loader, num_classes, num_cameras)
    """
    parser = DATASET_PARSERS.get(dataset_name)
    if parser is None:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_PARSERS.keys())}"
        )

    train_data, query_data, gallery_data = parser(root)

    num_classes = len(set(pid for _, pid, _ in train_data))
    num_cameras = len(set(cam for _, _, cam in train_data))

    train_transform = build_train_transforms(
        height,
        width,
        random_erasing_prob,
        color_jitter,
    )
    test_transform = build_test_transforms(height, width)

    train_dataset = ReIDDataset(train_data, train_transform)
    query_dataset = ReIDDataset(query_data, test_transform)
    gallery_dataset = ReIDDataset(gallery_data, test_transform)

    p = batch_size // num_instances
    sampler = PKSampler(train_data, p=p, k=num_instances, seed=seed)

    # Reproducible worker RNG + generator (only meaningful when a seed is given).
    worker_init = seed_worker if seed is not None else None
    generator = make_generator(seed) if seed is not None else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init,
        generator=generator,
    )
    query_loader = DataLoader(
        query_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init,
    )
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init,
    )

    return train_loader, query_loader, gallery_loader, num_classes, num_cameras
