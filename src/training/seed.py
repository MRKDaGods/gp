"""Reproducibility helpers: global RNG seeding + deterministic DataLoader workers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 3407, deterministic: bool = True) -> int:
    """Seed all RNGs (python, numpy, torch CPU+CUDA) for reproducible training."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    return seed


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: re-seed numpy/random in each worker process."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = 3407) -> torch.Generator:
    """Return a torch.Generator seeded for use as DataLoader(generator=...)."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
