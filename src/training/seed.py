"""Reproducibility helpers: global RNG seeding + deterministic DataLoader workers.

Use this to make ReID training runs reproducible end-to-end:

    from src.training.seed import set_seed, seed_worker, make_generator

    set_seed(3407, deterministic=True)        # call ONCE, before building data/model
    loader = DataLoader(..., worker_init_fn=seed_worker, generator=make_generator(3407))

Notes / caveats:
    - ``deterministic=True`` sets ``cudnn.deterministic=True`` and ``cudnn.benchmark=False``.
      NEVER flip ``cudnn.benchmark=True`` afterwards - that re-enables nondeterministic
      convolution autotuning and silently breaks reproducibility.
    - ``worker_init_fn`` is required because each DataLoader worker is a forked process;
      torch re-seeds each worker's torch RNG deterministically from the base seed, but
      numpy / python ``random`` are NOT re-seeded automatically. seed_worker fixes that.
    - Even fully seeded, exact bit-reproducibility also requires the same library
      versions (torch / torchvision / timm / numpy) and the same GPU/driver - some CUDA
      kernels (atomics, certain convolutions) remain nondeterministic unless you also call
      ``torch.use_deterministic_algorithms(True)`` (which can error on unsupported ops and
      slow training). We default to the practical cudnn-deterministic level.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 3407, deterministic: bool = True) -> int:
    """Seed all RNGs (python, numpy, torch CPU+CUDA) for reproducible training.

    Call this ONCE at the very start of training, before building the dataloaders
    and the model (weight init consumes the torch RNG).

    Args:
        seed: the global seed.
        deterministic: if True, also force cuDNN into deterministic mode
            (cudnn.deterministic=True, cudnn.benchmark=False) and set
            PYTHONHASHSEED. Leave True for reproducible runs; set False only
            if you knowingly want the (faster) nondeterministic cuDNN autotuner.

    Returns:
        The seed that was set (for logging).
    """
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
    """DataLoader ``worker_init_fn``: re-seed numpy/random in each worker process.

    torch already gives each worker a deterministic torch seed derived from the
    base generator; this propagates that to numpy and python ``random`` so any
    numpy-based augmentation/sampling in workers is reproducible too.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = 3407) -> torch.Generator:
    """Return a torch.Generator seeded for use as DataLoader(generator=...)."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
