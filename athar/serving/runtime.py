"""ReIDRuntime — explicit, thread-safe serving model lifecycle.

Replaces the two structural problems the v1 loaders carried:

- module-global device state (``_DEVICE``/``_PIN_MEMORY``/``_NUM_WORKERS``
  mutated by ``_set_runtime``) — now every extraction call computes its own
  parameters (see ``reid_loaders._loader_params``);
- a fixed ``functools.lru_cache(maxsize=2)`` that read
  ``REID_MODEL_CACHE_SIZE`` but could not honor it — now an actual LRU with
  a configurable capacity, explicit eviction, and stats.

A refcounted VRAM-budgeted DeviceManager is the Phase 4 follow-up; the
seam for it is this class.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from athar.serving.reid_loaders import LoadedReIDModel

logger = logging.getLogger(__name__)

DEFAULT_MAX_MODELS = 2


def _env_max_models() -> int:
    raw = os.getenv("REID_MODEL_CACHE_SIZE", str(DEFAULT_MAX_MODELS))
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "REID_MODEL_CACHE_SIZE=%r is not an int; using %d", raw, DEFAULT_MAX_MODELS
        )
        return DEFAULT_MAX_MODELS


class ReIDRuntime:
    """LRU cache of loaded serving models, keyed by (model_id, device)."""

    def __init__(
        self,
        max_models: Optional[int] = None,
        builder: Optional[Callable[[str, str], "LoadedReIDModel"]] = None,
    ) -> None:
        self.max_models = max_models if max_models is not None else _env_max_models()
        if self.max_models < 1:
            raise ValueError(f"max_models must be >= 1, got {self.max_models}")
        self._builder = builder
        self._cache: "OrderedDict[tuple[str, str], LoadedReIDModel]" = OrderedDict()
        self._lock = threading.RLock()

    def _build(self, model_id: str, device: str) -> "LoadedReIDModel":
        if self._builder is not None:
            return self._builder(model_id, device)
        from athar.serving.reid_loaders import _build_loaded_model

        return _build_loaded_model(model_id, device)

    def load(self, model_id: str, device: str) -> "LoadedReIDModel":
        from athar.serving.reid_loaders import _normalise_device

        key = (model_id, _normalise_device(device))
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached
        # Build OUTSIDE the lock: checkpoint loads take seconds and must not
        # serialize unrelated cache hits. Two threads may race to build the
        # same model; the first insert wins and the loser's copy is dropped.
        built = self._build(model_id, key[1])
        with self._lock:
            winner = self._cache.get(key)
            if winner is not None:
                self._cache.move_to_end(key)
                return winner
            self._cache[key] = built
            evicted = []
            while len(self._cache) > self.max_models:
                _, old = self._cache.popitem(last=False)
                evicted.append(old)
        for old in evicted:
            logger.info("ReIDRuntime evicted %s (%s)", old.model_id, old.device)
        if evicted:
            self._release_cuda()
        return built

    def extract(self, loaded: "LoadedReIDModel", images, batch_size: int = 32):
        from athar.serving.reid_loaders import extract_features

        return extract_features(loaded, images, batch_size=batch_size)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
        self._release_cuda()

    def stats(self) -> dict:
        with self._lock:
            return {
                "max_models": self.max_models,
                "loaded": [
                    {"model_id": m.model_id, "device": m.device, "loader": m.loader}
                    for m in self._cache.values()
                ],
            }

    @staticmethod
    def _release_cuda() -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_default_runtime: Optional[ReIDRuntime] = None
_default_lock = threading.Lock()


def default_runtime() -> ReIDRuntime:
    global _default_runtime
    with _default_lock:
        if _default_runtime is None:
            _default_runtime = ReIDRuntime()
        return _default_runtime


def reset_default_runtime() -> None:
    """Test hook: drop the singleton (does not clear CUDA memory by itself)."""
    global _default_runtime
    with _default_lock:
        _default_runtime = None
