"""ReIDRuntime — explicit, thread-safe serving model lifecycle.

Replaces the two structural problems the v1 loaders carried:

- module-global device state (``_DEVICE``/``_PIN_MEMORY``/``_NUM_WORKERS``
  mutated by ``_set_runtime``) — now every extraction call computes its own
  parameters (see ``reid_loaders._loader_params``);
- a fixed ``functools.lru_cache(maxsize=2)`` that read
  ``REID_MODEL_CACHE_SIZE`` but could not honor it — now an actual LRU with
  a configurable capacity, explicit eviction, and stats.

Phase 4 completion adds the placement guarantees serving needs:

- **Refcounted leases** (:meth:`ReIDRuntime.acquire`): a model held by a
  lease is never evicted mid-extraction, no matter what the LRU wants.
  Capacity overshoot with all entries leased is allowed (and logged) —
  breaking a running extraction is worse than a temporary oversubscription
  of the *count*; the *memory* budget below is never overshot.
- **VRAM budget** (:class:`~athar.serving.devices.DeviceManager`): bytes are
  reserved before a build and released on eviction. When a load cannot fit
  even after evicting every unleased model on that device, it fails with
  :class:`~athar.serving.devices.DeviceBudgetError` instead of a CUDA OOM.
- **Single build per key**: concurrent loads of the same (model, device)
  wait for the first builder instead of each deserializing a multi-GB
  checkpoint; different keys still build in parallel (build happens outside
  the lock).
"""

from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

from athar.serving.devices import DeviceBudgetError, DeviceManager

if TYPE_CHECKING:
    from athar.serving.reid_loaders import LoadedReIDModel

logger = logging.getLogger(__name__)

DEFAULT_MAX_MODELS = 2

_CacheKey = tuple[str, str]  # (model_id, normalized device)


def _env_max_models() -> int:
    raw = os.getenv("REID_MODEL_CACHE_SIZE", str(DEFAULT_MAX_MODELS))
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "REID_MODEL_CACHE_SIZE=%r is not an int; using %d", raw, DEFAULT_MAX_MODELS
        )
        return DEFAULT_MAX_MODELS


def _default_size_estimator(model_id: str) -> int:
    """Checkpoint file size ~= resident fp32 weight bytes. Activation and
    allocator overhead are covered by the DeviceManager headroom. Unknown
    models estimate 0 (budget-neutral) rather than blocking the load."""
    try:
        from athar.serving.reid_loaders import _primary_checkpoint, _registry_entry

        return _primary_checkpoint(_registry_entry(model_id)).stat().st_size
    except Exception as exc:  # noqa: BLE001 — estimation is advisory
        logger.warning("cannot estimate size of %s (%s); assuming 0 bytes", model_id, exc)
        return 0


@dataclass
class _Entry:
    loaded: "LoadedReIDModel"
    size_bytes: int
    refcount: int = 0


class ModelLease:
    """Holds a model resident until released. Use as a context manager or
    call :meth:`release` explicitly; releasing twice is a no-op."""

    def __init__(self, runtime: "ReIDRuntime", key: _CacheKey, loaded: "LoadedReIDModel"):
        self._runtime = runtime
        self._key = key
        self.model = loaded
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._runtime._release_lease(self._key)

    def __enter__(self) -> "ModelLease":
        return self

    def __exit__(self, *exc_info) -> None:
        self.release()


class ReIDRuntime:
    """Refcounted LRU of loaded serving models, keyed by (model_id, device)."""

    def __init__(
        self,
        max_models: Optional[int] = None,
        builder: Optional[Callable[[str, str], "LoadedReIDModel"]] = None,
        devices: Optional[DeviceManager] = None,
        size_estimator: Optional[Callable[[str], int]] = None,
    ) -> None:
        self.max_models = max_models if max_models is not None else _env_max_models()
        if self.max_models < 1:
            raise ValueError(f"max_models must be >= 1, got {self.max_models}")
        self._builder = builder
        self.devices = devices if devices is not None else DeviceManager()
        self._estimate = size_estimator or _default_size_estimator
        self._cache: "OrderedDict[_CacheKey, _Entry]" = OrderedDict()
        self._building: dict[_CacheKey, threading.Event] = {}
        self._lock = threading.RLock()

    def _build(self, model_id: str, device: str) -> "LoadedReIDModel":
        if self._builder is not None:
            return self._builder(model_id, device)
        from athar.serving.reid_loaders import _build_loaded_model

        return _build_loaded_model(model_id, device)

    # -- leases --------------------------------------------------------------
    def acquire(self, model_id: str, device: str) -> ModelLease:
        """Load (or hit) and pin the model until the lease is released."""
        from athar.serving.reid_loaders import _normalise_device

        key = (model_id, _normalise_device(device))
        while True:
            with self._lock:
                entry = self._cache.get(key)
                if entry is not None:
                    self._cache.move_to_end(key)
                    entry.refcount += 1
                    return ModelLease(self, key, entry.loaded)
                waiter = self._building.get(key)
                if waiter is None:
                    size = max(0, int(self._estimate(model_id)))
                    evicted = self._make_room(key[1], size)  # DeviceBudgetError on no fit
                    self.devices.reserve(key[1], size)
                    self._building[key] = threading.Event()
                    break
            # Another thread is building this exact key: wait, then re-check.
            waiter.wait()
        self._log_evictions(evicted)
        # Build OUTSIDE the lock: checkpoint loads take seconds and must not
        # serialize unrelated cache hits or other builds.
        try:
            built = self._build(model_id, key[1])
        except BaseException:
            with self._lock:
                self.devices.release(key[1], size)
                self._building.pop(key).set()
            raise
        with self._lock:
            self._cache[key] = _Entry(loaded=built, size_bytes=size, refcount=1)
            self._building.pop(key).set()
            evicted = self._enforce_capacity()
        self._log_evictions(evicted)
        return ModelLease(self, key, built)

    def _release_lease(self, key: _CacheKey) -> None:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return  # already evicted via clear() after forced release
            entry.refcount = max(0, entry.refcount - 1)
            evicted = self._enforce_capacity() if len(self._cache) > self.max_models else []
        self._log_evictions(evicted)

    # -- back-compat load ------------------------------------------------------
    def load(self, model_id: str, device: str) -> "LoadedReIDModel":
        """Unleased load: cached and evictable. Prefer :meth:`acquire` in
        code that extracts features — an eviction between ``load`` and use
        cannot be prevented without a lease."""
        lease = self.acquire(model_id, device)
        lease.release()
        return lease.model

    def extract(self, loaded: "LoadedReIDModel", images, batch_size: int = 32):
        from athar.serving.reid_loaders import extract_features

        return extract_features(loaded, images, batch_size=batch_size)

    # -- eviction (call with lock held; caller logs + releases CUDA) ----------
    def _evict(self, key: _CacheKey) -> _Entry:
        entry = self._cache.pop(key)
        self.devices.release(key[1], entry.size_bytes)
        return entry

    def _make_room(self, device: str, size: int) -> list[_CacheKey]:
        """Evict unleased LRU entries on ``device`` until ``size`` fits."""
        evicted: list[_CacheKey] = []
        while not self.devices.can_fit(device, size):
            victim = next(
                (k for k, e in self._cache.items() if k[1] == device and e.refcount == 0),
                None,
            )
            if victim is None:
                raise DeviceBudgetError(
                    f"cannot fit {size} bytes on {device}: budget "
                    f"{self.devices.budget(device)}, reserved {self.devices.reserved(device)}, "
                    f"and every resident model is leased"
                )
            self._evict(victim)
            evicted.append(victim)
        return evicted

    def _enforce_capacity(self) -> list[_CacheKey]:
        evicted: list[_CacheKey] = []
        while len(self._cache) > self.max_models:
            victim = next((k for k, e in self._cache.items() if e.refcount == 0), None)
            if victim is None:
                logger.warning(
                    "ReIDRuntime over capacity (%d > %d) but every model is leased; "
                    "allowing overshoot", len(self._cache), self.max_models,
                )
                break
            self._evict(victim)
            evicted.append(victim)
        return evicted

    def _log_evictions(self, evicted: list[_CacheKey]) -> None:
        for model_id, device in evicted:
            logger.info("ReIDRuntime evicted %s (%s)", model_id, device)
        if evicted:
            self._release_cuda()

    def clear(self) -> None:
        """Evict every unleased model; leased models stay (and are logged)."""
        with self._lock:
            victims = [k for k, e in self._cache.items() if e.refcount == 0]
            for key in victims:
                self._evict(key)
            for key, entry in self._cache.items():
                logger.warning(
                    "clear(): %s (%s) still leased %d time(s); keeping",
                    key[0], key[1], entry.refcount,
                )
        self._release_cuda()

    def stats(self) -> dict:
        with self._lock:
            return {
                "max_models": self.max_models,
                "loaded": [
                    {
                        "model_id": m.loaded.model_id,
                        "device": m.loaded.device,
                        "loader": m.loaded.loader,
                        "refcount": m.refcount,
                        "size_bytes": m.size_bytes,
                    }
                    for m in self._cache.values()
                ],
                "devices": self.devices.snapshot(),
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
