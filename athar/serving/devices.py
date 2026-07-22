"""DeviceManager: device inventory + memory budgets for model placement.

Serving keeps N models resident (ReIDRuntime LRU); this class answers the
question the LRU alone cannot: *does this model fit on that device right
now?* Each device has a byte budget (CUDA: detected total VRAM x headroom,
overridable; CPU: unlimited by default) and a reserved counter maintained
by whoever places models. The runtime reserves before building and releases
on eviction, so a burst of concurrent loads can never over-commit VRAM —
the failure is an explicit :class:`DeviceBudgetError` instead of a CUDA OOM
mid-extraction.

Budgets are byte *estimates* (checkpoint size ~= fp32 weight memory); the
default 0.9 headroom absorbs activation memory and allocator slack. Sites
that need tighter control set ``ATHAR_VRAM_BUDGET_MB`` or pass explicit
budgets.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_HEADROOM = 0.9
VRAM_BUDGET_ENV = "ATHAR_VRAM_BUDGET_MB"


class DeviceBudgetError(RuntimeError):
    """The requested placement cannot fit within the device budget even
    after evicting everything evictable. Raise the budget, release leases,
    or place on another device."""


def _env_vram_budget_bytes() -> Optional[int]:
    raw = os.getenv(VRAM_BUDGET_ENV)
    if raw is None:
        return None
    try:
        return max(0, int(float(raw))) * 1024 * 1024
    except ValueError:
        logger.warning("%s=%r is not a number; ignoring", VRAM_BUDGET_ENV, raw)
        return None


class DeviceManager:
    """Tracks per-device byte budgets and reservations. Thread-safe.

    ``budgets`` maps device strings (``"cpu"``, ``"cuda:0"``) to a byte
    budget, or ``None`` for unlimited. Devices not listed are resolved
    lazily: CPU is unlimited; CUDA devices get ``total_memory x headroom``
    (or the ``ATHAR_VRAM_BUDGET_MB`` override).
    """

    def __init__(
        self,
        budgets: Optional[dict[str, Optional[int]]] = None,
        headroom: float = DEFAULT_HEADROOM,
    ) -> None:
        if not 0.0 < headroom <= 1.0:
            raise ValueError(f"headroom must be in (0, 1], got {headroom}")
        self.headroom = headroom
        self._budgets: dict[str, Optional[int]] = dict(budgets or {})
        self._reserved: dict[str, int] = {}
        self._lock = threading.Lock()

    # -- budgets -----------------------------------------------------------
    def _resolve_budget(self, device: str) -> Optional[int]:
        if device in self._budgets:
            return self._budgets[device]
        if device.startswith("cuda"):
            env = _env_vram_budget_bytes()
            if env is not None:
                budget: Optional[int] = env
            else:
                budget = self._detect_cuda_budget(device)
        else:
            budget = None  # CPU: unlimited unless explicitly budgeted
        self._budgets[device] = budget
        return budget

    def _detect_cuda_budget(self, device: str) -> Optional[int]:
        try:
            import torch

            index = int(device.split(":", 1)[1]) if ":" in device else 0
            total = torch.cuda.get_device_properties(index).total_memory
            return int(total * self.headroom)
        except Exception as exc:  # noqa: BLE001 — budget detection is advisory
            logger.warning("cannot detect VRAM budget for %s (%s); unlimited", device, exc)
            return None

    def budget(self, device: str) -> Optional[int]:
        with self._lock:
            return self._resolve_budget(device)

    def reserved(self, device: str) -> int:
        with self._lock:
            return self._reserved.get(device, 0)

    # -- reservations --------------------------------------------------------
    def can_fit(self, device: str, nbytes: int) -> bool:
        with self._lock:
            budget = self._resolve_budget(device)
            if budget is None:
                return True
            return self._reserved.get(device, 0) + nbytes <= budget

    def reserve(self, device: str, nbytes: int) -> None:
        """Record a placement. Callers check :meth:`can_fit` first (under
        their own coordination lock); reserve itself never refuses."""
        if nbytes < 0:
            raise ValueError("nbytes must be >= 0")
        with self._lock:
            self._resolve_budget(device)
            self._reserved[device] = self._reserved.get(device, 0) + nbytes

    def release(self, device: str, nbytes: int) -> None:
        with self._lock:
            current = self._reserved.get(device, 0)
            if nbytes > current:
                logger.warning(
                    "release of %d bytes on %s exceeds reserved %d; clamping",
                    nbytes, device, current,
                )
                nbytes = current
            self._reserved[device] = current - nbytes

    def snapshot(self) -> dict:
        with self._lock:
            return {
                device: {
                    "budget_bytes": self._budgets.get(device),
                    "reserved_bytes": self._reserved.get(device, 0),
                }
                for device in sorted(set(self._budgets) | set(self._reserved))
            }
