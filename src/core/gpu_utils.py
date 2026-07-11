"""GPU capability helpers for choosing precision at runtime."""
from __future__ import annotations

from loguru import logger

_fp16_notice_logged = False


def _device_index(device: str) -> int:
    d = str(device).strip().lower()
    if ":" in d:
        try:
            return int(d.split(":")[1])
        except ValueError:
            return 0
    if d.isdigit():
        return int(d)
    return 0


def supports_fast_fp16(device: str) -> bool:
    """True only where half precision is genuinely fast: Volta+ (compute >= 7.0).

    Pascal / Maxwell consumer cards (GTX 9xx / 10xx, e.g. the 1050 Ti at CC 6.1)
    run FP16 at ~1/64 the FP32 rate, so `half=True` is a slowdown, not a speedup.
    """
    d = str(device).strip().lower()
    if not d or "cpu" in d:
        return False
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        return torch.cuda.get_device_capability(_device_index(device))[0] >= 7
    except Exception:
        return False


def effective_half(requested: bool, device: str) -> bool:
    """Honor a half=True request only on GPUs with fast FP16; else use FP32."""
    global _fp16_notice_logged
    if not requested:
        return False
    if supports_fast_fp16(device):
        return True
    if not _fp16_notice_logged:
        _fp16_notice_logged = True
        logger.info(
            f"FP16 requested but '{device}' has no fast half-precision (compute < 7.0) - "
            "using FP32, which is faster on Pascal/Maxwell GPUs."
        )
    return False
