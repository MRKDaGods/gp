"""Re-pool Stage 2 multi-query embeddings with robust aggregators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.stage2_features.robust_pool import (
    DEFAULT_MIN_K,
    aggregate_embedding_matrix,
    available_modes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=tuple(available_modes()), required=True)
    parser.add_argument("--min-k", type=int, default=DEFAULT_MIN_K)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--no-trim-padding", action="store_true")
    return parser.parse_args()


def repool_stage2(
    stage2_dir: Path,
    mode: str,
    min_k: int = DEFAULT_MIN_K,
    output: Path | None = None,
    summary: Path | None = None,
    trim_padding: bool = True,
) -> dict:
    stage2_dir = Path(stage2_dir)
    mq_path = stage2_dir / "multi_query_embeddings.npz"
    fallback_path = stage2_dir / "embeddings.npy"
    if not mq_path.exists():
        raise FileNotFoundError(mq_path)
    if not fallback_path.exists():
        raise FileNotFoundError(fallback_path)

    with np.load(mq_path) as data:
        if "embeddings" not in data:
            raise KeyError(f"{mq_path} does not contain an 'embeddings' array")
        multi_query_embeddings = data["embeddings"].astype(np.float32)

    fallback_embeddings = np.load(fallback_path).astype(np.float32)
    pooled, fallback_count = aggregate_embedding_matrix(
        multi_query_embeddings,
        mode=mode,
        fallback_embeddings=fallback_embeddings,
        min_k=min_k,
        trim_padding=trim_padding,
    )

    output = output or (stage2_dir / f"embeddings_{mode}.npy")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, pooled.astype(np.float32))

    norms = np.linalg.norm(pooled, axis=1)
    payload = {
        "mode": mode,
        "min_k": int(min_k),
        "stage2_dir": str(stage2_dir),
        "multi_query_path": str(mq_path),
        "fallback_path": str(fallback_path),
        "output": str(output),
        "input_shape": list(multi_query_embeddings.shape),
        "output_shape": list(pooled.shape),
        "fallback_count": int(fallback_count),
        "trim_padding": bool(trim_padding),
        "norm_min": float(norms.min()) if norms.size else None,
        "norm_max": float(norms.max()) if norms.size else None,
    }

    if summary is not None:
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload


def main() -> None:
    args = parse_args()
    payload = repool_stage2(
        stage2_dir=args.stage2_dir,
        mode=args.mode,
        min_k=args.min_k,
        output=args.output,
        summary=args.summary,
        trim_padding=not args.no_trim_padding,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
