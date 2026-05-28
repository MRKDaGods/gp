"""Generate qualitative retrieval panels for the VeRi-776 paper.

Inputs:
    experiments/veri776-paper/<exp_id>/features/stream1/{query,gallery}.npy
    experiments/veri776-paper/<exp_id>/features/stream2/{query,gallery}.npy
    experiments/veri776-paper/<exp_id>/features/index_map.json

Outputs:
    figures/paper/retrieval/panel_{1..6}.pdf
    figures/paper/retrieval/panels_log.json

Reproduction:
    python scripts/paper/generate_retrieval_panels.py --exp-id A5alpha

Use --dry-run to validate CLI parsing and post-processing import paths without
requiring feature files. If feature files are not present yet, the script exits
successfully with a Wave-2 reminder so it can be committed during Wave 1.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.stage4_association.query_expansion import average_query_expansion_batched
from src.training.evaluate_reid import compute_reranking

SEED = 20260528
DEFAULT_EXP_ID = "A5alpha"
FUSION_WEIGHTS = {"stream1": 0.3, "stream2": 0.7}
PANEL_SPECS = [
    ("success_fusion_corrects_stream", "Fusion corrects a stream error"),
    ("success_frontal_rear", "Frontal-to-rear viewpoint transition"),
    ("success_illumination", "Illumination shift handled correctly"),
    ("failure_same_make_model_color", "Same make/model/color confound"),
    ("failure_occlusion", "Extreme occlusion or small visible area"),
    ("failure_low_res_gallery", "Low-resolution gallery ambiguity"),
]


@dataclass(frozen=True)
class Item:
    row: int
    image_path: str
    vehicle_id: int
    camera_id: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Selection:
    category: str
    caption: str
    query_row: int
    interpretation: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-id", default=DEFAULT_EXP_ID, help="Experiment id under experiments/veri776-paper/.")
    parser.add_argument("--dry-run", action="store_true", help="Validate CLI and import paths only.")
    parser.add_argument("--w1", type=float, default=FUSION_WEIGHTS["stream1"], help="Stream-1 fusion weight.")
    parser.add_argument("--w2", type=float, default=FUSION_WEIGHTS["stream2"], help="Stream-2 fusion weight.")
    return parser.parse_args()


def feature_paths(exp_id: str) -> dict[str, Path]:
    root = REPO_ROOT / "experiments" / "veri776-paper" / exp_id / "features"
    return {
        "root": root,
        "stream1_query": root / "stream1" / "query.npy",
        "stream1_gallery": root / "stream1" / "gallery.npy",
        "stream2_query": root / "stream2" / "query.npy",
        "stream2_gallery": root / "stream2" / "gallery.npy",
        "index_map": root / "index_map.json",
    }


def missing_inputs(paths: dict[str, Path]) -> list[Path]:
    required = [paths[k] for k in ("stream1_query", "stream1_gallery", "stream2_query", "stream2_gallery", "index_map")]
    return [path for path in required if not path.exists()]


def l2_normalize(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-12)


def apply_aqe(query: np.ndarray, gallery: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    all_features = l2_normalize(np.concatenate([query, gallery], axis=0))
    similarity = all_features @ all_features.T
    indices = np.argsort(-similarity, axis=1)
    expanded = average_query_expansion_batched(all_features, indices, k=k, alpha=1.0)
    return expanded[: query.shape[0]], expanded[query.shape[0] :]


def postprocessed_similarity(
    query: np.ndarray,
    gallery: np.ndarray,
    *,
    aqe_k: int,
    rerank_k1: int,
    rerank_k2: int,
    rerank_lambda: float,
) -> np.ndarray:
    query_aqe, gallery_aqe = apply_aqe(query, gallery, k=aqe_k)
    distance = compute_reranking(
        query_aqe,
        gallery_aqe,
        k1=rerank_k1,
        k2=rerank_k2,
        lambda_value=rerank_lambda,
    )
    return 1.0 - distance


def load_index_map(path: Path) -> tuple[list[Item], list[Item]]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    def convert(records: Iterable[dict[str, Any]]) -> list[Item]:
        items = []
        for record in records:
            metadata = dict(record)
            items.append(
                Item(
                    row=int(record["row"]),
                    image_path=str(record.get("image_path", "")),
                    vehicle_id=int(record["vehicle_id"]),
                    camera_id=int(record["camera_id"]),
                    metadata=metadata,
                )
            )
        return items

    return convert(payload["query"]), convert(payload["gallery"])


def valid_ranking(similarity: np.ndarray, queries: list[Item], galleries: list[Item]) -> np.ndarray:
    ranking = np.argsort(-similarity, axis=1)
    filtered = np.empty_like(ranking)
    for q_idx, query in enumerate(queries):
        order = []
        for gallery_idx in ranking[q_idx]:
            gallery = galleries[int(gallery_idx)]
            if gallery.vehicle_id == query.vehicle_id and gallery.camera_id == query.camera_id:
                continue
            order.append(int(gallery_idx))
        if len(order) < ranking.shape[1]:
            order.extend([idx for idx in ranking[q_idx].tolist() if idx not in order])
        filtered[q_idx] = np.asarray(order[: ranking.shape[1]], dtype=ranking.dtype)
    return filtered


def hit_flags(ranking: np.ndarray, queries: list[Item], galleries: list[Item]) -> np.ndarray:
    return np.asarray([galleries[int(order[0])].vehicle_id == queries[q_idx].vehicle_id for q_idx, order in enumerate(ranking)])


def image_size(path: str) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.size
    except Exception:
        return None


def brightness(path: str) -> float | None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            array = np.asarray(image.convert("HSV"), dtype=np.float32)
        return float(array[..., 2].mean() / 255.0)
    except Exception:
        return None


def area_or_default(path: str, default: int) -> int:
    size = image_size(path)
    if size is None:
        return default
    return int(size[0] * size[1])


def same_vehicle_attributes(query: Item, gallery: Item) -> bool:
    keys = ("model", "model_str", "make", "make_str", "color", "colour", "color_str", "colour_str")
    compared = [key for key in keys if key in query.metadata and key in gallery.metadata]
    return bool(compared) and all(query.metadata[key] == gallery.metadata[key] for key in compared)


def first_unused(candidates: Iterable[int], used: set[int]) -> int | None:
    for candidate in candidates:
        if int(candidate) not in used:
            return int(candidate)
    return None


def choose_panels(
    queries: list[Item],
    galleries: list[Item],
    stream1_sim: np.ndarray,
    stream2_sim: np.ndarray,
    fusion_sim: np.ndarray,
) -> list[Selection]:
    rng = np.random.default_rng(SEED)
    stream1_rank = valid_ranking(stream1_sim, queries, galleries)
    stream2_rank = valid_ranking(stream2_sim, queries, galleries)
    fusion_rank = valid_ranking(fusion_sim, queries, galleries)
    stream1_hit = hit_flags(stream1_rank, queries, galleries)
    stream2_hit = hit_flags(stream2_rank, queries, galleries)
    fusion_hit = hit_flags(fusion_rank, queries, galleries)
    used: set[int] = set()
    selections: list[Selection] = []

    all_rows = np.arange(len(queries))
    success_rows = all_rows[fusion_hit]
    failure_rows = all_rows[~fusion_hit]

    def pick(candidates: Iterable[int], fallback: Iterable[int]) -> int:
        row = first_unused(candidates, used)
        if row is not None:
            return row
        row = first_unused(fallback, used)
        if row is not None:
            return row
        return int(rng.choice(all_rows))

    candidates = all_rows[(~stream1_hit) & (~stream2_hit) & fusion_hit]
    if len(candidates):
        margin = fusion_sim[candidates, fusion_rank[candidates, 0]] - np.maximum(
            stream1_sim[candidates, stream1_rank[candidates, 0]],
            stream2_sim[candidates, stream2_rank[candidates, 0]],
        )
        order = candidates[np.argsort(-margin)]
    else:
        order = success_rows
    row = pick(order, success_rows)
    used.add(row)
    selections.append(Selection(PANEL_SPECS[0][0], PANEL_SPECS[0][1], row, "Fusion top-1 is correct where at least one component stream is weaker; falls back to a fusion success if no dual-stream miss exists."))

    candidates = sorted(success_rows.tolist(), key=lambda idx: (-abs(queries[idx].camera_id - galleries[int(fusion_rank[idx, 0])].camera_id), queries[idx].vehicle_id, idx))
    row = pick(candidates, success_rows)
    used.add(row)
    selections.append(Selection(PANEL_SPECS[1][0], PANEL_SPECS[1][1], row, "Selected by maximum query/top-1 camera-index separation as a viewpoint-divergence proxy."))

    illuminated = []
    for idx in success_rows.tolist():
        q_brightness = brightness(queries[idx].image_path)
        g_brightness = brightness(galleries[int(fusion_rank[idx, 0])].image_path)
        if q_brightness is not None and g_brightness is not None and abs(q_brightness - g_brightness) > (30.0 / 255.0):
            illuminated.append(idx)
    row = pick(sorted(illuminated, key=lambda idx: (queries[idx].vehicle_id, idx)), success_rows)
    used.add(row)
    selections.append(Selection(PANEL_SPECS[2][0], PANEL_SPECS[2][1], row, "Selected by HSV-V brightness gap when images are available; otherwise falls back to a deterministic fusion success."))

    confounds = [idx for idx in failure_rows.tolist() if same_vehicle_attributes(queries[idx], galleries[int(fusion_rank[idx, 0])])]
    row = pick(sorted(confounds, key=lambda idx: (queries[idx].vehicle_id, idx)), failure_rows)
    used.add(row)
    selections.append(Selection(PANEL_SPECS[3][0], PANEL_SPECS[3][1], row, "Selected by matching vehicle metadata when present; otherwise uses a deterministic fusion failure as the confound placeholder."))

    occlusion_order = sorted(failure_rows.tolist(), key=lambda idx: (area_or_default(queries[idx].image_path, 10**12), queries[idx].vehicle_id, idx))
    row = pick(occlusion_order, failure_rows)
    used.add(row)
    selections.append(Selection(PANEL_SPECS[4][0], PANEL_SPECS[4][1], row, "Selected by smallest available query crop area as an occlusion proxy."))

    low_res_order = sorted(
        failure_rows.tolist(),
        key=lambda idx: (
            max(area_or_default(galleries[int(g)].image_path, 10**12) for g in fusion_rank[idx, :5]),
            queries[idx].vehicle_id,
            idx,
        ),
    )
    row = pick(low_res_order, failure_rows)
    selections.append(Selection(PANEL_SPECS[5][0], PANEL_SPECS[5][1], row, "Selected by the smallest maximum top-5 gallery crop area as a low-resolution proxy."))

    return selections


def load_image_for_panel(path: str) -> np.ndarray:
    try:
        from PIL import Image

        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((180, 180))
            canvas = Image.new("RGB", (180, 180), "white")
            x = (180 - image.width) // 2
            y = (180 - image.height) // 2
            canvas.paste(image, (x, y))
            return np.asarray(canvas)
    except Exception:
        placeholder = np.full((180, 180, 3), 245, dtype=np.uint8)
        placeholder[::12, :, :] = 220
        placeholder[:, ::12, :] = 220
        return placeholder


def render_panel(
    output_path: Path,
    query: Item,
    galleries: list[Item],
    gt_flags: list[bool],
    caption: str,
    failure: bool,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    fig, axes = plt.subplots(1, 6, figsize=(10.8, 2.45))
    panel_items = [query] + galleries
    labels = ["Query"] + [f"G{i}" for i in range(1, 6)]
    for idx, (axis, item, label) in enumerate(zip(axes, panel_items, labels)):
        axis.imshow(load_image_for_panel(item.image_path))
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(label, fontsize=8)
        color = "#333333"
        width = 1.2
        if idx > 0 and gt_flags[idx - 1]:
            color = "#1b9e3e"
            width = 3.0
        elif failure and idx == 1 and not gt_flags[0]:
            color = "#c62828"
            width = 3.0
        for spine in axis.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(width)
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def generate(exp_id: str, w1: float, w2: float) -> None:
    paths = feature_paths(exp_id)
    missing = missing_inputs(paths)
    if missing:
        print(f"Features for exp_id {exp_id} not yet produced by Wave 2 -- re-run after Wave 2 completes")
        for path in missing:
            print(f"missing: {path.relative_to(REPO_ROOT)}")
        return

    stream1_query = np.load(paths["stream1_query"])
    stream1_gallery = np.load(paths["stream1_gallery"])
    stream2_query = np.load(paths["stream2_query"])
    stream2_gallery = np.load(paths["stream2_gallery"])
    queries, galleries = load_index_map(paths["index_map"])

    stream1_sim = postprocessed_similarity(stream1_query, stream1_gallery, aqe_k=3, rerank_k1=80, rerank_k2=15, rerank_lambda=0.2)
    stream2_sim = postprocessed_similarity(stream2_query, stream2_gallery, aqe_k=10, rerank_k1=50, rerank_k2=10, rerank_lambda=0.1)
    fusion_sim = (w1 * stream1_sim) + (w2 * stream2_sim)
    fusion_rank = valid_ranking(fusion_sim, queries, galleries)
    selections = choose_panels(queries, galleries, stream1_sim, stream2_sim, fusion_sim)

    output_dir = REPO_ROOT / "figures" / "paper" / "retrieval"
    log_entries = []
    for panel_id, selection in enumerate(selections, start=1):
        query = queries[selection.query_row]
        gallery_rows = [int(row) for row in fusion_rank[selection.query_row, :5]]
        gallery_items = [galleries[row] for row in gallery_rows]
        gt_flags = [gallery.vehicle_id == query.vehicle_id for gallery in gallery_items]
        caption = f"{selection.caption}: {selection.interpretation}"
        render_panel(output_dir / f"panel_{panel_id}.pdf", query, gallery_items, gt_flags, caption, failure=selection.category.startswith("failure"))
        log_entries.append(
            {
                "panel_id": panel_id,
                "category": selection.category,
                "query_id": query.vehicle_id,
                "query_row": query.row,
                "query_image_path": query.image_path,
                "gallery_ids_shown": [gallery.vehicle_id for gallery in gallery_items],
                "gallery_rows_shown": gallery_rows,
                "gt_in_top5": bool(any(gt_flags)),
                "interpretation": selection.interpretation,
                "exp_id": exp_id,
                "fusion_weights": {"stream1": w1, "stream2": w2},
            }
        )

    (output_dir / "panels_log.json").write_text(json.dumps(log_entries, indent=2), encoding="utf-8")
    print(f"Wrote {len(log_entries)} retrieval panels to {output_dir.relative_to(REPO_ROOT)}")


def main() -> int:
    args = parse_args()
    if args.dry_run:
        _ = average_query_expansion_batched
        _ = compute_reranking
        print(f"dry-run ok: exp_id={args.exp_id}, weights=({args.w1}, {args.w2}), seed={SEED}")
        return 0
    generate(args.exp_id, args.w1, args.w2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())