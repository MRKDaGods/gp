"""Aggregate VeRi-776 paper campaign kernel outputs.

Inputs:
    experiments/veri776-paper/<exp_id>/eval_results.json
    experiments/veri776-paper/<exp_id>/recipe.json      (optional)
    experiments/veri776-paper/<exp_id>/train_log.json   (optional)

Outputs:
    experiments/veri776-paper/results.csv
    experiments/veri776-paper/results.json
    experiments/veri776-paper/REPORT.md

Reproduction:
    python scripts/paper/aggregate_paper_results.py
    python scripts/paper/aggregate_paper_results.py --exp-dir experiments/veri776-paper
    python scripts/paper/aggregate_paper_results.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXP_DIR = REPO_ROOT / "experiments" / "veri776-paper"
EXP_IDS = ["A1", "A2", "A3", "A4", "A5alpha", "A5beta", "S1_seed42", "S2_seed123", "S3_seed456"]
ABLATION_IDS = ["A1", "A2", "A3", "A4", "A5alpha", "A5beta"]
SEED_IDS = ["S1_seed42", "S2_seed123", "S3_seed456"]
SEED_VALUES = [42, 123, 456]
CSV_COLUMNS = [
    "experiment_name",
    "seed",
    "stream1_clip_init",
    "stream1_supcon",
    "stream1_adamw",
    "stream2_fixed",
    "aqe",
    "rerank",
    "tta",
    "fusion",
    "mAP",
    "rank1",
    "notes",
]
REQUIRED_EVAL_KEYS = {
    "exp_id",
    "seed",
    "stream1_clip_init",
    "stream1_supcon",
    "stream1_adamw",
    "stream2_fixed",
    "aqe",
    "rerank",
    "tta",
    "fusion",
    "mAP",
    "rank1",
    "notes",
    "recipe_signature",
}
DATASET_SPLIT = "VeRi-776 standard split (37781/576 train, 1678 query, 11579 gallery / 200 test IDs)"
SEED_VARIANCE_RECIPE = "A5alpha (CE-LS + Triplet + CenterLoss, LLRD=0.75)"
NO_RESULTS_MESSAGE = "No eval_results.json files found yet — Wave 3 not ready to aggregate. Re-run after kernel outputs are downloaded."


@dataclass(frozen=True)
class ExperimentRecord:
    exp_id: str
    status: str
    data: dict[str, Any]
    error: str | None = None
    recipe: dict[str, Any] | None = None
    train_summary: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", type=Path, default=DEFAULT_EXP_DIR, help="Directory containing per-experiment outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Validate CLI/imports and iterate exp_ids without writing files.")
    return parser.parse_args()


def run_git(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_number(value: Any, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def validate_eval_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("eval_results.json root must be an object")
    missing = sorted(REQUIRED_EVAL_KEYS - set(payload))
    if missing:
        raise ValueError(f"missing required keys: {', '.join(missing)}")

    validated = dict(payload)
    for key in (
        "stream1_clip_init",
        "stream1_supcon",
        "stream1_adamw",
        "stream2_fixed",
        "aqe",
        "rerank",
        "tta",
        "fusion",
    ):
        if not isinstance(validated[key], bool):
            raise ValueError(f"{key} must be boolean")
    validated["mAP"] = require_number(validated["mAP"], "mAP")
    validated["rank1"] = require_number(validated["rank1"], "rank1")
    if not isinstance(validated["notes"], str):
        raise ValueError("notes must be a string")
    if not isinstance(validated["recipe_signature"], str):
        raise ValueError("recipe_signature must be a string")
    return validated


def optional_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = load_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def summarize_train_log(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        payload = load_json(path)
    except (OSError, json.JSONDecodeError):
        return None

    final_epoch = None
    training_time = None
    if isinstance(payload, dict):
        final_epoch = payload.get("final_epoch") or payload.get("epoch") or payload.get("epochs_completed")
        training_time = payload.get("training_time") or payload.get("training_time_sec") or payload.get("wall_time_sec")
        history = payload.get("history") or payload.get("epochs") or payload.get("log")
        if final_epoch is None and isinstance(history, list) and history:
            last = history[-1]
            if isinstance(last, dict):
                final_epoch = last.get("epoch") or last.get("epochs_completed")
    elif isinstance(payload, list) and payload:
        last = payload[-1]
        if isinstance(last, dict):
            final_epoch = last.get("epoch") or last.get("epochs_completed")
            training_time = last.get("training_time") or last.get("training_time_sec") or last.get("wall_time_sec")

    pieces = []
    if final_epoch is not None:
        pieces.append(f"final epoch {final_epoch}")
    if training_time is not None:
        pieces.append(f"training time {format_training_time(training_time)}")
    return ", ".join(pieces) if pieces else None


def format_training_time(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        seconds = float(value)
        if seconds >= 3600:
            return f"{seconds / 3600:.2f} h"
        if seconds >= 60:
            return f"{seconds / 60:.1f} min"
        return f"{seconds:.0f} s"
    return str(value)


def collect_records(exp_dir: Path) -> list[ExperimentRecord]:
    records = []
    for exp_id in EXP_IDS:
        root = exp_dir / exp_id
        eval_path = root / "eval_results.json"
        recipe = optional_object(root / "recipe.json")
        train_summary = summarize_train_log(root / "train_log.json")
        if not eval_path.exists():
            records.append(
                ExperimentRecord(
                    exp_id=exp_id,
                    status="not run",
                    data={"exp_id": exp_id, "notes": "not run"},
                    error="missing eval_results.json",
                    recipe=recipe,
                    train_summary=train_summary,
                )
            )
            continue
        try:
            data = validate_eval_payload(load_json(eval_path))
            records.append(
                ExperimentRecord(
                    exp_id=exp_id,
                    status="completed",
                    data=data,
                    recipe=recipe,
                    train_summary=train_summary,
                )
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            records.append(
                ExperimentRecord(
                    exp_id=exp_id,
                    status="failed",
                    data={"exp_id": exp_id, "notes": f"failed: {exc}"},
                    error=str(exc),
                    recipe=recipe,
                    train_summary=train_summary,
                )
            )
    return records


def has_any_eval_file(exp_dir: Path) -> bool:
    return any((exp_dir / exp_id / "eval_results.json").exists() for exp_id in EXP_IDS)


def bool_cell(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return ""


def metric_cell(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.4f}".rstrip("0").rstrip(".")
    return ""


def csv_row(record: ExperimentRecord) -> dict[str, str]:
    if record.status != "completed":
        notes = "not run" if record.status == "not run" else str(record.data.get("notes", "failed"))
        return {column: (record.exp_id if column == "experiment_name" else notes if column == "notes" else "") for column in CSV_COLUMNS}
    data = record.data
    return {
        "experiment_name": str(data.get("exp_id") or record.exp_id),
        "seed": str(data.get("seed", "")),
        "stream1_clip_init": bool_cell(data.get("stream1_clip_init")),
        "stream1_supcon": bool_cell(data.get("stream1_supcon")),
        "stream1_adamw": bool_cell(data.get("stream1_adamw")),
        "stream2_fixed": bool_cell(data.get("stream2_fixed")),
        "aqe": bool_cell(data.get("aqe")),
        "rerank": bool_cell(data.get("rerank")),
        "tta": bool_cell(data.get("tta")),
        "fusion": bool_cell(data.get("fusion")),
        "mAP": metric_cell(data.get("mAP")),
        "rank1": metric_cell(data.get("rank1")),
        "notes": str(data.get("notes", "")),
    }


def experiment_json(record: ExperimentRecord) -> dict[str, Any]:
    payload = dict(record.data)
    payload.setdefault("exp_id", record.exp_id)
    payload["status"] = record.status
    if record.error:
        payload["error"] = record.error
    if record.recipe is not None:
        payload["recipe"] = record.recipe
    return payload


def completed_metric(records: dict[str, ExperimentRecord], exp_id: str, key: str) -> float | None:
    record = records.get(exp_id)
    if record is None or record.status != "completed":
        return None
    value = record.data.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def seed_variance(records: list[ExperimentRecord]) -> dict[str, Any]:
    maps = []
    ranks = []
    for record in records:
        if record.exp_id not in SEED_IDS or record.status != "completed":
            continue
        maps.append(float(record.data["mAP"]))
        ranks.append(float(record.data["rank1"]))

    n_completed = len(maps)
    return {
        "seeds": SEED_VALUES,
        "recipe": SEED_VARIANCE_RECIPE,
        "mAP_mean": float(np.mean(maps)) if maps else None,
        "mAP_std": float(np.std(maps, ddof=1)) if n_completed >= 2 else None,
        "rank1_mean": float(np.mean(ranks)) if ranks else None,
        "rank1_std": float(np.std(ranks, ddof=1)) if n_completed >= 2 else None,
        "n_completed": n_completed,
    }


def recommended_values(records: list[ExperimentRecord]) -> dict[str, Any]:
    by_id = {record.exp_id: record for record in records}
    alpha_map = completed_metric(by_id, "A5alpha", "mAP")
    beta_map = completed_metric(by_id, "A5beta", "mAP")
    alpha_rank = completed_metric(by_id, "A5alpha", "rank1")
    beta_rank = completed_metric(by_id, "A5beta", "rank1")
    tie_band = alpha_map is not None and beta_map is not None and abs(alpha_map - beta_map) <= 0.10

    if tie_band:
        headline = "both"
    elif alpha_map is not None and (beta_map is None or alpha_map > beta_map):
        headline = "A5alpha"
    elif beta_map is not None and (alpha_map is None or beta_map > alpha_map):
        headline = "A5beta"
    else:
        headline = "both"

    fusion_candidates = [record for record in records if record.status == "completed" and record.data.get("fusion") is True]
    best_fusion = max(fusion_candidates, key=lambda record: float(record.data["mAP"]), default=None)
    if best_fusion is not None:
        fusion_map = float(best_fusion.data["mAP"])
        fusion_rank = float(best_fusion.data["rank1"])
    else:
        fusion_map = None
        fusion_rank = None

    if headline == "A5beta":
        stream_map = beta_map
        stream_rank = beta_rank
    elif headline == "A5alpha":
        stream_map = alpha_map
        stream_rank = alpha_rank
    elif alpha_map is not None and beta_map is not None:
        stream_map = max(alpha_map, beta_map)
        stream_rank = alpha_rank if alpha_map >= beta_map else beta_rank
    else:
        stream_map = alpha_map if alpha_map is not None else beta_map
        stream_rank = alpha_rank if alpha_rank is not None else beta_rank

    return {
        "headline_recipe": headline,
        "stream1_full_mAP": stream_map,
        "stream1_full_rank1": stream_rank,
        "fusion_mAP": fusion_map,
        "fusion_rank1": fusion_rank,
        "tie_band_applies": tie_band,
    }


def write_csv(path: Path, records: list[ExperimentRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(csv_row(record))


def write_results_json(path: Path, records: list[ExperimentRecord]) -> None:
    failures = [record.exp_id for record in records if record.status != "completed"]
    payload = {
        "repo_commit": run_git(["rev-parse", "--short", "HEAD"]),
        "branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "dataset_split": DATASET_SPLIT,
        "experiments": [experiment_json(record) for record in records],
        "failures": failures,
        "seed_variance": seed_variance(records),
        "recommended_paper_table_values": recommended_values(records),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def table_row(cells: list[Any]) -> str:
    return "| " + " | ".join(markdown_cell(cell) for cell in cells) + " |"


def markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def value_or_not_run(record: ExperimentRecord, key: str) -> str:
    if record.status != "completed":
        return "not run" if record.status == "not run" else "failed"
    return metric_cell(record.data.get(key))


def delta_text(current: ExperimentRecord, baseline: ExperimentRecord | None, key: str) -> str:
    if current.status != "completed" or baseline is None or baseline.status != "completed":
        return "n/a"
    delta = float(current.data[key]) - float(baseline.data[key])
    return f"{delta:+.2f} pp"


def format_mean_std(mean: Any, std: Any) -> str:
    if mean is None:
        return "not run"
    if std is None:
        return f"{float(mean):.2f} +/- n/a"
    return f"{float(mean):.2f} +/- {float(std):.2f}"


def build_report(records: list[ExperimentRecord]) -> str:
    by_id = {record.exp_id: record for record in records}
    baseline = by_id.get("A1")
    variance = seed_variance(records)
    recommended = recommended_values(records)
    failures = [record for record in records if record.status != "completed"]
    lines = ["# VeRi-776 Paper Campaign — Results Report", ""]

    lines.extend(["## Ablation Findings", ""])
    lines.append(table_row(["Experiment", "Status", "mAP", "Rank-1", "Delta mAP vs A1", "Delta Rank-1 vs A1", "Notes"]))
    lines.append(table_row(["---", "---", "---", "---", "---", "---", "---"]))
    for exp_id in ABLATION_IDS:
        record = by_id[exp_id]
        lines.append(
            table_row(
                [
                    exp_id,
                    record.status,
                    value_or_not_run(record, "mAP"),
                    value_or_not_run(record, "rank1"),
                    delta_text(record, baseline, "mAP"),
                    delta_text(record, baseline, "rank1"),
                    str(record.data.get("notes", "")),
                ]
            )
        )
    lines.append("")
    lines.extend(component_commentary(by_id, baseline))
    lines.append("")

    lines.extend(["## Seed Variance", ""])
    lines.append(table_row(["Experiment", "Seed", "Status", "mAP", "Rank-1", "Training summary"]))
    lines.append(table_row(["---", "---", "---", "---", "---", "---"]))
    for exp_id in SEED_IDS:
        record = by_id[exp_id]
        seed = record.data.get("seed", exp_id.rsplit("seed", 1)[-1])
        lines.append(table_row([exp_id, seed, record.status, value_or_not_run(record, "mAP"), value_or_not_run(record, "rank1"), record.train_summary or "-"]))
    lines.append(table_row(["Mean +/- std", "42/123/456", f"n={variance['n_completed']}", format_mean_std(variance["mAP_mean"], variance["mAP_std"]), format_mean_std(variance["rank1_mean"], variance["rank1_std"]), "training-time variance"]))
    lines.append("")
    lines.append("Seed variance is training-time variance, not inference-time variance.")
    lines.append("")

    lines.extend(["## Recipe Disclosure Outcome", ""])
    lines.extend(recipe_outcome_lines(by_id, recommended))
    lines.append("")

    lines.extend(["## Failures", ""])
    if failures:
        for record in failures:
            reason = record.error or record.data.get("notes") or record.status
            lines.append(f"- {record.exp_id}: {reason}")
    else:
        lines.append("No failed or missing kernel outputs were detected.")
    lines.append("")

    lines.extend(["## Unresolved Limitations", ""])
    lines.append("- CityFlowV2 domain gap remains unresolved by this VeRi-776 campaign.")
    lines.append("- Two-stream inference has an added compute and storage cost compared with stream-1-only evaluation.")
    not_run = [record.exp_id for record in records if record.status == "not run"]
    if not_run:
        lines.append(f"- Ablations not run: {', '.join(not_run)}.")
    return "\n".join(lines) + "\n"


def component_commentary(by_id: dict[str, ExperimentRecord], baseline: ExperimentRecord | None) -> list[str]:
    comparisons = [
        ("A2", "CLIP init"),
        ("A3", "SupCon"),
        ("A4", "AdamW"),
        ("A5alpha", "A5alpha full recipe"),
        ("A5beta", "A5beta full recipe"),
    ]
    lines = []
    if baseline is None or baseline.status != "completed":
        lines.append("Baseline A1 is not completed, so per-component deltas are unavailable.")
        return lines
    for exp_id, label in comparisons:
        record = by_id[exp_id]
        if record.status == "completed":
            lines.append(f"- {label} changes mAP by {delta_text(record, baseline, 'mAP')} and Rank-1 by {delta_text(record, baseline, 'rank1')} versus A1.")
        else:
            lines.append(f"- {label} is {record.status}; no delta is reported.")
    return lines


def recipe_outcome_lines(by_id: dict[str, ExperimentRecord], recommended: dict[str, Any]) -> list[str]:
    alpha = by_id["A5alpha"]
    beta = by_id["A5beta"]
    lines = [f"Headline recipe: {recommended['headline_recipe']}."]
    if alpha.status == "completed" and beta.status == "completed":
        delta = float(alpha.data["mAP"]) - float(beta.data["mAP"])
        winner = "A5alpha" if delta > 0 else "A5beta" if delta < 0 else "neither recipe"
        lines.append(f"A5alpha minus A5beta is {delta:+.2f} pp mAP; {winner} leads on mAP.")
        lines.append(f"Tie band applies: {str(recommended['tie_band_applies']).lower()}.")
    elif alpha.status == "completed" or beta.status == "completed":
        completed = "A5alpha" if alpha.status == "completed" else "A5beta"
        missing = "A5beta" if completed == "A5alpha" else "A5alpha"
        lines.append(f"Only {completed} is completed; {missing} is not available for the alpha/beta decision.")
        lines.append("Tie band applies: false.")
    else:
        lines.append("Neither A5alpha nor A5beta is completed, so no recipe decision can be made yet.")
        lines.append("Tie band applies: false.")
    return lines


def write_report(path: Path, records: list[ExperimentRecord]) -> None:
    path.write_text(build_report(records), encoding="utf-8")


def print_summary(records: list[ExperimentRecord]) -> None:
    print("Experiment aggregation summary:")
    print("exp_id       status")
    print("------------ ---------")
    for record in records:
        print(f"{record.exp_id:<12} {record.status}")


def main() -> int:
    args = parse_args()
    exp_dir = args.exp_dir.resolve()
    records = collect_records(exp_dir)

    if args.dry_run:
        print(f"Dry run OK. Checked {len(records)} experiment slots under {exp_dir}.")
        print_summary(records)
        return 0

    if not has_any_eval_file(exp_dir):
        print(NO_RESULTS_MESSAGE)
        print_summary(records)
        return 0

    exp_dir.mkdir(parents=True, exist_ok=True)
    write_csv(exp_dir / "results.csv", records)
    write_results_json(exp_dir / "results.json", records)
    write_report(exp_dir / "REPORT.md", records)
    print_summary(records)
    print(f"Wrote {exp_dir / 'results.csv'}")
    print(f"Wrote {exp_dir / 'results.json'}")
    print(f"Wrote {exp_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())