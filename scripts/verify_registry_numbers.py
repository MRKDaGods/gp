"""Verify every metric claimed in configs/model_registry.yaml against its source."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "configs" / "model_registry.yaml"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "docs" / "_data" / "registry_verification.json"
DEFAULT_KAGGLE_MANIFEST_PATH = PROJECT_ROOT / "docs" / "_data" / "kaggle_kernel_summaries.json"

FLOAT_TOLERANCE = 0.0001
LINE_WINDOW = 5

NUMBER_PATTERN = re.compile(r"(?<![A-Za-z0-9_])-?\d+(?:\.\d+)?\s*(?:%|pp)?(?![A-Za-z0-9_])")


@dataclass(frozen=True)
class SourceSpec:
    raw: Any
    kind: str | None
    path: str | None
    kernel: str | None
    line_ref: str | None
    key_path: str | None


def load_registry(path: Path) -> dict[str, Any]:
    loaded = OmegaConf.load(path)
    data = OmegaConf.to_container(loaded, resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"Registry must be a mapping: {path}")
    return data


def parse_source(source: Any) -> SourceSpec:
    if isinstance(source, str):
        line_ref = None
        source_text = source
        if "#L" in source_text:
            source_text, line_suffix = source_text.split("#", 1)
            line_ref = line_suffix
        if ":" in source_text and not Path(source_text).suffix:
            kernel, artifact = source_text.split(":", 1)
            return SourceSpec(source, "kernel_summary", artifact, kernel, line_ref, None)
        return SourceSpec(source, None, source_text, None, line_ref, None)

    if isinstance(source, dict):
        return SourceSpec(
            raw=source,
            kind=source.get("kind"),
            path=source.get("path"),
            kernel=source.get("kernel"),
            line_ref=source.get("line_ref"),
            key_path=source.get("key_path") or source.get("json_path"),
        )

    return SourceSpec(source, None, None, None, None, None)


def resolve_local_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    clean = path_text.split("#", 1)[0]
    path = Path(clean)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def line_number_from_ref(line_ref: str | None) -> int | None:
    if not line_ref:
        return None
    match = re.search(r"L(\d+)", str(line_ref))
    if not match:
        return None
    return int(match.group(1))


def numeric_tokens(text: str) -> list[tuple[float, str]]:
    tokens: list[tuple[float, str]] = []
    for match in NUMBER_PATTERN.finditer(text):
        raw = match.group(0).strip()
        is_percent = raw.endswith("%")
        number_text = raw.removesuffix("pp").rstrip("%").strip()
        try:
            value = float(number_text)
        except ValueError:
            continue
        if is_percent:
            value /= 100.0
        tokens.append((value, raw))
    return tokens


def values_match(claimed: float, source_value: float) -> bool:
    return math.isclose(float(claimed), float(source_value), abs_tol=FLOAT_TOLERANCE, rel_tol=0.0)


def extract_from_markdown(path: Path, metric: dict[str, Any], source: SourceSpec) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    claimed = float(metric["value"])
    cited_line = line_number_from_ref(source.line_ref)

    def scan(start: int, end: int, *, require_metric_context: bool) -> dict[str, Any] | None:
        for index in range(max(1, start), min(len(lines), end) + 1):
            line = lines[index - 1]
            if require_metric_context and not line_mentions_metric(line, metric["name"]):
                continue
            for value, raw in numeric_tokens(line):
                if values_match(claimed, value):
                    return {
                        "source_value": value,
                        "matched_text": raw,
                        "matched_line": index,
                        "matched_excerpt": line.strip(),
                    }
        return None

    if cited_line is not None:
        near = scan(cited_line - LINE_WINDOW, cited_line + LINE_WINDOW, require_metric_context=True)
        if near is None:
            near = scan(cited_line - LINE_WINDOW, cited_line + LINE_WINDOW, require_metric_context=False)
        if near is not None:
            near["match_scope"] = "near_cited_line"
            return near
        return {
            "source_value": None,
            "matched_text": None,
            "matched_line": None,
            "matched_excerpt": None,
            "match_scope": "near_cited_line",
            "error": f"claimed value not found within +/-{LINE_WINDOW} lines of {source.line_ref}",
        }

    anywhere = scan(1, len(lines), require_metric_context=True)
    if anywhere is None:
        anywhere = scan(1, len(lines), require_metric_context=False)
    if anywhere is not None:
        anywhere["match_scope"] = "whole_file"
        return anywhere
    return {
        "source_value": None,
        "matched_text": None,
        "matched_line": None,
        "matched_excerpt": None,
        "match_scope": "whole_file",
        "error": "claimed value not found in markdown source",
    }


def get_by_key_path(data: Any, key_path: str) -> Any:
    cursor = data
    for part in key_path.split("."):
        if isinstance(cursor, dict):
            cursor = cursor[part]
        elif isinstance(cursor, list):
            cursor = cursor[int(part)]
        else:
            raise KeyError(key_path)
    return cursor


def walk_numeric_leaves(data: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], float]]:
    leaves: list[tuple[tuple[str, ...], float]] = []
    if isinstance(data, dict):
        for key, value in data.items():
            leaves.extend(walk_numeric_leaves(value, (*path, str(key))))
    elif isinstance(data, list):
        for index, value in enumerate(data):
            leaves.extend(walk_numeric_leaves(value, (*path, str(index))))
    elif isinstance(data, (int, float)) and not isinstance(data, bool):
        leaves.append((path, float(data)))
    return leaves


def metric_aliases(metric_name: str) -> set[str]:
    base = metric_name.lower()
    aliases = {base, base.replace("_", ""), base.replace("_", "-")}
    if base == "map":
        aliases.update({"m_ap", "mean_average_precision", "meanaverageprecision"})
    if base == "r1":
        aliases.update({"rank1", "rank_1", "top1", "top_1"})
    if base.endswith("idf1"):
        aliases.add("idf1")
    if "idf1" in base:
        aliases.add("idf1")
    if "moda" in base:
        aliases.add("moda")
    if "precision" in base:
        aliases.add("precision")
    if "recall" in base:
        aliases.add("recall")
    if "delta" in base:
        aliases.update({"delta", "regression", "lost", "catastrophic", "harmful"})
    return aliases


def line_mentions_metric(line: str, metric_name: str) -> bool:
    lower = line.lower().replace("_", "")
    return any(alias.replace("_", "") in lower for alias in metric_aliases(metric_name))


def path_matches_metric(path: tuple[str, ...], metric_name: str) -> bool:
    aliases = metric_aliases(metric_name)
    normalized_parts = {part.lower().replace("_", "") for part in path}
    normalized_path = ".".join(part.lower() for part in path)
    return bool(aliases & normalized_parts) or any(alias in normalized_path for alias in aliases)


def extract_from_json_data(data: Any, metric: dict[str, Any], source: SourceSpec) -> dict[str, Any]:
    claimed = float(metric["value"])
    if source.key_path:
        try:
            value = float(get_by_key_path(data, source.key_path))
        except Exception as exc:  # noqa: BLE001 - report exact source extraction failure
            return {"source_value": None, "json_path": source.key_path, "error": str(exc)}
        return {"source_value": value, "json_path": source.key_path}

    leaves = walk_numeric_leaves(data)
    metric_matches = [(path, value) for path, value in leaves if path_matches_metric(path, metric["name"])]
    for path, value in metric_matches:
        if values_match(claimed, value):
            return {"source_value": value, "json_path": ".".join(path), "match_scope": "metric_name"}
    for path, value in leaves:
        if values_match(claimed, value):
            return {"source_value": value, "json_path": ".".join(path), "match_scope": "fuzzy_value"}
    return {"source_value": None, "json_path": None, "error": "claimed value not found in JSON"}


def extract_from_json_file(path: Path, metric: dict[str, Any], source: SourceSpec) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return extract_from_json_data(data, metric, source)


def load_kaggle_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"kernels": []}
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_matches(path_text: str, wanted: str | None) -> bool:
    if not wanted:
        return True
    return Path(path_text).name.lower() == Path(wanted).name.lower() or path_text.lower().endswith(wanted.lower())


def extract_from_kaggle_manifest(
    manifest: dict[str, Any], metric: dict[str, Any], source: SourceSpec
) -> dict[str, Any]:
    kernel = source.kernel
    if not kernel:
        return {"source_value": None, "error": "kernel source missing slug"}

    kernels = manifest.get("kernels", [])
    entry = next((item for item in kernels if item.get("slug") == kernel), None)
    if entry is None:
        return {"source_value": None, "error": "kernel missing from Kaggle manifest"}
    if entry.get("status") == "UNAVAILABLE":
        return {"source_value": None, "error": f"kernel unavailable: {entry.get('reason')}"}

    for artifact in entry.get("summary_files", []):
        artifact_path = artifact.get("path", "")
        if not artifact_matches(artifact_path, source.path):
            continue
        data = artifact.get("data")
        if data is None:
            continue
        result = extract_from_json_data(data, metric, source)
        if result.get("source_value") is not None:
            result["artifact_path"] = artifact_path
            return result

    metrics = entry.get("metrics", {})
    result = extract_from_json_data(metrics, metric, source)
    if result.get("source_value") is not None:
        result["artifact_path"] = "manifest.metrics"
        return result
    return {"source_value": None, "error": "claimed value not found in Kaggle manifest"}


def verify_metric(
    model: dict[str, Any],
    metric: dict[str, Any],
    *,
    with_kaggle: bool,
    kaggle_manifest: dict[str, Any],
) -> dict[str, Any]:
    source = parse_source(metric.get("source"))
    verified = bool(metric.get("verified"))
    claimed = float(metric["value"])
    result: dict[str, Any] = {
        "model_id": model["id"],
        "metric": metric["name"],
        "claimed_value": claimed,
        "verified": verified,
        "source": source.raw,
        "status": "INFO" if not verified else "FAIL",
        "error": None,
    }

    extraction: dict[str, Any]
    if with_kaggle and source.kind == "kernel_summary":
        extraction = extract_from_kaggle_manifest(kaggle_manifest, metric, source)
    else:
        source_path = resolve_local_path(source.path)
        if source_path is None or not source_path.exists():
            result["error"] = f"source file missing: {source.path}"
            return result
        suffix = source_path.suffix.lower()
        try:
            if suffix == ".json":
                extraction = extract_from_json_file(source_path, metric, source)
            elif suffix in {".md", ".markdown", ".txt", ".log"}:
                extraction = extract_from_markdown(source_path, metric, source)
            else:
                extraction = {"source_value": None, "error": f"unsupported source type: {suffix}"}
        except Exception as exc:  # noqa: BLE001 - verifier must report, not crash per metric
            extraction = {"source_value": None, "error": str(exc)}

    result.update(extraction)
    source_value = result.get("source_value")
    matched = source_value is not None and values_match(claimed, float(source_value))
    if verified:
        result["status"] = "PASS" if matched else "FAIL"
    else:
        result["status"] = "INFO"
    if not matched and result.get("error") is None:
        result["error"] = f"source value {source_value} does not match claimed value {claimed}"

    if with_kaggle and source.kernel and source.kind != "kernel_summary":
        kaggle_source = SourceSpec(
            raw=source.raw,
            kind="kernel_summary",
            path=None,
            kernel=source.kernel,
            line_ref=None,
            key_path=source.key_path,
        )
        kaggle_result = extract_from_kaggle_manifest(kaggle_manifest, metric, kaggle_source)
        kaggle_value = kaggle_result.get("source_value")
        if kaggle_value is not None and values_match(claimed, float(kaggle_value)):
            result["kaggle_status"] = "PASS"
            result["kaggle_source_value"] = kaggle_value
            result["kaggle_artifact_path"] = kaggle_result.get("artifact_path")
            result["kaggle_json_path"] = kaggle_result.get("json_path")
        elif kaggle_result.get("error", "").startswith("kernel unavailable"):
            result["kaggle_status"] = "UNAVAILABLE"
            result["kaggle_error"] = kaggle_result.get("error")
        else:
            result["kaggle_status"] = "INFO"
            result["kaggle_error"] = kaggle_result.get("error")
    return result


def format_source(result: dict[str, Any]) -> str:
    source = result.get("source")
    if isinstance(source, dict):
        path = source.get("path") or "<missing path>"
        line_ref = source.get("line_ref")
        if result.get("matched_line"):
            return f"{path}#L{result['matched_line']} matched"
        if line_ref:
            return f"{path}#{line_ref}"
        return str(path)
    return str(source)


def write_report(path: Path, registry: dict[str, Any], results: list[dict[str, Any]], with_kaggle: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "passed": sum(1 for item in results if item["status"] == "PASS"),
        "failed": sum(1 for item in results if item["status"] == "FAIL"),
        "info": sum(1 for item in results if item["status"] == "INFO"),
    }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "with_kaggle": with_kaggle,
        "total_entries": len(registry.get("models", [])),
        "total_metrics": len(results),
        "summary": summary,
        "results": results,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def print_report(registry: dict[str, Any], results: list[dict[str, Any]]) -> None:
    print("=== Registry Verification Report ===")
    print(f"Total entries: {len(registry.get('models', []))}")
    print(f"Total metrics: {len(results)}")
    print()
    for item in results:
        status = item["status"]
        source = format_source(item)
        value = item["claimed_value"]
        suffix = f" (source: {source})"
        if item.get("error"):
            suffix = f" (source: {source} -- {item['error']})"
        print(f"[{status}] {item['model_id']} -> {item['metric']} = {value:g}{suffix}")
    print()
    passed = sum(1 for item in results if item["status"] == "PASS")
    failed = sum(1 for item in results if item["status"] == "FAIL")
    info = sum(1 for item in results if item["status"] == "INFO")
    print(f"Summary: {passed} passed, {failed} failed, {info} info-only")


def run_verification(
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    *,
    with_kaggle: bool = False,
    kaggle_manifest_path: Path = DEFAULT_KAGGLE_MANIFEST_PATH,
) -> tuple[int, list[dict[str, Any]]]:
    registry = load_registry(registry_path)
    kaggle_manifest = load_kaggle_manifest(kaggle_manifest_path) if with_kaggle else {"kernels": []}
    results = [
        verify_metric(model, metric, with_kaggle=with_kaggle, kaggle_manifest=kaggle_manifest)
        for model in registry.get("models", [])
        for metric in model.get("metrics", [])
    ]
    write_report(report_path, registry, results, with_kaggle)
    print_report(registry, results)
    return (1 if any(item["status"] == "FAIL" for item in results) else 0), results


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify model registry metric values against sources.")
    parser.add_argument("registry", nargs="?", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--with-kaggle", action="store_true", help="Cross-check kernel_summary sources against the Kaggle manifest.")
    parser.add_argument("--kaggle-manifest", default=str(DEFAULT_KAGGLE_MANIFEST_PATH))
    args = parser.parse_args()

    exit_code, _ = run_verification(
        Path(args.registry),
        Path(args.report),
        with_kaggle=args.with_kaggle,
        kaggle_manifest_path=Path(args.kaggle_manifest),
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())