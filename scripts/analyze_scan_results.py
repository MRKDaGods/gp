"""Download and analyze Kaggle scan results.

Usage:
    python scripts/analyze_scan_results.py [--download]
    python scripts/analyze_scan_results.py --file data/outputs/scan_results.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def download_results(output_dir: Path) -> Path | None:
    """Download scan_results.json from Kaggle kernel output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle", "kernels", "output",
        "mrkdagods/mtmc-10c-stages-4-5-association-eval",
        "-p", str(output_dir),
    ]
    print(f"Downloading: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"Download failed: {r.stderr}")
        return None

    results_file = output_dir / "scan_results.json"
    if results_file.exists():
        print(f"Downloaded: {results_file}")
        return results_file

    # Try to find it in subdirectories
    for f in output_dir.rglob("scan_results.json"):
        print(f"Found: {f}")
        return f

    print("scan_results.json not found in output")
    return None


def analyze(results_file: Path) -> None:
    """Analyze scan results."""
    data = json.loads(results_file.read_text())
    sort_key = data.get("sort_key", "HOTA")
    results = data.get("results", [])

    if not results:
        print("No results found!")
        return

    print(f"\n{'='*80}")
    print(f"SCAN RESULTS ANALYSIS ({len(results)} combinations)")
    print(f"{'='*80}")

    # Sort by HOTA then IDF1
    results.sort(key=lambda x: (x.get("HOTA", 0), x.get("IDF1", 0)), reverse=True)

    # Top 20 results
    print(f"\n--- TOP 20 by {sort_key} ---")
    header = f"{'#':<3} {'sim':<6} {'app_w':<7} {'bridge':<8} {'gal_th':<7} {'orph':<6} {'rr_l':<6} {'alg':<5} {'IDF1':>7} {'MOTA':>7} {'HOTA':>7} {'M_MOTA':>7}"
    print(header)
    for i, r in enumerate(results[:20]):
        alg = "agg" if r.get("algorithm", "") == "agglomerative" else "cc"
        print(
            f"{i+1:<3} {r['sim_thresh']:<6} {r['appearance_w']:<7} "
            f"{r['bridge_prune']:<8} {r.get('gallery_thresh','?'):<7} "
            f"{r.get('orphan_thresh',0):<6} {r.get('rerank_lambda',0):<6} "
            f"{alg:<5} {r['IDF1']:>7.3f} {r['MOTA']:>7.3f} {r['HOTA']:>7.3f} "
            f"{r.get('MTMC_MOTA',0):>7.3f}"
        )

    # Parameter sensitivity
    print(f"\n{'='*80}")
    print("PARAMETER SENSITIVITY ANALYSIS")
    print(f"{'='*80}")

    param_keys = [k for k in results[0] if k not in ("IDF1", "MOTA", "HOTA", "MTMC_MOTA", "time", "st_w")]
    for param_name in param_keys:
        param_vals = sorted(set(r.get(param_name) for r in results if param_name in r))
        if len(param_vals) < 2:
            continue
        print(f"\n--- {param_name} ---")
        for pval in param_vals:
            subset = [r for r in results if r.get(param_name) == pval]
            if not subset:
                continue
            avg_hota = sum(r["HOTA"] for r in subset) / len(subset)
            avg_idf1 = sum(r["IDF1"] for r in subset) / len(subset)
            avg_mota = sum(r["MOTA"] for r in subset) / len(subset)
            best_hota = max(r["HOTA"] for r in subset)
            best_idf1 = max(r["IDF1"] for r in subset)
            print(
                f"  {str(pval):<15} avg HOTA={avg_hota:.3f} avg IDF1={avg_idf1:.3f} "
                f"avg MOTA={avg_mota:.3f} | best HOTA={best_hota:.3f} IDF1={best_idf1:.3f} "
                f"(n={len(subset)})"
            )

    # Best combo
    best = results[0]
    print(f"\n{'='*80}")
    print(f"BEST COMBINATION:")
    for k, v in sorted(best.items()):
        if k not in ("time",):
            print(f"  {k}: {v}")

    # Metric distributions
    hotas = [r["HOTA"] for r in results if r["HOTA"] > 0]
    idf1s = [r["IDF1"] for r in results if r["IDF1"] > 0]
    motas = [r["MOTA"] for r in results]
    if hotas:
        print(f"\nHOTA distribution: min={min(hotas):.3f} median={sorted(hotas)[len(hotas)//2]:.3f} max={max(hotas):.3f}")
    if idf1s:
        print(f"IDF1 distribution: min={min(idf1s):.3f} median={sorted(idf1s)[len(idf1s)//2]:.3f} max={max(idf1s):.3f}")
    if motas:
        print(f"MOTA distribution: min={min(motas):.3f} median={sorted(motas)[len(motas)//2]:.3f} max={max(motas):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Kaggle scan results")
    parser.add_argument("--download", action="store_true", help="Download results from Kaggle")
    parser.add_argument("--file", type=str, help="Path to scan_results.json")
    parser.add_argument("--output-dir", type=str, default="data/outputs/nb10c_v22",
                        help="Download directory")
    args = parser.parse_args()

    if args.file:
        analyze(Path(args.file))
    elif args.download:
        results_file = download_results(Path(args.output_dir))
        if results_file:
            analyze(results_file)
    else:
        # Try to find an existing file
        candidates = [
            Path(args.output_dir) / "scan_results.json",
            Path("data/outputs/scan_results.json"),
        ]
        for c in candidates:
            if c.exists():
                analyze(c)
                return
        print("No results file found. Use --download to fetch from Kaggle, or --file <path>.")
        sys.exit(1)


if __name__ == "__main__":
    main()
