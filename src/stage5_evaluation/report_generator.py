"""Evaluation report generation in HTML and Markdown formats."""

from __future__ import annotations

from pathlib import Path
from typing import List

from loguru import logger

from src.core.data_models import EvaluationResult, GlobalTrajectory


def generate_report(
    result: EvaluationResult,
    trajectories: List[GlobalTrajectory],
    output_path: str | Path,
    fmt: str = "html",
) -> None:
    """Generate a human-readable evaluation report.

    Args:
        result: Evaluation metrics.
        trajectories: Global trajectories for summary statistics.
        output_path: Output file path.
        fmt: Format - "html" or "markdown".
    """
    if fmt == "html":
        content = _generate_html_report(result, trajectories)
    else:
        content = _generate_markdown_report(result, trajectories)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def _trajectory_stats(trajectories: List[GlobalTrajectory]) -> dict:
    """Compute summary statistics for trajectories."""
    if not trajectories:
        return {"total": 0}

    num_traj = len(trajectories)
    multi_cam = sum(1 for t in trajectories if t.num_cameras > 1)
    total_tracklets = sum(len(t.tracklets) for t in trajectories)
    durations = [t.total_duration for t in trajectories]
    cameras_per = [t.num_cameras for t in trajectories]

    return {
        "total": num_traj,
        "multi_camera": multi_cam,
        "single_camera": num_traj - multi_cam,
        "total_tracklets": total_tracklets,
        "avg_duration": sum(durations) / num_traj,
        "max_duration": max(durations),
        "avg_cameras": sum(cameras_per) / num_traj,
        "max_cameras": max(cameras_per),
    }


def _generate_markdown_report(
    result: EvaluationResult, trajectories: List[GlobalTrajectory]
) -> str:
    stats = _trajectory_stats(trajectories)
    lines = [
        "# MTMC Tracking Evaluation Report\n",
        "## Tracking Metrics\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| MOTA | {result.mota:.4f} |",
        f"| IDF1 | {result.idf1:.4f} |",
        f"| HOTA | {result.hota:.4f} |",
        f"| ID Switches | {result.id_switches} |",
        f"| Mostly Tracked | {result.mostly_tracked:.4f} |",
        f"| Mostly Lost | {result.mostly_lost:.4f} |",
        "",
        "## Trajectory Statistics\n",
        f"- **Total trajectories**: {stats['total']}",
        f"- **Multi-camera**: {stats.get('multi_camera', 0)}",
        f"- **Single-camera**: {stats.get('single_camera', 0)}",
        f"- **Total tracklets**: {stats.get('total_tracklets', 0)}",
        f"- **Avg duration**: {stats.get('avg_duration', 0):.1f}s",
        f"- **Avg cameras per trajectory**: {stats.get('avg_cameras', 0):.1f}",
    ]

    if result.details:
        lines.extend(["", "## Details\n"])
        for k, v in result.details.items():
            lines.append(f"- **{k}**: {v}")

    return "\n".join(lines)


def _generate_html_report(
    result: EvaluationResult, trajectories: List[GlobalTrajectory]
) -> str:
    stats = _trajectory_stats(trajectories)
    return f"""<!DOCTYPE html>
<html><head><title>MTMC Tracking Evaluation</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; }}
  h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: 10px; }}
  h2 {{ color: #16213e; margin-top: 30px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 10px 14px; text-align: left; }}
  th {{ background-color: #16213e; color: white; }}
  tr:nth-child(even) {{ background-color: #f5f5f5; }}
  .metric-value {{ font-weight: bold; font-size: 1.1em; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; margin: 15px 0; }}
  .stat-card {{ background: #f8f9fa; border-radius: 8px; padding: 15px; border-left: 4px solid #16213e; }}
  .stat-card .label {{ color: #666; font-size: 0.85em; }}
  .stat-card .value {{ font-size: 1.4em; font-weight: bold; color: #1a1a2e; }}
</style></head><body>
<h1>MTMC Tracking Evaluation Report</h1>

<h2>Tracking Metrics</h2>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>MOTA</td><td class="metric-value">{result.mota:.4f}</td></tr>
  <tr><td>IDF1</td><td class="metric-value">{result.idf1:.4f}</td></tr>
  <tr><td>HOTA</td><td class="metric-value">{result.hota:.4f}</td></tr>
  <tr><td>ID Switches</td><td class="metric-value">{result.id_switches}</td></tr>
  <tr><td>Mostly Tracked</td><td class="metric-value">{result.mostly_tracked:.4f}</td></tr>
  <tr><td>Mostly Lost</td><td class="metric-value">{result.mostly_lost:.4f}</td></tr>
</table>

<h2>Trajectory Statistics</h2>
<div class="stat-grid">
  <div class="stat-card"><div class="label">Total Trajectories</div><div class="value">{stats['total']}</div></div>
  <div class="stat-card"><div class="label">Multi-Camera</div><div class="value">{stats.get('multi_camera', 0)}</div></div>
  <div class="stat-card"><div class="label">Total Tracklets</div><div class="value">{stats.get('total_tracklets', 0)}</div></div>
  <div class="stat-card"><div class="label">Avg Duration</div><div class="value">{stats.get('avg_duration', 0):.1f}s</div></div>
  <div class="stat-card"><div class="label">Avg Cameras/Traj</div><div class="value">{stats.get('avg_cameras', 0):.1f}</div></div>
  <div class="stat-card"><div class="label">Max Cameras</div><div class="value">{stats.get('max_cameras', 0)}</div></div>
</div>

<h2>Ground Truth</h2>
<table>
  <tr><td>GT Identities</td><td>{result.num_gt_ids}</td></tr>
  <tr><td>Predicted Identities</td><td>{result.num_pred_ids}</td></tr>
</table>
</body></html>"""
