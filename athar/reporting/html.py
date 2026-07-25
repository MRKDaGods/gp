"""Chain-of-custody report HTML.

Input is the run's ``package.report`` artifact (report_inputs.json): the
evidence SHA-256 -> config hash -> identities skeleton written by the
package stage. The rendered chain is: video SHA → config hash → model
SHA → results. Model SHAs come from the pinned weights manifest — the
report only states hashes it can actually attest, never invents one.
"""

from __future__ import annotations

import base64
import html as html_escape
from pathlib import Path
from typing import Any, Mapping, Optional

WEIGHT_EXTENSIONS = (".pt", ".pth", ".onnx", ".bin", ".ckpt")

_LABELS = {
    "ar": {
        "title": "تقرير سلسلة العهدة — أثر",
        "run": "المعالجة",
        "role": "الدور",
        "profile": "الملف",
        "created": "الإنشاء",
        "config_hash": "بصمة الإعدادات",
        "evidence": "الأدلة (فيديو المصدر)",
        "camera": "الكاميرا",
        "path": "المسار",
        "sha256": "SHA-256",
        "duration": "المدة (ث)",
        "fps": "إطار/ث",
        "models": "النماذج المستخدمة",
        "model_file": "ملف النموذج",
        "unrecorded": "غير مسجل في بيان الأوزان",
        "no_models": "لا توجد نماذج مسجلة في الإعدادات المجمدة",
        "identities": "الهويات المستخرجة",
        "identity": "الهوية",
        "class": "الفئة",
        "cross_camera": "عبر الكاميرات",
        "members": "المسارات",
        "track": "المسار",
        "from": "من (ث)",
        "to": "إلى (ث)",
        "thumbnail": "صورة",
        "clip": "مقطع الدليل",
        "yes": "نعم",
        "no": "لا",
        "no_identities": "لا توجد هويات في هذه المعالجة",
        "footer": "وثيقة مولدة آليًا من مدخلات مجمدة؛ كل بند أعلاه قابل "
        "للتحقق عبر البصمات المذكورة.",
    },
    "en": {
        "title": "Chain-of-Custody Report — ATHAR",
        "run": "Run",
        "role": "Role",
        "profile": "Profile",
        "created": "Created",
        "config_hash": "Config hash",
        "evidence": "Evidence (source video)",
        "camera": "Camera",
        "path": "Path",
        "sha256": "SHA-256",
        "duration": "Duration (s)",
        "fps": "FPS",
        "models": "Models used",
        "model_file": "Model file",
        "unrecorded": "not pinned in the weights manifest",
        "no_models": "no model checkpoints recorded in the frozen config",
        "identities": "Extracted identities",
        "identity": "Identity",
        "class": "Class",
        "cross_camera": "Cross-camera",
        "members": "Tracklets",
        "track": "Track",
        "from": "From (s)",
        "to": "To (s)",
        "thumbnail": "Image",
        "clip": "Evidence clip",
        "yes": "yes",
        "no": "no",
        "no_identities": "no identities in this run",
        "footer": "Machine-generated from frozen inputs; every row above is "
        "verifiable through the listed hashes.",
    },
}

_CSS = """
body { font-family: 'IBM Plex Sans Arabic', 'Segoe UI', 'Noto Sans Arabic',
       sans-serif; color: #171717; margin: 2rem; font-size: 12px; }
h1 { font-size: 18px; border-bottom: 2px solid #171717; padding-bottom: 6px; }
h2 { font-size: 14px; margin-top: 1.4em; }
table { border-collapse: collapse; width: 100%; margin-top: 0.4em; }
th, td { border: 1px solid #bbb; padding: 4px 8px; text-align: start;
         vertical-align: top; }
th { background: #f0f0f0; }
code { font-family: Consolas, monospace; font-size: 11px;
       direction: ltr; unicode-bidi: embed; }
img.thumb { max-height: 64px; max-width: 64px; }
footer { margin-top: 2em; font-size: 10px; color: #555;
         border-top: 1px solid #bbb; padding-top: 6px; }
@page { size: A4; margin: 15mm; }
"""


def load_weight_shas(manifest_path: Path) -> dict[str, str]:
    """basename -> sha256 from the pinned weights manifest (empty when the
    manifest is absent — callers then report checkpoints as unrecorded)."""
    if not manifest_path.is_file():
        return {}
    import yaml

    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    shas: dict[str, str] = {}
    for file_set in (data.get("sets") or {}).values():
        for entry in file_set.get("files", []) or []:
            if entry.get("name") and entry.get("sha256"):
                shas[entry["name"]] = entry["sha256"]
    return shas


def models_from_config(
    values: Mapping[str, Any], weight_shas: Mapping[str, str]
) -> list[dict]:
    """Collect model checkpoint references out of the frozen config values
    and attach the pinned SHA when the weights manifest knows the file."""
    models: dict[str, Optional[str]] = {}
    for value in values.values():
        if isinstance(value, str) and value.lower().endswith(WEIGHT_EXTENSIONS):
            name = Path(value).name
            models[value] = weight_shas.get(name)
    return [
        {"path": path, "sha256": sha}
        for path, sha in sorted(models.items())
    ]


def _esc(value: Any) -> str:
    return html_escape.escape("" if value is None else str(value))


def _thumb_cell(relpath: Optional[str], run_dir: Optional[Path]) -> str:
    if not relpath or run_dir is None:
        return ""
    path = run_dir / relpath
    if not path.is_file():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img class="thumb" src="data:image/jpeg;base64,{encoded}" />'


def _clip_cell(member: Mapping[str, Any]) -> str:
    """Reference to the package-time evidence clip (path + pinned hash) —
    the clip file itself stays beside the run, referenced not embedded."""
    clip = member.get("clip")
    if not clip:
        return "—"
    sha = member.get("clip_sha256")
    sha_html = f"<br/><code>{_esc(sha)}</code>" if sha else ""
    return f"<code>{_esc(clip)}</code>{sha_html}"


def render_report_html(
    report: Mapping[str, Any],
    *,
    models: Optional[list[dict]] = None,
    run_dir: Optional[Path] = None,
    locale: str = "ar",
) -> str:
    """Self-contained report document (thumbnails inlined as data URIs)."""
    t = _LABELS.get(locale, _LABELS["ar"])
    direction = "rtl" if locale == "ar" else "ltr"
    run = report.get("run", {})
    parts: list[str] = [
        f'<!DOCTYPE html><html lang="{locale}" dir="{direction}"><head>'
        f'<meta charset="utf-8"><title>{t["title"]}</title>'
        f"<style>{_CSS}</style></head><body>",
        f"<h1>{t['title']}</h1>",
        "<table>",
        f"<tr><th>{t['run']}</th><td><code>{_esc(run.get('run_id'))}</code></td>"
        f"<th>{t['role']}</th><td>{_esc(run.get('role'))}</td></tr>",
        f"<tr><th>{t['profile']}</th><td>{_esc(run.get('profile'))}</td>"
        f"<th>{t['created']}</th><td><code>{_esc(run.get('created_at'))}</code></td></tr>",
        f"<tr><th>{t['config_hash']}</th>"
        f"<td colspan=3><code>{_esc(run.get('config_hash'))}</code></td></tr>",
        "</table>",
    ]

    parts.append(f"<h2>{t['evidence']}</h2><table><tr>")
    parts.append(
        f"<th>{t['camera']}</th><th>{t['path']}</th><th>{t['sha256']}</th>"
        f"<th>{t['duration']}</th><th>{t['fps']}</th></tr>"
    )
    for item in report.get("evidence", []):
        parts.append(
            f"<tr><td>{_esc(item.get('camera_id'))}</td>"
            f"<td><code>{_esc(item.get('original_path'))}</code></td>"
            f"<td><code>{_esc(item.get('sha256'))}</code></td>"
            f"<td>{_esc(item.get('duration_s'))}</td>"
            f"<td>{_esc(item.get('fps'))}</td></tr>"
        )
    parts.append("</table>")

    parts.append(f"<h2>{t['models']}</h2>")
    if models:
        parts.append(
            f"<table><tr><th>{t['model_file']}</th><th>{t['sha256']}</th></tr>"
        )
        for model in models:
            sha = model.get("sha256")
            sha_cell = (
                f"<code>{_esc(sha)}</code>" if sha else f"<i>{t['unrecorded']}</i>"
            )
            parts.append(
                f"<tr><td><code>{_esc(model.get('path'))}</code></td>"
                f"<td>{sha_cell}</td></tr>"
            )
        parts.append("</table>")
    else:
        parts.append(f"<p><i>{t['no_models']}</i></p>")

    parts.append(f"<h2>{t['identities']}</h2>")
    identities = report.get("identities", [])
    if not identities:
        parts.append(f"<p><i>{t['no_identities']}</i></p>")
    for identity in identities:
        cross = t["yes"] if identity.get("cross_camera") else t["no"]
        parts.append(
            f"<h3>{t['identity']} {_esc(identity.get('global_id'))} — "
            f"{_esc(identity.get('entity_class'))} "
            f"({t['cross_camera']}: {cross})</h3>"
        )
        parts.append(
            f"<table><tr><th>{t['camera']}</th><th>{t['track']}</th>"
            f"<th>{t['from']}</th><th>{t['to']}</th><th>{t['thumbnail']}</th>"
            f"<th>{t['clip']}</th></tr>"
        )
        for member in identity.get("members", []):
            parts.append(
                f"<tr><td>{_esc(member.get('camera_id'))}</td>"
                f"<td>{_esc(member.get('track_id'))}</td>"
                f"<td>{_esc(member.get('start_ts_scene_s'))}</td>"
                f"<td>{_esc(member.get('end_ts_scene_s'))}</td>"
                f"<td>{_thumb_cell(member.get('thumbnail'), run_dir)}</td>"
                f"<td>{_clip_cell(member)}</td></tr>"
            )
        parts.append("</table>")

    parts.append(f"<footer>{t['footer']}</footer></body></html>")
    return "".join(parts)
