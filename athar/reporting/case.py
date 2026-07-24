"""Case dossier HTML: the investigation-level companion to the per-run
chain-of-custody report.

Joins what the run report cannot see: the case file (title, owner,
status), every attached evidence run WITH its ingest hashes, the targets
and their attributed hypothesis decisions (who proposed, who confirmed or
rejected, when, on what score), and the case's slice of the hash-chained
audit log. Rendering and printing reuse the run-report machinery
(self-contained HTML -> Playwright PDF).
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Sequence

from athar.reporting.html import _CSS, _esc

_LABELS = {
    "ar": {
        "title": "ملف القضية — أثر",
        "case": "القضية",
        "case_title": "العنوان",
        "status": "الحالة",
        "owner": "المالك",
        "created": "الإنشاء",
        "updated": "آخر تحديث",
        "evidence_runs": "معالجات الأدلة المرفقة",
        "run": "المعالجة",
        "role": "الدور",
        "profile": "الملف",
        "config_hash": "بصمة الإعدادات",
        "camera": "الكاميرا",
        "sha256": "SHA-256",
        "attached_by": "أرفقها",
        "attached_at": "تاريخ الإرفاق",
        "manifest_missing": "بيان المعالجة غير متوفر على القرص",
        "no_runs": "لا توجد معالجات مرفقة",
        "targets": "الأهداف والقرارات",
        "target": "الهدف",
        "created_by": "أنشأه",
        "members": "المسارات المؤكدة",
        "track": "المسار",
        "no_members": "لا توجد مسارات مؤكدة",
        "hypotheses": "سجل الفرضيات",
        "kind": "النوع",
        "stream": "القناة",
        "score": "الدرجة",
        "probability": "الاحتمال",
        "uncalibrated": "غير معايَر",
        "decision": "القرار",
        "proposed_by": "اقترحها",
        "decided_by": "قررها",
        "decided_at": "تاريخ القرار",
        "pending": "قيد المراجعة",
        "confirmed": "مؤكدة",
        "rejected": "مرفوضة",
        "no_hypotheses": "لا توجد فرضيات",
        "no_targets": "لا توجد أهداف",
        "audit": "سجل التدقيق (مقتطف القضية)",
        "seq": "التسلسل",
        "ts": "التوقيت",
        "actor": "المستخدم",
        "action": "الإجراء",
        "detail": "التفاصيل",
        "hash": "البصمة",
        "no_audit": "لا توجد سجلات تدقيق لهذه القضية",
        "footer": "وثيقة مولدة آليًا؛ سجلات التدقيق أعلاه جزء من سلسلة بصمات "
        "مترابطة يمكن التحقق من سلامتها كاملة عبر واجهة التدقيق.",
    },
    "en": {
        "title": "Case File — ATHAR",
        "case": "Case",
        "case_title": "Title",
        "status": "Status",
        "owner": "Owner",
        "created": "Created",
        "updated": "Updated",
        "evidence_runs": "Attached evidence runs",
        "run": "Run",
        "role": "Role",
        "profile": "Profile",
        "config_hash": "Config hash",
        "camera": "Camera",
        "sha256": "SHA-256",
        "attached_by": "Attached by",
        "attached_at": "Attached at",
        "manifest_missing": "run manifest not on disk",
        "no_runs": "no runs attached",
        "targets": "Targets and decisions",
        "target": "Target",
        "created_by": "Created by",
        "members": "Confirmed tracklets",
        "track": "Track",
        "no_members": "no confirmed tracklets",
        "hypotheses": "Hypothesis log",
        "kind": "Kind",
        "stream": "Stream",
        "score": "Score",
        "probability": "Probability",
        "uncalibrated": "uncalibrated",
        "decision": "Decision",
        "proposed_by": "Proposed by",
        "decided_by": "Decided by",
        "decided_at": "Decided at",
        "pending": "pending",
        "confirmed": "confirmed",
        "rejected": "rejected",
        "no_hypotheses": "no hypotheses",
        "no_targets": "no targets",
        "audit": "Audit log (case slice)",
        "seq": "Seq",
        "ts": "Timestamp",
        "actor": "Actor",
        "action": "Action",
        "detail": "Detail",
        "hash": "Hash",
        "no_audit": "no audit records for this case",
        "footer": "Machine-generated; the audit rows above belong to a "
        "hash-chained log whose full integrity is verifiable through the "
        "audit API.",
    },
}


def _detail_cell(detail_json: str) -> str:
    """Compact detail rendering: drop the case_id every row shares."""
    try:
        detail = json.loads(detail_json)
    except (TypeError, ValueError):
        return _esc(detail_json)
    detail.pop("case_id", None)
    return _esc(json.dumps(detail, ensure_ascii=False, sort_keys=True)) if detail else ""


def render_case_report_html(
    case: Mapping[str, Any],
    audit_slice: Sequence[Mapping[str, Any]],
    *,
    run_evidence: Optional[Sequence[Mapping[str, Any]]] = None,
    locale: str = "ar",
) -> str:
    """``case`` is a CaseDetail dump; ``run_evidence`` optionally enriches
    each attached run with its manifest facts ({run_id, profile,
    config_hash, cameras: [{camera_id, sha256}]}, or {run_id,
    missing: True} when the store no longer has it)."""
    t = _LABELS.get(locale, _LABELS["ar"])
    direction = "rtl" if locale == "ar" else "ltr"
    evidence_by_run = {e["run_id"]: e for e in (run_evidence or [])}

    parts: list[str] = [
        f'<!DOCTYPE html><html lang="{locale}" dir="{direction}"><head>'
        f'<meta charset="utf-8"><title>{t["title"]}</title>'
        f"<style>{_CSS}</style></head><body>",
        f"<h1>{t['title']}</h1>",
        "<table>",
        f"<tr><th>{t['case']}</th><td><code>{_esc(case.get('case_id'))}</code></td>"
        f"<th>{t['status']}</th><td>{_esc(case.get('status'))}</td></tr>",
        f"<tr><th>{t['case_title']}</th><td>{_esc(case.get('title'))}</td>"
        f"<th>{t['owner']}</th><td>{_esc(case.get('owner'))}</td></tr>",
        f"<tr><th>{t['created']}</th><td><code>{_esc(case.get('created_at'))}</code></td>"
        f"<th>{t['updated']}</th><td><code>{_esc(case.get('updated_at'))}</code></td></tr>",
        "</table>",
    ]

    parts.append(f"<h2>{t['evidence_runs']}</h2>")
    runs = case.get("runs", [])
    if not runs:
        parts.append(f"<p><i>{t['no_runs']}</i></p>")
    for run in runs:
        evidence = evidence_by_run.get(run["run_id"], {})
        parts.append(
            f"<h3><code>{_esc(run.get('run_id'))}</code> — {_esc(run.get('role'))}</h3>"
        )
        parts.append("<table>")
        parts.append(
            f"<tr><th>{t['attached_by']}</th><td>{_esc(run.get('attached_by'))}</td>"
            f"<th>{t['attached_at']}</th><td><code>{_esc(run.get('attached_at'))}</code></td></tr>"
        )
        if evidence.get("missing"):
            parts.append(
                f"<tr><td colspan=4><i>{t['manifest_missing']}</i></td></tr></table>"
            )
            continue
        parts.append(
            f"<tr><th>{t['profile']}</th><td>{_esc(evidence.get('profile'))}</td>"
            f"<th>{t['config_hash']}</th>"
            f"<td><code>{_esc(evidence.get('config_hash'))}</code></td></tr>"
        )
        parts.append("</table>")
        cameras = evidence.get("cameras") or []
        if cameras:
            parts.append(
                f"<table><tr><th>{t['camera']}</th><th>{t['sha256']}</th></tr>"
            )
            for camera in cameras:
                parts.append(
                    f"<tr><td>{_esc(camera.get('camera_id'))}</td>"
                    f"<td><code>{_esc(camera.get('sha256'))}</code></td></tr>"
                )
            parts.append("</table>")

    parts.append(f"<h2>{t['targets']}</h2>")
    targets = case.get("targets", [])
    if not targets:
        parts.append(f"<p><i>{t['no_targets']}</i></p>")
    for target in targets:
        parts.append(
            f"<h3>{t['target']}: {_esc(target.get('label'))} "
            f"(<code>{_esc(target.get('target_id'))}</code>, "
            f"{t['created_by']}: {_esc(target.get('created_by'))})</h3>"
        )
        members = target.get("members", [])
        parts.append(f"<p><b>{t['members']}</b></p>")
        if members:
            parts.append(
                f"<table><tr><th>{t['run']}</th><th>{t['camera']}</th>"
                f"<th>{t['track']}</th></tr>"
            )
            for member in members:
                parts.append(
                    f"<tr><td><code>{_esc(member.get('run_id'))}</code></td>"
                    f"<td>{_esc(member.get('camera_id'))}</td>"
                    f"<td>{_esc(member.get('track_id'))}</td></tr>"
                )
            parts.append("</table>")
        else:
            parts.append(f"<p><i>{t['no_members']}</i></p>")

        hypotheses = target.get("hypotheses", [])
        parts.append(f"<p><b>{t['hypotheses']}</b></p>")
        if hypotheses:
            parts.append(
                f"<table><tr><th>{t['kind']}</th><th>{t['run']}</th>"
                f"<th>{t['camera']}</th><th>{t['track']}</th><th>{t['stream']}</th>"
                f"<th>{t['score']}</th><th>{t['probability']}</th>"
                f"<th>{t['decision']}</th><th>{t['proposed_by']}</th>"
                f"<th>{t['decided_by']}</th><th>{t['decided_at']}</th></tr>"
            )
            for hyp in hypotheses:
                probability = hyp.get("probability")
                probability_cell = (
                    f"{probability:.3f}" if probability is not None
                    else f"<i>{t['uncalibrated']}</i>"
                )
                decision = t.get(hyp.get("status", ""), _esc(hyp.get("status")))
                parts.append(
                    f"<tr><td>{_esc(hyp.get('kind'))}</td>"
                    f"<td><code>{_esc(hyp.get('run_id'))}</code></td>"
                    f"<td>{_esc(hyp.get('camera_id'))}</td>"
                    f"<td>{_esc(hyp.get('track_id'))}</td>"
                    f"<td>{_esc(hyp.get('stream'))}</td>"
                    f"<td>{_esc(round(hyp.get('raw_score', 0.0), 4))}</td>"
                    f"<td>{probability_cell}</td>"
                    f"<td>{decision}</td>"
                    f"<td>{_esc(hyp.get('proposed_by'))}</td>"
                    f"<td>{_esc(hyp.get('decided_by'))}</td>"
                    f"<td><code>{_esc(hyp.get('decided_at'))}</code></td></tr>"
                )
            parts.append("</table>")
        else:
            parts.append(f"<p><i>{t['no_hypotheses']}</i></p>")

    parts.append(f"<h2>{t['audit']}</h2>")
    if audit_slice:
        parts.append(
            f"<table><tr><th>{t['seq']}</th><th>{t['ts']}</th>"
            f"<th>{t['actor']}</th><th>{t['action']}</th><th>{t['detail']}</th>"
            f"<th>{t['hash']}</th></tr>"
        )
        for row in audit_slice:
            parts.append(
                f"<tr><td>{_esc(row.get('seq'))}</td>"
                f"<td><code>{_esc(row.get('ts'))}</code></td>"
                f"<td>{_esc(row.get('actor'))}</td>"
                f"<td>{_esc(row.get('action'))}</td>"
                f"<td><code>{_detail_cell(row.get('detail', ''))}</code></td>"
                f"<td><code>{_esc(str(row.get('hash', ''))[:16])}</code></td></tr>"
            )
        parts.append("</table>")
    else:
        parts.append(f"<p><i>{t['no_audit']}</i></p>")

    parts.append(f"<footer>{t['footer']}</footer></body></html>")
    return "".join(parts)
