"""Report export: chain-of-custody case/run reports (D19: Playwright PDF).

``html`` renders a fully self-contained document (inline CSS, thumbnails
embedded as data URIs — nothing fetched at view time, air-gap safe);
``pdf`` prints it through headless chromium, the only renderer that
shapes Arabic script correctly (WeasyPrint is Arabic-broken, D19).
"""

from athar.reporting.case import render_case_report_html
from athar.reporting.html import (
    load_weight_shas,
    models_from_config,
    render_report_html,
)
from athar.reporting.pdf import ReportError, html_to_pdf

__all__ = [
    "ReportError",
    "html_to_pdf",
    "load_weight_shas",
    "models_from_config",
    "render_case_report_html",
    "render_report_html",
]
