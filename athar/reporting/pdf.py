"""HTML -> PDF through headless chromium (Playwright).

Chromium is the one renderer that shapes Arabic correctly (D19;
WeasyPrint is Arabic-broken). The browser binary is installed once at
deployment time (``playwright install chromium``) — printing itself
makes no network requests and the input document is self-contained.
"""

from __future__ import annotations


class ReportError(RuntimeError):
    """Report rendering is unavailable or failed."""


def html_to_pdf(html: str) -> bytes:
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - env without playwright
        raise ReportError(
            "playwright is not installed; PDF export needs the 'playwright' "
            "package plus 'playwright install chromium'"
        ) from exc

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                page = browser.new_page()
                page.set_content(html, wait_until="load")
                return page.pdf(format="A4", print_background=True)
            finally:
                browser.close()
    except PlaywrightError as exc:
        raise ReportError(f"PDF rendering failed: {exc}") from exc
