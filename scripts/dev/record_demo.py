"""Record the MIE finals demo happy path as a 1080p video + screenshots.

Drives the real web app (localhost:3000) with Playwright and records the
full click-path the live demo follows: login -> cases -> case workspace
(targets, confirmed match, audit-backed hypotheses) -> live probe search
-> gallery run timeline (cross-camera identity -> evidence panel + clip
playback) -> map view -> PDF report download. Output lands in
mie-competition/assets/ as the stage fallback if the live demo dies.

Uses the system Edge/Chrome channel (NOT Playwright's bundled chromium:
it has no H.264 codecs, and the evidence clips are H.264 MP4 — the
player would render black in the recording).

Prereqs: API on :8000, web dev server on :3000, the finals case seeded
(scripts/dev/seed_demo_case.py).

Usage:
  python scripts/dev/record_demo.py --case CASE_ID --gallery RUN_ID \
      [--locale ar] [--out mie-competition/assets]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE = "http://localhost:3000"


def shot(page, out_dir: Path, name: str) -> None:
    path = out_dir / f"{name}.png"
    page.screenshot(path=str(path), full_page=False)
    print(f"  shot: {path.name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--gallery", required=True)
    parser.add_argument("--locale", default="ar", choices=("ar", "en"))
    parser.add_argument("--out", default="mie-competition/assets")
    parser.add_argument("--user", default="demo")
    parser.add_argument("--password", default="demo-pass-1")
    parser.add_argument("--theme", default="dark", choices=("dark", "light"))
    parser.add_argument("--identity", type=int, default=None,
                        help="global_id of the timeline identity to feature "
                             "(default: first cross-camera span)")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = out_dir / "_video_tmp"

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = None
        for channel in ("msedge", "chrome"):
            try:
                browser = p.chromium.launch(channel=channel, headless=True)
                print(f"browser channel: {channel}")
                break
            except Exception as exc:
                print(f"channel {channel} unavailable: {exc}")
        if browser is None:
            print("no branded browser channel (H.264 needed) — aborting",
                  file=sys.stderr)
            return 1

        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            record_video_dir=str(video_dir),
            record_video_size={"width": 1920, "height": 1080},
            color_scheme="dark" if args.theme == "dark" else "light",
        )
        page = context.new_page()
        L = args.locale

        # ---- login -----------------------------------------------------
        page.goto(f"{BASE}/{L}/login", wait_until="load")
        page.fill("#username", args.user)
        page.fill("#password", args.password)
        shot(page, out_dir, f"01-login-{L}")
        page.click("button[type='submit']")
        page.wait_for_load_state("load")
        time.sleep(1)

        # ---- cases list -------------------------------------------------
        page.goto(f"{BASE}/{L}/cases", wait_until="load")
        time.sleep(1.5)
        shot(page, out_dir, f"02-cases-{L}")

        # ---- case workspace --------------------------------------------
        page.goto(f"{BASE}/{L}/cases/{args.case}", wait_until="load")
        time.sleep(2.5)
        shot(page, out_dir, f"03-case-workspace-{L}")

        # ---- gallery run: timeline + map + evidence --------------------
        page.goto(f"{BASE}/{L}/runs/{args.gallery}", wait_until="load")
        time.sleep(6)  # timeline + map render
        # the live events feed scrolls the page — pin back to the timeline
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(1.5)
        shot(page, out_dir, f"04-run-timeline-{L}")

        # click a cross-camera identity span (spans are slivers — JS click;
        # cross-camera spans carry the border class)
        selector = (
            f'button[title^="#{args.identity} "]'
            if args.identity is not None
            else 'div[dir="ltr"] button.border'
        )
        clicked = page.evaluate(
            f"() => {{ const el = document.querySelector('{selector}');"
            "if (el) el.click(); return !!el; }"
        )
        if clicked:
            time.sleep(2)
            page.evaluate(
                "document.querySelector('video')?.scrollIntoView({block: 'center'})"
            )
            time.sleep(1)
            shot(page, out_dir, f"05-evidence-panel-{L}")
            video_el = page.query_selector("video")
            if video_el is not None:
                # let the clip actually play a few seconds into the recording
                page.evaluate("v => v.play()", video_el)
                time.sleep(7)
                shot(page, out_dir, f"06-clip-playing-{L}")

        # scroll to the map
        page.evaluate(
            "document.querySelector('.maplibregl-map')?.scrollIntoView({block: 'center'})"
        )
        time.sleep(2.5)
        shot(page, out_dir, f"07-map-{L}")

        # ---- PDF report -------------------------------------------------
        with page.expect_download(timeout=120000) as download_info:
            for text in ("تقرير PDF", "PDF report", "PDF"):
                btn = page.get_by_text(text, exact=False).first
                try:
                    btn.click(timeout=3000)
                    break
                except Exception:
                    continue
        download = download_info.value
        pdf_path = out_dir / f"athar-report-{args.gallery}-{L}.pdf"
        download.save_as(str(pdf_path))
        print(f"  report: {pdf_path.name} ({pdf_path.stat().st_size / 1e3:.0f} KB)")
        time.sleep(2)

        page.close()
        video_path = page.video.path() if page.video else None
        context.close()
        browser.close()

        if video_path:
            final = out_dir / f"demo-happy-path-{L}-{args.theme}.webm"
            Path(video_path).replace(final)
            print(f"video: {final} ({final.stat().st_size / 1e6:.1f} MB)")
        for leftover in video_dir.glob("*"):
            leftover.unlink()
        video_dir.rmdir() if video_dir.exists() else None

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
