"""
Capture the animation map box as individual PNG frames using Playwright headless Chromium.
No DPR scaling issues — uses device_scale_factor=1 in a fresh browser context.
"""
import time
import os
import pathlib
from playwright.sync_api import sync_playwright

OUT_DIR = pathlib.Path(__file__).parent / "frames_map_capture"
OUT_DIR.mkdir(exist_ok=True)
# Clear old frames
for f in OUT_DIR.glob("frame_*.png"):
    f.unlink()

HTML_FILE = pathlib.Path(__file__).parent / "index.html"
URL = HTML_FILE.as_uri()

FPS = 30
DURATION_S = 16          # capture a touch longer than the 15s animation
TOTAL_FRAMES = FPS * DURATION_S
FRAME_MS = 1000 / FPS

with sync_playwright() as p:
    # Fresh browser — device_scale_factor=1 removes DPR effects entirely
    browser = p.chromium.launch(headless=True)
    context = browser.new_context(
        viewport={"width": 1400, "height": 900},
        device_scale_factor=1,
    )
    page = context.new_page()
    page.goto(URL, wait_until="domcontentloaded")

    # Get actual map dimensions (should be 800x600 at DPR=1, viewport 1400)
    dims = page.evaluate("""() => ({
        dpr: window.devicePixelRatio,
        innerW: window.innerWidth,
        mapW: document.querySelector('.map-container').offsetWidth,
        mapH: document.querySelector('.map-container').offsetHeight,
    })""")
    print(f"DPR={dims['dpr']}  innerW={dims['innerW']}  map={dims['mapW']}x{dims['mapH']}")

    map_w = dims["mapW"]
    map_h = dims["mapH"]

    # If map is not 800x600, rescale camera coords to fit the actual container
    if abs(map_w - 800) > 5 or abs(map_h - 600) > 5:
        print(f"Rescaling camera coords from 800x600 -> {map_w}x{map_h}")
        page.evaluate(f"""() => {{
            const scaleX = {map_w} / 800;
            const scaleY = {map_h} / 600;
            CONFIG.cameras.forEach(cam => {{
                cam.x = Math.round(cam.x * scaleX);
                cam.y = Math.round(cam.y * scaleY);
            }});
            CONFIG.radarMaxRadiusPx = Math.round(CONFIG.radarMaxRadiusPx * Math.min(scaleX, scaleY));
            CONFIG.radarMinRadiusPx = Math.round(CONFIG.radarMinRadiusPx * Math.min(scaleX, scaleY));
            const size = 40 * CONFIG.cameraScale;
            document.querySelectorAll('.camera').forEach(pin => {{
                const label = pin.textContent.trim();
                const cam = CONFIG.cameras.find(c => String(c.id) === label);
                if (cam) {{
                    pin.style.left = (cam.x - size / 2) + 'px';
                    pin.style.top  = (cam.y - size / 2) + 'px';
                }}
            }});
            resetAnimation();
        }}""")

    # Start animation
    page.evaluate("() => document.getElementById('play-btn').click()")

    # Hide header and controls so only the map renders
    page.add_style_tag(content="""
        .header   { display: none !important; }
        .controls { display: none !important; }
        body, html { margin: 0; padding: 0; overflow: hidden; background: #0f172a; }
        .map-container {
            margin: 0 !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            border: none !important;
        }
    """)

    map_el = page.locator(".map-container").element_handle()

    print(f"Capturing {TOTAL_FRAMES} frames at {FPS}fps ...")
    start = time.monotonic()
    for i in range(TOTAL_FRAMES):
        idx = str(i).zfill(4)
        map_el.screenshot(path=str(OUT_DIR / f"frame_{idx}.png"))
        # Pace to ~30fps
        elapsed = time.monotonic() - start
        target  = (i + 1) / FPS
        remaining = target - elapsed
        if remaining > 0:
            time.sleep(remaining)

    print("Done. Frames saved to:", OUT_DIR)
    browser.close()
