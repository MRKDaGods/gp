# ATHAR v2 — Technology Stack Decisions

> Research date: **2026-07-21** (all versions verified against live registries).
> Decision rationale lives here; the checklist lives in [ROADMAP.md](../ROADMAP.md).
> v1's dependency setup (4-step `--no-deps` ritual, stale pins, dead libraries)
> is fully superseded by this document.

## Runtime & toolchain

| Decision | Choice | Rationale |
|---|---|---|
| Python | **3.13** (`>=3.13,<3.14`) | numpy 2.5 sets the ≥3.12 floor; boxmot 22 sets the <3.14 ceiling. Avoid free-threaded builds (torch dropped 3.13t). |
| Env/packaging | **uv** (project mode: `uv lock` / `uv sync`) | De-facto standard 2026. Torch CUDA index via `[tool.uv.sources]` + explicit `[[tool.uv.index]]` (cu130 Linux GPU, CPU wheels elsewhere). |
| Node / pkg mgr | **Node 24 LTS + pnpm 11** | Node 24 = Active LTS. Bun's `--production` reliability caveats are the wrong trade for air-gapped installs. |
| TypeScript | **5.9.x pinned** | TS 7.0 (Go-native) is `latest` on npm but ecosystem tooling compat is unproven; revisit deliberately, not by drift. |

## ML / CV stack

| Component | Choice | Rationale |
|---|---|---|
| PyTorch | **torch 2.13.0** + paired torchvision (cu130 on GPU hosts — driver ≥580; cu126 fallback; CPU wheels for light serving) | cu130 is the stable PyPI default; cu128 was removed in the 2.12 cycle. |
| Detector | **ultralytics 8.4.103 — YOLO26** generation. Production: **YOLO26x @ imgsz 1280** (COCO 57.5 mAP). **Parity profile: YOLO26m** (v1's detector — gates must match v1 components). | Accuracy-first forensic use: x-vs-m is +4.4 mAP and offline throughput is not the constraint. NMS-free e2e, STAL small-object gains suit distant CCTV targets. |
| 2nd-pass detector (optional) | **RF-DETR-XL/2XL** (`rfdetr`, 58.6–60.1 AP) | DETR-family beats YOLO on pure AP and degrades more gracefully in crowds (no NMS suppressing overlapping people). High-recall verification pass; licensing accepted per D12. |
| Tracker | **boxmot 22.0.0**. Production default: **BoostTrack + ReID**; fallback BoT-SORT-ReID. **Parity profile: v1's tracker + config.** | boxmot 19–22 fixed the broken metadata (no more `--no-deps`), added C++ trackers + built-in benchmark eval/tuner. BoostTrack tops MOT17/MOT20 among online methods. |
| OpenCV | **opencv-python-headless 4.13.0.92** — NOT 5.x | boxmot caps `opencv-python<5`; v1's 5.0.0.93 pin cannot carry over. Suppress boxmot's non-headless dep via uv `override-dependencies` (avoid duplicate `cv2`). Avoid 4.13.0.90 exactly (ultralytics-excluded FIPS crash). |
| Video decode | **torchcodec 0.15.0** primary (`seek_mode="exact"`, NVDEC on Linux, batch-to-tensor); **PyAV 18** for probing/metadata/odd codecs | PyTorch-blessed successor (torchvision/torchaudio decoders deprecated). cv2.VideoCapture seeking is NOT frame-accurate on long-GOP CCTV — v1's approach is retired. decord is dead (2021). |
| Vector search | **faiss-cpu 1.14.3**, `IndexFlatIP` exact; optional HNSW for interactive UX only | Exact search at 100k–1M vectors is milliseconds on CPU and forensically defensible (zero ANN recall loss). faiss-gpu: no official wheels, Linux-only community builds — skip; keeps Windows parity. |
| MOT metrics | **boxmot built-in eval** going forward; **sn-trackeval** (maintained fork) for parity-gate comparability with v1's TrackEval numbers | Upstream TrackEval is dormant (numpy-2 casualties); py-motmetrics discontinued — both dropped. |
| numpy | **2.5.1** | Whole stack is numpy-2 clean now. |
| Person ReID | Port v1 TransReID checkpoints (parity); **CION ReIDZoo** pretrained backbones = tier-2 retrain candidates (93.3 mAP Market / 74.3 MSMT17) | CION is the current person-ReID pretraining foundation with public weights. torchreid is frozen (2022) — dropped. |
| Vehicle ReID | Port v1 TransReID/CLIP-SENet/DINOv2 fusion (parity); joint multi-domain retrain per D4 | No CION-equivalent foundation for vehicles; CLIP-based remains SOTA on VeRi. |
| Face | **insightface 1.0.1**, `buffalo_l` pack (ONNX Runtime CPU backend) | Active (1.0 May 2026); buffalo_l recommended over antelopev2 by maintainers. |
| Gait | **OpenGait** (DeepGaitV2 / SkeletonGait++ baselines) | De-facto framework, TPAMI 2025; FoundationGait is the research edge to watch. |
| Inference optimization | **torch.compile** (`max-autotune`) default; **TensorRT via ultralytics export** for the detector only; **onnxruntime 1.27 CPU-only** | ORT-GPU on PyPI is CUDA 12.x — clashes with torch cu130. One GPU runtime (torch/TRT); ORT stays CPU for insightface + light serving. |

## Backend

| Component | Choice | Rationale |
|---|---|---|
| API | **FastAPI 0.139** + **uvicorn 0.51** (native multi-worker; no gunicorn) | Granian's edge is irrelevant when the bottleneck is GPU + SQLite. |
| DB | **SQLAlchemy 2.0.51 + aiosqlite + alembic**, SQLite WAL (`synchronous=NORMAL`, `busy_timeout=5000`) | sqlmodel lags SQLAlchemy; raw sqlite too low-level for the app DB. |
| Job queue | Hand-rolled `jobs` table (`UPDATE…RETURNING` claims, worker heartbeats) in a **separate SQLite file**, workers as **separate processes** | The 2026 SQLite-queue pattern; separate file avoids app-DB write contention; separate process avoids N web workers each running queues. |
| Auth | **Server-side sessions** (opaque HttpOnly cookie, sessions table) + **pwdlib\[argon2\]**; RBAC via `Depends(require_permission(...))` role→permission table | fastapi-users is maintenance-mode; passlib is dead on 3.13. Sessions are instantly revocable — right model for audited forensic tools. No external IdP (air-gapped). |
| Realtime | **SSE** via sse-starlette → browser EventSource, direct to FastAPI | One-directional job progress; proxy-friendly on-prem; auto-reconnect. WebSocket only if client→server streaming appears. |
| Audit/logging | **structlog** (structured JSON) + append-only hash-chained audit table | Chain-of-custody requirement (council red team). |
| PDF reports | **Playwright/Chromium `page.pdf()`** — reports reuse dashboard HTML/CSS | WeasyPrint has documented Arabic bidi bugs (#1686). Browser engine gives correct shaping free; vendor Chromium for air-gap (`PLAYWRIGHT_BROWSERS_PATH`). |

## Frontend

| Component | Choice | Rationale |
|---|---|---|
| Framework | **Next.js 16.2 + React 19.2** (App Router, Turbopack, Cache Components) | Auth-gated all-dynamic dashboard: fetch in Server Components, mutate via generated API client. |
| Styling | **Tailwind CSS 4.3** (CSS-first config, logical properties `ms-*`/`me-*`) | Logical properties make RTL near-free — mandatory for Arabic UI. |
| Components | **shadcn/ui (CLI v4) on Base UI** + next-themes | shadcn defaults to Base UI since Jan 2026 (MUI team, 1.0 stable, richer set: combobox/multi-select). Radix is slowing post-acquisition. |
| i18n/RTL | **next-intl 4** + `<html dir>` + logical properties | App Router standard, deep RSC integration, `[locale]` routing. |
| Tables | **TanStack Table v8** + react-virtual v3 (+ TanStack Query 5) | v9 still beta (July 2026) — not for production. |
| Maps (air-gapped) | **MapLibre GL 5 + PMTiles 4** (Protomaps regional extract + self-hosted glyphs/sprites) | THE 2026 offline-maps answer: one `.pmtiles` file, no tile server, zero external calls. |
| Charts | **Recharts** (standard) + **ECharts** (canvas, >10k points) | Recharts = shadcn-charts default; SVG chokes on dense series. |
| Video review | `<video>` + **requestVideoFrameCallback** + **react-konva** overlays; **WebCodecs** for frame-exact stepping; media-chrome player UI | No off-the-shelf annotated-review lib exists — we build this layer. Konva is reused for the cross-camera timeline (custom canvas — no good off-the-shelf timeline in 2026). |
| API client | **@hey-api/openapi-ts** (pinned exactly — pre-1.0) + TanStack Query plugin | Successor to openapi-typescript-codegen; codegen runs outside the air gap. |
| Fonts | `next/font/local`, vendored Arabic-capable fonts (IBM Plex Sans Arabic / Noto Naskh) — same files feed PDF pipeline | Never Google Fonts at runtime (air gap). |

## Pre-verified conflicts & constraints

1. **boxmot ⛌ OpenCV 5** — pin headless 4.13.0.92 + uv override (above).
2. **torch cu130 ⛌ onnxruntime-gpu (CUDA 12)** — ORT stays CPU-only; GPU inference via torch/TRT exclusively.
3. **Python window is exactly 3.12–3.13** (numpy floor, boxmot ceiling) → 3.13.
4. **torchvision pairing** — let uv resolve the torch-2.13 partner at lock time.
5. **Parity vs production profiles** — parity gates (P1–P3) MUST pin v1 components (YOLO26m, v1 tracker+config, TrackEval-compatible metrics via sn-trackeval). Upgrades (YOLO26x, BoostTrack, RF-DETR) live only in production profiles and get benchmarked against the parity baseline — never silently swapped.
6. **Air-gap install bundle must vendor**: Playwright Chromium, `.pmtiles` + glyphs/sprites, fonts, all wheels (devpi/verdaccio mirrors for in-gap rebuilds).

## Dropped from v1 (dead/superseded)

`decord` (dead 2021) · `py-motmetrics` (discontinued) · upstream `TrackEval` (dormant; sn-trackeval fork for parity only) · `torchreid` (frozen 2022) · `passlib` (broken ≥3.13) · OpenCV 5 pin (boxmot conflict) · cv2-based frame seeking (not frame-accurate) · Streamlit apps · gunicorn.
