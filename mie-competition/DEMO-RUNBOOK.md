# ATHAR — MIE Finals Demo Runbook

Finals: Saturday 2026-08-01, Nile University. Team code CS140.
Everything below runs OFFLINE on the demo laptop — no internet needed
once set up (air-gap by design; maps, fonts, weights all self-hosted).

The demo data is a REAL end-to-end run: 4 gallery cameras + 1 probe
camera of the El Shorouk compound (21.6 min each, synchronized), fully
processed by the production profile (YOLO26 detection, BotSort tracking,
5 ReID streams, FAISS indexing, cross-camera association, evidence
packaging) on a Kaggle T4 — imported here with the run store + footage.

## 0. One-time setup checklist (do this the night before)

- [ ] Repo at `E:\dev\src\gp`, branch `athar-v2`, latest pull.
- [ ] `.venv-v2` exists (CPU torch env — serving only, no GPU needed).
- [ ] `web/` deps installed: `pnpm --dir web install`.
- [ ] Demo runs present under `data/runs/`:
      gallery `run-20260730-201635-81157c`, probe `run-20260731-005450-b6a70b`
      (re-import: `python scripts/kaggle/shorouk_demo/fetch_results.py`).
- [ ] Footage present: `data/raw/shorouk_demo/c017,c018,c019,c020,c021/vdo.mp4`
      (re-download: `python scripts/kaggle/shorouk_demo/fetch_footage.py`).
- [ ] Demo case seeded: "El Shorouk — Finals Demo" (see §3).
- [ ] Dry-run the click-path (§4) once, end to end.

## 1. Boot (2 terminals, both from repo root)

Terminal 1 — API:

```bash
ATHAR_COOKIE_SECURE=0 .venv-v2/Scripts/python.exe -m uvicorn --factory athar.api.app:create_app --host 127.0.0.1 --port 8000
```

Terminal 2 — web:

```bash
pnpm --dir web dev
```

Open **http://localhost:3000** (MUST be `localhost`, NOT `127.0.0.1` —
the session cookie is SameSite and the two origins count as different
sites; login silently loops on 127.0.0.1).

Wait for "Ready" from Next (~5s) and check http://127.0.0.1:8000/health
returns `{"status":"ok"}`.

## 2. Login

| user | password | role | use |
|------|----------|------|-----|
| `demo` | `demo-pass-1` | investigator | THE demo login (owns the demo case) |
| `mie-shots` | `mie-finals-2026` | admin | backup/admin (sees all cases) |

Arabic RTL is the default (`/ar/...`); English mirror at `/en/...`.
Theme toggle is in the top bar — rehearse in **dark**.

## 3. Demo case (pre-seeded — only reseed if lost)

```bash
.venv-v2/Scripts/python.exe scripts/dev/seed_demo_case.py --gallery run-20260730-201635-81157c --probe run-20260731-005450-b6a70b
```

Creates "El Shorouk — Finals Demo" as `demo`: attaches both runs, runs a
probe→gallery search on the vehicle stream (`transreid_primary`) and the
person stream (`transreid_person`), files the top hits as hypotheses on
two targets, confirms the best of each — all through the audited API, so
the audit trail is genuine.

### What the data shows (verified 2026-07-31)

- Current seeded case: `case-20260731-020923-301c1b`. 574 identities in
  the gallery, **17 cross-camera** (10 vehicles, 7 persons), 342 pre-cut
  evidence clips.
- The timeline defaults to **"عبر الكاميرات فقط"** (cross-camera only) on
  dense runs — the toggle chip switches back to all identities.
- **Star identities to click on stage** (clean, verified clips):
  - **#109 — person** walking c018 → c019 → c021, confidence 65.6%,
    spatiotemporal evidence 0.969. The clip shows him mid-frame.
  - **#86 — car** c017 → c019, confidence 65.9%.
- The three LARGE identities (#5, #6 cars; #3 person) are appearance
  clusters — the graph groups the compound's recurring similar white
  cars / similar pedestrians. If asked, that IS the design (D7): clusters
  are ranked hypotheses; the investigator confirms/rejects per tracklet
  in the case workspace — never auto-asserted as one individual.
- Search scores show "(غير معاير)" — uncalibrated — deliberately: no
  probability is invented without a per-deployment calibration fit.
  That's the forensic-honesty pitch, not a gap.

## 4. The 3-minute click-path (rehearse this)

Story: *"Footage of a person and a vehicle of interest was captured at
camera c020. Find where else they appear across the compound."*

1. **Login** as `demo` → lands on القضايا (Cases). ~10s
2. Open **"El Shorouk — Finals Demo"** → case workspace: two targets
   (vehicle + person of interest), each with ranked hypotheses and one
   CONFIRMED cross-camera match; every action attributed + audited. ~30s
3. In the search panel: pick probe `run-20260731-005450-b6a70b` → run a live search
   (vehicle stream) → ranked hits with scores appear in ~1s (CPU FAISS).
   Attach a hit as a hypothesis live if asked "is this real?". ~40s
4. Click through to the **gallery run** `run-20260730-201635-81157c` →
   **cross-camera timeline**: per-camera lanes, colored identity spans
   (cross-camera-only filter is on by default). Click **identity #109**
   (blue spans on c018/c019/c021) → evidence panel: per-term match
   evidence (appearance/HSV/spatiotemporal bars), thumbnail, and the
   **evidence clip playing inline** — the person visibly walks through
   the frame. ~60s
5. Scroll to the **map**: camera pins on the actual compound (offline
   basemap), dashed lines tracing the identity's cross-camera journey.
   ~20s
6. Click **تقرير PDF** → Arabic chain-of-custody report downloads
   (evidence SHA-256 → config hash → model SHAs → identities with
   clips; ~7.7 MB, takes ~5s to generate — don't re-click). Open it,
   scroll once. ~25s

Total ≈ 3 min. Backup: the same path recorded at
`mie-competition/assets/demo-happy-path-ar-dark.webm`.

## 5. Troubleshooting

| symptom | fix |
|---------|-----|
| Port 8000/3000 in use | `netstat -ano \| findstr :8000` → `taskkill /PID <pid> /F` (same for 3000) |
| Login loops back to the login page | You opened `127.0.0.1:3000` — use `http://localhost:3000` |
| Clip shows "footage not on disk" (404) | `python scripts/kaggle/shorouk_demo/fetch_footage.py` then reload; pre-cut cross-camera clips play even without footage |
| Map has pins but no streets | `web/public/maplibre/` or `web/public/maps/basemap.pmtiles` missing — restore from git |
| Runs missing from القائمة | `python scripts/kaggle/shorouk_demo/fetch_results.py` (needs Kaggle token; do NOT do this on stage) |
| Case gone / demo DB broken | reseed per §3 (new case), or restore `data/app/app.db` from the `.pre-finals` backup |
| PDF button hangs | first PDF spawns headless Chromium (~3s warmup); wait, don't re-click |
| Everything on fire | play `mie-competition/assets/demo-happy-path-ar-dark.webm` full-screen |

## 6. What NOT to touch

- `data/runs/run-20260730-201635-81157c` + `run-20260731-005450-b6a70b` — the imported evidence
  (chain-of-custody hashes match the Kaggle-processed footage in
  `data/raw/shorouk_demo/`).
- Frozen parity baselines, calibrations, model registry — the demo needs
  none of them modified.
- No Kaggle/network access during the demo — everything is local.
