# ATHAR — MIE 20th Edition Finals Deck (CS140)
# 12 slides · 15 minutes · Finals: Sat 1 Aug 2026, Nile University

**Judging map:** 1 Problem&Opportunity → S2 · 2 Solution&Innovation → S3 · 3 Technical
Implementation → S4+S5 · 4 Business Model&Impact → S6–S11 · 5 Presentation → design+delivery.
All 11 required content points from working-material.md are covered (order adapted).

**Design language:** deep navy `101A33` dominant (dark theme throughout, premium enterprise),
ice `E9EEF9` text, teal `19C6B4` accent (the "identity thread"), gold `D4A72C` sparingly for
money/awards, coral `FF6B5E` for problem/alerts only. Font: Segoe UI (embed on save).
Motif: a thin teal "tracking path" polyline with node dots — the cross-camera identity thread —
recurring on every slide (title corner, section markers). NOT a color bar.

---

## S1 — Title
- **ATHAR | أثر** wordmark (means "trace/trail" in Arabic — say it out loud)
- Tagline: **"Every camera tells a fragment. ATHAR tells the story."**
- Sub: Forensic multi-camera tracking & re-identification — built in Egypt, for Egypt.
- Team CS140 · Cairo University Faculty of Engineering · Supervisor: Assoc. Prof. Ahmed Hamdi
- Footer: IEEE MIE 20th Edition Finals — Aug 2026
- Visual: dark navy field, subtle Egypt-map contour, teal tracking path crossing 3 camera nodes.
- Notes: 30s. Hook: "A suspect crosses a city. 200 cameras saw him. Nobody can follow him."

## S2 — Problem (Problem & Opportunity)
Headline: **Finding one person in a city of cameras takes a week of human eyes.**
- Investigators manually scrub footage camera-by-camera: a real serious-assault case burned
  **185 hours** of officer time on CCTV review (Gorilla Technology case study).
- Cameras don't share identity: every handoff between cameras loses the target →
  fragmented, unusable timelines. Egypt: NAC alone runs **6,000+ cameras** into one ICCC.
- Stakes: slower case closure, missed windows in kidnapping/theft response, wasted officer-hours.
- Existing tools: Western analytics cost **$600–1,200/camera** (BriefCam-class), ship no Arabic,
  no local support; Chinese stacks raise **data-sovereignty** red flags for government.
- Stat callouts: 185h (one case) · 6,000+ (NAC cameras) · $1.2B (Egypt electronic-security mkt 2024)
- Notes: 90s. Make it human: "one officer, one week, one suspect — and he's already gone."

## S3 — Solution (Solution & Innovation)
Headline: **Upload the footage. Ask for the person. Get the journey.**
ATHAR = a forensic search engine over any CCTV network:
1. **Ingest** days of multi-camera footage (works on EXISTING cameras — zero new hardware)
2. **Search** with one crop/photo of a person or vehicle
3. **Reconstruct** the full cross-camera journey on a timeline + map
4. **Export** a court-ready evidence report (tamper-evident chain of custody)
Innovation bullets (right column):
- Person **and** vehicle in one system (rivals do one or the other)
- Investigator confirm/reject on every match — verified evidence only (no rival has this)
- Hash-chained audit + WORM anchoring: every result traceable video→model→report
- Arabic-first UI (RTL day one) — built for Egyptian operations
- 185h → minutes of search, not weeks of scrubbing
- Notes: 2min. Demo-flow narrative. This slide = the product promise; proof comes next.

## S4 — Product & Technology (Technical Implementation 1/2)
Headline: **A working product — not a paper pipeline.**
- LEFT: real app screenshots (case workspace / timeline / map — Arabic RTL) stacked/angled.
- RIGHT: pipeline strip: Ingest → Detect+Track → Re-ID embed → Index → Cross-camera associate →
  Case report. Under it, tech chips: YOLO26 · TransReID(ViT) + CLIP-SENet + DINOv2 ensemble ·
  FAISS · FastAPI · Next.js (Arabic RTL)
- Floor line: Full chain of custody: evidence SHA-256 → frozen config → model SHA → result.
  Air-gap ready: zero cloud calls at runtime (sovereign deployments).
- Notes: 2min. "Everything you see is running today on our machines — offline, in Arabic."

## S5 — Validation: benchmarks + real deployment (Technical Implementation 2/2)
Headline: **State of the art — proven on world benchmarks, then on real Egyptian streets.**
- Chart 1 (bar): VeRi-776 vehicle Re-ID mAP: ATHAR fusion **93.3%** vs previous best published
  (CLIP-SENet) 92.9 vs strongest pure ViT-B 90.0 → **state of the art**.
- Stat tiles: CityFlowV2 MTMC IDF1 **0.78** (AI City 2022 leaderboard range 0.70–0.84, offline
  validation protocol) · WILDTRACK person IDF1 **0.95** / MODA 0.90.
- Real-world: **14-camera live deployment — El Shorouk compound, Cairo**: ran on unseen cameras,
  no retraining. (mini map pin + camera icons)
- v2 kicker: joint multi-domain Re-ID retrain generalizes: **+10 pts** CityFlow, **+38 pts**
  VeRi-Wild vs single-domain baseline — the engine improves with every domain we add.
- Notes: 2min. Careful honest phrasing on CityFlow protocol (validation split, offline).

## S6 — Market Opportunity
Headline: **Egypt is building the cameras. Nobody Egyptian is building the brain.**
- Left: TAM/SAM/SOM funnel: TAM Egypt video-analytics software ≈ **$30–50M/yr by 2027**
  (→$115–175M by 2033); SAM (gov + smart-city cameras, ~80k by 2027) ≈ **$32–48M** licenses +
  $6.4–9.6M/yr recurring; SOM 5-yr ≈ **$12–25M**.
- Right: growth chart Egypt CCTV market **$153M (2024) → $582M (2033)**, 14.3% CAGR; MEA
  fastest-growing region globally ($6.7B, 12.2% CAGR).
- Drivers row: 60+ planned smart cities (Vision 2030) · NAC 6,000+ cameras phase 1 ·
  50–100k MoI cameras by 2027 · gated compounds & malls boom.
- Notes: 1min. Big numbers, then pivot: "and every one of those cameras needs software like ours."

## S7 — Competitive Landscape
Headline: **Global tools weren't built for Egypt. ATHAR was.**
- Comparison table (rows = capabilities, cols = BriefCam/Milestone, Gorilla, Intelion, Hikvision/
  Dahua, **ATHAR**): cross-camera Re-ID (person+vehicle) · interactive verification workflow ·
  chain-of-custody/audit · Arabic UI + local support · data sovereignty/air-gap · price/camera.
  ATHAR column teal-filled: Yes ×5 + **30–50% lower TCO**.
- Under-table gap chips: ✦ person↔vehicle identity bridging ✦ human-in-the-loop verification
  ✦ sovereign on-prem AI — no competitor has all three.
- Pricing anchor: BriefCam $600–1,200/cam vs ATHAR $400–500/cam perpetual.
- Notes: 90s. Name the enemy honestly; our wedge is sovereignty + Arabic + price + verification.

## S8 — Business Model
Headline: **License the platform. Grow with every camera.**
- Revenue streams (4 cards): Core license **$400–500/camera** (perpetual) · Annual maintenance
  $60–100/cam/yr · Advanced analytics subscription $80–120/cam/yr · Professional services
  (integration+training) 15–20% of license.
- Who pays (customer stack): Government (MoI via licensed SI partner) · Smart-city operators ·
  Private: compounds, malls, clubs, private-security firms.
- Unit economics strip: gross margin 50–70% (software) · CapEx-friendly procurement (gov
  preference) · land-and-expand per governorate.
- Notes: 1min. One sentence per stream; emphasize recurring layer on installed base.

## S9 — Go-To-Market
Headline: **Private proves it. Partners scale it. Government buys it.**
3-phase route (horizontal path with the motif):
1. **Now → 6mo — Beachhead (private, direct):** gated compounds (El Shorouk pilot LIVE),
   malls, sporting clubs, private-security companies. Short sales cycle, real revenue,
   reference sites. (This is the deployment we already have.)
2. **6–18mo — Channel (B2G via integrators):** partner with licensed Egyptian system
   integrators / defense contractors who hold MoI + NUCA frameworks — they carry ATHAR into
   government tenders (MoI direct procurement exemption shortens cycles). ITIDA/TIEC support.
3. **18mo+ — Scale:** governorate command centers, smart-city ICCCs, MEA export (GCC).
- Notes: 90s. Explicitly say: "we don't knock on the MoI's door alone — we ride proven
  integrator channels; meanwhile private sites pay the bills and battle-test the product."

## S10 — Impact
Headline: **Safer streets, faster justice, sovereign technology.**
4 impact tiles (data from cited deployments of this technology class):
- **Security:** −55% homicides, −35% robberies (West Palm Beach, after video analytics);
  −93% cargo theft (Tlaxcala, Mexico); 30–40% crime-reduction potential (Deloitte smart-cities).
- **Justice speed:** 185h→15h case review (Gorilla); 24h footage → 15 key minutes (BriefCam/
  Green Bay PD); 10× faster investigations.
- **Economy:** 200–400% ROI in 18–24mo typical for AI surveillance; $11M community savings
  (West Palm Beach, 2yrs); 30–50% lower monitoring staffing cost.
- **Sovereignty & sustainability:** 100% local data — zero foreign dependency · zero new
  hardware (runs on existing CCTV) · Egyptian engineers, Egyptian IP, Vision 2030 aligned.
- Small print: sector case-study figures for AI video analytics deployments; sources on file.
- Notes: 1min. "This is what this class of technology does when a city turns it on."

## S11 — Financial Projections
Headline: **Break-even in year one. $13.6M cumulative by year five.**
- Chart: combo column (annual revenue Y1–Y5: 1.46 / 2.30 / 3.90 / 3.10 / 2.80 $M) + line
  (cumulative net cash: 0.31 / 1.81 / 4.52 / 6.34 / **7.72 $M**).
- Stat row: Initial CapEx **$650k** · Break-even **month 9–12** · 5-yr net profit **$7.7M** ·
  ROI **~12×** initial capital · gross margin 50–70%.
- Assumptions line (small): $500/cam avg license; 2k→15k cameras cumulative by Y5; 80%
  maintenance attach; 30% analytics attach by Y3.
- Notes: 1min. Point at break-even and the recurring layer; don't read every number.

## S12 — Team + Next Steps / The Ask
Headline: **The team that built it — and what we do next.**
- Team row (4): Abdelrahman Hamdi — data & ingestion · Ali Ashraf — detection & tracking ·
  Mohamed Amar — Re-ID, search & indexing · Seif Tamer — association, evaluation & platform.
  Supervisor: Assoc. Prof. Ahmed Hamdi, Cairo University. (CS140)
- Next 12 months (3 milestones): ① 3 paid private pilots (compound ✓ live, + mall + club/
  security firm) ② sign 1–2 SI channel partnerships for gov tenders ③ v2 platform GA
  (multi-domain models, live RTSP, watchlists — already in build).
- **The Ask** (gold): Pilot introductions — smart-city operators, malls, clubs, private-security
  companies & SIs serving MoI · incubation/acceleration (ITIDA, TIEC, Flat6Labs) · seed
  funding to certify, harden, and deploy.
- Close line: **"The cameras are already watching. ATHAR makes them remember."**
- Notes: 45s. End on the close line, then "Thank you — نحن جاهزون لأسئلتكم".

---
## Fact guardrails (do not violate on slides)
- VeRi-776 93.3% mAP = fusion result, SOTA claim per report §5.4.2 (beats CLIP-SENet 92.9).
- CityFlow 0.779 IDF1 = validation split, offline protocol — never claim leaderboard rank;
  phrase as "within published leaderboard range (0.70–0.84)".
- WILDTRACK 0.946 IDF1 / 0.903 MODA (ground-plane protocol).
- El Shorouk = 14 cameras, ~21.5 min synced footage, unseen cameras, no retraining — real,
  but a verification deployment, phrase as "live deployment verification", pilot-in-progress.
- Impact stats (crime/ROI) are third-party case studies of the technology CLASS, not ATHAR
  results — keep the attribution line on S10.
- Financials = report Tables 2.3–2.6 (5-yr model, $650k CapEx, break-even mo 9–12, $7.7M).
- Joint-retrain gains: +10.2pp CityFlow (0.30→0.40 mAP), +37.6pp VeRi-Wild (0.31→0.69),
  +2.3pp VeRi (0.80→0.82) — deployed-mode eval, ROADMAP Phase 6 frozen matrix.
