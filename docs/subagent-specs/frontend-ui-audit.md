# MTMC Frontend UI/UX Audit — Comprehensive Analysis
**Date:** May 27, 2026  
**Scope:** Next.js frontend (src/) — 7-stage pipeline, 28 major components, 6 Zustand stores, ~50 API functions

---

## Executive Summary — 10 Critical Issues

1. **No Visual Pipeline Narrative** — Users see "Upload, Detection, Selection, Inference, Timeline, Refinement, Output" with ZERO explanation of what each does or why it exists
2. **Stage Components Are Massive Monoliths** — Detection = 850 lines, Timeline = 2,100 lines, Inference = 1,250 lines; logic is tangled with UI
3. **No Information Architecture** — Sidebar nav shows abstract stage names; no "what does this stage need?" / "what outputs to use next?"
4. **Controls Are Scattered & Dense** — Inference stage alone has: 3 dataset selector cards + model picker + 5 dropdowns + 3 sliders + 2 tabs + location filters + date pickers + hyperparameter display
5. **Inconsistent Status System** — Stage 3 shows "Kaggle vs Local" toggle; Stages 1-2 hide it. Progress bars are sometimes muted, sometimes loud. Error states vary widely
6. **Cross-Stage State Chaos** — 6 Zustand stores (pipeline, video, detection, timeline, session, stageExecution); no clear ownership; downstream invalidation is manual
7. **No "Idle vs Running vs Done vs Failed" Visual Hierarchy** — Same badge size for all states; color coding is CSS-based magic; no glance-readability
8. **API Surface Is Messy** — 45+ functions; grouped by feature not by logical flow (ReID, pipeline, video, search, detections all separate); error handling is inconsistent
9. **Input/Output Contract Is Hidden** — Each stage page doesn't say "needs Stage X outputs, produces Stage Y inputs" anywhere
10. **Sidebar Badges Lack Semantic Value** — Kaggle creds icon exists but state is unclear; "active model" badge shows only on page entry, not in real-time

---

## 1. Site Map & Information Architecture

### 1.1 Route Structure

| Route | Component | Purpose | Status Badge |
|-------|-----------|---------|--------------|
| `/` | `page.tsx` | Splash screen → MainDashboard | —  |
| `/reid` | `reid/page.tsx` | ReID query tool (Phase 2, out-of-pipeline) | ⚠️ Separate from pipeline |
| `/fusion` | `fusion/page.tsx` | Fusion model evaluation | ⚠️ Separate from pipeline |
| `/eval` | `eval/page.tsx` | Evaluation runner | ⚠️ Separate from pipeline |

**Main Pipeline:** All 7 stages in `/page.tsx` via `MainDashboard` component + stage selector in sidebar

### 1.2 Sidebar Navigation

```
MTMC Pipeline
├── [Icon] Upload (Stage 0)       → UploadStage
├── [Icon] Detection (Stage 1)    → DetectionStage
├── [Icon] Selection (Stage 2)    → SelectionStage
├── [Icon] Inference (Stage 3)    → InferenceStage
├── [Icon] Timeline (Stage 4)     → TimelineStage
├── [Icon] Refinement (Stage 5)   → RefinementStage
└── [Icon] Output (Stage 6)       → OutputStage

Settings Panel:
├── [Gear icon] Kaggle Credentials
├── [Cloud icon] Kaggle Status Indicator
└── [?] Help / Docs (NOT PRESENT — missing!)
```

**Sidebar Issues:**
- No labels explaining what each stage does (only names)
- Icons are generic (Upload/Box/Database icons don't convey MEANING)
- Kaggle status indicator shows ✓ or ✗ only; no detail on rate limits, queue status
- No "last run time" or "% complete overall"
- Badge for "active model" exists in main dashboard header, not sidebar
- No indication of which stages are optional vs required

### 1.3 Main Dashboard Layout

```
┌─ Header (Pipeline Stage Selector Sidebar) ──────────────────┐
│ [Stage Icons] [Current Stage Label] [Model Badge] [Settings]│
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  [STAGE COMPONENT RENDERS HERE]                             │
│  (DetectionStage, InferenceStage, etc. — ~1000 lines each) │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**Issues:**
- Stage selection is tightly coupled to state updates; no "unsaved changes?" warning
- No visual indication of stage dependencies (Stage 3 requires Stage 1+2 complete)
- "Continue to Next" buttons exist on each stage but are hidden if preconditions aren't met
- No visual breadcrumb showing current stage in pipeline sequence

---

## 2. Per-Stage Component Inventory

### Stage 0: Upload (`upload-stage.tsx` — 428 lines)

**Purpose:** Import surveillance video OR pre-computed Kaggle artifacts

| Property | Value |
|----------|-------|
| **Inputs Needed** | None (initial stage) |
| **Outputs Produced** | runId, currentVideo (VideoFile) |
| **Backend Calls** | `uploadVideo()`, `importKaggleRunArtifacts()`, `getVideos()`, `runStage(1)` |
| **Controls/Knobs** | Drag-drop upload, file picker, Demo Mode button, Import Artifact Zip |

**Sections:**
1. **Header** — Title + Demo Mode quick-start
2. **Dataset Compatibility Alert** (blue info box) — Lists recommended/limited datasets
3. **Import Kaggle Artifacts Card** — Zip file picker + progress bar
4. **Grid (2 cols):** Left: Upload dropzone + progress bars | Right: Video gallery
5. **Quick Start Info Block** (3 columns) — Step 1/2/3 text

**Density:** Moderate. ~6 distinct sections, clean separation.

---

### Stage 1: Detection (`detection-stage.tsx` — 850 lines)

**Purpose:** Run YOLO + DeepOCSort on uploaded video; preview detections; select tracks to pass downstream

| Property | Value |
|----------|-------|
| **Inputs Needed** | currentVideo (from Stage 0) |
| **Outputs Produced** | selectedTrackIds (Set<number>), detections[] |
| **Backend Calls** | `runStage(1)`, `getAllDetections()`, `getDetections()`, `getPipelineStatus()` |
| **Execution Target** | Local only (GPU-required; typically runs on Kaggle) |

**Controls:** Play/Pause, Scrubber slider, Frame counter, Skip Back/Forward, Bounding box click-to-toggle, Multi-select checkbox toggle

**Knob Count:** ~7 interactive elements

**Density:** Moderate-to-high. Video canvas dominates; controls are compact. Sidebar adds 1 more dimension.

---

### Stage 2: Selection (`selection-stage.tsx` — 500 lines)

**Purpose:** Pick multi-camera tracklets to search for across dataset

| Property | Value |
|----------|-------|
| **Inputs Needed** | selectedTrackIds from Stage 1, currentVideo, stage1 outputs |
| **Outputs Produced** | Updated selectedTrackIds (persisted) |
| **Backend Calls** | `getTracklets()` |

**Controls:** Checkbox multi-select toggle, "Select All", "Deselect All", Tracklet card click, Remove (X) chips

**Knob Count:** ~4 major controls

**Density:** Moderate. Grid + sidebar are well-proportioned.

---

### Stage 3: Inference (`inference-stage.tsx` — 1,250 lines) **⚠️ PROBLEM AREA**

**Purpose:** Extract ReID embeddings (Stage 2) + build FAISS index (Stage 3); apply location/date filters; select model; run on local or Kaggle

| Property | Value |
|----------|-------|
| **Inputs Needed** | selectedTrackIds, currentVideo, runId |
| **Outputs Produced** | runId, stageProgress (2 & 3), modelMetadata |
| **Backend Calls** | `runStage(2)`, `runStage(3)`, `getPipelineStatus()`, `getDatasets()`, `fetchModel()` |

**Sections:**
1. Header + error/warning banners
2. Dataset Source Card
3. Execution Card (Local/Kaggle toggle)
4. Location Filter Card (3 dropdowns: Governorate → City → Zone)
5. Model Registry Card (Single vs Fusion tabs)
6. Effective Config Card (collapsed by default)
7. Active Pipeline Parameters Card (read-only)
8. Date & Time Range Card (2 date pickers)
9. Run Button + Processing Status Card

**Controls:** Dataset selector, Local/Kaggle toggle, 3 Location dropdowns, Model mode toggle, ModelPicker, Fusion weight sliders, 2 Date pickers, Run button

**Knob Count:** ~20+ interactive controls

**Density:** **VERY HIGH**
- 4+ cards in view simultaneously
- Nested dropdowns (location filtering)
- 2 date pickers with calendar popovers
- Fusion weight sliders (5 sliders if multiple models selected)
- Read-only config displays with nested dicts/arrays
- Multiple warning/error banners at top

---

### Stage 4: Timeline (`timeline-stage.tsx` — 2,100 lines) **⚠️ PROBLEM AREA**

**Purpose:** Run cross-camera association (Stage 4); visualize global trajectories; confirm/reject matches; apply alternatives

| Property | Value |
|----------|-------|
| **Inputs Needed** | selectedTrackIds, currentVideo, runId |
| **Outputs Produced** | tracks[], selectedTrackId, confirmedTracks |
| **Backend Calls** | `runStage(4)`, `queryTimeline()`, `getTrajectories()`, `getPipelineStatus()`, `getMatchedAlternatives()`, 6+ more |

**Sections:**
1. Header + badges + "Rerun Association" button
2. Left Sidebar: Progress card, Camera grid slider, Tracklet list, Top 5 Alternatives
3. Main Video Preview Grid (1–8 split-screen)
4. Timeline Controls (Play/pause, scrubber, zoom)
5. NLE-Style Timeline (ruler + camera lanes + tracklet blocks)

**Controls:** Play/pause, Skip back/forward, Scrubber, Zoom slider, Camera count slider, "Playing tracklets only" checkbox, Lane click, Tracklet row click, Confirm checkbox per row, Remove button per row, "Rerun Association" button, Alternative "Apply" button

**Knob Count:** ~12 major controls + per-row confirm/remove buttons

**Density:** **EXTREMELY DENSE**
- 6 distinct UI regions (header, left sidebar, video grid, controls, ruler, timeline)
- NLE-style timeline has horizontal + vertical scrolling
- Real-time playback with 4 FPS playhead updates
- Adaptive/filter mode changes layout dynamically
- Color coding (by camera) + confidence heatmap + state styling all at once

---

### Stage 5: Refinement (`refinement-stage.tsx` — 550 lines)

**Purpose:** Select reference frames for improved ReID search; run re-search

| Property | Value |
|----------|-------|
| **Inputs Needed** | confirmedTracks from Stage 4, runId |
| **Outputs Produced** | refinementFrames[], updated tracks |
| **Backend Calls** | `getTrackletSequence()`, `getRunFullFrameUrl()`, `getMatchedAlternatives()` |

**Controls:** Prev 10 / Play / Next 10 buttons, Scrubber, Speed selector, Frame thumbnail click, Remove (X) per selected frame, "Re-Search" button, "Clear Selection" button

**Knob Count:** ~7 major controls

**Density:** Low-to-moderate. Clean grid + sidebar layout.

---

### Stage 6: Output (`output-stage.tsx` — 900 lines)

**Purpose:** Render summary video; display trajectory map; export results (MP4/JSON/CSV)

| Property | Value |
|----------|-------|
| **Inputs Needed** | tracks, currentVideo, runId |
| **Outputs Produced** | Downloadable MP4, JSON, CSV; map visualization |
| **Backend Calls** | `generateSummaryVideo()`, `getTrajectories()`, `exportTrajectories()` |

**Controls:** Play/pause, Scrubber, Speed selector, Quality selector, Export format radio (MP4/JSON/CSV), Checkbox per trajectory, "Generate & Download" button

**Knob Count:** ~7 major controls

**Density:** Low. Clean separation: video on left, map + list on right.

---

## 3. Global State Audit

### 3.1 Zustand Stores

| Store Name | Purpose | Persisted? | Key Properties | Issues |
|------------|---------|-----------|-----------------|--------|
| `usePipelineStore` | Pipeline execution state | No | runId, stages[], currentStage, selectedModelId, fusion config, modelMode, error | Mixed responsibilities (run state + model selection + error) |
| `useStageExecutionStore` | Per-stage local/Kaggle target | Yes | stageExecutionTargets | Good separation |
| `useVideoStore` | Video selection & playback | No | videos[], currentVideo, currentFrame, isPlaying, playbackSpeed | Clear scope |
| `useDetectionStore` | Detection selection (Stage 1) | No | detections[], selectedTrackIds, multiSelectMode | Good; track-level selection persistent |
| `useTimelineStore` | Timeline state (Stage 4) | No | tracks[], selectedTrackId, zoom, confirmTrackId | Large; manages track metadata + visual state |
| `useSessionStore` | Session UI state | No | currentStage, locationFilter, dateTimeRange, refinementFrames | Scattered collection of unrelated state |

### 3.2 State Flow Issues

**Problem 1:** Scattered Responsibility
- `usePipelineStore` owns model selection; `useStageExecutionStore` owns execution target
- No clear ownership pattern

**Problem 2:** Cross-Stage Invalidation
- `downstreamInvalidateGeneration` counter manually incremented to force reload
- No clear pattern across all stages

**Problem 3:** Temporal State
- `currentStage` in `useSessionStore`; `runId`, `stages[]` in `usePipelineStore`
- No single source of truth

**Problem 4:** Timeline Store Complexity
- Tracks have multiple representations; alternative history by trackId in local state, not store

---

## 4. Visual & Status System Audit

### 4.1 Status Representation

| Status Type | Detection | Inference | Timeline | Inconsistency |
|-------------|-----------|-----------|----------|---|
| **Idle** | No indicator | Badge: "not started" | No indicator | Inconsistent |
| **Running** | Spinner + message | Progress bar + Kaggle panel | Progress card | Message placement differs |
| **Complete** | Badge + silent progression | Stage 2/3 badges | Badges + green checkmark | Badges used inconsistently |
| **Error** | Error banner (red, collapsed traceback) | Error banner (red, 1-2 lines) | Red box in progress card | Error styling differs |

### 4.2 Color Coding

**Current Palette:**
- Green: Success, selected (RGB: 34, 197, 79 — `#22c55e`)
- Red: Error (RGB: 239, 68, 68 — `#ef4444`)
- Yellow: Warning (RGB: 234, 179, 8 — `#eab308`)
- Blue: Info, active (RGB: 59, 130, 246 — `#3b82f6`)

**Issues:**
- No hierarchy: Alert box and badge use same colors
- Red/green only: Colorblind users cannot distinguish selected vs unselected
- Kaggle vs Local: No visual distinction

---

## 5. API Surface

### 5.1 Function Grouping

| Category | Functions | Issue |
|----------|-----------|-------|
| **Video Management** | uploadVideo, getVideos, getVideo, deleteVideo | Core |
| **Detection** | getDetections, getAllDetections | Stage 1 only |
| **Features & Indexing** | extractFeatures, buildIndex | Stage 2-3 only |
| **Tracklets & Search** | getTracklets, getMatchedSummary, queryTimeline, getTrajectories | Stage 4 only |
| **Pipeline Execution** | runStage, runFullPipeline, getPipelineStatus | Core |
| **Kaggle Integration** | getKaggleStatus, cancelKaggleKernel | Optional |
| **ReID** | singleCamReid, fusionReid | Phase 2 (out-of-pipeline) |
| **Evaluation** | submitEval, getEvalStatus | Phase 2 |

**Grouping Issue:** Functions are grouped by feature not by stage dependency. No way to ask "what do I call for Stage 2?"

### 5.2 Error Handling

**Pattern 1:** ApiError thrown on non-2xx response — caller must try/catch

**Pattern 2:** ApiResponse wrapper with `success`, `data`, `message`

**Inconsistency:** Mixed error patterns make error handling unpredictable

---

## 6. Pain Point Map — User Complaints → Code

### **Complaint #1: "I don't know what each stage does or how they connect"**

| Code Location | Problem | Fix |
|---------------|---------|-----|
| `main-dashboard.tsx` header | Stage labels are abstract with no tooltips | Add hover tooltips: "Stage 1: Run YOLO detector + DeepOCSort tracker" |
| Sidebar nav | No visual hierarchy showing dependencies | Add arrows/connectors; disable unmet prerequisites |
| Each stage page header | No "Inputs: X, Y" / "Outputs: Z" statement | Add 2-line banner at top |
| No route breadcrumbs | User doesn't see stage context | Add breadcrumb: "Home > Pipeline > Stage 3 of 7" |
| `/README.md` missing | No user-facing documentation | Create `docs/frontend-guide.md` with stage flowchart |

### **Complaint #2: "Too many controls/config knobs on each stage page"**

| Code Location | Problem | Count | Fix |
|---------------|---------|-------|-----|
| `inference-stage.tsx` | 20+ controls scattered across 7 cards | 20+ | Extract model picker to modal; fold dataset selector; hide advanced params |
| `timeline-stage.tsx` | Split-screen slider + zoom slider + play/pause + scrubber + lane selector + confirm checkboxes | 12+ | Move lane selector + confirm toggles to dedicated panel |
| All stages | No progressive disclosure | — | Use tabs/accordions to hide secondary controls |

### **Complaint #3: "Don't know what inputs/outputs each stage needs"**

| Code Location | Problem | Fix |
|---------------|---------|-----|
| Stage page headers | No statement of "requires:" or "produces:" | Add 2-line input/output contract banner |
| Stage components | No checking if inputs are available | Add precondition checker; show "Run Stage 1 first" if missing |
| API surface | No grouped functions by stage | Reorganize `api.ts` by stage (e.g., `stageN_*` prefix) |
| Sidebar nav | No indication of stage dependencies | Add dependency arrows or color coding |

### **Complaint #4: "Can't tell what's running vs done vs failed"**

| Code Location | Problem | Fix |
|---------------|---------|-----|
| Sidebar badges | Generic ✓ / ✗ for Kaggle; no color coding | Color-code each stage: Gray (idle), Blue (running), Green (done), Red (error) |
| `stageProgress[]` display | Status info in store but NOT shown in sidebar | Show progress dot + status badge next to each stage |
| `GlobalProcessingBanner` | Only shows when actively processing | Keep banner but change to "Last run: X min ago" |
| Error messages | Scattered across multiple banners | Consolidate: 1 persistent error zone at top |

### **Complaint #5: "Visual clutter — too dense, no hierarchy"**

| Code Location | Problem | Fix |
|---------------|---------|-----|
| `inference-stage.tsx` | 4+ cards all at full height; 7 cards vertically stacked | Use collapsible "Advanced" accordion; tabs for Location/Dates |
| `timeline-stage.tsx` | 6 UI regions all visible + lots of scrolling | Alternative selection in modal instead of sidebar |
| All stages | Cards use same border/shadow; no visual weight | Use CSS hierarchy: primary (filled) > secondary (outline) > tertiary (text) |

---

## 7. Component Reuse Opportunities

| Pattern | Locations | Lines | Opportunity |
|---------|-----------|-------|-------------|
| **Run Stage Button + Polling** | Detection, Inference, Timeline | ~100 each | Extract `<RunStageWidget stage={} />` |
| **Progress Card** | Inference, Timeline | ~50 each | Extract `<StageProgressCard />` |
| **Status Badge Set** | All stages | ~5 each | Extract `<StageBadges />` for consistency |
| **Dropdown Cascade** | Inference (location) | ~80 | Extract `<DependentSelectGroup />` |
| **Frame Thumbnail Grid** | Selection, Refinement | ~100 each | Extract `<FrameGrid />` |
| **Tracklet Preview** | Detection, Selection, Timeline | ~30 each | Extract `<TrackletPreview />` |
| **Export Panel** | Output | ~150 | Extract `<ExportDialog />` |
| **Error Banner** | All stages | ~20 each | Extract `<ErrorBanner />` |

### Missing Primitives

- `<LoadingState />` — Spinner + message + retry button (used ~6 times)
- `<ProgressIndicator />` — Bullet point status (idle → running → done/error)
- `<ControlGroup />` — Label + input + helper text (used ~15 times)

---

## 8. Key Files Reference Table

| File Path | Type | Lines | Purpose | Status |
|-----------|------|-------|---------|--------|
| `layout/main-dashboard.tsx` | Component | 270 | Stage selector, sidebar nav | Clean |
| `stages/upload-stage.tsx` | Component | 428 | Video upload / artifact import | Moderate |
| `stages/detection-stage.tsx` | Component | 850 | YOLO + tracking preview | Dense |
| `stages/selection-stage.tsx` | Component | 500 | Tracklet multi-select | Moderate |
| `stages/inference-stage.tsx` | Component | 1,250 | Model picker + feature extraction | **VERY DENSE** |
| `stages/timeline-stage.tsx` | Component | 2,100 | Association + trajectory viz | **EXTREMELY DENSE** |
| `stages/refinement-stage.tsx` | Component | 550 | Refinement frame selection | Moderate |
| `stages/output-stage.tsx` | Component | 900 | Summary video + export | Moderate |
| `store/index.ts` | State | 500+ | 6 Zustand stores | Complex |
| `lib/api.ts` | API | 700+ | 45+ backend functions | Scattered |

**Total Stage Components:** ~7,600 lines  
**Total API Layer:** ~700 lines  
**Total State:** ~500 lines

---

## 9. Recommendations for Redesign

### Priority 1 (Critical)

1. **Add Stage Info Banner** — Each page shows "Stage N of 7: [Name]" + "Inputs: X, Y" / "Outputs: Z"
2. **Sidebar Status Dots** — Color-code + icon each stage (idle/running/done/error)
3. **Simplify Inference Stage** — Move model picker to modal; hide advanced params
4. **Extract Shared Components** — `<RunStageWidget>`, `<StageProgressCard>`, `<ErrorBanner>`

### Priority 2 (High)

5. **Consolidate Status Display** — One error zone at top; badges only in headers
6. **API Reorganization** — Group functions by stage
7. **Improve Timeline UX** — Alternative selection in modal
8. **Color Accessibility** — Add icons/shapes to status indicators

### Priority 3 (Medium)

9. **Dependency Visualization** — Show stage dependencies in sidebar
10. **Progressive Disclosure** — Use accordions/tabs to hide 50% of secondary controls