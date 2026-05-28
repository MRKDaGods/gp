# MTMC Frontend UI/UX Redesign Spec
**Date:** May 27, 2026
**Source audit:** `docs/subagent-specs/frontend-ui-audit.md`
**Scope:** Next.js 14 App Router + Tailwind + shadcn/ui + Zustand. UI only — no backend, no new routes, no API shape changes.
**Goal:** Turn the 7-stage pipeline UI into a single guided+expert experience where every stage clearly states what it needs, what it produces, what it's doing, and which knobs matter.

---

## 1. Design Principles (the North Star)

1. **One canonical pipeline grammar.** Every stage uses the same shell: Contract Banner → Status Strip → Primary Workspace → Actions Footer. No bespoke layouts.
2. **State is glanceable, not readable.** Status is communicated by one badge component, one dot component, one color+icon pair. Five states, no exceptions.
3. **Progressive disclosure in three tiers.** Essential is always visible. Advanced is one click away (`<DisclosurePanel>`). Debug is two clicks away (collapsed by default, dev-tinted).
4. **Inputs and outputs are first-class.** Every stage page declares its upstream dependencies and downstream artifacts at the top. If upstream is missing, the run button is blocked with a fix-it link.
5. **Run location is a stage-level decision, not a global mode.** Local/Kaggle lives in the stage Actions Footer and is universal where it makes sense.
6. **The pipeline run is the protagonist.** A persistent `<PipelineRunHeader>` keeps runId, current stage, overall progress, and last-error visible everywhere.
7. **Colorblind-safe by construction.** State is always color + icon + label. Never color alone.
8. **No silent invalidation.** When upstream edits invalidate downstream artifacts, show a yellow “Stale — needs re-run” chip on every affected stage in the sidebar.

---

## 2. New Information Architecture

### 2.1 Three-zone layout

```
┌──────────────────────────────────────────────────────────────────┐
│ PipelineRunHeader   runId · stage · overall · last-run · error  │  ← 40px persistent
├────────┬─────────────────────────────────────────────────────────┤
│        │ ContractBanner (Stage N: Name · Needs · Produces)      │  ← 56px sticky
│        ├─────────────────────────────────────────────────────────┤
│Sidebar │                                                          │
│ 14/56  │                  Primary Workspace                       │
│ /208   │                  (the stage's main UI)                   │
│        │                                                          │
│        ├─────────────────────────────────────────────────────────┤
│        │ Actions Footer (Target · Run · Cancel · Continue →)    │  ← 64px sticky
└────────┴─────────────────────────────────────────────────────────┘
```

### 2.2 Sidebar redesign

The existing collapsible sidebar (`main-dashboard.tsx`) stays, but each stage row becomes a richer cell:

```
[●] 1  Detection                ✓ done · local
[◐] 2  Selection                ⟳ running · 47% · local
[○] 3  Inference                ⊘ blocked — needs Stage 2
[○] 4  Timeline                 ◇ stale · last run 12m ago
[!] 5  Refinement               ⚠ error · click to view
[○] 6  Output                   — idle
```

Per row:
- Left status dot (`<StageStatusDot />`) — colorblind-safe (filled, half, ring, dashed, exclamation)
- Stage number + label
- Right meta strip (collapsed-sidebar mode: just dot + cloud overlay; expanded: dot + label + secondary meta)
- Tiny cloud/server glyph remains (already implemented — keep)
- Hover tooltip shows full status sentence: e.g. “Running on Kaggle · started 4m ago · 47% complete”

Dependency arrows: keep implicit (the vertical list already implies order). Do NOT draw SVG arrows — that's noise. Instead, when a stage is `blocked`, its row shows a one-line reason (“needs Stage 2”) inline.

Bottom of sidebar keeps the existing Settings (Kaggle creds), Kaggle status panel, and Active Model chip. Add a new “?” help button that opens a side sheet with the same content as `docs/frontend-guide.md` (we'll create that doc in Phase 9 — for now the button opens a placeholder sheet).

### 2.3 PipelineRunHeader (new persistent strip — 40px)

A new component `<PipelineRunHeader />` mounted in `main-dashboard.tsx` above the stage workspace:

```
[runId: 2026-05-27_run_18] · Stage 3/7 · Inference  [████████░░ 47%]  last run: 12m ago    [⚠ 1 error]
```

- runId chip (click to copy)
- Current stage breadcrumb (Stage N/7 · Name)
- Overall progress bar (mean of all stages’ progress)
- “last run: Xm ago” relative timestamp (per-stage; updates every 30s)
- Error pill that opens a side panel listing the last 5 errors across all stages (sourced from `usePipelineStore.error` + per-stage error state — Phase 2 just wires the read; persistence is out of scope)

### 2.4 ContractBanner (per stage — 56px sticky under header)

Every stage page renders a `<ContractBanner />` as its first child. Example:

```
Stage 3 · Inference
Needs: track selections from Stage 2 · video frames from Stage 0
Produces: ReID embeddings · FAISS index (used by Stage 4)
                                                       [? help] [docs ↗]
```

- Stage number + name (large)
- “Needs:” line listing upstream artifacts as chips. Each chip links to the source stage. Missing chips render red with a “Fix in Stage 2” link.
- “Produces:” line listing downstream artifacts as chips. Each chip indicates freshness (✓ fresh / ◇ stale / — none yet).
- “? help” opens a popover with a 3-sentence plain-English explanation (content defined per-stage in §6).
- “docs ↗” opens a side sheet (Phase 9 — placeholder for now).

### 2.5 Help/tooltip strategy

- **Universal “?” icon** next to every control with a non-obvious effect. Tooltip text is one sentence ≤ 90 chars.
- **No info-icon spam on Essential controls.** Essential controls have inline helper text under the label.
- **Advanced and Debug panels** open with a 1–2 line header explaining what the section is for.
- **No tutorials/coach marks.** Solo-developer persona — they don't want hand-holding past first read.

---

## 3. Canonical Status System

### 3.1 The five canonical states

| State     | When                                                              | Color (text)        | Color (bg)            | Icon (lucide)      | Dot glyph         |
| --------- | ----------------------------------------------------------------- | ------------------- | --------------------- | ------------------ | ----------------- |
| `idle`    | Stage has never run and upstream is satisfied                     | `text-muted-fg`     | `bg-muted/30`         | `Circle`           | ○ (empty ring)    |
| `blocked` | Upstream artifacts missing or upstream stage failed               | `text-zinc-500`     | `bg-zinc-500/10`      | `Ban`              | ⊘ (dashed ring)   |
| `running` | Stage is executing (local or Kaggle)                              | `text-sky-600`      | `bg-sky-500/10`       | `Loader2 spin`     | ◐ (half-filled)   |
| `done`    | Stage completed successfully and outputs are fresh                | `text-emerald-600`  | `bg-emerald-500/10`   | `CheckCircle2`     | ● (filled)        |
| `stale`   | Stage completed previously but upstream changed since             | `text-amber-600`    | `bg-amber-500/10`     | `AlertTriangle`    | ◇ (filled diamond)|
| `error`   | Stage's last execution failed                                     | `text-rose-600`     | `bg-rose-500/10`      | `XCircle`          | ! (exclamation)   |

Total: six states. (Audit asked for five; `stale` is genuinely distinct from `idle` because upstream invalidation is already tracked via `downstreamInvalidateGeneration` and we want users to see it.) The label list is **idle, blocked, running, done, stale, error**.

### 3.2 Canonical components

Location: `frontend/src/components/pipeline/status/`

- **`<StageStatusBadge state={...} label?={...} size="sm"|"md" />`** — pill with icon + color + label. Used in contract banner, run buttons, error banners.
- **`<StageStatusDot state={...} size="sm"|"md" withCloudOverlay?={boolean} />`** — circle/ring/diamond glyph. Used in sidebar and run header.
- Both consume a single `StageState` enum exported from `frontend/src/components/pipeline/status/types.ts`. Existing `pipelineStages[].status` (idle/running/completed/error) is mapped via a `toStageState(stageProgress, isUpstreamStale)` helper.

### 3.3 Where the canonical components appear

| Surface                          | Component used                                              |
| -------------------------------- | ----------------------------------------------------------- |
| Sidebar row                      | `<StageStatusDot withCloudOverlay />`                       |
| `<PipelineRunHeader />` error    | `<StageStatusBadge state="error" size="sm" />` (only if any)|
| `<ContractBanner />` right side  | `<StageStatusBadge state={current} size="md" />`            |
| Actions Footer Run button        | Run button color is keyed off state; inline `<StageStatusBadge size="sm">` when running |
| Error banner header              | `<StageStatusBadge state="error" />`                        |

All other ad-hoc badge/spinner usages in `inference-stage.tsx`, `timeline-stage.tsx`, etc. are removed.

---

## 4. Progressive Disclosure Tier System

Three tiers, one component (`<DisclosurePanel tier="advanced" | "debug" defaultOpen={false} />`).

- **Essential** — Always visible. Inputs a user *must* set to run the stage. Target: ≤ 5 controls per stage.
- **Advanced** — Collapsed by default, plain expander. Tuning knobs that change results but have sensible defaults.
- **Debug** — Collapsed by default, dev-tinted (dashed border, “DEBUG” label). Read-only telemetry, raw config dumps, internal counters.

### 4.1 Per-stage knob mapping

#### Stage 0 — Upload
- **Essential**: drag-drop video, Demo Mode button.
- **Advanced**: Import Kaggle Artifacts zip picker.
- **Debug**: Dataset Compatibility Alert (collapsed; was a big blue info box) and the “Quick Start” 3-column info block (move into Debug — or remove entirely; recommendation: remove, replaced by ContractBanner help popover).

#### Stage 1 — Detection
- **Essential**: video preview canvas + scrubber, Run button.
- **Advanced**: multi-select toggle, “select all/deselect all”.
- **Debug**: frame counter, raw bbox-click hit log (currently inline — move to debug accordion).

#### Stage 2 — Selection
- **Essential**: tracklet grid with checkboxes, Run button.
- **Advanced**: Select-all / Deselect-all bulk controls.
- **Debug**: tracklet-count summary, per-camera distribution chart (if any).

#### Stage 3 — Inference (⚠ biggest cleanup)
- **Essential**:
  - Model picker (single OR fusion, controlled by a small inline single/fusion toggle, NOT a tabbed mega-card)
  - Run button + target toggle in Actions Footer
- **Advanced** (single `<DisclosurePanel tier="advanced">` with sub-tabs Location / Time / Fusion):
  - Location filter cascade (Governorate → City → Zone)
  - Date & Time Range (2 date pickers)
  - Fusion weight sliders (only shown when model mode = fusion)
- **Debug** (collapsed `<DisclosurePanel tier="debug">`):
  - Effective Config Card (read-only OmegaConf dump)
  - Active Pipeline Parameters Card (read-only)
  - Dataset source card details (raw paths, counts)

Dataset source selection is moved to a small inline `<DatasetSwitcher />` chip in the ContractBanner's “Needs” line (since dataset is an upstream concept, not a Stage-3 knob).

#### Stage 4 — Timeline (⚠ second biggest cleanup)
- **Essential**:
  - Multi-cam video preview grid (top)
  - NLE-style timeline (bottom)
  - Tracklet list (left rail)
- **Advanced** (`<DisclosurePanel tier="advanced">`):
  - Camera-count slider (split-screen 1–8)
  - Zoom slider
  - “Playing tracklets only” checkbox
  - Top-5 Alternatives panel (currently in left sidebar — move to a side sheet opened by clicking a tracklet's “alternatives” pill)
- **Debug**:
  - Association rerun controls (existing “Rerun Association” button stays as Essential — it's a primary action; the debug panel only holds raw match/conflict counts)
  - Confidence heatmap toggle
  - Lane coloring scheme selector

#### Stage 5 — Refinement
- **Essential**: frame thumbnail grid, Prev/Play/Next, Re-Search button.
- **Advanced**: scrubber, speed selector, Clear-Selection bulk button.
- **Debug**: per-frame embedding distance readout.

#### Stage 6 — Output
- **Essential**: video preview, Generate & Download button, export format radio (MP4/JSON/CSV).
- **Advanced**: quality selector, speed selector, per-trajectory checkbox grid.
- **Debug**: raw trajectory JSON preview.

---

## 5. Shared Component Plan

All new primitives live under `frontend/src/components/pipeline/`:

```
frontend/src/components/pipeline/
├── shell/
│   ├── StageShell.tsx              ← Contract banner + workspace slot + actions footer
│   ├── ContractBanner.tsx
│   └── ActionsFooter.tsx
├── status/
│   ├── StageStatusBadge.tsx
│   ├── StageStatusDot.tsx
│   ├── types.ts                    ← StageState enum + toStageState()
│   └── status-meta.ts              ← color/icon/label table
├── run/
│   ├── RunStageWidget.tsx          ← run button + progress + branches to KaggleStatusPanel
│   ├── StageProgressCard.tsx
│   └── ExecutionTargetToggle.tsx   ← thin wrapper around KaggleExecutionToggle, universalized
├── disclosure/
│   └── DisclosurePanel.tsx         ← collapsible w/ tier="advanced"|"debug" styling
├── feedback/
│   ├── ErrorBanner.tsx             ← unified red banner; replaces all per-stage error boxes
│   └── StalenessChip.tsx           ← amber chip "Stale — needs re-run"
├── media/
│   ├── PlaybackControls.tsx        ← prev/play/next/scrubber/speed (used in 1, 4, 5, 6)
│   ├── TrackletPreview.tsx         ← thumbnail + id + cam label (used in 1, 2, 4)
│   └── FrameGrid.tsx               ← multi-frame thumbnail grid (used in 2, 5)
└── header/
    └── PipelineRunHeader.tsx       ← persistent top strip
```

Each component replaces the listed audit Section-7 patterns. No new heavy deps — `<DisclosurePanel>` wraps shadcn's `Accordion`; `<PlaybackControls>` wraps `Slider` + `Button`; status components use `lucide-react` icons we already ship.

**Removed/absorbed:**
- `inference-stage.tsx` Effective Config Card → merged into Debug `<DisclosurePanel>`
- Per-stage ad-hoc error banners → `<ErrorBanner>`
- Per-stage progress cards → `<StageProgressCard>`
- All inline “⏳ Loading…” renders → consume `<StageStatusBadge state="running">`

---

## 6. Per-Stage Redesign

For each stage: new layout, components consumed, knob disposition, target line count, and (for 3 & 4) the specific layout fix.

### Stage 0 — Upload (target: 428 → ~220 lines)

```
[ContractBanner]   Stage 0 · Upload
                   Needs: —
                   Produces: video file · runId
─────────────────────────────────────────────
[                    Drag-drop dropzone                    ]
  or  [Demo Mode] [Browse files]

────── existing videos ──────
[VideoCard] [VideoCard] [VideoCard] …

[DisclosurePanel "Advanced"]
  └── Import Kaggle Artifacts (zip picker + progress)

[ActionsFooter]   target: n/a (upload is local)        [Continue → Stage 1]
```

- Consumes: `<StageShell>`, `<ContractBanner>`, `<ErrorBanner>`, `<DisclosurePanel tier="advanced">`, `<ActionsFooter>` (no run button — “Continue” only).
- Removed: Quick-Start 3-column info block; dataset-compatibility info card (content moved to ContractBanner help popover).

### Stage 1 — Detection (target: 850 → ~380 lines)

```
[ContractBanner]   Stage 1 · Detection
                   Needs: video [from Stage 0 ✓]
                   Produces: detections · track ids
─────────────────────────────────────────────
┌──────────────── Video canvas + bboxes ────────────────┐
│                                                       │
└───────────────────────────────────────────────────────┘
[PlaybackControls]   prev · play · scrubber · next

[DisclosurePanel "Advanced"]
  └── multi-select mode toggle · Select all · Deselect all

[DisclosurePanel "Debug"]
  └── frame counter · selected count · raw click log

[ActionsFooter]  [ExecutionTargetToggle local|kaggle]  [Run Stage 1]  [Continue → Stage 2]
```

- Consumes: `<StageShell>`, `<PlaybackControls>`, `<RunStageWidget>`, `<DisclosurePanel>` ×2, `<ErrorBanner>`.
- All ad-hoc badges/spinners replaced by canonical status surfaces.

### Stage 2 — Selection (target: 500 → ~260 lines)

```
[ContractBanner]   Stage 2 · Selection
                   Needs: detections [Stage 1 ✓]
                   Produces: selected tracklet ids
─────────────────────────────────────────────
[FrameGrid of tracklets — checkbox overlay]

[DisclosurePanel "Advanced"]
  └── Select all · Deselect all · multi-select toggle

[ActionsFooter]  [ExecutionTargetToggle]  [Continue → Stage 3]
```

- Consumes: `<StageShell>`, `<FrameGrid>`, `<TrackletPreview>`, `<DisclosurePanel>`.
- Stage 2 is purely a selection UI; no “run” button — “Continue” gates on `selectedTrackIds.size > 0`.

### Stage 3 — Inference (target: 1,250 → ~480 lines) ⚠ biggest win

The current 7-card vertical stack becomes one above-the-fold model decision + one collapsed advanced tab group + a debug panel.

```
[ContractBanner]   Stage 3 · Inference
                   Needs: tracklet ids [Stage 2 ✓] · dataset [chip: cityflowv2 ▾]
                   Produces: ReID embeddings · FAISS index
─────────────────────────────────────────────
┌─ Model ─────────────────────────────────────────────┐
│  [Single ▾]  [Fusion ▾]   ← inline toggle           │
│  ┌──── ModelPicker (selected: TransReID-ViT-B16) ──┐│
│  │ name · headline metric (IDF1 0.779) · checkpt   ││
│  └──────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────┘

[DisclosurePanel "Advanced"]   ← sub-tabs inside
  ┌─[Location][Time][Fusion]─────────────────────────┐
  │ Location: Governorate → City → Zone cascade     │
  │ Time:     [start date]   [end date]              │
  │ Fusion:   (visible only when mode=fusion)        │
  │           weight sliders · AQE-k · rerank        │
  └──────────────────────────────────────────────────┘

[DisclosurePanel "Debug"]
  ├── Effective Config (OmegaConf dump, copyable)
  ├── Active Pipeline Parameters (read-only)
  └── Dataset source detail (paths, counts)

[ActionsFooter]  [ExecutionTargetToggle local|kaggle]  [Run Stage 3]  [Continue → Stage 4]
                  └── KaggleStatusPanel slides in here when target=kaggle and running
```

- The 3-card pile (Dataset / Execution / Location) collapses: dataset becomes a ContractBanner chip, execution moves to ActionsFooter, location moves into Advanced.
- Single/Fusion stays as a tab BUT inside the Model card only — not as a top-level page tab.
- Date pickers move to Advanced.
- Consumes: `<StageShell>`, `<RunStageWidget>` (handles local-vs-Kaggle branching internally; embeds existing `<KaggleStatusPanel>`), `<DisclosurePanel>` ×2, `<ExecutionTargetToggle>`, existing `<ModelPicker>` and `<FusionModelPanel>` (unchanged).

### Stage 4 — Timeline (target: 2,100 → ~900 lines) ⚠ most complex

Current 6 regions (header / left sidebar / video grid / controls / ruler / timeline) → 3 visible regions + 2 side sheets.

```
[ContractBanner]   Stage 4 · Timeline
                   Needs: embeddings + index [Stage 3 ✓]
                   Produces: global trajectories
─────────────────────────────────────────────
┌─ Left rail (tracklet list) ─┐┌─ Video grid (split-screen) ─┐
│ TrackletPreview rows         ││                              │
│ click → opens Alternatives   ││                              │
│       side sheet             ││                              │
└──────────────────────────────┘└──────────────────────────────┘

[PlaybackControls]
─────────────────────────────────────────────
┌──────────── NLE timeline (ruler + lanes + blocks) ───────────┐
└──────────────────────────────────────────────────────────────┘

[DisclosurePanel "Advanced"]
  └── camera-count slider · zoom slider · "tracklets only" filter

[DisclosurePanel "Debug"]
  └── confidence heatmap · lane coloring scheme · raw counts

Side sheets (open on demand, NOT always-visible):
  • Alternatives — opened by clicking a tracklet's "alternatives" pill (was always-visible top-5 panel)
  • Rerun-Association config — opened by clicking "Rerun" in ActionsFooter, contains thresholds etc.

[ActionsFooter]  [ExecutionTargetToggle]  [Rerun Association ▾]  [Continue → Stage 5]
```

Region collapses:
- "Top 5 Alternatives" panel → on-demand side sheet.
- Camera-count + zoom sliders → Advanced (not always-visible).
- Confirm/Remove per-row stays inline on tracklet rows (essential).

- Consumes: `<StageShell>`, `<PlaybackControls>`, `<TrackletPreview>`, `<DisclosurePanel>` ×2, side sheets via shadcn `Sheet`.

### Stage 5 — Refinement (target: 550 → ~310 lines)

```
[ContractBanner]   Stage 5 · Refinement
                   Needs: confirmed trajectories [Stage 4 ✓]
                   Produces: refined trajectories
─────────────────────────────────────────────
┌── FrameGrid (selected reference frames) ──┐
│   [thumb] [thumb] [thumb] [×remove chips] │
└────────────────────────────────────────────┘

[PlaybackControls]   prev10 · play · next10 · speed

[DisclosurePanel "Advanced"]
  └── Clear selection · scrubber

[ActionsFooter]  [Re-Search]  [Continue → Stage 6]
```

- Stage 5 stays local (no Kaggle toggle needed; see §8).
- Consumes: `<StageShell>`, `<FrameGrid>`, `<PlaybackControls>`, `<DisclosurePanel>`.

### Stage 6 — Output (target: 900 → ~480 lines)

```
[ContractBanner]   Stage 6 · Output
                   Needs: refined trajectories [Stage 5 ✓]
                   Produces: MP4 · JSON · CSV
─────────────────────────────────────────────
┌── Video preview ──┐  ┌── Trajectory map ──┐
│                   │  │                    │
└───────────────────┘  └────────────────────┘
[PlaybackControls]

[ Export format:  ○ MP4   ○ JSON   ○ CSV ]
                              [Generate & Download]

[DisclosurePanel "Advanced"]
  └── quality selector · speed · per-trajectory checkboxes

[DisclosurePanel "Debug"]
  └── raw trajectory JSON preview
```

- Stage 6 stays local.
- Consumes: `<StageShell>`, `<PlaybackControls>`, `<DisclosurePanel>` ×2.

---

## 7. State Refactor (minimal path)

**Decision: KEEP the existing 6 stores. Do not consolidate.** Rationale:

- The audit's complaints about state are 80% UI-driven (scattered status surfaces, no canonical badge). Fixing the components alone resolves the felt pain.
- Consolidating Zustand stores risks regressing the persisted `useStageExecutionStore`, the `downstreamInvalidateGeneration` mechanism, and the carefully-scoped `useTimelineStore.applyTracksReplaceKeepingMeta` flows.
- Cost of consolidation is high (rewriting every consumer); benefit is low once `<StageStatusBadge>` and `<StageStatusDot>` give us a single read surface.

Minimal additions only:
1. **New selector helper** `frontend/src/store/selectors/stageState.ts` exporting `useStageState(stage: StageNumber): StageState` that composes `usePipelineStore.stages[stage].status`, `downstreamInvalidateGeneration`, and an upstream-satisfied check into the canonical `StageState` enum. All status components consume this selector.
2. **New tiny store** `useUIDisclosureStore` (persisted) keying `{ [stage]: { advancedOpen: boolean; debugOpen: boolean } }` so users' preference for collapsed-vs-open Advanced/Debug sticks. Lives at `frontend/src/store/ui-disclosure.ts`.
3. **No changes to existing store shapes.**

---

## 8. Kaggle Toggle Universalization

The `<KaggleExecutionToggle>` (currently only used by Stage 3) becomes the universal `<ExecutionTargetToggle>` mounted in `<ActionsFooter>` for every stage that genuinely has a Kaggle pipeline.

| Stage | Target options              | Default | Rationale |
| ----- | --------------------------- | ------- | --------- |
| 0 Upload      | local only           | local   | Upload is browser → backend; no Kaggle equivalent. Hide toggle. |
| 1 Detection   | local · kaggle       | kaggle  | GPU-heavy; Kaggle preferred per copilot-instructions. |
| 2 Selection   | local only           | local   | Pure UI selection. Hide toggle. |
| 3 Inference   | local · kaggle       | kaggle  | Already implemented. |
| 4 Timeline    | local · kaggle       | local   | CPU-OK; Kaggle still useful for large gallery rebuilds. |
| 5 Refinement  | local only           | local   | UI-driven re-search. Hide toggle. |
| 6 Output      | local only           | local   | Visualization. Hide toggle. |

`<ActionsFooter>` always renders the slot but only shows the toggle on stages where `executionTargetEnabled === true` (Stages 1, 3, 4). Persistence already exists via `useStageExecutionStore`.

When `target = kaggle` and a run is active, `<RunStageWidget>` slides the existing `<KaggleStatusPanel>` into a card under the run button (no layout breakage on local).

---

## 9. Implementation Phases

Each phase is independently verifiable. After each phase: run `cd frontend && pnpm tsc --noEmit && pnpm lint` and `pytest tests/`. Rollback = revert phase commit.

### Phase 1 — Shared primitives (no behavior change)
- Create `frontend/src/components/pipeline/{status,disclosure,feedback,header,shell,run,media}/` skeletons.
- Implement `StageStatusBadge`, `StageStatusDot`, `status/types.ts`, `status/status-meta.ts`.
- Implement `DisclosurePanel`, `ErrorBanner`, `StalenessChip`.
- Implement `useStageState` selector in `frontend/src/store/selectors/stageState.ts`.
- Add `useUIDisclosureStore`.
- **Files:** all new, no existing files touched.
- **Validate:** `tsc`, `lint`, `pytest`.
- **Rollback:** delete new files.

### Phase 2 — PipelineRunHeader + sidebar status dots
- Implement `<PipelineRunHeader />` and mount it in `main-dashboard.tsx` above the workspace.
- Replace the inline status circle in sidebar rows with `<StageStatusDot>` (preserve cloud overlay).
- Wire `useStageState` so sidebar reflects idle/blocked/running/done/stale/error.
- **Files:** `frontend/src/components/layout/main-dashboard.tsx`, new `frontend/src/components/pipeline/header/PipelineRunHeader.tsx`.
- **Validate:** visually verify sidebar dot colors + header chips render; `tsc`, `lint`.
- **Rollback:** revert main-dashboard edits.

### Phase 3 — StageShell + ContractBanner on all stages (no logic changes)
- Implement `<StageShell>`, `<ContractBanner>`, `<ActionsFooter>`.
- Wrap each of the 7 stage components in `<StageShell>`. Move existing run buttons / continue buttons into `<ActionsFooter>`. Move existing error renderings into `<ErrorBanner>`.
- Pull contract text (Needs/Produces) into a per-stage `contract.ts` data module under each stage folder.
- **Files:** `frontend/src/components/pipeline/shell/*`, all 7 `stages/*.tsx` wrappers.
- **Validate:** `tsc`, `lint`, every stage page still loads identically below the new banner+footer.
- **Rollback:** unwrap stages.

### Phase 4 — Redesign Stage 0/1/2
- Apply per-stage layouts from §6.
- Replace ad-hoc badges/spinners with `<StageStatusBadge>`.
- Wrap secondary controls in `<DisclosurePanel tier="advanced">` / `"debug"`.
- Extract `<PlaybackControls>`, `<TrackletPreview>`, `<FrameGrid>` and replace inline usages.
- **Files:** `stages/upload-stage.tsx`, `stages/detection-stage.tsx`, `stages/selection-stage.tsx`, new `pipeline/media/*`.
- **Validate:** `tsc`, `lint`, `pytest`. Manually walk a run through Stage 0→2.
- **Rollback:** revert stage files.

### Phase 5 — Redesign Stage 3 (Inference)
- Collapse 7 cards into Model card + Advanced (sub-tabs Location/Time/Fusion) + Debug.
- Move dataset selector into ContractBanner's Needs chip.
- Move execution toggle into ActionsFooter (consumes `<ExecutionTargetToggle>`).
- Integrate existing `<KaggleStatusPanel>` via `<RunStageWidget>`.
- **Files:** `stages/inference-stage.tsx`, `pipeline/run/RunStageWidget.tsx`, `pipeline/run/ExecutionTargetToggle.tsx`.
- **Validate:** `tsc`, `lint`, `pytest`. Run local + Kaggle inference end-to-end.
- **Rollback:** revert.

### Phase 6 — Redesign Stage 4 (Timeline)
- Move Top-5 Alternatives into a side sheet triggered by tracklet row.
- Move camera-count + zoom + filter into Advanced.
- Move Rerun-Association controls into ActionsFooter ▾ side sheet.
- Keep video grid + NLE timeline + left tracklet rail always-visible.
- **Files:** `stages/timeline-stage.tsx`, new side-sheet components co-located in `stages/timeline/`.
- **Validate:** `tsc`, `lint`, `pytest`. Full Stage-4 playthrough including confirm + rerun + alternatives.
- **Rollback:** revert.

### Phase 7 — Redesign Stage 5/6
- Apply per-stage layouts.
- Extract any remaining inline progress/badge/error patterns to canonical components.
- **Files:** `stages/refinement-stage.tsx`, `stages/output-stage.tsx`.
- **Validate:** `tsc`, `lint`, `pytest`.
- **Rollback:** revert.

### Phase 8 — Universalize Kaggle toggle
- Add `executionTargetEnabled` flag per stage in `pipeline/run/ExecutionTargetToggle.tsx`.
- Surface toggle in `<ActionsFooter>` for Stages 1, 3, 4.
- Verify `useStageExecutionStore` persistence still works for each.
- **Files:** `pipeline/shell/ActionsFooter.tsx`, `pipeline/run/ExecutionTargetToggle.tsx`, three stage files.
- **Validate:** `tsc`, `lint`, `pytest`. Verify per-stage persistence in localStorage.
- **Rollback:** revert.

### Phase 9 — Polish, a11y, micro-interactions
- Add focus rings, ARIA labels on status badges (`aria-label={state}`), `role="status"` for running indicators.
- Honor `prefers-reduced-motion` for spinners → pulse only.
- Add the `?` help side sheet with placeholder content (frontend guide doc deferred).
- Audit color contrast (status palette must hit WCAG AA on dark + light themes).
- **Files:** all pipeline/* components.
- **Validate:** axe-core check; `tsc`, `lint`, `pytest`.
- **Rollback:** revert.

### Phase 10 — Cleanup (optional, low priority)
- Delete dead code paths in stage files now unreached.
- Verify final line counts vs §6 targets.
- Update `frontend/README.md` (one-line note pointing to this spec).

---

## 10. Out of Scope (explicit)

- **No backend changes.** `backend/`, FastAPI routers, API request/response shapes — all frozen.
- **No new routes.** `/reid`, `/fusion`, `/eval` stay as-is; main pipeline stays at `/`.
- **No new heavy deps.** Allowed: nothing. Reuse shadcn `Accordion`, `Sheet`, `Tooltip`, `Tabs`, `Slider`, `Popover`, `Button`, `Calendar`, `Select` — all already in repo.
- **No state-store consolidation** beyond the two additions in §7.
- **No new top-level pages.** Side sheets only.
- **No tutorial overlays / coach marks / first-run tour.**
- **No mobile/responsive overhaul.** Target stays desktop, sidebar collapsible. Audit didn't flag mobile.
- **No theming/branding changes.** Existing dark theme + color palette stays; we only add status semantics.
- **No `docs/frontend-guide.md` content** — the help sheet renders a placeholder until that doc is written separately.

---

## 11. Risks & Open Questions (defaults chosen — Coder can execute without input)

| # | Risk / Question                                                                 | Chosen default                                                                                          |
|---|---------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| 1 | Stage-4 timeline at 2100 lines — feasible to refactor in one phase?             | Yes, but defer non-visual logic (playhead, scroll sync) untouched — only restructure JSX + extract.     |
| 2 | Should `stale` state require a backend signal or is `downstreamInvalidateGeneration` enough? | Sufficient. We treat any incremented generation since stage's last completion as `stale`.        |
| 3 | Where does the ContractBanner's "Needs" dataset chip persist its choice?        | Reuse existing dataset selection state in `useSessionStore` — chip is a thin renderer over it.          |
| 4 | Should `PipelineRunHeader` be hidden on `/reid`, `/fusion`, `/eval` routes?     | Yes — those routes are out of the pipeline. Header renders only inside `<MainDashboard>`.               |
| 5 | Does the universal Kaggle toggle on Stage 1 actually work today?                | The toggle UI ships now; if Stage 1 Kaggle execution isn't wired backend-side it will surface backend error — acceptable (out-of-scope).  |
| 6 | "Last run: Xm ago" — where do we store completion timestamps?                   | Add `completedAt?: number` to `StageProgress` in-memory only (no persistence). Stamp on status flip to `done`. Falls back to "—" if never run. |
| 7 | Side sheets for Stage-4 alternatives — does this hurt power users who relied on always-visible Top-5? | The sheet opens via a one-click pill on each tracklet row; opens persistently per tracklet selection. Acceptable trade for the density win. |
| 8 | Should we add Storybook for the new primitives?                                 | No — out of scope; adds a heavy dep. Document via TSDoc only.                                           |

---

## Executive Summary (15 bullets)

1. Introduce a single `<StageShell>` (ContractBanner + workspace + ActionsFooter) that wraps every stage — no bespoke stage layouts survive.
2. Add a persistent 40px `<PipelineRunHeader>` showing runId, current stage, overall progress, last-run timestamp, and an error pill.
3. Every stage gets a `<ContractBanner>` declaring upstream "Needs" and downstream "Produces" as chips — blocked stages link directly back to the failed prerequisite.
4. Replace every ad-hoc spinner/badge/error with two canonical components: `<StageStatusBadge>` (pill) and `<StageStatusDot>` (sidebar glyph), driven by a 6-state enum: idle, blocked, running, done, stale, error.
5. Status is colorblind-safe: every state has a color **and** an icon **and** a label.
6. Three-tier progressive disclosure (Essential / Advanced / Debug) via one `<DisclosurePanel>` primitive — solves the inference and timeline knob explosion without modes.
7. Stage 3 (Inference) collapses from 7 stacked cards to 1 Model card + 1 collapsed Advanced sub-tab group (Location/Time/Fusion) + 1 Debug panel — target: 1250 → ~480 lines.
8. Stage 4 (Timeline) keeps video grid + NLE timeline + tracklet rail as the 3 visible regions; Alternatives panel becomes a side sheet; camera/zoom sliders move to Advanced — target: 2100 → ~900 lines.
9. Dataset selection migrates from a Stage-3 card to a chip in the ContractBanner's "Needs" line (it's an upstream concept, not a stage knob).
10. Kaggle/local execution toggle universalizes to Stages 1, 3, 4 via `<ExecutionTargetToggle>` mounted in `<ActionsFooter>`; Stages 0/2/5/6 stay local-only.
11. State refactor is intentionally minimal: keep all 6 Zustand stores, add one selector (`useStageState`) and one tiny persisted UI store for disclosure panel state.
12. New primitives all live under `frontend/src/components/pipeline/` in 7 sub-folders (status, disclosure, feedback, header, shell, run, media) — clean import surface.
13. 9 implementation phases, each independently verifiable with `tsc --noEmit`, `pnpm lint`, and `pytest tests/`; phases 1–3 ship infrastructure with zero behavior change before any stage rewrite begins.
14. Hard guarantees preserved: Phase 1-15 features (fusion mode, Kaggle per-stage toggle, credentials modal, sidebar gear, Kaggle status panel + cancel, sidebar cloud icon, `useStageExecutionStore`, model registry overrides) all keep working — they are consumed by the new primitives, not replaced.
15. No backend changes, no new routes, no new heavy deps, no API shape changes — all explicitly out of scope per §10.
