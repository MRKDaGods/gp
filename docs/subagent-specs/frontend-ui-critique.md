# MTMC Frontend UI/UX Critique — Phase 8 Review
**Date:** May 27, 2026
**Branch:** `feature/pipeline-model-integration`
**Reviewer:** MTMC Planner
**Source spec:** `docs/subagent-specs/frontend-ui-redesign.md`
**Reviewed commits:** 235295c → 10086b8 (Phases 1–8)

---

## 1. Executive Verdict

The redesign delivered the **structural backbone** (StageShell, ContractBanner, ActionsFooter, status enum, DisclosurePanel, PipelineRunHeader, Kaggle toggle universalization), and on the spec's North-Star principles it's a clear win — every stage now has a consistent shell, a Needs/Produces contract, and a single status grammar. **But the redesign stopped at the bones.** The two highest-leverage spec items — (a) `stale` state propagation via `useStageState` selector and (b) the dataset chip migrating into the ContractBanner — are unimplemented. `StalenessChip` is dead code, `toStageStatus` cannot return `blocked`/`stale`, and the per-stage status the ContractBanner displays is hand-rolled in `main-dashboard.tsx` with a `TODO(phase-4)` comment. The Actions footers for Stages 1, 3, 4 are also visually overloaded: full ExecutionTargetToggle cards (with nested credentials panels) collide with Run/Cancel/Continue buttons in a strip that was speced as 64px. **Biggest remaining gap:** stale tracking and footer density. Both are addressable in one tight Phase 9 commit chain.

---

## 2. Must-Fix in Phase 9

Low-risk, high-impact, all surface-level.

- [main-dashboard.tsx#L107](frontend/src/components/layout/main-dashboard.tsx#L107) — Replace the `TODO(phase-4)` in `deriveSidebarStageStatus` with real `stale` detection: compare `pipelineStage.status === "completed"` against `usePipelineStore.downstreamInvalidateGeneration` (or a per-stage generation snapshot recorded on completion). Without this, the entire `stale` half of the canonical 6-state system is unreachable.
- [frontend/src/store/selectors/stageState.ts](frontend/src/store/selectors/stageState.ts) — **File does not exist.** Spec §7.1 required `useStageState(stage)` to be the single source of truth. Today every consumer (sidebar, ContractBanner, Detection, Inference, Refinement, Timeline) re-derives status differently. Create it. Have it return the canonical `StageStatus` including `blocked` and `stale`. Replace `toStageStatus` callsites in stages with the selector.
- [pipeline/status/types.ts](frontend/src/components/pipeline/status/types.ts#L16) — `toStageStatus` can only return `idle | running | done | error`. Either extend it (preferred) or delete it and force consumers through `useStageState`. As-is it silently degrades every status surface.
- [pipeline/header/PipelineRunHeader.tsx#L84-L95](frontend/src/components/pipeline/header/PipelineRunHeader.tsx#L84) — Two bugs: (a) error pill is hardcoded "1 error" — count the actual number of stages in error; (b) the fallback `<StageStatusBadge status="idle" label="No errors" />` is semantic noise and visually competes with the runId chip — drop the badge when no error exists.
- [main-dashboard.tsx#L390](frontend/src/components/layout/main-dashboard.tsx#L390) — `lastRunLabel="-"` is hardcoded. Spec §11 risk #6 mandated stamping `completedAt?: number` on `StageProgress` when status flips to `done`, then formatting as "Xm ago" in the header. Wire it.
- [inference-stage.tsx#L142-L150](frontend/src/components/stages/inference-stage.tsx#L142) — The "Model registry warning" inline yellow box bypasses `ErrorBanner`. Either route warnings through a new `<WarningBanner>` (cheap) or fold into `ErrorBanner` with a `severity` prop. Inconsistent today.
- [main-dashboard.tsx](frontend/src/components/layout/main-dashboard.tsx#L356) — `GlobalProcessingBanner` is rendered above `PipelineRunHeader`. These two strips duplicate runtime info (current stage + progress). Spec §2.3 makes PipelineRunHeader the canonical surface. Either remove `GlobalProcessingBanner` from the dashboard or scope it to "complete/error toast" mode only.
- [pipeline/feedback/StalenessChip.tsx](frontend/src/components/pipeline/feedback/StalenessChip.tsx) — Dead code; not imported anywhere. Once `useStageState` returns `stale`, mount this chip in the sidebar row's secondary line and in ContractBanner's `Produces` chips for stale-flagged outputs.
- [main-dashboard.tsx#L201-L207](frontend/src/components/layout/main-dashboard.tsx#L201) — The sidebar shows a `Server`/`Cloud` glyph for **every** stage row when expanded, including Stages 0/2/5/6 which are local-only by spec §8. Hide the glyph for those stages.

## 3. Should-Fix in Phase 9

Polish, lower priority.

- [inference-stage.tsx — InferenceActions L350-L390](frontend/src/components/stages/inference-stage.tsx#L350) — The footer mounts **two full ExecutionTargetToggle cards** (one per Stage 2 / Stage 3). Each toggle is a vertical block with description + nested Kaggle credentials card. In an ActionsFooter intended as ~64px sticky, that's >180px tall and pushes the workspace upward. Collapse to a single toggle that controls both backend stages (rare to want them on different targets) or use a compact icon-only toggle variant for footers.
- [pipeline/run/ExecutionTargetToggle.tsx](frontend/src/components/pipeline/run/ExecutionTargetToggle.tsx) — Add a `variant: "full" | "compact"` prop. `compact` = icon + switch only, no description, no inline creds card (creds open via popover). Use `compact` everywhere it lives in `ActionsFooter`.
- [pipeline/run/RunStageWidget.tsx](frontend/src/components/pipeline/run/RunStageWidget.tsx) — Renders both a Run Button **and** a `StageProgressCard` together. In ContextOK in stage body. In `ActionsFooter` (Detection uses it there) the embedded Card is too heavy. Either split into two widgets (`<RunButton>` for footer, `<StageProgressCard>` for body) or accept a `mode="button-only"` prop.
- [detection-stage.tsx#L1024-L1037](frontend/src/components/stages/detection-stage.tsx#L1024) — `DetectionStageActions` renders `RunStageWidget` (with embedded progress card) inside the footer. Use a slim Run button only.
- [upload-stage.tsx#L361](frontend/src/components/stages/upload-stage.tsx#L361) — `UploadStageActions` reuses `RunStageWidget` with `runLabel="Continue to Stage 1"`. That's overloading the "Run" button to mean "Continue & Run". Confusing. Use a standalone Continue button that triggers the same handler, no progress card.
- [inference-stage.tsx#L162-L184](frontend/src/components/stages/inference-stage.tsx#L162) — `InferenceStage` body has two side-by-side `RunStageWidget`s. With no `onRun` prop they degrade to progress cards only — but the conditional rendering inside the widget (`showKaggle ? <KaggleStatusPanel> : <StageProgressCard>`) makes intent unclear. Refactor to direct `StageProgressCard` + `KaggleStatusPanel` usage in the body, drop the widget for non-button cases.
- [inference-stage.tsx](frontend/src/components/stages/inference-stage.tsx#L142) — Per Spec §6 Stage 3: dataset selection was supposed to migrate to a **ContractBanner chip in the `Needs` line** with a `<DatasetSwitcher />` popover. It's still buried under `Advanced → InferenceSourceCard`. Surface it. ContractChip already supports `missing`/`stale` markers; add a generic `interactive` chip type.
- [timeline/AlternativesSheet.tsx#L89](frontend/src/components/stages/timeline/AlternativesSheet.tsx#L89) — Uses Radix `Dialog` styled as a right-slide sheet. shadcn `Sheet` isn't installed in the repo, so this is correct *for now*, but it lacks the trap-focus + slide-from-right animation polish of a real Sheet. Either: (a) install `@/components/ui/sheet` via shadcn CLI (one new file), or (b) leave as-is and add a one-line comment explaining the choice. Spec §10 prohibits new heavy deps but shadcn Sheet is zero-cost.
- [pipeline/shell/ContractBanner.tsx#L107](frontend/src/components/pipeline/shell/ContractBanner.tsx#L107) — `helpText` opens a Tooltip, not a Popover. Spec §2.4 said "popover with 3-sentence plain-English explanation". Tooltip closes on cursor exit, popover is sticky and clickable. Swap for `<Popover>`.
- [pipeline/header/PipelineRunHeader.tsx](frontend/src/components/pipeline/header/PipelineRunHeader.tsx#L93) — Error pill is not clickable. Spec §2.3 wanted it to open a side panel listing the last 5 errors. At minimum, make it a button that focuses the offending stage (`setCurrentStage(errorStageId)`).
- [main-dashboard.tsx#L370-L390](frontend/src/components/layout/main-dashboard.tsx#L370) — All visited stage components stay mounted as hidden `role="tabpanel"`. Timeline + Inference each carry expensive poll/effect chains. Consider unmounting after `N` minutes of inactivity, or guarding background polls with `if (!isActive) return;`. Not blocking, but worth flagging.

## 4. Defer Beyond Phase 9

- **Stage 1 Detection LOC reduction** (target was 380, current is ~973). The video canvas + DoubleBufferedFrameImg path is doing real work; extracting `<DetectionCanvas>` and a `useDetectionPlayback` hook is a separate dedicated refactor (~1 day). Tag for Phase 10.
- **Stage 4 Timeline LOC** (1707 lines). Sub-components are extracted to `stages/timeline/*`, but data-loading hooks (`useTimelineTracks`, `useAlternativesByTrack`, `useAutoAssociationRefresh`) are still inline. Worth a Phase 10 follow-up — does not block correctness.
- **Stage 6 Output LOC** (1196 lines). Map + share + QR block (~400 lines, L912-L1100) deserves its own `<OutputMapShareCard />`. Phase 10.
- **Help "?" side sheet** with placeholder docs (Spec §9 Phase 9). Low value until `docs/frontend-guide.md` exists. Defer.
- **`useUIDisclosureStore`** (Spec §7.2) for persisting per-stage Advanced/Debug open state. Nice-to-have, not blocking. Phase 10.
- **Per-stage `completedAt` persistence**. In-memory is fine for Phase 9 (Spec §11 risk #6 explicitly said no persistence). Cross-session "last run" is a separate effort.
- **Storybook for primitives** (Spec §11 risk #8). Out of scope, stays out of scope.

## 5. Spec Adherence Scorecard

| Spec section | Item | Status | Note |
|---|---|---|---|
| §1.1 | One canonical pipeline grammar (StageShell everywhere) | ✅ | All 7 stages wrapped in `main-dashboard.tsx` |
| §1.2 | Glanceable state (single badge/dot/6-state enum) | ⚠ | Enum exists; `toStageStatus` can't emit `blocked`/`stale`; consumers diverge |
| §1.3 | Three-tier disclosure (Essential/Advanced/Debug) | ✅ | `DisclosurePanel` adopted in all 7 stages |
| §1.4 | Inputs/outputs first-class | ⚠ | ContractBanner exists; dataset chip for Stage 3 still in Advanced |
| §1.5 | Run location is a stage-level decision | ✅ | `ExecutionTargetToggle` in footers for 1/3/4 |
| §1.6 | Pipeline run is the protagonist | ⚠ | `PipelineRunHeader` mounted; lastRunLabel + error pill not wired |
| §1.7 | Colorblind-safe (color + icon + label) | ✅ | `status-meta.ts` enforces all three |
| §1.8 | No silent invalidation (stale chip everywhere) | ❌ | `StalenessChip` is dead code; stale never computed |
| §2.2 | Sidebar row redesign | ✅ | StageStatusDot + meta strip + cloud glyph |
| §2.2 | Sidebar shows "needs Stage X" reason | ✅ | Implemented in `deriveSidebarStageStatus` text |
| §2.3 | PipelineRunHeader runId/stage/progress/error/lastrun | ⚠ | runId + stage + progress ✅; lastrun + error pill incomplete |
| §2.4 | ContractBanner Needs/Produces chips | ✅ | `STAGE_CONTRACTS` populated |
| §2.4 | ContractBanner "?" help popover (3-sentence) | ⚠ | Help text exists but uses Tooltip, not Popover |
| §3 | 6-state enum + canonical components | ⚠ | Enum/meta correct; selector path incomplete |
| §4.1 Stage 0 | Essential/Advanced disposition | ✅ | |
| §4.1 Stage 1 | Footer toggle, ad-hoc badges removed | ⚠ | Detection has DisclosurePanels but body still ~973 lines |
| §4.1 Stage 2 | Selection grid + Advanced disclosure | ✅ | Clean |
| §4.1 Stage 3 | Collapse 7 cards into Model/Advanced/Debug | ⚠ | Inference Model card + Advanced + Debug ✅; dataset chip ❌ |
| §4.1 Stage 4 | Alternatives → side sheet | ✅ | `AlternativesSheet` (Dialog-styled) |
| §4.1 Stage 4 | Camera/zoom in Advanced | ✅ | `TimelineAdvancedControls` |
| §4.1 Stage 5 | Refinement essentials | ✅ | |
| §4.1 Stage 6 | Output essentials + Advanced/Debug | ⚠ | Done, but Map/QR block still inline (~400 lines) |
| §5 | Shared primitives folder structure | ✅ | All 7 sub-folders present |
| §7.1 | `useStageState` selector | ❌ | File never created |
| §7.2 | `useUIDisclosureStore` | ❌ | File never created |
| §8 | Kaggle toggle universalization (1/3/4) | ✅ | Per Phase 8 audit |
| §9 Phase 9 | a11y, popovers, help sheet, axe-core | ⚠ | Partial; this critique is the gate |

Legend: ✅ done · ⚠ partial / works but spec-deviant · ❌ missing

## 6. Per-Stage UX Critique (user's first-open POV)

- **Stage 0 — Upload.** Walking in cold, the ContractBanner says "Needs: — / Produces: Video · runId" which is honest and instantly tells me this is the entry point. Drag-drop is dominant; Demo Mode is a sensible escape hatch. Advanced disclosure for Kaggle artifact import is correctly hidden. **Annoyance:** the `UploadStageActions` footer's "Continue to Stage 1" button is actually a `RunStageWidget` that also renders a `StageProgressCard` — so the footer carries a Card. Should be a button. Also: when no video is selected, the disabled button gives no hint why ("Select a video first" tooltip missing).
- **Stage 1 — Detection.** Video canvas dominates correctly. Status legible via ContractBanner. **Annoyance:** the footer is dense — ExecutionTargetToggle (full card with desc + creds) + Cancel + RunStageWidget (button + progress card) + Continue. It overflows the "64px" intent of an ActionsFooter and reads as a separate region rather than a strip. Selection mode toggle correctly tucked into Advanced. Frame counter / hit log correctly in Debug. Can I tell what this stage does in <5s? — yes, ContractBanner sells it.
- **Stage 2 — Selection.** Cleanest stage. ContractBanner explains the role, the grid is the workspace, Advanced + Debug correctly tucked. **Annoyance:** the Selected Tracklets aside takes 320px on the right at xl breakpoints — for a stage whose only job is "pick tracklets" this is screen real estate well-spent, but Multi-select toggle is inside Advanced — discoverable only after one click. Minor.
- **Stage 3 — Inference.** Big improvement vs pre-redesign, but **two problems**: (a) Dataset selection is buried — spec said it should be a ContractBanner `Needs` chip with a popover switcher; instead it's inside `Advanced → Source`. Users will not find it. (b) The footer has TWO `ExecutionTargetToggle`s side by side (Stage 2 features, Stage 3 indexing) — 99% of users want them to match. Confusing surface. The Model card up top is great; Advanced/Debug discipline is correctly enforced.
- **Stage 4 — Timeline.** Workspace is the right three regions (tracklet rail / video grid / NLE timeline). AlternativesSheet as a side dialog is the right call vs the old always-visible panel. **Annoyance:** the `Run Association` button in the footer dispatches a `window.dispatchEvent(TIMELINE_RERUN_ASSOCIATION_EVENT)` — works, but it's a hidden coupling between footer (mounted by `MainDashboard`) and body (mounted inside the same StageShell). Slightly brittle. Same for `Alternatives` button → custom event. If the body unmounts (rare but possible during stage switch animations) the event listener evaporates. Document or refactor to a shared Zustand action.
- **Stage 5 — Refinement.** Frame grid + Prev/Play/Next + Re-Search → footer. Reads cleanly. **Annoyance:** the stage body has its own sticky `border-b` header strip showing "N/16 reference frames" + chips — this duplicates the role of ContractBanner above. Consolidate the chips into the ContractBanner's `Produces` chips with counts.
- **Stage 6 — Output.** Map + video + export format radio — clear. **Annoyance:** the export format radio lives in the **footer**, not next to the Generate button in the body. So scanning the body you see "Generate & Download" implied but no format choice — you have to look down at the footer. Either move the radio into the body (right next to the button) or merge the body's `Generate` UI into the footer. The Map/QR block is 400 lines inline; consider extracting (Phase 10).

## 7. Phase 9 Work Plan

Five focused commits. Each ends with `cd frontend && pnpm tsc --noEmit && pnpm lint && pnpm build`.

### Commit 1 — `feat(ui): stale + blocked status selector`
**Goal:** wire the 6-state enum end to end.

- Create `frontend/src/store/selectors/stageState.ts` exporting `useStageState(stage: StageNumber): StageStatus`. Compose: `pipelineStages[stage].status` + `error` + upstream-satisfied check + `downstreamInvalidateGeneration` compared to a per-stage `completedAtGeneration` snapshot.
- Add `completedAt?: number` and `completedAtGeneration?: number` to `StageProgress` type (in `frontend/src/types/`). Stamp on `updateStageProgress` when status transitions to `completed`.
- Replace inline `toStageStatus(stageProgress)` callsites in: `detection-stage.tsx`, `inference-stage.tsx`, `timeline-stage.tsx`, `refinement-stage.tsx`, `output-stage.tsx`, `upload-stage.tsx`, `main-dashboard.tsx` (drop `deriveSidebarStageStatus`).
- Either extend `toStageStatus` to also emit `blocked`/`stale` (preferred) or delete it and force the selector everywhere.
- Mount `<StalenessChip>` in `ContractBanner` next to status badge when state is `stale`.

**Commit msg:** `feat(ui): unify status via useStageState selector with stale + blocked`

### Commit 2 — `fix(ui): PipelineRunHeader correctness + dedup global banner`
**Goal:** make the persistent header trustworthy.

- `pipeline/header/PipelineRunHeader.tsx`: count actual error stages (`stages.filter(s => s.status === "error").length`); drop the "No errors" idle badge; make the error pill a `<Button>` that calls a new `onSelectErrorStage?: (id: StageNumber) => void` and focuses the first stage in error.
- Compute `lastRunLabel` from the most-recent `stages[].completedAt` and format as "Xm ago" (use existing `formatDuration` or inline `Math.round((Date.now() - ts)/60000) + "m ago"`); update every 30s via `useEffect` interval.
- In `main-dashboard.tsx`: pass `lastRunLabel` and `onSelectErrorStage`; remove `<GlobalProcessingBanner />` from the dashboard or scope it to "complete/error toast" mode only (extract a `<RunCompleteToast>` component if its complete-state UX is worth preserving).

**Commit msg:** `fix(ui): wire PipelineRunHeader lastRun, error count, error focus; drop duplicate banner`

### Commit 3 — `refactor(ui): compact ExecutionTargetToggle + lean ActionsFooter`
**Goal:** restore the 64px footer intent.

- `pipeline/run/ExecutionTargetToggle.tsx`: add `variant?: "full" | "compact"` prop. `compact` renders icon + switch + small label only; opens credentials in a Popover trigger by a separate "creds" icon button. Default `variant="full"` to preserve existing call sites.
- `pipeline/run/RunStageWidget.tsx`: add `mode?: "button-only" | "card"`. `button-only` renders just the Run button (still consumes status for the spinner). Stage body callers stay `mode="card"` (default).
- `detection-stage.tsx → DetectionStageActions`: use `<ExecutionTargetToggle variant="compact">` + `<RunStageWidget mode="button-only">`. Drop the embedded progress card from the footer (it's shown inline elsewhere).
- `inference-stage.tsx → InferenceActions`: collapse two toggles into one (`stage={2}` with a sub-label explaining it controls both Stage-2 + Stage-3 backend dispatch) OR mark both as `compact`. Decide based on whether anyone actually splits targets — recommend single compact toggle.
- `timeline-stage.tsx → TimelineStageActions`: compact toggle.
- `upload-stage.tsx → UploadStageActions`: replace `RunStageWidget` with a plain `<Button>Continue to Stage 1</Button>` that calls the same handler; drop the progress card from the footer.

**Commit msg:** `refactor(ui): compact ExecutionTargetToggle, slim ActionsFooter, drop embedded progress cards`

### Commit 4 — `refactor(ui): dataset chip in ContractBanner Needs + warning banner + popover help`
**Goal:** finish Spec §2.4 + §6 Stage 3.

- Extend `ContractChip` with `interactive?: boolean` and an optional `onClick`. In `ContractBanner.tsx`, render interactive chips as buttons that open a popover.
- Create `frontend/src/components/stages/inference/InferenceDatasetChip.tsx` — a button rendering `dataset: cityflowv2 ▾` that opens a `<Popover>` containing the existing dataset selector UI (extract from `InferenceSourceCard`).
- In `main-dashboard.tsx`, override the Stage-3 contract on render: append the `InferenceDatasetChip` to `needs`. (Or expose a `extraNeeds` prop on `StageShell` to keep `STAGE_CONTRACTS` static.)
- `pipeline/feedback/ErrorBanner.tsx`: add `severity?: "error" | "warning"` prop. Replace the inline "Model registry warning" yellow box in `inference-stage.tsx` with `<ErrorBanner severity="warning" title="Model registry warning" message={...}>`.
- `pipeline/shell/ContractBanner.tsx`: swap `Tooltip` around the help icon for `<Popover>` (longer-form 3-sentence explanation now possible).
- `main-dashboard.tsx`: hide the `Server`/`Cloud` sidebar glyph for Stages 0/2/5/6 (`isLocalOnlyStage(id)` helper; reuse the Spec §8 table).

**Commit msg:** `refactor(ui): dataset chip in ContractBanner; warning severity; popover help; local-only sidebar`

### Commit 5 — `polish(ui): a11y pass + dead-code cleanup + Stage 5/6 chip dedup`
**Goal:** Phase 9 closeout.

- Sweep `pipeline/*` for missing `aria-label`s on icon-only buttons (header copy-runId button, sidebar dataset button, ExecutionTargetToggle creds icon).
- `pipeline/status/StageStatusBadge.tsx`: respect `prefers-reduced-motion` on the running spinner (Tailwind `motion-safe:animate-spin`).
- `refinement-stage.tsx`: remove the body's own sticky border-b chip strip; move chip-count into ContractBanner `Produces` chip ("reference frames: N/16").
- `output-stage.tsx`: move the export-format radio from `OutputStageActions` into the stage body adjacent to where Generate & Download is contextualized. Footer keeps only the Generate button + Continue (if Continue exists).
- Verify `StalenessChip` is now consumed (Commit 1) — if not, delete.
- Run `pnpm exec axe http://localhost:3000` (if axe-cli is wired) or document the manual a11y checklist in a code comment at top of `main-dashboard.tsx`.

**Commit msg:** `polish(ui): a11y labels, reduced-motion spinner, chip dedup in Stage 5/6, dead-code cleanup`

---

**Critic's note to the Coder:** Resist scope creep. The three deferred LOC reductions (Stages 1/4/6) are explicitly out of Phase 9 — do not refactor them while doing the above. If you find a bug in those files during Phase 9 work, file a Phase-10 note in `docs/findings.md` and move on.