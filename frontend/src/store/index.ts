import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import type { ModelEntry } from '@/services/models';
import type {
  Detection,
  GlobalTrajectory,
  SessionState,
  StageExecutionTarget,
  StageNumber,
  StageProgress,
  TimelineTrack,
  Tracklet,
  UserPreferences,
  VideoFile,
} from '@/types';

// Pipeline Store - Manages pipeline execution state

/** Canonical sidebar pipeline labels - shared by pipeline store reset + downstream flush. */
export const PIPELINE_STAGE_DEFAULTS: StageProgress[] = [
  { stage: 0, status: 'idle', progress: 0, message: 'Ingestion', completedAt: null, lastRunAt: null, staleSince: null },
  { stage: 1, status: 'idle', progress: 0, message: 'Detection & Tracking', completedAt: null, lastRunAt: null, staleSince: null },
  { stage: 2, status: 'idle', progress: 0, message: 'Feature Extraction', completedAt: null, lastRunAt: null, staleSince: null },
  { stage: 3, status: 'idle', progress: 0, message: 'Indexing', completedAt: null, lastRunAt: null, staleSince: null },
  { stage: 4, status: 'idle', progress: 0, message: 'Association', completedAt: null, lastRunAt: null, staleSince: null },
  { stage: 5, status: 'idle', progress: 0, message: 'Evaluation', completedAt: null, lastRunAt: null, staleSince: null },
  { stage: 6, status: 'idle', progress: 0, message: 'Visualization', completedAt: null, lastRunAt: null, staleSince: null },
];

export type PipelineModelMode = 'single' | 'fusion';

export type { StageExecutionTarget };

export interface PipelineFusionModel {
  modelId: string;
  weight: number;
}

export interface PipelineFusionConfig {
  models: PipelineFusionModel[];
  aqeK: number;
  rerank: boolean;
  k1: number;
  k2: number;
  lambda: number;
}

/** Source context for the active run, captured at ingestion (Stage 0) so every */
export interface RunInputContext {
  inputDir: string;
  cameras: string[];
  name: string;
  smoke: boolean;
}

/** Live per-stage telemetry parsed from the backend status during a run. */
export interface RunTelemetry {
  stageLabel?: string;
  completedStages?: number;
  totalStages?: number;
  camera?: string;
  camerasProcessed?: number;
  frame?: number;
  frameTotal?: number;
  /** Latest status message from the pipeline (e.g. "Detection - camera S01_c001"). */
  message?: string;
  /** Rolling tail of the pipeline subprocess output (verbose live log). */
  logTail?: string;
  /** Cameras discovered for the active stage. */
  camerasTotal?: number;
}

interface PipelineState {
  runId: string | null;
  galleryRunId: string | null;
  /** Source folder/cameras/smoke for the active run; set at Stage 0 ingestion. */
  runInput: RunInputContext | null;
  /** Pipeline stage number currently executing (0-6), or null when idle. Lets the */
  activeStage: number | null;
  /** Live telemetry for the in-flight stage run (camera, frame counts, etc.). */
  runTelemetry: RunTelemetry | null;
  /** Map camera id -> lat/lng for vehicle path map; from selected dataset's camera_coordinates.json */
  mapCameraCoordinates: Record<string, { lat: number; lng: number; label?: string }> | null;
  stages: StageProgress[];
  isRunning: boolean;
  currentStage: StageNumber;
  error: string | null;
  modelMode: PipelineModelMode;
  selectedModelId: string | null;
  selectedModelMeta: ModelEntry | null;
  fusion: PipelineFusionConfig | null;
  /** Incremented when upstream edits invalidate timeline/output; Stage 4 effect must not skip reload. */
  downstreamInvalidateGeneration: number;

  // Actions
  setRunId: (id: string | null) => void;
  setRunInput: (input: RunInputContext | null) => void;
  setActiveStage: (stage: number | null) => void;
  setRunTelemetry: (telemetry: RunTelemetry | null) => void;
  setGalleryRunId: (id: string | null) => void;
  setMapCameraCoordinates: (
    coords: Record<string, { lat: number; lng: number; label?: string }> | null
  ) => void;
  updateStageProgress: (stage: StageNumber, progress: Partial<StageProgress>) => void;
  setStageStatus: (stage: StageNumber, status: StageProgress['status'], message?: string) => void;
  setStageCancelled: (stage: StageNumber, message?: string) => void;
  setCurrentStage: (stage: StageNumber) => void;
  setIsRunning: (running: boolean) => void;
  setError: (error: string | null) => void;
  setModelMode: (mode: PipelineModelMode) => void;
  setSelectedModel: (id: string, meta: ModelEntry) => void;
  clearSelectedModel: () => void;
  setFusionConfig: (cfg: PipelineFusionConfig | null) => void;
  updateFusionWeights: (weights: PipelineFusionModel[]) => void;
  reset: () => void;
}

export const usePipelineStore = create<PipelineState>()(
  devtools(
    persist(
    (set) => ({
      runId: null,
      galleryRunId: null,
      runInput: null,
      activeStage: null,
      runTelemetry: null,
      mapCameraCoordinates: null,
      stages: PIPELINE_STAGE_DEFAULTS.map((s) => ({ ...s })),
      isRunning: false,
      currentStage: 0,
      error: null,
      modelMode: 'single',
      selectedModelId: null,
      selectedModelMeta: null,
      fusion: null,
      downstreamInvalidateGeneration: 0,

      setRunId: (id) => set({ runId: id }),

      setRunInput: (input) => set({ runInput: input }),

      setActiveStage: (stage) => set({ activeStage: stage }),

      setRunTelemetry: (telemetry) => set({ runTelemetry: telemetry }),

      setGalleryRunId: (id) => set({ galleryRunId: id }),

      setMapCameraCoordinates: (coords) => set({ mapCameraCoordinates: coords }),

      updateStageProgress: (stage, progress) =>
        set((state) => {
          const now = Date.now();
          const isCompleting = progress.status === 'completed';
          const isTerminal = progress.status === 'completed' || progress.status === 'error' || progress.status === 'cancelled';

          return {
            stages: state.stages.map((s) => {
              if (s.stage === stage) {
                const isStarting = progress.status === 'running' && s.status !== 'running';
                const next: StageProgress = { ...s, ...progress };

                if (isStarting) next.lastRunAt = now;
                if (isTerminal) {
                  next.completedAt = now;
                  next.lastRunAt = next.lastRunAt ?? now;
                  next.staleSince = null;
                }
                if (progress.status === 'running' || progress.status === 'idle') {
                  next.staleSince = null;
                }
                if (progress.status === 'error') {
                  next.error = progress.error ?? progress.message ?? s.error;
                }
                if (progress.status && progress.status !== 'error') {
                  next.error = progress.error;
                }

                return next;
              }

              if (isCompleting && s.stage > stage && s.completedAt !== null) {
                return { ...s, staleSince: now };
              }

              return s;
            }),
          };
        }),

      setStageStatus: (stage, status, message) =>
        set((state) => {
          const now = Date.now();
          const isCompleting = status === 'completed';
          const isTerminal = status === 'completed' || status === 'error' || status === 'cancelled';

          return {
            isRunning: state.isRunning && !isTerminal,
            stages: state.stages.map((s) => {
              if (s.stage === stage) {
                return {
                  ...s,
                  status,
                  progress: status === 'completed' ? 100 : status === 'running' ? s.progress : s.progress,
                  message: message ?? s.message,
                  completedAt: isTerminal ? now : s.completedAt,
                  lastRunAt: s.lastRunAt ?? now,
                  staleSince: status === 'running' || isTerminal ? null : s.staleSince,
                  error: status === 'error' ? message ?? s.error : undefined,
                };
              }

              if (isCompleting && s.stage > stage && s.completedAt !== null) {
                return { ...s, staleSince: now };
              }

              return s;
            }),
          };
        }),

      setStageCancelled: (stage, message) =>
        set((state) => {
          const now = Date.now();
          return {
            isRunning: false,
            stages: state.stages.map((s) =>
              s.stage === stage
                ? {
                    ...s,
                    status: 'cancelled',
                    message: message ?? `Stage ${stage} cancelled`,
                    completedAt: now,
                    lastRunAt: s.lastRunAt ?? now,
                    staleSince: null,
                    error: undefined,
                  }
                : s
            ),
          };
        }),

      setCurrentStage: (stage) => set({ currentStage: stage }),

      setIsRunning: (running) => set({ isRunning: running }),

      setError: (error) => set({ error }),

      setModelMode: (mode) =>
        set(
          mode === 'fusion'
            ? { modelMode: mode, selectedModelId: null, selectedModelMeta: null }
            : { modelMode: mode, fusion: null }
        ),

      setSelectedModel: (id, meta) => set({ selectedModelId: id, selectedModelMeta: meta }),

      clearSelectedModel: () => set({ selectedModelId: null, selectedModelMeta: null }),

      setFusionConfig: (cfg) => set({ fusion: cfg }),

      updateFusionWeights: (weights) =>
        set((state) => ({
          fusion: state.fusion ? { ...state.fusion, models: weights } : state.fusion,
        })),

      reset: () =>
        set({
          runId: null,
          galleryRunId: null,
          runInput: null,
          activeStage: null,
          runTelemetry: null,
          mapCameraCoordinates: null,
          stages: PIPELINE_STAGE_DEFAULTS.map((s) => ({ ...s })),
          isRunning: false,
          currentStage: 0,
          error: null,
          downstreamInvalidateGeneration: 0,
        }),
    }),
    {
      // Persist the run identity + per-stage status so a browser reload re-opens
      // the active run instead of dropping it. Only stable fields are saved;
      name: 'mtmc-pipeline-run',
      partialize: (s) => ({
        runId: s.runId,
        galleryRunId: s.galleryRunId,
        runInput: s.runInput,
        mapCameraCoordinates: s.mapCameraCoordinates,
        stages: s.stages,
        currentStage: s.currentStage,
        modelMode: s.modelMode,
        selectedModelId: s.selectedModelId,
        selectedModelMeta: s.selectedModelMeta,
        fusion: s.fusion,
      }),
      merge: (persisted, current) => {
        const p = (persisted ?? {}) as Partial<PipelineState>;
        // A stage left "running" when the tab was reloaded is no longer being
        // polled - downgrade it to idle so it isn't stuck spinning forever.
        const stages = (p.stages ?? current.stages).map((st) =>
          st.status === 'running'
            ? { ...st, status: 'idle' as const, progress: 0, message: 'Interrupted by reload - re-run if needed' }
            : st
        );
        return {
          ...current,
          ...p,
          stages,
          activeStage: null,
          isRunning: false,
          runTelemetry: null,
          error: null,
        };
      },
    }
    )
  )
);

interface StageExecutionState {
  stageExecutionTargets: Record<number, StageExecutionTarget>;
  setStageExecutionTarget: (stage: number, target: StageExecutionTarget) => void;
  getStageExecutionTarget: (stage: number) => StageExecutionTarget;
}

// Keep execution-target preferences in a second persisted store so pipeline run state,
// selected models, and fusion settings keep their existing transient/reset semantics.
export const useStageExecutionStore = create<StageExecutionState>()(
  persist(
    (set, get) => ({
      stageExecutionTargets: {},
      setStageExecutionTarget: (stage, target) =>
        set((state) => ({
          stageExecutionTargets: {
            ...state.stageExecutionTargets,
            [stage]: target,
          },
        })),
      getStageExecutionTarget: (stage) => get().stageExecutionTargets[stage] ?? 'local',
    }),
    {
      name: 'mtmc-stage-execution-targets',
      version: 1,
      partialize: (state) => ({ stageExecutionTargets: state.stageExecutionTargets }),
    }
  )
);

// Video Store - Manages uploaded videos and frames

interface VideoState {
  videos: VideoFile[];
  currentVideo: VideoFile | null;
  currentFrame: number;
  isPlaying: boolean;
  playbackSpeed: number;

  // Actions
  setVideos: (videos: VideoFile[]) => void;
  addVideo: (video: VideoFile) => void;
  removeVideo: (id: string) => void;
  setCurrentVideo: (video: VideoFile | null) => void;
  setCurrentFrame: (frame: number) => void;
  setIsPlaying: (playing: boolean) => void;
  setPlaybackSpeed: (speed: number) => void;
}

export const useVideoStore = create<VideoState>()(
  devtools(
    persist(
    (set) => ({
      videos: [],
      currentVideo: null,
      currentFrame: 0,
      isPlaying: false,
      playbackSpeed: 1,

      setVideos: (videos) => set({ videos }),

      addVideo: (video) =>
        set((state) => ({ videos: [...state.videos, video] })),

      removeVideo: (id) =>
        set((state) => ({
          videos: state.videos.filter((v) => v.id !== id),
          currentVideo:
            state.currentVideo?.id === id ? null : state.currentVideo,
        })),

      setCurrentVideo: (video) => set({ currentVideo: video, currentFrame: 0 }),

      setCurrentFrame: (frame) => set({ currentFrame: frame }),

      setIsPlaying: (playing) => set({ isPlaying: playing }),

      setPlaybackSpeed: (speed) => set({ playbackSpeed: speed }),
    }),
    {
      // Persist the loaded camera list + current selection so a reload re-opens
      // the run's footage. Playback position/speed are transient.
      name: 'mtmc-video',
      partialize: (s) => ({ videos: s.videos, currentVideo: s.currentVideo }),
    }
    )
  )
);

// Detection Store - Manages detections and selections

interface DetectionState {
  detections: Detection[];
  /** @deprecated Use selectedTrackIds for persistent selection */
  selectedIds: Set<string>;
  /** Track-level selection - persists across frame changes */
  selectedTrackIds: Set<number>;
  multiSelectMode: boolean;
  hoveredId: string | null;

  // Actions
  setDetections: (detections: Detection[]) => void;
  setDetectionsKeepSelection: (detections: Detection[]) => void;
  toggleSelection: (id: string) => void;
  /** Toggle selection by trackId (persistent across frames) */
  toggleTrackSelection: (trackId: number) => void;
  selectAll: () => void;
  /** Replace the selected-track set with an explicit list (e.g. select all tracks */
  selectTrackIds: (trackIds: number[]) => void;
  deselectAll: () => void;
  setMultiSelectMode: (enabled: boolean) => void;
  setHoveredId: (id: string | null) => void;
  getSelectedDetections: () => Detection[];
  reset: () => void;
}

export const useDetectionStore = create<DetectionState>()(
  devtools(
    persist(
    (set, get) => ({
      detections: [],
      selectedIds: new Set(),
      selectedTrackIds: new Set(),
      multiSelectMode: true,
      hoveredId: null,

      // Update detections WITHOUT clearing track selections
      setDetections: (detections) =>
        set({ detections }),

      setDetectionsKeepSelection: (detections) =>
        set({ detections }),

      // Legacy: toggle by detection.id (frame-specific)
      toggleSelection: (id) =>
        set((state) => {
          // Extract trackId from detection id format "det-{trackId}-{frameId}"
          const det = state.detections.find((d) => d.id === id);
          const trackId = det?.trackId;
          if (trackId === undefined || trackId === null) return state;

          const newTrackIds = new Set(state.selectedTrackIds);
          if (state.multiSelectMode) {
            if (newTrackIds.has(trackId)) {
              newTrackIds.delete(trackId);
            } else {
              newTrackIds.add(trackId);
            }
          } else {
            if (newTrackIds.has(trackId)) {
              newTrackIds.clear();
            } else {
              newTrackIds.clear();
              newTrackIds.add(trackId);
            }
          }
          return { selectedTrackIds: newTrackIds };
        }),

      // Toggle by trackId directly (persistent across frames)
      toggleTrackSelection: (trackId) =>
        set((state) => {
          const newSet = new Set(state.selectedTrackIds);
          if (state.multiSelectMode) {
            if (newSet.has(trackId)) {
              newSet.delete(trackId);
            } else {
              newSet.add(trackId);
            }
          } else {
            if (newSet.has(trackId)) {
              newSet.clear();
            } else {
              newSet.clear();
              newSet.add(trackId);
            }
          }
          return { selectedTrackIds: newSet };
        }),

      selectAll: () =>
        set((state) => ({
          selectedTrackIds: new Set(
            state.detections
              .map((d) => d.trackId)
              .filter((id): id is number => id != null)
          ),
        })),

      selectTrackIds: (trackIds) => set({ selectedTrackIds: new Set(trackIds) }),

      deselectAll: () =>
        set({ selectedIds: new Set(), selectedTrackIds: new Set() }),

      setMultiSelectMode: (enabled) => set({ multiSelectMode: enabled }),

      setHoveredId: (id) => set({ hoveredId: id }),

      getSelectedDetections: () => {
        const state = get();
        return state.detections.filter(
          (d) => d.trackId != null && state.selectedTrackIds.has(d.trackId)
        );
      },

      reset: () =>
        set({
          detections: [],
          selectedIds: new Set(),
          selectedTrackIds: new Set(),
          multiSelectMode: true,
          hoveredId: null,
        }),
    }),
    {
      // Persist only the user's tracking selection so it survives a page
      // refresh. Without this, Stage 4 loses the picked vehicle and falls
      name: 'detection-selection',
      partialize: (s) => ({
        selectedTrackIds: Array.from(s.selectedTrackIds),
        multiSelectMode: s.multiSelectMode,
      }),
      merge: (persisted, current) => {
        const p = (persisted ?? {}) as Partial<{ selectedTrackIds: number[]; multiSelectMode: boolean }>;
        return {
          ...current,
          multiSelectMode:
            typeof p.multiSelectMode === "boolean" ? p.multiSelectMode : current.multiSelectMode,
          selectedTrackIds: new Set<number>(Array.isArray(p.selectedTrackIds) ? p.selectedTrackIds : []),
        };
      },
    }
    ),
    { name: 'detection-store' }
  )
);

// Tracklet Store - Manages tracklets and trajectories

interface TrackletState {
  tracklets: Tracklet[];
  trajectories: GlobalTrajectory[];
  selectedTrackletIds: Set<number>;
  selectedTrajectoryId: number | null;

  // Actions
  setTracklets: (tracklets: Tracklet[]) => void;
  setTrajectories: (trajectories: GlobalTrajectory[]) => void;
  toggleTrackletSelection: (id: number) => void;
  selectTrajectory: (id: number | null) => void;
  clearSelections: () => void;
  reset: () => void;
}

export const useTrackletStore = create<TrackletState>()(
  devtools(
    (set) => ({
      tracklets: [],
      trajectories: [],
      selectedTrackletIds: new Set(),
      selectedTrajectoryId: null,

      setTracklets: (tracklets) => set({ tracklets }),

      setTrajectories: (trajectories) => set({ trajectories }),

      toggleTrackletSelection: (id) =>
        set((state) => {
          const newSet = new Set(state.selectedTrackletIds);
          if (newSet.has(id)) {
            newSet.delete(id);
          } else {
            newSet.add(id);
          }
          return { selectedTrackletIds: newSet };
        }),

      selectTrajectory: (id) => set({ selectedTrajectoryId: id }),

      clearSelections: () =>
        set({ selectedTrackletIds: new Set(), selectedTrajectoryId: null }),

      reset: () =>
        set({
          tracklets: [],
          trajectories: [],
          selectedTrackletIds: new Set(),
          selectedTrajectoryId: null,
        }),
    }),
    { name: 'tracklet-store' }
  )
);

// Timeline Store - Manages timeline view state

interface TimelineState {
  tracks: TimelineTrack[];
  zoom: number;
  scrollPosition: number;
  selectedTrackId: string | null;
  confirmedTracks: Set<string>;
  /** Once true (user confirmed/unconfirmed any clip), output filters to confirmed clips only. */
  timelineClipFilterEngaged: boolean;
  /** The timeline-load key (video/run/selection) the current `tracks` were matched for. */
  tracksContextKey: string | null;

  // Actions
  setTracks: (tracks: TimelineTrack[]) => void;
  /** Record which load-context the current tracks belong to (for refresh restore). */
  setTracksContextKey: (key: string | null) => void;
  /** Full reset after upstream stage edits (video, selection, re-inference). */
  resetAfterUpstreamEdit: () => void;
  /** Replace rows from Stage-4 loaders; keep user confirmations for the same row ids. */
  applyTracksReplaceKeepingMeta: (tracks: TimelineTrack[]) => void;
  /** Replace rows and rebuild confirmed Sets from each row's `confirmed` flag (e.g. refinement). */
  replaceTracksSyncingRowFlags: (tracks: TimelineTrack[]) => void;
  addTrack: (track: TimelineTrack) => void;
  removeTrack: (id: string) => void;
  reorderTracks: (fromIndex: number, toIndex: number) => void;
  selectTrack: (id: string | null) => void;
  confirmTrack: (id: string) => void;
  unconfirmTrack: (id: string) => void;
  setZoom: (zoom: number) => void;
  setScrollPosition: (position: number) => void;
  updateTrack: (id: string, updates: Partial<TimelineTrack>) => void;
}

export const useTimelineStore = create<TimelineState>()(
  devtools(
    persist(
    (set) => ({
      tracks: [],
      zoom: 1,
      scrollPosition: 0,
      selectedTrackId: null,
      confirmedTracks: new Set(),
      timelineClipFilterEngaged: false,
      tracksContextKey: null,

      setTracks: (tracks) =>
        set({ tracks, timelineClipFilterEngaged: false, confirmedTracks: new Set(), tracksContextKey: null }),

      setTracksContextKey: (key) => set({ tracksContextKey: key }),

      resetAfterUpstreamEdit: () =>
        set({
          tracks: [],
          confirmedTracks: new Set(),
          timelineClipFilterEngaged: false,
          selectedTrackId: null,
          tracksContextKey: null,
        }),

      applyTracksReplaceKeepingMeta: (tracks) =>
        set((state) => {
          const prevConfirmed = state.confirmedTracks;
          const engaged = state.timelineClipFilterEngaged;
          const merged = tracks.map((t) => ({
            ...t,
            confirmed: prevConfirmed.has(t.id),
          }));
          const nextConfirmed = new Set(
            merged.filter((t) => t.confirmed).map((t) => t.id)
          );
          return {
            tracks: merged,
            confirmedTracks: nextConfirmed,
            timelineClipFilterEngaged: engaged || nextConfirmed.size > 0,
          };
        }),

      replaceTracksSyncingRowFlags: (tracks) =>
        set(() => {
          const nextConfirmed = new Set(
            tracks.filter((t) => t.confirmed).map((t) => t.id)
          );
          return {
            tracks,
            confirmedTracks: nextConfirmed,
            timelineClipFilterEngaged: nextConfirmed.size > 0,
          };
        }),

      addTrack: (track) =>
        set((state) => ({ tracks: [...state.tracks, track] })),

      removeTrack: (id) =>
        set((state) => ({
          tracks: state.tracks.filter((t) => t.id !== id),
          confirmedTracks: new Set(
            Array.from(state.confirmedTracks).filter((tid) => tid !== id)
          ),
        })),

      reorderTracks: (fromIndex, toIndex) =>
        set((state) => {
          const newTracks = [...state.tracks];
          const [removed] = newTracks.splice(fromIndex, 1);
          newTracks.splice(toIndex, 0, removed);
          return { tracks: newTracks };
        }),

      selectTrack: (id) => set({ selectedTrackId: id }),

      confirmTrack: (id) =>
        set((state) => {
          const newSet = new Set(state.confirmedTracks);
          newSet.add(id);
          return {
            timelineClipFilterEngaged: true,
            confirmedTracks: newSet,
            tracks: state.tracks.map((t) =>
              t.id === id ? { ...t, confirmed: true } : t
            ),
          };
        }),

      unconfirmTrack: (id) =>
        set((state) => {
          const newSet = new Set(state.confirmedTracks);
          newSet.delete(id);
          return {
            timelineClipFilterEngaged: true,
            confirmedTracks: newSet,
            tracks: state.tracks.map((t) =>
              t.id === id ? { ...t, confirmed: false } : t
            ),
          };
        }),

      setZoom: (zoom) => set({ zoom }),

      setScrollPosition: (position) => set({ scrollPosition: position }),

      updateTrack: (id, updates) =>
        set((state) => ({
          tracks: state.tracks.map((t) =>
            t.id === id ? { ...t, ...updates } : t
          ),
        })),
    }),
    {
      // Persist the matched timeline result so refreshing Stage 4 restores it instead of
      // re-running the cross-camera association query. Upstream edits (selection change,
      // -v2: bumped to discard older caches that wrongly stored the single-camera "no
      // association yet" fallback, which blocked the auto-run-association-on-load path.
      name: 'mtmc-timeline-v2',
      partialize: (s) => ({
        tracks: s.tracks,
        tracksContextKey: s.tracksContextKey,
        confirmedTracks: Array.from(s.confirmedTracks),
        timelineClipFilterEngaged: s.timelineClipFilterEngaged,
        selectedTrackId: s.selectedTrackId,
      }),
      merge: (persisted, current) => {
        const p = (persisted ?? {}) as Partial<{
          tracks: TimelineTrack[];
          tracksContextKey: string | null;
          confirmedTracks: string[];
          timelineClipFilterEngaged: boolean;
          selectedTrackId: string | null;
        }>;
        return {
          ...current,
          tracks: Array.isArray(p.tracks) ? p.tracks : current.tracks,
          tracksContextKey: typeof p.tracksContextKey === "string" ? p.tracksContextKey : null,
          confirmedTracks: new Set<string>(Array.isArray(p.confirmedTracks) ? p.confirmedTracks : []),
          timelineClipFilterEngaged: Boolean(p.timelineClipFilterEngaged),
          selectedTrackId: typeof p.selectedTrackId === "string" ? p.selectedTrackId : null,
        };
      },
    }
    ),
    { name: 'timeline-store' }
  )
);

// Manual Stage Store - per-run completion of stages that run no pipeline
// (Selection / Refinement). Persisted by runId so loading a run restores their

interface ManualStageState {
  completedByRun: Record<string, number[]>;
  markManualStageDone: (runId: string, stage: number) => void;
  clearManualStage: (runId: string, stage: number) => void;
  /** Drop all markers for a run (e.g. when it's deleted, so a reused id starts clean). */
  clearRun: (runId: string) => void;
  getManualStagesDone: (runId: string) => number[];
}

export const useManualStageStore = create<ManualStageState>()(
  persist(
    (set, get) => ({
      completedByRun: {},
      markManualStageDone: (runId, stage) =>
        set((s) => {
          if (!runId) return s;
          const cur = s.completedByRun[runId] ?? [];
          if (cur.includes(stage)) return s;
          return { completedByRun: { ...s.completedByRun, [runId]: [...cur, stage].sort((a, b) => a - b) } };
        }),
      clearManualStage: (runId, stage) =>
        set((s) => {
          const cur = s.completedByRun[runId];
          if (!cur || !cur.includes(stage)) return s;
          return { completedByRun: { ...s.completedByRun, [runId]: cur.filter((x) => x !== stage) } };
        }),
      clearRun: (runId) =>
        set((s) => {
          if (!(runId in s.completedByRun)) return s;
          const next = { ...s.completedByRun };
          delete next[runId];
          return { completedByRun: next };
        }),
      getManualStagesDone: (runId) => (runId ? get().completedByRun[runId] ?? [] : []),
    }),
    { name: 'mtmc-manual-stages', version: 1 }
  )
);

// Session Store - Manages user session and preferences

interface SessionStore extends SessionState {
  preferences: UserPreferences;

  // Actions
  setCurrentStage: (stage: StageNumber) => void;
  setDemoMode: (enabled: boolean) => void;
  setSelectedVideo: (video: VideoFile | undefined) => void;
  addSelectedDetection: (id: string) => void;
  removeSelectedDetection: (id: string) => void;
  clearSelectedDetections: () => void;
  setLocationFilter: (filter: Partial<SessionState['locationFilter']>) => void;
  setDateTimeRange: (range: Partial<SessionState['dateTimeRange']>) => void;
  addConfirmedClip: (clip: TimelineTrack) => void;
  removeConfirmedClip: (id: string) => void;
  addRefinementFrame: (frameId: string) => void;
  removeRefinementFrame: (frameId: string) => void;
  clearRefinementFrames: () => void;
  clearConfirmedClips: () => void;
  updatePreferences: (prefs: Partial<UserPreferences>) => void;
  resetSession: () => void;
}

const defaultPreferences: UserPreferences = {
  theme: 'dark',
  gridSize: 3,
  maxSplits: 4,
  playbackSpeed: 1,
  showConfidence: true,
  showTrajectoryPaths: true,
  autoAdvance: true,
};

const initialSession: SessionState = {
  currentStage: 0,
  isDemoMode: false,
  selectedVideo: undefined,
  selectedDetections: [],
  selectedTracklets: [],
  confirmedClips: [],
  locationFilter: {},
  dateTimeRange: {},
  refinementFrames: [],
};

export const useSessionStore = create<SessionStore>()(
  devtools(
    persist(
      (set) => ({
        ...initialSession,
        preferences: defaultPreferences,

        setCurrentStage: (stage) => set({ currentStage: stage }),

        setDemoMode: (enabled) => set({ isDemoMode: enabled }),

        setSelectedVideo: (video) => set({ selectedVideo: video }),

        addSelectedDetection: (id) =>
          set((state) => ({
            selectedDetections: [...state.selectedDetections, id],
          })),

        removeSelectedDetection: (id) =>
          set((state) => ({
            selectedDetections: state.selectedDetections.filter((d) => d !== id),
          })),

        clearSelectedDetections: () => set({ selectedDetections: [] }),

        setLocationFilter: (filter) =>
          set((state) => ({
            locationFilter: { ...state.locationFilter, ...filter },
          })),

        setDateTimeRange: (range) =>
          set((state) => ({
            dateTimeRange: { ...state.dateTimeRange, ...range },
          })),

        addConfirmedClip: (clip) =>
          set((state) => ({
            confirmedClips: [...state.confirmedClips, clip],
          })),

        removeConfirmedClip: (id) =>
          set((state) => ({
            confirmedClips: state.confirmedClips.filter((c) => c.id !== id),
          })),

        addRefinementFrame: (frameId) =>
          set((state) => {
            if (state.refinementFrames.length >= 16) return state;
            if (state.refinementFrames.includes(frameId)) return state;
            return { refinementFrames: [...state.refinementFrames, frameId] };
          }),

        removeRefinementFrame: (frameId) =>
          set((state) => ({
            refinementFrames: state.refinementFrames.filter((f) => f !== frameId),
          })),

        clearRefinementFrames: () => set({ refinementFrames: [] }),

        clearConfirmedClips: () => set({ confirmedClips: [] }),

        updatePreferences: (prefs) =>
          set((state) => ({
            preferences: { ...state.preferences, ...prefs },
          })),

        resetSession: () => set({ ...initialSession }),
      }),
      {
        name: 'mtmc-session',
        partialize: (state) => ({
          preferences: state.preferences,
          locationFilter: state.locationFilter,
          // Persist the active stage so a refresh keeps you where you were
          // (e.g. a loaded run on Detection stays on Detection, instead of
          currentStage: state.currentStage,
        }),
      }
    ),
    { name: 'session-store' }
  )
);

// UI Store - Manages UI state

interface UIState {
  sidebarOpen: boolean;
  sidebarWidth: number;
  showSettings: boolean;
  showHelp: boolean;
  activeModal: string | null;
  notifications: Array<{
    id: string;
    type: 'info' | 'success' | 'warning' | 'error';
    message: string;
    timestamp: number;
  }>;

  // Actions
  toggleSidebar: () => void;
  setSidebarWidth: (width: number) => void;
  setShowSettings: (show: boolean) => void;
  setShowHelp: (show: boolean) => void;
  setActiveModal: (modal: string | null) => void;
  addNotification: (
    type: 'info' | 'success' | 'warning' | 'error',
    message: string
  ) => void;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;
}

export const useUIStore = create<UIState>()(
  devtools(
    (set) => ({
      sidebarOpen: true,
      sidebarWidth: 320,
      showSettings: false,
      showHelp: false,
      activeModal: null,
      notifications: [],

      toggleSidebar: () =>
        set((state) => ({ sidebarOpen: !state.sidebarOpen })),

      setSidebarWidth: (width) => set({ sidebarWidth: width }),

      setShowSettings: (show) => set({ showSettings: show }),

      setShowHelp: (show) => set({ showHelp: show }),

      setActiveModal: (modal) => set({ activeModal: modal }),

      addNotification: (type, message) =>
        set((state) => ({
          notifications: [
            ...state.notifications,
            {
              id: Math.random().toString(36).substr(2, 9),
              type,
              message,
              timestamp: Date.now(),
            },
          ],
        })),

      removeNotification: (id) =>
        set((state) => ({
          notifications: state.notifications.filter((n) => n.id !== id),
        })),

      clearNotifications: () => set({ notifications: [] }),
    }),
    { name: 'ui-store' }
  )
);
