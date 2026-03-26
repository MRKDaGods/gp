import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import type {
  Detection,
  GlobalTrajectory,
  SessionState,
  StageNumber,
  StageProgress,
  TimelineTrack,
  Tracklet,
  UserPreferences,
  VideoFile,
} from '@/types';

// ============================================================================
// Pipeline Store - Manages pipeline execution state
// ============================================================================

interface PipelineState {
  runId: string | null;
  galleryRunId: string | null;
  stages: StageProgress[];
  isRunning: boolean;
  currentStage: StageNumber;
  error: string | null;

  // Actions
  setRunId: (id: string | null) => void;
  setGalleryRunId: (id: string | null) => void;
  updateStageProgress: (stage: StageNumber, progress: Partial<StageProgress>) => void;
  setCurrentStage: (stage: StageNumber) => void;
  setIsRunning: (running: boolean) => void;
  setError: (error: string | null) => void;
  reset: () => void;
}

const initialStages: StageProgress[] = [
  { stage: 0, status: 'idle', progress: 0, message: 'Ingestion' },
  { stage: 1, status: 'idle', progress: 0, message: 'Detection & Tracking' },
  { stage: 2, status: 'idle', progress: 0, message: 'Feature Extraction' },
  { stage: 3, status: 'idle', progress: 0, message: 'Indexing' },
  { stage: 4, status: 'idle', progress: 0, message: 'Association' },
  { stage: 5, status: 'idle', progress: 0, message: 'Evaluation' },
  { stage: 6, status: 'idle', progress: 0, message: 'Visualization' },
];

export const usePipelineStore = create<PipelineState>()(
  devtools(
    (set) => ({
      runId: null,
      galleryRunId: null,
      stages: [...initialStages],
      isRunning: false,
      currentStage: 0,
      error: null,

      setRunId: (id) => set({ runId: id }),

      setGalleryRunId: (id) => set({ galleryRunId: id }),

      updateStageProgress: (stage, progress) =>
        set((state) => ({
          stages: state.stages.map((s) =>
            s.stage === stage ? { ...s, ...progress } : s
          ),
        })),

      setCurrentStage: (stage) => set({ currentStage: stage }),

      setIsRunning: (running) => set({ isRunning: running }),

      setError: (error) => set({ error }),

      reset: () =>
        set({
          runId: null,
          galleryRunId: null,
          stages: [...initialStages],
          isRunning: false,
          currentStage: 0,
          error: null,
        }),
    }),
    { name: 'pipeline-store' }
  )
);

// ============================================================================
// Video Store - Manages uploaded videos and frames
// ============================================================================

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
    { name: 'video-store' }
  )
);

// ============================================================================
// Detection Store - Manages detections and selections
// ============================================================================

interface DetectionState {
  detections: Detection[];
  selectedIds: Set<string>;
  multiSelectMode: boolean;
  hoveredId: string | null;

  // Actions
  setDetections: (detections: Detection[]) => void;
  setDetectionsKeepSelection: (detections: Detection[]) => void;
  toggleSelection: (id: string) => void;
  selectAll: () => void;
  deselectAll: () => void;
  setMultiSelectMode: (enabled: boolean) => void;
  setHoveredId: (id: string | null) => void;
  getSelectedDetections: () => Detection[];
}

export const useDetectionStore = create<DetectionState>()(
  devtools(
    (set, get) => ({
      detections: [],
      selectedIds: new Set(),
      multiSelectMode: false,
      hoveredId: null,

      setDetections: (detections) =>
        set({ detections, selectedIds: new Set() }),

      setDetectionsKeepSelection: (detections) =>
        set((state) => {
          const validIds = new Set(detections.map((d) => d.id));
          const kept = new Set([...state.selectedIds].filter((id) => validIds.has(id)));
          return { detections, selectedIds: kept };
        }),

      toggleSelection: (id) =>
        set((state) => {
          const newSet = new Set(state.selectedIds);
          if (state.multiSelectMode) {
            if (newSet.has(id)) {
              newSet.delete(id);
            } else {
              newSet.add(id);
            }
          } else {
            if (newSet.has(id)) {
              newSet.clear();
            } else {
              newSet.clear();
              newSet.add(id);
            }
          }
          return { selectedIds: newSet };
        }),

      selectAll: () =>
        set((state) => ({
          selectedIds: new Set(state.detections.map((d) => d.id)),
        })),

      deselectAll: () => set({ selectedIds: new Set() }),

      setMultiSelectMode: (enabled) => set({ multiSelectMode: enabled }),

      setHoveredId: (id) => set({ hoveredId: id }),

      getSelectedDetections: () => {
        const state = get();
        return state.detections.filter((d) => state.selectedIds.has(d.id));
      },
    }),
    { name: 'detection-store' }
  )
);

// ============================================================================
// Tracklet Store - Manages tracklets and trajectories
// ============================================================================

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
    }),
    { name: 'tracklet-store' }
  )
);

// ============================================================================
// Timeline Store - Manages timeline view state
// ============================================================================

interface TimelineState {
  tracks: TimelineTrack[];
  zoom: number;
  scrollPosition: number;
  selectedTrackId: string | null;
  confirmedTracks: Set<string>;

  // Actions
  setTracks: (tracks: TimelineTrack[]) => void;
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
    (set) => ({
      tracks: [],
      zoom: 1,
      scrollPosition: 0,
      selectedTrackId: null,
      confirmedTracks: new Set(),

      setTracks: (tracks) => set({ tracks }),

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
    { name: 'timeline-store' }
  )
);

// ============================================================================
// Session Store - Manages user session and preferences
// ============================================================================

interface SessionStore extends SessionState {
  preferences: UserPreferences;

  // Actions
  setCurrentStage: (stage: StageNumber) => void;
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
        }),
      }
    ),
    { name: 'session-store' }
  )
);

// ============================================================================
// UI Store - Manages UI state
// ============================================================================

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
