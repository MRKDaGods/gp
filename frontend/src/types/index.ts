// ============================================================================
// MTMC Tracker Types
// Mirrors Python data_models.py for frontend type safety
// ============================================================================

export type StageNumber = 0 | 1 | 2 | 3 | 4 | 5 | 6;

export type StageStatus = 'idle' | 'running' | 'completed' | 'error';

export interface StageProgress {
  stage: StageNumber;
  status: StageStatus;
  progress: number; // 0-100
  message: string;
  startedAt?: string;
  completedAt?: string;
  error?: string;
}

// Detection and tracking
export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface Detection {
  id: string;
  bbox: BoundingBox;
  confidence: number;
  classId: number;
  className: string;
  frameId: number;
  selected?: boolean;
}

export interface TrackletFrame {
  frameId: number;
  timestamp: number;
  bbox: BoundingBox;
  confidence: number;
}

export interface Tracklet {
  trackId: number;
  cameraId: string;
  classId: number;
  className: string;
  frames: TrackletFrame[];
  startTime: number;
  endTime: number;
  duration: number;
  numFrames: number;
  meanConfidence: number;
  thumbnail?: string; // base64 or URL
}

// Global trajectory (cross-camera)
export interface GlobalTrajectory {
  globalId: number;
  tracklets: Tracklet[];
  confidence: number;
  evidence: EvidenceRecord[];
  timeline: TimelineEntry[];
  cameraSequence: string[];
  timeSpan: [number, number];
  totalDuration: number;
  className: string;
  numCameras: number;
  isCrossCamera: boolean;
}

export interface EvidenceRecord {
  trackletA: string;
  trackletB: string;
  similarity: number;
  mergeStage: 'graph' | 'gallery_expansion' | 'orphan_pair';
}

export interface TimelineEntry {
  cameraId: string;
  start: number;
  end: number;
}

// Video and frames
export interface VideoFile {
  id: string;
  name: string;
  path: string;
  size: number;
  duration: number;
  fps: number;
  width: number;
  height: number;
  thumbnail?: string;
  uploadedAt: string;
}

export interface FrameInfo {
  frameId: number;
  cameraId: string;
  timestamp: number;
  framePath: string;
  width: number;
  height: number;
}

// Location hierarchy (Egypt-specific)
export interface LocationNode {
  id: string;
  name: string;
  nameAr?: string;
  children?: LocationNode[];
}

export interface CameraInfo {
  id: string;
  name: string;
  location: {
    governorate: string;
    city: string;
    zone: string;
    coordinates?: [number, number]; // [lat, lng]
  };
  status: 'online' | 'offline' | 'processing';
}

// Search and queries
export interface SearchResult {
  rank: number;
  trackletId: number;
  cameraId: string;
  startTime: number;
  endTime: number;
  similarity: number;
  globalId?: number;
  trajectoryConfidence: number;
  thumbnail?: string;
}

export interface WatchlistHit {
  subjectId: string;
  globalId: number;
  similarity: number;
  trajectoryConfidence: number;
  camerasSeen: string[];
  firstSeen: number;
  lastSeen: number;
  alertLevel: 'HIGH' | 'MEDIUM' | 'LOW';
}

// Evaluation metrics
export interface EvaluationResult {
  mota: number;
  idf1: number;
  mtmcIdf1: number;
  hota: number;
  idSwitches: number;
  mostlyTracked: number;
  mostlyLost: number;
  numGtIds: number;
  numPredIds: number;
  details: Record<string, unknown>;
}

// API responses
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface PipelineRunStatus {
  runId: string;
  runDir: string;
  stages: StageProgress[];
  startedAt: string;
  status: 'running' | 'completed' | 'error';
}

// Websocket messages
export interface WsMessage {
  type: 'progress' | 'detection' | 'tracklet' | 'trajectory' | 'error' | 'completed';
  stage?: StageNumber;
  data: unknown;
  timestamp: string;
}

// Timeline view types

/** One camera-segment within a global trajectory row */
export interface TrajectorySegment {
  cameraId: string;
  trackId: number;
  globalId?: number;
  start: number;
  end: number;
  color: string;
  representativeFrame?: number;
  representativeBbox?: number[];
}

export interface TimelineTrack {
  id: string;
  cameraId: string;          // primary camera (first seen for multi-cam, or only camera)
  trackletId: number;
  globalId?: number;
  startTime: number;         // earliest start across all segments
  endTime: number;           // latest end across all segments
  thumbnail?: string;
  selected: boolean;
  confirmed: boolean;
  alternatives?: AlternativeMatch[];
  representativeFrame?: number;
  representativeBbox?: number[];
  sampleFrames?: { frameId: number; bbox: number[] }[];
  // Notebook-style multi-camera segments (one per camera in the trajectory)
  segments?: TrajectorySegment[];
  // Display label e.g. "G-0001 | 3 cams | car"
  label?: string;
  confidence?: number;
  className?: string;
}

export interface AlternativeMatch {
  trackletId: number;
  cameraId: string;
  similarity: number;
  thumbnail?: string;
}

// Grid view types
export interface GridCell {
  id: string;
  cameraId: string;
  timestamp: number;
  framePath: string;
  detections?: Detection[];
  tracklets?: Tracklet[];
}

// Map types (future)
export interface VehiclePath {
  globalId: number;
  points: PathPoint[];
  className: string;
  color: string;
}

export interface PathPoint {
  lat: number;
  lng: number;
  timestamp: number;
  cameraId: string;
  confidence: number;
}

export interface HeatmapConfig {
  enabled: boolean;
  radius: number;
  intensity: number;
  timeRange: [number, number];
}

export interface SpatiotemporalConstraints {
  maxSpeed: number; // km/h
  searchRadius: number; // meters
  timeWindow: number; // seconds
}

// User preferences
export interface UserPreferences {
  theme: 'dark' | 'light' | 'system';
  gridSize: number; // 1-5 for grid view
  maxSplits: number; // 1-16 for split screen
  playbackSpeed: number;
  showConfidence: boolean;
  showTrajectoryPaths: boolean;
  autoAdvance: boolean;
}

// Session state
export interface SessionState {
  currentStage: StageNumber;
  selectedVideo?: VideoFile;
  selectedDetections: string[];
  selectedTracklets: number[];
  confirmedClips: TimelineTrack[];
  locationFilter: {
    governorate?: string;
    city?: string;
    zone?: string;
  };
  dateTimeRange: {
    start?: Date;
    end?: Date;
  };
  refinementFrames: string[];
}
