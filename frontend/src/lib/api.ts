import type {
  ApiResponse,
  Detection,
  EvaluationResult,
  FrameInfo,
  GlobalTrajectory,
  PipelineRunStatus,
  SearchResult,
  SingleStageRunStatus,
  StageNumber,
  TrackletSummary,
  VideoFile,
} from '@/types';

export function apiBase(): string {
  return process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';
}

export function apiUrl(path: string): string {
  const suffix = path.startsWith('/') ? path : `/${path}`;
  return `${apiBase()}${suffix}`;
}

export const API_BASE = apiBase();

function extractApiErrorMessage(data: unknown, fallback: string): string {
  if (data && typeof data === 'object') {
    const record = data as { detail?: unknown; message?: unknown; error?: unknown };
    const detail = record.detail;
    if (typeof detail === 'string' && detail.trim()) return detail;
    if (Array.isArray(detail) && detail.length > 0) return JSON.stringify(detail);
    if (typeof record.message === 'string' && record.message.trim()) return record.message;
    if (typeof record.error === 'string' && record.error.trim()) return record.error;
  }
  return fallback;
}

function normalizeVideoFile(raw: any): VideoFile {
  return {
    id: String(raw.id),
    name: String(raw.name ?? raw.filename ?? raw.path ?? 'video'),
    path: String(raw.path ?? ''),
    size: Number(raw.size ?? 0),
    duration: Number(raw.duration ?? 0),
    fps: Number(raw.fps ?? 0),
    width: Number(raw.width ?? 0),
    height: Number(raw.height ?? 0),
    thumbnail: typeof raw.thumbnail === 'string' ? raw.thumbnail : undefined,
    uploadedAt: String(raw.uploadedAt ?? new Date().toISOString()),
    cameraId:
      raw.cameraId != null && raw.cameraId !== '' ? String(raw.cameraId) : undefined,
    latestRunId:
      raw.latestRunId != null && raw.latestRunId !== ''
        ? String(raw.latestRunId)
        : undefined,
  };
}

/** Map API confidence to 0–1 for UI (handles 0–100 percent from some sources). */
function normalizeConfidence(raw: unknown): number {
  let c = Number(raw ?? 0);
  if (!Number.isFinite(c)) return 0;
  if (c > 1.5) c = c / 100;
  return Math.min(Math.max(c, 0), 1);
}

function normalizeDetections(rawList: any[]): Detection[] {
  return rawList.map((d, idx) => {
    const bboxArr = Array.isArray(d.bbox) ? d.bbox : null;
    const bboxObj = bboxArr && bboxArr.length === 4
      ? { x1: Number(bboxArr[0]), y1: Number(bboxArr[1]), x2: Number(bboxArr[2]), y2: Number(bboxArr[3]) }
      : {
          x1: Number(d.bbox?.x1 ?? d.bbox?.x ?? 0),
          y1: Number(d.bbox?.y1 ?? d.bbox?.y ?? 0),
          x2: Number(d.bbox?.x2 ?? (d.bbox?.x ?? 0) + (d.bbox?.width ?? 0)),
          y2: Number(d.bbox?.y2 ?? (d.bbox?.y ?? 0) + (d.bbox?.height ?? 0)),
        };

    const confRaw = d.confidence ?? d.score ?? d.detectionConfidence;

    return {
      id: String(d.id ?? `det-${idx}`),
      bbox: bboxObj,
      confidence: normalizeConfidence(confRaw),
      classId: Number(d.classId ?? -1),
      className: String(d.className ?? 'vehicle'),
      frameId: Number(d.frameId ?? 0),
      trackId: Number(d.trackId ?? d.track_id ?? -1),  
      selected: Boolean(d.selected),
    };
  });
}

class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public data?: unknown
  ) {
    super(message);
    this.name = 'ApiError';
  }

  static async fromResponse(response: Response): Promise<ApiError> {
    const contentType = response.headers.get('content-type') ?? '';
    const data = contentType.includes('application/json')
      ? await response.json().catch(() => undefined)
      : await response.text().catch(() => undefined);
    const message = extractApiErrorMessage(data, response.statusText || `HTTP ${response.status}`);
    return new ApiError(message, response.status, data);
  }

  static fromXhr(xhr: XMLHttpRequest, fallback: string): ApiError {
    const raw = xhr.responseText;
    let data: unknown = undefined;
    if (raw) {
      try {
        data = JSON.parse(raw);
      } catch {
        data = raw;
      }
    }
    return new ApiError(extractApiErrorMessage(data, fallback), xhr.status, data);
  }
}

export interface ReIDImageInputPayload {
  id: string;
  image_base64: string;
  metadata?: Record<string, unknown>;
}

export interface ReIDRankedMatch {
  galleryId: string;
  rank: number;
  score: number;
  distance?: number | null;
  metadata?: Record<string, unknown>;
}

export interface ReIDQueryResult {
  queryId: string;
  matches: ReIDRankedMatch[];
  latencyMs: number;
}

export interface SingleCamReIDRequestPayload {
  modelId: string;
  queries: ReIDImageInputPayload[];
  gallery: ReIDImageInputPayload[];
  topK?: number;
  rerank?: boolean;
  aqeK?: number;
}

export interface SingleCamReIDResponsePayload {
  success: boolean;
  modelId: string;
  device: string;
  featureDim: number;
  queryCount: number;
  galleryCount: number;
  results: ReIDQueryResult[];
  latencyMs: number;
}

export interface FusionReIDModelPayload {
  modelId: string;
  weight: number;
}

export interface FusionReIDComponentPayload {
  modelId: string;
  weight: number;
  featureDim: number;
  results: ReIDQueryResult[];
}

export interface FusionReIDRequestPayload {
  models: FusionReIDModelPayload[];
  queries: ReIDImageInputPayload[];
  gallery: ReIDImageInputPayload[];
  topK?: number;
  rerank?: boolean;
  aqeK?: number;
}

export interface FusionReIDResponsePayload {
  success: boolean;
  modelIds: string[];
  weights: number[];
  device: string;
  queryCount: number;
  galleryCount: number;
  results: ReIDQueryResult[];
  components: FusionReIDComponentPayload[];
  warnings: string[];
  latencyMs: number;
}

export type EvalType = "veri776_transreid" | "veri776_clipsenet" | "cityflow_transreid" | "veri776_14t_fusion";

export interface EvalRunRequestPayload {
  evalType: EvalType;
  configOverrides?: Record<string, unknown>;
}

export interface EvalJobResponsePayload {
  jobId: string;
  status: string;
}

export interface EvalJobStatusPayload {
  jobId: string;
  status: "queued" | "running" | "completed" | "failed" | string;
  createdAt: string;
  startedAt?: string | null;
  finishedAt?: string | null;
  error?: string | null;
  progress: Record<string, unknown>;
}

export interface EvalJobResultPayload {
  jobId: string;
  status: string;
  result: {
    summary?: Record<string, unknown>;
    result?: unknown;
    [key: string]: unknown;
  } | null;
}

async function fetchApi<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = apiUrl(endpoint);

  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });

  if (!response.ok) {
    throw await ApiError.fromResponse(response);
  }

  return response.json();
}

export async function singleCamReid(payload: SingleCamReIDRequestPayload): Promise<SingleCamReIDResponsePayload> {
  return fetchApi<SingleCamReIDResponsePayload>('/v1/reid/single_cam', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function fusionReid(payload: FusionReIDRequestPayload): Promise<FusionReIDResponsePayload> {
  return fetchApi<FusionReIDResponsePayload>('/v1/reid/fusion', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function submitEval(payload: EvalRunRequestPayload): Promise<EvalJobResponsePayload> {
  return fetchApi<EvalJobResponsePayload>('/v1/eval/run', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function getEvalStatus(jobId: string): Promise<EvalJobStatusPayload> {
  return fetchApi<EvalJobStatusPayload>(`/v1/eval/${encodeURIComponent(jobId)}/status`);
}

export async function getEvalResult(jobId: string): Promise<EvalJobResultPayload> {
  return fetchApi<EvalJobResultPayload>(`/v1/eval/${encodeURIComponent(jobId)}/result`);
}

// ============================================================================
// Video Management
// ============================================================================

export async function uploadVideo(
  file: File,
  onProgress?: (progress: number) => void
): Promise<ApiResponse<VideoFile>> {
  const formData = new FormData();
  formData.append('video', file);

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) {
        const progress = (event.loaded / event.total) * 100;
        onProgress(progress);
      }
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        const parsed = JSON.parse(xhr.responseText);
        resolve({
          ...parsed,
          data: parsed?.data ? normalizeVideoFile(parsed.data) : undefined,
        });
      } else {
        reject(ApiError.fromXhr(xhr, 'Upload failed'));
      }
    };

    xhr.onerror = () => reject(new ApiError('Network error', 0));

    xhr.open('POST', apiUrl('/videos/upload'));
    xhr.send(formData);
  });
}

export async function getVideos(): Promise<ApiResponse<VideoFile[]>> {
  const response = await fetchApi<ApiResponse<any[]>>('/videos');
  return {
    ...response,
    data: Array.isArray(response.data)
      ? response.data.map((v) => normalizeVideoFile(v))
      : [],
  };
}

export async function getVideo(id: string): Promise<ApiResponse<VideoFile>> {
  const response = await fetchApi<ApiResponse<any>>(`/videos/${id}`);
  return {
    ...response,
    data: response.data ? normalizeVideoFile(response.data) : undefined,
  };
}

export async function deleteVideo(id: string): Promise<ApiResponse<void>> {
  return fetchApi(`/videos/${id}`, { method: 'DELETE' });
}

// ============================================================================
// Pipeline Stage Execution
// ============================================================================

export interface FusionModelRequest {
  model_id: string;
  weight: number;
}

export interface FusionConfigRequest {
  models: FusionModelRequest[];
  aqe_k: number;
  k1: number;
  k2: number;
  lambda: number;
  rerank: boolean;
}

export interface KaggleRequestConfig {
  target: 'local' | 'kaggle';
  username?: string;
  key?: string;
  dataset_slug?: string;
}

export interface KaggleJobStatus {
  run_id: string;
  kernel_slug: string;
  kernel_url: string;
  dataset_slug: string;
  project_dataset_slug: string;
  status: 'queued' | 'running' | 'complete' | 'error' | 'cancelled' | 'unknown';
  stages: number[];
  started_at: string;
  last_polled_at: string | null;
  exit_code: number | null;
  outputs_downloaded_to: string | null;
  error?: string | null;
}

export interface RunStageRequest {
  runId?: string;
  videoId?: string;
  cameraId?: string;
  dataset?: string;
  model_id?: string | null;
  fusion?: FusionConfigRequest | null;
  smokeTest?: boolean;
  useCpu?: boolean;
  config?: Record<string, unknown>;
  kaggle?: KaggleRequestConfig | null;
}

export async function runStage(
  stage: StageNumber,
  request?: RunStageRequest
): Promise<ApiResponse<SingleStageRunStatus>> {
  return fetchApi(`/pipeline/run-stage/${stage}`, {
    method: 'POST',
    body: JSON.stringify(request || {}),
  });
}

export async function runFullPipeline(
  config?: Record<string, unknown>
): Promise<ApiResponse<PipelineRunStatus>> {
  return fetchApi('/pipeline/run', {
    method: 'POST',
    body: JSON.stringify(config || {}),
  });
}

export async function getPipelineStatus(
  runId: string
): Promise<ApiResponse<PipelineRunStatus>> {
  return fetchApi(`/pipeline/status/${runId}`);
}

export async function cancelPipeline(
  runId: string
): Promise<ApiResponse<void>> {
  return fetchApi(`/pipeline/cancel/${runId}`, { method: 'POST' });
}

export async function getKaggleStatus(runId: string): Promise<ApiResponse<KaggleJobStatus>> {
  return fetchApi(`/pipeline/kaggle-status/${encodeURIComponent(runId)}`);
}

export async function cancelKaggleKernel(runId: string): Promise<ApiResponse<KaggleJobStatus>> {
  return fetchApi(`/pipeline/kaggle-cancel/${encodeURIComponent(runId)}`, {
    method: 'POST',
  });
}

// ============================================================================
// Stage 1: Detection & Tracking
// ============================================================================

export async function getDetections(
  videoId: string,
  frameId?: number
): Promise<ApiResponse<Detection[]>> {
  const params = frameId !== undefined ? `?frameId=${frameId}` : '';
  const response = await fetchApi<ApiResponse<any[]>>(`/detections/${videoId}${params}`);
  return {
    ...response,
    data: Array.isArray(response.data) ? normalizeDetections(response.data) : [],
  };
}

/**
 * Fetch ALL detections for every frame at once. Returns a Map keyed by frame number.
 */
export async function getAllDetections(
  videoId: string
): Promise<Map<number, Detection[]>> {
  const response = await fetchApi<ApiResponse<Record<string, any[]>>>(`/detections/${videoId}/all`);
  const map = new Map<number, Detection[]>();
  if (response.data && typeof response.data === 'object') {
    for (const [frameKey, rawDets] of Object.entries(response.data)) {
      if (Array.isArray(rawDets)) {
        map.set(Number(frameKey), normalizeDetections(rawDets));
      }
    }
  }
  return map;
}

export async function getFrameWithDetections(
  videoId: string,
  frameId: number
): Promise<ApiResponse<{ frame: FrameInfo; detections: Detection[] }>> {
  const response = await fetchApi<ApiResponse<{ frame: any; detections: any[] }>>(
    `/frames/${videoId}/${frameId}/detections`
  );

  if (!response.data) {
    return response as ApiResponse<{ frame: FrameInfo; detections: Detection[] }>;
  }

  const frameRaw = response.data.frame ?? {};
  const normalizedFrame: FrameInfo = {
    frameId: Number(frameRaw.frameId ?? frameRaw.id ?? frameId),
    cameraId: String(frameRaw.cameraId ?? frameRaw.videoId ?? videoId),
    timestamp: Number(frameRaw.timestamp ?? 0),
    framePath: String(frameRaw.framePath ?? ''),
    width: Number(frameRaw.width ?? 0),
    height: Number(frameRaw.height ?? 0),
  };

  return {
    ...response,
    data: {
      frame: normalizedFrame,
      detections: normalizeDetections(response.data.detections ?? []),
    },
  };
}

// ============================================================================
// Stage 4: Association & Search
// ============================================================================

export async function getTracklets(
  cameraId?: string,
  videoId?: string,
  opts?: { allCameras?: boolean }
): Promise<ApiResponse<TrackletSummary[]>> {
  const query = new URLSearchParams();
  if (cameraId) query.set('cameraId', cameraId);
  if (videoId) query.set('videoId', videoId);
  if (opts?.allCameras) query.set('allCameras', 'true');
  const params = query.toString() ? `?${query.toString()}` : '';
  return fetchApi(`/tracklets${params}`);
}

export async function getMatchedSummary(runId: string): Promise<any> {
  return fetchApi(`/runs/${runId}/matched_summary`);
}

export interface MatchedAlternative {
  rank: number;
  globalId: number | null;
  cameraId: string;
  trackId: number;
  score: number;
  confidence: number;
  numCameras: number;
  className?: string;
  startTime?: number;
  endTime?: number;
  representativeFrame?: number;
  representativeBbox?: number[];
  label?: string;
  clipPath: string;
  previewUrl?: string;
  ok: boolean;
  message?: string;
}

export interface MatchedAlternativesPayload {
  runId: string;
  totalCameras: number;
  cameras: string[];
  subfolder: string;
  alternatives: MatchedAlternative[];
}

export async function getMatchedAlternatives(
  runId: string,
  options?: {
    topK?: number;
    anchorCameraId?: string;
    anchorTrackId?: number;
    excludeGlobalId?: number;
    excludeCameraId?: string;
    excludeTrackId?: number;
  }
): Promise<MatchedAlternativesPayload> {
  const q = new URLSearchParams();
  if (options?.topK != null) q.set("topK", String(options.topK));
  if (options?.anchorCameraId) q.set("anchorCameraId", String(options.anchorCameraId));
  if (options?.anchorTrackId != null) q.set("anchorTrackId", String(options.anchorTrackId));
  if (options?.excludeGlobalId != null) q.set("excludeGlobalId", String(options.excludeGlobalId));
  if (options?.excludeCameraId) q.set("excludeCameraId", String(options.excludeCameraId));
  if (options?.excludeTrackId != null) q.set("excludeTrackId", String(options.excludeTrackId));

  const raw = await fetchApi<any>(
    `/runs/${encodeURIComponent(runId)}/matched_alternatives${q.toString() ? `?${q.toString()}` : ""}`
  );

  const alternatives = Array.isArray(raw?.alternatives)
    ? raw.alternatives.map((item: any): MatchedAlternative => ({
        rank: Number(item?.rank ?? 0),
        globalId: item?.global_id == null ? null : Number(item.global_id),
        cameraId: String(item?.camera_id ?? "unknown"),
        trackId: Number(item?.track_id ?? -1),
        score: Number(item?.score ?? 0),
        confidence: Number(item?.confidence ?? 0),
        numCameras: Number(item?.num_cameras ?? 0),
        className: typeof item?.class_name === "string" ? item.class_name : undefined,
        startTime: item?.start_time_s == null ? undefined : Number(item.start_time_s),
        endTime: item?.end_time_s == null ? undefined : Number(item.end_time_s),
        representativeFrame:
          item?.representative_frame == null ? undefined : Number(item.representative_frame),
        representativeBbox:
          Array.isArray(item?.representative_bbox) && item.representative_bbox.length === 4
            ? item.representative_bbox.map((v: any) => Number(v))
            : undefined,
        label: typeof item?.label === "string" ? item.label : undefined,
        clipPath: String(item?.clip_path ?? item?.file ?? ""),
        ok: Boolean(item?.ok),
        message: typeof item?.msg === "string" ? item.msg : undefined,
      }))
    : [];

  return {
    runId: String(raw?.runId ?? runId),
    totalCameras: Number(raw?.totalCameras ?? 0),
    cameras: Array.isArray(raw?.cameras) ? raw.cameras.map((c: any) => String(c)) : [],
    subfolder: String(raw?.subfolder ?? "top5_alternatives"),
    alternatives,
  };
}

export function getMatchedAlternativeClipUrl(runId: string, clipPath: string): string {
  const safePath = String(clipPath)
    .split("/")
    .filter(Boolean)
    .map((part) => encodeURIComponent(part))
    .join("/");
  return apiUrl(`/runs/${encodeURIComponent(runId)}/matched_alternatives/${safePath}`);
}

/** Sampled frames for timeline tracklet preview (full frame + bbox sync). */
export interface TrackletSequenceFrame {
  frameId: number;
  bbox: number[];
  timeRel: number;
  timestamp: number | null;
}

export interface TrackletSequencePayload {
  width: number;
  height: number;
  cameraId: string;
  trackId: number;
  frames: TrackletSequenceFrame[];
}

export async function getTrackletSequence(
  runId: string,
  cameraId: string,
  trackId: number,
  maxFrames = 64
): Promise<TrackletSequencePayload> {
  const q = new URLSearchParams({
    cameraId,
    trackId: String(trackId),
    max_frames: String(maxFrames),
  });
  return fetchApi<TrackletSequencePayload>(
    `/runs/${encodeURIComponent(runId)}/tracklet_sequence?${q.toString()}`
  );
}

export function getRunFullFrameUrl(
  runId: string,
  cameraId: string,
  frameId: number
): string {
  const q = new URLSearchParams({
    cameraId,
    frameId: String(frameId),
  });
  return apiUrl(`/runs/${encodeURIComponent(runId)}/full_frame?${q.toString()}`);
}

export async function getTrajectories(
  runId: string
): Promise<ApiResponse<GlobalTrajectory[]>> {
  return fetchApi(`/trajectories/${runId}`);
}

export async function queryTimeline(
  probeRunId: string,
  videoId: string,
  selectedTrackIds: string[],
  opts?: {
    galleryRunId?: string | null;
    skipExports?: boolean;
  }
): Promise<ApiResponse<{
  stage4Available: boolean;
  mode: string;
  message: string;
  trajectories: GlobalTrajectory[];
  selectedTracklets: any[];
  diagnostics: {
    selectedCount: number;
    selectedKeyCount: number;
    trajectoryCount: number;
    matchedTrajectoryCount: number;
  };
}>> {
  const body: Record<string, unknown> = {
    runId: probeRunId,
    videoId,
    selectedTrackIds,
    skipExports: opts?.skipExports ?? false,
  };
  const g = opts?.galleryRunId;
  if (g) body.galleryRunId = g;
  return fetchApi('/timeline/query', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

export async function searchByTracklet(
  trackletId: number,
  cameraId: string,
  topK: number = 20
): Promise<ApiResponse<SearchResult[]>> {
  return fetchApi('/search/tracklet', {
    method: 'POST',
    body: JSON.stringify({ trackletId, cameraId, topK }),
  });
}

export async function searchTracklet(options: {
  trackletId: number;
  probeVideoId: string;
  galleryRunId: string;
  topK?: number;
}): Promise<ApiResponse<{ rank: number; score: number; cameraId: string; trackletId: number; globalId: number | null; runId: string }[]>> {
  return fetchApi('/search/tracklet', {
    method: 'POST',
    body: JSON.stringify({
      trackletId: options.trackletId,
      probeVideoId: options.probeVideoId,
      galleryRunId: options.galleryRunId,
      topK: options.topK ?? 20,
    }),
  });
}

// ============================================================================
// Stage 5: Evaluation
// ============================================================================

export async function getEvaluationResults(
  runId: string
): Promise<ApiResponse<EvaluationResult>> {
  return fetchApi(`/evaluation/${runId}`);
}

// ============================================================================
// Stage 6: Visualization & Export
// ============================================================================

export async function generateSummaryVideo(
  runId: string,
  config?: {
    /** Only stitch these clips (timeline camera/track); keys match matched/summary.json */
    includeClips?: Array<{ camera_id: string; track_id: number }>;
    globalIds?: number[];
    speedup?: number;
    dedupe?: boolean;
  }
): Promise<ApiResponse<{ videoUrl: string }>> {
  return fetchApi(`/visualization/summary/${runId}`, {
    method: 'POST',
    body: JSON.stringify(config || {}),
  });
}

export async function exportTrajectories(
  runId: string,
  format: 'json' | 'csv' | 'mot'
): Promise<ApiResponse<{ downloadUrl: string }>> {
  return fetchApi(`/export/${runId}?format=${format}`);
}

export async function importKaggleRunArtifacts(
  zipFile: File,
  options?: {
    runId?: string;
    videoId?: string;
    cameraId?: string;
  },
  onProgress?: (progress: number) => void
): Promise<ApiResponse<PipelineRunStatus>> {
  const formData = new FormData();
  formData.append('artifactsZip', zipFile);
  if (options?.runId) formData.append('runId', options.runId);
  if (options?.videoId) formData.append('videoId', options.videoId);
  if (options?.cameraId) formData.append('cameraId', options.cameraId);

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) {
        const progress = (event.loaded / event.total) * 100;
        onProgress(progress);
      }
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        reject(ApiError.fromXhr(xhr, 'Kaggle import failed'));
      }
    };

    xhr.onerror = () => reject(new ApiError('Network error', 0));

    xhr.open('POST', apiUrl('/runs/import-kaggle'));
    xhr.send(formData);
  });
}

// ============================================================================
// WebSocket Connection
// ============================================================================

export function createWebSocket(
  runId: string,
  onMessage: (data: unknown) => void,
  onError?: (error: Event) => void,
  onClose?: () => void
): WebSocket {
  const wsUrl = `${apiBase().replace('http', 'ws')}/ws/pipeline/${runId}`;
  const ws = new WebSocket(wsUrl);

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch {
      console.error('Failed to parse WebSocket message:', event.data);
    }
  };

  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
    onError?.(error);
  };

  ws.onclose = () => {
    onClose?.();
  };

  return ws;
}

// ============================================================================
// Utilities
// ============================================================================

export function getFrameUrl(videoId: string, frameId: number): string {
  return apiUrl(`/frames/${videoId}/${frameId}`);
}

export function getVideoStreamUrl(videoId: string): string {
  return apiUrl(`/videos/stream/${videoId}`);
}

// ============================================================================
// Dataset endpoints
// ============================================================================

/** Loaded from dataset/<name>/camera_coordinates.json via GET /datasets */
export interface CameraMapCoordinateEntry {
  lat: number;
  lng: number;
  label?: string;
}

export interface DatasetFolder {
  name: string;
  path: string;
  cameras: { id: string; hasVideo: boolean }[];
  cameraCount: number;
  videosFound: number;
  alreadyProcessed: boolean;
  hasGallery: boolean;
  isProcessing: boolean;
  runId: string | null;
  galleryRunId: string | null;
  /** Present when camera_coordinates.json exists with at least one valid camera. */
  cameraCoordinates?: Record<string, CameraMapCoordinateEntry> | null;
}

export async function getDatasets(): Promise<ApiResponse<DatasetFolder[]>> {
  return fetchApi<ApiResponse<DatasetFolder[]>>('/datasets');
}

export async function processDataset(
  folder: string
): Promise<ApiResponse<any>> {
  return fetchApi<ApiResponse<any>>(`/datasets/${encodeURIComponent(folder)}/process`, {
    method: 'POST',
  });
}

/** A camera discovered inside a dataset/folder input dir. */
export interface DatasetCamera {
  id: string;
  hasVideo: boolean;
  file?: string;
}

/** A selectable tracking dataset, read from configs/datasets/*.yaml. */
export interface AvailableDataset {
  name: string;
  configFile: string;
  inputDir: string;
  taskType?: string | null;
  layout: 'per_camera' | 'flat' | 'empty' | 'missing';
  available: boolean;
  cameraCount: number;
  videosFound: number;
  /** Native fps of the source video (probed). */
  sourceFps?: number | null;
  /** Rate Stage 0 samples frames at (from the dataset config). */
  sampleFps?: number | null;
  width?: number | null;
  height?: number | null;
  cameras: DatasetCamera[];
}

/** One entry (subfolder or video file) in the folder browser. */
export interface BrowseEntry {
  name: string;
  type: 'dir' | 'video';
  path: string;
}

export interface BrowseResult {
  root: string;
  path: string;
  parent: string | null;
  entries: BrowseEntry[];
  datasetLike: boolean;
  layout: AvailableDataset['layout'];
  cameras: DatasetCamera[];
  inputDir: string;
}

/** Curated tracking datasets (CityFlow, WILDTRACK, EPFL, ...) with availability. */
export async function getAvailableDatasets(): Promise<ApiResponse<AvailableDataset[]>> {
  return fetchApi<ApiResponse<AvailableDataset[]>>('/datasets/available');
}

/** Sandboxed folder browser under data/raw/ for ad-hoc dataset selection. */
export async function browseDatasetFolder(path = ''): Promise<ApiResponse<BrowseResult>> {
  const q = path ? `?path=${encodeURIComponent(path)}` : '';
  return fetchApi<ApiResponse<BrowseResult>>(`/datasets/browse${q}`);
}

/** Camera videos inside a chosen dataset/folder, shaped for the gallery. */
export async function getDatasetVideos(inputDir: string): Promise<ApiResponse<VideoFile[]>> {
  const res = await fetchApi<ApiResponse<any[]>>(
    `/datasets/videos?inputDir=${encodeURIComponent(inputDir)}`
  );
  return { ...res, data: (res.data ?? []).map(normalizeVideoFile) };
}

/** Start a pipeline run against a chosen input folder (dataset or custom). */
export async function runDatasetInput(payload: {
  inputDir: string;
  name?: string;
  stages?: string;
  smoke?: boolean;
  /** Optional subset of camera ids to process (multi-camera selection). */
  cameras?: string[];
  /** Reuse an existing run so a single pipeline stage runs incrementally
   *  against the same run dir. Omit to allocate a fresh run. */
  runId?: string | null;
}): Promise<ApiResponse<any>> {
  return fetchApi<ApiResponse<any>>('/datasets/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export interface RunStageMap {
  stage0: boolean;
  stage1: boolean;
  stage2: boolean;
  stage3: boolean;
  stage4: boolean;
  stage5: boolean;
  stage6: boolean;
}

export type RunStageState = "idle" | "running" | "done" | "error";

/** Per-stage status that merges disk artifacts with the in-flight run's live
 *  stage — so a running/failed stage shows even before it writes output. */
export type RunStageStatusMap = Record<keyof RunStageMap, RunStageState>;

export interface RunVideoRecord {
  id: string;
  cameraId: string;
  path: string;
  name: string;
}

export interface RunSummary {
  runId: string;
  root?: string;
  name?: string | null;
  source?: string | null;
  inputDir?: string | null;
  cameras: string[];
  smoke: boolean;
  videos: RunVideoRecord[];
  createdAt?: string | null;
  updatedAt?: string | null;
  stages: RunStageMap;
  /** Live per-stage status (merges disk presence with the running stage). */
  stageStatus?: RunStageStatusMap | null;
  /** Pipeline stage number currently executing (when running). */
  activeStage?: number | null;
  /** Human label of the stage currently executing (e.g. "Detection & Tracking"). */
  currentStageName?: string | null;
  /** Latest status/progress message from the running pipeline. */
  message?: string | null;
  /** Error summary when the run failed. */
  error?: string | null;
  trajectoryCount?: number | null;
  status?: string | null;
  progress?: number | null;
  sizeBytes?: number | null;
}

/** List all runs on disk (newest first). */
export async function getRuns(): Promise<ApiResponse<RunSummary[]>> {
  return fetchApi<ApiResponse<RunSummary[]>>('/runs');
}

/** Full detail for one run (for re-opening it). */
export async function getRunDetail(runId: string): Promise<ApiResponse<RunSummary>> {
  return fetchApi<ApiResponse<RunSummary>>(`/runs/${encodeURIComponent(runId)}`);
}

/** Delete a run — removes its directory from disk and clears server state. */
export async function deleteRun(runId: string): Promise<ApiResponse<{ runId: string; removed: boolean }>> {
  return fetchApi<ApiResponse<{ runId: string; removed: boolean }>>(`/runs/${encodeURIComponent(runId)}`, {
    method: 'DELETE',
  });
}

export async function saveDatasetCameraCoordinates(
  folder: string,
  coordinates: Record<string, CameraMapCoordinateEntry>
): Promise<ApiResponse<Record<string, CameraMapCoordinateEntry>>> {
  return fetchApi<ApiResponse<Record<string, CameraMapCoordinateEntry>>>(
    `/datasets/${encodeURIComponent(folder)}/camera-coordinates`,
    {
      method: 'PUT',
      body: JSON.stringify(coordinates),
    }
  );
}

export { ApiError };
