import type {
  ApiResponse,
  Detection,
  EvaluationResult,
  FrameInfo,
  GlobalTrajectory,
  PipelineRunStatus,
  SearchResult,
  StageNumber,
  Tracklet,
  VideoFile,
  WatchlistHit,
} from '@/types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

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
  };
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

    return {
      id: String(d.id ?? `det-${idx}`),
      bbox: bboxObj,
      confidence: Number(d.confidence ?? 0),
      classId: Number(d.classId ?? -1),
      className: String(d.className ?? 'vehicle'),
      frameId: Number(d.frameId ?? 0),
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
}

async function fetchApi<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE}${endpoint}`;

  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new ApiError(
      errorData.message || `HTTP ${response.status}`,
      response.status,
      errorData
    );
  }

  return response.json();
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
        reject(new ApiError('Upload failed', xhr.status));
      }
    };

    xhr.onerror = () => reject(new ApiError('Network error', 0));

    xhr.open('POST', `${API_BASE}/videos/upload`);
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

export async function runStage(
  stage: StageNumber,
  config?: Record<string, unknown>
): Promise<ApiResponse<PipelineRunStatus>> {
  return fetchApi(`/pipeline/run-stage/${stage}`, {
    method: 'POST',
    body: JSON.stringify(config || {}),
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
// Stage 2-3: Features & Indexing
// ============================================================================

export async function extractFeatures(
  trackletIds: number[],
  cameraId: string
): Promise<ApiResponse<void>> {
  return fetchApi('/features/extract', {
    method: 'POST',
    body: JSON.stringify({ trackletIds, cameraId }),
  });
}

export async function buildIndex(
  runId: string
): Promise<ApiResponse<void>> {
  return fetchApi(`/index/build/${runId}`, { method: 'POST' });
}

// ============================================================================
// Stage 4: Association & Search
// ============================================================================

export async function getTracklets(
  cameraId?: string,
  videoId?: string
): Promise<ApiResponse<Tracklet[]>> {
  const query = new URLSearchParams();
  if (cameraId) query.set('cameraId', cameraId);
  if (videoId) query.set('videoId', videoId);
  const params = query.toString() ? `?${query.toString()}` : '';
  return fetchApi(`/tracklets${params}`);
}

export async function getTrajectories(
  runId: string
): Promise<ApiResponse<GlobalTrajectory[]>> {
  return fetchApi(`/trajectories/${runId}`);
}

export async function searchByImage(
  imageData: string, // base64
  topK: number = 20,
  minSimilarity: number = 0.3
): Promise<ApiResponse<SearchResult[]>> {
  return fetchApi('/search/image', {
    method: 'POST',
    body: JSON.stringify({ imageData, topK, minSimilarity }),
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

export async function scanWatchlist(
  watchlist: { subjectId: string; embedding: number[] }[],
  threshold: number = 0.55
): Promise<ApiResponse<WatchlistHit[]>> {
  return fetchApi('/watchlist/scan', {
    method: 'POST',
    body: JSON.stringify({ watchlist, threshold }),
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
        reject(new ApiError('Kaggle import failed', xhr.status));
      }
    };

    xhr.onerror = () => reject(new ApiError('Network error', 0));

    xhr.open('POST', `${API_BASE}/runs/import-kaggle`);
    xhr.send(formData);
  });
}

// ============================================================================
// Corrections & Refinement
// ============================================================================

export async function mergeTracklets(
  trackletA: { trackletId: number; cameraId: string },
  trackletB: { trackletId: number; cameraId: string }
): Promise<ApiResponse<GlobalTrajectory>> {
  return fetchApi('/corrections/merge', {
    method: 'POST',
    body: JSON.stringify({ trackletA, trackletB }),
  });
}

export async function splitTrajectory(
  globalId: number,
  atTrackletId: number
): Promise<ApiResponse<GlobalTrajectory[]>> {
  return fetchApi('/corrections/split', {
    method: 'POST',
    body: JSON.stringify({ globalId, atTrackletId }),
  });
}

export async function confirmTracklet(
  trackletId: number,
  cameraId: string,
  globalId: number
): Promise<ApiResponse<void>> {
  return fetchApi('/corrections/confirm', {
    method: 'POST',
    body: JSON.stringify({ trackletId, cameraId, globalId }),
  });
}

// ============================================================================
// Location Data (Egypt Hierarchy)
// ============================================================================

export async function getGovernorates(): Promise<
  ApiResponse<{ id: string; name: string; nameAr: string }[]>
> {
  return fetchApi('/locations/governorates');
}

export async function getCities(
  governorateId: string
): Promise<ApiResponse<{ id: string; name: string; nameAr: string }[]>> {
  return fetchApi(`/locations/cities/${governorateId}`);
}

export async function getZones(
  cityId: string
): Promise<ApiResponse<{ id: string; name: string; nameAr: string }[]>> {
  return fetchApi(`/locations/zones/${cityId}`);
}

export async function getCameras(
  zoneId?: string
): Promise<ApiResponse<{ id: string; name: string; location: unknown }[]>> {
  const params = zoneId ? `?zoneId=${zoneId}` : '';
  return fetchApi(`/cameras${params}`);
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
  const wsUrl = `${API_BASE.replace('http', 'ws')}/ws/pipeline/${runId}`;
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
    console.log('WebSocket closed');
    onClose?.();
  };

  return ws;
}

// ============================================================================
// Utilities
// ============================================================================

export function getFrameUrl(videoId: string, frameId: number): string {
  return `${API_BASE}/frames/${videoId}/${frameId}`;
}

export function getThumbnailUrl(
  cameraId: string,
  trackletId: number,
  frameId?: number
): string {
  const params = frameId !== undefined ? `?frameId=${frameId}` : '';
  return `${API_BASE}/thumbnails/${cameraId}/${trackletId}${params}`;
}

export function getVideoStreamUrl(videoId: string): string {
  return `${API_BASE}/videos/stream/${videoId}`;
}

// ============================================================================
// Dataset endpoints
// ============================================================================

export interface DatasetFolder {
  name: string;
  path: string;
  cameras: { id: string; hasVideo: boolean }[];
  cameraCount: number;
  videosFound: number;
  alreadyProcessed: boolean;
  isProcessing: boolean;
  runId: string | null;
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

export { ApiError };
