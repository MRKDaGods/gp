"use client";

import {
  memo,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
  type Ref,
  type SyntheticEvent,
} from "react";
import {
  Loader2,
  AlertCircle,
  Square,
  Clock,
  Cpu,
  Server,
  Cloud,
  ScanLine,
  Video,
  Terminal,
} from "lucide-react";
import { cn, bboxToStyle } from "@/lib/utils";
import {
  classIconFor,
  classLabelFor,
  domainIcon,
  domainNoun,
  resolveDomain,
  trackedTitle,
} from "@/lib/class-meta";
import { typicalFrameGap, valuesForFrame } from "@/lib/frame-lookup";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { DisclosurePanel, ErrorBanner, PlaybackControls, toStageStatus } from "@/components/pipeline";
import type { StageStatus } from "@/components/pipeline/status/types";
import {
  useVideoStore,
  useDetectionStore,
  usePipelineStore,
  useSessionStore,
  useStageExecutionStore,
} from "@/store";
import { useDatasetStore } from "@/lib/store";
import { flushPipelineFromStage } from "@/lib/pipeline-flush";
import { getRunStageErrorMessage, inferCameraId, pollStageStatus } from "@/lib/pipeline-run";
import { useRunPipelineStage } from "@/hooks/use-pipeline-stage";
import { MultiCameraView } from "@/components/stages/multi-camera-view";
import {
  cancelPipeline,
  getAllDetections,
  getFrameUrl,
  getVideoStreamUrl,
  runStage,
  apiUrl,
} from "@/lib/api";
import type { BoundingBox, Detection, VideoFile } from "@/types";
import { DoubleBufferedFrameImg } from "@/components/ui/double-buffered-img";

function detectionCropUrl(
  videoId: string,
  frameId: number,
  bbox: BoundingBox,
  quality: number = 92,
  minEdge: number = 160
): string {
  const { x1, y1, x2, y2 } = bbox;
  return apiUrl(`/crops/${encodeURIComponent(videoId)}?frameId=${frameId}&x1=${x1}&y1=${y1}&x2=${x2}&y2=${y2}&quality=${quality}&minEdge=${minEdge}&pad=0.12`);
}

/** Frames per camera a quick-test run ingests/processes - mirrors the pipeline's */
const SMOKE_FRAME_LIMIT = 10;

/** A whole tracked vehicle across the run (aggregated from per-frame detections). */
interface TrackSummary {
  trackId: number;
  className: string;
  classId: number;
  firstFrame: number;
  lastFrame: number;
  frameCount: number;
  thumbFrameId: number;
  thumbBbox: BoundingBox;
  confidence: number;
}

/** Aggregate per-frame detections into one entry per track id. */
function summariseTracks(
  byFrame: Map<number, Detection[]>
): TrackSummary[] {
  const map = new Map<number, TrackSummary>();
  for (const [frameId, dets] of byFrame) {
    for (const d of dets) {
      if (d.trackId == null) continue;
      const cur = map.get(d.trackId);
      if (!cur) {
        map.set(d.trackId, {
          trackId: d.trackId,
          className: d.className,
          classId: d.classId,
          firstFrame: frameId,
          lastFrame: frameId,
          frameCount: 1,
          thumbFrameId: frameId,
          thumbBbox: d.bbox,
          confidence: d.confidence,
        });
      } else {
        if (frameId < cur.firstFrame) {
          cur.firstFrame = frameId;
          cur.thumbFrameId = frameId;
          cur.thumbBbox = d.bbox;
        }
        cur.lastFrame = Math.max(cur.lastFrame, frameId);
        cur.frameCount += 1;
        cur.confidence = Math.max(cur.confidence, d.confidence);
      }
    }
  }
  return [...map.values()].sort(
    (a, b) => a.firstFrame - b.firstFrame || a.trackId - b.trackId
  );
}

/** First frame each track appears - used for sidebar thumbs only. */
function buildTrackThumbnailSources(
  frameMap: Map<number, Detection[]>
): Map<number, { frameId: number; bbox: BoundingBox }> {
  const out = new Map<number, { frameId: number; bbox: BoundingBox }>();
  const frames = [...frameMap.keys()].sort((a, b) => a - b);
  for (const fi of frames) {
    for (const d of frameMap.get(fi) ?? []) {
      const tid = d.trackId;
      if (!Number.isFinite(tid) || tid < 0) continue;
      if (out.has(tid)) continue;
      out.set(tid, {
        frameId: fi,
        bbox: { x1: d.bbox.x1, y1: d.bbox.y1, x2: d.bbox.x2, y2: d.bbox.y2 },
      });
    }
  }
  return out;
}

function detectionCropThumbPropsEqual(
  prev: {
    videoId: string;
    classId: number;
    isSelected: boolean;
    cropFrameId: number;
    cropBbox: BoundingBox;
  },
  next: typeof prev
) {
  return (
    prev.videoId === next.videoId &&
    prev.classId === next.classId &&
    prev.isSelected === next.isSelected &&
    prev.cropFrameId === next.cropFrameId &&
    prev.cropBbox.x1 === next.cropBbox.x1 &&
    prev.cropBbox.y1 === next.cropBbox.y1 &&
    prev.cropBbox.x2 === next.cropBbox.x2 &&
    prev.cropBbox.y2 === next.cropBbox.y2
  );
}

/** Small class glyph for captions/filters - person gets a pedestrian icon, not a car. */
function ClassGlyph({ classId, className }: { classId: number; className?: string }) {
  const Icon = classIconFor(classId);
  return <Icon className={className} />;
}

/** Pill toggle for the Tracked-Vehicles class filter. */
function FilterChip({
  active,
  onClick,
  label,
  count,
  icon,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
  count: number;
  icon?: ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium transition-colors",
        active
          ? "border-accent-strong bg-accent-strong/15 text-foreground"
          : "border-border/60 bg-background/40 text-muted-foreground hover:text-foreground"
      )}
      aria-pressed={active}
    >
      {icon}
      {label}
      <span className="opacity-60">{count}</span>
    </button>
  );
}

function ClassIconFallback({
  classId,
  isSelected,
}: {
  classId: number;
  isSelected: boolean;
}) {
  const cls = cn("h-6 w-6", isSelected ? "text-success" : "text-muted-foreground");
  const Icon = classIconFor(classId);
  return <Icon className={cls} />;
}

/** Sidebar crop thumbnails: stable crop URL per track + load only when in view. */
const DetectionCropThumb = memo(function DetectionCropThumb({
  videoId,
  classId,
  isSelected,
  cropFrameId,
  cropBbox,
}: {
  videoId: string;
  classId: number;
  isSelected: boolean;
  cropFrameId: number;
  cropBbox: BoundingBox;
}) {
  const rootRef = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const el = rootRef.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setVisible(true);
          obs.disconnect();
        }
      },
      { root: null, rootMargin: "120px", threshold: 0.01 }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  const url = useMemo(
    () => detectionCropUrl(videoId, cropFrameId, cropBbox, 92, 160),
    [videoId, cropFrameId, cropBbox]
  );

  return (
    <div
      ref={rootRef}
      className="relative h-full w-full overflow-hidden bg-muted"
    >
      {!visible ? (
        <div className="h-full w-full animate-pulse bg-muted-foreground/15" aria-hidden />
      ) : failed ? (
        <div
          className={cn(
            "flex h-full w-full items-center justify-center",
            isSelected ? "bg-success/20" : "bg-muted"
          )}
        >
          <ClassIconFallback classId={classId} isSelected={isSelected} />
        </div>
      ) : (
        /* eslint-disable-next-line @next/next/no-img-element */
        <img
          src={url}
          alt=""
          className="h-full w-full object-cover"
          loading="lazy"
          decoding="async"
          onError={() => setFailed(true)}
        />
      )}
    </div>
  );
}, detectionCropThumbPropsEqual);

/** Stable FPS for frame index <-> time mapping (fallback 25 when metadata is missing). */
function effectivePlaybackFps(video: VideoFile | null): number {
  if (!video) return 25;
  const f = video.fps;
  return f > 0 ? Math.min(Math.max(f, 1), 120) : 25;
}

/** Align frame index with decoded media time using real duration (avoids fps-metadata mismatch -> sticky boxes). */
function timeToFrameIndex(tSec: number, durationSec: number, totalFrames: number): number {
  const maxF = Math.max(0, totalFrames - 1);
  if (maxF <= 0) return 0;
  if (!(durationSec > 0) || !Number.isFinite(durationSec)) return 0;
  const u = Math.min(1, Math.max(0, tSec / durationSec));
  return Math.min(maxF, Math.round(u * maxF));
}

function frameIndexToTimeSec(
  frame: number,
  durationSec: number,
  totalFrames: number,
  fallbackFps: number
): number {
  const maxF = Math.max(0, totalFrames - 1);
  if (maxF <= 0) return 0;
  if (durationSec > 0 && Number.isFinite(durationSec)) {
    return (frame / maxF) * durationSec;
  }
  return frame / fallbackFps;
}

/** Isolated `<video>` so parent frame updates don't reconcile the media element (smoother decode/paint). */
const DetectionStreamVideo = memo(function DetectionStreamVideo({
  streamUrl,
  videoRef,
  onLoadedMetadata,
  onStreamError,
}: {
  streamUrl: string;
  videoRef: Ref<HTMLVideoElement>;
  onLoadedMetadata: (e: SyntheticEvent<HTMLVideoElement>) => void;
  onStreamError: () => void;
}) {
  return (
    <video
      ref={videoRef}
      src={streamUrl}
      className="absolute inset-0 z-0 h-full w-full object-fill transform-gpu"
      muted
      loop
      playsInline
      preload="auto"
      onLoadedMetadata={onLoadedMetadata}
      onError={onStreamError}
    />
  );
});

function formatElapsed(ms: number): string {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  const r = s % 60;
  return m > 0 ? `${m}m ${r.toString().padStart(2, "0")}s` : `${r}s`;
}

function ProgressStat({
  icon,
  label,
  value,
}: {
  icon: ReactNode;
  label: string;
  value: ReactNode;
}) {
  return (
    <div className="flex items-center gap-2 rounded-md bg-muted/40 px-2.5 py-1.5">
      <span className="text-muted-foreground">{icon}</span>
      <span className="flex min-w-0 flex-col leading-tight">
        <span className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</span>
        <span className="truncate text-xs font-medium">{value}</span>
      </span>
    </div>
  );
}

/** Verbose Stage 1 progress panel - shows live telemetry and a Cancel control. */
function DetectionProgressPanel({
  status,
  progress,
  message,
  target,
  elapsedMs,
  videoName,
  stageLabel,
  completedStages,
  totalStages,
  camera,
  camerasProcessed,
  frame,
  frameTotal,
  liveMessage,
  logTail,
  errorMessage,
  onCancel,
  className,
}: {
  status: StageStatus;
  progress: number;
  message?: string;
  target: "local" | "kaggle";
  elapsedMs: number;
  videoName?: string;
  stageLabel?: string;
  completedStages?: number;
  totalStages?: number;
  camera?: string;
  camerasProcessed?: number;
  frame?: number;
  frameTotal?: number;
  liveMessage?: string;
  logTail?: string;
  errorMessage?: string;
  onCancel: () => void;
  className?: string;
}) {
  const running = status === "running";
  const failed = status === "error";
  const pct = Math.max(0, Math.min(100, Math.round(progress)));
  const stageValue =
    completedStages && totalStages
      ? `${stageLabel ?? "Stage"} * ${Math.min(completedStages, totalStages)}/${totalStages}`
      : stageLabel ?? "-";
  return (
    <div
      className={cn(
        "rounded-lg border bg-card shadow-sm",
        failed ? "border-destructive/50" : running ? "border-primary/40" : "border-border",
        className
      )}
    >
      <div className="flex items-center justify-between gap-3 border-b border-border/60 px-4 py-2.5">
        <div className="flex min-w-0 items-center gap-2">
          {failed ? (
            <AlertCircle className="h-4 w-4 shrink-0 text-destructive" />
          ) : running ? (
            <Loader2 className="h-4 w-4 shrink-0 animate-spin text-primary" />
          ) : (
            <ScanLine className="h-4 w-4 shrink-0 text-muted-foreground" />
          )}
          <span className="truncate text-sm font-semibold">
            {failed
              ? "Detection failed"
              : running
                ? "Detecting & tracking..."
                : "Stage 1 - Detection & Tracking"}
          </span>
        </div>
        {running ? (
          <Button
            type="button"
            variant="destructive"
            size="sm"
            className="h-7 shrink-0 gap-1.5 px-2.5"
            onClick={onCancel}
            aria-label="Cancel detection"
          >
            <Square className="h-3 w-3 fill-current" />
            Cancel
          </Button>
        ) : null}
      </div>
      <div className="space-y-3 px-4 py-3">
        <div>
          <div className="mb-1 flex items-center justify-between gap-3 text-xs">
            <span className="truncate text-muted-foreground">{message ?? "Working..."}</span>
            <span className="shrink-0 font-mono text-muted-foreground">{pct}%</span>
          </div>
          {/* Backend reports coarse per-stage milestones, not per-frame progress.
              While running we show the milestone fill PLUS a sliding indeterminate
              sweep so a long step reads as "actively working", not "frozen". */}
          <div className="relative h-2 w-full overflow-hidden rounded-full bg-muted">
            <div
              className="h-full rounded-full bg-primary/70 transition-[width] duration-500"
              style={{ width: `${pct}%` }}
            />
            {running ? (
              <div className="absolute inset-y-0 left-0 w-1/3 animate-indeterminate rounded-full bg-primary" />
            ) : null}
          </div>
        </div>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <ProgressStat
            icon={<Cpu className="h-3.5 w-3.5" />}
            label="Model"
            value="YOLOv26 + Deep OC-SORT"
          />
          <ProgressStat
            icon={target === "kaggle" ? <Cloud className="h-3.5 w-3.5" /> : <Server className="h-3.5 w-3.5" />}
            label="Compute"
            value={target === "kaggle" ? "Kaggle GPU" : "Local"}
          />
          <ProgressStat
            icon={<Clock className="h-3.5 w-3.5" />}
            label="Elapsed"
            value={formatElapsed(elapsedMs)}
          />
          <ProgressStat
            icon={<ScanLine className="h-3.5 w-3.5" />}
            label="Pipeline step"
            value={stageValue}
          />
        </div>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
          <span className="truncate">Source: {videoName ?? "-"}</span>
          {camera ? (
            <span className="font-mono">
              Camera {camera}
              {camerasProcessed ? ` * ${camerasProcessed} processed` : ""}
            </span>
          ) : null}
          {frame && frameTotal ? (
            <span className="font-mono">
              Frame {frame.toLocaleString()}/{frameTotal.toLocaleString()}
            </span>
          ) : null}
        </div>
        {/* What the pipeline is doing right now (more human than the % bar). */}
        {running && liveMessage ? (
          <p className="truncate text-xs text-foreground/80">{liveMessage}</p>
        ) : null}
        {/* Error detail - show WHY a run failed instead of silently reverting. */}
        {failed && errorMessage ? (
          <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
            <p className="font-medium">The detection run did not complete.</p>
            <p className="mt-0.5 break-words font-mono text-[11px] opacity-90">{errorMessage}</p>
          </div>
        ) : null}
        {/* Live, verbose pipeline log - the raw subprocess output, tailing. */}
        {(running || failed) && logTail ? (
          <div className="overflow-hidden rounded-md border border-border/60 bg-background/60">
            <div className="flex items-center gap-1.5 border-b border-border/40 px-2.5 py-1 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
              <Terminal className="h-3 w-3" />
              Live log
            </div>
            <pre className="max-h-36 overflow-auto whitespace-pre-wrap break-words px-2.5 py-2 font-mono text-[10px] leading-relaxed text-muted-foreground">
              {logTail}
            </pre>
          </div>
        ) : null}
      </div>
    </div>
  );
}

/** Camera tab bar - switch which camera's footage + detections are displayed. */
function CameraSwitcher({
  videos,
  currentId,
  onSelect,
}: {
  videos: VideoFile[];
  currentId: string | undefined;
  onSelect: (video: VideoFile) => void;
}) {
  if (videos.length <= 1) return null;
  return (
    <div className="mb-3 flex shrink-0 items-center gap-2 overflow-x-auto pb-1">
      <span className="shrink-0 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
        Cameras
      </span>
      <div className="flex gap-1.5">
        {videos.map((v) => {
          const active = v.id === currentId;
          const label = v.cameraId || v.name;
          return (
            <button
              key={v.id}
              type="button"
              onClick={() => onSelect(v)}
              className={cn(
                "flex shrink-0 items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs font-medium transition-colors",
                active
                  ? "border-accent-strong bg-accent-strong/15 text-foreground"
                  : "border-border bg-background/40 text-muted-foreground hover:bg-background hover:text-foreground"
              )}
              aria-pressed={active}
              title={v.name}
            >
              <Video className="h-3.5 w-3.5 shrink-0" />
              <span className="max-w-[10rem] truncate">{label}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

export function DetectionStage() {
  const { currentVideo, currentFrame, setCurrentFrame, isPlaying, setIsPlaying, videos, setCurrentVideo } =
    useVideoStore();
   const {detections,setDetections,selectedTrackIds,toggleTrackSelection,selectTrackIds,deselectAll,hoveredId,setHoveredId,}
    = useDetectionStore();
  const selectedDataset = useDatasetStore((s) => s.selectedDataset);
  // Filter the tracked-object list by class (All / per detected class).
  const [classFilter, setClassFilter] = useState<number | "all">("all");
  const { runId, stages, updateStageProgress, setIsRunning } = usePipelineStore();
  const { currentStage, setCurrentStage } = useSessionStore();
  // Live run telemetry is published to the store by the shared stage runner.
  const runTelemetry = usePipelineStore((s) => s.runTelemetry) ?? {};
  const runInput = usePipelineStore((s) => s.runInput);
  const stage1Progress = stages.find((stage) => stage.stage === 1);
  const stage1Status = toStageStatus(stage1Progress);

  // Single (one big camera) vs Grid (synced multi-camera wall). Grid is only
  // offered when the run has more than one camera.
  const [viewMode, setViewMode] = useState<"single" | "grid">("single");
  // Seek request handed to the multi-view so "jump to vehicle" works in the grid
  // without leaving it. The token forces a re-seek even for the same frame.
  const [multiSeekRequest, setMultiSeekRequest] = useState<
    { videoId: string; frame: number; token: number } | null
  >(null);
  const seekTokenRef = useRef(0);
  const [isLoading, setIsLoading] = useState(true);
  const [videoSize, setVideoSize] = useState({ width: 1920, height: 1080 });
  // Full source-video frame count (used for accurate frame<->time mapping).
  const [totalFrames, setTotalFrames] = useState(100);
  // Highest frame id that actually has detections - i.e. how far detection was
  // run. For a quick-test (10-frame) run this is ~9, so the scrubber/playback are
  const [detectedMaxFrame, setDetectedMaxFrame] = useState<number | null>(null);
  // All tracked vehicles across the run (not just the current frame).
  const [tracks, setTracks] = useState<TrackSummary[]>([]);
  const [videoError, setVideoError] = useState<string | null>(null);
  const [errorDetail, setErrorDetail] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerSize, setContainerSize] = useState({ width: 800, height: 450 });
  const detectionCacheRef = useRef<Map<number, typeof detections>>(new Map());
  /** Sorted detection frame ids + typical spacing, for tolerant nearest-frame lookups. */
  const sortedFrameKeysRef = useRef<number[]>([]);
  const frameGapRef = useRef(1);
  /** Stable sidebar thumb source per track (first occurrence in cached frames). */
  const trackThumbByTrackRef = useRef<Map<number, { frameId: number; bbox: BoundingBox }>>(
    new Map()
  );
  const videoRef = useRef<HTMLVideoElement>(null);
  /** JPEG frame-by-frame path when /videos/stream fails or unsupported. */
  const [useFrameFallback, setUseFrameFallback] = useState(false);

  const playbackFps = useMemo(() => effectivePlaybackFps(currentVideo), [currentVideo]);

  // Cameras in this run, scoped to the run's SELECTED cameras and ordered by
  // camera id. Scoping defends against a store that still holds other dataset
  const runCameraKey = (runInput?.cameras ?? []).join(",");
  const sortedVideos = useMemo(() => {
    const cams = runCameraKey ? runCameraKey.split(",").filter(Boolean) : [];
    const camSet = cams.length > 0 ? new Set(cams.map((c) => c.toLowerCase())) : null;
    const scoped = camSet
      ? videos.filter((v) => v.cameraId && camSet.has(v.cameraId.toLowerCase()))
      : videos;
    return [...scoped].sort((a, b) =>
      (a.cameraId || a.name).localeCompare(b.cameraId || b.name, undefined, { numeric: true })
    );
  }, [videos, runCameraKey]);

  // If the selected video isn't one of the run's cameras (e.g. a stale value
  // persisted from before), snap to the first in-scope camera so the viewer
  useEffect(() => {
    if (!runInput || sortedVideos.length === 0) return;
    if (currentVideo && sortedVideos.some((v) => v.id === currentVideo.id)) return;
    setCurrentVideo(sortedVideos[0]);
  }, [runInput, sortedVideos, currentVideo, setCurrentVideo]);

  const handleStreamMeta = useCallback(
    (e: SyntheticEvent<HTMLVideoElement>) => {
      const v = e.currentTarget;
      const dur = v.duration;
      if (dur && isFinite(dur) && currentVideo) {
        const fps = effectivePlaybackFps(currentVideo);
        setTotalFrames(Math.max(1, Math.floor(dur * fps)));
      }
    },
    [currentVideo]
  );

  const handleStreamError = useCallback(() => setUseFrameFallback(true), []);

  // Navigable frame count. The <video> element always streams the FULL source
  // file, but for a quick-test run only the first SMOKE_FRAME_LIMIT frames are
  const navTotalFrames = (() => {
    let cap = totalFrames;
    if (detectedMaxFrame != null) cap = Math.min(cap, detectedMaxFrame + 1);
    if (runInput?.smoke) cap = Math.min(cap, SMOKE_FRAME_LIMIT);
    return Math.max(1, cap);
  })();

  // Track ids visible in the current frame (for the "live" badge in the list).
  const inFrameTrackIds = useMemo(
    () => new Set(detections.map((d) => d.trackId)),
    [detections]
  );

  const frameSyncRef = useRef(currentFrame);
  frameSyncRef.current = currentFrame;

  const prevDetectionVideoIdRef = useRef<string | undefined>(undefined);
  useEffect(() => {
    const id = currentVideo?.id;
    if (prevDetectionVideoIdRef.current === undefined) {
      prevDetectionVideoIdRef.current = id;
      return;
    }
    if (prevDetectionVideoIdRef.current !== id) {
      prevDetectionVideoIdRef.current = id;
      // In a multi-camera dataset run every camera belongs to ONE run, so
      // switching the *viewed* camera is just a view change - it must not flush
      if (!usePipelineStore.getState().runInput) {
        flushPipelineFromStage(id ? 2 : 1);
      }
    }
  }, [currentVideo?.id]);

  // DISPLAY ONLY. Show detection artifacts for the current video, but ONLY once
  // detection has actually finished for this run (stage 1 === "done"). This never
  useEffect(() => {
    let cancelled = false;

    const loadForDisplay = async () => {
      if (!currentVideo) {
        trackThumbByTrackRef.current = new Map();
        detectionCacheRef.current = new Map();
        sortedFrameKeysRef.current = [];
        frameGapRef.current = 1;
        setTracks([]);
        setDetections([]);
        setIsLoading(false);
        return;
      }

      setVideoError(null);
      setErrorDetail(null);
      setIsPlaying(false);

      // Set totalFrames from metadata immediately (refined from <video> metadata when stream loads)
      const fps = effectivePlaybackFps(currentVideo);
      setTotalFrames(Math.max(Math.floor(currentVideo.duration * fps), 1));
      setVideoSize({ width: currentVideo.width || 1920, height: currentVideo.height || 1080 });

      // Only show detections once THIS run's detection has completed. Before that
      // (idle / running / error) the viewer stays empty - we never display stale
      if (stage1Status !== "done") {
        detectionCacheRef.current = new Map();
        trackThumbByTrackRef.current = new Map();
        sortedFrameKeysRef.current = [];
        frameGapRef.current = 1;
        setDetectedMaxFrame(null);
        setTracks([]);
        setDetections([]);
        setIsLoading(false);
        return;
      }

      setIsLoading(true);
      try {
        const allDets = await getAllDetections(currentVideo.id);
        if (cancelled) return;
        detectionCacheRef.current = allDets;
        trackThumbByTrackRef.current = buildTrackThumbnailSources(allDets);
        // Bound the scrubber to the frames detection actually produced.
        const frameKeys = [...allDets.keys()];
        const sortedKeys = [...frameKeys].sort((a, b) => a - b);
        sortedFrameKeysRef.current = sortedKeys;
        frameGapRef.current = typicalFrameGap(sortedKeys);
        setDetectedMaxFrame(frameKeys.length ? Math.max(...frameKeys) : null);
        setTracks(summariseTracks(allDets));
        setDetections(valuesForFrame(allDets, sortedKeys, frameSyncRef.current, frameGapRef.current));
      } catch {
        if (cancelled) return;
        detectionCacheRef.current = new Map();
        trackThumbByTrackRef.current = new Map();
        sortedFrameKeysRef.current = [];
        frameGapRef.current = 1;
        setDetectedMaxFrame(null);
        setTracks([]);
        setDetections([]);
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    };

    void loadForDisplay();

    return () => {
      cancelled = true;
    };
  }, [currentVideo, stage1Status, setDetections, setIsPlaying]);

  // Container size tracking - use clientWidth/clientHeight (excludes border)
  // and ResizeObserver for reliable layout tracking.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const updateSize = () => {
      setContainerSize({ width: el.clientWidth, height: el.clientHeight });
    };
    updateSize();
    const ro = new ResizeObserver(updateSize);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Look up cached detections for the current frame - no API call. Uses a tolerant
  // nearest-frame match so boxes still render on scrub/playback positions that fall
  // between the (possibly sparse) frames tracking produced, instead of flickering off.
  useEffect(() => {
    if (!currentVideo || isLoading) return;
    if (detectionCacheRef.current.size === 0) return;
    const next = valuesForFrame(
      detectionCacheRef.current,
      sortedFrameKeysRef.current,
      currentFrame,
      frameGapRef.current
    );
    const prev = useDetectionStore.getState().detections;
    if (prev === next) return;
    if (prev.length === 0 && next.length === 0) return;
    setDetections(next);
  }, [currentVideo, currentFrame, isLoading, setDetections]);

  useEffect(() => {
    setUseFrameFallback(false);
  }, [currentVideo?.id]);

  useEffect(() => {
    const max = Math.max(0, navTotalFrames - 1);
    if (currentFrame > max) {
      setCurrentFrame(max);
    }
  }, [navTotalFrames, currentFrame, setCurrentFrame]);

  // Pause playback when this stage isn't visible. All stages stay mounted (hidden via
  // CSS), so a running video/frame-fallback loop would otherwise keep playing - and
  useEffect(() => {
    if (currentStage !== 1 && isPlaying) setIsPlaying(false);
  }, [currentStage, isPlaying, setIsPlaying]);

  // Play/pause native video when not using JPEG fallback
  useEffect(() => {
    const v = videoRef.current;
    if (!v || !currentVideo || useFrameFallback) return;
    if (isPlaying) {
      void v.play().catch(() => setUseFrameFallback(true));
    } else {
      v.pause();
    }
  }, [isPlaying, currentVideo, useFrameFallback]);

  // While playing with stream: lock bbox/detection updates to decoded video frames (smoothest sync).
  useEffect(() => {
    const v = videoRef.current;
    if (!v || !currentVideo || useFrameFallback || !isPlaying) return;
    let lastEmitted = -1;
    let vfcId = 0;

    const navMax = Math.max(0, navTotalFrames - 1);
    const navMaxTime = frameIndexToTimeSec(navMax, v.duration, totalFrames, playbackFps);
    const tick = (timeSec?: number) => {
      const t = timeSec ?? v.currentTime;
      const dur = v.duration;
      // Loop playback within the processed range - a quick-test run only has the
      // first few frames, so don't keep playing the rest of the source video.
      if (Number.isFinite(navMaxTime) && navMaxTime > 0 && t > navMaxTime + 1e-3) {
        v.currentTime = 0;
        return;
      }
      const maxF = navMax;
      const f =
        dur > 0 && Number.isFinite(dur)
          ? Math.min(maxF, timeToFrameIndex(t, dur, totalFrames))
          : Math.min(maxF, Math.max(0, Math.floor(t * playbackFps + 1e-6)));
      if (f !== lastEmitted) {
        lastEmitted = f;
        setCurrentFrame(f);
      }
    };

    if (typeof v.requestVideoFrameCallback === "function") {
      const onFrame: VideoFrameRequestCallback = (_now, metadata) => {
        const mt = metadata?.mediaTime;
        tick(typeof mt === "number" && Number.isFinite(mt) ? mt : undefined);
        vfcId = v.requestVideoFrameCallback(onFrame);
      };
      vfcId = v.requestVideoFrameCallback(onFrame);
      return () => {
        if (typeof v.cancelVideoFrameCallback === "function") {
          v.cancelVideoFrameCallback(vfcId);
        }
      };
    }

    let raf = 0;
    const onRaf = () => {
      tick();
      raf = requestAnimationFrame(onRaf);
    };
    raf = requestAnimationFrame(onRaf);
    return () => cancelAnimationFrame(raf);
  }, [isPlaying, currentVideo, useFrameFallback, playbackFps, totalFrames, navTotalFrames, setCurrentFrame]);

  // JPEG fallback: advance frames with rAF + accumulated time (smoother than setInterval)
  useEffect(() => {
    if (!isPlaying || !currentVideo || !useFrameFallback) return;
    let acc = 0;
    let last = performance.now();
    let raf = 0;
    let idx = frameSyncRef.current;
    const frameMs = 1000 / playbackFps;
    const step = () => {
      const now = performance.now();
      acc += now - last;
      last = now;
      let advanced = false;
      while (acc >= frameMs) {
        acc -= frameMs;
        idx = idx + 1 >= navTotalFrames ? 0 : idx + 1;
        advanced = true;
      }
      if (advanced) {
        setCurrentFrame(idx);
      }
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [isPlaying, currentVideo, useFrameFallback, totalFrames, navTotalFrames, playbackFps, setCurrentFrame]);

  // When paused, keep the hidden video aligned with the current frame (scrubber / step)
  useEffect(() => {
    const v = videoRef.current;
    if (!v || !currentVideo || useFrameFallback) return;
    if (isPlaying) return;
    const t = frameIndexToTimeSec(currentFrame, v.duration, totalFrames, playbackFps);
    if (Number.isFinite(t) && Math.abs(v.currentTime - t) > 0.001) {
      v.currentTime = t;
    }
  }, [currentFrame, isPlaying, currentVideo, useFrameFallback, playbackFps, totalFrames]);

  const seekToFrame = useCallback(
    (frame: number) => {
      const boundedFrame = Math.min(Math.max(frame, 0), Math.max(navTotalFrames - 1, 0));
      setCurrentFrame(boundedFrame);
      const v = videoRef.current;
      if (v && !useFrameFallback) {
        v.currentTime = frameIndexToTimeSec(boundedFrame, v.duration, totalFrames, playbackFps);
      }
    },
    [navTotalFrames, totalFrames, playbackFps, useFrameFallback, setCurrentFrame]
  );

  // Jump to a tracked vehicle's frame. In multi-view we seek the grid's shared
  // clock (staying in the wall); in single view we pause + seek the player. The
  const jumpToVehicle = useCallback(
    (frame: number) => {
      if (viewMode === "grid" && sortedVideos.length > 1 && currentVideo) {
        seekTokenRef.current += 1;
        setMultiSeekRequest({ videoId: currentVideo.id, frame, token: seekTokenRef.current });
        return;
      }
      setViewMode("single");
      setIsPlaying(false);
      seekToFrame(frame);
    },
    [viewMode, sortedVideos.length, currentVideo, seekToFrame, setIsPlaying]
  );

  const togglePlayback = () => {
    if (!currentVideo || isLoading) return;
    setIsPlaying(!isPlaying);
  };

  const hasVideo = Boolean(currentVideo);
  const executionTarget = useStageExecutionStore((s) => s.getStageExecutionTarget)(1);

  // Live elapsed timer while Stage 1 is running.
  const [elapsedMs, setElapsedMs] = useState(0);
  const runStartRef = useRef<number | null>(null);
  useEffect(() => {
    if (stage1Status === "running") {
      if (runStartRef.current == null) runStartRef.current = Date.now();
      const id = setInterval(
        () => setElapsedMs(Date.now() - (runStartRef.current ?? Date.now())),
        1000
      );
      return () => clearInterval(id);
    }
    runStartRef.current = null;
    setElapsedMs(0);
  }, [stage1Status]);

  const handleCancelRun = useCallback(async () => {
    if (!runId) return;
    try {
      await cancelPipeline(runId);
    } finally {
      setIsRunning(false);
      updateStageProgress(1, { status: "idle", progress: 0, message: "Stage 1 cancelled" });
    }
  }, [runId, setIsRunning, updateStageProgress]);

  const countByClassId = (classId: number) =>
    detections.filter((d) => d.classId === classId).length;

  // Class breakdown + filtered list for the Tracked panel.
  const classCounts = useMemo(() => {
    const counts: Record<number, number> = {};
    for (const t of tracks) counts[t.classId] = (counts[t.classId] ?? 0) + 1;
    return counts;
  }, [tracks]);
  const presentClasses = useMemo(
    () => Object.keys(classCounts).map(Number).sort((a, b) => a - b),
    [classCounts]
  );
  // Whether this run tracks vehicles or people - prefer the real detected classes
  // (robust for re-opened runs) and fall back to the selected dataset before any exist.
  const domain = useMemo(
    () => resolveDomain(selectedDataset, presentClasses),
    [selectedDataset, presentClasses]
  );
  // Class ids visible in the CURRENT frame, for the on-video count strip.
  const framePresentClasses = useMemo(() => {
    const set = new Set<number>();
    for (const d of detections) if (d.classId != null) set.add(d.classId);
    return [...set].sort((a, b) => a - b);
  }, [detections]);
  const DomainIcon = domainIcon(domain);
  const filteredTracks = useMemo(
    () => (classFilter === "all" ? tracks : tracks.filter((t) => t.classId === classFilter)),
    [tracks, classFilter]
  );
  // If the active class filter no longer matches anything, fall back to All.
  useEffect(() => {
    if (classFilter !== "all" && !presentClasses.includes(classFilter)) {
      setClassFilter("all");
    }
  }, [classFilter, presentClasses]);

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden">
      {/* Main content */}
      <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden lg:flex-row">
        {/* Video area */}
        <div className="flex min-h-[200px] min-w-0 flex-1 flex-col overflow-hidden p-3 sm:p-4 lg:min-h-0">
          <ErrorBanner title="Detection failed" message={videoError} className="mb-3 shrink-0 sm:mb-4" />
          {sortedVideos.length > 1 ? (
            <div className="mb-3 flex shrink-0 flex-wrap items-center gap-2">
              <div className="inline-flex items-center gap-0.5 rounded-md border border-border/60 bg-muted/30 p-0.5">
                {(["single", "grid"] as const).map((mode) => (
                  <button
                    key={mode}
                    type="button"
                    onClick={() => setViewMode(mode)}
                    className={cn(
                      "rounded px-2.5 py-1 text-xs font-medium capitalize transition-colors",
                      viewMode === mode
                        ? "bg-background text-foreground shadow-sm"
                        : "text-muted-foreground hover:text-foreground"
                    )}
                    aria-pressed={viewMode === mode}
                  >
                    {mode === "grid" ? "Multi-view" : "Single"}
                  </button>
                ))}
              </div>
              {viewMode === "single" ? (
                <CameraSwitcher
                  videos={sortedVideos}
                  currentId={currentVideo?.id}
                  onSelect={setCurrentVideo}
                />
              ) : (
                <span className="text-xs text-muted-foreground">
                  One scrubber drives all tiles * nudge a camera&apos;s offset to align it
                </span>
              )}
            </div>
          ) : null}
          {stage1Status === "running" || stage1Status === "error" ? (
            <DetectionProgressPanel
              status={stage1Status}
              progress={stage1Progress?.progress ?? 0}
              message={stage1Progress?.message}
              target={executionTarget === "kaggle" ? "kaggle" : "local"}
              elapsedMs={elapsedMs}
              videoName={currentVideo?.name}
              stageLabel={runTelemetry.stageLabel}
              completedStages={runTelemetry.completedStages}
              totalStages={runTelemetry.totalStages}
              camera={runTelemetry.camera}
              camerasProcessed={runTelemetry.camerasProcessed}
              frame={runTelemetry.frame}
              frameTotal={runTelemetry.frameTotal}
              liveMessage={runTelemetry.message}
              logTail={runTelemetry.logTail}
              errorMessage={stage1Status === "error" ? stage1Progress?.message : undefined}
              onCancel={() => void handleCancelRun()}
              className="mb-3 shrink-0 sm:mb-4"
            />
          ) : null}
          {viewMode === "grid" && sortedVideos.length > 1 && hasVideo ? (
            <MultiCameraView
              videos={sortedVideos}
              detectionsReady={stage1Status === "done"}
              seekRequest={multiSeekRequest}
            />
          ) : (
          <>
          {/* Video container - camera view */}
          <div
            ref={containerRef}
            className="relative min-h-0 flex-1 overflow-hidden rounded-lg border border-border bg-black"
          >
            {!hasVideo ? (
              <div className="absolute inset-0 flex items-center justify-center bg-background">
                <div className="max-w-md text-center text-white/85 px-6">
                  <AlertCircle className="h-12 w-12 mx-auto mb-3 text-warning" />
                  <p className="font-medium">No video selected</p>
                  <p className="text-sm text-white/60 mt-2">
                    Go to the input stage and load a dataset. This stage runs YOLOv26 detection with Deep OC-SORT tracking.
                  </p>
                  <Button className="mt-4" variant="secondary" onClick={() => setCurrentStage(0)}>
                    Go To Upload
                  </Button>
                </div>
              </div>
            ) : isLoading ? (
              <div className="absolute inset-0 flex items-center justify-center bg-background">
                <div className="flex max-w-sm flex-col items-center gap-4 px-6 text-center">
                  <Loader2 className="h-14 w-14 text-primary animate-spin" />
                  <div>
                    {stage1Status === "running" ? (
                      <>
                        <p className="font-medium text-white">Detection in progress</p>
                        <p className="mt-1 text-sm text-white/60">
                          The annotated preview appears here once tracking finishes. Live status is shown above.
                        </p>
                      </>
                    ) : (
                      <p className="font-medium text-white">Loading detections...</p>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              <>
                {/* Prefer streamed MP4 for smooth playback; JPEG strips on stream failure (e.g. codec). */}
                {currentVideo && !useFrameFallback && (
                  <DetectionStreamVideo
                    streamUrl={getVideoStreamUrl(currentVideo.id)}
                    videoRef={videoRef}
                    onLoadedMetadata={handleStreamMeta}
                    onStreamError={handleStreamError}
                  />
                )}
                {currentVideo && useFrameFallback && (
                  <DoubleBufferedFrameImg
                    src={getFrameUrl(currentVideo.id, currentFrame)}
                    alt={`Frame ${currentFrame}`}
                    className="object-fill"
                    imgDecoding="async"
                  />
                )}

                <div className="absolute inset-0 z-[1] pointer-events-none">
                  <div className="absolute top-0 left-0 right-0 flex justify-between items-start p-3 bg-gradient-to-b from-black/70 to-transparent">
                    <div>
                      <div className="flex items-center gap-2">
                        <div
                          className={cn(
                            "h-2.5 w-2.5 rounded-full bg-destructive shadow-lg shadow-destructive/50",
                            !isPlaying && "animate-pulse"
                          )}
                        />
                        <span className="max-w-[min(100%,18rem)] truncate text-sm font-mono font-medium text-white sm:max-w-md">
                          {currentVideo?.name ?? "Selected video"}
                        </span>
                        <Badge variant="secondary" className="text-[10px] bg-white/10 text-white border-white/20">
                          LIVE
                        </Badge>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-white/80 text-xs font-mono">
                        {new Date().toLocaleDateString()} {new Date().toLocaleTimeString()}
                      </div>
                      <div className="text-white/50 text-[10px] font-mono">
                        {videoSize.width}x{videoSize.height} @ {playbackFps.toFixed(playbackFps >= 10 ? 0 : 1)}fps
                        {" "}
                        | Frame {currentFrame}
                      </div>
                    </div>
                  </div>

                  <div className="absolute bottom-0 left-0 right-0 p-3 bg-gradient-to-t from-black/80 to-transparent">
                    <div className="flex justify-between items-center text-white/80 text-xs font-mono">
                      <span className="flex items-center gap-3">
                        {framePresentClasses.length === 0 ? (
                          <span className="opacity-70">{detections.length} in frame</span>
                        ) : (
                          framePresentClasses.map((cid) => (
                            <span key={cid} className="flex items-center gap-1" title={classLabelFor(cid)}>
                              <ClassGlyph classId={cid} className="h-3.5 w-3.5" />
                              {countByClassId(cid)}
                            </span>
                          ))
                        )}
                      </span>
                      <span className="text-success">
                        {selectedTrackIds.size} selected for tracking
                      </span>
                    </div>
                  </div>
                </div>

                {/* Bounding boxes overlay - above frame stack (imgs use z-1/z-2) */}
                <div className="absolute inset-0 z-[5]" style={{ pointerEvents: "none" }}>
                  {detections.map((detection) => {
                    const isSelected = selectedTrackIds.has(detection.trackId);
                    const isHovered = hoveredId === detection.id;
                    const style = bboxToStyle(
                      detection.bbox,
                      containerSize.width,
                      containerSize.height,
                      videoSize.width,
                      videoSize.height
                    );

                    return (
                      <div
                        key={detection.id}
                        className={cn(
                          "absolute border-2 cursor-pointer",
                          !isPlaying && "transition-all duration-150",
                          isSelected
                            ? "border-success bg-success/20 shadow-lg shadow-success/30"
                            : "border-destructive bg-destructive/10",
                          isHovered && "ring-2 ring-white/50 scale-[1.02]"
                        )}
                        style={{
                          left: style.left,
                          top: style.top,
                          width: style.width,
                          height: style.height,
                          pointerEvents: "auto",
                        }}
                        onClick={() => toggleTrackSelection(detection.trackId)}
                        onMouseEnter={() => setHoveredId(detection.id)}
                        onMouseLeave={() => setHoveredId(null)}
                      >
                        {/* Label - leads with the track ID so vehicles are identifiable */}
                        <div
                          className={cn(
                            "absolute -top-6 left-0 flex items-center gap-1 rounded-sm px-1.5 py-0.5 text-xs text-white whitespace-nowrap",
                            isSelected ? "bg-success" : "bg-destructive"
                          )}
                        >
                          {detection.trackId != null && detection.trackId >= 0 && (
                            <span className="font-mono font-bold">#{detection.trackId}</span>
                          )}
                          <span className="font-medium capitalize">{detection.className}</span>
                          <span className="opacity-80">{(detection.confidence * 100).toFixed(0)}%</span>
                        </div>

                        {/* Selection checkmark */}
                        {isSelected && (
                          <div className="absolute -top-2 -right-2 h-5 w-5 bg-success rounded-full flex items-center justify-center border-2 border-white">
                            <svg className="h-3 w-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                            </svg>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </>
            )}
          </div>

          {/* Video controls */}
          <PlaybackControls
            className="mt-4 shrink-0"
            isPlaying={isPlaying}
            currentFrame={currentFrame}
            totalFrames={navTotalFrames}
            speedOptions={[]}
            onPlayPause={togglePlayback}
            onFrameChange={seekToFrame}
            onStepBack={() => seekToFrame(0)}
            onStepForward={() => seekToFrame(Math.max(navTotalFrames - 1, 0))}
          />
          </>
          )}
        </div>

        {/* Sidebar - Detection list */}
        <aside className="flex max-h-[42vh] min-h-0 w-full shrink-0 flex-col border-t border-border bg-muted/20 lg:max-h-none lg:w-80 lg:border-l lg:border-t-0">
          <div className="shrink-0 border-b bg-muted/30 p-3">
            <div className="flex items-center justify-between gap-2">
              <div className="flex min-w-0 items-center gap-2">
                <h3 className="font-semibold">{trackedTitle(domain)}</h3>
                {currentVideo?.cameraId && (
                  <Badge variant="outline" className="shrink-0 gap-1 font-mono text-[10px]">
                    <Video className="h-3 w-3" />
                    {currentVideo.cameraId}
                  </Badge>
                )}
              </div>
              <Badge variant="secondary" className="shrink-0">
                {classFilter === "all" ? tracks.length : `${filteredTracks.length} / ${tracks.length}`}
              </Badge>
            </div>
            {presentClasses.length > 0 && (
              <div className="mt-2.5 flex flex-wrap gap-1.5">
                <FilterChip active={classFilter === "all"} onClick={() => setClassFilter("all")} label="All" count={tracks.length} />
                {presentClasses.map((cid) => (
                  <FilterChip
                    key={cid}
                    active={classFilter === cid}
                    onClick={() => setClassFilter(cid)}
                    label={classLabelFor(cid)}
                    count={classCounts[cid]}
                    icon={<ClassGlyph classId={cid} className="h-3 w-3" />}
                  />
                ))}
              </div>
            )}
            <div className="mt-2.5 flex items-center justify-between gap-2">
              <span className="text-[11px] text-muted-foreground">Tap to view * tick to follow across cameras</span>
              <div className="flex shrink-0 items-center gap-1.5">
                <button
                  type="button"
                  className="text-[11px] font-medium text-accent-strong hover:underline disabled:opacity-40"
                  disabled={filteredTracks.length === 0}
                  onClick={() => selectTrackIds(Array.from(new Set([...selectedTrackIds, ...filteredTracks.map((t) => t.trackId)])))}
                >
                  Select all
                </button>
                <span className="text-muted-foreground/40">*</span>
                <button
                  type="button"
                  className="text-[11px] font-medium text-muted-foreground hover:text-destructive disabled:opacity-40"
                  disabled={selectedTrackIds.size === 0}
                  onClick={deselectAll}
                >
                  Clear
                </button>
              </div>
            </div>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden p-3">
            {filteredTracks.length === 0 ? (
              <div className="flex h-full flex-col items-center justify-center gap-2 px-6 text-center text-muted-foreground">
                <DomainIcon className="h-8 w-8 opacity-40" />
                {tracks.length === 0 ? (
                  <>
                    <p className="text-sm font-medium">No tracked {domainNoun(domain, { plural: true })} yet</p>
                    <p className="text-xs">
                      {stage1Status === "done" ? `Detection produced no ${domainNoun(domain, { plural: true })} for this camera.` : "Run detection to populate this list."}
                    </p>
                  </>
                ) : (
                  <p className="text-sm">No {classLabelFor(classFilter as number).toLowerCase()} tracks.</p>
                )}
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-2.5">
                {filteredTracks.map((track) => {
                  const isSelected = selectedTrackIds.has(track.trackId);
                  const inView = inFrameTrackIds.has(track.trackId);
                  const rangeLabel =
                    track.firstFrame === track.lastFrame
                      ? `frame ${track.firstFrame}`
                      : `${track.firstFrame}-${track.lastFrame}`;
                  return (
                    <div
                      key={`track-${track.trackId}`}
                      className={cn(
                        "group relative overflow-hidden rounded-lg border transition-colors",
                        isSelected
                          ? "border-accent-strong ring-1 ring-accent-strong/50"
                          : inView
                            ? "border-emerald-500/50"
                            : "border-border/60 hover:border-border"
                      )}
                      onMouseEnter={() => setHoveredId(`track-${track.trackId}`)}
                      onMouseLeave={() => setHoveredId(null)}
                    >
                      {/* Thumbnail - click to jump to this vehicle */}
                      <button
                        type="button"
                        onClick={() => jumpToVehicle(track.thumbFrameId)}
                        className="relative block aspect-square w-full bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring"
                        title={`Show #${track.trackId} in the viewer (frame ${track.thumbFrameId})`}
                        aria-label={`Show ${track.className} #${track.trackId} in the viewer`}
                      >
                        {currentVideo ? (
                          <DetectionCropThumb
                            videoId={currentVideo.id}
                            classId={track.classId}
                            isSelected={isSelected}
                            cropFrameId={track.thumbFrameId}
                            cropBbox={track.thumbBbox}
                          />
                        ) : (
                          <div className="flex h-full w-full items-center justify-center">
                            <ClassGlyph classId={track.classId} className="h-7 w-7 text-muted-foreground" />
                          </div>
                        )}
                        {/* legibility gradient */}
                        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-9 bg-gradient-to-t from-black/75 to-transparent" />
                        {/* id + live - badge turns green when the vehicle is on screen now */}
                        <span
                          className={cn(
                            "pointer-events-none absolute left-1 top-1 flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] font-bold text-white",
                            inView ? "bg-emerald-600/90" : "bg-black/70"
                          )}
                          title={inView ? "On screen in the current frame" : undefined}
                        >
                          {inView && (
                            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-200" />
                          )}
                          #{track.trackId}
                        </span>
                        {/* class + confidence */}
                        <div className="pointer-events-none absolute inset-x-1.5 bottom-1 flex items-center justify-between text-white">
                          <span className="flex items-center gap-1 text-[11px] font-medium capitalize">
                            <ClassGlyph classId={track.classId} className="h-3 w-3" />
                            {track.className}
                          </span>
                          <span className="text-[10px] tabular-nums text-white/85">{(track.confidence * 100).toFixed(0)}%</span>
                        </div>
                      </button>
                      {/* Selection toggle */}
                      <button
                        type="button"
                        onClick={() => toggleTrackSelection(track.trackId)}
                        aria-pressed={isSelected}
                        aria-label={isSelected ? `Deselect #${track.trackId}` : `Select #${track.trackId}`}
                        className={cn(
                          "absolute right-1 top-1 z-10 flex h-5 w-5 items-center justify-center rounded-full border-2 transition-colors",
                          isSelected
                            ? "border-accent-strong bg-accent-strong text-white"
                            : "border-white/70 bg-black/40 text-transparent hover:border-white hover:text-white/70"
                        )}
                      >
                        <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                        </svg>
                      </button>
                      {/* caption - frame coverage */}
                      <div className="flex items-center justify-between gap-1 px-1.5 py-1 text-[10px] text-muted-foreground">
                        <span className="truncate" title={`Frames ${track.firstFrame}-${track.lastFrame}`}>{rangeLabel}</span>
                        <span className="shrink-0">{track.frameCount}f</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          <div className="shrink-0 space-y-3 border-t bg-muted/30 p-3">
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">Selected for tracking</span>
              <div className="flex items-center gap-2">
                <Badge variant={selectedTrackIds.size > 0 ? "default" : "secondary"}>
                  {selectedTrackIds.size} / {tracks.length}
                </Badge>
                {selectedTrackIds.size > 0 && (
                  <button
                    onClick={deselectAll}
                    className="text-[11px] text-muted-foreground transition-colors hover:text-destructive"
                  >
                    Clear
                  </button>
                )}
              </div>
            </div>
            <DisclosurePanel title="Debug" tier="debug" description="Frame and request telemetry.">
              <div className="space-y-2 text-xs text-muted-foreground">
                <div className="flex justify-between gap-3"><span>Frame</span><span className="font-mono">{currentFrame}/{Math.max(navTotalFrames - 1, 0)}</span></div>
                <div className="flex justify-between gap-3"><span>Detected frames</span><span className="font-mono">{detectedMaxFrame != null ? detectedMaxFrame + 1 : 0}</span></div>
                <div className="flex justify-between gap-3"><span>Raw detections</span><span className="font-mono">{detections.length}</span></div>
                <div className="flex justify-between gap-3"><span>Selected tracks</span><span className="font-mono">{selectedTrackIds.size}</span></div>
                <div className="flex justify-between gap-3"><span>Model</span><span className="font-mono">YOLOv26 + Deep OC-SORT</span></div>
                <div className="break-all font-mono">videoId: {currentVideo?.id ?? "none"}</div>
                {errorDetail ? <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-all rounded bg-background p-2">{errorDetail}</pre> : null}
              </div>
            </DisclosurePanel>
          </div>
        </aside>
      </div>
    </div>
  );
}

export function DetectionStageActions() {
  const { selectedTrackIds } = useDetectionStore();
  const { setCurrentStage } = useSessionStore();
  const { runId, setRunId, stages, updateStageProgress, setError } = usePipelineStore();
  const runInput = usePipelineStore((s) => s.runInput);
  const currentVideo = useVideoStore((s) => s.currentVideo);
  const runPipelineStage = useRunPipelineStage();
  const stageProgress = stages.find((stage) => stage.stage === 1);
  const status = toStageStatus(stageProgress);
  // Dataset flow: ingestion (stage 0) ran separately and must finish first.
  // Probe flow: there's no separate ingestion step - stage 1 ingests THIS video
  // and detects/tracks it in one backend call (execute_stage runs "0,1" together).
  const ingestDone = toStageStatus(stages.find((s) => s.stage === 0)) === "done";
  const datasetFlow = Boolean(runInput);

  const running = status === "running";
  const done = status === "done";
  const canRun = datasetFlow ? ingestDone && !running : Boolean(currentVideo) && !running;

  const handleRun = async () => {
    if (datasetFlow) {
      await runPipelineStage({ pipelineStage: 1, uiStage: 1, label: "detection" });
      return;
    }

    if (!currentVideo) {
      updateStageProgress(1, { status: "error", progress: 100, message: "No probe video selected. Go back to Upload." });
      return;
    }

    setError(null);
    updateStageProgress(1, { status: "running", progress: 0, message: "Running ingestion + detection & tracking..." });

    try {
      const response = await runStage(1, {
        runId: runId ?? undefined,
        videoId: currentVideo.id,
        cameraId: inferCameraId(currentVideo),
      });
      const nextRunId = response.data?.runId ?? runId;
      if (nextRunId) setRunId(nextRunId);
      if (nextRunId) await pollStageStatus(nextRunId, 1, updateStageProgress);
    } catch (error) {
      const message = getRunStageErrorMessage(error);
      setError(message);
      updateStageProgress(1, { status: "error", progress: 100, message });
    }
  };

  const handleCancel = async () => {
    if (!runId) return;
    await cancelPipeline(runId);
  };

  return (
    <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
      {running ? (
        <>
          <span className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin text-primary" />
            Detection running...
          </span>
          <Button type="button" variant="destructive" onClick={() => void handleCancel()} aria-label="Cancel detection run">
            <Square className="mr-2 h-3 w-3 fill-current" />
            Cancel
          </Button>
        </>
      ) : (
        <Button
          type="button"
          onClick={() => void handleRun()}
          disabled={!canRun}
          title={
            datasetFlow
              ? (!ingestDone ? "Waiting for ingestion to finish" : undefined)
              : (!currentVideo ? "Upload or select a probe video first" : undefined)
          }
          aria-label="Run detection & tracking"
        >
          {done ? "Re-run detection" : "Run detection"}
        </Button>
      )}
      <Button
        type="button"
        onClick={() => setCurrentStage(2)}
        disabled={running || selectedTrackIds.size === 0}
        aria-label="Continue to Stage 2 selection"
      >
        Continue to Stage 2
      </Button>
    </div>
  );
}
