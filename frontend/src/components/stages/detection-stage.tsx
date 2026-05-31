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
  Car,
  Truck,
  Bus,
  AlertCircle,
  X,
  Square,
  Clock,
  Cpu,
  Server,
  Cloud,
  ScanLine,
} from "lucide-react";
import { cn, bboxToStyle } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { DisclosurePanel, ErrorBanner, ExecutionTargetToggle, PlaybackControls, RunStageWidget, toStageStatus } from "@/components/pipeline";
import type { StageStatus } from "@/components/pipeline/status/types";
import {
  useVideoStore,
  useDetectionStore,
  usePipelineStore,
  useSessionStore,
  useStageExecutionStore,
} from "@/store";
import { flushPipelineFromStage } from "@/lib/pipeline-flush";
import { useStartStage1 } from "@/hooks/use-start-stage1";
import {
  cancelPipeline,
  getDetections,
  getAllDetections,
  getPipelineStatus,
  getFrameUrl,
  getVideoStreamUrl,
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

/**
 * First frame each track appears — used for sidebar thumbs only.
 * Playback updates overlay boxes every frame; if thumbs used that data, the
 * crop URL would change ~30×/s and flood `/api/crops` (freezing the video).
 */
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

function ClassIconFallback({
  classId,
  isSelected,
}: {
  classId: number;
  isSelected: boolean;
}) {
  const cls = cn("h-6 w-6", isSelected ? "text-success" : "text-muted-foreground");
  if (classId === 2) return <Car className={cls} />;
  if (classId === 7) return <Truck className={cls} />;
  if (classId === 5) return <Bus className={cls} />;
  return <Car className={cls} />;
}

/**
 * Sidebar crop thumbnails: stable crop URL per track + load only when in view.
 */
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
      className={cn(
        "relative h-16 w-16 shrink-0 overflow-hidden rounded-md border bg-muted",
        isSelected ? "border-success/70 ring-1 ring-success/40" : "border-border"
      )}
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

/** Stable FPS for frame index ↔ time mapping (fallback 25 when metadata is missing). */
function effectivePlaybackFps(video: VideoFile | null): number {
  if (!video) return 25;
  const f = video.fps;
  return f > 0 ? Math.min(Math.max(f, 1), 120) : 25;
}

/** Align frame index with decoded media time using real duration (avoids fps-metadata mismatch → sticky boxes). */
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

/** Isolated `<video>` so parent frame updates don’t reconcile the media element (smoother decode/paint). */
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

/** Verbose Stage 1 progress panel — shows live telemetry and a Cancel control. */
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
  onCancel: () => void;
  className?: string;
}) {
  const running = status === "running";
  const pct = Math.max(0, Math.min(100, Math.round(progress)));
  const stageValue =
    completedStages && totalStages
      ? `${stageLabel ?? "Stage"} · ${Math.min(completedStages, totalStages)}/${totalStages}`
      : stageLabel ?? "—";
  return (
    <div
      className={cn(
        "rounded-lg border bg-card shadow-sm",
        running ? "border-primary/40" : "border-border",
        className
      )}
    >
      <div className="flex items-center justify-between gap-3 border-b border-border/60 px-4 py-2.5">
        <div className="flex min-w-0 items-center gap-2">
          {running ? (
            <Loader2 className="h-4 w-4 shrink-0 animate-spin text-primary" />
          ) : (
            <ScanLine className="h-4 w-4 shrink-0 text-muted-foreground" />
          )}
          <span className="truncate text-sm font-semibold">
            {running ? "Detecting & tracking…" : "Stage 1 — Detection & Tracking"}
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
            <span className="truncate text-muted-foreground">{message ?? "Working…"}</span>
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
          <span className="truncate">Source: {videoName ?? "—"}</span>
          {camera ? (
            <span className="font-mono">
              Camera {camera}
              {camerasProcessed ? ` · ${camerasProcessed} processed` : ""}
            </span>
          ) : null}
          {frame && frameTotal ? (
            <span className="font-mono">
              Frame {frame.toLocaleString()}/{frameTotal.toLocaleString()}
            </span>
          ) : null}
        </div>
      </div>
    </div>
  );
}

export function DetectionStage() {
  const { currentVideo, currentFrame, setCurrentFrame, isPlaying, setIsPlaying } =
    useVideoStore();
   const {detections,setDetections,selectedTrackIds,toggleTrackSelection,selectAll,deselectAll,multiSelectMode,setMultiSelectMode,hoveredId,setHoveredId,}
    = useDetectionStore();
  const { runId, stages, updateStageProgress, setIsRunning } = usePipelineStore();
  const { setCurrentStage } = useSessionStore();

  const [isLoading, setIsLoading] = useState(true);
  const [videoSize, setVideoSize] = useState({ width: 1920, height: 1080 });
  const [totalFrames, setTotalFrames] = useState(100);
  const [videoError, setVideoError] = useState<string | null>(null);
  const [errorDetail, setErrorDetail] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerSize, setContainerSize] = useState({ width: 800, height: 450 });
  const detectionCacheRef = useRef<Map<number, typeof detections>>(new Map());
  /** Stable sidebar thumb source per track (first occurrence in cached frames). */
  const trackThumbByTrackRef = useRef<Map<number, { frameId: number; bbox: BoundingBox }>>(
    new Map()
  );
  const videoRef = useRef<HTMLVideoElement>(null);
  /** JPEG frame-by-frame path when /videos/stream fails or unsupported. */
  const [useFrameFallback, setUseFrameFallback] = useState(false);
  /** Live run telemetry parsed from the backend status (stage milestones + camera). */
  const [runTelemetry, setRunTelemetry] = useState<{
    stageLabel?: string;
    completedStages?: number;
    totalStages?: number;
    camera?: string;
    camerasProcessed?: number;
    frame?: number;
    frameTotal?: number;
  }>({});

  const playbackFps = useMemo(() => effectivePlaybackFps(currentVideo), [currentVideo]);

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
      flushPipelineFromStage(id ? 2 : 1);
    }
  }, [currentVideo?.id]);

  // Wait for active stage1 run (if any), then load detections.
  useEffect(() => {
    let cancelled = false;
    let interval: ReturnType<typeof setInterval> | null = null;

    const fetchAllDetections = async () => {
      if (!currentVideo) return;

      try {
        const allDets = await getAllDetections(currentVideo.id);
        if (cancelled) return;
        detectionCacheRef.current = allDets;
        trackThumbByTrackRef.current = buildTrackThumbnailSources(allDets);
        // Show detections for current frame
        setDetections(allDets.get(currentFrame) ?? []);
      } catch {
        if (cancelled) return;
        // Fallback: fetch single frame
        const response = await getDetections(currentVideo.id, currentFrame);
        if (cancelled) return;
        const dets = response.data ?? [];
        trackThumbByTrackRef.current = buildTrackThumbnailSources(
          new Map([[currentFrame, dets]])
        );
        setDetections(dets);
      }
    };

    const loadInitialDetections = async () => {
      if (!currentVideo) {
        trackThumbByTrackRef.current = new Map();
        setDetections([]);
        setIsLoading(false);
        updateStageProgress(1, {
          status: "idle",
          progress: 0,
          message: "Waiting for video selection",
        });
        return;
      }

      setVideoError(null);
      setErrorDetail(null);
      setIsLoading(true);
      setIsPlaying(false);

      // Set totalFrames from metadata immediately (refined from <video> metadata when stream loads)
      if (currentVideo) {
        const fps = effectivePlaybackFps(currentVideo);
        setTotalFrames(Math.max(Math.floor(currentVideo.duration * fps), 1));
        setVideoSize({ width: currentVideo.width || 1920, height: currentVideo.height || 1080 });
      }

      if (!runId) {
        try {
          await fetchAllDetections();
          if (cancelled) return;
          updateStageProgress(1, {
            status: "completed",
            progress: 100,
            message: "Stage 1 artifacts loaded",
          });
        } catch (err) {
          if (cancelled) return;
          const msg = err instanceof Error ? err.message : String(err);
          setVideoError(`Failed to load detections: ${msg}`);
          trackThumbByTrackRef.current = new Map();
          setDetections([]);
          updateStageProgress(1, {
            status: "error",
            progress: 100,
            message: `Stage 1 failed: ${msg}`,
          });
        } finally {
          if (!cancelled) {
            setIsLoading(false);
          }
        }
        return;
      }

      updateStageProgress(1, {
        status: "running",
        progress: 5,
        message: "Running Stage 1 (YOLOv26 + Deep OC-SORT)...",
      });

      const pollStatus = async () => {
        try {
          const statusResponse = await getPipelineStatus(runId);
          if (cancelled) return;

          const statusData: any = statusResponse.data;
          const status = statusData?.status;
          const progress = Number(statusData?.progress ?? 0);
          const message = String(statusData?.message ?? "Running Stage 1...");

          setRunTelemetry({
            stageLabel: statusData?.currentStageName ? String(statusData.currentStageName) : undefined,
            completedStages: statusData?.completedStages != null ? Number(statusData.completedStages) : undefined,
            totalStages: statusData?.totalStages != null ? Number(statusData.totalStages) : undefined,
            camera: statusData?.currentCamera ? String(statusData.currentCamera) : undefined,
            camerasProcessed: statusData?.camerasProcessed != null ? Number(statusData.camerasProcessed) : undefined,
            frame: statusData?.currentFrame != null ? Number(statusData.currentFrame) : undefined,
            frameTotal: statusData?.totalFrames != null ? Number(statusData.totalFrames) : undefined,
          });

          if (status === "completed") {
            if (interval) clearInterval(interval);
            await fetchAllDetections();
            if (cancelled) return;
            updateStageProgress(1, {
              status: "completed",
              progress: 100,
              message,
            });
            setRunTelemetry({});
            setIsRunning(false);
            setIsLoading(false);
            return;
          }

          if (status === "error") {
            if (interval) clearInterval(interval);
            const errMsg = statusData?.error
              ? String(statusData.error)
              : statusData?.message
              ? String(statusData.message)
              : "Stage 1 failed — unknown error";
            const detail = statusData?.errorDetail ? String(statusData.errorDetail) : null;
            setVideoError(errMsg);
            setErrorDetail(detail);
            trackThumbByTrackRef.current = new Map();
            setDetections([]);
            updateStageProgress(1, {
              status: "error",
              progress: 100,
              message: errMsg,
            });
            setRunTelemetry({});
            setIsRunning(false);
            setIsLoading(false);
            return;
          }

          updateStageProgress(1, {
            status: "running",
            progress,
            message,
          });
        } catch (err) {
          if (cancelled) return;
          const msg = err instanceof Error ? err.message : String(err);
          setVideoError(`Failed to poll Stage 1 status: ${msg}`);
          setErrorDetail(null);
          if (interval) clearInterval(interval);
          setIsLoading(false);
        }
      };

      await pollStatus();
      if (!cancelled) {
        interval = setInterval(() => {
          void pollStatus();
        }, 1200);
      }
    };

    loadInitialDetections();

    return () => {
      cancelled = true;
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [currentVideo, runId, setDetections, setIsPlaying, setIsRunning, updateStageProgress]);

  // Container size tracking — use clientWidth/clientHeight (excludes border)
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

  // Look up cached detections for the current frame — no API call.
  useEffect(() => {
    if (!currentVideo || isLoading) return;
    const cached = detectionCacheRef.current.get(currentFrame);
    if (cached) {
      const prev = useDetectionStore.getState().detections;
      if (prev === cached) return;
      setDetections(cached);
    } else if (detectionCacheRef.current.size > 0) {
      const prev = useDetectionStore.getState().detections;
      if (prev.length === 0) return;
      setDetections([]);
    }
  }, [currentVideo, currentFrame, isLoading, setDetections]);

  useEffect(() => {
    setUseFrameFallback(false);
  }, [currentVideo?.id]);

  useEffect(() => {
    const max = Math.max(0, totalFrames - 1);
    if (currentFrame > max) {
      setCurrentFrame(max);
    }
  }, [totalFrames, currentFrame, setCurrentFrame]);

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

    const tick = (timeSec?: number) => {
      const t = timeSec ?? v.currentTime;
      const dur = v.duration;
      const maxF = Math.max(0, totalFrames - 1);
      const f =
        dur > 0 && Number.isFinite(dur)
          ? timeToFrameIndex(t, dur, totalFrames)
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
  }, [isPlaying, currentVideo, useFrameFallback, playbackFps, totalFrames, setCurrentFrame]);

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
        idx = idx + 1 >= totalFrames ? 0 : idx + 1;
        advanced = true;
      }
      if (advanced) {
        setCurrentFrame(idx);
      }
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [isPlaying, currentVideo, useFrameFallback, totalFrames, playbackFps, setCurrentFrame]);

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
      const boundedFrame = Math.min(Math.max(frame, 0), Math.max(totalFrames - 1, 0));
      setCurrentFrame(boundedFrame);
      const v = videoRef.current;
      if (v && !useFrameFallback) {
        v.currentTime = frameIndexToTimeSec(boundedFrame, v.duration, totalFrames, playbackFps);
      }
    },
    [totalFrames, playbackFps, useFrameFallback, setCurrentFrame]
  );

  const togglePlayback = () => {
    if (!currentVideo || isLoading) return;
    setIsPlaying(!isPlaying);
  };

  const hasVideo = Boolean(currentVideo);
  const stage1Progress = stages.find((stage) => stage.stage === 1);
  const stage1Status = toStageStatus(stage1Progress);
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

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden">
      {/* Main content */}
      <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden lg:flex-row">
        {/* Video area */}
        <div className="flex min-h-[200px] min-w-0 flex-1 flex-col overflow-hidden p-3 sm:p-4 lg:min-h-0">
          <ErrorBanner title="Detection failed" message={videoError} className="mb-3 shrink-0 sm:mb-4" />
          {stage1Status === "running" ? (
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
              onCancel={() => void handleCancelRun()}
              className="mb-3 shrink-0 sm:mb-4"
            />
          ) : null}
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
                      <p className="font-medium text-white">Loading detections…</p>
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
                        <span className="flex items-center gap-1">
                          <Car className="h-3.5 w-3.5" />
                          {countByClassId(2)}
                        </span>
                        <span className="flex items-center gap-1">
                          <Truck className="h-3.5 w-3.5" />
                          {countByClassId(7)}
                        </span>
                        <span className="flex items-center gap-1">
                          <Bus className="h-3.5 w-3.5" />
                          {countByClassId(5)}
                        </span>
                      </span>
                      <span className="text-success">
                        {selectedTrackIds.size} selected for tracking
                      </span>
                    </div>
                  </div>
                </div>

                {/* Bounding boxes overlay — above frame stack (imgs use z-1/z-2) */}
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
                        {/* Label */}
                        <div
                          className={cn(
                            "absolute -top-6 left-0 px-2 py-0.5 text-xs font-medium text-white rounded-sm whitespace-nowrap",
                            isSelected ? "bg-success" : "bg-destructive"
                          )}
                        >
                          {detection.className} {(detection.confidence * 100).toFixed(0)}%
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
            totalFrames={totalFrames}
            speedOptions={[]}
            onPlayPause={togglePlayback}
            onFrameChange={seekToFrame}
            onStepBack={() => seekToFrame(0)}
            onStepForward={() => seekToFrame(Math.max(totalFrames - 1, 0))}
          />
        </div>

        {/* Sidebar - Detection list */}
        <aside className="flex max-h-[42vh] min-h-0 w-full shrink-0 flex-col border-t border-border bg-muted/20 lg:max-h-none lg:w-80 lg:border-l lg:border-t-0">
          <div className="shrink-0 border-b bg-muted/30 p-4">
            <div className="flex items-center justify-between gap-3">
              <div className="min-w-0">
                <h3 className="font-semibold">Detected Vehicles</h3>
                <p className="text-sm text-muted-foreground">Click boxes or list items to select</p>
              </div>
              <Badge variant="secondary" className="shrink-0">{detections.length}</Badge>
            </div>
          </div>
          <div className="shrink-0 space-y-3 border-b p-3">
            <DisclosurePanel title="Advanced" description="Selection controls for the current detection set.">
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <Checkbox id="stage1-multi-select" checked={multiSelectMode} onCheckedChange={(checked) => setMultiSelectMode(checked === true)} />
                  <Label htmlFor="stage1-multi-select" className="text-sm">Multi-select mode</Label>
                </div>
                <div className="flex flex-wrap gap-2">
                  <Button type="button" variant="outline" size="sm" onClick={selectAll} disabled={detections.length === 0} aria-label="Select all visible detections">
                    Select All
                  </Button>
                  <Button type="button" variant="outline" size="sm" onClick={deselectAll} disabled={selectedTrackIds.size === 0} aria-label="Deselect all selected detections">
                    Deselect All
                  </Button>
                </div>
              </div>
            </DisclosurePanel>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden p-3">
            <div className="space-y-2">
              {detections.map((detection) => {
              const isSelected = selectedTrackIds.has(detection.trackId); 
              const isHovered = hoveredId === detection.id;
              const thumbSrc =
                trackThumbByTrackRef.current.get(detection.trackId) ?? null;
              const cropFrameId = thumbSrc?.frameId ?? detection.frameId;
              const cropBbox = thumbSrc?.bbox ?? detection.bbox;
                return (
                  <div
                    key={`track-${detection.trackId}`}
                    className={cn(
                      "flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all",
                      isSelected
                        ? "border-success bg-success/10"
                        : "border-transparent bg-background/50 hover:bg-background",
                      isHovered && "ring-1 ring-primary"
                    )}
                    onClick={() => toggleTrackSelection(detection.trackId)}
                    onMouseEnter={() => setHoveredId(detection.id)}
                    onMouseLeave={() => setHoveredId(null)}
                  >
                    {currentVideo ? (
                      <DetectionCropThumb
                        videoId={currentVideo.id}
                        classId={detection.classId}
                        isSelected={isSelected}
                        cropFrameId={cropFrameId}
                        cropBbox={cropBbox}
                      />
                    ) : (
                      <div
                        className={cn(
                          "flex h-16 w-16 shrink-0 items-center justify-center rounded-md",
                          isSelected ? "bg-success/20" : "bg-muted"
                        )}
                      >
                        {detection.classId === 2 && (
                          <Car className={cn("h-6 w-6", isSelected ? "text-success" : "text-muted-foreground")} />
                        )}
                        {detection.classId === 7 && (
                          <Truck className={cn("h-6 w-6", isSelected ? "text-success" : "text-muted-foreground")} />
                        )}
                        {detection.classId === 5 && (
                          <Bus className={cn("h-6 w-6", isSelected ? "text-success" : "text-muted-foreground")} />
                        )}
                      </div>
                    )}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-medium capitalize">{detection.className}</span>
                        <Badge variant="secondary" className="text-[10px]">
                          {(detection.confidence * 100).toFixed(0)}%
                        </Badge>
                      </div>
                      <p className="text-xs text-muted-foreground">
                        ID: {detection.id}
                      </p>
                    </div>
                    <div className={cn(
                      "h-5 w-5 rounded-full border-2 flex items-center justify-center transition-colors",
                      isSelected ? "bg-success border-success" : "border-muted-foreground/30"
                    )}>
                      {isSelected && (
                        <svg className="h-3 w-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                        </svg>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="shrink-0 space-y-3 border-t bg-muted/30 p-4">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm text-muted-foreground">Selected</span>
              <div className="flex items-center gap-2">
                <Badge variant={selectedTrackIds.size > 0 ? "default" : "secondary"}>
                  {selectedTrackIds.size} / {detections.length}
                </Badge>
                {selectedTrackIds.size > 0 && (
                  <button
                    onClick={deselectAll}
                    className="text-[11px] text-muted-foreground hover:text-destructive transition-colors"
                  >
                    Clear
                  </button>
                )}
              </div>
            </div>
            {selectedTrackIds.size > 0 && (
              <div className="mb-3 flex flex-wrap gap-1">
                {Array.from(selectedTrackIds).sort((a, b) => a - b).map((id) => (
                  <button
                    key={id}
                    onClick={() => toggleTrackSelection(id)}
                    className="group flex items-center gap-0.5 rounded-full border bg-muted/50 px-2 py-0.5 text-[10px] font-mono transition-colors hover:bg-destructive/10 hover:border-destructive/30"
                  >
                    #{id}
                    <X className="h-2.5 w-2.5 text-muted-foreground group-hover:text-destructive" />
                  </button>
                ))}
              </div>
            )}
            <DisclosurePanel title="Debug" tier="debug" description="Frame and request telemetry.">
              <div className="space-y-2 text-xs text-muted-foreground">
                <div className="flex justify-between gap-3"><span>Frame</span><span className="font-mono">{currentFrame}/{Math.max(totalFrames - 1, 0)}</span></div>
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
  const { currentVideo } = useVideoStore();
  const { selectedTrackIds } = useDetectionStore();
  const { setCurrentStage } = useSessionStore();
  const { runId, stages, setIsRunning, updateStageProgress } = usePipelineStore();
  const getStageExecutionTarget = useStageExecutionStore((state) => state.getStageExecutionTarget);
  const startStage1 = useStartStage1();
  const stageProgress = stages.find((stage) => stage.stage === 1);
  const status = toStageStatus(stageProgress);
  const executionTarget = getStageExecutionTarget(1);

  // Drive controls off the stage STATUS, not the `isRunning` store flag: runs
  // launched from Stage 0 (runDatasetInput) never set that flag, which is why
  // the "Run Stage 1" button stayed clickable during an active run.
  const running = status === "running";
  const done = status === "done";

  const handleRun = async () => {
    await startStage1();
  };

  const handleCancel = async () => {
    if (!runId) return;
    try {
      await cancelPipeline(runId);
    } finally {
      setIsRunning(false);
      updateStageProgress(1, { status: "idle", progress: 0, message: "Stage 1 cancelled" });
    }
  };

  return (
    <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
      {running ? (
        <>
          <span className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin text-primary" />
            Detection running…
          </span>
          <Button type="button" variant="destructive" onClick={() => void handleCancel()} aria-label="Cancel Stage 1 run">
            <Square className="mr-2 h-3 w-3 fill-current" />
            Cancel
          </Button>
        </>
      ) : (
        <>
          <ExecutionTargetToggle stage={1} variant="compact" />
          <RunStageWidget
            target={executionTarget}
            runId={runId}
            status={status}
            progress={stageProgress?.progress ?? 0}
            message={stageProgress?.message}
            isRunning={false}
            disabled={!currentVideo}
            runLabel={done ? "Re-run Stage 1" : "Run Stage 1"}
            mode="button-only"
            onRun={() => void handleRun()}
          />
        </>
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
