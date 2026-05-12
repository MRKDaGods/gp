"use client";

import {
  memo,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type Ref,
  type SyntheticEvent,
} from "react";
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Loader2,
  Car,
  Truck,
  Bus,
  AlertCircle,
  X,
} from "lucide-react";
import { cn, bboxToStyle } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import {
  useVideoStore,
  useDetectionStore,
  usePipelineStore,
  useSessionStore,
} from "@/store";
import { flushPipelineFromStage } from "@/lib/pipeline-flush";
import {
  getDetections,
  getAllDetections,
  getPipelineStatus,
  getFrameUrl,
  getVideoStreamUrl,
} from "@/lib/api";
import type { BoundingBox, Detection, VideoFile } from "@/types";
import { DoubleBufferedFrameImg } from "@/components/ui/double-buffered-img";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8004/api";

function detectionCropUrl(
  videoId: string,
  frameId: number,
  bbox: BoundingBox,
  quality: number = 92,
  minEdge: number = 160
): string {
  const { x1, y1, x2, y2 } = bbox;
  return `${API_BASE}/crops/${encodeURIComponent(videoId)}?frameId=${frameId}&x1=${x1}&y1=${y1}&x2=${x2}&y2=${y2}&quality=${quality}&minEdge=${minEdge}&pad=0.12`;
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
  const cls = cn("h-6 w-6", isSelected ? "text-green-500" : "text-muted-foreground");
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
        isSelected ? "border-green-500/70 ring-1 ring-green-500/40" : "border-border"
      )}
    >
      {!visible ? (
        <div className="h-full w-full animate-pulse bg-muted-foreground/15" aria-hidden />
      ) : failed ? (
        <div
          className={cn(
            "flex h-full w-full items-center justify-center",
            isSelected ? "bg-green-500/20" : "bg-muted"
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

export function DetectionStage() {
  const { currentVideo, currentFrame, setCurrentFrame, isPlaying, setIsPlaying } =
    useVideoStore();
   const {detections,setDetections,selectedTrackIds,toggleTrackSelection,deselectAll,hoveredId,setHoveredId,}
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

          if (status === "completed") {
            if (interval) clearInterval(interval);
            await fetchAllDetections();
            if (cancelled) return;
            updateStageProgress(1, {
              status: "completed",
              progress: 100,
              message,
            });
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

  const handleProceed = () => {
    if (selectedTrackIds.size > 0) {
      setCurrentStage(2);
    }
  };

  const hasVideo = Boolean(currentVideo);

  const countByClassId = (classId: number) =>
    detections.filter((d) => d.classId === classId).length;

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden">
      {/* Header */}
      <header className="flex shrink-0 flex-col gap-3 border-b px-4 py-3 sm:flex-row sm:items-center sm:justify-between sm:px-6">
        <div className="min-w-0">
          <h1 className="text-lg font-semibold">Stage 1: Vehicle Detection</h1>
          <p className="text-sm text-muted-foreground">
            YOLOv26 + Deep OC-SORT on CityFlowV2 footage
          </p>
        </div>
        <div className="flex flex-wrap items-center justify-end gap-2">
          <Badge variant="secondary">
            {detections.length} vehicles detected
          </Badge>
          <Badge variant="outline" className="bg-green-500/10 text-green-500 border-green-500/30">
            {selectedTrackIds.size} selected
          </Badge>
          <Button className="shrink-0" onClick={handleProceed} disabled={selectedTrackIds.size === 0}>
            Continue to Selection
          </Button>
        </div>
      </header>

      {/* Main content */}
      <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden lg:flex-row">
        {/* Video area */}
        <div className="flex min-h-[200px] min-w-0 flex-1 flex-col overflow-hidden p-3 sm:p-4 lg:min-h-0">
          {/* Video container - CityFlow camera view */}
          <div
            ref={containerRef}
            className="relative min-h-0 flex-1 overflow-hidden rounded-lg border border-border bg-black"
          >
            {!hasVideo ? (
              <div className="absolute inset-0 flex items-center justify-center bg-slate-900">
                <div className="max-w-md text-center text-white/85 px-6">
                  <AlertCircle className="h-12 w-12 mx-auto mb-3 text-amber-400" />
                  <p className="font-medium">No video selected</p>
                  <p className="text-sm text-white/60 mt-2">
                    Go back to Upload and pick a CityFlowV2 video. The stage will run YOLOv26 detection with Deep OC-SORT tracking.
                  </p>
                  <Button className="mt-4" variant="secondary" onClick={() => setCurrentStage(0)}>
                    Go To Upload
                  </Button>
                </div>
              </div>
            ) : isLoading ? (
              <div className="absolute inset-0 flex items-center justify-center bg-slate-900">
                <div className="flex flex-col items-center gap-4">
                  <Loader2 className="h-14 w-14 text-primary animate-spin" />
                  <div className="text-center">
                    <p className="text-white font-medium">Processing Video</p>
                    <p className="text-white/60 text-sm">Running YOLOv26 + Deep OC-SORT...</p>
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
                            "h-2.5 w-2.5 rounded-full bg-red-500 shadow-lg shadow-red-500/50",
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
                      <span className="text-green-400">
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
                            ? "border-green-500 bg-green-500/20 shadow-lg shadow-green-500/30"
                            : "border-red-500 bg-red-500/10",
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
                            isSelected ? "bg-green-600" : "bg-red-600"
                          )}
                        >
                          {detection.className} {(detection.confidence * 100).toFixed(0)}%
                        </div>

                        {/* Selection checkmark */}
                        {isSelected && (
                          <div className="absolute -top-2 -right-2 h-5 w-5 bg-green-500 rounded-full flex items-center justify-center border-2 border-white">
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

            {videoError && (
              <div className="absolute top-3 left-3 right-3 z-20 rounded-md border border-red-500/50 bg-red-950/80 backdrop-blur-sm text-sm text-red-100 overflow-hidden">
                <div className="flex items-start gap-2 px-3 py-2">
                  <AlertCircle className="h-4 w-4 text-red-400 flex-shrink-0 mt-0.5" />
                  <div className="flex-1 min-w-0">
                    <p className="font-semibold text-red-300 text-xs uppercase tracking-wide mb-1">Pipeline Error</p>
                    <p className="break-words text-red-100">{videoError}</p>
                    {errorDetail && (
                      <details className="mt-2">
                        <summary className="text-xs text-red-400 cursor-pointer hover:text-red-300">Show full traceback</summary>
                        <pre className="mt-1 text-[10px] text-red-300/80 font-mono whitespace-pre-wrap break-all max-h-48 overflow-y-auto bg-black/40 rounded p-2">{errorDetail}</pre>
                      </details>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Video controls */}
          <div className="mt-4 flex shrink-0 flex-wrap items-center gap-3 rounded-lg bg-muted/30 p-3 sm:gap-4">
            <div className="flex shrink-0 items-center gap-1">
              <Button variant="ghost" size="icon" onClick={() => seekToFrame(0)} disabled={isLoading || !hasVideo}>
                <SkipBack className="h-4 w-4" />
              </Button>
              <Button
                variant="default"
                size="icon"
                onClick={togglePlayback}
                disabled={isLoading || !hasVideo}
              >
                {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              </Button>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => seekToFrame(Math.max(totalFrames - 1, 0))}
                disabled={isLoading || !hasVideo}
              >
                <SkipForward className="h-4 w-4" />
              </Button>
            </div>

            <div className="min-w-[120px] flex-1 basis-[160px]">
              <Slider
                value={[currentFrame]}
                max={Math.max(totalFrames - 1, 0)}
                step={1}
                onValueChange={(v) => seekToFrame(v[0])}
                disabled={isLoading || !hasVideo}
              />
            </div>

            <div className="shrink-0 text-right font-mono text-sm text-muted-foreground">
              {currentFrame}/{Math.max(totalFrames - 1, 0)} frames
            </div>
          </div>
        </div>

        {/* Sidebar - Detection list */}
        <aside className="flex max-h-[42vh] min-h-0 w-full shrink-0 flex-col border-t border-border bg-muted/20 lg:max-h-none lg:w-80 lg:border-l lg:border-t-0">
          <div className="shrink-0 border-b bg-muted/30 p-4">
            <h3 className="font-semibold">Detected Vehicles</h3>
            <p className="text-sm text-muted-foreground">
              Click boxes or list items to select
            </p>
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
                        ? "border-green-500 bg-green-500/10"
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
                          isSelected ? "bg-green-500/20" : "bg-muted"
                        )}
                      >
                        {detection.classId === 2 && (
                          <Car className={cn("h-6 w-6", isSelected ? "text-green-500" : "text-muted-foreground")} />
                        )}
                        {detection.classId === 7 && (
                          <Truck className={cn("h-6 w-6", isSelected ? "text-green-500" : "text-muted-foreground")} />
                        )}
                        {detection.classId === 5 && (
                          <Bus className={cn("h-6 w-6", isSelected ? "text-green-500" : "text-muted-foreground")} />
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
                      isSelected ? "bg-green-500 border-green-500" : "border-muted-foreground/30"
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

          {/* Selection summary */}
          <div className="shrink-0 border-t bg-muted/30 p-4">
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
            <Button
              className="w-full"
              onClick={handleProceed}
              disabled={selectedTrackIds.size === 0}
            >
              {selectedTrackIds.size > 0
                ? `Track ${selectedTrackIds.size} Vehicle${selectedTrackIds.size > 1 ? 's' : ''}`
                : "Select vehicles to continue"
              }
            </Button>
          </div>
        </aside>
      </div>
    </div>
  );
}
