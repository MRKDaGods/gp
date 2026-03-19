"use client";

import { useEffect, useState, useRef } from "react";
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
} from "lucide-react";
import { cn, bboxToStyle } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import {
  useVideoStore,
  useDetectionStore,
  usePipelineStore,
  useSessionStore,
} from "@/store";
import { getDetections, getAllDetections, getPipelineStatus, getVideoStreamUrl } from "@/lib/api";

export function DetectionStage() {
  const { currentVideo, currentFrame, setCurrentFrame, isPlaying, setIsPlaying } =
    useVideoStore();
  const { detections, setDetections, selectedIds, toggleSelection, hoveredId, setHoveredId } =
    useDetectionStore();
  const { runId, stages, updateStageProgress, setIsRunning } = usePipelineStore();
  const { setCurrentStage } = useSessionStore();

  const [isLoading, setIsLoading] = useState(true);
  const [videoSize, setVideoSize] = useState({ width: 1920, height: 1080 });
  const [totalFrames, setTotalFrames] = useState(100);
  const [videoError, setVideoError] = useState<string | null>(null);
  const [videoFallback, setVideoFallback] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [containerSize, setContainerSize] = useState({ width: 800, height: 450 });
  const detectionCacheRef = useRef<Map<number, typeof detections>>(new Map());

  const stage1Progress = stages.find((s) => s.stage === 1);

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
        // Show detections for current frame
        setDetections(allDets.get(currentFrame) ?? []);
      } catch {
        if (cancelled) return;
        // Fallback: fetch single frame
        const response = await getDetections(currentVideo.id, currentFrame);
        if (cancelled) return;
        setDetections(response.data ?? []);
      }
    };

    const loadInitialDetections = async () => {
      if (!currentVideo) {
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
      setVideoFallback(false);
      setIsLoading(true);
      setIsPlaying(false);

      // Set totalFrames from metadata immediately (will be refined when video loads)
      if (currentVideo) {
        const fps = Math.max(currentVideo.fps || 10, 1);
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
        message: "Running Stage 1 (YOLOv8 + Deep OC-SORT)...",
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
            setVideoError(String(statusData?.error ?? "Stage 1 backend run failed."));
            setDetections([]);
            updateStageProgress(1, {
              status: "error",
              progress: 100,
              message: String(statusData?.error ?? "Stage 1 failed"),
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
      setDetections(cached);
    } else if (detectionCacheRef.current.size > 0) {
      // Cache loaded but no data for this frame → empty
      setDetections([]);
    }
    // If cache is empty we haven't bulk-loaded yet; the initial load effect
    // will populate it.
  }, [currentVideo, currentFrame, isLoading, setDetections]);

  // Keep store frame synced with video playback time.
  useEffect(() => {
    const videoElement = videoRef.current;
    if (!videoElement || !currentVideo) return;

    const onLoadedMetadata = () => {
      const videoWidth = videoElement.videoWidth || currentVideo.width || 1920;
      const videoHeight = videoElement.videoHeight || currentVideo.height || 1080;
      setVideoSize({ width: videoWidth, height: videoHeight });

      const fps = Math.max(currentVideo.fps || 10, 1);
      const frames = Math.max(Math.floor(videoElement.duration * fps), 1);
      setTotalFrames(frames);
    };

    const onTimeUpdate = () => {
      const fps = Math.max(currentVideo.fps || 10, 1);
      const frame = Math.min(Math.floor(videoElement.currentTime * fps), Math.max(totalFrames - 1, 0));
      if (frame !== currentFrame) {
        setCurrentFrame(frame);
      }
    };

    const onPlay = () => setIsPlaying(true);
    const onPause = () => setIsPlaying(false);
    const onEnded = () => setIsPlaying(false);
    const onError = () => {
      setVideoFallback(true);
      // Set totalFrames from metadata so slider works in fallback mode
      if (currentVideo) {
        const fps = Math.max(currentVideo.fps || 10, 1);
        setTotalFrames(Math.max(Math.floor(currentVideo.duration * fps), 1));
        setVideoSize({ width: currentVideo.width || 1920, height: currentVideo.height || 1080 });
      }
    };

    videoElement.addEventListener("loadedmetadata", onLoadedMetadata);
    videoElement.addEventListener("timeupdate", onTimeUpdate);
    videoElement.addEventListener("play", onPlay);
    videoElement.addEventListener("pause", onPause);
    videoElement.addEventListener("ended", onEnded);
    videoElement.addEventListener("error", onError);

    return () => {
      videoElement.removeEventListener("loadedmetadata", onLoadedMetadata);
      videoElement.removeEventListener("timeupdate", onTimeUpdate);
      videoElement.removeEventListener("play", onPlay);
      videoElement.removeEventListener("pause", onPause);
      videoElement.removeEventListener("ended", onEnded);
      videoElement.removeEventListener("error", onError);
    };
  }, [currentVideo, currentFrame, setCurrentFrame, setIsPlaying, totalFrames]);

  const seekToFrame = (frame: number) => {
    const boundedFrame = Math.min(Math.max(frame, 0), Math.max(totalFrames - 1, 0));
    setCurrentFrame(boundedFrame);

    if (videoFallback || !currentVideo || !videoRef.current) return;
    const fps = Math.max(currentVideo.fps || 10, 1);
    videoRef.current.currentTime = boundedFrame / fps;
  };

  const togglePlayback = async () => {
    if (!currentVideo || isLoading) return;

    if (videoFallback) {
      setIsPlaying(!isPlaying);
      return;
    }

    if (!videoRef.current) return;
    if (videoRef.current.paused) {
      try {
        await videoRef.current.play();
      } catch {
        // Fall back to timer-based playback
        setVideoFallback(true);
        setIsPlaying(true);
      }
      return;
    }

    videoRef.current.pause();
  };

  // Timer-based frame advancement for fallback mode
  useEffect(() => {
    if (!videoFallback || !isPlaying || !currentVideo) return;
    const fps = Math.max(currentVideo.fps || 10, 1);
    let frame = currentFrame;
    const interval = setInterval(() => {
      frame = frame + 1 >= totalFrames ? 0 : frame + 1;
      setCurrentFrame(frame);
    }, 1000 / fps);
    return () => clearInterval(interval);
  }, [videoFallback, isPlaying, currentVideo, totalFrames, setCurrentFrame]);

  const handleProceed = () => {
    if (selectedIds.size > 0) {
      setCurrentStage(2);
    }
  };

  const hasVideo = Boolean(currentVideo);

  const countByClassId = (classId: number) =>
    detections.filter((d) => d.classId === classId).length;

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <header className="flex h-14 items-center justify-between border-b px-6">
        <div>
          <h1 className="text-lg font-semibold">Stage 1: Vehicle Detection</h1>
          <p className="text-sm text-muted-foreground">
            YOLOv8 + Deep OC-SORT on CityFlowV2 footage
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="secondary">
            {detections.length} vehicles detected
          </Badge>
          <Badge variant="outline" className="bg-green-500/10 text-green-500 border-green-500/30">
            {selectedIds.size} selected
          </Badge>
          <Button onClick={handleProceed} disabled={selectedIds.size === 0}>
            Continue to Selection
          </Button>
        </div>
      </header>

      {/* Main content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Video area */}
        <div className="flex-1 flex flex-col p-4">
          {/* Progress bar during detection */}
          {stage1Progress?.status === "running" && (
            <div className="mb-4 p-3 bg-muted/50 rounded-lg">
              <div className="flex justify-between text-sm mb-2">
                <span className="flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  {stage1Progress.message}
                </span>
                <span className="font-mono">{stage1Progress.progress}%</span>
              </div>
              <Progress value={stage1Progress.progress} className="h-2" />
            </div>
          )}

          {/* Video container - CityFlow camera view */}
          <div
            ref={containerRef}
            className="relative flex-1 bg-black rounded-lg overflow-hidden border border-border"
          >
            {!hasVideo ? (
              <div className="absolute inset-0 flex items-center justify-center bg-slate-900">
                <div className="max-w-md text-center text-white/85 px-6">
                  <AlertCircle className="h-12 w-12 mx-auto mb-3 text-amber-400" />
                  <p className="font-medium">No video selected</p>
                  <p className="text-sm text-white/60 mt-2">
                    Go back to Upload and pick a CityFlowV2 video. The stage will run YOLOv8 detection with Deep OC-SORT tracking.
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
                    <p className="text-white/60 text-sm">Running YOLOv8 + Deep OC-SORT...</p>
                  </div>
                </div>
              </div>
            ) : (
              <>
                {videoFallback ? (
                  <div className="absolute inset-0 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
                    {/* Synthetic camera feed background */}
                    <svg className="absolute inset-0 w-full h-full opacity-15" preserveAspectRatio="none">
                      <line x1="50%" y1="25%" x2="15%" y2="100%" stroke="white" strokeWidth="1.5" />
                      <line x1="50%" y1="25%" x2="85%" y2="100%" stroke="white" strokeWidth="1.5" />
                      <line x1="50%" y1="25%" x2="50%" y2="100%" stroke="#fbbf24" strokeWidth="1.5" strokeDasharray="8,6" />
                      <line x1="0%" y1="65%" x2="100%" y2="65%" stroke="white" strokeWidth="0.5" strokeDasharray="12,8" />
                    </svg>
                    <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(59,130,246,0.08),transparent_70%)]" />
                  </div>
                ) : (
                  <video
                    ref={videoRef}
                    src={currentVideo ? getVideoStreamUrl(currentVideo.id) : undefined}
                    className="absolute inset-0 h-full w-full object-fill"
                    preload="metadata"
                    muted
                    playsInline
                  />
                )}

                <div className="absolute inset-0 pointer-events-none">
                  <div className="absolute top-0 left-0 right-0 flex justify-between items-start p-3 bg-gradient-to-b from-black/70 to-transparent">
                    <div>
                      <div className="flex items-center gap-2">
                        <div className="h-2.5 w-2.5 rounded-full bg-red-500 animate-pulse shadow-lg shadow-red-500/50" />
                        <span className="text-white text-sm font-mono font-medium">
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
                        {videoSize.width}x{videoSize.height} @ {currentVideo?.fps || 10}fps | Frame {currentFrame}
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
                        {selectedIds.size} selected for tracking
                      </span>
                    </div>
                  </div>
                </div>

                {/* Bounding boxes overlay */}
                <div className="absolute inset-0" style={{ pointerEvents: "none" }}>
                  {detections.map((detection) => {
                    const isSelected = selectedIds.has(detection.id);
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
                          "absolute border-2 cursor-pointer transition-all duration-150",
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
                        onClick={() => toggleSelection(detection.id)}
                        onMouseEnter={() => setHoveredId(detection.id)}
                        onMouseLeave={() => setHoveredId(null)}
                      >
                        {/* Vehicle silhouette inside bbox */}
                        <div className="absolute inset-0 flex items-center justify-center opacity-40">
                          {detection.classId === 2 && <Car className="w-1/2 h-1/2 text-white" />}
                          {detection.classId === 7 && <Truck className="w-1/2 h-1/2 text-white" />}
                          {detection.classId === 5 && <Bus className="w-1/2 h-1/2 text-white" />}
                        </div>

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
              <div className="absolute top-3 left-3 right-3 z-20 rounded-md border border-red-500/40 bg-red-500/15 px-3 py-2 text-sm text-red-100">
                {videoError}
              </div>
            )}
          </div>

          {/* Video controls */}
          <div className="mt-4 flex items-center gap-4 p-3 bg-muted/30 rounded-lg">
            <div className="flex items-center gap-1">
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

            <div className="flex-1">
              <Slider
                value={[currentFrame]}
                max={Math.max(totalFrames - 1, 0)}
                step={1}
                onValueChange={(v) => seekToFrame(v[0])}
                disabled={isLoading || !hasVideo}
              />
            </div>

            <div className="text-sm text-muted-foreground font-mono min-w-[100px] text-right">
              {currentFrame}/{Math.max(totalFrames - 1, 0)} frames
            </div>
          </div>
        </div>

        {/* Sidebar - Detection list */}
        <aside className="w-80 border-l flex flex-col bg-muted/20">
          <div className="p-4 border-b bg-muted/30">
            <h3 className="font-semibold">Detected Vehicles</h3>
            <p className="text-sm text-muted-foreground">
              Click boxes or list items to select
            </p>
          </div>
          <div className="flex-1 overflow-auto p-3">
            <div className="space-y-2">
              {detections.map((detection) => {
                const isSelected = selectedIds.has(detection.id);
                const isHovered = hoveredId === detection.id;
                return (
                  <div
                    key={detection.id}
                    className={cn(
                      "flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all",
                      isSelected
                        ? "border-green-500 bg-green-500/10"
                        : "border-transparent bg-background/50 hover:bg-background",
                      isHovered && "ring-1 ring-primary"
                    )}
                    onClick={() => toggleSelection(detection.id)}
                    onMouseEnter={() => setHoveredId(detection.id)}
                    onMouseLeave={() => setHoveredId(null)}
                  >
                    <div className={cn(
                      "h-10 w-10 rounded flex items-center justify-center",
                      isSelected ? "bg-green-500/20" : "bg-muted"
                    )}>
                      {detection.classId === 2 && <Car className={cn("h-5 w-5", isSelected ? "text-green-500" : "text-muted-foreground")} />}
                      {detection.classId === 7 && <Truck className={cn("h-5 w-5", isSelected ? "text-green-500" : "text-muted-foreground")} />}
                      {detection.classId === 5 && <Bus className={cn("h-5 w-5", isSelected ? "text-green-500" : "text-muted-foreground")} />}
                    </div>
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
          <div className="p-4 border-t bg-muted/30">
            <div className="flex justify-between items-center mb-3">
              <span className="text-sm text-muted-foreground">Selected</span>
              <Badge variant={selectedIds.size > 0 ? "default" : "secondary"}>
                {selectedIds.size} / {detections.length}
              </Badge>
            </div>
            <Button
              className="w-full"
              onClick={handleProceed}
              disabled={selectedIds.size === 0}
            >
              {selectedIds.size > 0
                ? `Track ${selectedIds.size} Vehicle${selectedIds.size > 1 ? 's' : ''}`
                : "Select vehicles to continue"
              }
            </Button>
          </div>
        </aside>
      </div>
    </div>
  );
}
