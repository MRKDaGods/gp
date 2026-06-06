"use client";

import { useEffect, useRef, useState } from "react";
import { Car, Loader2 } from "lucide-react";

import { TrackletFrameView } from "@/components/ui/double-buffered-img";
import { apiUrl, getRunFullFrameUrl, getTrackletSequence, type TrackletSequenceFrame } from "@/lib/api";
import { cn, formatDuration } from "@/lib/utils";

import type { TimelinePreviewCamera } from "./types";

function shouldUseRunCropsForCamera(runId: string | undefined, _cameraId: string): boolean {
  if (!runId) return false;
  return true;
}

export interface TimelineVideoGridProps {
  cameras: TimelinePreviewCamera[];
  splitCount: number;
  tracksLoading?: boolean;
  currentVideoId?: string;
  currentVideoTime: number;
  trackletPickTime: number;
  isPlaying: boolean;
  probeRunId?: string;
  cropRunId?: string;
}

export function TimelineVideoGrid({
  cameras,
  splitCount,
  tracksLoading = false,
  currentVideoId,
  currentVideoTime,
  trackletPickTime,
  isPlaying,
  probeRunId,
  cropRunId,
}: TimelineVideoGridProps) {
  const columns = Math.ceil(Math.sqrt(splitCount));
  const rows = Math.ceil(splitCount / columns);

  return (
    <div
      className="relative shrink-0 border-b border-border/60 bg-background p-2"
      style={{ height: "clamp(200px, min(42vh, 50dvh), 560px)" }}
    >
      <div
        className="grid h-full min-h-0 min-w-0 gap-1"
        style={{
          gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`,
          gridTemplateRows: `repeat(${rows}, minmax(0, 1fr))`,
        }}
      >
        {cameras.map((camera) => (
          <CameraPreview
            key={camera.id}
            camera={camera}
            isActive={Boolean(camera.segment)}
            isPast={camera.isPast}
            isNext={camera.isNext}
            absCurrentTime={currentVideoTime}
            trackletPickTime={trackletPickTime}
            isPlaying={isPlaying}
            primarySeg={camera.primarySeg}
            probeRunId={probeRunId}
            videoId={currentVideoId}
            cropRunId={cropRunId}
          />
        ))}
      </div>
      {tracksLoading && (
        <div
          className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 rounded-md bg-background/85 px-4 backdrop-blur"
          role="status"
          aria-live="polite"
          aria-label="Loading timeline previews"
        >
          <Loader2 className="h-10 w-10 animate-spin text-primary" />
          <p className="text-center text-sm text-muted-foreground">Loading camera previews...</p>
        </div>
      )}
    </div>
  );
}

function TimelinePreviewAspectShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-b from-slate-900 via-slate-950 to-black p-1 sm:p-1.5">
      <div
        className={cn(
          "flex min-h-0 min-w-0 items-center justify-center overflow-hidden rounded-md",
          "border border-white/10 bg-black shadow-inner",
          "aspect-video h-full w-auto max-h-full max-w-full"
        )}
      >
        <div className="relative flex h-full min-h-0 w-full min-w-0 items-center justify-center">
          {children}
        </div>
      </div>
    </div>
  );
}

function CameraPreview({
  camera,
  isActive,
  isPast,
  isNext,
  absCurrentTime,
  trackletPickTime,
  isPlaying,
  primarySeg,
  probeRunId,
  videoId,
  cropRunId,
}: {
  camera: { id: string; name: string; location: string; activeTrack?: any };
  isActive: boolean;
  isPast?: boolean;
  isNext?: boolean;
  absCurrentTime: number;
  trackletPickTime: number;
  isPlaying: boolean;
  primarySeg?: { globalId?: number; cameraId: string; trackId: number; start: number; end: number };
  probeRunId?: string;
  videoId?: string;
  cropRunId?: string;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const seekWallRef = useRef(absCurrentTime);
  const clipStartWallRef = useRef(primarySeg?.start ?? 0);
  const [clipFailed, setClipFailed] = useState(false);
  const [stableScrubTime, setStableScrubTime] = useState(absCurrentTime);
  const [trackSeq, setTrackSeq] = useState<{
    key: string;
    width: number;
    height: number;
    frames: TrackletSequenceFrame[];
  } | null>(null);

  const clipUrl = (() => {
    if (!probeRunId || !primarySeg) return null;
    const { globalId, cameraId, trackId } = primarySeg;
    if (globalId == null || trackId == null) return null;
    const safeCam = String(cameraId).replace(/[/\\]/g, "_");
    const filename = `global_${globalId}_cam_${safeCam}_track_${trackId}.mp4`;
    return apiUrl(`/runs/${probeRunId}/matched_clips/${filename}`);
  })();

  const trackSeqKey =
    primarySeg?.cameraId != null && primarySeg.trackId != null
      ? `${String(primarySeg.cameraId)}|${Number(primarySeg.trackId)}`
      : "";

  const segmentIdentityKey = `${trackSeqKey}|${primarySeg?.globalId ?? ""}|${clipUrl ?? ""}`;
  const prevSegmentIdentityRef = useRef<string>("");

  useEffect(() => {
    if (!cropRunId || !primarySeg?.cameraId || primarySeg.trackId == null) {
      setTrackSeq(null);
      return;
    }
    const key = `${String(primarySeg.cameraId)}|${Number(primarySeg.trackId)}`;
    let cancelled = false;
    setTrackSeq((prev) => (prev?.key === key ? prev : null));
    void (async () => {
      try {
        const data = await getTrackletSequence(
          cropRunId,
          String(primarySeg.cameraId),
          Number(primarySeg.trackId),
          120
        );
        if (cancelled) return;
        if (data?.frames?.length) {
          setTrackSeq({
            key,
            width: data.width,
            height: data.height,
            frames: data.frames,
          });
        } else {
          setTrackSeq(null);
        }
      } catch {
        if (!cancelled) setTrackSeq(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [cropRunId, primarySeg?.cameraId, primarySeg?.trackId]);

  useEffect(() => {
    if (segmentIdentityKey !== prevSegmentIdentityRef.current) {
      prevSegmentIdentityRef.current = segmentIdentityKey;
      setStableScrubTime(absCurrentTime);
    }
  }, [segmentIdentityKey, absCurrentTime]);

  useEffect(() => {
    if (isPlaying) {
      setStableScrubTime(absCurrentTime);
      return;
    }
    const id = window.setTimeout(() => setStableScrubTime(absCurrentTime), 90);
    return () => window.clearTimeout(id);
  }, [absCurrentTime, isPlaying]);

  seekWallRef.current = isPlaying ? absCurrentTime : stableScrubTime;

  useEffect(() => {
    setClipFailed(false);
  }, [clipUrl]);

  const clipStartSec = primarySeg?.start ?? 0;
  clipStartWallRef.current = clipStartSec;

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !clipUrl) return;
    const onCanPlay = () => {
      const currentVideo = videoRef.current;
      if (!currentVideo) return;
      currentVideo.currentTime = Math.max(0, seekWallRef.current - clipStartWallRef.current);
    };
    video.addEventListener("canplay", onCanPlay, { once: true });
    return () => video.removeEventListener("canplay", onCanPlay);
  }, [clipUrl]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !clipUrl) return;
    if (isPlaying) {
      const seekTo = Math.max(0, absCurrentTime - clipStartSec);
      if (Math.abs(video.currentTime - seekTo) > 1.25) video.currentTime = seekTo;
      return;
    }
    const seekTo = Math.max(0, stableScrubTime - clipStartSec);
    if (Math.abs(video.currentTime - seekTo) < 0.04) return;
    video.currentTime = seekTo;
  }, [isPlaying, absCurrentTime, stableScrubTime, clipUrl, clipStartSec]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !clipUrl) return;
    // Only the camera that currently holds the vehicle plays. Otherwise every tile with a
    // clip plays at once, so a past camera's clip keeps replaying ("looping") after the
    // vehicle has already handed off to the next camera.
    if (isPlaying && isActive) {
      video.play().catch(() => {});
    } else {
      video.pause();
    }
  }, [isPlaying, isActive, clipUrl]);

  const cropUrl = (() => {
    if (!camera.activeTrack) return null;
    const track = camera.activeTrack;
    const bbox = track.representativeBbox;
    const frameId = track.representativeFrame;
    if (frameId == null) return null;
    const bboxParams = bbox && bbox.length === 4
      ? `x1=${bbox[0]}&y1=${bbox[1]}&x2=${bbox[2]}&y2=${bbox[3]}`
      : "x1=0&y1=0&x2=9999&y2=9999";
    if (cropRunId && shouldUseRunCropsForCamera(cropRunId, track.cameraId)) {
      return apiUrl(`/crops/run/${cropRunId}?cameraId=${encodeURIComponent(track.cameraId)}&frameId=${frameId}&${bboxParams}`);
    }
    if (videoId) {
      return apiUrl(`/crops/${videoId}?frameId=${frameId}&${bboxParams}`);
    }
    return null;
  })();

  const trackletFramePick = (() => {
    if (!trackSeq?.frames?.length || !primarySeg || !trackSeqKey || trackSeq.key !== trackSeqKey) {
      return null;
    }
    const segStart = primarySeg.start;
    const segEnd = Math.max(primarySeg.end, segStart + 1e-3);
    const pickTime = isPlaying ? trackletPickTime : stableScrubTime;
    const u = Math.min(1, Math.max(0, (pickTime - segStart) / (segEnd - segStart)));
    const frames = trackSeq.frames;
    let best = frames[0];
    let bestDistance = 1;
    for (const frame of frames) {
      const distance = Math.abs(frame.timeRel - u);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = frame;
      }
    }
    return { frame: best };
  })();

  const trackletFullSrc =
    trackletFramePick && cropRunId
      ? getRunFullFrameUrl(cropRunId, String(primarySeg!.cameraId), trackletFramePick.frame.frameId)
      : null;

  const showTrackletFrames = Boolean(trackletFullSrc && trackletFramePick);
  const showVideoClip = Boolean(clipUrl && !clipFailed && !showTrackletFrames);
  const showCropOnly = Boolean(cropUrl && !showTrackletFrames && !showVideoClip);

  const ringClass = showTrackletFrames
    ? isActive
      ? "ring-2 ring-success"
      : isPast
        ? "ring-1 ring-warning/60 opacity-70"
        : isNext
          ? "ring-1 ring-info/60 opacity-70"
          : "opacity-50"
    : clipUrl && !clipFailed
      ? isActive
        ? "ring-2 ring-success"
        : isPast
          ? "ring-1 ring-warning/60 opacity-70"
          : isNext
            ? "ring-1 ring-info/60 opacity-70"
            : "opacity-50"
      : isActive
        ? "ring-2 ring-success"
        : isPast
          ? "ring-1 ring-warning/50 opacity-50"
          : isNext
            ? "ring-1 ring-info/50 opacity-40"
            : "opacity-30";

  const statusLabel = isActive ? null : isPast ? "PAST" : isNext ? "NEXT" : null;
  const statusColor = isPast ? "text-warning" : "text-info";

  return (
    <div className={cn("relative h-full min-h-0 w-full min-w-0 overflow-hidden rounded", ringClass)}>
      <TimelinePreviewAspectShell>
        {showTrackletFrames ? (
          <TrackletFrameView src={trackletFullSrc!} bbox={trackletFramePick!.frame.bbox} />
        ) : showVideoClip ? (
          <video
            key={clipUrl}
            ref={videoRef}
            src={clipUrl!}
            poster={cropUrl ?? undefined}
            className="max-h-full max-w-full object-contain"
            muted
            playsInline
            preload="auto"
            onError={() => setClipFailed(true)}
          />
        ) : showCropOnly ? (
          <img src={cropUrl!} alt={camera.id} className="max-h-full max-w-full object-contain" draggable={false} />
        ) : (
          <div className="relative h-full min-h-0 w-full min-w-0">
            <svg className="pointer-events-none absolute inset-0 h-full w-full opacity-20" preserveAspectRatio="none">
              <line x1="50%" y1="30%" x2="20%" y2="100%" stroke="white" strokeWidth="1" />
              <line x1="50%" y1="30%" x2="80%" y2="100%" stroke="white" strokeWidth="1" />
            </svg>
            {isActive && camera.activeTrack && (
              <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
                <div
                  className="flex h-6 w-10 items-center justify-center rounded border-2"
                  style={{
                    borderColor: camera.activeTrack.color || "hsl(var(--success))",
                    backgroundColor: camera.activeTrack.color ? `${camera.activeTrack.color}33` : "hsl(var(--success) / 0.2)",
                  }}
                >
                  <Car className="h-4 w-4 text-white" />
                </div>
              </div>
            )}
          </div>
        )}
      </TimelinePreviewAspectShell>

      <div className="absolute left-0 right-0 top-0 bg-black/60 p-1">
        <div className="flex items-center gap-1">
          <div
            className={cn(
              "h-1.5 w-1.5 rounded-full",
              isActive
                ? isPlaying
                  ? "bg-success"
                  : "animate-pulse bg-success"
                : isPast
                  ? "bg-warning"
                  : isNext
                    ? "bg-info"
                    : "bg-gray-500"
            )}
          />
          <span className="font-mono text-[10px] text-white">{camera.id}</span>
          {statusLabel && <span className={cn("ml-auto text-[9px] font-bold", statusColor)}>{statusLabel}</span>}
        </div>
      </div>

      <div className="absolute bottom-0 left-0 right-0 bg-black/60 p-1">
        <span className="font-mono text-[9px] text-white/70">{formatDuration(absCurrentTime)}</span>
      </div>
    </div>
  );
}
