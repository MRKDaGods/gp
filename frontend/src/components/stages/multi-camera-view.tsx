"use client";

import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { GripVertical, Loader2, Minus, Plus, RotateCcw } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { PlaybackControls } from "@/components/pipeline";
import { cn, bboxToStyle } from "@/lib/utils";
import { getAllDetections, getVideoStreamUrl } from "@/lib/api";
import type { BoundingBox, Detection, VideoFile } from "@/types";

/** Synthetic master-timeline resolution (frames/sec) for the shared scrubber. */
const MASTER_FPS = 10;
/** Drift (s) under which a slaved tile is considered in sync - no correction. */
const SOFT_DRIFT = 0.08;
/** Drift (s) past which we hard-seek a slave to catch up (last resort). */
const HARD_DRIFT = 0.6;
/** Gentle rate trims to converge a slightly-off slave without re-buffering. */
const RATE_AHEAD = 0.9; // slave is ahead -> slow down
const RATE_BEHIND = 1.1; // slave is behind -> speed up
/** Seconds per manual offset nudge (for aligning cameras that start at different times). */
const OFFSET_STEP = 0.2;

interface VideoMeta {
  duration: number;
  width: number;
  height: number;
}

interface CameraDetections {
  byFrame: Map<number, Detection[]>;
  maxFrame: number;
}

/** Map a local media time to the nearest extracted detection-frame index. */
function timeToFrame(tSec: number, durationSec: number, maxFrame: number): number {
  if (maxFrame <= 0 || !(durationSec > 0)) return 0;
  const u = Math.min(1, Math.max(0, tSec / durationSec));
  return Math.round(u * maxFrame);
}

/** One synchronized camera tile: native <video> + bbox overlays + offset nudge. */
const CameraPane = memo(function CameraPane({
  video,
  registerRef,
  detections,
  outOfRange,
  videoW,
  videoH,
  offset,
  onNudgeOffset,
  onResetOffset,
  onMeta,
  isOver,
  isDragging,
  onDragStart,
  onDragOver,
  onDrop,
  onDragEnd,
}: {
  video: VideoFile;
  registerRef: (id: string, el: HTMLVideoElement | null) => void;
  detections: Detection[];
  outOfRange: boolean;
  videoW: number;
  videoH: number;
  offset: number;
  onNudgeOffset: (id: string, delta: number) => void;
  onResetOffset: (id: string) => void;
  onMeta: (id: string, meta: VideoMeta) => void;
  isOver: boolean;
  isDragging: boolean;
  onDragStart: (id: string) => void;
  onDragOver: (id: string) => void;
  onDrop: (id: string) => void;
  onDragEnd: () => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 320, height: 180 });
  const [buffering, setBuffering] = useState(false);

  // Stable ref callback so frequent overlay re-renders don't detach the <video>.
  const setVideoEl = useCallback(
    (el: HTMLVideoElement | null) => registerRef(video.id, el),
    [registerRef, video.id]
  );

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const update = () => setSize({ width: el.clientWidth, height: el.clientHeight });
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const handleGripDragStart = (e: React.DragEvent) => {
    e.dataTransfer.effectAllowed = "move";
    try {
      e.dataTransfer.setData("text/plain", video.id);
    } catch {
      /* some browsers restrict setData */
    }
    onDragStart(video.id);
  };
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
    onDragOver(video.id);
  };
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    onDrop(video.id);
  };

  return (
    <div className="flex min-h-0 min-w-0 flex-col">
      <div
        ref={containerRef}
        onDragOver={handleDragOver}
        onDragEnter={handleDragOver}
        onDrop={handleDrop}
        className={cn(
          "relative min-h-0 flex-1 overflow-hidden rounded-md border bg-black transition-shadow",
          isOver ? "border-accent-strong ring-2 ring-accent-strong/60" : "border-border",
          isDragging && "opacity-50"
        )}
      >
        {/* object-fill (not contain) keeps bbox coords aligned to the container. */}
        {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
        <video
          ref={setVideoEl}
          src={getVideoStreamUrl(video.id)}
          className="absolute inset-0 z-0 h-full w-full object-fill"
          draggable={false}
          muted
          playsInline
          preload="auto"
          onWaiting={() => setBuffering(true)}
          onStalled={() => setBuffering(true)}
          onSeeking={() => setBuffering(true)}
          onPlaying={() => setBuffering(false)}
          onCanPlay={() => setBuffering(false)}
          onSeeked={() => setBuffering(false)}
          onLoadedMetadata={(e) => {
            const v = e.currentTarget;
            onMeta(video.id, {
              duration: Number.isFinite(v.duration) ? v.duration : 0,
              width: v.videoWidth || video.width || 1920,
              height: v.videoHeight || video.height || 1080,
            });
          }}
        />

        {/* Buffering spinner - shown while this tile waits for more data. */}
        {buffering && !outOfRange && (
          <div className="absolute inset-0 z-[4] flex items-center justify-center bg-black/30">
            <span className="flex items-center gap-1.5 rounded-md bg-black/75 px-2.5 py-1.5 text-[11px] text-white/90">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              buffering...
            </span>
          </div>
        )}

        {/* Bounding boxes */}
        {!outOfRange && (
          <div className="absolute inset-0 z-[5]" style={{ pointerEvents: "none" }}>
            {detections.map((d) => {
              const style = bboxToStyle(d.bbox as BoundingBox, size.width, size.height, videoW, videoH);
              return (
                <div
                  key={d.id}
                  className="absolute border-2 border-destructive bg-destructive/10"
                  style={{ left: style.left, top: style.top, width: style.width, height: style.height }}
                >
                  <div className="absolute -top-5 left-0 flex items-center gap-1 whitespace-nowrap rounded-sm bg-destructive px-1.5 py-px text-[10px] text-white">
                    {d.trackId != null && d.trackId >= 0 && (
                      <span className="font-mono font-bold">#{d.trackId}</span>
                    )}
                    <span className="font-medium capitalize">{d.className}</span>
                    <span className="opacity-80">{Math.round(d.confidence * 100)}%</span>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Camera label + drag handle */}
        <div className="absolute left-1.5 top-1.5 z-[6] flex items-center gap-1.5">
          <span
            draggable
            onDragStart={handleGripDragStart}
            onDragEnd={onDragEnd}
            className="flex cursor-grab items-center rounded bg-black/70 px-1 py-0.5 text-white/80 hover:bg-black/85 active:cursor-grabbing"
            title="Drag to reorder"
            aria-label={`Drag to reorder ${video.cameraId ?? video.name}`}
          >
            <GripVertical className="h-3.5 w-3.5" />
          </span>
          <span className="rounded bg-black/70 px-1.5 py-0.5 font-mono text-[11px] font-medium text-white">
            {video.cameraId || video.name}
          </span>
          {!outOfRange && (
            <Badge variant="secondary" className="border-white/20 bg-white/10 text-[9px] text-white">
              {detections.length}
            </Badge>
          )}
        </div>

        {/* Out-of-range veil (camera doesn't cover the master time after offset) */}
        {outOfRange && (
          <div className="absolute inset-0 z-[6] flex items-center justify-center bg-black/70">
            <span className="rounded bg-black/60 px-2 py-1 text-[11px] text-white/70">
              outside this camera&apos;s range
            </span>
          </div>
        )}

        {/* Per-camera offset nudge - align cameras that start at different times */}
        <div className="absolute bottom-1.5 right-1.5 z-[6] flex items-center gap-0.5 rounded-md bg-black/70 p-0.5 text-white">
          <button
            type="button"
            className="rounded p-1 hover:bg-white/15"
            onClick={() => onNudgeOffset(video.id, -OFFSET_STEP)}
            title="Shift this camera earlier"
            aria-label={`Shift ${video.cameraId ?? video.name} earlier`}
          >
            <Minus className="h-3 w-3" />
          </button>
          <span className="min-w-[3.2rem] text-center font-mono text-[10px] tabular-nums">
            {offset >= 0 ? "+" : ""}
            {offset.toFixed(1)}s
          </span>
          <button
            type="button"
            className="rounded p-1 hover:bg-white/15"
            onClick={() => onNudgeOffset(video.id, OFFSET_STEP)}
            title="Shift this camera later"
            aria-label={`Shift ${video.cameraId ?? video.name} later`}
          >
            <Plus className="h-3 w-3" />
          </button>
          {offset !== 0 && (
            <button
              type="button"
              className="rounded p-1 hover:bg-white/15"
              onClick={() => onResetOffset(video.id)}
              title="Reset offset"
              aria-label={`Reset ${video.cameraId ?? video.name} offset`}
            >
              <RotateCcw className="h-3 w-3" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
});

/** Pick a grid column count that keeps tiles reasonably sized. */
function gridColsClass(n: number): string {
  if (n <= 1) return "grid-cols-1";
  if (n === 2) return "grid-cols-2";
  if (n <= 4) return "grid-cols-2";
  if (n <= 6) return "grid-cols-3";
  return "grid-cols-4";
}

/** Synchronized multi-camera "video wall". A single master clock (global seconds) */
export function MultiCameraView({
  videos,
  detectionsReady,
  seekRequest,
}: {
  videos: VideoFile[];
  /** Whether detection has finished for this run (gates overlay loading). */
  detectionsReady: boolean;
  /** External "jump to this camera's frame" request (from the sidebar). The */
  seekRequest?: { videoId: string; frame: number; token: number } | null;
}) {
  // Which cameras are tiled in the grid (VSCode-style tabs toggle these).
  const [includedIds, setIncludedIds] = useState<Set<string>>(
    () => new Set(videos.map((v) => v.id))
  );
  // Display order of the tiles (drag-to-reorder).
  const [order, setOrder] = useState<string[]>(() => videos.map((v) => v.id));
  const [dragId, setDragId] = useState<string | null>(null);
  const [overId, setOverId] = useState<string | null>(null);
  const dragIdRef = useRef<string | null>(null);
  dragIdRef.current = dragId;
  const [isPlaying, setIsPlaying] = useState(false);
  const [gTime, setGTime] = useState(0);
  const [meta, setMeta] = useState<Record<string, VideoMeta>>({});
  const [offset, setOffset] = useState<Record<string, number>>({});
  const [detData, setDetData] = useState<Record<string, CameraDetections>>({});
  const [loadingDet, setLoadingDet] = useState(false);

  const elRefs = useRef<Record<string, HTMLVideoElement | null>>({});
  const metaRef = useRef(meta);
  const offsetRef = useRef(offset);
  const gRef = useRef(0);
  // The tile whose real playback position currently drives the master clock.
  const masterIdRef = useRef<string | null>(null);
  metaRef.current = meta;
  offsetRef.current = offset;

  // Keep the included set valid as the run's camera list changes.
  useEffect(() => {
    setIncludedIds((prev) => {
      const live = new Set(videos.map((v) => v.id));
      const next = new Set([...prev].filter((id) => live.has(id)));
      if (next.size === 0) videos.forEach((v) => next.add(v.id));
      return next;
    });
  }, [videos]);

  // Seed metadata from the video records so the timeline has a length before the
  // <video> elements report their own (refined) metadata.
  useEffect(() => {
    setMeta((prev) => {
      const next = { ...prev };
      for (const v of videos) {
        if (!next[v.id]) {
          next[v.id] = { duration: v.duration || 0, width: v.width || 1920, height: v.height || 1080 };
        }
      }
      return next;
    });
  }, [videos]);

  // Load per-camera detections once tracking is done.
  useEffect(() => {
    let cancelled = false;
    if (!detectionsReady) {
      setDetData({});
      return;
    }
    setLoadingDet(true);
    void (async () => {
      const entries = await Promise.all(
        videos.map(async (v) => {
          try {
            const byFrame = await getAllDetections(v.id);
            const keys = [...byFrame.keys()];
            const maxFrame = keys.length ? Math.max(...keys) : 0;
            return [v.id, { byFrame, maxFrame }] as const;
          } catch {
            return [v.id, { byFrame: new Map<number, Detection[]>(), maxFrame: 0 }] as const;
          }
        })
      );
      if (cancelled) return;
      setDetData(Object.fromEntries(entries));
      setLoadingDet(false);
    })();
    return () => {
      cancelled = true;
    };
  }, [videos, detectionsReady]);

  // Keep the display order in sync as the run's camera list changes (append new,
  // drop removed) without disturbing the user's existing arrangement.
  useEffect(() => {
    setOrder((prev) => {
      const live = videos.map((v) => v.id);
      const kept = prev.filter((id) => live.includes(id));
      const added = live.filter((id) => !kept.includes(id));
      const next = [...kept, ...added];
      const same = next.length === prev.length && next.every((id, i) => id === prev[i]);
      return same ? prev : next;
    });
  }, [videos]);

  const includedVideos = useMemo(
    () => videos.filter((v) => includedIds.has(v.id)),
    [videos, includedIds]
  );

  // Visible tiles in the user's chosen order.
  const orderedIncluded = useMemo(() => {
    const byId = new Map(videos.map((v) => [v.id, v]));
    return order
      .filter((id) => includedIds.has(id))
      .map((id) => byId.get(id))
      .filter((v): v is VideoFile => Boolean(v));
  }, [order, includedIds, videos]);

  const handleDragStart = useCallback((id: string) => setDragId(id), []);
  const handleDragOver = useCallback((id: string) => setOverId(id), []);
  const handleDragEnd = useCallback(() => {
    setDragId(null);
    setOverId(null);
  }, []);
  const handleDrop = useCallback((targetId: string) => {
    const draggedId = dragIdRef.current;
    setOverId(null);
    setDragId(null);
    if (!draggedId || draggedId === targetId) return;
    setOrder((prev) => {
      const next = prev.filter((id) => id !== draggedId);
      const ti = next.indexOf(targetId);
      if (ti < 0) return prev;
      next.splice(ti, 0, draggedId);
      return next;
    });
  }, []);

  // Global timeline length = furthest point any visible camera reaches.
  const globalDuration = useMemo(() => {
    let max = 0;
    for (const v of includedVideos) {
      const dur = meta[v.id]?.duration || 0;
      const off = offset[v.id] || 0;
      max = Math.max(max, off + dur);
    }
    return max;
  }, [includedVideos, meta, offset]);

  const includedKey = includedVideos.map((v) => v.id).join(",");

  // Park every tile EXACTLY on global time `g` (used for scrubbing / pausing).
  const applyG = useCallback((g: number, playing: boolean) => {
    for (const id of Object.keys(elRefs.current)) {
      const el = elRefs.current[id];
      if (!el) continue;
      const dur = metaRef.current[id]?.duration || el.duration || 0;
      const target = g - (offsetRef.current[id] || 0);
      const inRange = dur > 0 && target >= 0 && target <= dur;
      el.playbackRate = 1;
      if (inRange) {
        try {
          el.currentTime = target;
        } catch {
          /* not seekable yet */
        }
        if (playing && el.paused) void el.play().catch(() => undefined);
        if (!playing && !el.paused) el.pause();
      } else if (!el.paused) {
        el.pause();
      }
    }
  }, []);

  // Master-driven playback loop. Instead of a free-running wall clock (which
  // outruns buffering videos and triggers seek-storms), we read the master
  useEffect(() => {
    if (!isPlaying) return;
    if (globalDuration <= 0) return;
    const ids = includedVideos.map((v) => v.id);

    const isReadyInRange = (id: string, g: number): boolean => {
      const el = elRefs.current[id];
      const dur = metaRef.current[id]?.duration || el?.duration || 0;
      const target = g - (offsetRef.current[id] || 0);
      return Boolean(el) && dur > 0 && target >= 0 && target <= dur && (el as HTMLVideoElement).readyState >= 2;
    };

    // Keep the current master if it's still valid; else pick the in-range tile
    // with the furthest reach (so the clock survives longest before a switch).
    const pickMaster = (g: number): string | null => {
      const cur = masterIdRef.current;
      if (cur && ids.includes(cur) && isReadyInRange(cur, g)) return cur;
      let best: string | null = null;
      let bestReach = -1;
      for (const id of ids) {
        if (!isReadyInRange(id, g)) continue;
        const reach = (offsetRef.current[id] || 0) + (metaRef.current[id]?.duration || 0);
        if (reach > bestReach) {
          bestReach = reach;
          best = id;
        }
      }
      return best;
    };

    let raf = 0;
    let last = performance.now();
    const tick = () => {
      const now = performance.now();
      const dt = (now - last) / 1000;
      last = now;

      // Smoothness gate: if any in-range tile hasn't buffered enough to play
      // forward (readyState < HAVE_FUTURE_DATA), hold the shared clock - park the
      const gNow = gRef.current;
      const inRangeIds = ids.filter((id) => {
        const el = elRefs.current[id];
        const dur = metaRef.current[id]?.duration || el?.duration || 0;
        const t = gNow - (offsetRef.current[id] || 0);
        return Boolean(el) && dur > 0 && t >= 0 && t <= dur;
      });
      const stalled = inRangeIds.some((id) => {
        const el = elRefs.current[id] as HTMLVideoElement;
        // A tile that has errored can never become ready - don't let it deadlock
        // the whole wall; only genuinely-buffering tiles hold the clock.
        return !el.error && el.readyState < 3;
      });
      if (stalled && inRangeIds.length > 0) {
        for (const id of ids) {
          const el = elRefs.current[id];
          if (!el) continue;
          el.playbackRate = 1;
          if (el.readyState >= 3) {
            if (!el.paused) el.pause(); // ready -> wait for the others
          } else if (el.paused) {
            void el.play().catch(() => undefined); // not ready -> keep buffering
          }
        }
        raf = requestAnimationFrame(tick);
        return; // hold G steady until everyone is ready
      }

      let g = gRef.current;
      const masterId = pickMaster(g);
      masterIdRef.current = masterId;
      const master = masterId ? elRefs.current[masterId] : null;
      if (master) {
        if (master.paused) void master.play().catch(() => undefined);
        master.playbackRate = 1;
        // The clock IS the master's real position - it cannot outrun the video.
        g = master.currentTime + (offsetRef.current[masterId as string] || 0);
      } else {
        g = g + dt; // no ready master (gap/buffering) - advance gently
      }

      if (g >= globalDuration) {
        // Loop the review: reset everyone to their start.
        g = 0;
        gRef.current = 0;
        setGTime(0);
        for (const id of ids) {
          const el = elRefs.current[id];
          if (!el) continue;
          el.playbackRate = 1;
          try {
            el.currentTime = Math.max(0, -(offsetRef.current[id] || 0));
          } catch {
            /* not seekable yet */
          }
        }
        raf = requestAnimationFrame(tick);
        return;
      }

      gRef.current = g;
      setGTime(g);

      // Converge the slaves toward the master clock.
      for (const id of ids) {
        if (id === masterId) continue;
        const el = elRefs.current[id];
        if (!el) continue;
        const dur = metaRef.current[id]?.duration || el.duration || 0;
        const target = g - (offsetRef.current[id] || 0);
        const inRange = dur > 0 && target >= 0 && target <= dur;
        if (!inRange) {
          el.playbackRate = 1;
          if (!el.paused) el.pause();
          continue;
        }
        if (el.paused) void el.play().catch(() => undefined);
        const drift = el.currentTime - target; // + = ahead of master
        const ad = Math.abs(drift);
        if (ad > HARD_DRIFT) {
          try {
            el.currentTime = target; // far off -> snap (last resort)
          } catch {
            /* not seekable yet */
          }
          el.playbackRate = 1;
        } else if (ad > SOFT_DRIFT) {
          el.playbackRate = drift < 0 ? RATE_BEHIND : RATE_AHEAD; // gentle trim
        } else {
          el.playbackRate = 1;
        }
      }

      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // includedKey/globalDuration in deps so the loop picks up tile/length changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isPlaying, globalDuration, includedKey]);

  // Pausing must actually stop native playback (and clear any rate trim). The
  // master loop only advances time while playing; without this the <video>
  useEffect(() => {
    if (isPlaying) return;
    for (const id of Object.keys(elRefs.current)) {
      const el = elRefs.current[id];
      if (!el) continue;
      el.playbackRate = 1;
      if (!el.paused) el.pause();
    }
  }, [isPlaying]);

  // When paused, keep the videos parked on the scrubbed frame.
  const seekTo = useCallback(
    (g: number) => {
      const clamped = Math.min(Math.max(g, 0), globalDuration || 0);
      gRef.current = clamped;
      setGTime(clamped);
      applyG(clamped, isPlaying);
    },
    [applyG, globalDuration, isPlaying]
  );

  // External "jump to vehicle" - park the wall at the global time where the
  // requested camera shows `frame` (frame -> local time -> +offset), staying in
  const lastSeekTokenRef = useRef(0);
  useEffect(() => {
    if (!seekRequest || seekRequest.token === lastSeekTokenRef.current) return;
    lastSeekTokenRef.current = seekRequest.token;
    const { videoId, frame } = seekRequest;
    setIncludedIds((prev) => (prev.has(videoId) ? prev : new Set(prev).add(videoId)));
    const dur = metaRef.current[videoId]?.duration || 0;
    const maxFrame = detData[videoId]?.maxFrame || 0;
    const local =
      dur > 0 && maxFrame > 0 ? (Math.min(Math.max(frame, 0), maxFrame) / maxFrame) * dur : 0;
    const g = local + (offsetRef.current[videoId] || 0);
    setIsPlaying(false);
    gRef.current = g;
    setGTime(g);
    applyG(g, false);
  }, [seekRequest, detData, applyG]);

  const handleMeta = useCallback((id: string, m: VideoMeta) => {
    setMeta((prev) => ({ ...prev, [id]: m }));
  }, []);
  const nudgeOffset = useCallback((id: string, delta: number) => {
    setOffset((prev) => ({ ...prev, [id]: Math.round(((prev[id] || 0) + delta) * 10) / 10 }));
  }, []);
  const resetOffset = useCallback((id: string) => {
    setOffset((prev) => ({ ...prev, [id]: 0 }));
  }, []);
  const registerRef = useCallback((id: string, el: HTMLVideoElement | null) => {
    elRefs.current[id] = el;
  }, []);

  const toggleIncluded = useCallback((id: string) => {
    setIncludedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        if (next.size > 1) next.delete(id); // keep at least one tile
      } else {
        next.add(id);
      }
      return next;
    });
  }, []);

  // Per-camera detections + range flag for the current master time.
  const paneState = useMemo(() => {
    const out: Record<string, { dets: Detection[]; outOfRange: boolean }> = {};
    for (const v of includedVideos) {
      const m = meta[v.id];
      const d = detData[v.id];
      const dur = m?.duration || 0;
      const target = gTime - (offset[v.id] || 0);
      const oor = dur > 0 && (target < -0.05 || target > dur + 0.05);
      let dets: Detection[] = [];
      if (d && dur > 0 && !oor) {
        dets = d.byFrame.get(timeToFrame(target, dur, d.maxFrame)) ?? [];
      }
      out[v.id] = { dets, outOfRange: oor };
    }
    return out;
  }, [includedVideos, meta, detData, offset, gTime]);

  const masterTotal = Math.max(1, Math.round((globalDuration || 0) * MASTER_FPS));
  const masterFrame = Math.min(masterTotal - 1, Math.round(gTime * MASTER_FPS));

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* Tab strip - toggle which cameras are tiled (VSCode split-editor style) */}
      <div className="mb-2 flex shrink-0 flex-wrap items-center gap-1.5">
        <span className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
          Tiles
        </span>
        {videos.map((v) => {
          const on = includedIds.has(v.id);
          return (
            <button
              key={v.id}
              type="button"
              onClick={() => toggleIncluded(v.id)}
              className={cn(
                "rounded-md border px-2.5 py-1 text-xs font-medium transition-colors",
                on
                  ? "border-accent-strong bg-accent-strong/15 text-foreground"
                  : "border-border bg-background/40 text-muted-foreground hover:text-foreground"
              )}
              aria-pressed={on}
              title={on ? "Hide this camera" : "Show this camera"}
            >
              {v.cameraId || v.name}
            </button>
          );
        })}
        {loadingDet && (
          <span className="ml-1 flex items-center gap-1 text-[11px] text-muted-foreground">
            <Loader2 className="h-3 w-3 animate-spin" />
            loading overlays...
          </span>
        )}
      </div>

      {/* The grid of synced camera tiles */}
      <div className={cn("grid min-h-0 flex-1 gap-2", gridColsClass(orderedIncluded.length))}>
        {orderedIncluded.map((v) => {
          const m = meta[v.id];
          const ps = paneState[v.id] ?? { dets: [], outOfRange: false };
          return (
            <CameraPane
              key={v.id}
              video={v}
              registerRef={registerRef}
              detections={ps.dets}
              outOfRange={ps.outOfRange}
              videoW={m?.width || v.width || 1920}
              videoH={m?.height || v.height || 1080}
              offset={offset[v.id] || 0}
              onNudgeOffset={nudgeOffset}
              onResetOffset={resetOffset}
              onMeta={handleMeta}
              isOver={overId === v.id && dragId !== v.id}
              isDragging={dragId === v.id}
              onDragStart={handleDragStart}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              onDragEnd={handleDragEnd}
            />
          );
        })}
      </div>

      {/* One master scrubber drives every tile */}
      <PlaybackControls
        className="mt-3 shrink-0"
        isPlaying={isPlaying}
        currentFrame={masterFrame}
        totalFrames={masterTotal}
        speedOptions={[]}
        onPlayPause={() => setIsPlaying((p) => !p)}
        onFrameChange={(f) => seekTo(f / MASTER_FPS)}
        onStepBack={() => seekTo(0)}
        onStepForward={() => seekTo(globalDuration)}
      />
    </div>
  );
}
