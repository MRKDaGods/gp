"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Download,
  Play,
  Pause,
  Camera,
  Maximize2,
  Car,
  Truck,
  Bus,
  CheckCircle2,
  Loader2,
  TrendingUp,
  Route,
  Gauge,
} from "lucide-react";
import { getCameraColor, formatDuration } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Separator } from "@/components/ui/separator";
import { Label } from "@/components/ui/label";
import { usePipelineStore, useVideoStore } from "@/store";
import {
  exportTrajectories,
  generateSummaryVideo,
  getMatchedSummary,
  getTracklets,
  getTrajectories,
  getVideo,
} from "@/lib/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8004/api";
const outputPalette = ["#22c55e", "#3b82f6", "#f97316", "#e11d48", "#06b6d4", "#8b5cf6", "#f59e0b"];

interface OutputTrajectory {
  id: number;
  vehicleId: string;
  cameras: string[];
  duration: number;
  vehicleType: string;
  confidence: number;
  color: string;
}

function trajectoryFromGlobal(item: Record<string, unknown>, index: number): OutputTrajectory {
  const gid = Number(item.global_id ?? item.globalId ?? index + 1);

  let cameras: string[] = [];
  const camSeq = item.cameraSequence ?? item.camera_sequence;
  const visited = item.cameras_visited;
  if (Array.isArray(camSeq) && camSeq.length) {
    cameras = camSeq as string[];
  } else if (Array.isArray(visited) && visited.length) {
    cameras = visited as string[];
  } else if (Array.isArray(item.timeline) && (item.timeline as unknown[]).length) {
    cameras = Array.from(
      new Set(
        (item.timeline as { camera_id?: string; cameraId?: string }[]).map(
          (e) => e.cameraId ?? e.camera_id ?? ""
        )
      )
    ).filter(Boolean);
  } else if (Array.isArray(item.tracklets) && (item.tracklets as unknown[]).length) {
    cameras = Array.from(
      new Set(
        (item.tracklets as { camera_id?: string; cameraId?: string }[]).map(
          (t) => t.cameraId ?? t.camera_id ?? ""
        )
      )
    ).filter(Boolean);
  }

  const span = (item.timeSpan ?? item.time_span) as [number, number] | undefined;
  let duration = Number(item.totalDuration ?? item.total_duration ?? NaN);
  if (!Number.isFinite(duration) && span && span.length >= 2) {
    duration = Math.max(0, Number(span[1]) - Number(span[0]));
  }
  if (!Number.isFinite(duration) && Array.isArray(item.tracklets)) {
    const tl = item.tracklets as { start_time?: number; end_time?: number; startTime?: number; endTime?: number }[];
    const times = tl.flatMap((t) =>
      [t.start_time, t.end_time, t.startTime, t.endTime].filter((x): x is number => typeof x === "number")
    );
    if (times.length) duration = Math.max(...times) - Math.min(...times);
  }
  if (!Number.isFinite(duration)) duration = 0;

  const t0 = item.tracklets as { class_name?: string; className?: string }[] | undefined;
  const vehicleType = String(
    item.className ?? item.class ?? item.class_name ?? t0?.[0]?.className ?? t0?.[0]?.class_name ?? "sedan"
  ).toLowerCase();

  const confidence = Number(item.confidence ?? 0.8);

  return {
    id: gid,
    vehicleId: `G-${String(gid).padStart(4, "0")}`,
    cameras: Array.from(new Set(cameras)),
    duration,
    vehicleType,
    confidence: Number.isFinite(confidence) ? confidence : 0.8,
    color: outputPalette[index % outputPalette.length],
  };
}

function trajectoryFromTrackletSummary(item: any, index: number): OutputTrajectory {
  return {
    id: Number(item.id ?? index + 1),
    vehicleId: `T-${String(item.id ?? index + 1).padStart(4, "0")}`,
    cameras: [String(item.cameraId ?? item.camera_id ?? "unknown")],
    duration: Number(item.duration ?? 0),
    vehicleType: String(item.className ?? "sedan").toLowerCase(),
    confidence: Number(item.confidence ?? 0.8),
    color: outputPalette[index % outputPalette.length],
  };
}

function formatClock(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function OutputStage() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTimeSec, setCurrentTimeSec] = useState(0);
  const [durationSec, setDurationSec] = useState(0);
  const [videoLoadError, setVideoLoadError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [trajectories, setTrajectories] = useState<OutputTrajectory[]>([]);
  const [dataSource, setDataSource] = useState<"real" | "none">("none");
  const [error, setError] = useState<string | null>(null);
  const [exportFormat, setExportFormat] = useState<"mp4" | "json" | "csv">("mp4");
  const [isExporting, setIsExporting] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(2);
  const [hydratedLatestRunId, setHydratedLatestRunId] = useState<string | null>(null);
  const [outputFetchState, setOutputFetchState] = useState<"idle" | "loading" | "ready">("idle");
  const [summaryVideoUrl, setSummaryVideoUrl] = useState<string | null>(null);
  const [matchedSummary, setMatchedSummary] = useState<any>(null);

  const { runId } = usePipelineStore();
  const { currentVideo } = useVideoStore();

  const effectiveRunId = useMemo(
    () => runId ?? currentVideo?.latestRunId ?? hydratedLatestRunId ?? null,
    [runId, currentVideo?.latestRunId, hydratedLatestRunId]
  );

  const backendOrigin = API_BASE.endsWith("/api") ? API_BASE.slice(0, -4) : API_BASE;
  const toAbsoluteUrl = (url: string) => {
    if (url.startsWith("http://") || url.startsWith("https://")) return url;
    if (url.startsWith("/")) return `${backendOrigin}${url}`;
    return `${backendOrigin}/${url}`;
  };

  const rawStreamUrl = currentVideo ? `${API_BASE}/videos/stream/${currentVideo.id}` : null;
  const streamUrl = summaryVideoUrl ?? rawStreamUrl;

  // -- Playback helpers ---------------------------------------------------

  const playbackProgressPct = useMemo(() => {
    if (!durationSec || durationSec <= 0) return 0;
    return Math.min(100, Math.max(0, (currentTimeSec / durationSec) * 100));
  }, [currentTimeSec, durationSec]);

  const onVideoTimeUpdate = useCallback(() => {
    const v = videoRef.current;
    if (v) setCurrentTimeSec(v.currentTime);
  }, []);

  const onVideoLoaded = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    setDurationSec(Number.isFinite(v.duration) ? v.duration : 0);
    setVideoLoadError(null);
  }, []);

  const togglePlayback = useCallback(async () => {
    const v = videoRef.current;
    if (!v) return;
    try {
      if (v.paused) await v.play();
      else v.pause();
    } catch {
      setVideoLoadError("Playback was blocked or failed.");
    }
  }, []);

  const seekToPercent = useCallback(
    (pct: number) => {
      const v = videoRef.current;
      if (!v || !durationSec) return;
      const t = (pct / 100) * durationSec;
      v.currentTime = t;
      setCurrentTimeSec(t);
    },
    [durationSec]
  );

  useEffect(() => {
    setVideoLoadError(null);
    setCurrentTimeSec(0);
    setDurationSec(0);
  }, [streamUrl]);

  useEffect(() => {
    const v = videoRef.current;
    if (v) v.playbackRate = playbackSpeed;
  }, [playbackSpeed, streamUrl]);

  // -- Export handlers -----------------------------------------------------

  const handleDownloadVideo = async () => {
    if (isExporting) return;
    if (!effectiveRunId) {
      if (streamUrl) window.open(streamUrl, "_blank", "noopener,noreferrer");
      return;
    }
    try {
      setIsExporting(true);
      const response = await generateSummaryVideo(effectiveRunId);
      const videoUrl = response.data?.videoUrl;
      if (videoUrl) window.open(toAbsoluteUrl(videoUrl), "_blank", "noopener,noreferrer");
      else if (streamUrl) window.open(streamUrl, "_blank", "noopener,noreferrer");
    } finally {
      setIsExporting(false);
    }
  };

  const handleExportTracklets = async () => {
    if (!effectiveRunId || isExporting) return;
    const fmt = exportFormat === "mp4" ? "json" : exportFormat;
    try {
      setIsExporting(true);
      const response = await exportTrajectories(effectiveRunId, fmt);
      const downloadUrl = response.data?.downloadUrl;
      if (downloadUrl) window.open(toAbsoluteUrl(downloadUrl), "_blank", "noopener,noreferrer");
    } finally {
      setIsExporting(false);
    }
  };

  // -- Data loading --------------------------------------------------------

  useEffect(() => {
    const ac = new AbortController();

    const loadOutputData = async () => {
      if (!currentVideo) {
        setTrajectories([]);
        setSummaryVideoUrl(null);
        setMatchedSummary(null);
        setDataSource("none");
        setError(null);
        setHydratedLatestRunId(null);
        setOutputFetchState("idle");
        return;
      }

      setOutputFetchState("loading");
      setError(null);

      try {
        let eff = runId ?? currentVideo.latestRunId ?? null;
        if (!eff) {
          const res = await getVideo(currentVideo.id);
          if (ac.signal.aborted) return;
          eff = res.data?.latestRunId ?? null;
          setHydratedLatestRunId(eff);
        } else {
          setHydratedLatestRunId(null);
        }
        if (ac.signal.aborted) return;

        if (eff) {
          // Fire summary video, matched summary, and trajectories in parallel
          const [summaryRes, msRes, trajRes] = await Promise.allSettled([
            generateSummaryVideo(eff),
            getMatchedSummary(eff),
            getTrajectories(eff),
          ]);
          if (ac.signal.aborted) return;

          if (summaryRes.status === "fulfilled") {
            const url = summaryRes.value.data?.videoUrl;
            setSummaryVideoUrl(url ? toAbsoluteUrl(url) : null);
          } else {
            setSummaryVideoUrl(null);
          }

          if (msRes.status === "fulfilled") {
            setMatchedSummary(msRes.value);
          } else {
            setMatchedSummary(null);
          }

          if (trajRes.status === "fulfilled") {
            const raw = trajRes.value.data;
            const globalTrajectories = Array.isArray(raw) ? raw : [];
            if (globalTrajectories.length > 0) {
              setTrajectories(
                globalTrajectories.map((row, i) =>
                  trajectoryFromGlobal(row as unknown as Record<string, unknown>, i)
                )
              );
              setDataSource("real");
              return;
            }
          }
        }

        if (ac.signal.aborted) return;

        const trackletResponse = await getTracklets(undefined, currentVideo.id);
        if (ac.signal.aborted) return;
        const summary = Array.isArray(trackletResponse.data) ? trackletResponse.data : [];
        if (summary.length > 0) {
          setTrajectories(summary.map(trajectoryFromTrackletSummary));
          setDataSource("real");
          return;
        }
      } catch (err) {
        if (ac.signal.aborted) return;
        setError(String(err instanceof Error ? err.message : err || "Failed to load trajectory data"));
        setTrajectories([]);
        setDataSource("none");
        return;
      }

      if (!ac.signal.aborted) {
        setTrajectories([]);
        setDataSource("none");
      }
    };

    void loadOutputData().finally(() => {
      if (ac.signal.aborted) return;
      setOutputFetchState(currentVideo ? "ready" : "idle");
    });

    return () => { ac.abort(); };
  }, [currentVideo?.id, runId, currentVideo?.latestRunId]);

  // -- Derived stats -------------------------------------------------------

  const outputStats = useMemo(() => {
    const ms = matchedSummary;
    const hasMatched = ms && typeof ms.totalCameras === "number";

    const camerasFromTrajectories = new Set<string>();
    let crossCameraFromTrajectories = 0;
    trajectories.forEach((traj) => {
      traj.cameras.forEach((cam) => camerasFromTrajectories.add(cam));
      if (traj.cameras.length > 1) crossCameraFromTrajectories += 1;
    });

    const camerasAnalyzed = hasMatched
      ? Math.max(ms.totalCameras, camerasFromTrajectories.size)
      : camerasFromTrajectories.size;

    const uniqueVehicles = hasMatched
      ? Math.max(ms.totalMatchedTracklets ?? 0, trajectories.length)
      : trajectories.length;

    const crossCameraMatches = hasMatched
      ? ms.totalMatchedTrajectories ?? 0
      : crossCameraFromTrajectories;

    const clips = Array.isArray(ms?.clips) ? ms.clips : [];
    const clipConfidences = clips.map((c: any) => Number(c.confidence ?? 0)).filter((c: number) => c > 0);
    const avgConfidence = clipConfidences.length > 0
      ? clipConfidences.reduce((a: number, b: number) => a + b, 0) / clipConfidences.length
      : trajectories.length > 0
        ? trajectories.reduce((sum, traj) => sum + traj.confidence, 0) / trajectories.length
        : 0;

    return { camerasAnalyzed, uniqueVehicles, crossCameraMatches, avgConfidence };
  }, [trajectories, matchedSummary]);

  // -- Render --------------------------------------------------------------

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
      {/* Header */}
      <header className="flex shrink-0 flex-col gap-3 border-b px-4 py-3 sm:flex-row sm:items-center sm:justify-between sm:px-6">
        <div className="min-w-0">
          <h1 className="text-lg font-semibold">Results & Export</h1>
          <p className="text-sm text-muted-foreground">
            Multi-camera tracking summary
          </p>
        </div>
        <div className="flex flex-wrap items-center justify-end gap-2">
          {outputFetchState === "loading" && (
            <Badge variant="secondary" className="gap-1.5">
              <Loader2 className="h-3 w-3 animate-spin" aria-hidden />
              Loading…
            </Badge>
          )}
          {dataSource === "real" && outputFetchState === "ready" && (
            <Badge variant="outline" className="bg-green-500/10 text-green-500 border-green-500/30">
              <CheckCircle2 className="h-3 w-3 mr-1" />
              Ready
            </Badge>
          )}
        </div>
      </header>

      {/* Error banner */}
      {error && (
        <div className="flex shrink-0 items-start gap-3 border-b border-destructive/30 bg-destructive/10 px-4 py-3 sm:px-6">
          <Route className="mt-0.5 h-5 w-5 shrink-0 text-destructive" />
          <div className="min-w-0">
            <p className="text-sm font-medium text-destructive">Failed to load output data</p>
            <p className="break-words text-xs text-muted-foreground">{error}</p>
          </div>
        </div>
      )}

      {/* Two-panel content */}
      <div className="flex min-h-0 flex-1 flex-col overflow-hidden lg:flex-row">

        {/* Left panel: video + trajectory list */}
        <div className="min-w-0 flex-1 overflow-x-hidden overflow-y-auto p-4">
          {/* Video player */}
          <div
            className="relative w-full overflow-hidden rounded-lg border border-border bg-slate-900"
            style={{ height: "calc(100vh - 10rem)" }}
          >
            {streamUrl ? (
              <video
                ref={videoRef}
                key={streamUrl}
                src={streamUrl}
                className="pointer-events-none absolute inset-0 h-full w-full object-contain"
                autoPlay
                muted
                loop
                playsInline
                controls={false}
                onTimeUpdate={onVideoTimeUpdate}
                onLoadedMetadata={onVideoLoaded}
                onPlay={() => setIsPlaying(true)}
                onPause={() => setIsPlaying(false)}
                onError={() =>
                  setVideoLoadError("Could not load video stream.")
                }
              />
            ) : (
              <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-b from-slate-700 via-slate-800 to-slate-900">
                <p className="text-sm text-white/40">No video available</p>
              </div>
            )}

            {videoLoadError && (
              <div className="absolute bottom-16 left-3 right-3 rounded-md bg-red-950/90 px-3 py-2 text-center text-xs text-red-100">
                {videoLoadError}
              </div>
            )}

            <div className="pointer-events-none absolute top-3 right-3 rounded bg-black/60 px-2 py-1">
              <span className="text-xs font-mono text-white/80">
                {outputStats.uniqueVehicles} Tracked
              </span>
            </div>

            {/* Controls */}
            <div className="pointer-events-auto absolute bottom-0 left-0 right-0 z-10 bg-gradient-to-t from-black/80 to-transparent p-3">
              <div className="flex items-center gap-3">
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-white hover:text-white hover:bg-white/20"
                  onClick={() => void togglePlayback()}
                >
                  {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                </Button>
                <Slider
                  className="flex-1"
                  value={[playbackProgressPct]}
                  max={100}
                  step={0.1}
                  onValueChange={(v) => seekToPercent(v[0])}
                  disabled={!durationSec}
                />
                <span className="shrink-0 text-xs font-mono tabular-nums text-white">
                  {formatClock(currentTimeSec)} / {formatClock(durationSec)}
                </span>
                <span className="shrink-0 rounded bg-white/10 px-1.5 py-0.5 text-[10px] font-mono text-white/70">
                  {playbackSpeed}x
                </span>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-white hover:text-white hover:bg-white/20"
                  onClick={() => {
                    const v = videoRef.current;
                    if (!v) return;
                    if (!document.fullscreenElement) void v.parentElement?.requestFullscreen?.();
                    else void document.exitFullscreen();
                  }}
                >
                  <Maximize2 className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </div>

        </div>

        {/* Right sidebar */}
        <aside className="w-full shrink-0 overflow-y-auto border-t border-border bg-muted/20 p-4 lg:w-80 lg:border-l lg:border-t-0">

          {/* Quick stats */}
          <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Summary
          </h3>
          <div className="grid grid-cols-2 gap-3">
            <StatRow icon={Camera} label="Cameras" value={String(outputStats.camerasAnalyzed)} />
            <StatRow icon={Car} label="Vehicles" value={String(outputStats.uniqueVehicles)} />
            <StatRow icon={TrendingUp} label="Cross-cam" value={String(outputStats.crossCameraMatches)} />
            <StatRow icon={Gauge} label="Avg. Conf." value={`${(outputStats.avgConfidence * 100).toFixed(0)}%`} />
          </div>

          <Separator className="my-5" />

          {/* Playback speed */}
          <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Playback Speed — {playbackSpeed}x
          </h3>
          <Slider
            value={[playbackSpeed]}
            min={1}
            max={16}
            step={1}
            onValueChange={(v) => setPlaybackSpeed(v[0])}
          />

          <Separator className="my-5" />

          {/* Export */}
          <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Export
          </h3>
          <div className="space-y-3">
            <div className="space-y-1.5">
              <Label className="text-xs">Format</Label>
              <select
                className="w-full rounded border bg-background px-3 py-2 text-sm"
                value={exportFormat}
                onChange={(e) => setExportFormat(e.target.value as "mp4" | "json" | "csv")}
              >
                <option value="mp4">MP4 Video</option>
                <option value="json">JSON Tracklets</option>
                <option value="csv">CSV Export</option>
              </select>
            </div>
            <Button className="w-full" onClick={handleDownloadVideo} disabled={isExporting}>
              <Download className="mr-2 h-4 w-4" />
              Download Video
            </Button>
            <Button
              variant="outline"
              className="w-full"
              onClick={handleExportTracklets}
              disabled={!effectiveRunId || isExporting || outputFetchState === "loading"}
            >
              <Download className="mr-2 h-4 w-4" />
              Export Tracklets
            </Button>
          </div>
        </aside>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function TrajectoryItem({ trajectory }: { trajectory: OutputTrajectory }) {
  const vt = trajectory.vehicleType;
  const Icon = vt.includes("truck") ? Truck : vt.includes("bus") ? Bus : Car;

  return (
    <div className="flex items-center gap-3 rounded-lg bg-muted/50 p-3 transition-colors hover:bg-muted">
      <div
        className="flex h-9 w-9 items-center justify-center rounded"
        style={{ backgroundColor: `${trajectory.color}20`, color: trajectory.color }}
      >
        <Icon className="h-4 w-4" />
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">{trajectory.vehicleId}</span>
          <Badge variant="secondary" className="text-[10px]">
            {(trajectory.confidence * 100).toFixed(0)}%
          </Badge>
        </div>
        <div className="mt-0.5 flex items-center gap-1">
          {trajectory.cameras.map((cam, i) => (
            <div key={`${cam}-${i}`} className="flex items-center">
              <div className="h-2 w-2 rounded-full" style={{ backgroundColor: getCameraColor(cam) }} />
              {i < trajectory.cameras.length - 1 && <span className="mx-0.5 text-muted-foreground">→</span>}
            </div>
          ))}
          <span className="ml-1.5 text-[10px] text-muted-foreground">{trajectory.cameras.length} cam</span>
        </div>
      </div>
      <span className="shrink-0 text-xs font-mono text-muted-foreground">{formatDuration(trajectory.duration)}</span>
    </div>
  );
}

function StatRow({ icon: Icon, label, value }: { icon: React.ElementType; label: string; value: string }) {
  return (
    <div className="flex items-center gap-2.5 rounded-md bg-muted/40 px-3 py-2">
      <Icon className="h-4 w-4 shrink-0 text-primary" />
      <div className="min-w-0">
        <p className="text-lg font-bold leading-tight">{value}</p>
        <p className="text-[10px] text-muted-foreground">{label}</p>
      </div>
    </div>
  );
}
