"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  ZoomIn,
  ZoomOut,
  Check,
  X,
  ChevronDown,
  Layers,
  Camera,
  ArrowRight,
  Car,
} from "lucide-react";
import { cn, formatDuration, getCameraColor } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  useTimelineStore,
  usePipelineStore,
  useSessionStore,
  useVideoStore,
  useDetectionStore,
} from "@/store";
import { getTracklets, getTrajectories, runStage, getPipelineStatus } from "@/lib/api";
import type { TimelineTrack } from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";


export function TimelineStage() {
  const {
    tracks,
    setTracks,
    zoom,
    setZoom,
    selectedTrackId,
    selectTrack,
    confirmTrack,
    unconfirmTrack,
    removeTrack,
  } = useTimelineStore();
  const { runId, updateStageProgress, stages } = usePipelineStore();
  const { setCurrentStage } = useSessionStore();
  const { currentVideo } = useVideoStore();
  const { selectedIds } = useDetectionStore();

  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [splitCount, setSplitCount] = useState(6);
  const [showProgress, setShowProgress] = useState(true);
  const timelineRef = useRef<HTMLDivElement>(null);

  const inferredDuration =
    tracks.length > 0
      ? Math.max(1, ...tracks.map((track) => track.endTime + 2))
      : Math.max(1, currentVideo?.duration ?? 0);

  const totalDuration = Math.max(inferredDuration, 1);

  const camerasForPreview = (() => {
    const cameraIds = Array.from(new Set(tracks.map((track) => track.cameraId)));
    if (cameraIds.length === 0) return [];

    return cameraIds.map((cameraId) => ({
      id: cameraId,
      scene: cameraId.split("_")[0] ?? "Unknown",
      name: cameraId,
      location: "Camera",
    }));
  })();

  const buildTracksFromSummary = (summary: any[]): TimelineTrack[] => {
    if (!currentVideo || summary.length === 0) return [];
    const fps = Math.max(currentVideo.fps || 10, 1);

    return summary.map((item: any, index: number) => {
      const startFrame = Number(item.startFrame ?? 0);
      const endFrame = Number(item.endFrame ?? startFrame);
      const startTime = startFrame / fps;
      const endTime = endFrame / fps;

      return {
        id: `real-${item.cameraId}-${item.id}-${index}`,
        cameraId: String(item.cameraId ?? "unknown"),
        trackletId: Number(item.id ?? index),
        startTime,
        endTime: Math.max(endTime, startTime + 0.1),
        selected: false,
        confirmed: index === 0,
        representativeFrame: item.representativeFrame,
        representativeBbox: item.representativeBbox,
        sampleFrames: item.sampleFrames,
      };
    });
  };

  const buildTracksFromTrajectories = (trajectories: any[]): TimelineTrack[] => {
    if (!Array.isArray(trajectories) || trajectories.length === 0) return [];

    // Filter to only trajectories that include a tracklet selected in stage 2
    const filtered = selectedIds.size > 0
      ? trajectories.filter((traj: any) => {
          const tracklets = Array.isArray(traj.tracklets) ? traj.tracklets : [];
          return tracklets.some((t: any) => selectedIds.has(String(t.track_id ?? t.trackId)));
        })
      : trajectories;

    const rows: TimelineTrack[] = [];
    filtered.forEach((trajectory: any, trajectoryIndex: number) => {
      const globalId = Number(trajectory.globalId ?? trajectory.global_id ?? trajectoryIndex + 1);
      // Support both camelCase and snake_case field names from the backend
      const timeline = Array.isArray(trajectory.timeline) ? trajectory.timeline : [];
      const tracklets: any[] = Array.isArray(trajectory.tracklets) ? trajectory.tracklets : [];

      timeline.forEach((entry: any, entryIndex: number) => {
        // Backend returns snake_case: camera_id, track_id
        const cameraId = String(entry.camera_id ?? entry.cameraId ?? "unknown");
        const trackId = entry.track_id ?? entry.trackId;
        const startTime = Number(entry.start ?? 0);
        const endTime = Number(entry.end ?? startTime + 0.1);

        // Pull representative frame + bbox from the matching tracklet's frames
        let representativeFrame: number | undefined;
        let representativeBbox: number[] | undefined;
        const matchedTracklet = tracklets.find(
          (t: any) =>
            (t.track_id ?? t.trackId) === trackId &&
            (t.camera_id ?? t.cameraId) === cameraId
        );
        if (matchedTracklet) {
          const frames: any[] = Array.isArray(matchedTracklet.frames) ? matchedTracklet.frames : [];
          const midFrame = frames[Math.floor(frames.length / 2)];
          if (midFrame) {
            representativeFrame = Number(midFrame.frame_id ?? midFrame.frameId ?? 0);
            representativeBbox = midFrame.bbox;
          }
        }

        rows.push({
          id: `traj-${globalId}-${cameraId}-${entryIndex}`,
          cameraId,
          trackletId: globalId,
          globalId,
          startTime,
          endTime: Math.max(endTime, startTime + 0.1),
          selected: false,
          confirmed: true,
          representativeFrame,
          representativeBbox,
        });
      });
    });

    return rows;
  };

  // Stage 4: run real association, then load tracks
  useEffect(() => {
    let cancelled = false;

    const loadTracks = async () => {
      if (!currentVideo) return;

      try {
        // If we already have stage4 trajectory artifacts, load them directly
        if (runId) {
          const trajectoryResponse = await getTrajectories(runId);
          if (cancelled) return;

          const trajectoryRows = buildTracksFromTrajectories(
            Array.isArray(trajectoryResponse.data) ? trajectoryResponse.data : []
          );
          if (trajectoryRows.length > 0) {
            setTracks(trajectoryRows);
            updateStageProgress(4, { status: "completed", progress: 100, message: "Association loaded" });
            return;
          }

          // No stage4 artifacts yet — run stage 4 now
          updateStageProgress(4, { status: "running", progress: 5, message: "Running cross-camera association..." });
          const stageResp = await runStage(4, { runId, videoId: currentVideo.id });
          if (cancelled) return;
          const stage4RunId = (stageResp.data as any)?.runId ?? runId;

          // Poll until done
          let done = false;
          while (!done && !cancelled) {
            await new Promise((r) => setTimeout(r, 1500));
            if (cancelled) return;
            const statusResp = await getPipelineStatus(stage4RunId);
            if (cancelled) return;
            const statusData: any = statusResp.data;
            const status = statusData?.status;
            const progress = Number(statusData?.progress ?? 0);
            const message = String(statusData?.message ?? "Running...");
            updateStageProgress(4, { progress, message });
            if (status === "completed" || status === "error") done = true;
            if (status === "error") {
              updateStageProgress(4, { status: "error", message: String(statusData?.error ?? "Stage 4 failed") });
              // Fall through to load stage1 tracklets
              break;
            }
          }

          if (!cancelled) {
            const traj2 = await getTrajectories(stage4RunId);
            if (cancelled) return;
            const rows2 = buildTracksFromTrajectories(Array.isArray(traj2.data) ? traj2.data : []);
            if (rows2.length > 0) {
              setTracks(rows2);
              updateStageProgress(4, { status: "completed", progress: 100, message: "Association complete" });
              return;
            }
          }
        }

        // Fall back to stage1 tracklets (single-camera view)
        const response = await getTracklets(undefined, currentVideo.id);
        if (cancelled) return;
        let summary = Array.isArray(response.data) ? response.data : [];

        // Filter to only selected tracklets from Stage 2
        if (selectedIds.size > 0) {
          summary = summary.filter((item: any) => selectedIds.has(String(item.id)));
        }
        const realTracks = buildTracksFromSummary(summary);
        if (realTracks.length > 0) {
          setTracks(realTracks);
          updateStageProgress(4, { status: "completed", progress: 100, message: "Showing stage 1 tracklets" });
        }
      } catch (err) {
        if (!cancelled) {
          updateStageProgress(4, { status: "error", progress: 0, message: String(err) });
        }
      }
    };

    void loadTracks();

    return () => {
      cancelled = true;
    };
  }, [currentVideo, runId, selectedIds, setTracks]);

  // Playback simulation
  useEffect(() => {
    if (!isPlaying) return;
    const interval = setInterval(() => {
      setCurrentTime((t) => (t >= totalDuration ? 0 : t + 0.5));
    }, 500);
    return () => clearInterval(interval);
  }, [isPlaying, totalDuration]);

  useEffect(() => {
    setCurrentTime((t) => Math.min(t, totalDuration));
  }, [totalDuration]);


  const stage4Progress = stages.find((s) => s.stage === 4);

  const timeToPixel = useCallback(
    (time: number) => {
      const baseWidth = 1200;
      return (time / totalDuration) * baseWidth * zoom;
    },
    [zoom]
  );

  const handleTrackClick = (trackId: string) => {
    selectTrack(trackId === selectedTrackId ? null : trackId);
  };

  const handleConfirmToggle = (trackId: string, isConfirmed: boolean) => {
    if (isConfirmed) {
      unconfirmTrack(trackId);
    } else {
      confirmTrack(trackId);
    }
  };

  const handleProceed = () => {
    setCurrentStage(5);
  };

  const confirmedCount = tracks.filter((t) => t.confirmed).length;
  const timelineDataSource = tracks.some((t) => t.id.startsWith("real-") || t.id.startsWith("traj-")) ? "real" : "demo";

  const visibleCameras = camerasForPreview.slice(0, splitCount);
  const activeCamerasForGrid = visibleCameras.map((cam) => {
    const activeTrack = tracks.find(
      (t) => t.cameraId === cam.id && currentTime >= t.startTime && currentTime <= t.endTime
    );
    return { ...cam, activeTrack };
  });

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <header className="flex h-14 items-center justify-between border-b px-6">
        <div>
          <h1 className="text-lg font-semibold">Stage 4: Cross-Camera Timeline</h1>
          <p className="text-sm text-muted-foreground">
            DeepOCSORT tracklet association across CityFlow cameras
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant={timelineDataSource === "real" ? "secondary" : "destructive"}>
            {timelineDataSource === "real" ? "Real Artifacts" : "No Data"}
          </Badge>
          <Badge variant="secondary">{tracks.length} tracklets</Badge>
          <Badge variant="outline" className="bg-green-500/10 text-green-500 border-green-500/30">
            {confirmedCount} confirmed
          </Badge>
          <Button onClick={handleProceed}>
            Continue to Refinement
            <ArrowRight className="ml-2 h-4 w-4" />
          </Button>
        </div>
      </header>

      {/* Main content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left panel */}
        <aside className="w-64 border-r flex flex-col bg-muted/20">
          <div className="p-4 border-b">
            <h3 className="font-semibold flex items-center gap-2">
              <Layers className="h-4 w-4" />
              Association Progress
            </h3>
          </div>
          <div className="p-4">
            {stage4Progress?.status === "running" ? (
              <div className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span className="text-xs">{stage4Progress.message}</span>
                  <span className="font-mono">{stage4Progress.progress}%</span>
                </div>
                <Progress value={stage4Progress.progress} className="h-2" />
              </div>
            ) : (
              <div className="flex items-center gap-2 text-green-500">
                <Check className="h-4 w-4" />
                <span className="text-sm">Association complete</span>
              </div>
            )}
          </div>

          <Separator />

          {/* Split screen control */}
          <div className="p-4">
            <h4 className="text-sm font-medium mb-3">Camera Grid</h4>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Cameras</span>
                <span className="text-sm font-medium">{splitCount}</span>
              </div>
              <Slider
                value={[splitCount]}
                min={1}
                max={6}
                step={1}
                onValueChange={(v) => setSplitCount(v[0])}
              />
            </div>
          </div>

          <Separator />

          {/* Tracklet list */}
          <div className="flex-1 overflow-auto p-4">
            <h4 className="text-sm font-medium mb-3">Tracklets</h4>
            <div className="space-y-2">
              {tracks.map((track) => (
                <TrackletItem
                  key={track.id}
                  track={track}
                  isSelected={selectedTrackId === track.id}
                  onClick={() => handleTrackClick(track.id)}
                  onConfirm={() => handleConfirmToggle(track.id, track.confirmed)}
                  onRemove={() => removeTrack(track.id)}
                />
              ))}
            </div>
          </div>
        </aside>

        {/* Main timeline area */}
        <div className="flex-1 flex flex-col">
          {/* Video preview area - CityFlow camera grid */}
          <div className="h-56 border-b bg-slate-900 p-2">
            <div
              className="grid gap-1 h-full"
              style={{
                gridTemplateColumns: `repeat(${Math.min(splitCount, 3)}, 1fr)`,
                gridTemplateRows: `repeat(${Math.ceil(splitCount / 3)}, 1fr)`,
              }}
            >
              {activeCamerasForGrid.map((cam) => (
                <CameraPreview
                  key={cam.id}
                  camera={cam}
                  isActive={!!cam.activeTrack}
                  currentTime={currentTime}
                  videoId={currentVideo?.id}
                  runId={runId ?? undefined}
                />
              ))}
            </div>
          </div>

          {/* Timeline controls */}
          <div className="flex items-center gap-4 p-4 border-b bg-muted/30">
            <div className="flex items-center gap-1">
              <Button variant="ghost" size="icon" onClick={() => setCurrentTime(0)}>
                <SkipBack className="h-4 w-4" />
              </Button>
              <Button
                variant="default"
                size="icon"
                onClick={() => setIsPlaying(!isPlaying)}
              >
                {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              </Button>
              <Button variant="ghost" size="icon" onClick={() => setCurrentTime(totalDuration)}>
                <SkipForward className="h-4 w-4" />
              </Button>
            </div>

            <div className="text-sm font-mono text-muted-foreground">
              {formatDuration(currentTime)} / {formatDuration(totalDuration)}
            </div>

            <div className="flex-1">
              <Slider
                value={[currentTime]}
                max={totalDuration}
                step={0.5}
                onValueChange={(v) => setCurrentTime(v[0])}
              />
            </div>

            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setZoom(Math.max(0.5, zoom - 0.25))}
              >
                <ZoomOut className="h-4 w-4" />
              </Button>
              <span className="text-sm w-14 text-center font-mono">{(zoom * 100).toFixed(0)}%</span>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setZoom(Math.min(4, zoom + 0.25))}
              >
                <ZoomIn className="h-4 w-4" />
              </Button>
            </div>
          </div>

          {/* Timeline tracks */}
          <div className="flex-1 overflow-hidden bg-muted/10">
            <ScrollArea className="h-full">
              <div ref={timelineRef} className="p-4 min-w-max">
                {/* Time ruler */}
                <div className="h-6 mb-2 relative border-b border-muted">
                  {Array.from({ length: Math.ceil(totalDuration / 10) + 1 }).map((_, i) => (
                    <div
                      key={i}
                      className="absolute flex flex-col items-center"
                      style={{ left: timeToPixel(i * 10) }}
                    >
                      <div className="h-3 w-px bg-muted-foreground/30" />
                      <span className="text-[10px] text-muted-foreground font-mono">
                        {formatDuration(i * 10)}
                      </span>
                    </div>
                  ))}
                  {/* Playhead on ruler */}
                  <div
                    className="absolute top-0 h-3 w-0.5 bg-red-500"
                    style={{ left: timeToPixel(currentTime) }}
                  />
                </div>

                {/* Track rows */}
                <div className="space-y-1">
                  {tracks.map((track) => (
                    <TimelineRow
                      key={track.id}
                      track={track}
                      totalDuration={totalDuration}
                      zoom={zoom}
                      isSelected={selectedTrackId === track.id}
                      onClick={() => handleTrackClick(track.id)}
                      timeToPixel={timeToPixel}
                      currentTime={currentTime}
                      videoId={currentVideo?.id}
                      runId={runId ?? undefined}
                    />
                  ))}
                </div>
              </div>
              <ScrollBar orientation="horizontal" />
            </ScrollArea>
          </div>
        </div>
      </div>
    </div>
  );
}

function CameraPreview({
  camera,
  isActive,
  currentTime,
  videoId,
  runId,
}: {
  camera: { id: string; name: string; location: string; activeTrack?: TimelineTrack };
  isActive: boolean;
  currentTime: number;
  videoId?: string;
  runId?: string;
}) {
  // Build a crop URL from the active track's representative frame
  const cropUrl = (() => {
    if (!isActive || !camera.activeTrack) return null;
    const t = camera.activeTrack;
    const bbox = t.representativeBbox;
    const frameId = t.representativeFrame;
    if (!bbox || bbox.length !== 4 || frameId == null) return null;
    const bboxParams = `x1=${bbox[0]}&y1=${bbox[1]}&x2=${bbox[2]}&y2=${bbox[3]}`;
    if (runId) {
      return `${API_BASE}/crops/run/${runId}?cameraId=${t.cameraId}&frameId=${frameId}&${bboxParams}`;
    }
    if (videoId) {
      return `${API_BASE}/crops/${videoId}?frameId=${frameId}&${bboxParams}`;
    }
    return null;
  })();

  return (
    <div className={cn(
      "relative rounded overflow-hidden transition-all",
      isActive ? "ring-2 ring-green-500" : "opacity-60"
    )}>
      {/* Camera feed — actual crop or placeholder */}
      <div className="absolute inset-0 bg-gradient-to-b from-slate-700 via-slate-800 to-slate-900">
        {cropUrl ? (
          <img
            src={cropUrl}
            alt={camera.id}
            className="absolute inset-0 w-full h-full object-cover"
          />
        ) : (
          <>
            <svg className="absolute inset-0 w-full h-full opacity-20" preserveAspectRatio="none">
              <line x1="50%" y1="30%" x2="20%" y2="100%" stroke="white" strokeWidth="1" />
              <line x1="50%" y1="30%" x2="80%" y2="100%" stroke="white" strokeWidth="1" />
            </svg>
            {isActive && camera.activeTrack && (
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">
                <div
                  className="w-10 h-6 rounded border-2 flex items-center justify-center"
                  style={{ borderColor: (camera.activeTrack as any).color || "#22c55e", backgroundColor: `${(camera.activeTrack as any).color || "#22c55e"}33` }}
                >
                  <Car className="h-4 w-4 text-white" />
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {/* Camera info overlay */}
      <div className="absolute top-0 left-0 right-0 p-1 bg-black/60">
        <div className="flex items-center gap-1">
          <div className={cn("h-1.5 w-1.5 rounded-full", isActive ? "bg-green-500 animate-pulse" : "bg-gray-500")} />
          <span className="text-white text-[10px] font-mono">{camera.id}</span>
        </div>
      </div>

      {/* Timestamp */}
      <div className="absolute bottom-0 left-0 right-0 p-1 bg-black/60">
        <span className="text-white/70 text-[9px] font-mono">
          {formatDuration(currentTime)}
        </span>
      </div>
    </div>
  );
}

interface TrackletItemProps {
  track: TimelineTrack;
  isSelected: boolean;
  onClick: () => void;
  onConfirm: () => void;
  onRemove: () => void;
}

function TrackletItem({ track, isSelected, onClick, onConfirm, onRemove }: TrackletItemProps) {
  const trackColor = (track as any).color || getCameraColor(track.cameraId);

  return (
    <div
      className={cn(
        "p-2 rounded-lg border cursor-pointer transition-all",
        isSelected && "border-primary bg-primary/5",
        track.confirmed && !isSelected && "border-green-500/50 bg-green-500/5"
      )}
      onClick={onClick}
    >
      <div className="flex items-center gap-2">
        <div
          className="h-3 w-3 rounded-full flex-shrink-0"
          style={{ backgroundColor: trackColor }}
        />
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium truncate">{track.cameraId}</p>
          <p className="text-[10px] text-muted-foreground">
            {formatDuration(track.startTime)} - {formatDuration(track.endTime)}
          </p>
        </div>
        <div className="flex gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6"
            onClick={(e) => {
              e.stopPropagation();
              onConfirm();
            }}
          >
            <Check className={cn("h-3 w-3", track.confirmed ? "text-green-500" : "text-muted-foreground")} />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6"
            onClick={(e) => {
              e.stopPropagation();
              onRemove();
            }}
          >
            <X className="h-3 w-3 text-muted-foreground hover:text-red-500" />
          </Button>
        </div>
      </div>
    </div>
  );
}

interface TimelineRowProps {
  track: TimelineTrack;
  totalDuration: number;
  zoom: number;
  isSelected: boolean;
  onClick: () => void;
  timeToPixel: (time: number) => number;
  currentTime: number;
  videoId?: string;
  runId?: string;
}

function TimelineRow({ track, totalDuration, zoom, isSelected, onClick, timeToPixel, currentTime, videoId, runId }: TimelineRowProps) {
  const isCurrentlyActive = currentTime >= track.startTime && currentTime <= track.endTime;
  const trackColor = (track as any).color || getCameraColor(track.cameraId);

  // Build crop URL for the representative frame
  const cropUrl = (() => {
    const bbox = track.representativeBbox;
    const frameId = track.representativeFrame;
    if (!bbox || bbox.length !== 4 || frameId == null) return null;
    const bboxParams = `x1=${bbox[0]}&y1=${bbox[1]}&x2=${bbox[2]}&y2=${bbox[3]}`;
    if (runId) {
      return `${API_BASE}/crops/run/${runId}?cameraId=${track.cameraId}&frameId=${frameId}&${bboxParams}`;
    }
    if (videoId) {
      return `${API_BASE}/crops/${videoId}?frameId=${frameId}&${bboxParams}`;
    }
    return null;
  })();

  return (
    <div
      className={cn(
        "h-10 relative rounded border bg-background/50 cursor-pointer transition-all",
        isSelected && "border-primary ring-1 ring-primary",
        track.confirmed && !isSelected && "border-green-500/30",
        isCurrentlyActive && "bg-primary/5"
      )}
      style={{ width: timeToPixel(totalDuration) }}
      onClick={onClick}
    >
      {/* Camera label */}
      <div className="absolute left-2 top-1/2 -translate-y-1/2 flex items-center gap-2 z-10">
        <div className="h-2 w-2 rounded-full" style={{ backgroundColor: trackColor }} />
        <span className="text-[10px] font-mono text-muted-foreground">{track.cameraId}</span>
      </div>

      {/* Track clip */}
      <div
        className={cn(
          "absolute top-1 bottom-1 rounded transition-all overflow-hidden",
          track.confirmed && "ring-1 ring-green-500/50",
          isCurrentlyActive && "ring-2 ring-white/50"
        )}
        style={{
          left: timeToPixel(track.startTime),
          width: Math.max(timeToPixel(track.endTime - track.startTime), 20),
          backgroundColor: trackColor,
        }}
      >
        {cropUrl ? (
          <img
            src={cropUrl}
            alt={`Track ${track.trackletId}`}
            className="absolute inset-0 w-full h-full object-cover opacity-80"
            loading="lazy"
          />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center">
            <Car className="h-4 w-4 text-white/80" />
          </div>
        )}

        {/* Alternatives dropdown */}
        {track.alternatives && track.alternatives.length > 0 && (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="absolute right-0 top-1/2 -translate-y-1/2 h-6 w-6 bg-black/30 hover:bg-black/50"
                onClick={(e) => e.stopPropagation()}
              >
                <ChevronDown className="h-3 w-3 text-white" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent>
              {track.alternatives.map((alt) => (
                <DropdownMenuItem key={`${alt.cameraId}-${alt.trackletId}`}>
                  <div className="flex items-center gap-2">
                    <div className="h-2 w-2 rounded-full" style={{ backgroundColor: getCameraColor(alt.cameraId) }} />
                    <span className="text-sm">{alt.cameraId}</span>
                    <Badge variant="secondary" className="ml-auto text-[10px]">
                      {(alt.similarity * 100).toFixed(0)}%
                    </Badge>
                  </div>
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        )}
      </div>

      {/* Playhead indicator on this row */}
      {isCurrentlyActive && (
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-red-500 z-20"
          style={{ left: timeToPixel(currentTime) }}
        />
      )}
    </div>
  );
}
