"use client";

import { useState, useMemo, useEffect } from "react";
import {
  Image,
  X,
  Check,
  RefreshCw,
  Search,
  ChevronLeft,
  ChevronRight,
  ArrowRight,
  Play,
  Pause,
} from "lucide-react";
import { cn, getCameraColor, formatDuration } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Slider } from "@/components/ui/slider";
import { Separator } from "@/components/ui/separator";
import {
  useTimelineStore,
  useSessionStore,
  usePipelineStore,
  useVideoStore,
} from "@/store";
import {
  getTrackletSequence,
  getRunFullFrameUrl,
  getMatchedAlternatives,
} from "@/lib/api";
import { TrackletFrameView } from "@/components/ui/double-buffered-img";
import type { TimelineTrack } from "@/types";



export function RefinementStage() {
  const { tracks, setTracks } = useTimelineStore();
  const { runId, galleryRunId } = usePipelineStore();
  const { currentVideo } = useVideoStore();
  const {
    refinementFrames,
    addRefinementFrame,
    removeRefinementFrame,
    clearRefinementFrames,
    setCurrentStage,
  } = useSessionStore();

  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [isPlaying, setIsPlaying] = useState(false);
  const [framesLoading, setFramesLoading] = useState(false);
  const [reSearchRunning, setReSearchRunning] = useState(false);
  const [reSearchStatus, setReSearchStatus] = useState<string | null>(null);
  const [refinementCandidateFrames, setRefinementCandidateFrames] = useState<Array<{
    id: string;
    frameId: number;
    timestamp: number;
    cameraId: string;
    trackId: number;
    imageUrl?: string;
    bbox?: number[];
  }>>([]);

  const confirmedTrackList = tracks.filter((t) => t.confirmed);
  const selectedFrameCount = refinementFrames.length;
  const maxFrames = 16;
  const cropRunId = galleryRunId ?? runId ?? null;

  const frameById = useMemo(() => {
    const m = new Map<string, (typeof refinementCandidateFrames)[number]>();
    for (const f of refinementCandidateFrames) m.set(f.id, f);
    return m;
  }, [refinementCandidateFrames]);

  useEffect(() => {
    let cancelled = false;

    const loadFrames = async () => {
      if (!cropRunId || confirmedTrackList.length === 0) {
        if (!cancelled) setRefinementCandidateFrames([]);
        return;
      }

      setFramesLoading(true);
      const rows: Array<{
        id: string;
        frameId: number;
        timestamp: number;
        cameraId: string;
        trackId: number;
        imageUrl?: string;
        bbox?: number[];
      }> = [];

      for (const track of confirmedTrackList) {
        try {
          const sequence = await getTrackletSequence(cropRunId, track.cameraId, track.trackletId, 16);
          if (cancelled) return;

          const samples = Array.isArray(sequence.frames) ? sequence.frames.slice(0, 8) : [];
          if (samples.length > 0) {
            for (const sample of samples) {
              const start = Number(track.startTime ?? 0);
              const end = Number(track.endTime ?? start + 0.1);
              const rel = Number(sample.timeRel ?? 0);
              rows.push({
                id: `${track.id}-frame-${sample.frameId}`,
                frameId: sample.frameId,
                timestamp: Number.isFinite(sample.timestamp as number)
                  ? Number(sample.timestamp)
                  : start + rel * Math.max(end - start, 0.1),
                cameraId: track.cameraId,
                trackId: track.trackletId,
                imageUrl: getRunFullFrameUrl(cropRunId, track.cameraId, sample.frameId),
                bbox: Array.isArray(sample.bbox) ? sample.bbox : undefined,
              });
            }
            continue;
          }
        } catch {
          // Fall through to representative frame fallback below.
        }

        if (track.representativeFrame != null) {
          rows.push({
            id: `${track.id}-rep-${track.representativeFrame}`,
            frameId: Number(track.representativeFrame),
            timestamp: Number(track.startTime ?? 0),
            cameraId: track.cameraId,
            trackId: track.trackletId,
            imageUrl: getRunFullFrameUrl(cropRunId, track.cameraId, Number(track.representativeFrame)),
            bbox: Array.isArray(track.representativeBbox) ? track.representativeBbox : undefined,
          });
        }
      }

      if (!cancelled) {
        setRefinementCandidateFrames(rows.slice(0, 64));
        setCurrentFrameIndex((v) => Math.min(v, Math.max(rows.length - 1, 0)));
      }
      if (!cancelled) setFramesLoading(false);
    };

    void loadFrames();
    return () => {
      cancelled = true;
      setFramesLoading(false);
    };
  }, [cropRunId, confirmedTrackList]);

  const handleFrameSelect = (frameId: string) => {
    if (refinementFrames.includes(frameId)) {
      removeRefinementFrame(frameId);
    } else if (selectedFrameCount < maxFrames) {
      addRefinementFrame(frameId);
    }
  };

  const handleReSearch = () => {
    void (async () => {
      if (reSearchRunning) return;
      if (!runId) {
        setReSearchStatus("No active run to refine.");
        return;
      }

      const selected = refinementFrames
        .map((id) => frameById.get(id))
        .filter((f): f is NonNullable<typeof f> => Boolean(f));

      if (selected.length === 0) {
        setReSearchStatus("Select at least one frame first.");
        return;
      }

      setReSearchRunning(true);
      setReSearchStatus("Running re-search on selected frames...");

      try {
        const anchorMap = new Map<string, { cameraId: string; trackId: number }>();
        for (const s of selected) {
          anchorMap.set(`${s.cameraId}:${s.trackId}`, {
            cameraId: s.cameraId,
            trackId: s.trackId,
          });
        }
        const anchors = Array.from(anchorMap.values());

        const batches = await Promise.all(
          anchors.map(async (a) => {
            const resp = await getMatchedAlternatives(runId, {
              topK: 5,
              anchorCameraId: a.cameraId,
              anchorTrackId: a.trackId,
            });
            return resp.alternatives;
          })
        );

        type Agg = {
          scoreSum: number;
          count: number;
          alt: any;
        };
        const agg = new Map<string, Agg>();
        for (const alternatives of batches) {
          for (const alt of alternatives) {
            const key = `${alt.cameraId}:${alt.trackId}`;
            const prev = agg.get(key);
            if (!prev) {
              agg.set(key, {
                scoreSum: Number(alt.score ?? 0),
                count: 1,
                alt,
              });
            } else {
              prev.scoreSum += Number(alt.score ?? 0);
              prev.count += 1;
              if (Number(alt.score ?? 0) > Number(prev.alt?.score ?? 0)) {
                prev.alt = alt;
              }
            }
          }
        }

        const refinedRows = Array.from(agg.values())
          .map((x) => ({
            avgScore: x.count > 0 ? x.scoreSum / x.count : Number(x.alt?.score ?? 0),
            alt: x.alt,
          }))
          .sort((a, b) => b.avgScore - a.avgScore)
          .slice(0, 24);

        const refinedTracks: TimelineTrack[] = refinedRows.map((row, idx) => {
          const alt = row.alt;
          const start = Number.isFinite(alt.startTime) ? Number(alt.startTime) : 0;
          const endRaw = Number.isFinite(alt.endTime) ? Number(alt.endTime) : start + 0.1;
          const end = Math.max(endRaw, start + 0.1);
          const seg = {
            cameraId: alt.cameraId,
            trackId: Number(alt.trackId),
            globalId: alt.globalId ?? undefined,
            start,
            end,
            color: getCameraColor(alt.cameraId),
            representativeFrame: alt.representativeFrame,
            representativeBbox: alt.representativeBbox,
          };

          return {
            id: `refined-${alt.cameraId}-${alt.trackId}-${idx}`,
            cameraId: alt.cameraId,
            trackletId: Number(alt.trackId),
            globalId: alt.globalId ?? undefined,
            startTime: start,
            endTime: end,
            selected: false,
            confirmed: true,
            representativeFrame: alt.representativeFrame,
            representativeBbox: alt.representativeBbox,
            segments: [seg],
            label: alt.label ?? `Refined · ${alt.cameraId} · track ${alt.trackId}`,
            confidence: row.avgScore,
            className: alt.className ?? "vehicle",
          };
        });

        if (refinedTracks.length > 0) {
          setTracks(refinedTracks);
          clearRefinementFrames();
          setReSearchStatus(
            `Re-search complete: ${refinedTracks.length} refined candidates from ${selected.length} selected frame(s).`
          );
        } else {
          setReSearchStatus("No refined matches were found from selected frames.");
        }
      } catch (err: any) {
        setReSearchStatus(String(err?.message ?? "Re-search failed"));
      } finally {
        setReSearchRunning(false);
      }
    })();
  };

  const handleProceed = () => {
    setCurrentStage(6);
  };

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden">
      {/* Header */}
      <header className="flex shrink-0 flex-col gap-3 border-b px-4 py-3 sm:flex-row sm:items-center sm:justify-between sm:px-6">
        <div className="min-w-0">
          <h1 className="text-lg font-semibold">Stage 5: Refinement</h1>
          <p className="text-sm text-muted-foreground">
            Select reference frames for improved search accuracy
          </p>
        </div>
        <div className="flex flex-wrap items-center justify-end gap-2">
          <Badge variant="secondary">
            {selectedFrameCount}/{maxFrames} frames selected
          </Badge>
          <Button className="shrink-0" onClick={handleProceed}>
            Continue to Output
            <ArrowRight className="ml-2 h-4 w-4" />
          </Button>
        </div>
      </header>

      {/* Main content */}
      <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden lg:flex-row">
        {/* Left panel - Confirmed tracklets */}
        <aside className="flex max-h-[36vh] min-h-0 w-full shrink-0 flex-col border-b border-border lg:max-h-none lg:w-64 lg:border-b-0 lg:border-r">
          <div className="shrink-0 border-b p-4">
            <h3 className="font-semibold">Confirmed Tracklets</h3>
            <p className="text-sm text-muted-foreground">
              {confirmedTrackList.length} reference clips
            </p>
          </div>
          <ScrollArea className="min-h-0 flex-1">
            <div className="p-4 space-y-2">
              {confirmedTrackList.map((track) => (
                <div
                  key={track.id}
                  className="p-3 rounded-lg border bg-green-500/5 border-green-500/30"
                >
                  <div className="flex items-center gap-2">
                    <div
                      className="h-3 w-3 rounded-full"
                      style={{ backgroundColor: getCameraColor(track.cameraId) }}
                    />
                    <span className="font-medium text-sm">{track.cameraId}</span>
                    <Badge variant="secondary" className="ml-auto text-[10px]">
                      #{track.trackletId}
                    </Badge>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {formatDuration(track.startTime)} - {formatDuration(track.endTime)}
                  </p>
                </div>
              ))}
            </div>
          </ScrollArea>
        </aside>

        {/* Main area */}
        <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
          {/* Frame viewer */}
          <div className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden p-3 sm:p-4">
            {framesLoading && (
              <p className="mb-3 text-xs text-muted-foreground">Loading refinement frames...</p>
            )}
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
              {refinementCandidateFrames.slice(0, 30).map((frame) => {
                const isSelected = refinementFrames.includes(frame.id);
                return (
                  <FrameCard
                    key={frame.id}
                    frame={frame}
                    isSelected={isSelected}
                    onSelect={() => handleFrameSelect(frame.id)}
                    disabled={!isSelected && selectedFrameCount >= maxFrames}
                  />
                );
              })}
            </div>
          </div>

          {/* Playback controls */}
          <div className="shrink-0 border-t p-3 sm:p-4">
            <div className="flex flex-wrap items-center gap-3 sm:gap-4">
              {/* Navigation */}
              <div className="flex items-center gap-1">
                <Button
                  variant="ghost"
                  size="icon-sm"
                  onClick={() => setCurrentFrameIndex(Math.max(0, currentFrameIndex - 10))}
                >
                  <ChevronLeft className="h-4 w-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setIsPlaying(!isPlaying)}
                >
                  {isPlaying ? (
                    <Pause className="h-5 w-5" />
                  ) : (
                    <Play className="h-5 w-5" />
                  )}
                </Button>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  onClick={() =>
                    setCurrentFrameIndex(Math.min(refinementCandidateFrames.length - 1, currentFrameIndex + 10))
                  }
                >
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>

              {/* Timeline scrubber */}
              <div className="min-w-[120px] flex-1 basis-[160px]">
                <Slider
                  value={[currentFrameIndex]}
                  max={Math.max(refinementCandidateFrames.length - 1, 0)}
                  step={1}
                  onValueChange={(v) => setCurrentFrameIndex(v[0])}
                />
              </div>

              {/* Speed control */}
              <div className="flex items-center gap-2">
                <span className="text-sm text-muted-foreground">Speed:</span>
                <select
                  value={playbackSpeed}
                  onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
                  className="bg-muted rounded px-2 py-1 text-sm"
                >
                  <option value={0.25}>0.25x</option>
                  <option value={0.5}>0.5x</option>
                  <option value={1}>1x</option>
                  <option value={1.5}>1.5x</option>
                  <option value={2}>2x</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        {/* Right panel - Selected frames */}
        <aside className="w-80 border-l flex flex-col">
          <div className="p-4 border-b">
            <h3 className="font-semibold">Selected Reference Frames</h3>
            <p className="text-sm text-muted-foreground">
              Used as ground truth for re-search
            </p>
          </div>
          <ScrollArea className="min-h-0 flex-1">
            <div className="p-4">
              {refinementFrames.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  <Image className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p>No frames selected</p>
                  <p className="text-sm">Click frames to add as reference</p>
                </div>
              ) : (
                <div className="grid grid-cols-2 gap-2">
                  {refinementFrames.map((frameId) => {
                    const frame = refinementCandidateFrames.find((f) => f.id === frameId);
                    if (!frame) return null;
                    return (
                      <div
                        key={frameId}
                        className="relative aspect-video bg-muted rounded-md overflow-hidden group"
                      >
                        {frame.imageUrl ? (
                          <img
                            src={frame.imageUrl}
                            alt={`${frame.cameraId} frame ${frame.frameId}`}
                            className="absolute inset-0 h-full w-full object-cover"
                            draggable={false}
                          />
                        ) : (
                          <div className="absolute inset-0 flex items-center justify-center">
                            <Image className="h-6 w-6 text-muted-foreground/50" />
                          </div>
                        )}
                        <Button
                          variant="ghost"
                          size="icon-sm"
                          className="absolute top-1 right-1 h-5 w-5 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity"
                          onClick={() => removeRefinementFrame(frameId)}
                        >
                          <X className="h-3 w-3 text-white" />
                        </Button>
                        <div
                          className="absolute bottom-0 left-0 right-0 h-1"
                          style={{ backgroundColor: getCameraColor(frame.cameraId) }}
                        />
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </ScrollArea>

          {/* Actions */}
          <div className="p-4 border-t space-y-2">
            {reSearchStatus && (
              <p className="text-xs text-muted-foreground">{reSearchStatus}</p>
            )}
            <Button
              className="w-full"
              onClick={handleReSearch}
              disabled={refinementFrames.length === 0 || reSearchRunning || !currentVideo}
            >
              <Search className="mr-2 h-4 w-4" />
              {reSearchRunning ? "Re-searching..." : "Re-Search with Selected"}
            </Button>
            <Button
              variant="outline"
              className="w-full"
              onClick={clearRefinementFrames}
              disabled={refinementFrames.length === 0}
            >
              <RefreshCw className="mr-2 h-4 w-4" />
              Clear Selection
            </Button>
          </div>
        </aside>
      </div>
    </div>
  );
}

interface FrameCardProps {
  frame: {
    id: string;
    frameId: number;
    timestamp: number;
    cameraId: string;
    imageUrl?: string;
    bbox?: number[];
  };
  isSelected: boolean;
  onSelect: () => void;
  disabled: boolean;
}

function FrameCard({ frame, isSelected, onSelect, disabled }: FrameCardProps) {
  return (
    <div
      className={cn(
        "relative aspect-video rounded-lg overflow-hidden cursor-pointer transition-all",
        "border-2",
        isSelected
          ? "border-green-500 shadow-lg shadow-green-500/20"
          : "border-transparent hover:border-primary/50",
        disabled && !isSelected && "opacity-50 cursor-not-allowed"
      )}
      onClick={() => !disabled && onSelect()}
    >
      {frame.imageUrl ? (
        frame.bbox ? (
          <TrackletFrameView src={frame.imageUrl} bbox={frame.bbox} />
        ) : (
          <img
            src={frame.imageUrl}
            alt={`${frame.cameraId} frame ${frame.frameId}`}
            className="absolute inset-0 h-full w-full object-cover"
            draggable={false}
          />
        )
      ) : (
        <div className="absolute inset-0 bg-muted flex items-center justify-center">
          <Image className="h-8 w-8 text-muted-foreground/30" />
        </div>
      )}

      {/* Selection indicator */}
      {isSelected && (
        <div className="absolute top-2 right-2 h-5 w-5 bg-green-500 rounded-full flex items-center justify-center">
          <Check className="h-3 w-3 text-white" />
        </div>
      )}

      {/* Info overlay */}
      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-2">
        <div className="flex items-center justify-between">
          <div
            className="flex items-center gap-1"
          >
            <div
              className="h-2 w-2 rounded-full"
              style={{ backgroundColor: getCameraColor(frame.cameraId) }}
            />
            <span className="text-[10px] text-white">{frame.cameraId}</span>
          </div>
          <span className="text-[10px] text-white/70">
            {formatDuration(frame.timestamp)}
          </span>
        </div>
      </div>
    </div>
  );
}
