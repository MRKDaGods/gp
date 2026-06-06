"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArrowRight, Check, Image as ImageIcon, RefreshCw, Search, X } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { DisclosurePanel, ErrorBanner, PlaybackControls } from "@/components/pipeline";
import { getMatchedAlternatives, getRunFullFrameUrl, getTrackletSequence } from "@/lib/api";
import { cn, formatDuration, getCameraColor } from "@/lib/utils";
import { useManualStageStore, usePipelineStore, useSessionStore, useTimelineStore, useVideoStore } from "@/store";
import { TrackletFrameView } from "@/components/ui/double-buffered-img";
import type { TimelineTrack } from "@/types";

const MAX_REFINEMENT_FRAMES = 16;
const REFINEMENT_RESEARCH_EVENT = "mtmc:stage5:research";

type RefinementFrame = {
  id: string;
  frameId: number;
  timestamp: number;
  cameraId: string;
  trackId: number;
  imageUrl?: string;
  bbox?: number[];
};

export function RefinementStage() {
  const { tracks, replaceTracksSyncingRowFlags } = useTimelineStore();
  const { runId, galleryRunId } = usePipelineStore();
  const { currentVideo } = useVideoStore();
  const { refinementFrames, addRefinementFrame, removeRefinementFrame, clearRefinementFrames } = useSessionStore();

  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [isPlaying, setIsPlaying] = useState(false);
  const [framesLoading, setFramesLoading] = useState(false);
  const [reSearchRunning, setReSearchRunning] = useState(false);
  const [reSearchStatus, setReSearchStatus] = useState<string | null>(null);
  const [reSearchError, setReSearchError] = useState<string | null>(null);
  const [refinementCandidateFrames, setRefinementCandidateFrames] = useState<RefinementFrame[]>([]);

  const confirmedTrackList = useMemo(() => tracks.filter((track) => track.confirmed), [tracks]);
  const cropRunId = galleryRunId ?? runId ?? null;
  const selectedFrameCount = refinementFrames.length;

  const frameById = useMemo(() => {
    const frames = new Map<string, RefinementFrame>();
    refinementCandidateFrames.forEach((frame) => frames.set(frame.id, frame));
    return frames;
  }, [refinementCandidateFrames]);

  useEffect(() => {
    let cancelled = false;

    const loadFrames = async () => {
      if (!cropRunId || confirmedTrackList.length === 0) {
        if (!cancelled) setRefinementCandidateFrames([]);
        return;
      }

      setFramesLoading(true);
      const rows = await Promise.all(
        confirmedTrackList.map(async (track): Promise<RefinementFrame[]> => {
          try {
            const sequence = await getTrackletSequence(cropRunId, track.cameraId, track.trackletId, 16);
            const samples = Array.isArray(sequence.frames) ? sequence.frames.slice(0, 8) : [];
            if (samples.length > 0) {
              return samples.map((sample) => {
                const start = Number(track.startTime ?? 0);
                const end = Number(track.endTime ?? start + 0.1);
                const rel = Number(sample.timeRel ?? 0);
                return {
                  id: `${track.id}-frame-${sample.frameId}`,
                  frameId: sample.frameId,
                  timestamp: Number.isFinite(sample.timestamp as number)
                    ? Number(sample.timestamp)
                    : start + rel * Math.max(end - start, 0.1),
                  cameraId: track.cameraId,
                  trackId: track.trackletId,
                  imageUrl: getRunFullFrameUrl(cropRunId, track.cameraId, sample.frameId),
                  bbox: Array.isArray(sample.bbox) ? sample.bbox : undefined,
                };
              });
            }
          } catch {
            // Fall back to the representative frame already present on the track.
          }

          if (track.representativeFrame == null) return [];
          return [{
            id: `${track.id}-rep-${track.representativeFrame}`,
            frameId: Number(track.representativeFrame),
            timestamp: Number(track.startTime ?? 0),
            cameraId: track.cameraId,
            trackId: track.trackletId,
            imageUrl: getRunFullFrameUrl(cropRunId, track.cameraId, Number(track.representativeFrame)),
            bbox: Array.isArray(track.representativeBbox) ? track.representativeBbox : undefined,
          }];
        })
      );

      if (cancelled) return;
      const nextFrames = rows.flat().slice(0, 64);
      setRefinementCandidateFrames(nextFrames);
      setCurrentFrameIndex((value) => Math.min(value, Math.max(nextFrames.length - 1, 0)));
      setFramesLoading(false);
    };

    void loadFrames();
    return () => {
      cancelled = true;
      setFramesLoading(false);
    };
  }, [cropRunId, confirmedTrackList]);

  useEffect(() => {
    if (!isPlaying || refinementCandidateFrames.length <= 1) return;
    const id = window.setInterval(() => {
      setCurrentFrameIndex((index) => (index + 1) % refinementCandidateFrames.length);
    }, Math.max(120, 900 / playbackSpeed));
    return () => window.clearInterval(id);
  }, [isPlaying, playbackSpeed, refinementCandidateFrames.length]);

  // If the user comes back and changes their reference frames (or a re-search / clear empties
  // them), Refinement is being re-worked — un-mark it done (and drop its per-run completion
  // marker) so it isn't shown as finished while edits are pending.
  //
  // CRITICAL: depend ONLY on `refinementFrames`, and read runId/stores via getState. Listing
  // `runId` as a dep made this fire when a run is LOADED (runId changes) and wipe the done
  // status that useLoadRun had just restored — making Refinement go stale on open-run/refresh.
  const skipFramesInvalidateRef = useRef(true);
  useEffect(() => {
    if (skipFramesInvalidateRef.current) {
      skipFramesInvalidateRef.current = false;
      return;
    }
    const pipeline = usePipelineStore.getState();
    const rid = pipeline.runId;
    if (rid) useManualStageStore.getState().clearManualStage(rid, 5);
    const stage5 = pipeline.stages.find((st) => st.stage === 5);
    if (stage5 && (stage5.status === "completed" || stage5.progress >= 100)) {
      pipeline.updateStageProgress(5, { status: "idle", progress: 0, message: "" });
    }
  }, [refinementFrames]);

  const handleFrameSelect = (frameId: string) => {
    if (refinementFrames.includes(frameId)) {
      removeRefinementFrame(frameId);
    } else if (selectedFrameCount < MAX_REFINEMENT_FRAMES) {
      addRefinementFrame(frameId);
    }
  };

  const handleReSearch = useCallback(async () => {
    if (reSearchRunning) return;
    setReSearchError(null);

    if (!runId) {
      setReSearchError("No active run to refine.");
      return;
    }

    const selected = refinementFrames.map((id) => frameById.get(id)).filter((frame): frame is RefinementFrame => Boolean(frame));
    if (selected.length === 0) {
      setReSearchError("Select at least one frame first.");
      return;
    }

    setReSearchRunning(true);
    setReSearchStatus("Running re-search on selected frames...");

    try {
      const anchors = Array.from(
        new Map(selected.map((frame) => [`${frame.cameraId}:${frame.trackId}`, { cameraId: frame.cameraId, trackId: frame.trackId }])).values()
      );
      const batches = await Promise.all(
        anchors.map(async (anchor) => {
          const response = await getMatchedAlternatives(runId, {
            topK: 5,
            anchorCameraId: anchor.cameraId,
            anchorTrackId: anchor.trackId,
          });
          return response.alternatives;
        })
      );

      const aggregated = new Map<string, { scoreSum: number; count: number; alt: any }>();
      batches.flat().forEach((alt) => {
        const key = `${alt.cameraId}:${alt.trackId}`;
        const prev = aggregated.get(key);
        if (!prev) {
          aggregated.set(key, { scoreSum: Number(alt.score ?? 0), count: 1, alt });
          return;
        }
        prev.scoreSum += Number(alt.score ?? 0);
        prev.count += 1;
        if (Number(alt.score ?? 0) > Number(prev.alt?.score ?? 0)) prev.alt = alt;
      });

      const refinedTracks: TimelineTrack[] = Array.from(aggregated.values())
        .map((item) => ({ avgScore: item.count > 0 ? item.scoreSum / item.count : Number(item.alt?.score ?? 0), alt: item.alt }))
        .sort((a, b) => b.avgScore - a.avgScore)
        .slice(0, 24)
        .map((row, index) => buildRefinedTrack(row.alt, row.avgScore, index));

      if (refinedTracks.length > 0) {
        replaceTracksSyncingRowFlags(refinedTracks);
        clearRefinementFrames();
        setReSearchStatus(`Re-search complete: ${refinedTracks.length} refined candidates from ${selected.length} selected frame(s).`);
      } else {
        setReSearchStatus("No refined matches were found from selected frames.");
      }
    } catch (error) {
      setReSearchError(String(error instanceof Error ? error.message : error || "Re-search failed"));
    } finally {
      setReSearchRunning(false);
    }
  }, [clearRefinementFrames, frameById, refinementFrames, reSearchRunning, replaceTracksSyncingRowFlags, runId]);

  useEffect(() => {
    const listener = () => void handleReSearch();
    window.addEventListener(REFINEMENT_RESEARCH_EVENT, listener);
    return () => window.removeEventListener(REFINEMENT_RESEARCH_EVENT, listener);
  }, [handleReSearch]);

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden">
      <div className="shrink-0 space-y-3 border-b px-4 py-3 sm:px-6">
        <ErrorBanner title="Refinement failed" message={reSearchError} />
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant="outline">{selectedFrameCount}/{MAX_REFINEMENT_FRAMES} reference frames</Badge>
          {refinementFrames.map((frameId) => {
            const frame = frameById.get(frameId);
            return (
              <button
                key={frameId}
                type="button"
                onClick={() => removeRefinementFrame(frameId)}
                className="group flex items-center gap-1 rounded-full border bg-muted/50 px-2 py-1 text-[11px] font-mono transition-colors hover:border-destructive/30 hover:bg-destructive/10"
                aria-label={`Remove reference frame ${frameId}`}
              >
                {frame ? `${frame.cameraId}:${frame.frameId}` : frameId}
                <X className="h-3 w-3 text-muted-foreground group-hover:text-destructive" />
              </button>
            );
          })}
          {reSearchStatus ? <span className="text-xs text-muted-foreground">{reSearchStatus}</span> : null}
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-4 sm:p-6">
        {framesLoading ? <p className="mb-3 text-xs text-muted-foreground">Loading refinement frames...</p> : null}
        {refinementCandidateFrames.length === 0 ? (
          <div className="flex min-h-[260px] flex-col items-center justify-center rounded-md border border-dashed text-muted-foreground">
            <ImageIcon className="mb-3 h-10 w-10 opacity-50" aria-hidden />
            <p className="font-medium">No confirmed reference frames</p>
            <p className="text-sm">Confirm tracks in Stage 4 before refining.</p>
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6">
            {refinementCandidateFrames.slice(0, 30).map((frame, index) => (
              <FrameCard
                key={frame.id}
                frame={frame}
                active={index === currentFrameIndex}
                isSelected={refinementFrames.includes(frame.id)}
                disabled={!refinementFrames.includes(frame.id) && selectedFrameCount >= MAX_REFINEMENT_FRAMES}
                onSelect={() => handleFrameSelect(frame.id)}
              />
            ))}
          </div>
        )}
      </div>

      <div className="shrink-0 space-y-3 border-t p-4 sm:px-6">
        <PlaybackControls
          isPlaying={isPlaying}
          currentFrame={currentFrameIndex}
          totalFrames={refinementCandidateFrames.length}
          onPlayPause={() => setIsPlaying((value) => !value)}
          onFrameChange={setCurrentFrameIndex}
          onStepBack={() => setCurrentFrameIndex((value) => Math.max(0, value - 1))}
          onStepForward={() => setCurrentFrameIndex((value) => Math.min(refinementCandidateFrames.length - 1, value + 1))}
        />

        <DisclosurePanel title="Advanced" description="Playback speed and larger navigation jumps.">
          <div className="flex flex-wrap items-center gap-2">
            <Button type="button" variant="outline" size="sm" onClick={() => setCurrentFrameIndex((value) => Math.max(0, value - 10))}>
              Prev 10
            </Button>
            <Button type="button" variant="outline" size="sm" onClick={() => setCurrentFrameIndex((value) => Math.min(refinementCandidateFrames.length - 1, value + 10))}>
              Next 10
            </Button>
            {[0.25, 0.5, 1, 1.5, 2].map((speed) => (
              <Button key={speed} type="button" variant={playbackSpeed === speed ? "secondary" : "ghost"} size="sm" onClick={() => setPlaybackSpeed(speed)}>
                {speed}x
              </Button>
            ))}
          </div>
        </DisclosurePanel>

        <DisclosurePanel title="Debug" tier="debug" description="Raw frame and selection state from the session store.">
          <div className="grid gap-3 text-xs text-muted-foreground sm:grid-cols-2 lg:grid-cols-4">
            <DebugStat label="Confirmed tracks" value={confirmedTrackList.length} />
            <DebugStat label="Candidate frames" value={refinementCandidateFrames.length} />
            <DebugStat label="Selected frames" value={selectedFrameCount} />
            <DebugStat label="Active frame" value={currentFrameIndex} />
            <div className="break-all font-mono sm:col-span-2 lg:col-span-4">refinementFrames: {JSON.stringify(refinementFrames)}</div>
            <div className="break-all font-mono sm:col-span-2 lg:col-span-4">currentVideo: {currentVideo?.id ?? "none"}</div>
          </div>
        </DisclosurePanel>
      </div>
    </div>
  );
}

export function RefinementStageActions() {
  const { refinementFrames, clearRefinementFrames, setCurrentStage } = useSessionStore();
  const { currentVideo } = useVideoStore();
  const updateStageProgress = usePipelineStore((s) => s.updateStageProgress);
  const runId = usePipelineStore((s) => s.runId);
  const markManualStageDone = useManualStageStore((s) => s.markManualStageDone);

  const handleContinue = () => {
    // Refinement is a manual review with no pipeline run of its own, so nothing else marks
    // it complete. Stamp it done when the user finishes and moves on, so the nav reflects it,
    // and record it per-run so loading the run later restores the checkmark.
    updateStageProgress(5, { status: "completed", progress: 100, message: "Refinement reviewed" });
    if (runId) markManualStageDone(runId, 5);
    setCurrentStage(6);
  };

  return (
    <>
      <Button type="button" variant="outline" onClick={clearRefinementFrames} disabled={refinementFrames.length === 0}>
        <RefreshCw className="mr-2 h-4 w-4" />
        Clear Selection
      </Button>
      <Button
        type="button"
        variant="outline"
        onClick={() => window.dispatchEvent(new Event(REFINEMENT_RESEARCH_EVENT))}
        disabled={refinementFrames.length === 0 || !currentVideo}
      >
        <Search className="mr-2 h-4 w-4" />
        Re-Search
      </Button>
      <Button type="button" onClick={handleContinue}>
        Continue to Output
        <ArrowRight className="ml-2 h-4 w-4" />
      </Button>
    </>
  );
}

function buildRefinedTrack(alt: any, avgScore: number, index: number): TimelineTrack {
  const start = Number.isFinite(alt.startTime) ? Number(alt.startTime) : 0;
  const endRaw = Number.isFinite(alt.endTime) ? Number(alt.endTime) : start + 0.1;
  const end = Math.max(endRaw, start + 0.1);
  const segment = {
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
    id: `refined-${alt.cameraId}-${alt.trackId}-${index}`,
    cameraId: alt.cameraId,
    trackletId: Number(alt.trackId),
    globalId: alt.globalId ?? undefined,
    startTime: start,
    endTime: end,
    selected: false,
    confirmed: true,
    representativeFrame: alt.representativeFrame,
    representativeBbox: alt.representativeBbox,
    segments: [segment],
    label: alt.label ?? `Refined · ${alt.cameraId} · track ${alt.trackId}`,
    confidence: avgScore,
    className: alt.className ?? "vehicle",
  };
}

function FrameCard({ frame, active, isSelected, onSelect, disabled }: { frame: RefinementFrame; active: boolean; isSelected: boolean; onSelect: () => void; disabled: boolean }) {
  return (
    <button
      type="button"
      className={cn(
        "relative aspect-video overflow-hidden rounded-md border-2 text-left transition-all",
        isSelected ? "border-success shadow-lg shadow-success/20" : "border-transparent hover:border-primary/50",
        active && "ring-2 ring-primary/40",
        disabled && !isSelected && "cursor-not-allowed opacity-50"
      )}
      onClick={() => !disabled && onSelect()}
    >
      {frame.imageUrl ? (
        frame.bbox ? (
          <TrackletFrameView src={frame.imageUrl} bbox={frame.bbox} />
        ) : (
          <img src={frame.imageUrl} alt={`${frame.cameraId} frame ${frame.frameId}`} className="absolute inset-0 h-full w-full object-cover" draggable={false} />
        )
      ) : (
        <span className="absolute inset-0 flex items-center justify-center bg-muted">
          <ImageIcon className="h-8 w-8 text-muted-foreground/30" aria-hidden />
        </span>
      )}
      {isSelected ? (
        <span className="absolute right-2 top-2 flex h-5 w-5 items-center justify-center rounded-full bg-success">
          <Check className="h-3 w-3 text-white" />
        </span>
      ) : null}
      <span className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 to-transparent p-2">
        <span className="flex items-center justify-between gap-2">
          <span className="flex items-center gap-1 text-[10px] text-white">
            <span className="h-2 w-2 rounded-full" style={{ backgroundColor: getCameraColor(frame.cameraId) }} />
            {frame.cameraId}
          </span>
          <span className="text-[10px] text-white/70">{formatDuration(frame.timestamp)}</span>
        </span>
      </span>
    </button>
  );
}

function DebugStat({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex justify-between gap-3 rounded bg-muted/40 px-3 py-2">
      <span>{label}</span>
      <span className="font-mono">{value}</span>
    </div>
  );
}