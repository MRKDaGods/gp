"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CheckCircle2, Layers, Loader2, MousePointer2, RefreshCw, Search, Video, XCircle } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { DisclosurePanel, ErrorBanner } from "@/components/pipeline";
import { apiUrl, getTracklets } from "@/lib/api";
import { flushPipelineFromStage } from "@/lib/pipeline-flush";
import { cn } from "@/lib/utils";
import { classColorFor, classLabelFor } from "@/lib/class-meta";
import { useDetectionStore, useManualStageStore, usePipelineStore, useSessionStore, useVideoStore } from "@/store";

/** Pill toggle for the selection filters (camera / class). */
function FilterPill({
  active,
  onClick,
  label,
  count,
  icon,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
  count?: number;
  icon?: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs font-medium transition-colors",
        active
          ? "border-accent-strong bg-accent-strong/15 text-foreground"
          : "border-border/60 bg-background/40 text-muted-foreground hover:text-foreground"
      )}
      aria-pressed={active}
    >
      {icon}
      {label}
      {count != null && <span className="opacity-60">{count}</span>}
    </button>
  );
}

function uploadVideoCropUrl(
  videoId: string,
  frameId: number,
  bbox: number[],
  opts?: { quality?: number; minEdge?: number; pad?: number }
): string {
  if (!bbox || bbox.length !== 4) return "";
  const q = opts?.quality ?? 92;
  const minEdge = opts?.minEdge ?? 200;
  const pad = opts?.pad ?? 0.12;
  const [x1, y1, x2, y2] = bbox;
  return apiUrl(`/crops/${encodeURIComponent(videoId)}?frameId=${frameId}&x1=${x1}&y1=${y1}&x2=${x2}&y2=${y2}&quality=${q}&minEdge=${minEdge}&pad=${pad}`);
}

interface SampleFrame {
  frameId: number;
  bbox: number[];
}

interface TrackletSummary {
  id: number;
  cameraId: string;
  startFrame: number;
  endFrame: number;
  numFrames: number;
  duration: number;
  className: string;
  classId: number;
  confidence: number;
  representativeFrame: number;
  representativeBbox: number[];
  sampleFrames?: SampleFrame[];
}

export function SelectionStage() {
  const {
    selectedTrackIds,
    toggleTrackSelection,
    deselectAll,
    multiSelectMode,
    setMultiSelectMode,
  } = useDetectionStore();
  const { currentVideo, videos, setCurrentVideo } = useVideoStore();

  const [tracklets, setTracklets] = useState<TrackletSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  // Filters
  const [cameraFilter, setCameraFilter] = useState<string>("all");
  const [classFilter, setClassFilter] = useState<number | "all">("all");
  const [search, setSearch] = useState("");

  useEffect(() => {
    setMultiSelectMode(true);
  }, [setMultiSelectMode]);

  const fetchTracklets = useCallback(async () => {
    if (!currentVideo) {
      setTracklets([]);
      setLoading(false);
      return;
    }

    setLoading(true);
    setLoadError(null);
    try {
      // Load tracklets for EVERY camera in the run so they can be searched and
      // filtered by camera, not just the currently-viewed one.
      const resp = await getTracklets(undefined, currentVideo.id, { allCameras: true });
      setTracklets(Array.isArray(resp.data) ? resp.data : []);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setLoadError(`Failed to load tracklets: ${msg}`);
      setTracklets([]);
    } finally {
      setLoading(false);
    }
  }, [currentVideo]);

  useEffect(() => {
    void fetchTracklets();
  }, [fetchTracklets]);

  // Camera + class breakdown and the filtered, searched tracklet list.
  const cameras = useMemo(() => {
    const counts = new Map<string, number>();
    for (const t of tracklets) {
      if (!t.cameraId) continue;
      counts.set(t.cameraId, (counts.get(t.cameraId) ?? 0) + 1);
    }
    return [...counts.entries()].sort((a, b) => a[0].localeCompare(b[0], undefined, { numeric: true }));
  }, [tracklets]);
  const classCounts = useMemo(() => {
    const counts: Record<number, number> = {};
    for (const t of tracklets) counts[t.classId] = (counts[t.classId] ?? 0) + 1;
    return counts;
  }, [tracklets]);
  const presentClasses = useMemo(
    () => Object.keys(classCounts).map(Number).sort((a, b) => a - b),
    [classCounts]
  );
  const filteredTracklets = useMemo(() => {
    const q = search.trim().replace(/^#/, "");
    return tracklets.filter((t) => {
      if (cameraFilter !== "all" && t.cameraId !== cameraFilter) return false;
      if (classFilter !== "all" && t.classId !== classFilter) return false;
      if (q && !String(t.id).includes(q)) return false;
      return true;
    });
  }, [tracklets, cameraFilter, classFilter, search]);

  // Picking a camera also focuses that camera's video so the downstream timeline
  // query (which is scoped to currentVideo) targets the right camera.
  const pickCamera = useCallback(
    (cam: string) => {
      setCameraFilter(cam);
      if (cam !== "all") {
        const match = videos.find((v) => v.cameraId === cam);
        if (match && match.id !== currentVideo?.id) setCurrentVideo(match);
      }
    },
    [videos, currentVideo?.id, setCurrentVideo]
  );

  const selectionSig = useMemo(
    () => Array.from(selectedTrackIds).sort((a, b) => a - b).join(","),
    [selectedTrackIds]
  );
  const skipSelectionFlushRef = useRef(true);
  useEffect(() => {
    if (skipSelectionFlushRef.current) {
      skipSelectionFlushRef.current = false;
      return;
    }
    // The selection is consumed ONLY by the Stage-4 timeline query - feature
    // extraction (Stage 2) and indexing (Stage 3) run over the whole run and never
    flushPipelineFromStage(4);
  }, [selectionSig]);

  const selectedTracklets = useMemo(
    () => tracklets.filter((tracklet) => selectedTrackIds.has(tracklet.id)),
    [selectedTrackIds, tracklets]
  );

  const groupedSelected = selectedTracklets.reduce((acc, tracklet) => {
    const className = classLabelFor(tracklet.classId, tracklet.className);
    acc[className] = [...(acc[className] ?? []), tracklet];
    return acc;
  }, {} as Record<string, TrackletSummary[]>);

  const selectAllTracklets = () => {
    filteredTracklets.forEach((tracklet) => {
      if (!selectedTrackIds.has(tracklet.id)) toggleTrackSelection(tracklet.id);
    });
  };

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden">
      <div className="shrink-0 space-y-2.5 border-b px-4 py-3 sm:px-6">
        <ErrorBanner title="Tracklet loading failed" message={loadError} />
        <div className="flex flex-wrap items-center gap-2">
          <div className="relative w-full max-w-[220px]">
            <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search by #id..."
              className="h-8 pl-8 text-sm"
              aria-label="Search tracklets by id"
            />
          </div>
          <Badge variant="outline" className="ml-auto shrink-0">
            {selectedTrackIds.size} selected * {filteredTracklets.length} shown
          </Badge>
          {selectedTrackIds.size > 0 && (
            <Button type="button" variant="ghost" size="sm" className="h-7 px-2 text-xs" onClick={deselectAll}>
              Clear
            </Button>
          )}
        </div>
        {cameras.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">Camera</span>
            <FilterPill active={cameraFilter === "all"} onClick={() => pickCamera("all")} label="All" count={tracklets.length} />
            {cameras.map(([cam, n]) => (
              <FilterPill
                key={cam}
                active={cameraFilter === cam}
                onClick={() => pickCamera(cam)}
                label={cam}
                count={n}
                icon={<Video className="h-3 w-3" />}
              />
            ))}
          </div>
        )}
        {presentClasses.length > 1 && (
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">Type</span>
            <FilterPill active={classFilter === "all"} onClick={() => setClassFilter("all")} label="All" count={tracklets.length} />
            {presentClasses.map((cid) => (
              <FilterPill
                key={cid}
                active={classFilter === cid}
                onClick={() => setClassFilter(cid)}
                label={classLabelFor(cid)}
                count={classCounts[cid]}
              />
            ))}
          </div>
        )}
      </div>

      {loading ? (
        <div className="flex h-full items-center justify-center">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
          <span className="ml-3 text-muted-foreground">Loading tracklets...</span>
        </div>
      ) : (
        <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden xl:flex-row">
          <div className="min-h-[200px] min-w-0 flex-1 overflow-x-hidden overflow-y-auto p-4 sm:p-6 xl:min-h-0">
            {filteredTracklets.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
                <MousePointer2 className="mb-4 h-12 w-12 opacity-50" />
                {tracklets.length === 0 ? (
                  <>
                    <p className="text-lg font-medium">No tracklets found</p>
                    <p className="text-sm">Run detection first to generate tracklets.</p>
                  </>
                ) : (
                  <>
                    <p className="text-lg font-medium">No matches</p>
                    <p className="text-sm">No tracklets match the current filters.</p>
                  </>
                )}
              </div>
            ) : (
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                {filteredTracklets.map((tracklet) => (
                  <TrackletCard
                    key={`${tracklet.cameraId}-${tracklet.id}`}
                    tracklet={tracklet}
                    videoId={currentVideo?.id}
                    isSelected={selectedTrackIds.has(tracklet.id)}
                    onToggle={() => toggleTrackSelection(tracklet.id)}
                  />
                ))}
              </div>
            )}
          </div>

          <aside className="flex max-h-[42vh] min-h-0 w-full shrink-0 flex-col border-t border-border bg-muted/20 xl:max-h-none xl:w-80 xl:border-l xl:border-t-0">
            <div className="shrink-0 border-b p-4">
              <h3 className="flex items-center gap-2 font-semibold">
                <Layers className="h-4 w-4" />
                Selected Tracklets
              </h3>
              <p className="text-sm text-muted-foreground">{selectedTrackIds.size} tracklets will seed inference</p>
            </div>

            <ScrollArea className="min-h-0 flex-1">
              <div className="space-y-4 p-4">
                {Object.entries(groupedSelected).map(([className, items]) => (
                  <div key={className}>
                    <div className="mb-2 flex items-center gap-2">
                      <div className="h-3 w-3 rounded-full" style={{ backgroundColor: classColorFor(items[0].classId) }} />
                      <span className="font-medium capitalize">{className}</span>
                      <Badge variant="secondary" className="ml-auto">{items.length}</Badge>
                    </div>
                    <div className="space-y-1 pl-5">
                      {items.map((item) => (
                        <div key={item.id} className="flex items-center justify-between rounded bg-muted/50 px-2 py-1 text-sm">
                          <span className="text-muted-foreground">Track #{item.id}</span>
                          <span>{item.numFrames} frames</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}

                {selectedTrackIds.size === 0 ? (
                  <div className="py-8 text-center text-muted-foreground">
                    <MousePointer2 className="mx-auto mb-2 h-8 w-8 opacity-50" />
                    <p>No tracklets selected</p>
                    <p className="text-sm">Click tracklets to select them</p>
                  </div>
                ) : null}
              </div>
            </ScrollArea>

            <div className="shrink-0 space-y-3 border-t p-3">
              <DisclosurePanel title="Advanced" description="Bulk controls and selection behavior.">
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <Checkbox id="stage2-multi-select" checked={multiSelectMode} onCheckedChange={(checked) => setMultiSelectMode(checked === true)} />
                    <Label htmlFor="stage2-multi-select" className="text-sm">Multi-select mode</Label>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <Button type="button" variant="outline" size="sm" onClick={selectAllTracklets} disabled={tracklets.length === 0} aria-label="Select all tracklets">
                      Select All
                    </Button>
                    <Button type="button" variant="outline" size="sm" onClick={deselectAll} disabled={selectedTrackIds.size === 0} aria-label="Deselect all tracklets">
                      Deselect All
                    </Button>
                  </div>
                </div>
              </DisclosurePanel>

              <DisclosurePanel title="Debug" tier="debug" description="Raw counts, IDs, and refresh controls.">
                <div className="space-y-3 text-xs text-muted-foreground">
                  <div className="flex justify-between gap-3"><span>Tracklet count</span><span className="font-mono">{tracklets.length}</span></div>
                  <div className="flex justify-between gap-3"><span>Selected count</span><span className="font-mono">{selectedTrackIds.size}</span></div>
                  <div className="break-all font-mono">videoId: {currentVideo?.id ?? "none"}</div>
                  <div className="max-h-24 overflow-auto break-all font-mono">ids: {tracklets.map((tracklet) => tracklet.id).join(", ") || "none"}</div>
                  <Button type="button" variant="outline" size="sm" onClick={() => void fetchTracklets()} aria-label="Refresh tracklets">
                    <RefreshCw className="mr-2 h-4 w-4" />
                    Refresh
                  </Button>
                </div>
              </DisclosurePanel>
            </div>
          </aside>
        </div>
      )}
    </div>
  );
}

export function SelectionStageActions() {
  const { selectedTrackIds } = useDetectionStore();
  const { setCurrentStage } = useSessionStore();
  const updateStageProgress = usePipelineStore((s) => s.updateStageProgress);
  const runId = usePipelineStore((s) => s.runId);
  const markManualStageDone = useManualStageStore((s) => s.markManualStageDone);

  const handleContinue = () => {
    // Selection is a manual pick with no pipeline run of its own - stamp it done when the
    // user finishes and moves on, so the nav reflects it (mirrors Refinement), and record it
    updateStageProgress(2, { status: "completed", progress: 100, message: "Tracklets selected" });
    if (runId) markManualStageDone(runId, 2);
    setCurrentStage(3);
  };

  return (
    <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
      <Button type="button" onClick={handleContinue} disabled={selectedTrackIds.size === 0} aria-label="Continue to Stage 3 inference">
        Continue to Inference
      </Button>
    </div>
  );
}

function TrackletCard({
  tracklet,
  videoId,
  isSelected,
  onToggle,
}: {
  tracklet: TrackletSummary;
  videoId?: string;
  isSelected: boolean;
  onToggle: () => void;
}) {
  const [imgError, setImgError] = useState(false);
  const [frameIdx, setFrameIdx] = useState(0);
  const [hovered, setHovered] = useState(false);
  const preloadedRef = useRef(false);

  const cropUrls = useMemo(() => {
    const urls: string[] = [];
    if (!videoId) return urls;

    const samples = tracklet.sampleFrames;
    if (samples && samples.length > 0) {
      for (const sample of samples) {
        const url = uploadVideoCropUrl(videoId, sample.frameId, sample.bbox);
        if (url) urls.push(url);
      }
    }

    if (urls.length === 0) {
      const url = uploadVideoCropUrl(videoId, tracklet.representativeFrame, tracklet.representativeBbox);
      if (url) urls.push(url);
    }

    return urls;
  }, [tracklet.representativeBbox, tracklet.representativeFrame, tracklet.sampleFrames, videoId]);

  const currentUrl = cropUrls.length > 0 ? cropUrls[frameIdx % cropUrls.length] : null;

  // Preload the remaining sample frames only on first hover. An idle grid of many
  // tracklets must NOT fire a crop request for every frame of every card up front.
  useEffect(() => {
    if (!hovered || preloadedRef.current || cropUrls.length <= 1) return;
    preloadedRef.current = true;
    for (const url of cropUrls) {
      const img = new Image();
      img.src = url;
    }
  }, [hovered, cropUrls]);

  // Scrub through the sample frames ONLY while hovered. Inactive stages stay mounted in
  // this dashboard (hidden via CSS), so an always-on interval kept re-fetching crops
  useEffect(() => {
    if (!hovered || cropUrls.length <= 1) {
      setFrameIdx(0);
      return;
    }
    const id = setInterval(() => {
      setFrameIdx((index) => (index + 1) % cropUrls.length);
    }, 250);
    return () => clearInterval(id);
  }, [hovered, cropUrls.length]);

  return (
    <Card
      className={cn(
        "cursor-pointer transition-all hover:shadow-md",
        isSelected ? "border-success shadow-lg shadow-success/20" : "hover:border-destructive/50"
      )}
      onClick={onToggle}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onFocus={() => setHovered(true)}
      onBlur={() => setHovered(false)}
      role="button"
      tabIndex={0}
      aria-label={`${isSelected ? "Deselect" : "Select"} tracklet ${tracklet.id}`}
      aria-pressed={isSelected}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onToggle();
        }
      }}
    >
      <CardContent className="p-0">
        <div className="relative flex aspect-video items-center justify-center overflow-hidden rounded-t-lg bg-muted">
          {currentUrl && !imgError ? (
            <img
              src={currentUrl}
              alt={`${tracklet.className} Track #${tracklet.id}`}
              className="h-full w-full bg-black/40 object-contain"
              loading="lazy"
              decoding="async"
              onError={() => setImgError(true)}
            />
          ) : (
            <div className="flex flex-col items-center justify-center text-muted-foreground">
              <span className="text-xs">No image</span>
            </div>
          )}

          {cropUrls.length > 1 ? (
            <div className="absolute bottom-8 right-2 rounded bg-black/60 px-1.5 py-0.5">
              <span className="font-mono text-[10px] text-white">{(frameIdx % cropUrls.length) + 1}/{cropUrls.length}</span>
            </div>
          ) : null}

          <div className={cn("absolute right-2 top-2 flex h-6 w-6 items-center justify-center rounded-full", isSelected ? "bg-success" : "bg-destructive")}>
            {isSelected ? <CheckCircle2 className="h-4 w-4 text-white" /> : <XCircle className="h-4 w-4 text-white" />}
          </div>

          <Badge className="absolute bottom-2 left-2" style={{ backgroundColor: classColorFor(tracklet.classId) }}>
            {classLabelFor(tracklet.classId, tracklet.className)}
          </Badge>

          {tracklet.cameraId ? (
            <Badge variant="secondary" className="absolute left-2 top-2 gap-1 border-white/20 bg-black/70 font-mono text-[10px] text-white">
              <Video className="h-2.5 w-2.5" />
              {tracklet.cameraId}
            </Badge>
          ) : null}
        </div>

        <div className="space-y-1 p-3">
          <div className="flex items-center justify-between gap-3">
            <span className="truncate text-sm font-medium">Track #{tracklet.id}</span>
            <span className="text-sm text-muted-foreground">{(tracklet.confidence * 100).toFixed(0)}%</span>
          </div>
          <div className="flex items-center justify-between gap-3 text-xs text-muted-foreground">
            <span>{tracklet.numFrames} frames</span>
            <span>F{tracklet.startFrame}-{tracklet.endFrame}</span>
          </div>
          {tracklet.duration > 0 ? <div className="text-xs text-muted-foreground">{tracklet.duration.toFixed(1)}s duration</div> : null}
        </div>
      </CardContent>
    </Card>
  );
}
