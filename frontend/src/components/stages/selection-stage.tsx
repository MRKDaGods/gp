"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CheckCircle2, Layers, Loader2, MousePointer2, RefreshCw, X, XCircle } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { DisclosurePanel, ErrorBanner } from "@/components/pipeline";
import { apiUrl, getTracklets } from "@/lib/api";
import { flushPipelineFromStage } from "@/lib/pipeline-flush";
import { cn, getClassColor } from "@/lib/utils";
import { useDetectionStore, useSessionStore, useVideoStore } from "@/store";

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
  const { currentVideo } = useVideoStore();

  const [tracklets, setTracklets] = useState<TrackletSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

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
      const resp = await getTracklets(undefined, currentVideo.id);
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
    flushPipelineFromStage(2);
  }, [selectionSig]);

  const selectedTracklets = useMemo(
    () => tracklets.filter((tracklet) => selectedTrackIds.has(tracklet.id)),
    [selectedTrackIds, tracklets]
  );

  const groupedSelected = selectedTracklets.reduce((acc, tracklet) => {
    const className = tracklet.className ?? "vehicle";
    acc[className] = [...(acc[className] ?? []), tracklet];
    return acc;
  }, {} as Record<string, TrackletSummary[]>);

  const selectAllTracklets = () => {
    tracklets.forEach((tracklet) => {
      if (!selectedTrackIds.has(tracklet.id)) toggleTrackSelection(tracklet.id);
    });
  };

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden">
      <div className="shrink-0 space-y-3 border-b px-4 py-3 sm:px-6">
        <ErrorBanner title="Tracklet loading failed" message={loadError} />
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant="outline">{selectedTrackIds.size} of {tracklets.length} selected</Badge>
          {Array.from(selectedTrackIds).sort((a, b) => a - b).slice(0, 18).map((id) => (
            <button
              key={id}
              type="button"
              onClick={() => toggleTrackSelection(id)}
              className="group flex items-center gap-1 rounded-full border bg-muted/50 px-2 py-1 text-[11px] font-mono transition-colors hover:border-destructive/30 hover:bg-destructive/10"
              aria-label={`Remove tracklet ${id} from selection`}
            >
              #{id}
              <X className="h-3 w-3 text-muted-foreground group-hover:text-destructive" />
            </button>
          ))}
          {selectedTrackIds.size > 18 ? <Badge variant="secondary">+{selectedTrackIds.size - 18}</Badge> : null}
        </div>
      </div>

      {loading ? (
        <div className="flex h-full items-center justify-center">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
          <span className="ml-3 text-muted-foreground">Loading tracklets...</span>
        </div>
      ) : (
        <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden xl:flex-row">
          <div className="min-h-[200px] min-w-0 flex-1 overflow-x-hidden overflow-y-auto p-4 sm:p-6 xl:min-h-0">
            {tracklets.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
                <MousePointer2 className="mb-4 h-12 w-12 opacity-50" />
                <p className="text-lg font-medium">No tracklets found</p>
                <p className="text-sm">Run detection first to generate tracklets.</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                {tracklets.map((tracklet) => (
                  <TrackletCard
                    key={tracklet.id}
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
                      <div className="h-3 w-3 rounded-full" style={{ backgroundColor: getClassColor(items[0].classId ?? 2) }} />
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

  return (
    <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
      <Button type="button" onClick={() => setCurrentStage(3)} disabled={selectedTrackIds.size === 0} aria-label="Continue to Stage 3 inference">
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
  const [preloaded, setPreloaded] = useState(false);

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

  useEffect(() => {
    if (cropUrls.length <= 1) {
      setPreloaded(true);
      return;
    }
    let loaded = 0;
    for (const url of cropUrls) {
      const img = new Image();
      img.src = url;
      img.onload = img.onerror = () => {
        loaded += 1;
        if (loaded >= cropUrls.length) setPreloaded(true);
      };
    }
  }, [cropUrls]);

  useEffect(() => {
    if (cropUrls.length <= 1 || !preloaded) return;
    const id = setInterval(() => {
      setFrameIdx((index) => (index + 1) % cropUrls.length);
    }, 250);
    return () => clearInterval(id);
  }, [cropUrls.length, preloaded]);

  return (
    <Card
      className={cn(
        "cursor-pointer transition-all hover:shadow-md",
        isSelected ? "border-green-500 shadow-lg shadow-green-500/20" : "hover:border-red-500/50"
      )}
      onClick={onToggle}
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

          <div className={cn("absolute right-2 top-2 flex h-6 w-6 items-center justify-center rounded-full", isSelected ? "bg-green-500" : "bg-red-500")}>
            {isSelected ? <CheckCircle2 className="h-4 w-4 text-white" /> : <XCircle className="h-4 w-4 text-white" />}
          </div>

          <Badge className="absolute bottom-2 left-2" style={{ backgroundColor: getClassColor(tracklet.classId ?? 2) }}>
            {tracklet.className ?? "vehicle"}
          </Badge>
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
