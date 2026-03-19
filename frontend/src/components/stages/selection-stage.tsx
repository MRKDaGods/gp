"use client";

import { useState, useEffect, useCallback } from "react";
import {
  CheckCircle2,
  XCircle,
  MousePointer2,
  Layers,
  ArrowRight,
  Loader2,
} from "lucide-react";
import { cn, getClassColor } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  useDetectionStore,
  useSessionStore,
  useVideoStore,
} from "@/store";
import { getTracklets } from "@/lib/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

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
    selectedIds,
    toggleSelection,
    selectAll,
    deselectAll,
    multiSelectMode,
    setMultiSelectMode,
    setDetectionsKeepSelection,
  } = useDetectionStore();
  const { setCurrentStage } = useSessionStore();
  const { currentVideo } = useVideoStore();

  const [tracklets, setTracklets] = useState<TrackletSummary[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchTracklets = useCallback(async () => {
    if (!currentVideo) return;
    setLoading(true);
    try {
      const resp: any = await getTracklets(undefined, currentVideo.id);
      const data = resp?.data ?? resp;
      const list: TrackletSummary[] = Array.isArray(data) ? data : [];
      setTracklets(list);

      // Map stage-1 detection IDs ("det-{trackId}-{frameId}") to tracklet IDs
      const prevSelected = useDetectionStore.getState().selectedIds;
      const trackIdSet = new Set<string>();
      prevSelected.forEach((detId) => {
        const m = detId.match(/^det-(\d+)-/);
        if (m) trackIdSet.add(m[1]);
        else trackIdSet.add(detId); // already a plain tracklet id
      });

      // Build new detections keyed by tracklet id
      const newDetections = list.map((t) => ({
        id: String(t.id),
        bbox: {
          x1: t.representativeBbox?.[0] ?? 0,
          y1: t.representativeBbox?.[1] ?? 0,
          x2: t.representativeBbox?.[2] ?? 0,
          y2: t.representativeBbox?.[3] ?? 0,
        },
        confidence: t.confidence,
        classId: t.classId ?? 2,
        className: t.className ?? "vehicle",
        frameId: t.representativeFrame ?? t.startFrame ?? 0,
      }));

      // Set detections and manually restore matched selections
      setDetectionsKeepSelection(newDetections);

      // Re-select tracklets whose track_id was selected in stage 1
      const matchedIds = newDetections
        .filter((d) => trackIdSet.has(d.id))
        .map((d) => d.id);
      if (matchedIds.length > 0) {
        // Enable multi-select so toggleSelection adds instead of replacing
        setMultiSelectMode(true);
        matchedIds.forEach((id) => {
          if (!useDetectionStore.getState().selectedIds.has(id)) {
            toggleSelection(id);
          }
        });
      }
    } catch {
      setTracklets([]);
    } finally {
      setLoading(false);
    }
  }, [currentVideo, setDetectionsKeepSelection, setMultiSelectMode, toggleSelection]);

  useEffect(() => {
    void fetchTracklets();
  }, [fetchTracklets]);

  const selectedTracklets = tracklets.filter((t) => selectedIds.has(String(t.id)));

  // Group selected by class
  const groupedSelected = selectedTracklets.reduce((acc, t) => {
    const cls = t.className ?? "vehicle";
    if (!acc[cls]) acc[cls] = [];
    acc[cls].push(t);
    return acc;
  }, {} as Record<string, TrackletSummary[]>);

  const handleProceed = () => {
    if (selectedIds.size > 0) {
      setCurrentStage(3);
    }
  };

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        <span className="ml-3 text-muted-foreground">Loading tracklets...</span>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <header className="flex h-14 items-center justify-between border-b px-6">
        <div>
          <h1 className="text-lg font-semibold">Stage 2: Tracklet Selection</h1>
          <p className="text-sm text-muted-foreground">
            Select vehicle tracklets to track across cameras
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline">
            {selectedIds.size} of {tracklets.length} selected
          </Badge>
          <Button onClick={handleProceed} disabled={selectedIds.size === 0}>
            Run Inference
            <ArrowRight className="ml-2 h-4 w-4" />
          </Button>
        </div>
      </header>

      {/* Main content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Main selection area */}
        <div className="flex-1 p-6 overflow-auto">
          {/* Controls */}
          <Card className="mb-6">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-6">
                  <div className="flex items-center gap-2">
                    <Checkbox
                      id="multi-select"
                      checked={multiSelectMode}
                      onCheckedChange={(checked) =>
                        setMultiSelectMode(checked === true)
                      }
                    />
                    <Label htmlFor="multi-select" className="text-sm">
                      Multi-select mode
                    </Label>
                  </div>
                  <Separator orientation="vertical" className="h-6" />
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <div className="flex items-center gap-1">
                      <div className="h-3 w-3 rounded-full bg-red-500" />
                      <span>Unselected</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <div className="h-3 w-3 rounded-full bg-green-500" />
                      <span>Selected</span>
                    </div>
                  </div>
                </div>
                <div className="flex gap-2">
                  <Button variant="outline" size="sm" onClick={selectAll}>
                    <CheckCircle2 className="mr-2 h-4 w-4" />
                    Select All
                  </Button>
                  <Button variant="outline" size="sm" onClick={deselectAll}>
                    <XCircle className="mr-2 h-4 w-4" />
                    Deselect All
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          {tracklets.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
              <MousePointer2 className="h-12 w-12 mb-4 opacity-50" />
              <p className="text-lg font-medium">No tracklets found</p>
              <p className="text-sm">Run detection first (Stage 1) to generate tracklets.</p>
            </div>
          ) : (
            <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {tracklets.map((tracklet) => {
                const isSelected = selectedIds.has(String(tracklet.id));
                return (
                  <TrackletCard
                    key={tracklet.id}
                    tracklet={tracklet}
                    videoId={currentVideo?.id}
                    isSelected={isSelected}
                    onToggle={() => toggleSelection(String(tracklet.id))}
                  />
                );
              })}
            </div>
          )}
        </div>

        {/* Sidebar - Selected summary */}
        <aside className="w-80 border-l flex flex-col">
          <div className="p-4 border-b">
            <h3 className="font-semibold flex items-center gap-2">
              <Layers className="h-4 w-4" />
              Selected Tracklets
            </h3>
            <p className="text-sm text-muted-foreground">
              {selectedIds.size} tracklets will be tracked
            </p>
          </div>

          <ScrollArea className="flex-1">
            <div className="p-4 space-y-4">
              {Object.entries(groupedSelected).map(([className, items]) => (
                <div key={className}>
                  <div className="flex items-center gap-2 mb-2">
                    <div
                      className="h-3 w-3 rounded-full"
                      style={{ backgroundColor: getClassColor(items[0].classId ?? 2) }}
                    />
                    <span className="font-medium capitalize">{className}</span>
                    <Badge variant="secondary" className="ml-auto">
                      {items.length}
                    </Badge>
                  </div>
                  <div className="space-y-1 pl-5">
                    {items.map((item) => (
                      <div
                        key={item.id}
                        className="flex items-center justify-between text-sm py-1 px-2 rounded bg-muted/50"
                      >
                        <span className="text-muted-foreground">Track #{item.id}</span>
                        <span>{item.numFrames} frames</span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}

              {selectedIds.size === 0 && (
                <div className="text-center py-8 text-muted-foreground">
                  <MousePointer2 className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p>No tracklets selected</p>
                  <p className="text-sm">Click on tracklets to select them</p>
                </div>
              )}
            </div>
          </ScrollArea>

          {/* Action footer */}
          <div className="p-4 border-t">
            <Button
              className="w-full"
              onClick={handleProceed}
              disabled={selectedIds.size === 0}
            >
              Continue with {selectedIds.size} tracklets
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </div>
        </aside>
      </div>
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

  // Build array of crop URLs from sampleFrames (fall back to single representative)
  const cropUrls: string[] = [];
  if (videoId) {
    const samples = tracklet.sampleFrames;
    if (samples && samples.length > 0) {
      for (const sf of samples) {
        if (sf.bbox && sf.bbox.length === 4) {
          cropUrls.push(
            `${API_BASE}/crops/${videoId}?frameId=${sf.frameId}&x1=${sf.bbox[0]}&y1=${sf.bbox[1]}&x2=${sf.bbox[2]}&y2=${sf.bbox[3]}`
          );
        }
      }
    }
    if (cropUrls.length === 0) {
      const bbox = tracklet.representativeBbox;
      if (bbox && bbox.length === 4) {
        cropUrls.push(
          `${API_BASE}/crops/${videoId}?frameId=${tracklet.representativeFrame}&x1=${bbox[0]}&y1=${bbox[1]}&x2=${bbox[2]}&y2=${bbox[3]}`
        );
      }
    }
  }

  const currentUrl = cropUrls.length > 0 ? cropUrls[frameIdx % cropUrls.length] : null;

  // Preload all crop images for smooth playback
  useEffect(() => {
    if (cropUrls.length <= 1) { setPreloaded(true); return; }
    let loaded = 0;
    for (const url of cropUrls) {
      const img = new Image();
      img.src = url;
      img.onload = img.onerror = () => { if (++loaded >= cropUrls.length) setPreloaded(true); };
    }
  }, [cropUrls.length, videoId]);

  // Auto-loop through frames continuously like a video (~4 FPS)
  useEffect(() => {
    if (cropUrls.length <= 1 || !preloaded) return;
    const id = setInterval(() => {
      setFrameIdx((i) => (i + 1) % cropUrls.length);
    }, 250);
    return () => clearInterval(id);
  }, [cropUrls.length, preloaded]);

  return (
    <Card
      className={cn(
        "cursor-pointer transition-all hover:shadow-md",
        isSelected
          ? "border-green-500 shadow-green-500/20 shadow-lg"
          : "hover:border-red-500/50"
      )}
      onClick={onToggle}
    >
      <CardContent className="p-0">
        {/* Vehicle crop thumbnail */}
        <div
          className={cn(
            "relative aspect-video flex items-center justify-center",
            "overflow-hidden rounded-t-lg bg-muted"
          )}
        >
          {currentUrl && !imgError ? (
            <img
              src={currentUrl}
              alt={`${tracklet.className} Track #${tracklet.id}`}
              className="h-full w-full object-cover"
              loading="lazy"
              onError={() => setImgError(true)}
            />
          ) : (
            <div className="flex flex-col items-center justify-center text-muted-foreground">
              <span className="text-2xl">🚗</span>
              <span className="text-xs mt-1">No image</span>
            </div>
          )}
          {/* Frame counter */}
          {cropUrls.length > 1 && (
            <div className="absolute bottom-8 right-2 bg-black/60 rounded px-1.5 py-0.5">
              <span className="text-white text-[10px] font-mono">
                {(frameIdx % cropUrls.length) + 1}/{cropUrls.length}
              </span>
            </div>
          )}

          {/* Selection indicator */}
          <div
            className={cn(
              "absolute top-2 right-2 h-6 w-6 rounded-full flex items-center justify-center",
              isSelected ? "bg-green-500" : "bg-red-500"
            )}
          >
            {isSelected ? (
              <CheckCircle2 className="h-4 w-4 text-white" />
            ) : (
              <XCircle className="h-4 w-4 text-white" />
            )}
          </div>

          {/* Class badge */}
          <Badge
            className="absolute bottom-2 left-2"
            style={{ backgroundColor: getClassColor(tracklet.classId ?? 2) }}
          >
            {tracklet.className ?? "vehicle"}
          </Badge>
        </div>

        {/* Info */}
        <div className="p-3 space-y-1">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Track #{tracklet.id}</span>
            <span className="text-sm text-muted-foreground">
              {(tracklet.confidence * 100).toFixed(0)}%
            </span>
          </div>
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>{tracklet.numFrames} frames</span>
            <span>
              F{tracklet.startFrame}–{tracklet.endFrame}
            </span>
          </div>
          {tracklet.duration > 0 && (
            <div className="text-xs text-muted-foreground">
              {tracklet.duration.toFixed(1)}s duration
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
