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

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8004/api";

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
    selectAll,
    deselectAll,
    multiSelectMode,
    setMultiSelectMode,
    setDetections,             
  } = useDetectionStore();
  const { setCurrentStage } = useSessionStore();
  const { currentVideo } = useVideoStore();

  const [tracklets, setTracklets] = useState<TrackletSummary[]>([]);
  const [loading, setLoading] = useState(true);

  // Selection stage always uses multi-select so users can pick multiple tracklets
  useEffect(() => {
    setMultiSelectMode(true);
  }, [setMultiSelectMode]);

    const fetchTracklets = useCallback(async () => {
    if (!currentVideo) return;
    setLoading(true);
    try {
      const resp: any = await getTracklets(undefined, currentVideo.id);
      const data = resp?.data ?? resp;
      const list: TrackletSummary[] = Array.isArray(data) ? data : [];
      setTracklets(list);
    } catch {
      setTracklets([]);
    } finally {
      setLoading(false);
    }
  }, [currentVideo]);

  useEffect(() => {
    void fetchTracklets();
  }, [fetchTracklets]);

  const selectedTracklets = tracklets.filter((t) => selectedTrackIds.has(t.id));
  // Group selected by class
  const groupedSelected = selectedTracklets.reduce((acc, t) => {
    const cls = t.className ?? "vehicle";
    if (!acc[cls]) acc[cls] = [];
    acc[cls].push(t);
    return acc;
  }, {} as Record<string, TrackletSummary[]>);

  const handleProceed = () => {
    if (selectedTrackIds.size > 0) {
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
    <div className="flex h-full min-h-0 flex-col overflow-hidden">
      {/* Header */}
      <header className="flex shrink-0 flex-col gap-3 border-b px-4 py-3 sm:flex-row sm:items-center sm:justify-between sm:px-6">
        <div className="min-w-0">
          <h1 className="text-lg font-semibold">Stage 2: Tracklet Selection</h1>
          <p className="text-sm text-muted-foreground">
            Select vehicle tracklets to track across cameras
          </p>
        </div>
        <div className="flex flex-wrap items-center justify-end gap-2">
          <Badge variant="outline">
            {selectedTrackIds.size} of {tracklets.length} selected
          </Badge>
          <Button className="shrink-0" onClick={handleProceed} disabled={selectedTrackIds.size === 0}>
            Run Inference
            <ArrowRight className="ml-2 h-4 w-4" />
          </Button>
        </div>
      </header>

      {/* Main content */}
      <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden xl:flex-row">
        {/* Main selection area */}
        <div className="min-h-[200px] min-w-0 flex-1 overflow-x-hidden overflow-y-auto p-4 sm:p-6 xl:min-h-0">
          {/* Controls */}
          <Card className="mb-6">
            <CardContent className="p-4">
              <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                <div className="flex min-w-0 flex-wrap items-center gap-4 sm:gap-6">
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
                <div className="flex shrink-0 flex-wrap gap-2">
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
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {tracklets.map((tracklet) => {
                  const isSelected = selectedTrackIds.has(tracklet.id);          // ← number, not String()
                  return (
                  <TrackletCard
                    key={tracklet.id}
                    tracklet={tracklet}
                    videoId={currentVideo?.id}
                    isSelected={isSelected}
                    onToggle={() => toggleTrackSelection(tracklet.id)}
                  />
                );
              })}
            </div>
          )}
        </div>

        {/* Sidebar - Selected summary */}
        <aside className="flex max-h-[40vh] min-h-0 w-full shrink-0 flex-col border-t border-border xl:max-h-none xl:w-80 xl:border-l xl:border-t-0">
          <div className="shrink-0 border-b p-4">
            <h3 className="font-semibold flex items-center gap-2">
              <Layers className="h-4 w-4" />
              Selected Tracklets
            </h3>
            <p className="text-sm text-muted-foreground">
              {selectedTrackIds.size} tracklets will be tracked
            </p>
          </div>

          <ScrollArea className="min-h-0 flex-1">
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

              {selectedTrackIds.size === 0 && (
                <div className="text-center py-8 text-muted-foreground">
                  <MousePointer2 className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p>No tracklets selected</p>
                  <p className="text-sm">Click on tracklets to select them</p>
                </div>
              )}
            </div>
          </ScrollArea>

          {/* Action footer */}
          <div className="shrink-0 border-t p-4">
            <Button
              className="w-full"
              onClick={handleProceed}
              disabled={selectedTrackIds.size === 0}
            >
              Continue with {selectedTrackIds.size} tracklets
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
