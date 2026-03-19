"use client";

import { useState, useMemo } from "react";
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
} from "@/store";



export function RefinementStage() {
  const { tracks, confirmedTracks } = useTimelineStore();
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

  const confirmedTrackList = tracks.filter((t) => t.confirmed);
  const selectedFrameCount = refinementFrames.length;
  const maxFrames = 16;

  // Build frames from confirmed tracks (real data, no mocks)
  const refinementCandidateFrames = useMemo(() => {
    if (confirmedTrackList.length === 0) return [];
    return confirmedTrackList.flatMap((track) => {
      const numFrames = Math.max(1, Math.round((track.endTime - track.startTime) * 2));
      return Array.from({ length: Math.min(numFrames, 10) }, (_, i) => ({
        id: `${track.id}-frame-${i}`,
        frameId: Math.round(track.startTime * 10 + i * ((track.endTime - track.startTime) * 10 / Math.max(numFrames, 1))),
        timestamp: track.startTime + i * ((track.endTime - track.startTime) / Math.max(numFrames, 1)),
        cameraId: track.cameraId,
        thumbnail: undefined as string | undefined,
      }));
    });
  }, [confirmedTrackList]);

  const handleFrameSelect = (frameId: string) => {
    if (refinementFrames.includes(frameId)) {
      removeRefinementFrame(frameId);
    } else if (selectedFrameCount < maxFrames) {
      addRefinementFrame(frameId);
    }
  };

  const handleReSearch = () => {
    // Trigger re-search with selected frames
    console.log("Re-searching with frames:", refinementFrames);
  };

  const handleProceed = () => {
    setCurrentStage(6);
  };

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <header className="flex h-14 items-center justify-between border-b px-6">
        <div>
          <h1 className="text-lg font-semibold">Stage 5: Refinement</h1>
          <p className="text-sm text-muted-foreground">
            Select reference frames for improved search accuracy
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="secondary">
            {selectedFrameCount}/{maxFrames} frames selected
          </Badge>
          <Button onClick={handleProceed}>
            Continue to Output
            <ArrowRight className="ml-2 h-4 w-4" />
          </Button>
        </div>
      </header>

      {/* Main content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left panel - Confirmed tracklets */}
        <aside className="w-64 border-r flex flex-col">
          <div className="p-4 border-b">
            <h3 className="font-semibold">Confirmed Tracklets</h3>
            <p className="text-sm text-muted-foreground">
              {confirmedTrackList.length} reference clips
            </p>
          </div>
          <ScrollArea className="flex-1">
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
        <div className="flex-1 flex flex-col">
          {/* Frame viewer */}
          <div className="flex-1 p-4">
            <div className="grid grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
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
          <div className="border-t p-4">
            <div className="flex items-center gap-4">
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
              <div className="flex-1">
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
          <ScrollArea className="flex-1">
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
                        <div className="absolute inset-0 flex items-center justify-center">
                          <Image className="h-6 w-6 text-muted-foreground/50" />
                        </div>
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
            <Button
              className="w-full"
              onClick={handleReSearch}
              disabled={refinementFrames.length === 0}
            >
              <Search className="mr-2 h-4 w-4" />
              Re-Search with Selected
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
      {/* Thumbnail placeholder */}
      <div className="absolute inset-0 bg-muted flex items-center justify-center">
        <Image className="h-8 w-8 text-muted-foreground/30" />
      </div>

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
