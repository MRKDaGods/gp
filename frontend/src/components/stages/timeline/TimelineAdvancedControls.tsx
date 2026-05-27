"use client";

import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";

export interface TimelineAdvancedControlsProps {
  effectiveSplitCount: number;
  cameraCount: number;
  playbackFilterActive: boolean;
  playingTrackletsOnly: boolean;
  zoom: number;
  tracksCount: number;
  onSplitCountChange: (count: number) => void;
  onPlayingTrackletsOnlyChange: (value: boolean) => void;
  onZoomChange: (zoom: number) => void;
}

export function TimelineAdvancedControls({
  effectiveSplitCount,
  cameraCount,
  playbackFilterActive,
  playingTrackletsOnly,
  zoom,
  tracksCount,
  onSplitCountChange,
  onPlayingTrackletsOnlyChange,
  onZoomChange,
}: TimelineAdvancedControlsProps) {
  return (
    <div className="grid gap-4 md:grid-cols-2">
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Cameras</span>
          <span className="text-sm font-medium">{effectiveSplitCount}</span>
        </div>
        {playbackFilterActive && (
          <p className="text-[10px] leading-snug text-muted-foreground">
            Playback: {cameraCount} camera{cameraCount !== 1 ? "s" : ""} with active tracklets now
          </p>
        )}
        {playingTrackletsOnly ? (
          <p className="text-[10px] leading-snug text-muted-foreground">
            Grid size follows active cameras (slider off in adaptive mode).
          </p>
        ) : null}
        <Slider
          value={[Math.min(effectiveSplitCount, 8)]}
          min={1}
          max={8}
          step={1}
          disabled={playingTrackletsOnly}
          onValueChange={(value) => onSplitCountChange(value[0])}
        />
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Zoom</span>
          <span className="text-sm font-medium">{(zoom * 100).toFixed(0)}%</span>
        </div>
        <Slider value={[zoom]} min={0.5} max={4} step={0.25} onValueChange={(value) => onZoomChange(value[0])} />
      </div>

      <div className="flex items-start gap-2 md:col-span-2">
        <Checkbox
          id="timeline-playing-only-advanced"
          className="mt-0.5"
          checked={playingTrackletsOnly}
          disabled={tracksCount === 0}
          onCheckedChange={(value) => onPlayingTrackletsOnlyChange(value === true)}
        />
        <Label
          htmlFor="timeline-playing-only-advanced"
          className="cursor-pointer text-xs font-normal leading-snug text-muted-foreground"
        >
          Playing tracklets only: trajectory list and preview grid follow the playhead.
        </Label>
      </div>
    </div>
  );
}
