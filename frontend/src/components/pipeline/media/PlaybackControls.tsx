"use client";

import { Pause, Play, SkipBack, SkipForward } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { cn } from "@/lib/utils";

export interface PlaybackControlsProps {
  isPlaying?: boolean;
  currentFrame?: number;
  totalFrames?: number;
  positionLabel?: string;
  speed?: number;
  speedOptions?: number[];
  onPlayPause?: () => void;
  onFrameChange?: (frame: number) => void;
  onStepBack?: () => void;
  onStepForward?: () => void;
  onSpeedChange?: (speed: number) => void;
  className?: string;
}

export function PlaybackControls({
  isPlaying = false,
  currentFrame = 0,
  totalFrames = 0,
  positionLabel,
  speed = 1,
  speedOptions = [0.5, 1, 2, 4],
  onPlayPause,
  onFrameChange,
  onStepBack,
  onStepForward,
  onSpeedChange,
  className,
}: PlaybackControlsProps) {
  const maxFrame = Math.max(0, totalFrames - 1);

  return (
    <div className={cn("flex flex-col gap-3 rounded-md border bg-card p-3 sm:flex-row sm:items-center", className)}>
      <div className="flex items-center gap-1">
        <Button type="button" variant="ghost" size="icon" className="h-8 w-8" onClick={onStepBack} aria-label="Previous frame">
          <SkipBack className="h-4 w-4" />
        </Button>
        <Button type="button" variant="outline" size="icon" className="h-8 w-8" onClick={onPlayPause} aria-label={isPlaying ? "Pause" : "Play"}>
          {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
        </Button>
        <Button type="button" variant="ghost" size="icon" className="h-8 w-8" onClick={onStepForward} aria-label="Next frame">
          <SkipForward className="h-4 w-4" />
        </Button>
      </div>
      <div className="flex min-w-0 flex-1 items-center gap-3">
        <Slider
          value={[Math.min(currentFrame, maxFrame)]}
          min={0}
          max={maxFrame}
          step={1}
          onValueChange={([value]) => onFrameChange?.(value ?? 0)}
          aria-label="Playback frame"
        />
        <span className="w-24 text-right font-mono text-xs text-muted-foreground">
          {positionLabel ?? `${currentFrame}/${maxFrame}`}
        </span>
      </div>
      {onSpeedChange ? (
        <div className="flex flex-wrap items-center gap-1">
          {speedOptions.map((option) => (
            <Button
              key={option}
              type="button"
              variant={option === speed ? "secondary" : "ghost"}
              size="sm"
              className="h-8 px-2 text-xs"
              onClick={() => onSpeedChange(option)}
            >
              {option}x
            </Button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
