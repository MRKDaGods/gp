"use client";

import type { RefObject } from "react";

import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { cn, formatDuration } from "@/lib/utils";

import type { TimelineCameraLane } from "./types";

export interface NLETimelineProps {
  timelineRef?: RefObject<HTMLDivElement>;
  cameraLanes: TimelineCameraLane[];
  selectedLaneId: string | null;
  timelineStart: number;
  timelineEnd: number;
  currentTime: number;
  playheadVideoTime: number;
  rulerTickCount: number;
  rulerTickInterval: number;
  rulerPlayheadLeft: number;
  timeToPixel: (time: number) => number;
  onLaneClick: (laneId: string) => void;
}

export function NLETimeline({
  timelineRef,
  cameraLanes,
  selectedLaneId,
  timelineStart,
  timelineEnd,
  currentTime,
  playheadVideoTime,
  rulerTickCount,
  rulerTickInterval,
  rulerPlayheadLeft,
  timeToPixel,
  onLaneClick,
}: NLETimelineProps) {
  return (
    <div className="flex min-h-0 min-w-0 flex-1 overflow-hidden border-t border-border/50 bg-background">
      <div
        className="flex w-[5.5rem] shrink-0 flex-col border-r border-border/60 bg-muted/25"
        aria-label="Camera lanes"
      >
        <div className="flex h-10 shrink-0 items-end border-b border-border/60 pb-1 pl-2.5">
          <span className="text-[9px] font-semibold uppercase tracking-wider text-muted-foreground">
            Camera
          </span>
        </div>
        {cameraLanes.map((lane) => {
          const confidences = lane.segments
            .map((segment) => Number(segment.confidence ?? 0))
            .filter((value) => Number.isFinite(value) && value > 0);
          const best = confidences.length > 0 ? Math.max(...confidences) : 0;
          const isSelected = selectedLaneId === lane.id;
          return (
            <button
              key={lane.id}
              type="button"
              className={cn(
                "flex h-10 shrink-0 flex-col items-stretch justify-center border-b border-border/50 px-2.5 text-left transition-colors",
                "hover:bg-muted/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring",
                isSelected && "bg-primary/12"
              )}
              onClick={() => onLaneClick(lane.id)}
              title={lane.label}
            >
              <span className="truncate font-mono text-[11px] font-semibold leading-tight text-foreground">
                {lane.cameraId}
              </span>
              {best > 0 && (
                <span className="text-[9px] tabular-nums text-muted-foreground">
                  {Math.round(best * 100)}% match
                </span>
              )}
            </button>
          );
        })}
      </div>

      <ScrollArea className="h-full min-h-0 min-w-0 flex-1">
        <div ref={timelineRef} className="min-w-max pb-3 pl-2 pr-4 pt-2">
          <div
            className="relative mb-0 h-10 overflow-visible border-b border-border/30 bg-muted/5"
            style={{ width: timeToPixel(timelineEnd) }}
          >
            {Array.from({ length: rulerTickCount }).map((_, index) => {
              const absoluteTime = timelineStart + index * rulerTickInterval;
              return (
                <div
                  key={index}
                  className="absolute bottom-0 flex -translate-x-1/2 flex-col items-center"
                  style={{ left: timeToPixel(absoluteTime) }}
                >
                  <span className="mb-0.5 select-none text-[9px] tabular-nums tracking-tight text-muted-foreground/45">
                    {formatDuration(absoluteTime)}
                  </span>
                  <div className="h-1.5 w-px bg-border/80" />
                </div>
              );
            })}
            <div
              className="pointer-events-none absolute inset-y-0 z-30 w-px -translate-x-1/2 bg-foreground/32"
              style={{ left: rulerPlayheadLeft }}
              aria-hidden
            />
          </div>

          <div className="flex flex-col">
            {cameraLanes.map((lane) => (
              <TimelineRow
                key={lane.id}
                lane={lane}
                timelineEnd={timelineEnd}
                isSelected={selectedLaneId === lane.id}
                onClick={() => onLaneClick(lane.id)}
                timeToPixel={timeToPixel}
                currentTime={currentTime}
                playheadVideoTime={playheadVideoTime}
              />
            ))}
          </div>
        </div>
        <ScrollBar orientation="horizontal" />
      </ScrollArea>
    </div>
  );
}

interface TimelineRowProps {
  lane: TimelineCameraLane;
  timelineEnd: number;
  isSelected: boolean;
  onClick: () => void;
  timeToPixel: (time: number) => number;
  currentTime: number;
  playheadVideoTime: number;
}

function TimelineRow({
  lane,
  timelineEnd,
  isSelected,
  onClick,
  timeToPixel,
  currentTime,
  playheadVideoTime,
}: TimelineRowProps) {
  const isCurrentlyActive = lane.segments.some(
    (segment) => playheadVideoTime >= segment.start && playheadVideoTime <= segment.end
  );

  return (
    <button
      type="button"
      className={cn(
        "relative h-10 cursor-pointer border-b border-border/40 text-left transition-colors",
        "hover:bg-muted/30 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring",
        isSelected && "bg-primary/8",
        isCurrentlyActive && "bg-muted/20"
      )}
      style={{ width: timeToPixel(timelineEnd) }}
      onClick={onClick}
      title={lane.label}
    >
      <div className="pointer-events-none absolute inset-x-0 top-1/2 h-6 -translate-y-1/2 bg-muted/40" />
      {lane.segments.map((segment, index) => {
        const segmentLeft = timeToPixel(segment.sumStart);
        const segmentWidth = Math.max(timeToPixel(segment.sumEnd) - timeToPixel(segment.sumStart), 3);
        const isSegmentActive = playheadVideoTime >= segment.start && playheadVideoTime <= segment.end;

        return (
          <div
            key={`${segment.cameraId}-${segment.trackId}-${segment.trajectoryId}-${index}`}
            className={cn(
              "absolute top-1/2 h-2.5 -translate-y-1/2 rounded-full border transition-shadow",
              "border-black/20 shadow-sm",
              isSegmentActive && "z-[5] h-3 shadow-md ring-2 ring-white/50",
              segment.confirmed && "ring-1 ring-green-500/80 ring-offset-1 ring-offset-background"
            )}
            style={{
              left: segmentLeft,
              width: segmentWidth,
              backgroundColor: segment.color,
              opacity: isSegmentActive ? 1 : 0.72,
            }}
            title={`G-${String(segment.globalId ?? 0).padStart(4, "0")} - ${formatDuration(segment.start)} to ${formatDuration(segment.end)}`}
          />
        );
      })}

      <div
        className="pointer-events-none absolute inset-y-0 z-20 w-px -translate-x-1/2 bg-foreground/22"
        style={{ left: timeToPixel(currentTime) }}
        aria-hidden
      />
    </button>
  );
}
