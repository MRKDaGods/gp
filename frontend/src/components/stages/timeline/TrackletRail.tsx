"use client";

import type { RefObject } from "react";
import { Check, Loader2, X } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { cn, formatDuration, getCameraColor } from "@/lib/utils";
import type { StageProgress, TimelineTrack } from "@/types";

export interface TrackletRailProps {
  listRef?: RefObject<HTMLDivElement>;
  tracks: TimelineTrack[];
  visibleTracks: TimelineTrack[];
  tracksLoading: boolean;
  selectedTrackId: string | null;
  selectedTrackletCount: number;
  stage4Progress?: StageProgress;
  playingTrackletsOnly: boolean;
  activeAtPlayheadIds: Set<string>;
  onPlayingTrackletsOnlyChange: (value: boolean) => void;
  onSelectTrack: (trackId: string) => void;
  onConfirmToggle: (trackId: string, isConfirmed: boolean) => void;
  onRemoveTrack: (trackId: string) => void;
}

export function TrackletRail({
  listRef,
  tracks,
  visibleTracks,
  tracksLoading,
  selectedTrackId,
  selectedTrackletCount,
  stage4Progress,
  playingTrackletsOnly,
  activeAtPlayheadIds,
  onPlayingTrackletsOnlyChange,
  onSelectTrack,
  onConfirmToggle,
  onRemoveTrack,
}: TrackletRailProps) {
  return (
    <div ref={listRef} className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden p-4">
      <div className="mb-2 flex items-start gap-2">
        <Checkbox
          id="timeline-playing-only"
          className="mt-0.5"
          checked={playingTrackletsOnly}
          disabled={tracks.length === 0}
          onCheckedChange={(value) => onPlayingTrackletsOnlyChange(value === true)}
        />
        <Label
          htmlFor="timeline-playing-only"
          className="cursor-pointer text-xs font-normal leading-snug text-muted-foreground"
        >
          Adaptive mode: trajectory list and preview grid follow the playhead - only identities
          and cameras with live segments at the current video time (updates while scrubbing or playing).
        </Label>
      </div>
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <h4 className="text-sm font-medium">Trajectories</h4>
        {tracks.length > 0 && playingTrackletsOnly && (
          <Badge variant="outline" className="text-[10px] font-normal tabular-nums">
            {visibleTracks.length} at playhead
          </Badge>
        )}
      </div>
      {tracksLoading ? (
        <div className="mt-3 flex items-center gap-2 text-muted-foreground">
          <Loader2 className="h-4 w-4 shrink-0 animate-spin" />
          <span className="text-xs">Loading trajectories and previews...</span>
        </div>
      ) : tracks.length === 0 ? (
        <p className="mt-2 text-xs text-muted-foreground">
          {selectedTrackletCount > 0
            ? stage4Progress?.message || "No trajectories match selected tracklets. Run Stage 4 to associate them."
            : "No tracklet data yet."}
        </p>
      ) : playingTrackletsOnly && visibleTracks.length === 0 ? (
        <p className="mt-2 text-xs text-amber-600/90 dark:text-amber-400/90">
          No trajectory spans the current video time. Move the playhead to a segment.
        </p>
      ) : (
        <div className="space-y-2">
          {visibleTracks.map((track) => (
            <TrackletItem
              key={track.id}
              track={track}
              isSelected={selectedTrackId === track.id}
              isActiveAtPlayhead={!playingTrackletsOnly && activeAtPlayheadIds.has(track.id)}
              onClick={() => onSelectTrack(track.id)}
              onConfirm={() => onConfirmToggle(track.id, track.confirmed)}
              onRemove={() => onRemoveTrack(track.id)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

interface TrackletItemProps {
  track: TimelineTrack;
  isSelected: boolean;
  isActiveAtPlayhead?: boolean;
  onClick: () => void;
  onConfirm: () => void;
  onRemove: () => void;
}

function TrackletItem({
  track,
  isSelected,
  isActiveAtPlayhead = false,
  onClick,
  onConfirm,
  onRemove,
}: TrackletItemProps) {
  const cameraCount = track.segments ? new Set(track.segments.map((segment) => segment.cameraId)).size : 1;
  const primaryColor = track.segments?.[0]?.color ?? getCameraColor(track.cameraId);

  return (
    <div
      data-track-id={track.id}
      className={cn(
        "cursor-pointer rounded-lg border p-2 transition-all",
        isSelected && "border-primary bg-primary/5",
        track.confirmed && !isSelected && "border-green-500/50 bg-green-500/5",
        isActiveAtPlayhead && !isSelected && "border-l-2 border-l-emerald-500 bg-emerald-500/5"
      )}
      onClick={onClick}
    >
      <div className="flex items-center gap-2">
        <div className="flex flex-shrink-0 gap-0.5">
          {track.segments ? (
            Array.from(new Set(track.segments.map((segment) => segment.cameraId))).slice(0, 4).map((cameraId) => (
              <div key={cameraId} className="h-3 w-1.5 rounded-full" style={{ backgroundColor: getCameraColor(cameraId) }} />
            ))
          ) : (
            <div className="h-3 w-3 rounded-full" style={{ backgroundColor: primaryColor }} />
          )}
        </div>
        <div className="min-w-0 flex-1">
          <p className="truncate text-xs font-medium">{track.label ?? track.cameraId}</p>
          <p className="text-[10px] text-muted-foreground">
            {formatDuration(track.startTime)} - {formatDuration(track.endTime)}
            {cameraCount > 1 && <span className="ml-1 text-blue-400">- {cameraCount} cams</span>}
          </p>
          {typeof track.confidence === "number" && track.confidence > 0 && (
            <p className="text-[9px] text-muted-foreground">
              confidence: {(track.confidence * 100).toFixed(0)}%
            </p>
          )}
        </div>
        <div className="flex gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6"
            onClick={(event) => {
              event.stopPropagation();
              onConfirm();
            }}
          >
            <Check className={cn("h-3 w-3", track.confirmed ? "text-green-500" : "text-muted-foreground")} />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6"
            onClick={(event) => {
              event.stopPropagation();
              onRemove();
            }}
          >
            <X className="h-3 w-3 text-muted-foreground hover:text-red-500" />
          </Button>
        </div>
      </div>
    </div>
  );
}
