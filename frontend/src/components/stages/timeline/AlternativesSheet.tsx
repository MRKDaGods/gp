"use client";

import { Loader2 } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { getMatchedAlternativeClipUrl, type MatchedAlternative } from "@/lib/api";
import type { TimelineTrack } from "@/types";

export interface AlternativesPanelProps {
  selectedTrack: TimelineTrack | null | undefined;
  alternatives: MatchedAlternative[];
  alternativesLoading: boolean;
  alternativesError: string | null;
  alternativesCameraCount: number;
  probeRunId?: string | null;
  onApplyAlternative: (alternative: MatchedAlternative) => void;
}

export function AlternativesPanel({
  selectedTrack,
  alternatives,
  alternativesLoading,
  alternativesError,
  alternativesCameraCount,
  probeRunId,
  onApplyAlternative,
}: AlternativesPanelProps) {
  return (
    <div>
      <div className="flex items-center justify-between gap-2">
        <h4 className="text-sm font-medium">Top 5 Alternatives</h4>
        {selectedTrack ? (
          <Badge variant="outline" className="text-[10px]">
            {selectedTrack.cameraId} - #{selectedTrack.trackletId} - {alternativesCameraCount || "-"} cams
          </Badge>
        ) : null}
      </div>

      {!selectedTrack ? (
        <p className="mt-2 text-xs text-muted-foreground">
          Select a trajectory to load alternatives from matched/top5_alternatives.
        </p>
      ) : alternativesLoading ? (
        <div className="mt-2 flex items-center gap-2 text-muted-foreground">
          <Loader2 className="h-4 w-4 shrink-0 animate-spin" />
          <span className="text-xs">Loading top alternatives...</span>
        </div>
      ) : alternativesError ? (
        <p className="mt-2 text-xs text-muted-foreground">{alternativesError}</p>
      ) : alternatives.length === 0 ? (
        <p className="mt-2 text-xs text-muted-foreground">No alternative clips were found for this selection.</p>
      ) : (
        <div className="mt-2 space-y-2">
          {alternatives.map((alternative) => (
            <AlternativeTrackletItem
              key={`${alternative.rank}-${alternative.cameraId}-${alternative.trackId}-${alternative.clipPath}`}
              alternative={alternative}
              videoUrl={
                alternative.previewUrl
                  ? alternative.previewUrl
                  : probeRunId && alternative.clipPath
                    ? getMatchedAlternativeClipUrl(probeRunId, alternative.clipPath)
                    : ""
              }
              onUse={() => onApplyAlternative(alternative)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export interface AlternativesSheetProps extends AlternativesPanelProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function AlternativesSheet({ open, onOpenChange, ...panelProps }: AlternativesSheetProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="left-auto right-0 top-0 h-dvh max-h-dvh w-full max-w-[420px] translate-x-0 translate-y-0 rounded-none border-y-0 border-r-0 p-0 sm:rounded-none data-[state=closed]:slide-out-to-right data-[state=open]:slide-in-from-right">
        <DialogHeader className="border-b px-4 py-4 pr-12">
          <DialogTitle>Alternatives</DialogTitle>
          <DialogDescription>Review candidate swaps for the selected trajectory.</DialogDescription>
        </DialogHeader>
        <div className="min-h-0 flex-1 overflow-y-auto p-4">
          <AlternativesPanel {...panelProps} />
        </div>
      </DialogContent>
    </Dialog>
  );
}

function AlternativeTrackletItem({
  alternative,
  videoUrl,
  onUse,
}: {
  alternative: MatchedAlternative;
  videoUrl: string;
  onUse: () => void;
}) {
  return (
    <div className="rounded-md border border-border/60 bg-muted/20 p-2">
      <div className="mb-1 flex items-center justify-between gap-2">
        <span className="text-[10px] font-semibold text-blue-400">ALT #{alternative.rank}</span>
        <span className="text-[10px] tabular-nums text-muted-foreground">
          score {(alternative.score * 100).toFixed(1)}%
        </span>
      </div>

      {videoUrl ? (
        <video
          src={videoUrl}
          className="mb-2 h-20 w-full rounded object-cover"
          controls
          muted
          playsInline
          preload="metadata"
        />
      ) : null}

      <div className="space-y-0.5 text-[10px] text-muted-foreground">
        <p className="font-mono text-foreground/90">
          {alternative.cameraId} - track {alternative.trackId}
        </p>
        <p>
          global {alternative.globalId ?? "?"} - {Math.max(1, alternative.numCameras)} cams
        </p>
      </div>

      <Button variant="outline" size="sm" className="mt-2 h-6 w-full text-[10px]" onClick={onUse}>
        Use In Timeline
      </Button>
    </div>
  );
}
