"use client";

import { Badge } from "@/components/ui/badge";
import type { StageProgress } from "@/types";

export interface TimelineDebugPanelProps {
  runId?: string | null;
  galleryRunId?: string | null;
  resolvedProbeRunId?: string | null;
  currentStage: number;
  triggerReload: number;
  downstreamInvalidateGeneration: number;
  tracksCount: number;
  confirmedCount: number;
  selectedTrackletCount: number;
  cameraLaneCount: number;
  alternativesCount: number;
  alternativesCameraCount: number;
  timelineDataSource: string;
  stage4Progress?: StageProgress;
}

export function TimelineDebugPanel({
  runId,
  galleryRunId,
  resolvedProbeRunId,
  currentStage,
  triggerReload,
  downstreamInvalidateGeneration,
  tracksCount,
  confirmedCount,
  selectedTrackletCount,
  cameraLaneCount,
  alternativesCount,
  alternativesCameraCount,
  timelineDataSource,
  stage4Progress,
}: TimelineDebugPanelProps) {
  const rows = [
    ["run_id", runId ?? "-"],
    ["gallery_run_id", galleryRunId ?? "-"],
    ["resolved_probe_run_id", resolvedProbeRunId ?? "-"],
    ["current_stage", String(currentStage)],
    ["generation", String(downstreamInvalidateGeneration)],
    ["reload_counter", String(triggerReload)],
    ["tracks", String(tracksCount)],
    ["confirmed", String(confirmedCount)],
    ["selected_tracklets", String(selectedTrackletCount)],
    ["camera_lanes", String(cameraLaneCount)],
    ["alternatives", String(alternativesCount)],
    ["alternative_cameras", String(alternativesCameraCount)],
    ["data_source", timelineDataSource],
    ["stage4_status", stage4Progress?.status ?? "-"],
    ["stage4_progress", stage4Progress ? `${stage4Progress.progress}%` : "-"],
    ["stage4_message", stage4Progress?.message ?? "-"],
  ];

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-2">
        <Badge variant="outline">{tracksCount} tracks</Badge>
        <Badge variant="outline">{cameraLaneCount} lanes</Badge>
        <Badge variant="outline">{confirmedCount} confirmed</Badge>
      </div>
      <dl className="grid gap-2 text-xs sm:grid-cols-2">
        {rows.map(([label, value]) => (
          <div key={label} className="min-w-0 rounded border bg-muted/20 p-2">
            <dt className="font-mono text-[10px] uppercase tracking-wide text-muted-foreground">{label}</dt>
            <dd className="mt-1 break-words font-mono text-foreground/90">{value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
