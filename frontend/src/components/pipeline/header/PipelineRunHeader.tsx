"use client";

import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, Clipboard, GitBranch } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Progress } from "@/components/ui/progress";
import { useStageState } from "@/hooks/useStageState";
import { cn } from "@/lib/utils";
import type { StageNumber, StageProgress } from "@/types";

import { StageStatusBadge } from "../status/StageStatusBadge";

export interface PipelineRunHeaderStage {
  id: StageNumber;
  label: string;
}

export interface PipelineRunHeaderProps {
  runId?: string | null;
  currentStage?: StageNumber;
  stages?: StageProgress[];
  stageLabels?: PipelineRunHeaderStage[];
  error?: string | null;
  lastRunLabel?: string;
  onSelectErrorStage?: (stage: StageNumber) => void;
  className?: string;
}

const DEFAULT_STAGE_LABELS: PipelineRunHeaderStage[] = [
  { id: 0, label: "Upload" },
  { id: 1, label: "Detection" },
  { id: 2, label: "Selection" },
  { id: 3, label: "Inference" },
  { id: 4, label: "Timeline" },
  { id: 5, label: "Refinement" },
  { id: 6, label: "Output" },
];

function overallProgress(stages: StageProgress[]): number {
  if (stages.length === 0) return 0;
  return stages.reduce((total, stage) => total + Math.max(0, Math.min(100, stage.progress)), 0) / stages.length;
}

function formatRelativeTimestamp(timestamp: number | null): string {
  if (!timestamp) return "-";
  const seconds = Math.max(0, Math.floor((Date.now() - timestamp) / 1000));
  if (seconds < 10) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

export function PipelineRunHeader({
  runId = null,
  currentStage = 0,
  stages = [],
  stageLabels = DEFAULT_STAGE_LABELS,
  error = null,
  lastRunLabel = "-",
  onSelectErrorStage,
  className,
}: PipelineRunHeaderProps) {
  const [, setClockTick] = useState(0);
  const label = stageLabels.find((stage) => stage.id === currentStage)?.label ?? `Stage ${currentStage}`;
  const currentStageState = useStageState(currentStage);
  const progress = useMemo(() => overallProgress(stages), [stages]);
  const errorStages = useMemo(
    () => stages
      .filter((stage) => stage.status === "error" || Boolean(stage.error))
      .map((stage) => ({
        stage: stage.stage,
        label: stageLabels.find((candidate) => candidate.id === stage.stage)?.label ?? `Stage ${stage.stage}`,
        message: stage.error ?? stage.message ?? error ?? "Stage failed",
      }))
      .slice(-5)
      .reverse(),
    [error, stageLabels, stages]
  );
  const latestRunAt = useMemo(
    () => stages.reduce<number | null>((latest, stage) => {
      const value = stage.lastRunAt ?? null;
      return value !== null && (latest === null || value > latest) ? value : latest;
    }, null),
    [stages]
  );
  const resolvedLastRunLabel = latestRunAt ? formatRelativeTimestamp(latestRunAt) : lastRunLabel;

  useEffect(() => {
    const interval = window.setInterval(() => setClockTick((tick) => tick + 1), 30000);
    return () => window.clearInterval(interval);
  }, []);

  const copyRunId = () => {
    if (runId && typeof navigator !== "undefined") {
      void navigator.clipboard?.writeText(runId);
    }
  };

  return (
    <div className={cn("flex h-10 shrink-0 items-center gap-3 border-b border-border/60 bg-background/80 px-4 text-sm backdrop-blur sm:px-6", className)}>
      <Button
        type="button"
        variant="outline"
        size="sm"
        className="h-7 max-w-[220px] gap-1.5 px-2 font-mono text-xs"
        onClick={copyRunId}
        disabled={!runId}
        title={runId ? "Copy runId" : "No active run"}
        aria-label={runId ? `Copy runId ${runId}` : "No active runId to copy"}
      >
        <Clipboard className="h-3.5 w-3.5" />
        <span className="truncate">runId: {runId ?? "-"}</span>
      </Button>
      <div className="flex min-w-0 items-center gap-1.5 text-muted-foreground">
        <GitBranch className="h-3.5 w-3.5 shrink-0" />
        <span className="truncate">Stage {currentStage + 1}/7 * {label}</span>
        <StageStatusBadge status={currentStageState.status} />
      </div>
      <div className="hidden min-w-[140px] max-w-[220px] flex-1 items-center gap-2 md:flex">
        <Progress value={progress} className="h-1.5" />
        <span className="w-9 text-right font-mono text-xs text-muted-foreground">{Math.round(progress)}%</span>
      </div>
      <span className="ml-auto hidden text-xs text-muted-foreground sm:inline">last run: {resolvedLastRunLabel}</span>
      {errorStages.length > 0 ? (
        <Popover>
          <PopoverTrigger asChild>
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-7 gap-1 border-status-error/40 bg-status-error/10 px-2 text-xs text-status-error hover:bg-status-error/15 hover:text-status-error"
            >
              <AlertTriangle className="h-3.5 w-3.5" />
              {errorStages.length} {errorStages.length === 1 ? "error" : "errors"}
            </Button>
          </PopoverTrigger>
          <PopoverContent align="end" className="w-80 space-y-3 p-3">
            <div className="text-sm font-medium">Recent errors</div>
            <div className="space-y-2">
              {errorStages.map((stageError) => (
                <button
                  key={`${stageError.stage}-${stageError.message}`}
                  type="button"
                  className="w-full rounded-md border border-border/60 bg-muted/30 px-3 py-2 text-left text-sm transition-colors hover:border-status-error/40 hover:bg-status-error/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                  onClick={() => onSelectErrorStage?.(stageError.stage)}
                >
                  <div className="font-medium">Stage {stageError.stage} * {stageError.label}</div>
                  <div className="mt-1 line-clamp-2 text-xs text-muted-foreground">{stageError.message}</div>
                </button>
              ))}
            </div>
          </PopoverContent>
        </Popover>
      ) : null}
    </div>
  );
}
