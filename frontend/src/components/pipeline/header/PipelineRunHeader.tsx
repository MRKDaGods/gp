"use client";

import { useMemo } from "react";
import { AlertTriangle, Clipboard, GitBranch } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
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

export function PipelineRunHeader({
  runId = null,
  currentStage = 0,
  stages = [],
  stageLabels = DEFAULT_STAGE_LABELS,
  error = null,
  lastRunLabel = "-",
  className,
}: PipelineRunHeaderProps) {
  const label = stageLabels.find((stage) => stage.id === currentStage)?.label ?? `Stage ${currentStage}`;
  const currentStageState = useStageState(currentStage);
  const progress = useMemo(() => overallProgress(stages), [stages]);

  const copyRunId = () => {
    if (runId && typeof navigator !== "undefined") {
      void navigator.clipboard?.writeText(runId);
    }
  };

  return (
    <div className={cn("flex h-10 shrink-0 items-center gap-3 border-b bg-card px-4 text-sm sm:px-6", className)}>
      <Button
        type="button"
        variant="outline"
        size="sm"
        className="h-7 max-w-[220px] gap-1.5 px-2 font-mono text-xs"
        onClick={copyRunId}
        disabled={!runId}
        title={runId ? "Copy runId" : "No active run"}
      >
        <Clipboard className="h-3.5 w-3.5" />
        <span className="truncate">runId: {runId ?? "-"}</span>
      </Button>
      <div className="flex min-w-0 items-center gap-1.5 text-muted-foreground">
        <GitBranch className="h-3.5 w-3.5 shrink-0" />
        <span className="truncate">Stage {currentStage + 1}/7 · {label}</span>
      </div>
      <div className="hidden min-w-[140px] max-w-[220px] flex-1 items-center gap-2 md:flex">
        <Progress value={progress} className="h-1.5" />
        <span className="w-9 text-right font-mono text-xs text-muted-foreground">{Math.round(progress)}%</span>
      </div>
      <span className="ml-auto hidden text-xs text-muted-foreground sm:inline">last run: {lastRunLabel}</span>
      {error ? (
        <Badge variant="outline" className="gap-1 border-rose-500/30 bg-rose-500/10 text-rose-600">
          <AlertTriangle className="h-3.5 w-3.5" />
          1 error
        </Badge>
      ) : (
        <StageStatusBadge status={currentStageState.status} />
      )}
    </div>
  );
}
