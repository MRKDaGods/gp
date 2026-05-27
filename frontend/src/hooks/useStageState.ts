import { usePipelineStore, useStageExecutionStore } from "@/store";
import type { StageExecutionTarget, StageNumber, StageProgress } from "@/types";
import { toStageStatus, type StageStatus } from "@/components/pipeline/status/types";

export interface StageStateSnapshot {
  status: StageStatus;
  progress: StageProgress | null;
  error: string | null;
  lastRunAt: number | null;
  completedAt: number | null;
  isStale: boolean;
  executionTarget: StageExecutionTarget;
}

function stageIsDone(stage?: StageProgress | null): boolean {
  return Boolean(stage && (stage.status === "completed" || stage.progress >= 100));
}

export function deriveStageState(
  stage: StageNumber,
  stages: StageProgress[],
  executionTarget: StageExecutionTarget
): StageStateSnapshot {
  const progress = stages.find((candidate) => candidate.stage === stage) ?? null;
  const previousStage = stage > 0 ? stages.find((candidate) => candidate.stage === stage - 1) ?? null : null;
  const blocked = stage > 0 && !stageIsDone(previousStage);
  const isStale = progress !== null
    && progress.staleSince !== null
    && progress.completedAt !== null
    && progress.staleSince > progress.completedAt;
  const status = toStageStatus(progress, { blocked, stale: isStale });

  return {
    status,
    progress,
    error: progress?.error ?? (status === "error" ? progress?.message ?? null : null),
    lastRunAt: progress?.lastRunAt ?? null,
    completedAt: progress?.completedAt ?? null,
    isStale,
    executionTarget,
  };
}

export function useStageState(stage: StageNumber): StageStateSnapshot {
  const stages = usePipelineStore((state) => state.stages);
  const executionTarget = useStageExecutionStore((state) => state.getStageExecutionTarget(stage));

  return deriveStageState(stage, stages, executionTarget);
}