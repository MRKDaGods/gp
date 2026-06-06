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

/** The UI nav stages don't map 1:1 to pipeline prerequisites. "Selection" (UI */
const PREREQUISITE_OVERRIDE: Partial<Record<StageNumber, StageNumber>> = {
  3: 1,
  // Refinement (UI 5) is an OPTIONAL manual review with no pipeline run of its own, so it
  // must not gate Output. Output's real prerequisite is the cross-camera association (UI 4).
  6: 4,
};

/** The stage that must be done before `stage` can run (null for stage 0). */
export function prerequisiteStage(stage: StageNumber): StageNumber | null {
  if (stage <= 0) return null;
  return PREREQUISITE_OVERRIDE[stage] ?? ((stage - 1) as StageNumber);
}

export function deriveStageState(
  stage: StageNumber,
  stages: StageProgress[],
  executionTarget: StageExecutionTarget
): StageStateSnapshot {
  const progress = stages.find((candidate) => candidate.stage === stage) ?? null;
  const prereq = prerequisiteStage(stage);
  const previousStage = prereq != null ? stages.find((candidate) => candidate.stage === prereq) ?? null : null;
  const blocked = prereq != null && !stageIsDone(previousStage);
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