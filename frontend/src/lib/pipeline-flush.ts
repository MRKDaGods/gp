import type { StageNumber } from "@/types";
import {
  PIPELINE_STAGE_DEFAULTS,
  useManualStageStore,
  usePipelineStore,
  useSessionStore,
  useTimelineStore,
} from "@/store";

/** Manual UI stages (no pipeline run) whose per-run completion markers must be cleared
 *  when an upstream stage at-or-before them is invalidated. */
const MANUAL_STAGES: StageNumber[] = [2, 5];

/**
 * Reset pipeline progress for this stage and all later stages, clear timeline/refinement/output session data,
 * and bump `downstreamInvalidateGeneration` so Stage 4 reloads even if selection/run ids match prior values.
 *
 * @param firstPipelineStageToInvalidate - 0-6; this stage and all higher-numbered stages return to idle defaults.
 */
export function flushPipelineFromStage(firstPipelineStageToInvalidate: StageNumber): void {
  usePipelineStore.setState((state) => ({
    downstreamInvalidateGeneration: state.downstreamInvalidateGeneration + 1,
    stages: state.stages.map((s) => {
      if (s.stage < firstPipelineStageToInvalidate) return s;
      const base = PIPELINE_STAGE_DEFAULTS.find((d) => d.stage === s.stage);
      return base ? { ...base } : s;
    }),
  }));

  useTimelineStore.getState().resetAfterUpstreamEdit();
  useSessionStore.getState().clearRefinementFrames();
  useSessionStore.getState().clearConfirmedClips();

  // Drop per-run completion markers for any manual stage at or after the invalidated point,
  // so loading the run later doesn't restore a "done" checkmark for work that's now stale.
  const runId = usePipelineStore.getState().runId;
  if (runId) {
    MANUAL_STAGES.forEach((manualStage) => {
      if (manualStage >= firstPipelineStageToInvalidate) {
        useManualStageStore.getState().clearManualStage(runId, manualStage);
      }
    });
  }
}
