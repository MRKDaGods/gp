import type { StageProgress } from "@/types";

export const STAGE_STATUSES = ["idle", "blocked", "running", "done", "stale", "error"] as const;

export type StageStatus = (typeof STAGE_STATUSES)[number];

export interface StageStatusMeta {
  label: string;
  sentence: string;
  textClassName: string;
  bgClassName: string;
  borderClassName: string;
  dotClassName: string;
}

export function toStageStatus(stage?: Pick<StageProgress, "status" | "progress" | "error"> | null): StageStatus {
  if (!stage) return "idle";
  if (stage.status === "error" || stage.error) return "error";
  if (stage.status === "running" || (stage.progress > 0 && stage.progress < 100)) return "running";
  if (stage.status === "completed" || stage.progress >= 100) return "done";
  return "idle";
}
