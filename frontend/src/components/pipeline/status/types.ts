import type { StageProgress } from "@/types";

export const STAGE_STATUSES = ["idle", "blocked", "running", "done", "stale", "error", "cancelled"] as const;

export type StageStatus = (typeof STAGE_STATUSES)[number];

export interface StageStatusMeta {
  label: string;
  sentence: string;
  textClassName: string;
  bgClassName: string;
  borderClassName: string;
  dotClassName: string;
}

export interface ToStageStatusOptions {
  blocked?: boolean;
  stale?: boolean;
}

export function toStageStatus(
  stage?: Pick<StageProgress, "status" | "progress" | "error" | "completedAt" | "staleSince"> | null,
  options: ToStageStatusOptions = {}
): StageStatus {
  if (!stage) return "idle";
  if (stage.status === "cancelled") return "cancelled";
  if (stage.status === "error" || stage.error) return "error";
  if (stage.status === "running" || (stage.progress > 0 && stage.progress < 100)) return "running";
  if (options.blocked) return "blocked";
  if (options.stale || (stage.staleSince !== null && stage.completedAt !== null && stage.staleSince > stage.completedAt)) return "stale";
  if (stage.status === "completed" || stage.progress >= 100) return "done";
  return "idle";
}
