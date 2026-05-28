import type { LucideIcon } from "lucide-react";
import { AlertTriangle, Ban, CheckCircle2, Circle, Loader2, OctagonX, XCircle } from "lucide-react";

import type { StageStatus, StageStatusMeta } from "./types";

export interface StageStatusMetaWithIcon extends StageStatusMeta {
  icon: LucideIcon;
}

export const STAGE_STATUS_META: Record<StageStatus, StageStatusMetaWithIcon> = {
  idle: {
    label: "Idle",
    sentence: "Idle and ready when upstream stages are complete.",
    textClassName: "text-status-idle",
    bgClassName: "bg-status-idle/10",
    borderClassName: "border-status-idle/30",
    dotClassName: "border-status-idle/50 text-status-idle",
    icon: Circle,
  },
  blocked: {
    label: "Blocked",
    sentence: "Blocked until an upstream stage finishes successfully.",
    textClassName: "text-status-blocked",
    bgClassName: "bg-status-blocked/10",
    borderClassName: "border-status-blocked/40",
    dotClassName: "border-status-blocked/60 text-status-blocked",
    icon: Ban,
  },
  running: {
    label: "Running",
    sentence: "Running now.",
    textClassName: "text-status-running",
    bgClassName: "bg-status-running/10",
    borderClassName: "border-status-running/40",
    dotClassName: "border-status-running bg-status-running/15 text-status-running",
    icon: Loader2,
  },
  done: {
    label: "Done",
    sentence: "Complete and ready for downstream stages.",
    textClassName: "text-status-done",
    bgClassName: "bg-status-done/10",
    borderClassName: "border-status-done/40",
    dotClassName: "border-status-done bg-status-done text-success-foreground",
    icon: CheckCircle2,
  },
  stale: {
    label: "Stale",
    sentence: "Stale because upstream inputs changed.",
    textClassName: "text-status-stale",
    bgClassName: "bg-status-stale/10",
    borderClassName: "border-status-stale/40",
    dotClassName: "border-status-stale bg-status-stale/15 text-status-stale",
    icon: AlertTriangle,
  },
  error: {
    label: "Error",
    sentence: "Last run failed and needs attention.",
    textClassName: "text-status-error",
    bgClassName: "bg-status-error/10",
    borderClassName: "border-status-error/40",
    dotClassName: "border-status-error bg-status-error/10 text-status-error",
    icon: XCircle,
  },
  cancelled: {
    label: "Cancelled",
    sentence: "Last run was cancelled.",
    textClassName: "text-status-cancelled",
    bgClassName: "bg-status-cancelled/10",
    borderClassName: "border-status-cancelled/40",
    dotClassName: "border-status-cancelled bg-status-cancelled/10 text-status-cancelled",
    icon: OctagonX,
  },
};

export function statusMeta(status: StageStatus): StageStatusMetaWithIcon {
  return STAGE_STATUS_META[status];
}
