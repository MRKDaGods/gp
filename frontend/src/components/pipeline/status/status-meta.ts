import type { LucideIcon } from "lucide-react";
import { AlertTriangle, Ban, CheckCircle2, Circle, Loader2, XCircle } from "lucide-react";

import type { StageStatus, StageStatusMeta } from "./types";

export interface StageStatusMetaWithIcon extends StageStatusMeta {
  icon: LucideIcon;
}

export const STAGE_STATUS_META: Record<StageStatus, StageStatusMetaWithIcon> = {
  idle: {
    label: "Idle",
    sentence: "Idle and ready when upstream stages are complete.",
    textClassName: "text-muted-foreground",
    bgClassName: "bg-muted/30",
    borderClassName: "border-muted-foreground/20",
    dotClassName: "border-muted-foreground/60 text-muted-foreground",
    icon: Circle,
  },
  blocked: {
    label: "Blocked",
    sentence: "Blocked until an upstream stage finishes successfully.",
    textClassName: "text-zinc-500",
    bgClassName: "bg-zinc-500/10",
    borderClassName: "border-zinc-500/20",
    dotClassName: "border-zinc-500/60 text-zinc-500",
    icon: Ban,
  },
  running: {
    label: "Running",
    sentence: "Running now.",
    textClassName: "text-sky-600",
    bgClassName: "bg-sky-500/10",
    borderClassName: "border-sky-500/20",
    dotClassName: "border-sky-500 bg-sky-500/15 text-sky-600",
    icon: Loader2,
  },
  done: {
    label: "Done",
    sentence: "Complete and ready for downstream stages.",
    textClassName: "text-emerald-600",
    bgClassName: "bg-emerald-500/10",
    borderClassName: "border-emerald-500/20",
    dotClassName: "border-emerald-500 bg-emerald-500 text-white",
    icon: CheckCircle2,
  },
  stale: {
    label: "Stale",
    sentence: "Stale because upstream inputs changed.",
    textClassName: "text-amber-600",
    bgClassName: "bg-amber-500/10",
    borderClassName: "border-amber-500/20",
    dotClassName: "border-amber-500 bg-amber-500/15 text-amber-600",
    icon: AlertTriangle,
  },
  error: {
    label: "Error",
    sentence: "Last run failed and needs attention.",
    textClassName: "text-rose-600",
    bgClassName: "bg-rose-500/10",
    borderClassName: "border-rose-500/20",
    dotClassName: "border-rose-500 bg-rose-500/10 text-rose-600",
    icon: XCircle,
  },
};

export function statusMeta(status: StageStatus): StageStatusMetaWithIcon {
  return STAGE_STATUS_META[status];
}
