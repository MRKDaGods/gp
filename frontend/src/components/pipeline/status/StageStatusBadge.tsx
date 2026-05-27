import { cn } from "@/lib/utils";

import { statusMeta } from "./status-meta";
import type { StageStatus } from "./types";

export interface StageStatusBadgeProps {
  status?: StageStatus;
  label?: string;
  size?: "sm" | "md";
  className?: string;
}

export function StageStatusBadge({ status = "idle", label, size = "sm", className }: StageStatusBadgeProps) {
  const meta = statusMeta(status);
  const Icon = meta.icon;

  return (
    <span
      role={status === "running" ? "status" : undefined}
      aria-label={`Stage status: ${label ?? meta.label}`}
      className={cn(
        "inline-flex shrink-0 items-center gap-1.5 rounded-full border font-medium",
        size === "sm" ? "px-2 py-0.5 text-xs" : "px-2.5 py-1 text-sm",
        meta.bgClassName,
        meta.borderClassName,
        meta.textClassName,
        className
      )}
    >
      <Icon className={cn(size === "sm" ? "h-3.5 w-3.5" : "h-4 w-4", status === "running" && "animate-spin")} />
      <span>{label ?? meta.label}</span>
    </span>
  );
}
