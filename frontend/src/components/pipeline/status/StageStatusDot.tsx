import { Cloud } from "lucide-react";

import { cn } from "@/lib/utils";

import { statusMeta } from "./status-meta";
import type { StageStatus } from "./types";

export interface StageStatusDotProps {
  status?: StageStatus;
  size?: "sm" | "md";
  withCloudOverlay?: boolean;
  className?: string;
}

export function StageStatusDot({ status = "idle", size = "md", withCloudOverlay = false, className }: StageStatusDotProps) {
  const meta = statusMeta(status);
  const Icon = meta.icon;
  const dotSize = size === "sm" ? "h-5 w-5" : "h-6 w-6";
  const iconSize = size === "sm" ? "h-3 w-3" : "h-3.5 w-3.5";

  return (
    <span className="relative inline-flex shrink-0" aria-label={`Stage status: ${meta.label}`} title={meta.label}>
      <span
        className={cn(
          "inline-flex items-center justify-center rounded-full border text-[10px] font-semibold",
          dotSize,
          meta.dotClassName,
          status === "blocked" && "border-dashed",
          status === "stale" && "rotate-45 rounded-sm",
          className
        )}
      >
        <Icon className={cn(iconSize, status === "running" && "motion-safe:animate-spin", status === "stale" && "-rotate-45")} />
      </span>
      {withCloudOverlay ? (
        <Cloud className="absolute -right-1 -top-1 h-3 w-3 rounded-full bg-card text-sky-500 ring-1 ring-card" />
      ) : null}
    </span>
  );
}
