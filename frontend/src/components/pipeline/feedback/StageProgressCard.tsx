import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

import { StageStatusBadge } from "../status/StageStatusBadge";
import type { StageStatus } from "../status/types";

export interface StageProgressCardProps {
  title?: string;
  progress?: number;
  status?: StageStatus;
  message?: string;
  className?: string;
}

export function StageProgressCard({
  title = "Stage progress",
  progress = 0,
  status = "idle",
  message,
  className,
}: StageProgressCardProps) {
  const boundedProgress = Math.max(0, Math.min(100, progress));

  return (
    <Card className={cn("overflow-hidden", className)}>
      <CardHeader className="flex flex-row items-center justify-between gap-3 space-y-0 pb-3">
        <CardTitle className="truncate text-base">{title}</CardTitle>
        <StageStatusBadge status={status} />
      </CardHeader>
      <CardContent className="space-y-2">
        <div className="flex items-center justify-between gap-3 text-sm">
          <span className="truncate text-muted-foreground">{message ?? "Waiting to run"}</span>
          <span className="font-mono text-xs text-muted-foreground">{Math.round(boundedProgress)}%</span>
        </div>
        <Progress value={boundedProgress} className="h-2" />
      </CardContent>
    </Card>
  );
}
