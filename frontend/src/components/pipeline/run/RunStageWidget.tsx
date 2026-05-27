import type { ReactNode } from "react";

import { Button } from "@/components/ui/button";
import { KaggleStatusPanel } from "@/components/stages/kaggle-status-panel";
import { cn } from "@/lib/utils";
import type { StageExecutionTarget } from "@/types";

import { StageProgressCard } from "../feedback/StageProgressCard";
import type { StageStatus } from "../status/types";

export interface RunStageWidgetProps {
  target?: StageExecutionTarget;
  runId?: string | null;
  status?: StageStatus;
  progress?: number;
  message?: string;
  isRunning?: boolean;
  disabled?: boolean;
  runLabel?: string;
  onRun?: () => void;
  children?: ReactNode;
  className?: string;
}

export function RunStageWidget({
  target = "local",
  runId = null,
  status = "idle",
  progress = 0,
  message,
  isRunning = false,
  disabled = false,
  runLabel = "Run stage",
  onRun,
  children,
  className,
}: RunStageWidgetProps) {
  const showKaggle = target === "kaggle" && Boolean(runId) && (isRunning || status === "running");

  return (
    <div className={cn("space-y-3", className)}>
      <div className="flex flex-wrap items-center gap-2">
        <Button type="button" onClick={onRun} disabled={disabled || isRunning}>
          {isRunning ? "Running..." : runLabel}
        </Button>
        {children}
      </div>
      {showKaggle && runId ? (
        <KaggleStatusPanel runId={runId} />
      ) : (
        <StageProgressCard status={status} progress={progress} message={message} />
      )}
    </div>
  );
}
