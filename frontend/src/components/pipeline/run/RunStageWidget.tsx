import type { ReactNode } from "react";

import { KaggleCredentialsModal } from "@/components/settings/kaggle-credentials-modal";
import { Button } from "@/components/ui/button";
import { KaggleStatusPanel } from "@/components/stages/kaggle-status-panel";
import { cn } from "@/lib/utils";
import { useKaggleCredentialsStore } from "@/lib/kaggle-credentials-store";
import { usePipelineStore, useStageExecutionStore } from "@/store";
import type { StageExecutionTarget, StageNumber } from "@/types";

import { StageProgressCard } from "../feedback/StageProgressCard";
import { toStageStatus, type StageStatus } from "../status/types";

export interface RunStageWidgetProps {
  stage?: StageNumber;
  title?: string;
  target?: StageExecutionTarget;
  runId?: string | null;
  status?: StageStatus;
  progress?: number;
  message?: string;
  eta?: string | null;
  isRunning?: boolean;
  disabled?: boolean;
  runLabel?: string;
  runIcon?: ReactNode;
  mode?: "full" | "button-only";
  onRun?: () => void;
  children?: ReactNode;
  className?: string;
}

export function RunStageWidget({
  stage,
  title,
  target = "local",
  runId = null,
  status,
  progress,
  message,
  eta,
  isRunning = false,
  disabled = false,
  runLabel = "Run stage",
  runIcon,
  mode = "full",
  onRun,
  children,
  className,
}: RunStageWidgetProps) {
  const stageProgress = usePipelineStore((state) => stage !== undefined ? state.stages.find((candidate) => candidate.stage === stage) : null);
  const stageTarget = useStageExecutionStore((state) => stage !== undefined ? state.getStageExecutionTarget(stage) : target);
  const modalOpen = useKaggleCredentialsStore((state) => state.modalOpen);
  const setModalOpen = useKaggleCredentialsStore((state) => state.setModalOpen);
  const resolvedStatus = status ?? toStageStatus(stageProgress);
  const resolvedProgress = progress ?? stageProgress?.progress ?? 0;
  const resolvedMessage = eta ? `${message ?? stageProgress?.message ?? "Waiting to run"} · ETA ${eta}` : message ?? stageProgress?.message;
  const resolvedTitle = title ?? (stage !== undefined ? `Stage ${stage}` : runLabel);
  const showKaggle = stageTarget === "kaggle" && Boolean(runId) && (isRunning || resolvedStatus === "running");
  const showControls = Boolean(onRun || children);

  if (mode === "button-only") {
    return (
      <div className={cn("flex flex-wrap items-center gap-2", className)}>
        {onRun ? (
          <Button type="button" onClick={onRun} disabled={disabled || isRunning}>
            {runIcon}
            {isRunning ? "Running..." : runLabel}
          </Button>
        ) : null}
        {children}
        <KaggleCredentialsModal open={modalOpen} onOpenChange={setModalOpen} />
      </div>
    );
  }

  return (
    <div className={cn("space-y-3", className)}>
      {showControls ? (
        <div className="flex flex-wrap items-center gap-2">
          {onRun ? (
            <Button type="button" onClick={onRun} disabled={disabled || isRunning}>
              {runIcon}
              {isRunning ? "Running..." : runLabel}
            </Button>
          ) : null}
          {children}
        </div>
      ) : null}
      {showKaggle && runId ? (
        <KaggleStatusPanel runId={runId} stage={stage} />
      ) : (
        <StageProgressCard title={resolvedTitle} status={resolvedStatus} progress={resolvedProgress} message={resolvedMessage} />
      )}
      <KaggleCredentialsModal open={modalOpen} onOpenChange={setModalOpen} />
    </div>
  );
}
