"use client";

import { useCallback } from "react";

import { useToast } from "@/hooks/use-toast";
import { ApiError, runStage } from "@/lib/api";
import { useKaggleCredentialsStore } from "@/lib/kaggle-credentials-store";
import { flushPipelineFromStage } from "@/lib/pipeline-flush";
import { usePipelineStore, useStageExecutionStore, useVideoStore } from "@/store";

export interface StartStage1Options {
  resetRunId?: boolean;
  afterStart?: () => void;
}

function getStage1ErrorMessage(error: unknown): string {
  if (error instanceof ApiError && error.status === 429) {
    return "Both Kaggle slots busy — try again later";
  }
  return error instanceof Error ? error.message : String(error);
}

export function useStartStage1() {
  const currentVideo = useVideoStore((state) => state.currentVideo);
  const setRunId = usePipelineStore((state) => state.setRunId);
  const setIsRunning = usePipelineStore((state) => state.setIsRunning);
  const updateStageProgress = usePipelineStore((state) => state.updateStageProgress);
  const getStageExecutionTarget = useStageExecutionStore((state) => state.getStageExecutionTarget);
  const { toast } = useToast();

  return useCallback(async (options: StartStage1Options = {}) => {
    if (!currentVideo) return;

    const executionTarget = getStageExecutionTarget(1);
    const credentials = useKaggleCredentialsStore.getState().credentials;
    const kaggle = executionTarget === "kaggle"
      ? { target: "kaggle" as const, username: credentials?.username, key: credentials?.key }
      : null;

    flushPipelineFromStage(1);
    if (options.resetRunId) setRunId(null);
    setIsRunning(true);
    updateStageProgress(1, { status: "running", progress: 0, message: `Queued Stage 1 for ${currentVideo.name}` });

    try {
      const response = await runStage(1, {
        videoId: currentVideo.id,
        config: { tracker: "deepocsort" },
        kaggle,
      });
      const nextRunId = response.data?.runId ?? (response.data as any)?.id ?? null;
      if (nextRunId) setRunId(nextRunId);
      options.afterStart?.();
    } catch (error) {
      if (kaggle?.target === "kaggle" && error instanceof ApiError && error.status === 401) {
        useKaggleCredentialsStore.getState().openCredentialsModal();
      }
      const message = getStage1ErrorMessage(error);
      setIsRunning(false);
      updateStageProgress(1, { status: "error", progress: 100, message: `Failed to start Stage 1: ${message}` });
      toast({ title: "Failed to start Stage 1", description: message, variant: "destructive" });
    }
  }, [currentVideo, getStageExecutionTarget, setIsRunning, setRunId, toast, updateStageProgress]);
}