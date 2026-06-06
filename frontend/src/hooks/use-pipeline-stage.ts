"use client";

import { useCallback } from "react";

import { useToast } from "@/hooks/use-toast";
import { ApiError, getPipelineStatus, runDatasetInput } from "@/lib/api";
import { usePipelineStore } from "@/store";
import type { StageNumber } from "@/types";

export type StageRunResult = "completed" | "cancelled" | "error" | "aborted";

export interface RunPipelineStageOptions {
  /** Pipeline stage to execute (0=ingest, 1=track, 2=features, 3=index, 4=assoc, 5=eval, 6=viz). */
  pipelineStage: number;
  /** UI stage whose status/progress this run drives (0..6). */
  uiStage: StageNumber;
  /** Human label for messages/toasts, e.g. "detection". */
  label: string;
  /** Raw backend status payload on each poll tick (for stage-specific telemetry). */
  onProgress?: (data: Record<string, unknown>) => void;
  /** Return true to stop polling early (e.g. component unmounted). */
  shouldStop?: () => boolean;
}

const POLL_INTERVAL_MS = 1200;
const ACTIVE_STATUSES = new Set(["running", "queued", "pending", "processing", "starting"]);

/**
 * Run ONE pipeline stage against the active run and poll it to completion.
 *
 * Each stage runs incrementally against the same run_id (stages read prior
 * stages' outputs from outputs/<run_id>/stageN), so nothing cascades and
 * nothing auto-starts - a stage only runs when its own page invokes this.
 *
 * Returns the terminal status so callers can chain gating (await one stage,
 * then enable the next). Polling is owned by the call, not a standing effect,
 * so simultaneously-mounted stage pages never fight over the shared run status.
 */
export function useRunPipelineStage() {
  const setRunId = usePipelineStore((s) => s.setRunId);
  const setActiveStage = usePipelineStore((s) => s.setActiveStage);
  const setRunTelemetry = usePipelineStore((s) => s.setRunTelemetry);
  const updateStageProgress = usePipelineStore((s) => s.updateStageProgress);
  const { toast } = useToast();

  return useCallback(
    async (opts: RunPipelineStageOptions): Promise<StageRunResult> => {
      const { pipelineStage, uiStage, label } = opts;
      const input = usePipelineStore.getState().runInput;
      if (!input) {
        toast({
          title: "No active run",
          description: "Start a run from the Upload stage (ingestion) first.",
          variant: "destructive",
        });
        return "error";
      }

      const existingRunId = usePipelineStore.getState().runId;
      updateStageProgress(uiStage, {
        status: "running",
        progress: 0,
        message: `Starting ${label}...`,
      });
      setActiveStage(pipelineStage);
      setRunTelemetry(null);

      let activeRunId: string | null = existingRunId;
      try {
        const res = await runDatasetInput({
          inputDir: input.inputDir,
          name: input.name,
          cameras: input.cameras,
          // Coerce to a definite boolean so the backend never sees `undefined`
          // (which it would treat as "full video").
          smoke: Boolean(input.smoke),
          runId: existingRunId,
          stages: String(pipelineStage),
        });
        activeRunId = (res.data?.runId ?? res.data?.id ?? existingRunId) as string | null;
        if (activeRunId) setRunId(activeRunId);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        updateStageProgress(uiStage, {
          status: "error",
          progress: 100,
          message: `Failed to start ${label}: ${msg}`,
        });
        setActiveStage(null);
        toast({ title: `Failed to start ${label}`, description: msg, variant: "destructive" });
        return "error";
      }

      if (!activeRunId) {
        setActiveStage(null);
        return "error";
      }

      // Poll loop - owned by this call; resolves on a terminal backend status.
      // eslint-disable-next-line no-constant-condition
      while (true) {
        await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
        if (opts.shouldStop?.()) {
          setActiveStage(null);
          return "aborted";
        }

        let data: Record<string, any> | undefined;
        try {
          data = (await getPipelineStatus(activeRunId)).data as Record<string, any>;
        } catch (err) {
          // 404 = backend forgot this run (restart) - stop rather than loop forever.
          if (err instanceof ApiError && err.status === 404) {
            updateStageProgress(uiStage, { status: "idle", progress: 0, message: "Run not found" });
            setActiveStage(null);
            setRunTelemetry(null);
            return "error";
          }
          // Transient error - keep polling.
          continue;
        }

        opts.onProgress?.(data ?? {});
        const status = String(data?.status ?? "");
        const progress = Number(data?.progress ?? 0);
        const message = String(data?.message ?? `Running ${label}...`);

        setRunTelemetry({
          stageLabel: data?.currentStageName ? String(data.currentStageName) : undefined,
          completedStages: data?.completedStages != null ? Number(data.completedStages) : undefined,
          totalStages: data?.totalStages != null ? Number(data.totalStages) : undefined,
          camera: data?.currentCamera ? String(data.currentCamera) : undefined,
          camerasProcessed: data?.camerasProcessed != null ? Number(data.camerasProcessed) : undefined,
          camerasTotal: data?.camerasTotal != null ? Number(data.camerasTotal) : undefined,
          frame: data?.currentFrame != null ? Number(data.currentFrame) : undefined,
          frameTotal: data?.totalFrames != null ? Number(data.totalFrames) : undefined,
          message: data?.message ? String(data.message) : undefined,
          logTail: data?.logTail ? String(data.logTail) : undefined,
        });

        if (status === "completed") {
          updateStageProgress(uiStage, { status: "completed", progress: 100, message });
          setActiveStage(null);
          setRunTelemetry(null);
          return "completed";
        }
        if (status === "cancelled") {
          updateStageProgress(uiStage, { status: "idle", progress: 0, message: "Run cancelled" });
          setActiveStage(null);
          setRunTelemetry(null);
          return "cancelled";
        }
        if (status === "error") {
          const errMsg = data?.error ? String(data.error) : message || "Stage failed";
          updateStageProgress(uiStage, { status: "error", progress: 100, message: errMsg });
          setActiveStage(null);
          // Keep the failing log/telemetry so the stage panel can show WHY it
          // failed (instead of silently reverting with no detail).
          setRunTelemetry({
            stageLabel: data?.currentStageName ? String(data.currentStageName) : undefined,
            message: errMsg,
            logTail: data?.logTail
              ? String(data.logTail)
              : data?.errorDetail
                ? String(data.errorDetail)
                : undefined,
          });
          return "error";
        }
        if (ACTIVE_STATUSES.has(status)) {
          updateStageProgress(uiStage, { status: "running", progress, message });
        }
        // Unknown non-terminal status: keep polling without faking progress.
      }
    },
    [setActiveStage, setRunId, setRunTelemetry, toast, updateStageProgress]
  );
}
