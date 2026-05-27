"use client";

import { useEffect, useState } from "react";
import { ApiError, getKaggleStatus, type KaggleJobStatus } from "@/lib/api";
import { usePipelineStore } from "@/store";
import type { StageNumber } from "@/types";

const POLL_INTERVAL_MS = 5_000;
const MAX_ERROR_BACKOFF_MS = 60_000;
const TERMINAL_STATUSES = new Set<KaggleJobStatus["status"]>([
  "complete",
  "error",
  "cancelled",
]);

export interface UseKaggleStatusResult {
  status: KaggleJobStatus | null;
  isPolling: boolean;
  error: string | null;
  isLoading: boolean;
}

export interface UseKaggleStatusOptions {
  stage?: StageNumber;
}

function getErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    const data = error.data as { detail?: unknown; message?: unknown } | undefined;
    return String(data?.detail ?? data?.message ?? error.message);
  }
  return error instanceof Error ? error.message : "Unable to fetch Kaggle status";
}

export function useKaggleStatus(runId: string | null, options: UseKaggleStatusOptions = {}): UseKaggleStatusResult {
  const [status, setStatus] = useState<KaggleJobStatus | null>(null);
  const [isPolling, setIsPolling] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    let errorDelay = POLL_INTERVAL_MS;

    const clearPendingTimeout = () => {
      if (timeoutId) {
        clearTimeout(timeoutId);
        timeoutId = null;
      }
    };

    if (!runId) {
      setStatus(null);
      setError(null);
      setIsLoading(false);
      setIsPolling(false);
      return clearPendingTimeout;
    }

    const scheduleNext = (delayMs: number) => {
      clearPendingTimeout();
      timeoutId = setTimeout(() => {
        void fetchStatus(false);
      }, delayMs);
    };

    const fetchStatus = async (initial: boolean) => {
      if (cancelled) return;
      if (initial) {
        setIsLoading(true);
      }
      setIsPolling(true);

      try {
        const response = await getKaggleStatus(runId);
        if (cancelled) return;

        const nextStatus = response.data ?? null;
        setStatus(nextStatus);
        setError(null);
        errorDelay = POLL_INTERVAL_MS;

        if (nextStatus && options.stage !== undefined && TERMINAL_STATUSES.has(nextStatus.status)) {
          const pipelineStore = usePipelineStore.getState();
          if (nextStatus.status === "complete") {
            pipelineStore.setStageStatus(options.stage, "completed", `Stage ${options.stage} complete on Kaggle`);
          } else if (nextStatus.status === "error") {
            pipelineStore.setStageStatus(options.stage, "error", nextStatus.error ?? `Stage ${options.stage} failed on Kaggle`);
          } else if (nextStatus.status === "cancelled") {
            pipelineStore.setStageCancelled(options.stage, `Stage ${options.stage} cancelled on Kaggle`);
          }
        }

        if (!nextStatus || TERMINAL_STATUSES.has(nextStatus.status)) {
          setIsPolling(false);
          clearPendingTimeout();
        } else {
          scheduleNext(POLL_INTERVAL_MS);
        }
      } catch (err) {
        if (cancelled) return;

        if (err instanceof ApiError && err.status === 404) {
          setStatus(null);
          setError(null);
          setIsPolling(false);
          clearPendingTimeout();
          return;
        }

        setError(getErrorMessage(err));
        setIsPolling(true);
        scheduleNext(errorDelay);
        errorDelay = Math.min(errorDelay * 2, MAX_ERROR_BACKOFF_MS);
      } finally {
        if (!cancelled && initial) {
          setIsLoading(false);
        }
      }
    };

    void fetchStatus(true);

    return () => {
      cancelled = true;
      clearPendingTimeout();
    };
  }, [runId, options.stage]);

  return { status, isPolling, error, isLoading };
}
