"use client";

import { useEffect, useState } from "react";
import { ApiError, getKaggleStatus, type KaggleJobStatus } from "@/lib/api";

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

function getErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    const data = error.data as { detail?: unknown; message?: unknown } | undefined;
    return String(data?.detail ?? data?.message ?? error.message);
  }
  return error instanceof Error ? error.message : "Unable to fetch Kaggle status";
}

export function useKaggleStatus(runId: string | null): UseKaggleStatusResult {
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
  }, [runId]);

  return { status, isPolling, error, isLoading };
}
