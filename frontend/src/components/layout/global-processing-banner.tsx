"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { Loader2, CheckCircle2, XCircle, ChevronDown, ChevronUp, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { usePipelineStore, useVideoStore } from "@/store";
import { getPipelineStatus, cancelPipeline } from "@/lib/api";

const STAGE_LABELS: Record<number, string> = {
  0: "Ingestion",
  1: "Detection & Tracking",
  2: "Feature Extraction",
  3: "Indexing",
  4: "Cross-Camera Association",
  5: "Evaluation",
  6: "Visualization",
};

export function GlobalProcessingBanner() {
  const { runId, isRunning, stages, updateStageProgress, setIsRunning, setError } =
    usePipelineStore();
  const { currentVideo } = useVideoStore();

  const [expanded, setExpanded] = useState(true);
  const [showComplete, setShowComplete] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [lastStatus, setLastStatus] = useState<any>(null);
  const startTimeRef = useRef<number | null>(null);
  const pollFailCountRef = useRef(0);
  const completeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Find the current active stage from the store
  const runningStage = stages.find((s) => s.status === "running");
  const errorStage = stages.find((s) => s.status === "error");

  // Video name: prefer backend status (authoritative), fall back to current selection
  const videoName = lastStatus?.videoName ?? currentVideo?.name ?? null;

  // Overall progress: use backend status if available, else derive from store
  const overallProgress = lastStatus?.progress ?? runningStage?.progress ?? 0;
  const overallMessage =
    lastStatus?.message ?? runningStage?.message ?? "Processing...";
  const currentStageName =
    lastStatus?.currentStageName ?? (runningStage ? STAGE_LABELS[runningStage.stage] : null);
  const completedStages = lastStatus?.completedStages ?? 0;
  const totalStages = lastStatus?.totalStages ?? 1;

  // Poll backend for status while running
  useEffect(() => {
    if (!isRunning || !runId) return;
    let cancelled = false;

    const poll = async () => {
      try {
        const resp = await getPipelineStatus(runId);
        if (cancelled) return;
        const data: any = resp.data;
        setLastStatus(data);

        if (data?.status === "completed") {
          setIsRunning(false);
          setShowComplete(true);
          setLastStatus(data);
        } else if (data?.status === "error") {
          setIsRunning(false);
          setError(data?.error ?? "Pipeline failed");
          setLastStatus(data);
        }
        pollFailCountRef.current = 0;
      } catch (err) {
        pollFailCountRef.current += 1;
        if (pollFailCountRef.current >= 5) {
          const msg = err instanceof Error ? err.message : String(err);
          setIsRunning(false);
          setError(`Backend unreachable after 5 attempts: ${msg}`);
        }
      }
    };

    const interval = setInterval(poll, 1500);
    void poll();

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [isRunning, runId, setIsRunning, setError]);

  // Elapsed timer
  useEffect(() => {
    if (isRunning) {
      if (startTimeRef.current === null) {
        startTimeRef.current = Date.now();
      }
      const tick = setInterval(() => {
        setElapsed(Math.floor((Date.now() - (startTimeRef.current ?? Date.now())) / 1000));
      }, 1000);
      return () => clearInterval(tick);
    } else {
      startTimeRef.current = null;
    }
  }, [isRunning]);

  // Auto-dismiss completion banner after 8s
  useEffect(() => {
    if (showComplete) {
      completeTimerRef.current = setTimeout(() => setShowComplete(false), 8000);
      return () => {
        if (completeTimerRef.current) clearTimeout(completeTimerRef.current);
      };
    }
  }, [showComplete]);

  // Reset last status when runId changes
  useEffect(() => {
    setLastStatus(null);
    setShowComplete(false);
  }, [runId]);

  const handleCancel = useCallback(async () => {
    if (runId) {
      try {
        await cancelPipeline(runId);
      } catch { /* ignore */ }
    }
    setIsRunning(false);
  }, [runId, setIsRunning]);

  const formatElapsed = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  // Nothing to show
  if (!isRunning && !showComplete && !errorStage) return null;

  // Completed state
  if (showComplete && !isRunning) {
    return (
      <div className="border-b border-green-500/30 bg-green-500/10 px-6 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <CheckCircle2 className="h-5 w-5 text-green-500" />
            <span className="font-medium text-green-400">
              Pipeline Complete
            </span>
            <span className="text-sm text-green-400/70">
              {lastStatus?.message ?? "All stages finished successfully"}
            </span>
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6 text-green-400/70 hover:text-green-400"
            onClick={() => setShowComplete(false)}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>
    );
  }

  // Error state (sticky until dismissed)
  if (errorStage && !isRunning) {
    return (
      <div className="border-b border-red-500/30 bg-red-500/10 px-6 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <XCircle className="h-5 w-5 text-red-500" />
            <span className="font-medium text-red-400">Pipeline Error</span>
            <span className="text-sm text-red-400/70 truncate max-w-[600px]">
              {errorStage.message}
            </span>
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6 text-red-400/70 hover:text-red-400"
            onClick={() => {
              updateStageProgress(errorStage.stage as any, { status: "idle", message: "" });
            }}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>
    );
  }

  // Running state
  return (
    <div className="border-b border-primary/30 bg-primary/5">
      {/* Collapsed / slim bar */}
      <div
        className="flex items-center justify-between px-6 py-2 cursor-pointer"
        onClick={() => setExpanded((e) => !e)}
      >
        <div className="flex items-center gap-3 min-w-0">
          <Loader2 className="h-4 w-4 animate-spin text-primary flex-shrink-0" />
          <span className="font-medium text-sm truncate">
            {currentStageName
              ? `Processing${videoName ? ` "${videoName}"` : ""} — ${currentStageName}`
              : overallMessage}
          </span>
          <Badge variant="secondary" className="text-xs flex-shrink-0">
            {Math.round(overallProgress)}%
          </Badge>
          {totalStages > 1 && (
            <span className="text-xs text-muted-foreground flex-shrink-0">
              Stage {completedStages + 1}/{totalStages}
            </span>
          )}
        </div>
        <div className="flex items-center gap-3 flex-shrink-0">
          <span className="text-xs text-muted-foreground font-mono">
            {formatElapsed(elapsed)}
          </span>
          <Button
            variant="ghost"
            size="sm"
            className="h-6 px-2 text-xs text-destructive hover:text-destructive"
            onClick={(e) => {
              e.stopPropagation();
              void handleCancel();
            }}
          >
            Cancel
          </Button>
          {expanded ? (
            <ChevronUp className="h-4 w-4 text-muted-foreground" />
          ) : (
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          )}
        </div>
      </div>

      {/* Progress bar (always visible) */}
      <div className="px-6 pb-1">
        <Progress value={overallProgress} className="h-1.5" />
      </div>

      {/* Expanded detail */}
      {expanded && (
        <div className="px-6 pb-3 pt-1">
          <div className="flex flex-wrap gap-2">
            {[0, 1, 2, 3, 4].map((stageNum) => {
              if (totalStages <= 1 && stageNum > 0) return null;
              const label = STAGE_LABELS[stageNum] ?? `Stage ${stageNum}`;
              const stageData = stages.find((s) => s.stage === stageNum);
              const isActive =
                lastStatus?.currentStageNum === stageNum ||
                stageData?.status === "running";
              const isDone =
                (completedStages > 0 &&
                  stageNum <
                    (lastStatus?.currentStageNum ?? stageNum + 1)) ||
                stageData?.status === "completed";

              return (
                <div
                  key={stageNum}
                  className={cn(
                    "flex items-center gap-1.5 rounded-full px-3 py-1 text-xs border transition-colors",
                    isDone && "border-green-500/40 bg-green-500/10 text-green-400",
                    isActive && !isDone && "border-primary/40 bg-primary/10 text-primary",
                    !isDone && !isActive && "border-muted text-muted-foreground opacity-50"
                  )}
                >
                  {isDone ? (
                    <CheckCircle2 className="h-3 w-3" />
                  ) : isActive ? (
                    <Loader2 className="h-3 w-3 animate-spin" />
                  ) : (
                    <div className="h-3 w-3 rounded-full border" />
                  )}
                  {label}
                </div>
              );
            })}
          </div>
          <p className="mt-2 text-xs text-muted-foreground">{overallMessage}</p>
        </div>
      )}
    </div>
  );
}
