"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Database, FolderOpen, Loader2, Plus, RefreshCw, Trash2, Video } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useToast } from "@/hooks/use-toast";
import { useLoadRun } from "@/hooks/use-load-run";
import { deleteRun, getRuns, type RunStageState, type RunSummary } from "@/lib/api";
import { cn, formatBytes } from "@/lib/utils";
import { useDetectionStore, useManualStageStore, usePipelineStore, useSessionStore, useTimelineStore, useVideoStore } from "@/store";

/** Drop run-scoped selection/timeline state so a cleared workspace doesn't inherit a stale pick. */
function clearRunScopedSelection() {
  useDetectionStore.getState().deselectAll();
  useTimelineStore.getState().resetAfterUpstreamEdit();
}

const STAGE_LABELS = ["Ingest", "Detect", "Features", "Index", "Assoc", "Eval", "Viz"];

/** Run-list filter: separate quick tests (smoke runs) from full runs. */
type RunFilter = "all" | "tests" | "full";
const FILTER_LABELS: Record<RunFilter, string> = {
  all: "All",
  tests: "Quick tests",
  full: "Full runs",
};
const FILTER_ORDER: RunFilter[] = ["all", "tests", "full"];

function matchesFilter(run: RunSummary, filter: RunFilter): boolean {
  if (filter === "tests") return Boolean(run.smoke);
  if (filter === "full") return !run.smoke;
  return true;
}

function relativeTime(iso?: string | null): string {
  if (!iso) return "unknown";
  const t = new Date(iso).getTime();
  if (!Number.isFinite(t)) return "unknown";
  const s = Math.max(0, Math.floor((Date.now() - t) / 1000));
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

const STAGE_DOT_STYLES: Record<RunStageState, string> = {
  done: "bg-success/15 text-success",
  running: "bg-amber-500/20 text-amber-500 ring-1 ring-amber-500/40 animate-pulse",
  error: "bg-destructive/15 text-destructive ring-1 ring-destructive/40",
  idle: "bg-muted text-muted-foreground/50",
};

const STAGE_DOT_TITLES: Record<RunStageState, string> = {
  done: "done",
  running: "running...",
  error: "failed",
  idle: "not run",
};

function StageDots({ run }: { run: RunSummary }) {
  return (
    <div className="flex flex-wrap gap-1">
      {STAGE_LABELS.map((label, i) => {
        const key = `stage${i}` as keyof RunSummary["stages"];
        // Prefer live per-stage status; fall back to disk-presence booleans.
        const state: RunStageState =
          run.stageStatus?.[key] ?? (run.stages[key] ? "done" : "idle");
        return (
          <span
            key={label}
            className={cn("rounded px-1.5 py-px text-[10px] font-medium", STAGE_DOT_STYLES[state])}
            title={`${label}: ${STAGE_DOT_TITLES[state]}`}
          >
            {label}
          </span>
        );
      })}
    </div>
  );
}

export function RunManagerDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (v: boolean) => void }) {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [filter, setFilter] = useState<RunFilter>("all");
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const activeRunId = usePipelineStore((s) => s.runId);
  const resetPipeline = usePipelineStore((s) => s.reset);
  const setSessionStage = useSessionStore((s) => s.setCurrentStage);
  const setVideos = useVideoStore((s) => s.setVideos);
  const setCurrentVideo = useVideoStore((s) => s.setCurrentVideo);
  const loadRun = useLoadRun();
  const { toast } = useToast();

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const res = await getRuns();
      setRuns(res.data ?? []);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      toast({ title: "Couldn't list runs", description: msg, variant: "destructive" });
    } finally {
      setLoading(false);
    }
  }, [toast]);

  useEffect(() => {
    if (open) void refresh();
  }, [open, refresh]);

  // Reset filter/selection each time the dialog opens for a clean slate.
  useEffect(() => {
    if (open) {
      setFilter("all");
      setSelectedIds(new Set());
    }
  }, [open]);

  // Drop selections for runs that no longer exist (after a refresh/delete).
  useEffect(() => {
    setSelectedIds((prev) => {
      if (prev.size === 0) return prev;
      const live = new Set(runs.map((r) => r.runId));
      const next = new Set([...prev].filter((id) => live.has(id)));
      return next.size === prev.size ? prev : next;
    });
  }, [runs]);

  const counts = useMemo(
    () => ({
      all: runs.length,
      tests: runs.filter((r) => r.smoke).length,
      full: runs.filter((r) => !r.smoke).length,
    }),
    [runs]
  );

  const filteredRuns = useMemo(
    () => runs.filter((r) => matchesFilter(r, filter)),
    [runs, filter]
  );

  // Selection is scoped to what's currently visible under the active filter.
  const selectedVisible = useMemo(
    () => filteredRuns.filter((r) => selectedIds.has(r.runId)),
    [filteredRuns, selectedIds]
  );
  const allVisibleSelected =
    filteredRuns.length > 0 && selectedVisible.length === filteredRuns.length;

  const toggleSelect = useCallback((runId: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(runId)) next.delete(runId);
      else next.add(runId);
      return next;
    });
  }, []);

  const toggleSelectAllVisible = useCallback(() => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      const ids = filteredRuns.map((r) => r.runId);
      const everyOn = ids.length > 0 && ids.every((id) => next.has(id));
      if (everyOn) ids.forEach((id) => next.delete(id));
      else ids.forEach((id) => next.add(id));
      return next;
    });
  }, [filteredRuns]);

  // While the dialog is open and a run is in progress, poll so its stage status
  // and progress update live (e.g. detection moving from "running" to "done").
  useEffect(() => {
    if (!open) return;
    const anyRunning = runs.some((r) => r.status === "running" || r.status === "queued");
    if (!anyRunning) return;
    const id = setInterval(() => void refresh(), 2500);
    return () => clearInterval(id);
  }, [open, runs, refresh]);

  const handleLoad = async (runId: string) => {
    setBusyId(runId);
    try {
      const ok = await loadRun(runId);
      if (ok) onOpenChange(false);
    } finally {
      setBusyId(null);
    }
  };

  const handleDelete = async (run: RunSummary) => {
    const label = run.name ? `${run.name} (run ${run.runId})` : `run ${run.runId}`;
    if (!window.confirm(`Delete ${label}? This permanently removes its files from disk and cannot be undone.`)) {
      return;
    }
    setBusyId(run.runId);
    try {
      await deleteRun(run.runId);
      useManualStageStore.getState().clearRun(run.runId);
      if (run.runId === activeRunId) {
        // The active run was deleted - clear the workspace.
        resetPipeline();
        clearRunScopedSelection();
        setVideos([]);
        setCurrentVideo(null);
        setSessionStage(0);
      }
      setRuns((prev) => prev.filter((r) => r.runId !== run.runId));
      toast({ title: "Run deleted", description: `${label} removed from disk.`, variant: "success" });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      toast({ title: "Delete failed", description: msg, variant: "destructive" });
    } finally {
      setBusyId(null);
    }
  };

  const clearWorkspaceIfActiveDeleted = useCallback(
    (deletedIds: string[]) => {
      if (activeRunId && deletedIds.includes(activeRunId)) {
        resetPipeline();
        clearRunScopedSelection();
        setVideos([]);
        setCurrentVideo(null);
        setSessionStage(0);
      }
    },
    [activeRunId, resetPipeline, setVideos, setCurrentVideo, setSessionStage]
  );

  const handleDeleteSelected = async () => {
    const ids = selectedVisible.map((r) => r.runId);
    if (ids.length === 0) return;
    if (
      !window.confirm(
        `Delete ${ids.length} run${ids.length === 1 ? "" : "s"}? This permanently removes their files from disk and cannot be undone.`
      )
    ) {
      return;
    }
    setBusyId("__bulk__");
    try {
      const results = await Promise.allSettled(ids.map((id) => deleteRun(id)));
      const okIds = ids.filter((_, i) => results[i].status === "fulfilled");
      const failed = ids.length - okIds.length;
      okIds.forEach((id) => useManualStageStore.getState().clearRun(id));
      clearWorkspaceIfActiveDeleted(okIds);
      const okSet = new Set(okIds);
      setRuns((prev) => prev.filter((r) => !okSet.has(r.runId)));
      setSelectedIds((prev) => {
        const next = new Set(prev);
        okIds.forEach((id) => next.delete(id));
        return next;
      });
      if (okIds.length > 0) {
        toast({
          title: `Deleted ${okIds.length} run${okIds.length === 1 ? "" : "s"}`,
          description: failed > 0 ? `${failed} could not be deleted.` : "Removed from disk.",
          variant: failed > 0 ? "destructive" : "success",
        });
      } else {
        toast({ title: "Delete failed", description: "No runs were deleted.", variant: "destructive" });
      }
    } finally {
      setBusyId(null);
    }
  };

  const handleNewRun = () => {
    resetPipeline();
    clearRunScopedSelection();
    setVideos([]);
    setCurrentVideo(null);
    setSessionStage(0);
    onOpenChange(false);
    toast({ title: "New run", description: "Pick a dataset and cameras in the Upload stage to start." });
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Database className="h-5 w-5" />
            Runs
          </DialogTitle>
          <DialogDescription>Open a previous run, start a new one, or delete runs to free disk space.</DialogDescription>
        </DialogHeader>

        <div className="flex items-center justify-between gap-2">
          <Button type="button" size="sm" onClick={handleNewRun} className="gap-1.5">
            <Plus className="h-4 w-4" />
            New run
          </Button>
          <Button type="button" variant="outline" size="sm" onClick={() => void refresh()} disabled={loading} className="gap-1.5">
            <RefreshCw className={cn("h-3.5 w-3.5", loading && "animate-spin")} />
            Refresh
          </Button>
        </div>

        {/* Filter + bulk actions */}
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="inline-flex items-center gap-0.5 rounded-md border border-border/60 bg-muted/30 p-0.5">
            {FILTER_ORDER.map((f) => (
              <button
                key={f}
                type="button"
                onClick={() => setFilter(f)}
                className={cn(
                  "rounded px-2.5 py-1 text-xs font-medium transition-colors",
                  filter === f
                    ? "bg-background text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                )}
              >
                {FILTER_LABELS[f]}
                <span className="ml-1 text-[10px] opacity-60">{counts[f]}</span>
              </button>
            ))}
          </div>
          <div className="flex items-center gap-2">
            {filteredRuns.length > 0 && (
              <button
                type="button"
                onClick={toggleSelectAllVisible}
                className="text-xs text-muted-foreground hover:text-foreground"
              >
                {allVisibleSelected ? "Clear selection" : "Select all"}
              </button>
            )}
            <Button
              type="button"
              variant="destructive"
              size="sm"
              className="h-7 gap-1.5 px-2.5 text-xs"
              disabled={selectedVisible.length === 0 || busyId === "__bulk__"}
              onClick={() => void handleDeleteSelected()}
            >
              {busyId === "__bulk__" ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Trash2 className="h-3.5 w-3.5" />
              )}
              Delete selected{selectedVisible.length > 0 ? ` (${selectedVisible.length})` : ""}
            </Button>
          </div>
        </div>

        <ScrollArea className="h-[55vh] pr-3">
          {loading && runs.length === 0 ? (
            <div className="flex h-40 items-center justify-center text-muted-foreground">
              <Loader2 className="h-6 w-6 animate-spin" />
            </div>
          ) : runs.length === 0 ? (
            <div className="flex h-40 flex-col items-center justify-center gap-2 text-center text-muted-foreground">
              <FolderOpen className="h-8 w-8 opacity-40" />
              <p className="text-sm">No runs yet. Start one from the Upload stage.</p>
            </div>
          ) : filteredRuns.length === 0 ? (
            <div className="flex h-40 flex-col items-center justify-center gap-2 text-center text-muted-foreground">
              <FolderOpen className="h-8 w-8 opacity-40" />
              <p className="text-sm">No {FILTER_LABELS[filter].toLowerCase()} to show.</p>
            </div>
          ) : (
            <div className="space-y-2">
              {filteredRuns.map((run) => {
                const isActive = run.runId === activeRunId;
                const busy = busyId === run.runId;
                const selected = selectedIds.has(run.runId);
                return (
                  <div
                    key={run.runId}
                    className={cn(
                      "rounded-lg border p-3 transition-colors",
                      selected
                        ? "border-destructive/40 bg-destructive/5"
                        : isActive
                          ? "border-accent-strong/50 bg-accent-strong/5"
                          : "border-border/60 bg-card"
                    )}
                  >
                    <div className="flex items-start gap-3">
                      <Checkbox
                        className="mt-0.5 shrink-0"
                        checked={selected}
                        onCheckedChange={() => toggleSelect(run.runId)}
                        aria-label={`Select run ${run.runId}`}
                      />
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          <span className="truncate font-medium">{run.name || `Run ${run.runId}`}</span>
                          <span className="font-mono text-xs text-muted-foreground">#{run.runId}</span>
                          {isActive && <Badge className="text-[10px]">active</Badge>}
                          {run.status && run.status !== "ready" && (
                            <Badge variant="secondary" className="text-[10px] capitalize">{run.status}</Badge>
                          )}
                        </div>
                        <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
                          <span className="inline-flex items-center gap-1"><Video className="h-3 w-3" />{run.cameras?.length ?? 0} cameras</span>
                          {run.smoke && <Badge variant="outline" className="text-[10px]">quick test</Badge>}
                          <span>{relativeTime(run.updatedAt ?? run.createdAt)}</span>
                          {typeof run.sizeBytes === "number" && run.sizeBytes > 0 && <span>{formatBytes(run.sizeBytes)}</span>}
                          {typeof run.trajectoryCount === "number" && <span>{run.trajectoryCount} trajectories</span>}
                        </div>
                        <div className="mt-2"><StageDots run={run} /></div>
                        {run.status === "running" && (run.currentStageName || run.message) && (
                          <p className="mt-1.5 flex items-center gap-1.5 text-xs text-amber-500">
                            <Loader2 className="h-3 w-3 animate-spin" />
                            <span className="truncate">
                              {run.currentStageName ? `${run.currentStageName}` : "Running"}
                              {typeof run.progress === "number" ? ` * ${Math.round(run.progress)}%` : ""}
                            </span>
                          </p>
                        )}
                        {run.status === "error" && run.error && (
                          <p className="mt-1.5 truncate text-xs text-destructive" title={run.error}>
                            {run.error}
                          </p>
                        )}
                      </div>
                      <div className="flex shrink-0 flex-col gap-1.5">
                        <Button type="button" size="sm" className="h-7 px-2 text-xs" disabled={busy || isActive} onClick={() => void handleLoad(run.runId)}>
                          {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : isActive ? "Open" : "Load"}
                        </Button>
                        <Button type="button" variant="outline" size="sm" className="h-7 gap-1 px-2 text-xs text-destructive hover:bg-destructive/10 hover:text-destructive" disabled={busy} onClick={() => void handleDelete(run)}>
                          <Trash2 className="h-3.5 w-3.5" />
                          Delete
                        </Button>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
}
