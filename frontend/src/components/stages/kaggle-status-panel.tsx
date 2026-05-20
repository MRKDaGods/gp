"use client";

import { useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  Cloud,
  ExternalLink,
  Loader2,
  OctagonX,
  XCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { cancelKaggleKernel, type KaggleJobStatus } from "@/lib/api";
import { useKaggleStatus } from "@/hooks/use-kaggle-status";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";

export interface KaggleStatusPanelProps {
  runId: string;
  className?: string;
}

const STATUS_STYLES: Record<KaggleJobStatus["status"], string> = {
  queued: "border-slate-300 bg-slate-100 text-slate-700",
  running: "border-blue-300 bg-blue-100 text-blue-700",
  complete: "border-green-300 bg-green-100 text-green-700",
  error: "border-red-300 bg-red-100 text-red-700",
  cancelled: "border-amber-300 bg-amber-100 text-amber-700",
  unknown: "border-slate-300 bg-slate-100 text-slate-700",
};

function formatRelativeTime(value: string | null): string {
  if (!value) return "not polled yet";
  const timestamp = new Date(value).getTime();
  if (!Number.isFinite(timestamp)) return "unknown";

  const seconds = Math.max(0, Math.floor((Date.now() - timestamp) / 1000));
  if (seconds < 5) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

function truncatePath(path: string | null): string {
  if (!path) return "pending";
  const parts = path.replace(/\\/g, "/").split("/").filter(Boolean);
  return parts.slice(-2).join("/") || path;
}

function StatusBadge({ status }: { status: KaggleJobStatus["status"] }) {
  return (
    <Badge variant="outline" className={cn("capitalize", STATUS_STYLES[status])}>
      {status}
    </Badge>
  );
}

function RunningMessage({ status }: { status: KaggleJobStatus["status"] }) {
  if (status === "queued") {
    return (
      <div className="flex items-center gap-2 rounded-md border bg-muted/40 p-3 text-sm">
        <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
        Waiting for a free Kaggle GPU slot
      </div>
    );
  }

  if (status === "running") {
    return (
      <div className="flex items-center gap-2 rounded-md border border-blue-200 bg-blue-50 p-3 text-sm text-blue-800">
        <Loader2 className="h-4 w-4 animate-spin" />
        Kernel is executing on Kaggle
      </div>
    );
  }

  return null;
}

function TerminalMessage({ status }: { status: KaggleJobStatus }) {
  if (status.status === "complete") {
    const exitCodeOk = status.exit_code === 0 || status.exit_code === null;
    return (
      <div className="flex items-start gap-2 rounded-md border border-green-200 bg-green-50 p-3 text-sm text-green-800">
        <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0" />
        <div className="min-w-0 space-y-1">
          <div className="font-medium">Outputs downloaded</div>
          <div className="truncate font-mono text-xs">{truncatePath(status.outputs_downloaded_to)}</div>
          <div className={cn("text-xs", exitCodeOk ? "text-green-700" : "text-red-700")}>
            Exit code: {status.exit_code ?? "pending"}
          </div>
        </div>
      </div>
    );
  }

  if (status.status === "error") {
    return (
      <Alert variant="destructive">
        <XCircle className="h-4 w-4" />
        <AlertDescription className="space-y-2">
          <div>{status.error ?? "Kaggle reported an error for this kernel."}</div>
          <a
            href={status.kernel_url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-sm font-medium underline underline-offset-4"
          >
            View Kaggle logs
            <ExternalLink className="h-3.5 w-3.5" />
          </a>
        </AlertDescription>
      </Alert>
    );
  }

  if (status.status === "cancelled") {
    return (
      <div className="flex items-center gap-2 rounded-md border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
        <OctagonX className="h-4 w-4" />
        Kernel was cancelled
      </div>
    );
  }

  return null;
}

export function KaggleStatusPanel({ runId, className }: KaggleStatusPanelProps) {
  const { status, isPolling, error, isLoading } = useKaggleStatus(runId);
  const [cancelError, setCancelError] = useState<string | null>(null);
  const [isCancelling, setIsCancelling] = useState(false);

  const isActive = status?.status === "queued" || status?.status === "running";

  const handleCancel = async () => {
    setCancelError(null);
    setIsCancelling(true);
    try {
      await cancelKaggleKernel(runId);
    } catch (err) {
      setCancelError(err instanceof Error ? err.message : "Unable to cancel Kaggle kernel");
    } finally {
      setIsCancelling(false);
    }
  };

  if (isLoading) {
    return (
      <Card className={cn("border-blue-200 bg-blue-50/40", className)}>
        <CardContent className="flex items-center gap-3 p-6 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" />
          Loading Kaggle status...
        </CardContent>
      </Card>
    );
  }

  if (!status && !error) {
    return null;
  }

  return (
    <Card className={cn("overflow-hidden border-blue-200 bg-blue-50/30", className)}>
      <CardHeader className="flex flex-row items-center justify-between gap-3 space-y-0 pb-3">
        <CardTitle className="flex min-w-0 items-center gap-2 text-base">
          <Cloud className="h-4 w-4 shrink-0 text-blue-600" />
          <span className="truncate">Running on Kaggle</span>
        </CardTitle>
        {status ? <StatusBadge status={status.status} /> : null}
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        {error ? (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        ) : null}

        {status ? (
          <>
            <div className="grid gap-3 sm:grid-cols-[120px_1fr]">
              <div className="text-muted-foreground">Kernel</div>
              <div className="min-w-0">
                <a
                  href={status.kernel_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex max-w-full items-center gap-1 font-mono text-xs text-blue-700 underline-offset-4 hover:underline"
                >
                  <span className="truncate">{status.kernel_slug}</span>
                  <ExternalLink className="h-3.5 w-3.5 shrink-0" />
                </a>
              </div>
              <div className="text-muted-foreground">Last polled</div>
              <div>{formatRelativeTime(status.last_polled_at)}</div>
              <div className="text-muted-foreground">Stages</div>
              <div className="flex flex-wrap gap-1.5">
                {status.stages.map((stage) => (
                  <Badge key={stage} variant="secondary">
                    Stage {stage}
                  </Badge>
                ))}
              </div>
            </div>

            <RunningMessage status={status.status} />
            <TerminalMessage status={status} />
          </>
        ) : null}

        {cancelError ? (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{cancelError}</AlertDescription>
          </Alert>
        ) : null}
      </CardContent>
      {isActive ? (
        <CardFooter className="flex items-center justify-between gap-3 border-t bg-background/60 px-6 py-3">
          <div className="text-xs text-muted-foreground">
            {isPolling ? "Polling every 5 seconds" : "Polling paused"}
          </div>
          <Button
            type="button"
            size="sm"
            variant="destructive"
            onClick={() => void handleCancel()}
            disabled={isCancelling}
          >
            {isCancelling ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
            Cancel kernel
          </Button>
        </CardFooter>
      ) : null}
    </Card>
  );
}
