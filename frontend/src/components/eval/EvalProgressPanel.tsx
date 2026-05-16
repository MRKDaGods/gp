"use client";

import { useEffect, useMemo, useState } from "react";
import { CheckCircle2, Loader2, XCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { getEvalResult, getEvalStatus, type EvalJobResultPayload, type EvalJobStatusPayload } from "@/lib/api";
import { cn } from "@/lib/utils";

interface EvalProgressPanelProps {
  jobId: string | null;
}

function statusVariant(status: string) {
  if (status === "completed") return "success" as const;
  if (status === "failed") return "destructive" as const;
  return "secondary" as const;
}

function flattenSummary(result: EvalJobResultPayload | null): Array<[string, string]> {
  const summary = result?.result?.summary;
  if (!summary || typeof summary !== "object") return [];
  return Object.entries(summary).map(([key, value]) => [key, typeof value === "number" ? value.toFixed(5).replace(/0+$/, "").replace(/\.$/, "") : String(value)]);
}

export function EvalProgressPanel({ jobId }: EvalProgressPanelProps) {
  const [status, setStatus] = useState<EvalJobStatusPayload | null>(null);
  const [result, setResult] = useState<EvalJobResultPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setStatus(null);
    setResult(null);
    setError(null);
    if (!jobId) return;

    const currentJobId = jobId;
    let cancelled = false;
    async function poll() {
      try {
        const nextStatus = await getEvalStatus(currentJobId);
        if (cancelled) return;
        setStatus(nextStatus);
        if (nextStatus.status === "completed") {
          const nextResult = await getEvalResult(currentJobId);
          if (!cancelled) setResult(nextResult);
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "Eval polling failed");
      }
    }

    void poll();
    const interval = window.setInterval(() => void poll(), 5000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [jobId]);

  const percent = useMemo(() => Number(status?.progress?.percent ?? (status?.status === "completed" ? 100 : 0)), [status]);
  const summaryRows = flattenSummary(result);

  if (!jobId) {
    return <div className="rounded-md border border-dashed p-4 text-sm text-muted-foreground">No eval job submitted.</div>;
  }

  return (
    <Card>
      <CardContent className="space-y-4 p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="min-w-0">
            <div className="text-xs text-muted-foreground">Current job</div>
            <div className="truncate font-mono text-sm">{jobId}</div>
          </div>
          <Badge variant={statusVariant(status?.status ?? "queued")} className="gap-1">
            {status?.status === "completed" ? <CheckCircle2 className="h-3 w-3" /> : status?.status === "failed" ? <XCircle className="h-3 w-3" /> : <Loader2 className="h-3 w-3 animate-spin" />}
            {status?.status ?? "queued"}
          </Badge>
        </div>

        <div className="space-y-2">
          <Progress value={Math.min(Math.max(percent, 0), 100)} className="h-2" />
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>{String(status?.progress?.stage ?? "queued")}</span>
            <span>{Math.round(percent)}%</span>
          </div>
        </div>

        {error && <div className="rounded-md border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">{error}</div>}
        {status?.error && <div className="rounded-md border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">{status.error}</div>}

        {summaryRows.length > 0 && (
          <div className="overflow-hidden rounded-md border">
            <table className="w-full text-sm">
              <tbody>
                {summaryRows.map(([key, value]) => (
                  <tr key={key} className="border-b last:border-b-0">
                    <th className="w-1/2 bg-muted/40 px-3 py-2 text-left font-medium">{key}</th>
                    <td className="px-3 py-2 font-mono">{value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {result?.result?.result !== undefined && (
          <pre className={cn("max-h-[420px] overflow-auto rounded-md border bg-muted/30 p-3 text-xs", "whitespace-pre-wrap break-words")}>{JSON.stringify(result.result.result, null, 2)}</pre>
        )}
      </CardContent>
    </Card>
  );
}