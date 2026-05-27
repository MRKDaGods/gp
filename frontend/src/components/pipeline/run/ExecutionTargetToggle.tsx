"use client";

import { useState } from "react";
import { AlertTriangle, Cloud, KeyRound, Server } from "lucide-react";

import { KaggleCredentialsStatus } from "@/components/layout/kaggle-status-indicator";
import { KaggleCredentialsModal } from "@/components/settings/kaggle-credentials-modal";
import { Button } from "@/components/ui/button";
import { useKaggleCredentialsStore } from "@/lib/kaggle-credentials-store";
import { cn } from "@/lib/utils";
import { useStageExecutionStore } from "@/store";

export interface ExecutionTargetToggleProps {
  stage: number;
  variant?: "full" | "compact";
  className?: string;
}

export function ExecutionTargetToggle({ stage, variant = "full", className }: ExecutionTargetToggleProps) {
  const [modalOpen, setModalOpen] = useState(false);
  const target = useStageExecutionStore((state) => state.getStageExecutionTarget(stage));
  const setStageExecutionTarget = useStageExecutionStore((state) => state.setStageExecutionTarget);
  const credentials = useKaggleCredentialsStore((state) => state.credentials);
  const isKaggle = target === "kaggle";
  const Icon = isKaggle ? Cloud : Server;

  if (variant === "compact") {
    return (
      <div className={cn("flex flex-wrap items-center gap-2", className)}>
        <div className="inline-flex items-center rounded-md border bg-muted/20 p-0.5" role="group" aria-label={`Stage ${stage} execution target`}>
          <button
            type="button"
            className={cn(
              "inline-flex h-8 items-center gap-1.5 rounded px-2 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
              !isKaggle ? "bg-background text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
            )}
            onClick={() => setStageExecutionTarget(stage, "local")}
            aria-pressed={!isKaggle}
          >
            <Server className="h-3.5 w-3.5" />
            Local
          </button>
          <button
            type="button"
            className={cn(
              "inline-flex h-8 items-center gap-1.5 rounded px-2 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
              isKaggle ? "bg-background text-sky-600 shadow-sm" : "text-muted-foreground hover:text-foreground"
            )}
            onClick={() => setStageExecutionTarget(stage, "kaggle")}
            aria-pressed={isKaggle}
          >
            <Cloud className="h-3.5 w-3.5" />
            Kaggle
          </button>
        </div>

        {isKaggle ? (
          <Button
            type="button"
            variant={credentials ? "ghost" : "outline"}
            size="sm"
            className={cn("h-8 gap-1.5 px-2 text-xs", !credentials && "border-amber-500/40 text-amber-600")}
            onClick={() => setModalOpen(true)}
            aria-label="Configure Kaggle credentials"
          >
            {credentials ? <KeyRound className="h-3.5 w-3.5" /> : <AlertTriangle className="h-3.5 w-3.5" />}
            {credentials ? credentials.username : "Credentials"}
          </Button>
        ) : null}

        <KaggleCredentialsModal open={modalOpen} onOpenChange={setModalOpen} />
      </div>
    );
  }

  return (
    <div className={cn("space-y-3", className)}>
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-sm font-medium">
            <Icon className={cn("h-4 w-4", isKaggle ? "text-sky-600" : "text-muted-foreground")} />
            <span>Run on Kaggle</span>
          </div>
          <p className="text-xs text-muted-foreground">
            Local uses this server. Kaggle offloads to your Kaggle account or the server default.
          </p>
        </div>

        <button
          type="button"
          role="switch"
          aria-checked={isKaggle}
          aria-label={`Run stage ${stage} on Kaggle`}
          onClick={() => setStageExecutionTarget(stage, isKaggle ? "local" : "kaggle")}
          className={cn(
            "relative inline-flex h-6 w-11 shrink-0 items-center rounded-full border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
            isKaggle ? "border-sky-600 bg-sky-600" : "border-input bg-muted"
          )}
        >
          <span
            className={cn(
              "inline-block h-5 w-5 rounded-full bg-background shadow-sm transition-transform",
              isKaggle ? "translate-x-5" : "translate-x-0.5"
            )}
          />
        </button>
      </div>

      {isKaggle ? (
        <div className="space-y-2 rounded-md border bg-muted/30 p-3 text-sm">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-muted-foreground">Kaggle credentials:</span>
            <KaggleCredentialsStatus />
            {credentials ? <span className="font-mono text-xs text-muted-foreground">{credentials.username}</span> : null}
          </div>

          {!credentials ? (
            <div className="flex items-start gap-2 text-xs text-yellow-700 dark:text-yellow-300">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span>No credentials set; falls back to the server Kaggle account.</span>
            </div>
          ) : null}

          <Button type="button" variant="link" className="h-auto p-0 text-xs" onClick={() => setModalOpen(true)}>
            Configure credentials
          </Button>
        </div>
      ) : null}

      <KaggleCredentialsModal open={modalOpen} onOpenChange={setModalOpen} />
    </div>
  );
}