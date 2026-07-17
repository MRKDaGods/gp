"use client";

import type { ReactNode } from "react";
import { CheckCircle2, Database, FolderOpen, Loader2, Video } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";
import { usePipelineStore, useVideoStore } from "@/store";

import {
  UPLOADED_DATASET,
  useInferenceDatasets,
  useInferenceSourceStore,
} from "./InferenceSourceCard";

/**
 * Primary "search within" selector for Inference. You pick the dataset your probe
 * is searched against. Precomputed datasets are used immediately; a not-yet-
 * processed dataset is auto-processed inline when you run the search.
 */
export function InferenceSearchTargetCard() {
  const currentVideo = useVideoStore((state) => state.currentVideo);
  const selectedModelMeta = usePipelineStore((state) => state.selectedModelMeta);
  const { datasets, selectedDataset, datasetsLoading, fetchDatasets } = useInferenceDatasets();
  const setSelectedDataset = useInferenceSourceStore((state) => state.setSelectedDataset);
  const processingDataset = useInferenceSourceStore((state) => state.processingDataset);
  const processProgress = useInferenceSourceStore((state) => state.processProgress);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Database className="h-5 w-5" />
          Search within
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Model and gallery are independent: a model is a ReID recipe we can
            run against any dataset. So the folder picker is ALWAYS shown; the
            model (if any) only sets the recipe/config, never the search target. */}
        {selectedModelMeta ? (
          <p className="text-xs text-muted-foreground">
            Model <span className="font-medium text-foreground">{selectedModelMeta.id}</span> sets the
            ReID recipe (<span className="uppercase">{selectedModelMeta.dataset}</span>). Choose which
            gallery to search below.
          </p>
        ) : (
          <p className="text-xs text-muted-foreground">
            Choose where to search for your probe. Precomputed datasets run
            immediately; a fresh dataset is processed automatically first.
          </p>
        )}

        <TargetOption
          active={selectedDataset === UPLOADED_DATASET}
          icon={<Video className="h-4 w-4 text-primary" />}
          title="Uploaded video only"
          description={currentVideo?.name ?? "No probe uploaded"}
          disabled={Boolean(processingDataset)}
          onClick={() => setSelectedDataset(UPLOADED_DATASET)}
        />

        {datasetsLoading ? (
          <div className="flex items-center gap-2 rounded-md border border-dashed p-3 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            Loading datasets...
          </div>
        ) : null}

        {datasets.map((dataset) => {
          const isThisProcessing =
            processingDataset === dataset.name || (dataset.isProcessing ?? false);
          return (
            <TargetOption
              key={dataset.name}
              active={selectedDataset === dataset.name}
              icon={<FolderOpen className="h-4 w-4 text-warning" />}
              title={dataset.name}
              description={`${dataset.cameraCount} camera${dataset.cameraCount === 1 ? "" : "s"}, ${dataset.videosFound} video${dataset.videosFound === 1 ? "" : "s"}`}
              status={
                isThisProcessing
                  ? "processing"
                  : dataset.hasGallery
                    ? "ready"
                    : "unprocessed"
              }
              disabled={Boolean(processingDataset) && processingDataset !== dataset.name}
              onClick={() => setSelectedDataset(dataset.name)}
            />
          );
        })}

        {!datasetsLoading && datasets.length === 0 ? (
          <div className="rounded-md border border-dashed p-3 text-center text-sm text-muted-foreground">
            <p>No dataset folders found in dataset/</p>
            <Button type="button" variant="ghost" size="sm" className="mt-1" onClick={() => void fetchDatasets()}>
              Retry
            </Button>
          </div>
        ) : null}

        {processingDataset && processProgress ? (
          <div className="space-y-1 rounded-md border border-primary/30 bg-primary/5 p-3">
            <div className="flex items-center justify-between text-sm">
              <span className="flex items-center gap-2 text-foreground">
                <Loader2 className="h-4 w-4 animate-spin text-primary" />
                Processing {processingDataset}...
              </span>
              <span className="font-mono text-xs">{Math.round(processProgress.progress)}%</span>
            </div>
            <Progress value={processProgress.progress} className="h-2" />
            <p className="truncate text-xs text-muted-foreground">{processProgress.message}</p>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}

function StatusBadge({ status }: { status: "ready" | "unprocessed" | "processing" }) {
  if (status === "ready") {
    return (
      <Badge variant="outline" className="border-success/30 text-success">
        Precomputed
      </Badge>
    );
  }
  if (status === "processing") {
    return (
      <Badge variant="outline" className="border-primary/30 text-primary">
        <Loader2 className="mr-1 h-3 w-3 animate-spin" />
        Processing
      </Badge>
    );
  }
  return (
    <Badge variant="outline" className="border-warning/30 text-warning">
      Not processed
    </Badge>
  );
}

function TargetOption({
  active,
  icon,
  title,
  description,
  status,
  disabled,
  onClick,
}: {
  active: boolean;
  icon: ReactNode;
  title: string;
  description: string;
  status?: "ready" | "unprocessed" | "processing";
  disabled?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      className={cn(
        "flex w-full items-center gap-3 rounded-md border p-3 text-left transition-colors hover:border-primary/50 hover:bg-muted/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        active && "border-primary bg-primary/5",
        disabled && "cursor-not-allowed opacity-50"
      )}
      onClick={onClick}
      aria-pressed={active}
    >
      {icon}
      <span className="min-w-0 flex-1">
        <span className="block truncate text-sm font-medium">{title}</span>
        <span className="block truncate text-xs text-muted-foreground">{description}</span>
      </span>
      {status ? <StatusBadge status={status} /> : null}
      {active ? <CheckCircle2 className="h-4 w-4 shrink-0 text-primary" /> : null}
    </button>
  );
}
