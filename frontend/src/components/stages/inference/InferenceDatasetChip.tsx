"use client";

import type { ReactNode } from "react";
import { CheckCircle2, ChevronDown, FolderOpen, Loader2, Video } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import { usePipelineStore, useVideoStore } from "@/store";

import { useInferenceDatasets, useInferenceSourceStore } from "./InferenceSourceCard";

export function InferenceDatasetChip() {
  const currentVideo = useVideoStore((state) => state.currentVideo);
  const selectedModelMeta = usePipelineStore((state) => state.selectedModelMeta);
  const { datasets, selectedDataset, datasetsLoading, fetchDatasets } = useInferenceDatasets();
  const setSelectedDataset = useInferenceSourceStore((state) => state.setSelectedDataset);
  const selectedDatasetLabel = selectedModelMeta?.dataset ?? (selectedDataset === "__uploaded__" ? currentVideo?.name ?? "uploaded video" : selectedDataset);

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          type="button"
          variant="outline"
          size="sm"
          className="h-6 max-w-[240px] gap-1.5 truncate border-primary/30 bg-primary/5 px-2 text-xs"
          aria-label="Choose inference dataset source"
        >
          <FolderOpen className="h-3.5 w-3.5 shrink-0" />
          <span className="truncate">dataset: {selectedDatasetLabel}</span>
          <ChevronDown className="h-3 w-3 shrink-0" />
        </Button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-80 space-y-3 p-3">
        <div className="space-y-1">
          <div className="text-sm font-medium">Dataset source</div>
          {selectedModelMeta ? (
            <p className="text-xs text-muted-foreground">The selected registry model fixes the dataset to {selectedModelMeta.dataset}.</p>
          ) : (
            <p className="text-xs text-muted-foreground">Choose the upstream source used by Stage 2 features and Stage 3 indexing.</p>
          )}
        </div>

        {selectedModelMeta ? (
          <Badge variant="secondary" className="uppercase">{selectedModelMeta.dataset}</Badge>
        ) : (
          <div className="space-y-2">
            <DatasetOption
              active={selectedDataset === "__uploaded__"}
              icon={<Video className="h-4 w-4 text-primary" />}
              title="Uploaded video"
              description={currentVideo?.name ?? "No video uploaded"}
              onClick={() => setSelectedDataset("__uploaded__")}
            />

            {datasetsLoading ? (
              <div className="flex items-center gap-2 rounded-md border border-dashed p-3 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                Loading datasets...
              </div>
            ) : null}

            {datasets.map((dataset) => (
              <DatasetOption
                key={dataset.name}
                active={selectedDataset === dataset.name}
                icon={<FolderOpen className="h-4 w-4 text-warning" />}
                title={dataset.name}
                description={`${dataset.cameraCount} cameras, ${dataset.videosFound} videos`}
                badge={dataset.alreadyProcessed ? "Processed" : undefined}
                onClick={() => setSelectedDataset(dataset.name)}
              />
            ))}

            {!datasetsLoading && datasets.length === 0 ? (
              <div className="rounded-md border border-dashed p-3 text-center text-sm text-muted-foreground">
                <p>No dataset folders found in dataset/</p>
                <Button type="button" variant="ghost" size="sm" className="mt-1" onClick={() => void fetchDatasets()}>
                  Retry
                </Button>
              </div>
            ) : null}
          </div>
        )}
      </PopoverContent>
    </Popover>
  );
}

function DatasetOption({
  active,
  icon,
  title,
  description,
  badge,
  onClick,
}: {
  active: boolean;
  icon: ReactNode;
  title: string;
  description: string;
  badge?: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className={cn(
        "flex w-full items-center gap-3 rounded-md border p-3 text-left transition-colors hover:border-primary/50 hover:bg-muted/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        active && "border-primary bg-primary/5"
      )}
      onClick={onClick}
      aria-pressed={active}
    >
      {icon}
      <span className="min-w-0 flex-1">
        <span className="block truncate text-sm font-medium">{title}</span>
        <span className="block truncate text-xs text-muted-foreground">{description}</span>
      </span>
      {badge ? <Badge variant="outline" className="text-[10px]">{badge}</Badge> : null}
      {active ? <CheckCircle2 className="h-4 w-4 shrink-0 text-primary" /> : null}
    </button>
  );
}