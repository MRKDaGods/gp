"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  ExternalLink,
  Loader2,
  RefreshCw,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { fetchModels, type ModelEntry, type ModelStatus, type ModelTaskType } from "@/services/models";

type StatusFilter = "all" | "production" | "research";

interface ModelPickerProps {
  selectedId: string | null;
  onSelect: (modelId: string | null) => void;
  onModelChange?: (model: ModelEntry | null) => void;
  taskType?: ModelTaskType;
  multiSelect?: boolean;
  selectedIds?: string[];
  onMultiSelect?: (modelIds: string[]) => void;
  allowUnavailableSelection?: boolean;
  compact?: boolean;
}

const TASK_LABELS: Record<ModelTaskType, string> = {
  mtmc_vehicle: "Vehicle MTMC",
  mtmc_person: "Person MTMC",
  single_cam_reid: "Single-Cam ReID",
  detector_only: "Detector",
};

const TASK_ORDER: ModelTaskType[] = ["mtmc_vehicle", "mtmc_person", "single_cam_reid", "detector_only"];

const STATUS_STYLES: Record<ModelStatus, string> = {
  production: "border-green-600/30 bg-green-600/10 text-green-700",
  research: "border-yellow-600/30 bg-yellow-600/10 text-yellow-700",
  dead_end: "border-muted bg-muted text-muted-foreground",
  reference: "border-blue-600/30 bg-blue-600/10 text-blue-700",
};

const METRIC_LABELS: Record<string, string> = {
  mtmc_idf1: "IDF1",
  idf1_groundplane: "IDF1",
  idf1: "IDF1",
  map: "mAP",
  map_post_rerank: "mAP rerank",
  r1: "R1",
  r5: "R5",
  moda: "MODA",
  modp: "MODP",
  precision: "Precision",
  recall: "Recall",
};

function formatMetricName(name: string): string {
  return METRIC_LABELS[name] ?? name.replace(/_/g, " ").toUpperCase();
}

function formatMetricValue(value: number): string {
  if (!Number.isFinite(value)) return "n/a";
  return value >= 10 ? value.toFixed(2) : value.toFixed(5).replace(/0+$/, "").replace(/\.$/, "");
}

function getKernelHref(ref: string): string {
  if (/^https?:\/\//i.test(ref)) return ref;
  if (ref.includes("/")) return `https://www.kaggle.com/code/${ref}`;
  return ref;
}

function getDisabledReason(model: ModelEntry, allowUnavailableSelection = false): string | null {
  if (allowUnavailableSelection) return null;
  if (!model.runnable_locally) return "This registry entry is Kaggle-only or metadata-only.";
  if (model.missing_checkpoints.length > 0) {
    return `Missing weights: ${model.missing_checkpoints.join(", ")}`;
  }
  if (!model.pipeline_config) return "This entry does not define a runnable pipeline config.";
  return null;
}

export function ModelPicker({
  selectedId,
  onSelect,
  onModelChange,
  taskType,
  multiSelect = false,
  selectedIds = [],
  onMultiSelect,
  allowUnavailableSelection = false,
  compact = false,
}: ModelPickerProps) {
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [showDeadEnds, setShowDeadEnds] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadModels = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      const status = statusFilter === "all" ? undefined : statusFilter;
      const entries = await fetchModels({ task_type: taskType, status, include_dead_ends: showDeadEnds });
      setModels(entries);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load model registry");
      setModels([]);
    } finally {
      setIsLoading(false);
    }
  }, [showDeadEnds, statusFilter, taskType]);

  useEffect(() => {
    void loadModels();
  }, [loadModels]);

  const selectedModel = useMemo(
    () => models.find((model) => model.id === selectedId) ?? null,
    [models, selectedId]
  );

  useEffect(() => {
    onModelChange?.(selectedModel);
  }, [onModelChange, selectedModel]);

  const grouped = useMemo(() => {
    return TASK_ORDER.map((taskType) => ({
      taskType,
      models: models.filter((model) => model.task_type === taskType),
    })).filter((group) => group.models.length > 0);
  }, [models]);

  const toggleMulti = useCallback((modelId: string) => {
    if (!onMultiSelect) return;
    const next = selectedIds.includes(modelId)
      ? selectedIds.filter((id) => id !== modelId)
      : [...selectedIds, modelId];
    onMultiSelect(next);
  }, [onMultiSelect, selectedIds]);

  return (
    <TooltipProvider>
      <div className="space-y-4">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex flex-wrap gap-2">
            {(["all", "production", "research"] as StatusFilter[]).map((status) => (
              <Button
                key={status}
                type="button"
                variant={statusFilter === status ? "default" : "outline"}
                size="sm"
                onClick={() => setStatusFilter(status)}
                className="h-8"
              >
                {status === "all" ? "All" : status[0].toUpperCase() + status.slice(1)}
              </Button>
            ))}
            <Button
              type="button"
              variant={showDeadEnds ? "secondary" : "outline"}
              size="sm"
              onClick={() => setShowDeadEnds((value) => !value)}
              className="h-8"
            >
              Show dead ends
            </Button>
          </div>
          <div className="flex items-center gap-2">
            {!multiSelect && selectedId && (
              <Button type="button" variant="ghost" size="sm" onClick={() => onSelect(null)}>
                Use legacy config
              </Button>
            )}
            <Button type="button" variant="ghost" size="icon" onClick={() => void loadModels()} aria-label="Refresh models">
              <RefreshCw className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {isLoading && (
          <div className="flex items-center gap-2 rounded-lg border border-dashed p-4 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            Loading model registry...
          </div>
        )}

        {error && (
          <div className="flex items-start gap-2 rounded-lg border border-destructive/30 bg-destructive/10 p-4 text-sm">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-destructive" />
            <div className="min-w-0">
              <p className="font-medium text-destructive">Could not load models</p>
              <p className="break-words text-muted-foreground">{error}</p>
            </div>
          </div>
        )}

        {!isLoading && !error && grouped.length === 0 && (
          <div className="rounded-lg border border-dashed p-4 text-sm text-muted-foreground">
            No registry entries match the current filters.
          </div>
        )}

        {grouped.map((group) => (
          <section key={group.taskType} className="space-y-3">
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-semibold">{TASK_LABELS[group.taskType]}</h3>
              <Badge variant="secondary" className="text-[10px]">
                {group.models.length}
              </Badge>
            </div>
            <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
              {group.models.map((model) => {
                const isSelected = multiSelect ? selectedIds.includes(model.id) : selectedId === model.id;
                const disabledReason = getDisabledReason(model, allowUnavailableSelection);
                const metrics = model.metrics.slice(0, compact ? 1 : 2);

                return (
                  <Card
                    key={model.id}
                    className={cn(
                      "border-2 transition-all",
                      isSelected ? "border-primary bg-primary/5 shadow-md" : "border-muted",
                      model.status === "dead_end" && "opacity-75"
                    )}
                  >
                    <CardContent className={cn("space-y-4 p-4", compact && "space-y-3 p-3")}>
                      <div className="flex items-start justify-between gap-3">
                        <div className="min-w-0 space-y-1">
                          <p className={cn("font-semibold leading-snug", model.status === "dead_end" && "line-through")}>
                            {model.name}
                          </p>
                          <p className="text-xs text-muted-foreground">{model.dataset}</p>
                        </div>
                        <span className={cn("shrink-0 rounded-full border px-2 py-0.5 text-[10px] font-semibold", STATUS_STYLES[model.status])}>
                          {model.status}
                        </span>
                      </div>

                      <div className="grid gap-2 sm:grid-cols-2">
                        {metrics.map((metric) => (
                          <div key={`${model.id}-${metric.name}`} className="rounded-md border bg-background p-2">
                            <div className="flex items-center justify-between gap-2">
                              <span className="text-xs text-muted-foreground">{formatMetricName(metric.name)}</span>
                              {metric.verified ? (
                                <Badge variant="outline" className="gap-1 border-green-600 text-[10px] text-green-700">
                                  <CheckCircle2 className="h-3 w-3" />
                                  verified
                                </Badge>
                              ) : (
                                <Badge variant="outline" className="text-[10px] text-muted-foreground">
                                  unverified
                                </Badge>
                              )}
                            </div>
                            <p className="mt-1 text-lg font-semibold leading-none">
                              {formatMetricName(metric.name)} = {formatMetricValue(metric.value)}
                            </p>
                          </div>
                        ))}
                      </div>

                      {!compact && <p className="line-clamp-3 text-sm text-muted-foreground">{model.description}</p>}

                      <div className="flex flex-wrap items-center gap-2">
                        {model.requirements.gpu_required && (
                          <Badge variant="outline" className="text-[10px]">GPU {model.requirements.min_vram_gb}GB</Badge>
                        )}
                        {model.missing_checkpoints.length > 0 && (
                          <Badge variant="warning" className="text-[10px]">Weights missing</Badge>
                        )}
                        {model.notebook_or_kernel_ref && (
                          <Button asChild variant="ghost" size="sm" className="h-7 px-2 text-xs">
                            <a href={getKernelHref(model.notebook_or_kernel_ref)} target="_blank" rel="noreferrer">
                              Notebook/kernel
                              <ExternalLink className="ml-1 h-3 w-3" />
                            </a>
                          </Button>
                        )}
                      </div>

                      <div className="flex items-center justify-between gap-2">
                        {!model.runnable_locally && model.notebook_or_kernel_ref ? (
                          <Button asChild variant="outline" size="sm">
                            <a href={getKernelHref(model.notebook_or_kernel_ref)} target="_blank" rel="noreferrer">
                              Reproduce on Kaggle
                              <ExternalLink className="ml-2 h-4 w-4" />
                            </a>
                          </Button>
                        ) : <span />}
                        {disabledReason ? (
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <span className="inline-flex">
                                <Button type="button" size="sm" disabled>
                                  Select
                                </Button>
                              </span>
                            </TooltipTrigger>
                            <TooltipContent>{disabledReason}</TooltipContent>
                          </Tooltip>
                        ) : (
                          <Button
                            type="button"
                            size="sm"
                            variant={isSelected ? "default" : "outline"}
                            onClick={() => multiSelect ? toggleMulti(model.id) : onSelect(isSelected ? null : model.id)}
                          >
                            {isSelected ? "Selected" : multiSelect ? "Add" : "Select"}
                          </Button>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </section>
        ))}
      </div>
    </TooltipProvider>
  );
}