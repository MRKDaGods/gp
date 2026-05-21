"use client";

import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Upload,
  Box,
  Scan,
  Database,
  GitBranch,
  BarChart3,
  Film,
  ChevronLeft,
  ChevronRight,
  Loader2,
  FolderOpen,
  Check,
  Cpu,
  Settings,
  Cloud,
  Server,
} from "lucide-react";
import { useSessionStore, useUIStore, usePipelineStore, useStageExecutionStore } from "@/store";
import { KaggleCredentialsModal } from "@/components/settings/kaggle-credentials-modal";
import { useHasKaggleCredentials } from "@/lib/kaggle-credentials-store";
import type { StageNumber } from "@/types";
import type { ModelEntry, ModelMetric } from "@/services/models";
import { GlobalProcessingBanner } from "@/components/layout/global-processing-banner";

import { UploadStage } from "@/components/stages/upload-stage";
import { DetectionStage } from "@/components/stages/detection-stage";
import { SelectionStage } from "@/components/stages/selection-stage";
import { InferenceStage } from "@/components/stages/inference-stage";
import { TimelineStage } from "@/components/stages/timeline-stage";
import { RefinementStage } from "@/components/stages/refinement-stage";
import { OutputStage } from "@/components/stages/output-stage";
import { DatasetProcessing } from "@/components/stages/dataset-processing";
import type { ComponentType } from "react";

const stages = [
  { id: 0 as StageNumber, label: "Upload", icon: Upload },
  { id: 1 as StageNumber, label: "Detection", icon: Scan },
  { id: 2 as StageNumber, label: "Selection", icon: Box },
  { id: 3 as StageNumber, label: "Inference", icon: Database },
  { id: 4 as StageNumber, label: "Timeline", icon: GitBranch },
  { id: 5 as StageNumber, label: "Refinement", icon: BarChart3 },
  { id: 6 as StageNumber, label: "Output", icon: Film },
];

const PIPELINE_STAGE_COMPONENTS: { id: StageNumber; Component: ComponentType }[] = [
  { id: 0, Component: UploadStage },
  { id: 1, Component: DetectionStage },
  { id: 2, Component: SelectionStage },
  { id: 3, Component: InferenceStage },
  { id: 4, Component: TimelineStage },
  { id: 5, Component: RefinementStage },
  { id: 6, Component: OutputStage },
];

const HEADLINE_METRIC_PRIORITY = ["IDF1", "mAP", "R1"];

function getHeadlineMetric(model: ModelEntry | null): ModelMetric | null {
  if (!model) return null;

  for (const metricName of HEADLINE_METRIC_PRIORITY) {
    const metric = model.metrics.find(
      (candidate) => candidate.verified && candidate.name.toLowerCase() === metricName.toLowerCase()
    );
    if (metric) return metric;
  }

  return null;
}

function formatModelBadgeBody(
  modelMode: "single" | "fusion",
  selectedModelMeta: ModelEntry | null,
  fusion: { models: Array<{ modelId: string; weight: number }> } | null
): { primary: string; secondary: string | null; isFallback: boolean } {
  if (modelMode === "single" && selectedModelMeta) {
    const metric = getHeadlineMetric(selectedModelMeta);

    return {
      primary: selectedModelMeta.name,
      secondary: metric ? `${metric.name} ${metric.value.toFixed(4)}` : null,
      isFallback: false,
    };
  }

  if (modelMode === "fusion" && fusion?.models?.length) {
    const firstModelNames = fusion.models.slice(0, 2).map((model) => model.modelId).join(", ");

    return {
      primary: `Fusion · ${fusion.models.length} models`,
      secondary: firstModelNames || null,
      isFallback: false,
    };
  }

  return { primary: "Using legacy config", secondary: null, isFallback: true };
}

export function MainDashboard() {
  const { currentStage, setCurrentStage } = useSessionStore();
  const { sidebarOpen, toggleSidebar } = useUIStore();
  const pipelineStages = usePipelineStore((s) => s.stages);
  const modelMode = usePipelineStore((s) => s.modelMode);
  const selectedModelMeta = usePipelineStore((s) => s.selectedModelMeta);
  const fusion = usePipelineStore((s) => s.fusion);
  const getStageExecutionTarget = useStageExecutionStore(
    (s) => (stage: StageNumber) => s.stageExecutionTargets[stage] ?? s.getStageExecutionTarget(stage)
  );
  const hasKaggleCredentials = useHasKaggleCredentials();
  const [datasetView, setDatasetView] = useState(false);
  const [kaggleSettingsOpen, setKaggleSettingsOpen] = useState(false);
  const [visitedPipelineStages, setVisitedPipelineStages] = useState<Set<StageNumber>>(
    () => new Set([currentStage])
  );

  useEffect(() => {
    setVisitedPipelineStages((prev) => new Set(prev).add(currentStage));
  }, [currentStage]);

  const getStageStatus = (stageId: number) =>
    pipelineStages.find((s) => s.stage === stageId)?.status ?? "idle";

  const modelBadgeBody = formatModelBadgeBody(modelMode, selectedModelMeta, fusion);
  const openInferenceStage = () => {
    setDatasetView(false);
    setCurrentStage(3);
  };

  return (
    <div className="flex h-dvh max-h-dvh min-h-0 overflow-hidden bg-background">
      {/* Sidebar */}
      <aside
        className={cn(
          "flex min-h-0 min-w-0 flex-shrink-0 flex-col overflow-x-hidden border-r bg-card transition-all duration-300",
          sidebarOpen ? "w-56" : "w-14"
        )}
      >
        {/* Toggle at the top */}
        <div className={cn("flex shrink-0 items-center border-b", sidebarOpen ? "justify-end px-2 py-2" : "justify-center py-2")}>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={toggleSidebar}
            aria-label={sidebarOpen ? "Collapse sidebar" : "Expand sidebar"}
          >
            {sidebarOpen ? <ChevronLeft className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
          </Button>
        </div>

        {/* Pipeline stages */}
        <nav className="flex min-h-0 flex-1 flex-col gap-0.5 overflow-y-auto px-2 py-3">
          {stages.map((stage) => {
            const isActive = !datasetView && currentStage === stage.id;
            const status = getStageStatus(stage.id);
            const isCompleted = status === "completed";
            const isRunning = status === "running";
            const isError = status === "error";
            const executionTarget = getStageExecutionTarget(stage.id);
            const isKaggleStage = executionTarget === "kaggle";

            return (
              <Tooltip key={stage.id} delayDuration={0}>
                <TooltipTrigger asChild>
                  <button
                    onClick={() => { setDatasetView(false); setCurrentStage(stage.id); }}
                    className={cn(
                      "group flex w-full items-center gap-3 rounded-md px-2 py-2 text-sm font-medium transition-colors",
                      isActive
                        ? "bg-primary text-primary-foreground"
                        : "text-muted-foreground hover:bg-muted hover:text-foreground",
                      !sidebarOpen && "justify-center px-0"
                    )}
                  >
                    <div className="flex shrink-0 items-center gap-1.5">
                      <div className="relative">
                        <div className={cn(
                          "flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-[11px] font-semibold",
                          isActive && "bg-primary-foreground/20 text-primary-foreground",
                          isCompleted && !isActive && "bg-green-600/15 text-green-500",
                          isError && !isActive && "bg-red-600/15 text-red-500",
                          !isActive && !isCompleted && !isError && "bg-muted-foreground/10 text-muted-foreground",
                        )}>
                          {isRunning ? (
                            <Loader2 className="h-3 w-3 animate-spin" />
                          ) : isCompleted ? (
                            <Check className="h-3 w-3" />
                          ) : isError ? (
                            <span className="text-[10px]">!</span>
                          ) : (
                            stage.id
                          )}
                        </div>
                        {/* Collapsed local stages skip the Server badge to keep the compact rail legible. */}
                        {!sidebarOpen && isKaggleStage && (
                          <Cloud className="absolute -right-1 -top-1 h-3 w-3 rounded-full bg-card text-blue-500 ring-1 ring-card" />
                        )}
                      </div>
                      {sidebarOpen && (
                        <span
                          className="flex h-4 w-4 items-center justify-center"
                          title={isKaggleStage ? "Kaggle execution" : "Local execution"}
                          aria-label={isKaggleStage ? "Kaggle execution" : "Local execution"}
                        >
                          {isKaggleStage ? (
                            <Cloud className="h-3 w-3 text-blue-500" />
                          ) : (
                            <Server className="h-3 w-3 text-muted-foreground" />
                          )}
                        </span>
                      )}
                    </div>
                    {sidebarOpen && (
                      <span className="truncate">{stage.label}</span>
                    )}
                  </button>
                </TooltipTrigger>
                {!sidebarOpen && (
                  <TooltipContent side="right">
                    {isKaggleStage ? `${stage.label} · Kaggle execution` : stage.label}
                  </TooltipContent>
                )}
              </Tooltip>
            );
          })}

          <div className="mt-auto" />

          {/* Kaggle credentials */}
          <Tooltip delayDuration={0}>
            <TooltipTrigger asChild>
              <button
                onClick={() => setKaggleSettingsOpen(true)}
                className={cn(
                  "group flex w-full items-center rounded-md text-sm font-medium transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                  hasKaggleCredentials ? "text-foreground" : "text-muted-foreground",
                  sidebarOpen ? "gap-3 px-2 py-2" : "h-9 justify-center px-0"
                )}
                aria-label="Kaggle credentials"
              >
                <span className="relative flex h-6 w-6 shrink-0 items-center justify-center">
                  <Settings className="h-4 w-4" />
                  {hasKaggleCredentials && (
                    <span className="absolute right-0 top-0 h-2 w-2 rounded-full bg-green-500 ring-2 ring-card" />
                  )}
                </span>
                {sidebarOpen && <span className="truncate">Kaggle credentials</span>}
              </button>
            </TooltipTrigger>
            {!sidebarOpen && (
              <TooltipContent side="right">Kaggle credentials</TooltipContent>
            )}
          </Tooltip>

          {/* Active model */}
          <Tooltip delayDuration={0}>
            <TooltipTrigger asChild>
              <button
                onClick={openInferenceStage}
                className={cn(
                  "transition-colors hover:border-primary/40 hover:bg-muted/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                  sidebarOpen
                    ? "rounded-md border bg-muted/30 px-3 py-2 text-left text-xs"
                    : "flex h-9 w-full items-center justify-center rounded-md text-muted-foreground hover:text-foreground"
                )}
                aria-label="Open active model selection"
              >
                {sidebarOpen ? (
                  <>
                    <div className="mb-1 flex items-center gap-1.5 text-muted-foreground">
                      <Cpu className="h-3.5 w-3.5" />
                      <span>Active model</span>
                    </div>
                    <div
                      className={cn(
                        "truncate font-medium",
                        modelBadgeBody.isFallback && "font-normal italic text-muted-foreground"
                      )}
                    >
                      {modelBadgeBody.primary}
                    </div>
                    {modelBadgeBody.secondary && (
                      <div className="truncate text-muted-foreground">{modelBadgeBody.secondary}</div>
                    )}
                  </>
                ) : (
                  <Cpu className="h-4 w-4" />
                )}
              </button>
            </TooltipTrigger>
            {!sidebarOpen && (
              <TooltipContent side="right">
                {modelBadgeBody.secondary
                  ? `${modelBadgeBody.primary} · ${modelBadgeBody.secondary}`
                  : modelBadgeBody.primary}
              </TooltipContent>
            )}
          </Tooltip>

          {/* Spacer */}
          <div className="my-2 h-px bg-border" />

          {/* Dataset */}
          <Tooltip delayDuration={0}>
            <TooltipTrigger asChild>
              <button
                onClick={() => setDatasetView(true)}
                className={cn(
                  "group flex w-full items-center gap-3 rounded-md px-2 py-2 text-sm font-medium transition-colors",
                  datasetView
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground",
                  !sidebarOpen && "justify-center px-0"
                )}
              >
                <FolderOpen className="h-4 w-4 shrink-0" />
                {sidebarOpen && <span className="truncate">Dataset</span>}
              </button>
            </TooltipTrigger>
            {!sidebarOpen && (
              <TooltipContent side="right">Dataset</TooltipContent>
            )}
          </Tooltip>
        </nav>
      </aside>

      {/* Main content */}
      <main className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
        <GlobalProcessingBanner />
        <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
          {datasetView ? (
            <DatasetProcessing />
          ) : (
            <div className="relative flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
              {PIPELINE_STAGE_COMPONENTS.map(({ id, Component }) =>
                visitedPipelineStages.has(id) ? (
                  <div
                    key={id}
                    role="tabpanel"
                    id={`pipeline-stage-${id}`}
                    aria-hidden={currentStage !== id}
                    className={cn(
                      "flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden",
                      currentStage !== id && "hidden"
                    )}
                  >
                    <Component />
                  </div>
                ) : null
              )}
            </div>
          )}
        </div>
      </main>
      <KaggleCredentialsModal open={kaggleSettingsOpen} onOpenChange={setKaggleSettingsOpen} />
    </div>
  );
}
