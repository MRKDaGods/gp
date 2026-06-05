"use client";

import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
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
  FolderOpen,
  Info,
  LogOut,
  Cpu,
  Settings,
  Cloud,
  Radar,
  Check,
  Loader2,
  X,
} from "lucide-react";
import { RunManagerDialog } from "@/components/runs/run-manager-dialog";
import { useSessionStore, useUIStore, usePipelineStore, useVideoStore, useDetectionStore, useTrackletStore, useTimelineStore } from "@/store";
import { KaggleCredentialsModal } from "@/components/settings/kaggle-credentials-modal";
import { useHasKaggleCredentials } from "@/lib/kaggle-credentials-store";
import type { StageNumber } from "@/types";
import type { ModelEntry, ModelMetric } from "@/services/models";

import { UploadStage, UploadStageActions } from "@/components/stages/upload-stage";
import { DetectionStage, DetectionStageActions } from "@/components/stages/detection-stage";
import { SelectionStage, SelectionStageActions } from "@/components/stages/selection-stage";
import { InferenceActions, InferenceStage } from "@/components/stages/inference-stage";
import { InferenceDatasetChip } from "@/components/stages/inference/InferenceDatasetChip";
import { TimelineStage, TimelineStageActions } from "@/components/stages/timeline-stage";
import { RefinementStage, RefinementStageActions } from "@/components/stages/refinement-stage";
import { OutputStage, OutputStageActions } from "@/components/stages/output-stage";
import { DatasetProcessing } from "@/components/stages/dataset-processing";
import { PipelineRunHeader, StageShell, StalenessChip, stageContract, statusMeta, type StageStatus } from "@/components/pipeline";
import { useStageState, prerequisiteStage } from "@/hooks/useStageState";
import type { ComponentType, ReactNode } from "react";

const stages = [
  { id: 0 as StageNumber, label: "Upload", icon: Upload },
  { id: 1 as StageNumber, label: "Detection", icon: Scan },
  { id: 2 as StageNumber, label: "Selection", icon: Box },
  { id: 3 as StageNumber, label: "Inference", icon: Database },
  { id: 4 as StageNumber, label: "Timeline", icon: GitBranch },
  { id: 5 as StageNumber, label: "Refinement", icon: BarChart3 },
  { id: 6 as StageNumber, label: "Output", icon: Film },
];

const PIPELINE_STAGE_COMPONENTS: { id: StageNumber; Component: ComponentType; Actions?: ComponentType }[] = [
  { id: 0, Component: UploadStage, Actions: UploadStageActions },
  { id: 1, Component: DetectionStage, Actions: DetectionStageActions },
  { id: 2, Component: SelectionStage, Actions: SelectionStageActions },
  { id: 3, Component: InferenceStage, Actions: InferenceActions },
  { id: 4, Component: TimelineStage, Actions: TimelineStageActions },
  { id: 5, Component: RefinementStage, Actions: RefinementStageActions },
  { id: 6, Component: OutputStage, Actions: OutputStageActions },
];

const HEADLINE_METRIC_PRIORITY = ["IDF1", "mAP", "R1"];
const TARGET_VISIBLE_STAGES = new Set<StageNumber>([1, 3, 4]);

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

function sidebarStatusSentence(
  stageLabel: string,
  status: StageStatus,
  progress: number,
  executionTarget: "local" | "kaggle"
): string {
  if (status === "running") {
    return `Running on ${executionTarget === "kaggle" ? "Kaggle" : "local"} · ${Math.round(progress)}% complete`;
  }
  if (status === "blocked") return `${stageLabel} is blocked - complete the previous stage first`;
  if (status === "done") return `${stageLabel} is done · ${executionTarget} execution`;
  if (status === "error") return `${stageLabel} has an error · click to view`;
  return `${stageLabel} is ${statusMeta(status).label.toLowerCase()} · ${executionTarget} execution`;
}

/** Clean stepper indicator for the sidebar — no loud dashed/amber states. */
function StageStepDot({
  status,
  isActive,
  withCloudOverlay = false,
}: {
  status: StageStatus;
  isActive: boolean;
  withCloudOverlay?: boolean;
}) {
  const base =
    "flex h-6 w-6 shrink-0 items-center justify-center rounded-full border transition-colors";
  let body: ReactNode;
  if (status === "done") {
    body = (
      <span className={cn(base, "border-success bg-success text-success-foreground")}>
        <Check className="h-3.5 w-3.5" strokeWidth={3} />
      </span>
    );
  } else if (status === "running") {
    body = (
      <span className={cn(base, "border-accent-strong bg-accent-strong/15 text-accent-strong")}>
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
      </span>
    );
  } else if (status === "error") {
    body = (
      <span className={cn(base, "border-destructive bg-destructive/10 text-destructive")}>
        <X className="h-3.5 w-3.5" strokeWidth={3} />
      </span>
    );
  } else if (status === "stale") {
    body = (
      <span className={cn(base, "border-warning/60 bg-warning/10 text-warning")}>
        <span className="h-1.5 w-1.5 rounded-full bg-current" />
      </span>
    );
  } else if (isActive) {
    body = (
      <span className={cn(base, "border-accent-strong bg-accent-strong/20 text-accent-strong")}>
        <span className="h-2 w-2 rounded-full bg-current" />
      </span>
    );
  } else {
    // idle / blocked — quiet, neutral
    body = (
      <span className={cn(base, "border-border/70 bg-transparent text-muted-foreground/50")}>
        <span className="h-1.5 w-1.5 rounded-full bg-current" />
      </span>
    );
  }
  return (
    <span className="relative inline-flex">
      {body}
      {withCloudOverlay ? (
        <Cloud className="absolute -right-1 -top-1 h-3 w-3 rounded-full bg-card text-accent-strong ring-1 ring-card" />
      ) : null}
    </span>
  );
}

function SidebarStageRow({
  stage,
  isActive,
  sidebarOpen,
  onSelect,
}: {
  stage: (typeof stages)[number];
  isActive: boolean;
  sidebarOpen: boolean;
  onSelect: () => void;
}) {
  const stageState = useStageState(stage.id);
  const progress = stageState.progress?.progress ?? 0;
  const statusSentence = sidebarStatusSentence(stage.label, stageState.status, progress, stageState.executionTarget);
  const showExecutionTarget = TARGET_VISIBLE_STAGES.has(stage.id);
  const isKaggleStage = stageState.executionTarget === "kaggle";

  return (
    <Tooltip delayDuration={0}>
      <TooltipTrigger asChild>
        <button
          onClick={onSelect}
          className={cn(
            "group relative flex w-full items-center gap-3 rounded-lg px-2.5 py-2 text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
            isActive
              ? "bg-muted/70 text-foreground"
              : "text-muted-foreground hover:bg-muted/40 hover:text-foreground",
            !sidebarOpen && "justify-center px-0"
          )}
        >
          {isActive && sidebarOpen && (
            <span className="absolute inset-y-1.5 left-0 w-0.5 rounded-full bg-accent-strong" aria-hidden />
          )}
          <div className="flex shrink-0 items-center gap-1.5">
            <StageStepDot
              status={stageState.status}
              isActive={isActive}
              withCloudOverlay={!sidebarOpen && showExecutionTarget && isKaggleStage}
            />
            {sidebarOpen && showExecutionTarget && isKaggleStage && (
              <span
                className="flex h-4 w-4 items-center justify-center"
                title="Runs on Kaggle GPU"
                aria-label="Runs on Kaggle GPU"
              >
                <Cloud className="h-3 w-3 text-accent-strong" />
              </span>
            )}
          </div>
          {sidebarOpen && (
            <span className="flex min-w-0 flex-1 flex-col items-start gap-0.5">
              <span className={cn("truncate", isActive ? "font-semibold" : "font-medium")}>{stage.label}</span>
              {stageState.status === "stale" ? (
                <StalenessChip label="Stale" />
              ) : (
                <span className="max-w-full truncate text-[11px] font-normal opacity-70">
                  {stageState.status === "blocked" ? `needs Stage ${prerequisiteStage(stage.id) ?? stage.id - 1}` : `${statusMeta(stageState.status).label.toLowerCase()}${stageState.status === "running" ? ` · ${Math.round(progress)}%` : ""}`}
                </span>
              )}
            </span>
          )}
        </button>
      </TooltipTrigger>
      <TooltipContent side="right">{statusSentence}</TooltipContent>
    </Tooltip>
  );
}

function PipelineStagePanel({
  id,
  Component,
  Actions,
  currentStage,
  setDatasetView,
  setCurrentStage,
}: {
  id: StageNumber;
  Component: ComponentType;
  Actions?: ComponentType;
  currentStage: StageNumber;
  setDatasetView: (value: boolean) => void;
  setCurrentStage: (stage: StageNumber) => void;
}) {
  const stageState = useStageState(id);
  const prereq = prerequisiteStage(id);
  const blockedBy = stageState.status === "blocked" && prereq != null
    ? { label: `Stage ${prereq}`, stage: prereq }
    : null;
  const baseContract = stageContract(id);
  const needs = id === 3
    ? [...(baseContract.needs ?? []), { label: "Dataset", render: <InferenceDatasetChip /> }]
    : baseContract.needs;

  return (
    <div
      role="tabpanel"
      id={`pipeline-stage-${id}`}
      aria-hidden={currentStage !== id}
      className={cn(
        "flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden",
        currentStage !== id && "hidden"
      )}
    >
      <StageShell
        contract={{
          ...baseContract,
          needs,
          status: stageState.status,
          blockedBy,
          produces: stageState.isStale ? baseContract.produces?.map((chip) => ({ ...chip, stale: true })) : baseContract.produces,
          onNavigateToStage: (stageId) => {
            setDatasetView(false);
            setCurrentStage(stageId);
          },
        }}
        actions={Actions ? { run: <Actions /> } : undefined}
      >
        <Component />
      </StageShell>
    </div>
  );
}

export function MainDashboard() {
  const { currentStage, isDemoMode, setCurrentStage, setDemoMode, resetSession } = useSessionStore();
  const { sidebarOpen, toggleSidebar } = useUIStore();
  const runId = usePipelineStore((s) => s.runId);
  const pipelineStages = usePipelineStore((s) => s.stages);
  const pipelineError = usePipelineStore((s) => s.error);
  const modelMode = usePipelineStore((s) => s.modelMode);
  const selectedModelMeta = usePipelineStore((s) => s.selectedModelMeta);
  const fusion = usePipelineStore((s) => s.fusion);
  const resetPipeline = usePipelineStore((s) => s.reset);
  const setCurrentVideo = useVideoStore((s) => s.setCurrentVideo);
  const setIsPlaying = useVideoStore((s) => s.setIsPlaying);
  const resetDetections = useDetectionStore((s) => s.reset);
  const resetTracklets = useTrackletStore((s) => s.reset);
  const resetTimeline = useTimelineStore((s) => s.resetAfterUpstreamEdit);
  const hasKaggleCredentials = useHasKaggleCredentials();
  const [datasetView, setDatasetView] = useState(false);
  const [kaggleSettingsOpen, setKaggleSettingsOpen] = useState(false);
  const [runManagerOpen, setRunManagerOpen] = useState(false);
  const [visitedPipelineStages, setVisitedPipelineStages] = useState<Set<StageNumber>>(
    () => new Set([currentStage])
  );

  useEffect(() => {
    setVisitedPipelineStages((prev) => new Set(prev).add(currentStage));
  }, [currentStage]);

  const modelBadgeBody = formatModelBadgeBody(modelMode, selectedModelMeta, fusion);
  const openInferenceStage = () => {
    setDatasetView(false);
    setCurrentStage(3);
  };

  const exitDemoMode = () => {
    resetPipeline();
    resetDetections();
    resetTracklets();
    resetTimeline();
    setCurrentVideo(null);
    setIsPlaying(false);
    resetSession();
    setDemoMode(false);
    setDatasetView(false);
  };

  return (
    <div className="flex h-dvh max-h-dvh min-h-0 overflow-hidden bg-background">
      {/* Sidebar */}
      <aside
        className={cn(
          "flex min-h-0 min-w-0 flex-shrink-0 flex-col overflow-x-hidden border-r border-border/60 bg-card/40 transition-all duration-300",
          sidebarOpen ? "w-56" : "w-14"
        )}
      >
        {/* Brand + collapse toggle */}
        <div className={cn("flex h-12 shrink-0 items-center border-b border-border/60 px-2", sidebarOpen ? "justify-between" : "justify-center")}>
          {sidebarOpen && (
            <div className="flex items-center gap-2 pl-1">
              <Radar className="h-5 w-5 text-primary" />
              <span className="text-sm font-semibold tracking-tight">ATHAR</span>
            </div>
          )}
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
            return (
              <SidebarStageRow
                key={stage.id}
                stage={stage}
                isActive={isActive}
                sidebarOpen={sidebarOpen}
                onSelect={() => { setDatasetView(false); setCurrentStage(stage.id); }}
              />
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
                    <span className="absolute right-0 top-0 h-2 w-2 rounded-full bg-success ring-2 ring-card" />
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

          {/* Runs manager */}
          <Tooltip delayDuration={0}>
            <TooltipTrigger asChild>
              <button
                onClick={() => setRunManagerOpen(true)}
                aria-label="Manage runs"
                className={cn(
                  "group flex w-full items-center gap-3 rounded-md px-2 py-2 text-sm font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                  !sidebarOpen && "justify-center px-0"
                )}
              >
                <Database className="h-4 w-4 shrink-0" />
                {sidebarOpen && <span className="truncate">Runs</span>}
              </button>
            </TooltipTrigger>
            {!sidebarOpen && (
              <TooltipContent side="right">Runs</TooltipContent>
            )}
          </Tooltip>

          {/* Dataset */}
          <Tooltip delayDuration={0}>
            <TooltipTrigger asChild>
              <button
                onClick={() => setDatasetView(true)}
                aria-label="Open dataset workspace"
                className={cn(
                  "group flex w-full items-center gap-3 rounded-md px-2 py-2 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
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
        <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
          {datasetView ? (
            <DatasetProcessing />
          ) : (
            <div className="relative flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
              <PipelineRunHeader
                runId={runId}
                currentStage={currentStage}
                stages={pipelineStages}
                stageLabels={stages}
                error={pipelineError}
                onSelectErrorStage={(stageId) => {
                  setDatasetView(false);
                  setCurrentStage(stageId);
                }}
              />
              {isDemoMode ? (
                <div className="flex shrink-0 items-center gap-3 border-b border-border/60 bg-accent-strong/10 px-4 py-2 text-sm sm:px-6">
                  <Info className="h-4 w-4 shrink-0 text-accent-strong" aria-hidden="true" />
                  <Badge variant="outline" className="border-accent-strong/40 text-accent-strong">Demo run</Badge>
                  <span className="min-w-0 flex-1 truncate text-muted-foreground">Demo run loaded from a local sample video. Backend pipeline work has not started.</span>
                  <Button type="button" variant="destructive" size="sm" className="h-7 gap-1.5 px-2 text-xs" onClick={exitDemoMode}>
                    <LogOut className="h-3.5 w-3.5" />
                    Exit demo
                  </Button>
                </div>
              ) : null}
              {PIPELINE_STAGE_COMPONENTS.map(({ id, Component, Actions }) =>
                visitedPipelineStages.has(id) ? (
                  <PipelineStagePanel
                    key={id}
                    id={id}
                    Component={Component}
                    Actions={Actions}
                    currentStage={currentStage}
                    setDatasetView={setDatasetView}
                    setCurrentStage={setCurrentStage}
                  />
                ) : null
              )}
            </div>
          )}
        </div>
      </main>
      <KaggleCredentialsModal open={kaggleSettingsOpen} onOpenChange={setKaggleSettingsOpen} />
      <RunManagerDialog open={runManagerOpen} onOpenChange={setRunManagerOpen} />
    </div>
  );
}
