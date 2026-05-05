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
} from "lucide-react";
import { useSessionStore, useUIStore, usePipelineStore } from "@/store";
import type { StageNumber } from "@/types";
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

export function MainDashboard() {
  const { currentStage, setCurrentStage } = useSessionStore();
  const { sidebarOpen, toggleSidebar } = useUIStore();
  const pipelineStages = usePipelineStore((s) => s.stages);
  const [datasetView, setDatasetView] = useState(false);
  const [visitedPipelineStages, setVisitedPipelineStages] = useState<Set<StageNumber>>(
    () => new Set([currentStage])
  );

  useEffect(() => {
    setVisitedPipelineStages((prev) => new Set(prev).add(currentStage));
  }, [currentStage]);

  const getStageStatus = (stageId: number) =>
    pipelineStages.find((s) => s.stage === stageId)?.status ?? "idle";

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
                    {sidebarOpen && (
                      <span className="truncate">{stage.label}</span>
                    )}
                  </button>
                </TooltipTrigger>
                {!sidebarOpen && (
                  <TooltipContent side="right">{stage.label}</TooltipContent>
                )}
              </Tooltip>
            );
          })}

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
    </div>
  );
}
