"use client";

import { useState } from "react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Camera,
  Upload,
  Box,
  Scan,
  Database,
  GitBranch,
  BarChart3,
  Film,
  Map,
  Settings,
  HelpCircle,
  ChevronLeft,
  ChevronRight,
  Home,
  Loader2,
  FolderOpen,
} from "lucide-react";
import { useSessionStore, useUIStore, usePipelineStore } from "@/store";
import type { StageNumber } from "@/types";
import { Logo, LogoIcon } from "@/components/logo";
import { GlobalProcessingBanner } from "@/components/layout/global-processing-banner";

// Stage components
import { UploadStage } from "@/components/stages/upload-stage";
import { DetectionStage } from "@/components/stages/detection-stage";
import { SelectionStage } from "@/components/stages/selection-stage";
import { InferenceStage } from "@/components/stages/inference-stage";
import { TimelineStage } from "@/components/stages/timeline-stage";
import { RefinementStage } from "@/components/stages/refinement-stage";
import { OutputStage } from "@/components/stages/output-stage";
import { DatasetProcessing } from "@/components/stages/dataset-processing";

const stages = [
  { id: 0 as StageNumber, label: "Upload", icon: Upload, shortLabel: "Upload" },
  { id: 1 as StageNumber, label: "Detection", icon: Scan, shortLabel: "Detect" },
  { id: 2 as StageNumber, label: "Selection", icon: Box, shortLabel: "Select" },
  { id: 3 as StageNumber, label: "Inference", icon: Database, shortLabel: "ReID" },
  { id: 4 as StageNumber, label: "Timeline", icon: GitBranch, shortLabel: "Track" },
  { id: 5 as StageNumber, label: "Refinement", icon: BarChart3, shortLabel: "Refine" },
  { id: 6 as StageNumber, label: "Output", icon: Film, shortLabel: "Output" },
];

export function MainDashboard() {
  const { currentStage, setCurrentStage } = useSessionStore();
  const { sidebarOpen, toggleSidebar } = useUIStore();
  const pipelineStages = usePipelineStore((s) => s.stages);
  const [datasetView, setDatasetView] = useState(false);

  const stageIsRunning = (stageId: number) =>
    pipelineStages.find((s) => s.stage === stageId)?.status === "running";

  return (
    <div className="flex h-dvh max-h-dvh min-h-0 overflow-hidden bg-background">
      {/* Sidebar */}
      <aside
        className={cn(
          "flex min-h-0 min-w-0 flex-shrink-0 flex-col overflow-x-hidden border-r bg-card transition-all duration-300",
          sidebarOpen ? "w-64" : "w-16"
        )}
      >
        {/* Logo */}
        <div
          className={cn(
            "flex h-14 shrink-0 items-center border-b",
            sidebarOpen ? "px-4" : "justify-center px-0"
          )}
        >
          {sidebarOpen ? (
            <Logo size="sm" />
          ) : (
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary">
              <LogoIcon size={20} className="text-primary-foreground" />
            </div>
          )}
        </div>

        {/* Navigation */}
        <ScrollArea className="min-h-0 flex-1 px-2 py-4">
          <nav className="flex flex-col gap-1">
            {stages.map((stage) => {
              const Icon = stage.icon;
              const isActive = !datasetView && currentStage === stage.id;
              const isPast = currentStage > stage.id;
              const running = stageIsRunning(stage.id);

              return (
                <Tooltip key={stage.id} delayDuration={0}>
                  <TooltipTrigger asChild>
                    <Button
                      variant={isActive ? "secondary" : "ghost"}
                      className={cn(
                        "justify-start gap-3",
                        !sidebarOpen && "justify-center px-2",
                        isPast && "text-muted-foreground"
                      )}
                      onClick={() => {
                        setDatasetView(false);
                        setCurrentStage(stage.id);
                      }}
                    >
                      <div
                        className={cn(
                          "flex h-6 w-6 items-center justify-center rounded-full text-xs font-medium",
                          isActive && "bg-primary text-primary-foreground",
                          isPast && "bg-green-600 text-white",
                          !isActive && !isPast && "bg-muted text-muted-foreground"
                        )}
                      >
                        {running ? (
                          <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        ) : isPast ? (
                          "✓"
                        ) : (
                          stage.id
                        )}
                      </div>
                      {sidebarOpen && <span>{stage.label}</span>}
                    </Button>
                  </TooltipTrigger>
                  {!sidebarOpen && (
                    <TooltipContent side="right">
                      {stage.label}
                    </TooltipContent>
                  )}
                </Tooltip>
              );
            })}
          </nav>

          <Separator className="my-4" />

          {/* Dataset processing */}
          <nav className="flex flex-col gap-1">
            <Tooltip delayDuration={0}>
              <TooltipTrigger asChild>
                <Button
                  variant={datasetView ? "secondary" : "ghost"}
                  className={cn(
                    "justify-start gap-3",
                    !sidebarOpen && "justify-center px-2"
                  )}
                  onClick={() => setDatasetView(true)}
                >
                  <FolderOpen className="h-4 w-4" />
                  {sidebarOpen && <span>Dataset</span>}
                </Button>
              </TooltipTrigger>
              {!sidebarOpen && (
                <TooltipContent side="right">
                  Dataset Processing
                </TooltipContent>
              )}
            </Tooltip>
          </nav>

          <Separator className="my-4" />

          {/* Future: Map view */}
          <nav className="flex flex-col gap-1">
            <Tooltip delayDuration={0}>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  className={cn(
                    "justify-start gap-3 opacity-50",
                    !sidebarOpen && "justify-center px-2"
                  )}
                  disabled
                >
                  <Map className="h-4 w-4" />
                  {sidebarOpen && <span>Map View (Coming Soon)</span>}
                </Button>
              </TooltipTrigger>
              {!sidebarOpen && (
                <TooltipContent side="right">
                  Map View (Coming Soon)
                </TooltipContent>
              )}
            </Tooltip>
          </nav>
        </ScrollArea>

        {/* Footer — when collapsed (w-16), row layout overflows; stack vertically so expand stays reachable */}
        <div className={cn("shrink-0 border-t", sidebarOpen ? "p-2" : "px-1 py-2")}>
          <div
            className={cn(
              "flex gap-1",
              sidebarOpen
                ? "flex-row items-center justify-between"
                : "flex-col items-center justify-center gap-0.5"
            )}
          >
            <div className={cn("flex gap-1", !sidebarOpen && "flex-col items-center")}>
              <Tooltip delayDuration={0}>
                <TooltipTrigger asChild>
                  <Button variant="ghost" size="icon-sm">
                    <Settings className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Settings</TooltipContent>
              </Tooltip>
              <Tooltip delayDuration={0}>
                <TooltipTrigger asChild>
                  <Button variant="ghost" size="icon-sm">
                    <HelpCircle className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Help</TooltipContent>
              </Tooltip>
            </div>
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={toggleSidebar}
              className={cn(!sidebarOpen && "shrink-0")}
              aria-label={sidebarOpen ? "Collapse sidebar" : "Expand sidebar"}
            >
              {sidebarOpen ? (
                <ChevronLeft className="h-4 w-4" />
              ) : (
                <ChevronRight className="h-4 w-4" />
              )}
            </Button>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
        <GlobalProcessingBanner />
        <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
          {datasetView ? (
            <DatasetProcessing />
          ) : (
            <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
              <StageContent stage={currentStage} />
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

function StageContent({ stage }: { stage: StageNumber }) {
  switch (stage) {
    case 0:
      return <UploadStage />;
    case 1:
      return <DetectionStage />;
    case 2:
      return <SelectionStage />;
    case 3:
      return <InferenceStage />;
    case 4:
      return <TimelineStage />;
    case 5:
      return <RefinementStage />;
    case 6:
      return <OutputStage />;
    default:
      return <UploadStage />;
  }
}
