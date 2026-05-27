"use client";

import { useCallback } from "react";
import { ArrowRight, Database, Loader2, Search } from "lucide-react";
import { create } from "zustand";

import { DisclosurePanel, ErrorBanner, ExecutionTargetToggle, RunStageWidget, toStageStatus } from "@/components/pipeline";
import { InferenceDebugPanel } from "@/components/stages/inference/InferenceDebugPanel";
import { InferenceModelCard } from "@/components/stages/inference/InferenceModelCard";
import { InferenceSourceCard, useInferenceDatasets } from "@/components/stages/inference/InferenceSourceCard";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ApiError, cancelPipeline, getDatasets, getPipelineStatus, runStage, type DatasetFolder, type FusionConfigRequest } from "@/lib/api";
import { useKaggleCredentialsStore } from "@/lib/kaggle-credentials-store";
import { flushPipelineFromStage } from "@/lib/pipeline-flush";
import type { ModelEntry } from "@/services/models";
import { useDetectionStore, usePipelineStore, useSessionStore, useStageExecutionStore, useVideoStore } from "@/store";
import type { PipelineRunStatus, RunModelMetadata, StageNumber } from "@/types";

function getRunStageErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 401) return "Kaggle credentials missing or invalid. Configure them in the sidebar settings.";
    if (error.status === 429) return "Both Kaggle GPU slots are busy. Try again later or run locally.";
    if (error.status === 400) {
      const data = error.data as { detail?: unknown; message?: unknown } | undefined;
      return String(data?.detail ?? data?.message ?? error.message);
    }
    if (error.status === 500) return "Kaggle dispatch failed. Falling back to local? Check backend logs.";
  }

  return error instanceof Error ? error.message : "Inference failed";
}

function extractRunModelMetadata(data: any, selectedModel: ModelEntry | null): RunModelMetadata {
  return {
    modelId: data?.model_id ?? data?.modelId ?? selectedModel?.id ?? null,
    resolvedConfig: data?.resolved_config ?? data?.resolvedConfig ?? selectedModel?.pipeline_config ?? null,
    appliedOverrides: Array.isArray(data?.applied_overrides)
      ? data.applied_overrides
      : Array.isArray(data?.appliedOverrides)
        ? data.appliedOverrides
        : selectedModel?.model_overrides ?? [],
    warnings: Array.isArray(data?.warnings) ? data.warnings : [],
    fusion_resolved: data?.fusion_resolved ?? data?.fusionResolved ?? null,
  };
}

interface InferenceRunState {
  activeStage: 2 | 3 | null;
  runModelMetadata: RunModelMetadata | null;
  lastRunStageResponse: PipelineRunStatus | null;
  kagglePanelRunId: string | null;
  setActiveStage: (stage: 2 | 3 | null) => void;
  setRunModelMetadata: (metadata: RunModelMetadata | null) => void;
  setLastRunStageResponse: (response: PipelineRunStatus | null) => void;
  setKagglePanelRunId: (runId: string | null) => void;
  resetRunArtifacts: () => void;
}

const useInferenceRunStore = create<InferenceRunState>((set) => ({
  activeStage: null,
  runModelMetadata: null,
  lastRunStageResponse: null,
  kagglePanelRunId: null,
  setActiveStage: (activeStage) => set({ activeStage }),
  setRunModelMetadata: (runModelMetadata) => set({ runModelMetadata }),
  setLastRunStageResponse: (lastRunStageResponse) => set({ lastRunStageResponse }),
  setKagglePanelRunId: (kagglePanelRunId) => set({ kagglePanelRunId }),
  resetRunArtifacts: () => set({ activeStage: null, runModelMetadata: null, lastRunStageResponse: null, kagglePanelRunId: null }),
}));

function inferCameraId(video: { name: string; path: string } | null): string {
  if (!video) return "S02_c008";
  const candidate = `${video.name} ${video.path}`;
  const match = candidate.match(/S\d{2}_c\d{3}/i);
  return (match?.[0] ?? "S02_c008").toUpperCase();
}

async function pollStageStatus(
  activeRunId: string,
  stage: 2 | 3,
  updateStageProgress: (stage: StageNumber, progress: any) => void
) {
  while (true) {
    const statusResponse = await getPipelineStatus(activeRunId);
    const statusData: any = statusResponse.data;
    const status = String(statusData?.status ?? "running");
    const progress = Number(statusData?.progress ?? 0);
    const message = String(statusData?.message ?? `Stage ${stage} running...`);

    updateStageProgress(stage, { status: "running", progress, message });

    if (status === "completed") {
      updateStageProgress(stage, { status: "completed", progress: 100, message: `Stage ${stage} complete` });
      return;
    }

    if (status === "error") {
      throw new Error(String(statusData?.error ?? `Stage ${stage} failed`));
    }

    await new Promise((resolve) => setTimeout(resolve, 1200));
  }
}

function buildFusionPayload(fusion: ReturnType<typeof usePipelineStore.getState>["fusion"]): FusionConfigRequest | null {
  if (!fusion || fusion.models.length < 2) return null;
  return {
    models: fusion.models.map((model) => ({ model_id: model.modelId, weight: model.weight })),
    aqe_k: fusion.aqeK,
    k1: fusion.k1,
    k2: fusion.k2,
    lambda: fusion.lambda,
    rerank: fusion.rerank,
  };
}

export function InferenceStage() {
  const selectedTrackIds = useDetectionStore((state) => state.selectedTrackIds);
  const runId = usePipelineStore((state) => state.runId);
  const stages = usePipelineStore((state) => state.stages);
  const error = usePipelineStore((state) => state.error);
  const runModelMetadata = useInferenceRunStore((state) => state.runModelMetadata);
  const activeStage = useInferenceRunStore((state) => state.activeStage);
  const kagglePanelRunId = useInferenceRunStore((state) => state.kagglePanelRunId);
  const getStageExecutionTarget = useStageExecutionStore((state) => state.getStageExecutionTarget);
  const stage2Progress = stages.find((stage) => stage.stage === 2);
  const stage3Progress = stages.find((stage) => stage.stage === 3);
  const stage2Status = toStageStatus(stage2Progress);
  const stage3Status = toStageStatus(stage3Progress);
  const errorMessage = error
    ?? (stage2Status === "error" ? stage2Progress?.message : null)
    ?? (stage3Status === "error" ? stage3Progress?.message : null);

  return (
    <div className="min-h-0 flex-1 overflow-x-hidden overflow-y-auto p-4 sm:p-6">
      <div className="mx-auto max-w-4xl space-y-4">
        <Badge variant="secondary" className="w-fit">{selectedTrackIds.size} objects selected</Badge>

        <ErrorBanner title="Inference failed" message={errorMessage} />

        {runModelMetadata?.warnings.length ? (
          <div className="rounded-md border border-yellow-600/30 bg-yellow-600/10 p-3 text-sm">
            <p className="font-medium text-yellow-800 dark:text-yellow-200">Model registry warning</p>
            <ul className="mt-2 space-y-1 text-xs text-muted-foreground">
              {runModelMetadata.warnings.map((warning) => <li key={warning}>{warning}</li>)}
            </ul>
          </div>
        ) : null}

        <InferenceModelCard />

        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
          <RunStageWidget
            stage={2}
            title="Stage 2 Features"
            target={getStageExecutionTarget(2)}
            runId={activeStage === 2 ? kagglePanelRunId ?? runId : runId}
            status={stage2Status}
            progress={stage2Progress?.progress ?? 0}
            message={stage2Progress?.message}
            isRunning={stage2Status === "running"}
            className="min-w-0"
          />
          <RunStageWidget
            stage={3}
            title="Stage 3 Index"
            target={getStageExecutionTarget(3)}
            runId={activeStage === 3 ? kagglePanelRunId ?? runId : runId}
            status={stage3Status}
            progress={stage3Progress?.progress ?? 0}
            message={stage3Progress?.message}
            isRunning={stage3Status === "running"}
            className="min-w-0"
          />
        </div>

        <DisclosurePanel title="Advanced" tier="advanced" description="Dataset source, location cascade, and date range filters.">
          <InferenceSourceCard />
        </DisclosurePanel>

        <DisclosurePanel title="Debug" tier="debug" description="Resolved config, fusion payload, and read-only pipeline parameters.">
          <InferenceDebugPanel runModelMetadata={runModelMetadata} />
        </DisclosurePanel>
      </div>
    </div>
  );
}

export function InferenceActions() {
  const selectedTrackIds = useDetectionStore((state) => state.selectedTrackIds);
  const { setCurrentStage, locationFilter, dateTimeRange } = useSessionStore();
  const currentVideo = useVideoStore((state) => state.currentVideo);
  const setCurrentVideo = useVideoStore((state) => state.setCurrentVideo);
  const {
    runId,
    setRunId,
    galleryRunId: storeGalleryRunId,
    setGalleryRunId,
    setMapCameraCoordinates,
    setIsRunning,
    updateStageProgress,
    stages,
    setError,
    modelMode,
    selectedModelId,
    selectedModelMeta,
    fusion,
  } = usePipelineStore();
  const getStageExecutionTarget = useStageExecutionStore((state) => state.getStageExecutionTarget);
  const { datasets, selectedDataset } = useInferenceDatasets();
  const setActiveStage = useInferenceRunStore((state) => state.setActiveStage);
  const setRunModelMetadata = useInferenceRunStore((state) => state.setRunModelMetadata);
  const setLastRunStageResponse = useInferenceRunStore((state) => state.setLastRunStageResponse);
  const setKagglePanelRunId = useInferenceRunStore((state) => state.setKagglePanelRunId);
  const resetRunArtifacts = useInferenceRunStore((state) => state.resetRunArtifacts);
  const activeStage = useInferenceRunStore((state) => state.activeStage);
  const stage2Progress = stages.find((stage) => stage.stage === 2);
  const stage3Progress = stages.find((stage) => stage.stage === 3);
  const stage2Status = toStageStatus(stage2Progress);
  const stage3Status = toStageStatus(stage3Progress);
  const fusionRunDisabled = modelMode === "fusion" && (!fusion || fusion.models.length < 2);
  const isRunning = stage2Status === "running" || stage3Status === "running";

  const buildStageRequest = useCallback((stage: 2 | 3) => {
    const useDataset = !selectedModelMeta && selectedDataset && selectedDataset !== "__uploaded__";
    const selectedDs = useDataset ? datasets.find((dataset) => dataset.name === selectedDataset) : null;
    const effectiveDataset = selectedModelMeta?.dataset ?? (useDataset ? selectedDataset : undefined);
    const fusionPayload = modelMode === "fusion" ? buildFusionPayload(fusion) : null;
    const executionTarget = getStageExecutionTarget(stage);
    const credentials = useKaggleCredentialsStore.getState().credentials;

    return {
      useDataset,
      selectedDs,
      effectiveDataset,
      fusionPayload,
      modelIdForRequest: fusionPayload ? null : selectedModelId,
      executionTarget,
      kaggleRequestPart: executionTarget === "kaggle"
        ? { kaggle: { target: "kaggle" as const, username: credentials?.username, key: credentials?.key } }
        : {},
    };
  }, [datasets, fusion, getStageExecutionTarget, modelMode, selectedDataset, selectedModelId, selectedModelMeta]);

  const runBackendStage = useCallback(async (stage: 2 | 3) => {
    const probeVideo = currentVideo;
    const probeRunId = runId;
    const request = buildStageRequest(stage);

    resetRunArtifacts();
    setActiveStage(stage);
    setError(null);

    if (modelMode === "fusion" && !request.fusionPayload) {
      const message = "Pick at least 2 models for fusion mode";
      setError(message);
      updateStageProgress(stage, { status: "error", progress: 100, message });
      return;
    }

    if (!probeVideo) {
      updateStageProgress(stage, { status: "error", progress: 100, message: "No probe video selected. Go back to Upload." });
      return;
    }

    if (!probeRunId) {
      updateStageProgress(stage, { status: "error", progress: 100, message: "Run Stage 1 (Detection & Tracking) on your uploaded video first." });
      return;
    }

    if (request.useDataset && request.selectedDs?.cameraCoordinates && Object.keys(request.selectedDs.cameraCoordinates).length > 0) {
      setMapCameraCoordinates(request.selectedDs.cameraCoordinates);
    } else {
      setMapCameraCoordinates(null);
    }

    if (request.useDataset && request.selectedDs?.galleryRunId) {
      setGalleryRunId(request.selectedDs.galleryRunId);
    }

    flushPipelineFromStage(4);
    setIsRunning(true);
    updateStageProgress(stage, {
      status: "running",
      progress: 0,
      message: stage === 2 ? "Extracting feature vectors from selected tracklets..." : "Building FAISS search index...",
    });

    try {
      const response = await runStage(stage, {
        runId: probeRunId,
        videoId: probeVideo.id,
        cameraId: inferCameraId(probeVideo),
        dataset: request.effectiveDataset,
        model_id: request.modelIdForRequest,
        fusion: request.fusionPayload,
        ...request.kaggleRequestPart,
        config: {
          dataset: request.effectiveDataset,
          datasetName: request.useDataset ? selectedDataset : undefined,
          filters: { location: locationFilter, dateTimeRange },
        },
      });

      const responseData = response.data ?? null;
      setLastRunStageResponse(responseData);
      setRunModelMetadata(extractRunModelMetadata(responseData as any, selectedModelMeta));
      const nextRunId = (responseData as any)?.runId ?? probeRunId;
      if (nextRunId) setRunId(nextRunId);

      if (nextRunId && responseData?.execution_target === "kaggle") {
        setKagglePanelRunId(nextRunId);
        updateStageProgress(stage, {
          status: "running",
          progress: 0,
          message: stage === 2 ? "Kaggle kernel queued for feature extraction" : "Kaggle kernel queued for indexing",
        });
        return;
      }

      if (nextRunId) await pollStageStatus(nextRunId, stage, updateStageProgress);
      setCurrentVideo(probeVideo);

      if (stage === 3 && !storeGalleryRunId && !(request.useDataset && request.selectedDs?.galleryRunId)) {
        try {
          const responseDatasets: any = await getDatasets();
          const datasetList: DatasetFolder[] = Array.isArray(responseDatasets?.data) ? responseDatasets.data : [];
          const bestDataset = request.useDataset
            ? datasetList.find((dataset) => dataset.name === selectedDataset)
            : datasetList.find((dataset) => dataset.hasGallery);
          if (bestDataset?.galleryRunId) setGalleryRunId(bestDataset.galleryRunId);
          if (bestDataset?.cameraCoordinates && Object.keys(bestDataset.cameraCoordinates).length > 0) {
            setMapCameraCoordinates(bestDataset.cameraCoordinates);
          }
        } catch {
          // Dataset refresh is best-effort after indexing.
        }
      }
    } catch (error) {
      const message = getRunStageErrorMessage(error);
      setError(message);
      updateStageProgress(stage, { status: "error", progress: 100, message });
    } finally {
      setIsRunning(false);
      setActiveStage(null);
    }
  }, [buildStageRequest, currentVideo, dateTimeRange, locationFilter, modelMode, resetRunArtifacts, runId, selectedDataset, selectedModelMeta, setActiveStage, setCurrentVideo, setError, setGalleryRunId, setIsRunning, setKagglePanelRunId, setLastRunStageResponse, setMapCameraCoordinates, setRunId, setRunModelMetadata, storeGalleryRunId, updateStageProgress]);

  const handleCancelLocal = async () => {
    if (!runId) return;
    try {
      await cancelPipeline(runId);
    } finally {
      setIsRunning(false);
      if (activeStage) updateStageProgress(activeStage, { status: "idle", progress: 0, message: `Stage ${activeStage} cancelled` });
      setActiveStage(null);
    }
  };

  return (
    <div className="flex flex-col gap-3 xl:flex-row xl:items-center">
      <div className="flex flex-wrap items-center gap-2">
        <ExecutionTargetToggle stage={2} variant="compact" />
        <ExecutionTargetToggle stage={3} variant="compact" />
      </div>
      <div className="flex flex-wrap items-center justify-end gap-2">
        {isRunning ? (
          <Button type="button" variant="outline" onClick={() => void handleCancelLocal()} aria-label="Cancel local inference run">
            Cancel
          </Button>
        ) : null}
        <RunStageWidget
          mode="button-only"
          runLabel="Run Features"
          isRunning={stage2Status === "running"}
          disabled={isRunning || fusionRunDisabled || selectedTrackIds.size === 0}
          onRun={() => void runBackendStage(2)}
          runIcon={stage2Status === "running" ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Search className="mr-2 h-4 w-4" />}
        />
        <RunStageWidget
          mode="button-only"
          runLabel="Run Index"
          isRunning={stage3Status === "running"}
          disabled={isRunning || fusionRunDisabled || !runId}
          onRun={() => void runBackendStage(3)}
          runIcon={stage3Status === "running" ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Database className="mr-2 h-4 w-4" />}
        />
        <Button type="button" onClick={() => setCurrentStage(4)} disabled={stage3Progress?.progress !== 100} aria-label="Continue to Stage 4 timeline">
          Continue to Stage 4
          <ArrowRight className="ml-2 h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}