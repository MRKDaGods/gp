"use client";

import { useCallback, useState } from "react";
import { ArrowRight, Loader2 } from "lucide-react";
import { create } from "zustand";

import { DisclosurePanel, ErrorBanner, ExecutionTargetToggle, toStageStatus } from "@/components/pipeline";
import { InferenceDebugPanel } from "@/components/stages/inference/InferenceDebugPanel";
import { InferenceModelCard } from "@/components/stages/inference/InferenceModelCard";
import { InferenceSearchTargetCard } from "@/components/stages/inference/InferenceSearchTargetCard";
import { InferenceSourceCard, UPLOADED_DATASET, useEnsureGalleryReady, useInferenceDatasets, useInferenceSourceStore } from "@/components/stages/inference/InferenceSourceCard";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Progress } from "@/components/ui/progress";
import { ApiError, cancelPipeline, getDatasets, runStage, type DatasetFolder, type FusionConfigRequest } from "@/lib/api";
import { useKaggleCredentialsStore } from "@/lib/kaggle-credentials-store";
import { flushPipelineFromStage } from "@/lib/pipeline-flush";
import { getRunStageErrorMessage, inferCameraId, pollStageStatus } from "@/lib/pipeline-run";
import { useRunPipelineStage } from "@/hooks/use-pipeline-stage";
import type { ModelEntry } from "@/services/models";
import { useDetectionStore, usePipelineStore, useSessionStore, useStageExecutionStore, useVideoStore } from "@/store";
import type { RunModelMetadata, SingleStageRunStatus, StageNumber } from "@/types";

function extractRunModelMetadata(data: SingleStageRunStatus | null, selectedModel: ModelEntry | null): RunModelMetadata {
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
  lastRunStageResponse: SingleStageRunStatus | null;
  kagglePanelRunId: string | null;
  setActiveStage: (stage: 2 | 3 | null) => void;
  setRunModelMetadata: (metadata: RunModelMetadata | null) => void;
  setLastRunStageResponse: (response: SingleStageRunStatus | null) => void;
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

/** Compact progress for the probe's feature-extraction / indexing steps. Only
 * shows while running or once complete - the dataset precompute has its own
 * progress inside the "Search within" card. Replaces the two large per-stage
 * widgets so the page shows one clear status line instead of raw pipeline stages. */
function InferenceProgressStrip() {
  const stages = usePipelineStore((state) => state.stages);
  const stage2 = stages.find((s) => s.stage === 2);
  const stage3 = stages.find((s) => s.stage === 3);
  const stage2Status = toStageStatus(stage2);
  const stage3Status = toStageStatus(stage3);

  const active =
    stage3Status === "running" ? { label: "Building search index", progress: stage3?.progress ?? 0 }
    : stage2Status === "running" ? { label: "Extracting features from your selection", progress: stage2?.progress ?? 0 }
    : null;

  if (active) {
    return (
      <div className="space-y-1 rounded-md border border-primary/30 bg-primary/5 p-3">
        <div className="flex items-center justify-between text-sm">
          <span className="flex items-center gap-2 text-foreground">
            <Loader2 className="h-4 w-4 animate-spin text-primary" />
            {active.label}...
          </span>
          <span className="font-mono text-xs">{Math.round(active.progress)}%</span>
        </div>
        <Progress value={active.progress} className="h-2" />
      </div>
    );
  }

  if (stage3Status === "done") {
    return (
      <div className="rounded-md border border-success/30 bg-success/10 p-3 text-sm text-success">
        Search index ready - continue to the Timeline to see matches.
      </div>
    );
  }

  return null;
}

export function InferenceStage() {
  const selectedTrackIds = useDetectionStore((state) => state.selectedTrackIds);
  const stages = usePipelineStore((state) => state.stages);
  const error = usePipelineStore((state) => state.error);
  const runModelMetadata = useInferenceRunStore((state) => state.runModelMetadata);
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

        <ErrorBanner
          severity="warning"
          title="Model registry warning"
          message={runModelMetadata?.warnings.length ? runModelMetadata.warnings.join("\n") : null}
        />

        <InferenceModelCard />

        <InferenceSearchTargetCard />

        <InferenceProgressStrip />

        <DisclosurePanel title="Advanced" tier="advanced" description="Compute target, location cascade, and date range filters.">
          <div className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2 rounded-md border p-3">
              <span className="text-sm font-medium">Compute on</span>
              <ExecutionTargetToggle stage={2} stages={[2, 3]} variant="compact" />
            </div>
            <InferenceSourceCard />
          </div>
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
  const runInput = usePipelineStore((state) => state.runInput);
  const runDatasetStage = useRunPipelineStage();
  const { datasets, selectedDataset } = useInferenceDatasets();
  const ensureGalleryReady = useEnsureGalleryReady();
  const setActiveStage = useInferenceRunStore((state) => state.setActiveStage);
  const setRunModelMetadata = useInferenceRunStore((state) => state.setRunModelMetadata);
  const setLastRunStageResponse = useInferenceRunStore((state) => state.setLastRunStageResponse);
  const setKagglePanelRunId = useInferenceRunStore((state) => state.setKagglePanelRunId);
  const resetRunArtifacts = useInferenceRunStore((state) => state.resetRunArtifacts);
  const [showComputeDialog, setShowComputeDialog] = useState(false);
  const stage2Progress = stages.find((stage) => stage.stage === 2);
  const stage3Progress = stages.find((stage) => stage.stage === 3);
  const stage2Status = toStageStatus(stage2Progress);
  const stage3Status = toStageStatus(stage3Progress);
  const fusionRunDisabled = modelMode === "fusion" && (!fusion || fusion.models.length < 2);
  // A dataset auto-processing inline counts as "busy" too, so the run buttons
  // can't be re-triggered mid-precompute.
  const processingDataset = useInferenceSourceStore((state) => state.processingDataset);
  const isRunning = stage2Status === "running" || stage3Status === "running" || Boolean(processingDataset);

  // What must be true before the single "Continue" action can run the pipeline.
  // Dataset flow (reopened run): detection (stage 1) must be done. Probe flow:
  // at least one track must be selected. Fusion mode needs >= 2 models.
  const datasetFlow = Boolean(runInput);
  const detectionDone = toStageStatus(stages.find((s) => s.stage === 1)) === "done";
  const preconditionsUnmet = fusionRunDisabled || (datasetFlow ? !detectionDone : selectedTrackIds.size === 0);
  // Index already built -> "Continue" just navigates; otherwise it runs the pipeline.
  const inferenceDone = stage3Status === "done" || (stage3Progress?.progress ?? 0) >= 100;

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

  // Run one backend stage (2=features, 3=index). Returns "completed" when it
  // finished locally, "queued" when dispatched to Kaggle (async - don't chain),
  // or "error". The single "Continue" button chains 2 -> 3 using these results.
  const runBackendStage = useCallback(async (stage: 2 | 3): Promise<"completed" | "queued" | "error"> => {
    // Per-stage dataset flow: a run was created at ingestion (Stage 0). Run this
    // pipeline stage incrementally against the same run - nothing cascades and
    if (runInput) {
      setError(null);
      resetRunArtifacts();
      flushPipelineFromStage(4);
      const result = await runDatasetStage({
        pipelineStage: stage,
        uiStage: stage as StageNumber,
        label: stage === 2 ? "feature extraction" : "indexing",
      });
      if (result === "completed" && stage === 3) {
        // Best-effort: surface the dataset gallery + camera coords for Timeline/map.
        try {
          const responseDatasets: any = await getDatasets();
          const datasetList: DatasetFolder[] = Array.isArray(responseDatasets?.data) ? responseDatasets.data : [];
          const best = datasetList.find((d) => d.hasGallery);
          if (best?.galleryRunId) setGalleryRunId(best.galleryRunId);
          if (best?.cameraCoordinates && Object.keys(best.cameraCoordinates).length > 0) {
            setMapCameraCoordinates(best.cameraCoordinates);
          }
        } catch {
          // best-effort
        }
      }
      return result === "completed" ? "completed" : "error";
    }

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
      return "error";
    }

    if (!probeVideo) {
      updateStageProgress(stage, { status: "error", progress: 100, message: "No probe video selected. Go back to Upload." });
      return "error";
    }

    if (!probeRunId) {
      updateStageProgress(stage, { status: "error", progress: 100, message: "Run Stage 1 (Detection & Tracking) on your uploaded video first." });
      return "error";
    }

    // Auto-process the "search within" dataset if it isn't precomputed yet, then
    // search against its gallery. No-op for the uploaded-probe target. This is
    // what lets the user pick a fresh dataset at Inference and have it processed
    // inline before the search runs.
    if (request.useDataset && selectedDataset !== UPLOADED_DATASET) {
      try {
        const gallery = await ensureGalleryReady(selectedDataset);
        if (gallery.galleryRunId) setGalleryRunId(gallery.galleryRunId);
        if (gallery.cameraCoordinates && Object.keys(gallery.cameraCoordinates).length > 0) {
          setMapCameraCoordinates(gallery.cameraCoordinates);
        } else {
          setMapCameraCoordinates(null);
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : "Dataset processing failed";
        setError(message);
        updateStageProgress(stage, { status: "error", progress: 100, message });
        return "error";
      }
    } else {
      setMapCameraCoordinates(null);
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
      setRunModelMetadata(extractRunModelMetadata(responseData, selectedModelMeta));
      const nextRunId = responseData?.runId ?? probeRunId;
      if (nextRunId) setRunId(nextRunId);

      if (nextRunId && responseData?.execution_target === "kaggle") {
        setKagglePanelRunId(nextRunId);
        updateStageProgress(stage, {
          status: "running",
          progress: 0,
          message: stage === 2 ? "Kaggle kernel queued for feature extraction" : "Kaggle kernel queued for indexing",
        });
        return "queued";
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
      return "completed";
    } catch (error) {
      if (request.executionTarget === "kaggle" && error instanceof ApiError && error.status === 401) {
        useKaggleCredentialsStore.getState().openCredentialsModal();
      }
      const message = getRunStageErrorMessage(error);
      setError(message);
      updateStageProgress(stage, { status: "error", progress: 100, message });
      return "error";
    } finally {
      setIsRunning(false);
      setActiveStage(null);
    }
  }, [buildStageRequest, currentVideo, dateTimeRange, ensureGalleryReady, locationFilter, modelMode, resetRunArtifacts, runId, runInput, runDatasetStage, selectedDataset, selectedModelMeta, setActiveStage, setCurrentVideo, setError, setGalleryRunId, setIsRunning, setKagglePanelRunId, setLastRunStageResponse, setMapCameraCoordinates, setRunId, setRunModelMetadata, storeGalleryRunId, updateStageProgress]);

  // The single primary action: run features -> index on the probe (auto-processing
  // the chosen dataset first if needed), then advance to the Timeline (Stage 4).
  const runInferenceAndContinue = useCallback(async () => {
    const res2 = await runBackendStage(2);
    if (res2 === "error") return;
    if (res2 === "queued") return; // Kaggle runs async - the panel tracks it; don't chain locally.
    const res3 = await runBackendStage(3);
    if (res3 === "completed") setCurrentStage(4);
  }, [runBackendStage, setCurrentStage]);

  // Cancel whichever stage is currently running. The run is incremental against
  // one run_id, so cancelling terminates the active subprocess; the poll loop
  // then settles the UI state.
  const handleCancel = async () => {
    if (!runId) return;
    const stage = stage3Status === "running" ? 3 : 2;
    try {
      await cancelPipeline(runId);
    } finally {
      setIsRunning(false);
      updateStageProgress(stage, { status: "idle", progress: 0, message: `Stage ${stage} cancelled` });
      setActiveStage(null);
    }
  };

  // Does the chosen "search within" dataset still need to be precomputed?
  const selectedDs = !selectedModelMeta && selectedDataset !== UPLOADED_DATASET
    ? datasets.find((d) => d.name === selectedDataset)
    : undefined;
  const datasetNeedsCompute = Boolean(selectedDs && !selectedDs.hasGallery);

  const handleContinue = () => {
    // Already indexed - nothing to run, just advance.
    if (inferenceDone && !isRunning) {
      setCurrentStage(4);
      return;
    }
    // Fresh dataset -> confirm the (potentially long) precompute first.
    if (datasetNeedsCompute) {
      setShowComputeDialog(true);
      return;
    }
    void runInferenceAndContinue();
  };

  const continueDisabled = isRunning || (!inferenceDone && preconditionsUnmet);
  const continueLabel = isRunning
    ? "Working..."
    : inferenceDone
      ? "Continue to Stage 4"
      : "Run & Continue";

  return (
    <>
      <div className="flex flex-wrap items-center justify-end gap-2">
        {isRunning ? (
          <Button type="button" variant="destructive" onClick={() => void handleCancel()} aria-label="Cancel inference run">
            Cancel
          </Button>
        ) : null}
        <Button
          type="button"
          onClick={handleContinue}
          disabled={continueDisabled}
          title={
            continueDisabled && !isRunning
              ? (datasetFlow ? "Run Detection (Stage 1) first" : "Select at least one object in Selection first")
              : undefined
          }
          aria-label="Run inference and continue to Stage 4 timeline"
        >
          {isRunning ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
          {continueLabel}
          {!isRunning ? <ArrowRight className="ml-2 h-4 w-4" /> : null}
        </Button>
      </div>

      <Dialog open={showComputeDialog} onOpenChange={setShowComputeDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Process this dataset first?</DialogTitle>
            <DialogDescription>
              <span className="font-medium text-foreground">{selectedDataset}</span> hasn&apos;t been
              precomputed yet. It needs to be processed before your probe can be searched against it -
              this can take a while for a large dataset. It only has to be done once; after that the
              search runs instantly.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => setShowComputeDialog(false)}>
              Cancel
            </Button>
            <Button
              type="button"
              onClick={() => {
                setShowComputeDialog(false);
                void runInferenceAndContinue();
              }}
            >
              Process &amp; continue
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}