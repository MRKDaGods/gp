"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import {
  MapPin,
  Calendar,
  Database,
  Cpu,
  ArrowRight,
  Loader2,
  CheckCircle2,
  XCircle,
  FolderOpen,
  Camera,
  Video,
  Info,
  Server,
} from "lucide-react";
import { format } from "date-fns";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { ModelPicker } from "@/components/ModelPicker";
import { FusionModelPanel } from "@/components/stages/fusion-model-panel";
import { KaggleExecutionToggle } from "@/components/stages/kaggle-execution-toggle";
import { KaggleStatusPanel } from "@/components/stages/kaggle-status-panel";
import type { ModelEntry } from "@/services/models";
import { fetchModel } from "@/services/models";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Calendar as CalendarComponent } from "@/components/ui/calendar";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  useDetectionStore,
  useSessionStore,
  usePipelineStore,
  useStageExecutionStore,
  useVideoStore,
} from "@/store";
import { ApiError, getPipelineStatus, runStage, getDatasets, type DatasetFolder, type FusionConfigRequest } from "@/lib/api";
import { useKaggleCredentialsStore } from "@/lib/kaggle-credentials-store";
import { flushPipelineFromStage } from "@/lib/pipeline-flush";
import type { PipelineRunStatus, RunModelMetadata } from "@/types";

function getRunStageErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 401) {
      return "Kaggle credentials missing or invalid. Configure them in the sidebar settings.";
    }
    if (error.status === 429) {
      return "Both Kaggle GPU slots are busy. Try again later or run locally.";
    }
    if (error.status === 400) {
      const data = error.data as { detail?: unknown; message?: unknown } | undefined;
      return String(data?.detail ?? data?.message ?? error.message);
    }
    if (error.status === 500) {
      return "Kaggle dispatch failed. Falling back to local? Check backend logs.";
    }
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

// Egypt location hierarchy data
const locationData = {
  governorates: [
    { id: "cairo", name: "Cairo", nameAr: "القاهرة" },
    { id: "giza", name: "Giza", nameAr: "الجيزة" },
    { id: "alexandria", name: "Alexandria", nameAr: "الإسكندرية" },
    { id: "aswan", name: "Aswan", nameAr: "أسوان" },
    { id: "luxor", name: "Luxor", nameAr: "الأقصر" },
  ],
  cities: {
    cairo: [
      { id: "downtown", name: "Downtown", nameAr: "وسط البلد" },
      { id: "heliopolis", name: "Heliopolis", nameAr: "مصر الجديدة" },
      { id: "maadi", name: "Maadi", nameAr: "المعادي" },
      { id: "nasr_city", name: "Nasr City", nameAr: "مدينة نصر" },
    ],
    giza: [
      { id: "dokki", name: "Dokki", nameAr: "الدقي" },
      { id: "mohandessin", name: "Mohandessin", nameAr: "المهندسين" },
      { id: "haram", name: "Haram", nameAr: "الهرم" },
    ],
    alexandria: [
      { id: "sidi_gaber", name: "Sidi Gaber", nameAr: "سيدي جابر" },
      { id: "stanley", name: "Stanley", nameAr: "ستانلي" },
    ],
  },
  zones: {
    downtown: [
      { id: "tahrir", name: "Tahrir Square", nameAr: "ميدان التحرير" },
      { id: "ramses", name: "Ramses", nameAr: "رمسيس" },
      { id: "attaba", name: "Attaba", nameAr: "العتبة" },
    ],
    heliopolis: [
      { id: "korba", name: "Korba", nameAr: "كوربة" },
      { id: "merghany", name: "Merghany", nameAr: "الميرغني" },
    ],
    maadi: [
      { id: "degla", name: "Degla", nameAr: "دجلة" },
      { id: "sarayat", name: "Sarayat", nameAr: "سرايات" },
    ],
  },
};

export function InferenceStage() {
  const { selectedTrackIds } = useDetectionStore();
  const { setCurrentStage, locationFilter, setLocationFilter, dateTimeRange, setDateTimeRange } =
    useSessionStore();
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
    setModelMode,
    selectedModelId,
    selectedModelMeta,
    setSelectedModel,
    clearSelectedModel,
    fusion,
  } = usePipelineStore();
  const { currentVideo, setCurrentVideo } = useVideoStore();
  const getStageExecutionTarget = useStageExecutionStore((state) => state.getStageExecutionTarget);

  const [isProcessing, setIsProcessing] = useState(false);
  const [processStep, setProcessStep] = useState(0);
  const [runModelMetadata, setRunModelMetadata] = useState<RunModelMetadata | null>(null);
  const [lastRunStageResponse, setLastRunStageResponse] = useState<PipelineRunStatus | null>(null);
  const [kagglePanelRunId, setKagglePanelRunId] = useState<string | null>(null);
  const progressSectionRef = useRef<HTMLDivElement>(null);

  // Dataset folder selector
  const [datasets, setDatasets] = useState<DatasetFolder[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>("__uploaded__");
  const [datasetsLoading, setDatasetsLoading] = useState(true);

  const fetchDatasets = useCallback(async () => {
    try {
      setDatasetsLoading(true);
      const resp: any = await getDatasets();
      // Backend returns { success, data: [...] }
      const data = resp?.data ?? resp;
      const arr = Array.isArray(data) ? data : [];
      console.log("[InferenceStage] datasets loaded:", arr.length, arr.map((d: any) => d.name));
      setDatasets(arr);
    } catch (err) {
      console.error("[InferenceStage] Failed to fetch datasets:", err);
      setDatasets([]);
    } finally {
      setDatasetsLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchDatasets();
  }, [fetchDatasets]);

  const fusionContextSig = fusion?.models.map((model) => `${model.modelId}:${model.weight.toFixed(4)}`).join("|") ?? "none";
  const inferenceContextSig = `${selectedDataset}:${modelMode}:${selectedModelId ?? "legacy"}:${fusionContextSig}`;
  const skipInferContextFlushRef = useRef(true);
  useEffect(() => {
    if (skipInferContextFlushRef.current) {
      skipInferContextFlushRef.current = false;
      return;
    }
    flushPipelineFromStage(4);
  }, [inferenceContextSig]);

  useEffect(() => {
    if (!isProcessing) return;
    const id = requestAnimationFrame(() => {
      progressSectionRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    });
    return () => cancelAnimationFrame(id);
  }, [isProcessing]);

  const selectedCount = selectedTrackIds.size;

  const stage2Progress = stages.find((s) => s.stage === 2);
  const stage3Progress = stages.find((s) => s.stage === 3);
  const showKaggleStatusPanel = Boolean(
    kagglePanelRunId && lastRunStageResponse?.execution_target === "kaggle"
  );
  const effectiveDatasetLabel = selectedModelMeta?.dataset ?? selectedDataset;
  const fusionModelCount = fusion?.models.length ?? 0;
  const fusionRunDisabled = modelMode === "fusion" && fusionModelCount < 2;

  const handleSingleModelSelect = useCallback(
    (modelId: string | null) => {
      if (!modelId) {
        clearSelectedModel();
        return;
      }

      void fetchModel(modelId)
        .then((model) => setSelectedModel(model.id, model))
        .catch(() => {
          clearSelectedModel();
        });
    },
    [clearSelectedModel, setSelectedModel]
  );

  const handleSingleModelChange = useCallback(
    (model: ModelEntry | null) => {
      if (model) {
        setSelectedModel(model.id, model);
      } else if (!selectedModelId) {
        clearSelectedModel();
      }
    },
    [clearSelectedModel, selectedModelId, setSelectedModel]
  );

  // Get available cities based on selected governorate
  const availableCities = locationFilter.governorate
    ? locationData.cities[locationFilter.governorate as keyof typeof locationData.cities] || []
    : [];

  // Get available zones based on selected city
  const availableZones = locationFilter.city
    ? locationData.zones[locationFilter.city as keyof typeof locationData.zones] || []
    : [];

  const inferCameraId = () => {
    if (!currentVideo) return "S02_c008";
    const candidate = `${currentVideo.name} ${currentVideo.path}`;
    const match = candidate.match(/S\d{2}_c\d{3}/i);
    return (match?.[0] ?? "S02_c008").toUpperCase();
  };

  const pollStageStatus = async (activeRunId: string, stage: 2 | 3) => {
    while (true) {
      const statusResponse = await getPipelineStatus(activeRunId);
      const statusData: any = statusResponse.data;
      const status = String(statusData?.status ?? "running");
      const progress = Number(statusData?.progress ?? 0);
      const message = String(statusData?.message ?? `Stage ${stage} running...`);

      updateStageProgress(stage, { status: "running", progress, message });

      if (status === "completed") {
        updateStageProgress(stage, {
          status: "completed",
          progress: 100,
          message: `Stage ${stage} complete`,
        });
        return;
      }

      if (status === "error") {
        throw new Error(String(statusData?.error ?? `Stage ${stage} failed`));
      }

      await new Promise((resolve) => setTimeout(resolve, 1200));
    }
  };

  const handleRunInference = async () => {
    const useDataset = !selectedModelMeta && selectedDataset && selectedDataset !== "__uploaded__";
    const selectedDs = useDataset ? datasets.find((d) => d.name === selectedDataset) : null;
    const effectiveDataset = selectedModelMeta?.dataset ?? (useDataset ? selectedDataset : undefined);
    const fusionPayload: FusionConfigRequest | null =
      modelMode === "fusion" && fusion && fusion.models.length >= 2
        ? {
            models: fusion.models.map((model) => ({
              model_id: model.modelId,
              weight: model.weight,
            })),
            aqe_k: fusion.aqeK,
            k1: fusion.k1,
            k2: fusion.k2,
            lambda: fusion.lambda,
            rerank: fusion.rerank,
          }
        : null;
    const modelIdForRequest = fusionPayload ? null : selectedModelId;
    const executionTarget = getStageExecutionTarget(2);
    const kaggleCreds = useKaggleCredentialsStore.getState().credentials;
    const kagglePayload = executionTarget === "kaggle"
      ? {
          target: "kaggle" as const,
          username: kaggleCreds?.username,
          key: kaggleCreds?.key,
        }
      : null;
    const kaggleRequestPart = kagglePayload ? { kaggle: kagglePayload } : {};
    setRunModelMetadata(null);
    setLastRunStageResponse(null);
    setKagglePanelRunId(null);
    setError(null);

    if (modelMode === "fusion" && !fusionPayload) {
      setError("Pick at least 2 models for fusion mode");
      updateStageProgress(2, {
        status: "error",
        progress: 100,
        message: "Pick at least 2 models for fusion mode",
      });
      return;
    }

    if (
      useDataset &&
      selectedDs?.cameraCoordinates &&
      Object.keys(selectedDs.cameraCoordinates).length > 0
    ) {
      setMapCameraCoordinates(selectedDs.cameraCoordinates);
    } else {
      setMapCameraCoordinates(null);
    }

    // Stage 2/3 ALWAYS runs on the PROBE (uploaded) video so we get its feature vector.
    // The dataset only contributes its galleryRunId for cross-camera matching.
    const probeVideo = currentVideo;
    const probeRunId = runId; // set by stage 1 in upload-stage

    if (!probeVideo) {
      updateStageProgress(2, {
        status: "error",
        progress: 100,
        message: "No probe video selected. Go back to Upload.",
      });
      return;
    }

    if (!probeRunId) {
      updateStageProgress(2, {
        status: "error",
        progress: 100,
        message: "Run Stage 1 (Detection & Tracking) on your uploaded video first.",
      });
      return;
    }

    flushPipelineFromStage(4);

    // If gallery is already precomputed, store its runId immediately.
    if (useDataset && selectedDs?.galleryRunId) {
      setGalleryRunId(selectedDs.galleryRunId);
    }

    const cameraId = inferCameraId(); // probe camera (from currentVideo path/name)

    setIsProcessing(true);
    setIsRunning(true);

    try {
      setProcessStep(1);
      updateStageProgress(2, { status: "running", progress: 0, message: "Extracting feature vectors from selected tracklet..." });

      const stage2Response = await runStage(2, {
        runId: probeRunId,
        videoId: probeVideo.id,
        cameraId,
        dataset: effectiveDataset,
        model_id: modelIdForRequest,
        fusion: fusionPayload,
        ...kaggleRequestPart,
        config: {
          dataset: effectiveDataset,
          datasetName: useDataset ? selectedDataset : undefined,
        },
      });
      const stage2Data = stage2Response.data ?? null;
      setLastRunStageResponse(stage2Data);
      setRunModelMetadata(extractRunModelMetadata(stage2Data as any, selectedModelMeta));
      const stage2RunId = (stage2Data as any)?.runId ?? probeRunId;
      if (stage2RunId) {
        setRunId(stage2RunId);
        if (stage2Data?.execution_target === "kaggle") {
          setKagglePanelRunId(stage2RunId);
          updateStageProgress(2, {
            status: "running",
            progress: 0,
            message: "Kaggle kernel queued for feature extraction",
          });
          return;
        }
        await pollStageStatus(stage2RunId, 2);
      }

      setProcessStep(2);
      updateStageProgress(3, { status: "running", progress: 0, message: "Building search index..." });

      const stage3Response = await runStage(3, {
        runId: stage2RunId ?? probeRunId,
        videoId: probeVideo.id,
        cameraId,
        dataset: effectiveDataset,
        model_id: modelIdForRequest,
        fusion: fusionPayload,
        ...kaggleRequestPart,
        config: {
          dataset: effectiveDataset,
          datasetName: useDataset ? selectedDataset : undefined,
        },
      });
      const stage3Data = stage3Response.data ?? null;
      setLastRunStageResponse(stage3Data);
      setRunModelMetadata(extractRunModelMetadata(stage3Data as any, selectedModelMeta));
      const stage3RunId = (stage3Data as any)?.runId ?? stage2RunId;
      if (stage3RunId) {
        setRunId(stage3RunId);
        if (stage3Data?.execution_target === "kaggle") {
          setKagglePanelRunId(stage3RunId);
          updateStageProgress(3, {
            status: "running",
            progress: 0,
            message: "Kaggle kernel queued for indexing",
          });
          return;
        }
        await pollStageStatus(stage3RunId, 3);
      }

      setCurrentVideo(probeVideo);

      // If gallery not already set, discover available datasets now.
      if (!storeGalleryRunId && !(useDataset && selectedDs?.galleryRunId)) {
        try {
          const dsResp: any = await getDatasets();
          const dsList: DatasetFolder[] = Array.isArray(dsResp?.data) ? dsResp.data : [];
          const bestDs = useDataset
            ? dsList.find((d) => d.name === selectedDataset)
            : dsList.find((d) => d.hasGallery);
          if (bestDs?.galleryRunId) {
            setGalleryRunId(bestDs.galleryRunId);
          }
          if (
            bestDs?.cameraCoordinates &&
            Object.keys(bestDs.cameraCoordinates).length > 0
          ) {
            setMapCameraCoordinates(bestDs.cameraCoordinates);
          }
        } catch (_) { /* non-fatal */ }
      }

      setCurrentStage(4);
    } catch (error) {
      const message = getRunStageErrorMessage(error);
      setError(message);
      updateStageProgress(2, {
        status: "error",
        progress: 100,
        message,
      });
      updateStageProgress(3, {
        status: "error",
        progress: 100,
        message,
      });
    } finally {
      setIsRunning(false);
      setIsProcessing(false);
    }
  };

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden">
      {/* Header */}
      <header className="flex shrink-0 flex-col gap-2 border-b px-4 py-3 sm:flex-row sm:items-center sm:justify-between sm:px-6">
        <div className="min-w-0">
          <h1 className="text-lg font-semibold">Stage 2-3: Inference</h1>
          <p className="text-sm text-muted-foreground">
            Configure location filters and run ReID feature extraction
          </p>
        </div>
        <Badge className="w-fit shrink-0" variant="secondary">
          {selectedCount} objects to process
        </Badge>
      </header>

      {/* Error banner */}
      {(stage2Progress?.status === "error" || stage3Progress?.status === "error") && (
        <div className="flex shrink-0 items-start gap-3 overflow-x-auto border-b border-destructive/30 bg-destructive/10 px-4 py-3 sm:px-6">
          <XCircle className="mt-0.5 h-5 w-5 shrink-0 text-destructive" />
          <div className="min-w-0">
            <p className="text-sm font-medium text-destructive">Inference Failed</p>
            <p className="break-words text-xs text-muted-foreground">
              {stage2Progress?.status === "error" ? stage2Progress.message : stage3Progress?.message}
            </p>
          </div>
        </div>
      )}

      {runModelMetadata?.warnings.length ? (
        <div className="flex shrink-0 items-start gap-3 overflow-x-auto border-b border-yellow-600/30 bg-yellow-600/10 px-4 py-3 sm:px-6">
          <Info className="mt-0.5 h-5 w-5 shrink-0 text-yellow-700" />
          <div className="min-w-0">
            <p className="text-sm font-medium text-yellow-800">Model registry warning</p>
            <ul className="space-y-1 text-xs text-muted-foreground">
              {runModelMetadata.warnings.map((warning) => (
                <li key={warning} className="break-words">{warning}</li>
              ))}
            </ul>
          </div>
        </div>
      ) : null}

      {/* Main content */}
      <div className="min-h-0 flex-1 overflow-x-hidden overflow-y-auto p-4 sm:p-6">
        <div className="max-w-4xl mx-auto space-y-6">
          {/* Dataset source selector */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FolderOpen className="h-5 w-5" />
                Dataset Source
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <Label>Run inference on</Label>
                {selectedModelMeta ? (
                  <div className="rounded-lg border bg-muted/50 p-3">
                    <div className="flex flex-wrap items-center gap-2 text-sm">
                      <Badge variant="secondary" className="uppercase">
                        {effectiveDatasetLabel}
                      </Badge>
                      <span className="text-muted-foreground">
                        Dataset is derived from the selected registry model.
                      </span>
                    </div>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                    <div
                      className={cn(
                        "rounded-lg border-2 p-4 cursor-pointer transition-all hover:shadow-md",
                        selectedDataset === "__uploaded__"
                          ? "border-primary bg-primary/5 shadow-md"
                          : "border-muted hover:border-primary/50"
                      )}
                      onClick={() => setSelectedDataset("__uploaded__")}
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <Video className="h-5 w-5 text-primary" />
                        <span className="font-medium text-sm">Uploaded Video</span>
                      </div>
                      <p className="text-xs text-muted-foreground truncate">
                        {currentVideo?.name ?? "No video uploaded"}
                      </p>
                    </div>

                    {datasetsLoading && (
                      <div className="rounded-lg border-2 border-dashed border-muted p-4 flex items-center justify-center">
                        <Loader2 className="h-4 w-4 animate-spin mr-2" />
                        <span className="text-sm text-muted-foreground">Loading datasets...</span>
                      </div>
                    )}

                    {datasets.map((ds) => (
                      <div
                        key={ds.name}
                        className={cn(
                          "rounded-lg border-2 p-4 cursor-pointer transition-all hover:shadow-md",
                          selectedDataset === ds.name
                            ? "border-primary bg-primary/5 shadow-md"
                            : "border-muted hover:border-primary/50"
                        )}
                        onClick={() => setSelectedDataset(ds.name)}
                      >
                        <div className="flex items-center gap-2 mb-2">
                          <FolderOpen className="h-5 w-5 text-amber-500" />
                          <span className="font-medium text-sm">{ds.name}</span>
                          {ds.alreadyProcessed && (
                            <CheckCircle2 className="h-4 w-4 text-green-500 ml-auto" />
                          )}
                        </div>
                        <p className="text-xs text-muted-foreground">
                          {ds.cameraCount} cameras, {ds.videosFound} videos
                        </p>
                        {ds.alreadyProcessed && (
                          <Badge variant="outline" className="mt-2 text-[10px] text-green-600 border-green-600">
                            Pre-processed
                          </Badge>
                        )}
                      </div>
                    ))}

                    {!datasetsLoading && datasets.length === 0 && (
                      <div className="rounded-lg border-2 border-dashed border-muted p-4 text-center col-span-2">
                        <p className="text-sm text-muted-foreground">No dataset folders found in dataset/</p>
                        <Button variant="ghost" size="sm" className="mt-1" onClick={() => void fetchDatasets()}>
                          Retry
                        </Button>
                      </div>
                    )}
                  </div>
                )}
                {!selectedModelMeta && selectedDataset && selectedDataset !== "__uploaded__" && (
                  <div className="rounded-lg border p-3 bg-muted/50">
                    {(() => {
                      const ds = datasets.find((d) => d.name === selectedDataset);
                      if (!ds) return null;
                      return (
                        <div className="space-y-2">
                          <div className="flex items-center justify-between text-sm">
                            <span className="font-medium">{ds.name}</span>
                            {ds.alreadyProcessed ? (
                              <Badge variant="outline" className="text-green-600 border-green-600">
                                <CheckCircle2 className="h-3 w-3 mr-1" />
                                Processed
                              </Badge>
                            ) : (
                              <Badge variant="secondary">Not processed</Badge>
                            )}
                          </div>
                          <div className="flex gap-4 text-xs text-muted-foreground">
                            <span className="flex items-center gap-1">
                              <Camera className="h-3 w-3" />
                              {ds.cameraCount} cameras
                            </span>
                            <span className="flex items-center gap-1">
                              <Video className="h-3 w-3" />
                              {ds.videosFound} videos
                            </span>
                          </div>
                        </div>
                      );
                    })()}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* InferenceStage execution covers both Stage 2 feature extraction and Stage 3 indexing. */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Server className="h-5 w-5" />
                Execution
              </CardTitle>
            </CardHeader>
            <CardContent>
              <KaggleExecutionToggle stage={2} />
            </CardContent>
          </Card>

          {/* Location filters */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <MapPin className="h-5 w-5" />
                Location Filter
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
                {/* Governorate */}
                <div className="space-y-2">
                  <Label>Governorate</Label>
                  <Select
                    value={locationFilter.governorate}
                    onValueChange={(value) =>
                      setLocationFilter({ governorate: value, city: undefined, zone: undefined })
                    }
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select governorate" />
                    </SelectTrigger>
                    <SelectContent>
                      {locationData.governorates.map((gov) => (
                        <SelectItem key={gov.id} value={gov.id}>
                          <span className="flex items-center gap-2">
                            {gov.name}
                            <span className="text-muted-foreground text-xs">
                              {gov.nameAr}
                            </span>
                          </span>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {/* City */}
                <div className="space-y-2">
                  <Label>City</Label>
                  <Select
                    value={locationFilter.city}
                    onValueChange={(value) =>
                      setLocationFilter({ city: value, zone: undefined })
                    }
                    disabled={!locationFilter.governorate}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select city" />
                    </SelectTrigger>
                    <SelectContent>
                      {availableCities.map((city) => (
                        <SelectItem key={city.id} value={city.id}>
                          <span className="flex items-center gap-2">
                            {city.name}
                            <span className="text-muted-foreground text-xs">
                              {city.nameAr}
                            </span>
                          </span>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {/* Zone */}
                <div className="space-y-2">
                  <Label>Zone</Label>
                  <Select
                    value={locationFilter.zone}
                    onValueChange={(value) => setLocationFilter({ zone: value })}
                    disabled={!locationFilter.city}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select zone" />
                    </SelectTrigger>
                    <SelectContent>
                      {availableZones.map((zone) => (
                        <SelectItem key={zone.id} value={zone.id}>
                          <span className="flex items-center gap-2">
                            {zone.name}
                            <span className="text-muted-foreground text-xs">
                              {zone.nameAr}
                            </span>
                          </span>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Model selector */}
          <Card>
            <CardHeader className="space-y-3">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <CardTitle className="flex items-center gap-2">
                  <Cpu className="h-5 w-5" />
                  Model Registry
                </CardTitle>
                <Tabs value={modelMode} onValueChange={(value) => setModelMode(value as "single" | "fusion")}>
                  <TabsList>
                    <TabsTrigger value="single">Single model</TabsTrigger>
                    <TabsTrigger value="fusion">Fusion (2-3 models)</TabsTrigger>
                  </TabsList>
                </Tabs>
              </div>
            </CardHeader>
            <CardContent className="p-4">
              {modelMode === "single" ? (
                <ModelPicker
                  selectedId={selectedModelId}
                  onSelect={handleSingleModelSelect}
                  onModelChange={handleSingleModelChange}
                />
              ) : (
                <FusionModelPanel />
              )}
            </CardContent>
          </Card>

          {(selectedModelMeta || runModelMetadata || (modelMode === "fusion" && fusion)) && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground uppercase tracking-wide">
                  {runModelMetadata?.fusion_resolved ? "Effective Config (Fusion)" : "Effective Config"}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-xs">
                {runModelMetadata?.fusion_resolved ? (
                  <div className="space-y-3">
                    <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                      <div>
                        <div className="text-muted-foreground">Run model_id</div>
                        <div className="font-mono">{runModelMetadata.modelId ?? "resolved by backend"}</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">Pipeline YAML</div>
                        <div className="font-mono">{runModelMetadata.resolvedConfig ?? "configs/default.yaml"}</div>
                      </div>
                    </div>
                    <div className="rounded-lg border bg-muted/40 p-3">
                      <div className="mb-2 text-sm font-medium">Fusion models</div>
                      <div className="space-y-1.5">
                        {runModelMetadata.fusion_resolved.models.map((model) => (
                          <div
                            key={model.model_id}
                            className={cn(
                              "grid grid-cols-[1fr_auto_auto] items-center gap-3 rounded-md px-2 py-1",
                              model.primary && "border border-primary/30 bg-primary/10"
                            )}
                          >
                            <span className="truncate font-mono">{model.model_id}</span>
                            {model.primary ? <Badge variant="secondary">Primary</Badge> : <span className="text-muted-foreground">Secondary</span>}
                            <span className="font-mono tabular-nums">{(model.weight * 100).toFixed(0)}%</span>
                          </div>
                        ))}
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 rounded-lg border bg-muted/40 p-3 font-mono sm:grid-cols-5">
                      <div><span className="text-muted-foreground">aqe_k </span>{runModelMetadata.fusion_resolved.aqe_k}</div>
                      <div><span className="text-muted-foreground">k1 </span>{runModelMetadata.fusion_resolved.k1}</div>
                      <div><span className="text-muted-foreground">k2 </span>{runModelMetadata.fusion_resolved.k2}</div>
                      <div><span className="text-muted-foreground">λ </span>{runModelMetadata.fusion_resolved.lambda}</div>
                      <div><span className="text-muted-foreground">rerank </span>{runModelMetadata.fusion_resolved.rerank ? "true" : "false"}</div>
                    </div>
                    <details className="rounded-lg border bg-muted/40 p-3">
                      <summary className="cursor-pointer text-sm font-medium">Applied overrides</summary>
                      <pre className="mt-3 max-h-56 overflow-auto whitespace-pre-wrap rounded bg-background p-3 font-mono text-[11px]">
                        {runModelMetadata.appliedOverrides.length > 0
                          ? runModelMetadata.appliedOverrides.join("\n")
                          : "No registry overrides applied."}
                      </pre>
                    </details>
                  </div>
                ) : modelMode === "fusion" && fusion ? (
                  <div className="space-y-3">
                    <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                      <div>
                        <div className="text-muted-foreground">Mode</div>
                        <div className="font-mono">Fusion ({fusion.models.length} models)</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">Pipeline payload</div>
                        <div className="font-mono">Sent as top-level fusion</div>
                      </div>
                    </div>
                    <div className="rounded-lg border bg-muted/40 p-3">
                      <div className="mb-2 text-sm font-medium">Fusion models</div>
                      <div className="space-y-1.5">
                        {fusion.models.length > 0 ? fusion.models.map((model) => (
                          <div key={model.modelId} className="grid grid-cols-[1fr_auto] gap-3">
                            <span className="truncate font-mono">{model.modelId}</span>
                            <span className="font-mono tabular-nums">{(model.weight * 100).toFixed(0)}%</span>
                          </div>
                        )) : (
                          <div className="text-muted-foreground">Pick at least 2 models</div>
                        )}
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 rounded-lg border bg-muted/40 p-3 font-mono sm:grid-cols-5">
                      <div><span className="text-muted-foreground">aqe_k </span>{fusion.aqeK}</div>
                      <div><span className="text-muted-foreground">k1 </span>{fusion.k1}</div>
                      <div><span className="text-muted-foreground">k2 </span>{fusion.k2}</div>
                      <div><span className="text-muted-foreground">λ </span>{fusion.lambda}</div>
                      <div><span className="text-muted-foreground">rerank </span>{fusion.rerank ? "true" : "false"}</div>
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                      <div>
                        <div className="text-muted-foreground">Model</div>
                        <div className="font-mono">{runModelMetadata?.modelId ?? selectedModelMeta?.id ?? "legacy"}</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">Pipeline YAML</div>
                        <div className="font-mono">{runModelMetadata?.resolvedConfig ?? selectedModelMeta?.pipeline_config ?? "configs/default.yaml"}</div>
                      </div>
                    </div>
                    <details className="rounded-lg border bg-muted/40 p-3">
                      <summary className="cursor-pointer text-sm font-medium">Applied overrides</summary>
                      <pre className="mt-3 max-h-56 overflow-auto whitespace-pre-wrap rounded bg-background p-3 font-mono text-[11px]">
                        {(runModelMetadata?.appliedOverrides ?? selectedModelMeta?.model_overrides ?? []).length > 0
                          ? (runModelMetadata?.appliedOverrides ?? selectedModelMeta?.model_overrides ?? []).join("\n")
                          : "No registry overrides applied."}
                      </pre>
                    </details>
                  </>
                )}
              </CardContent>
            </Card>
          )}

          {/* Active Pipeline Parameters (read-only, notebook-aligned) */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground uppercase tracking-wide">
                Active Pipeline Parameters
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-xs">
              <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
                {modelMode === "fusion" && fusion ? (
                  <>
                    <div className="text-muted-foreground">Mode</div>
                    <div className="font-mono">Fusion ({fusion.models.length} models)</div>
                    {fusion.models.map((model) => (
                      <div key={`active-${model.modelId}`} className="contents">
                        <div className="truncate text-muted-foreground">{model.modelId}</div>
                        <div className="font-mono tabular-nums">weight {(model.weight * 100).toFixed(0)}%</div>
                      </div>
                    ))}
                    <div className="text-muted-foreground">Fusion hyperparams</div>
                    <div className="font-mono">aqe_k={fusion.aqeK} · k1={fusion.k1} · k2={fusion.k2} · λ={fusion.lambda} · rerank={fusion.rerank ? "true" : "false"}</div>
                  </>
                ) : null}
                <div className="text-muted-foreground">Detector</div>
                <div className="font-mono">YOLOv26 · conf 0.25 · IoU 0.65</div>
                <div className="text-muted-foreground">Tracker</div>
                <div className="font-mono">DeepOCSort · max_age 30</div>
                <div className="text-muted-foreground">ReID backbone</div>
                <div className="font-mono">TransReID ViT-Base · 768D → 280D PCA</div>
                <div className="text-muted-foreground">Samples / tracklet</div>
                <div className="font-mono">32 · flip_augment ✓ · cam_BN ✓</div>
                <div className="text-muted-foreground">Quality filter</div>
                <div className="font-mono">Laplacian var ≥ 15 · temp 3.0</div>
                <div className="text-muted-foreground">Matching</div>
                <div className="font-mono">FAISS IndexFlatIP · threshold 0.60</div>
                <div className="text-muted-foreground">Solver</div>
                <div className="font-mono">conflict_free_cc · AQE k=5 α=5.0</div>
                <div className="text-muted-foreground">FIC whitening</div>
                <div className="font-mono">reg 0.3 · gallery_expansion rounds=2</div>
              </div>
            </CardContent>
          </Card>

          {/* Date/Time range */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Calendar className="h-5 w-5" />
                Date & Time Range
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                {/* Start date */}
                <div className="space-y-2">
                  <Label>Start Date</Label>
                  <Popover>
                    <PopoverTrigger asChild>
                      <Button
                        variant="outline"
                        className="w-full justify-start text-left font-normal"
                      >
                        <Calendar className="mr-2 h-4 w-4" />
                        {dateTimeRange.start
                          ? format(dateTimeRange.start, "PPP")
                          : "Select start date"}
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-auto p-0">
                      <CalendarComponent
                        mode="single"
                        selected={dateTimeRange.start}
                        onSelect={(date) => setDateTimeRange({ start: date })}
                        initialFocus
                      />
                    </PopoverContent>
                  </Popover>
                </div>

                {/* End date */}
                <div className="space-y-2">
                  <Label>End Date</Label>
                  <Popover>
                    <PopoverTrigger asChild>
                      <Button
                        variant="outline"
                        className="w-full justify-start text-left font-normal"
                      >
                        <Calendar className="mr-2 h-4 w-4" />
                        {dateTimeRange.end
                          ? format(dateTimeRange.end, "PPP")
                          : "Select end date"}
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-auto p-0">
                      <CalendarComponent
                        mode="single"
                        selected={dateTimeRange.end}
                        onSelect={(date) => setDateTimeRange({ end: date })}
                        initialFocus
                      />
                    </PopoverContent>
                  </Popover>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Run button — progress is below so it stays in view after click */}
          <div className="flex justify-center">
            <Button
              size="lg"
              onClick={handleRunInference}
              disabled={isProcessing || fusionRunDisabled}
              className="min-w-[200px]"
            >
              {isProcessing ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <Database className="mr-2 h-4 w-4" />
                  Run Inference
                  <ArrowRight className="ml-2 h-4 w-4" />
                </>
              )}
            </Button>
          </div>

          {/* Processing status (below run — scrollIntoView when started) */}
          {(isProcessing || showKaggleStatusPanel) && (
            <div ref={progressSectionRef} className="scroll-mt-4">
              {showKaggleStatusPanel && kagglePanelRunId ? (
                <KaggleStatusPanel runId={kagglePanelRunId} />
              ) : (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Cpu className="h-5 w-5 animate-pulse" />
                      Processing
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <ProcessingStep
                      step={1}
                      currentStep={processStep}
                      title="Stage 2: Feature Extraction"
                      description="Extracting ReID embeddings using TransReID"
                      progress={stage2Progress?.progress || 0}
                      message={stage2Progress?.message}
                    />
                    <ProcessingStep
                      step={2}
                      currentStep={processStep}
                      title="Stage 3: Indexing"
                      description="Building FAISS vector index"
                      progress={stage3Progress?.progress || 0}
                      message={stage3Progress?.message}
                    />
                  </CardContent>
                </Card>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function ProcessingStep({
  step,
  currentStep,
  title,
  description,
  progress,
  message,
}: {
  step: number;
  currentStep: number;
  title: string;
  description: string;
  progress: number;
  message?: string;
}) {
  const isActive = currentStep === step;
  const isComplete = currentStep > step;
  const isPending = currentStep < step;

  return (
    <div
      className={cn(
        "p-4 rounded-lg border transition-colors",
        isActive && "border-primary bg-primary/5",
        isComplete && "border-green-500 bg-green-500/5",
        isPending && "opacity-50"
      )}
    >
      <div className="flex items-center gap-3 mb-2">
        {isComplete ? (
          <CheckCircle2 className="h-5 w-5 text-green-500" />
        ) : isActive ? (
          <Loader2 className="h-5 w-5 animate-spin text-primary" />
        ) : (
          <div className="h-5 w-5 rounded-full border-2" />
        )}
        <div>
          <p className="font-medium">{title}</p>
          <p className="text-sm text-muted-foreground">{description}</p>
        </div>
      </div>
      {isActive && (
        <div className="ml-8 space-y-2">
          <Progress value={progress} className="h-2" />
          {message && (
            <p className="text-xs text-muted-foreground">{message}</p>
          )}
        </div>
      )}
    </div>
  );
}
