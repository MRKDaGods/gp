"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import {
  MapPin,
  Calendar,
  Clock,
  Database,
  Cpu,
  ArrowRight,
  ChevronDown,
  Loader2,
  CheckCircle2,
  XCircle,
  FolderOpen,
  Camera,
  Video,
} from "lucide-react";
import { format } from "date-fns";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
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
  useVideoStore,
} from "@/store";
import { getPipelineStatus, runStage, getDatasets, type DatasetFolder } from "@/lib/api";

const REID_MODELS = [
  {
    id: "transreid_cityflowv2_best",
    label: "TransReID — CityFlowV2 Fine-tuned (Best)",
    path: "models/reid/transreid_cityflowv2_best.pth",
    badge: "Recommended",
  },
  {
    id: "vehicle_transreid_vit_base_veri776",
    label: "TransReID — VeRi-776",
    path: "models/reid/vehicle_transreid_vit_base_veri776.pth",
    badge: null,
  },
  {
    id: "vehicle_osnet_veri776",
    label: "OSNet — VeRi-776 (Lightweight)",
    path: "models/reid/vehicle_osnet_veri776.pth",
    badge: "Fast",
  },
  {
    id: "vehicle_resnet50ibn_veri776",
    label: "ResNet-50-IBN — VeRi-776",
    path: "models/reid/vehicle_resnet50ibn_veri776.pth",
    badge: null,
  },
];

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
  const { selectedTrackIds, detections } = useDetectionStore();
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
  } = usePipelineStore();
  const { currentVideo, videos, setCurrentVideo } = useVideoStore();

  const [isProcessing, setIsProcessing] = useState(false);
  const [processStep, setProcessStep] = useState(0);
  const [selectedModel, setSelectedModel] = useState("transreid_cityflowv2_best");
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
  const selectedDetections = detections.filter(
    (d) => d.trackId != null && selectedTrackIds.has(d.trackId)
  );

  const stage2Progress = stages.find((s) => s.stage === 2);
  const stage3Progress = stages.find((s) => s.stage === 3);

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

  const inferCameraIdFromVideo = (video?: { name?: string; path?: string } | null, fallback = "S02_c008") => {
    if (!video) return fallback;
    const candidate = `${video.name ?? ""} ${video.path ?? ""}`;
    const match = candidate.match(/S\d{2}_c\d{3}/i);
    return (match?.[0] ?? fallback).toUpperCase();
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
    const useDataset = selectedDataset && selectedDataset !== "__uploaded__";
    const selectedDs = useDataset ? datasets.find((d) => d.name === selectedDataset) : null;

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

      const modelPath = REID_MODELS.find((m) => m.id === selectedModel)?.path
        ?? "models/reid/transreid_cityflowv2_best.pth";

      const stage2Response = await runStage(2, {
        runId: probeRunId,
        videoId: probeVideo.id,
        cameraId,
        config: { reid_model_path: modelPath },
      });
      const stage2RunId = (stage2Response.data as any)?.runId ?? probeRunId;
      if (stage2RunId) {
        setRunId(stage2RunId);
        await pollStageStatus(stage2RunId, 2);
      }

      setProcessStep(2);
      updateStageProgress(3, { status: "running", progress: 0, message: "Building search index..." });

      const stage3Response = await runStage(3, {
        runId: stage2RunId ?? probeRunId,
        videoId: probeVideo.id,
        cameraId,
      });
      const stage3RunId = (stage3Response.data as any)?.runId ?? stage2RunId;
      if (stage3RunId) {
        setRunId(stage3RunId);
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
      const message = error instanceof Error ? error.message : "Inference failed";
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
                {/* Dataset cards - much more visible than dropdown */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                  {/* Uploaded video card */}
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

                  {/* Loading state */}
                  {datasetsLoading && (
                    <div className="rounded-lg border-2 border-dashed border-muted p-4 flex items-center justify-center">
                      <Loader2 className="h-4 w-4 animate-spin mr-2" />
                      <span className="text-sm text-muted-foreground">Loading datasets...</span>
                    </div>
                  )}

                  {/* Dataset folder cards */}
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

                  {/* Empty state */}
                  {!datasetsLoading && datasets.length === 0 && (
                    <div className="rounded-lg border-2 border-dashed border-muted p-4 text-center col-span-2">
                      <p className="text-sm text-muted-foreground">No dataset folders found in dataset/</p>
                      <Button variant="ghost" size="sm" className="mt-1" onClick={() => void fetchDatasets()}>
                        Retry
                      </Button>
                    </div>
                  )}
                </div>
                {selectedDataset && selectedDataset !== "__uploaded__" && (
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
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Cpu className="h-5 w-5" />
                ReID Model
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <Label>Vehicle Re-Identification Model</Label>
                <Select value={selectedModel} onValueChange={setSelectedModel}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select model" />
                  </SelectTrigger>
                  <SelectContent>
                    {REID_MODELS.map((m) => (
                      <SelectItem key={m.id} value={m.id}>
                        <span className="flex items-center gap-2">
                          {m.label}
                          {m.badge && (
                            <span className="ml-1 rounded bg-primary/10 px-1.5 py-0.5 text-xs font-medium text-primary">
                              {m.badge}
                            </span>
                          )}
                        </span>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  Selected: <code className="font-mono">{REID_MODELS.find(m => m.id === selectedModel)?.path}</code>
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Active Pipeline Parameters (read-only, notebook-aligned) */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground uppercase tracking-wide">
                Active Pipeline Parameters
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-xs">
              <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
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
              disabled={isProcessing}
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
          {isProcessing && (
            <div ref={progressSectionRef} className="scroll-mt-4">
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
