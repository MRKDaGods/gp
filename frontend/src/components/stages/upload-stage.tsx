"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AlertCircle, CheckCircle2, ExternalLink, FileArchive, FileVideo, Folder, Loader2, Play, Upload } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { DisclosurePanel, ErrorBanner, toStageStatus } from "@/components/pipeline";
import { DatasetPicker } from "@/components/stages/dataset-picker";
import { Lock, RotateCcw } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useRunPipelineStage } from "@/hooks/use-pipeline-stage";
import { getDatasetVideos, getFrameUrl, getVideos, importKaggleRunArtifacts, uploadVideo } from "@/lib/api";
import { cn, formatBytes, formatDuration } from "@/lib/utils";
import { type AppDataset, useDatasetStore } from "@/lib/store";
import { useDetectionStore, usePipelineStore, useSessionStore, useTimelineStore, useVideoStore } from "@/store";
import type { VideoFile } from "@/types";

/** Map a loaded dataset/folder name to the vehicle/person model family so the
 *  inference-stage model picker stays consistent with what was loaded. */
function inferAppDataset(name: string): AppDataset | null {
  const n = name.toLowerCase();
  if (/cityflow|aic|veri/.test(n)) return "cityflowv2";
  if (/wildtrack|epfl|person/.test(n)) return "wildtrack";
  return null;
}

function inferCameraId(video: VideoFile): string {
  const candidate = `${video.name} ${video.path}`;
  const match = candidate.match(/S\d{2}_c\d{3}/i);
  return (match?.[0] ?? "S02_c008").toUpperCase();
}

export function UploadStage() {
  const { videos, setVideos, addVideo, setCurrentVideo, currentVideo } = useVideoStore();
  const { setRunId, updateStageProgress } = usePipelineStore();
  const runId = usePipelineStore((s) => s.runId);
  const setRunInput = usePipelineStore((s) => s.setRunInput);
  const pipelineStages = usePipelineStore((s) => s.stages);
  const resetPipeline = usePipelineStore((s) => s.reset);
  const setSelectedDataset = useDatasetStore((s) => s.setSelectedDataset);
  const runPipelineStage = useRunPipelineStage();
  const { toast } = useToast();

  // Once a run exists (ingestion has created it), lock the input so the user
  // can't swap the dataset/cameras out from under the downstream per-stage runs.
  // Reset clears the run and unlocks.
  const stage0Status = toStageStatus(pipelineStages.find((s) => s.stage === 0));
  const inputLocked = Boolean(runId);

  const [isDragging, setIsDragging] = useState(false);
  const [activeDataset, setActiveDataset] = useState<string | null>(null);
  const [activeInputDir, setActiveInputDir] = useState<string | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [smokeRun, setSmokeRun] = useState(true);
  const [isStartingMtmc, setIsStartingMtmc] = useState(false);
  const [isLoadingVideos, setIsLoadingVideos] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState<Record<string, number>>({});
  const [artifactImportProgress, setArtifactImportProgress] = useState(0);
  const [isImportingArtifacts, setIsImportingArtifacts] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const artifactsInputRef = useRef<HTMLInputElement>(null);
  const enableKaggleImport = process.env.NEXT_PUBLIC_ENABLE_KAGGLE_IMPORT !== "false";

  useEffect(() => {
    const loadVideos = async () => {
      setIsLoadingVideos(true);
      setLoadError(null);
      try {
        const response = await getVideos();
        if (response.success && response.data) {
          // Show genuine user uploads on mount. Dataset footage is loaded on
          // demand via the picker - BUT keep any persisted dataset cameras (from
          // a prior run restored after reload) so re-opening a run still shows its
          // footage. Merge uploads with the persisted dataset videos.
          const uploadsOnly = response.data.filter((v) =>
            v.path.replace(/\\/g, "/").includes("/uploads/")
          );
          const persistedDatasetVideos = useVideoStore
            .getState()
            .videos.filter((v) => Boolean(v.cameraId));
          const seen = new Set(persistedDatasetVideos.map((v) => v.id));
          const merged = [...persistedDatasetVideos];
          for (const u of uploadsOnly) {
            if (!seen.has(u.id)) merged.push(u);
          }
          setVideos(merged);
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        setLoadError(`Could not fetch videos from backend: ${msg}`);
        toast({ title: "Failed to load videos", description: msg, variant: "destructive" });
      } finally {
        setIsLoadingVideos(false);
      }
    };

    void loadVideos();
  }, [setVideos, toast]);

  const handleFiles = useCallback(
    async (files: File[]) => {
      setUploadError(null);
      for (const file of files) {
        const fileId = `${file.name}-${Date.now()}`;
        setUploadProgress((prev) => ({ ...prev, [fileId]: 0 }));

        try {
          const response = await uploadVideo(file, (progress) => {
            setUploadProgress((prev) => ({ ...prev, [fileId]: progress }));
          });

          if (response.success && response.data) {
            addVideo(response.data);
            setCurrentVideo(response.data);
            toast({ title: "Upload complete", description: `${file.name} has been uploaded successfully`, variant: "success" });
          }
        } catch (error) {
          const msg = error instanceof Error ? error.message : String(error);
          setUploadError(`Failed to upload ${file.name}: ${msg}`);
          toast({ title: "Upload failed", description: msg, variant: "destructive" });
        } finally {
          setUploadProgress((prev) => {
            const { [fileId]: _removed, ...rest } = prev;
            return rest;
          });
        }
      }
    },
    [addVideo, setCurrentVideo, toast]
  );

  const handleDrop = useCallback(
    async (event: React.DragEvent) => {
      event.preventDefault();
      setIsDragging(false);

      const files = Array.from(event.dataTransfer.files).filter((file) => file.type.startsWith("video/"));
      if (files.length === 0) {
        setUploadError("Please drop video files only.");
        toast({ title: "Invalid files", description: "Please drop video files only", variant: "destructive" });
        return;
      }

      await handleFiles(files);
    },
    [handleFiles, toast]
  );

  const handleFileSelect = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(event.target.files || []);
      if (files.length > 0) await handleFiles(files);
      event.currentTarget.value = "";
    },
    [handleFiles]
  );

  const handleImportArtifactsSelect = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      event.currentTarget.value = "";
      if (!file) return;
      if (!file.name.toLowerCase().endsWith(".zip")) {
        setUploadError("Please select a .zip file exported from Kaggle outputs.");
        toast({ title: "Invalid artifact file", description: "Please select a .zip file exported from Kaggle outputs.", variant: "destructive" });
        return;
      }

      const videoForLink = currentVideo ?? videos[0] ?? null;
      const cameraId = videoForLink ? inferCameraId(videoForLink) : undefined;

      try {
        setIsImportingArtifacts(true);
        setArtifactImportProgress(0);
        setUploadError(null);

        const response = await importKaggleRunArtifacts(file, { videoId: videoForLink?.id, cameraId }, setArtifactImportProgress);
        const importedRunId = response.data?.runId ?? response.data?.id ?? null;
        if (importedRunId) {
          setRunId(importedRunId);
          updateStageProgress(6, { status: "completed", progress: 100, message: "Kaggle artifacts imported" });
        }
        if (videoForLink) setCurrentVideo(videoForLink);

        toast({
          title: "Kaggle artifacts imported",
          description: importedRunId ? `Run ${importedRunId.slice(0, 8)} is ready for timeline/output.` : "Artifacts imported successfully.",
          variant: "success",
        });
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        setUploadError(`Import failed: ${msg}`);
        toast({ title: "Artifact import failed", description: msg, variant: "destructive" });
      } finally {
        setIsImportingArtifacts(false);
        setTimeout(() => setArtifactImportProgress(0), 500);
      }
    },
    [currentVideo, setCurrentVideo, setRunId, toast, updateStageProgress, videos]
  );

  const handleResetInput = useCallback(async () => {
    const dir = activeInputDir;
    resetPipeline();
    // Clear the current video too: DetectionStage stays mounted and will
    // auto-load on-disk artifacts (and re-mark stage 1 "done") for whatever
    // video is still selected. Dropping it keeps the input genuinely unlocked.
    setCurrentVideo(null);
    setSelectedIds(new Set());
    // The run narrowed the gallery to its selected cameras; restore the dataset's
    // FULL camera list so the user can pick a different subset without reloading.
    if (dir) {
      try {
        const res = await getDatasetVideos(dir);
        setVideos(res.data ?? []);
      } catch {
        /* keep whatever is in the store */
      }
    } else {
      setActiveDataset(null);
      setActiveInputDir(null);
    }
    toast({
      title: "Input unlocked",
      description: "Pipeline reset - you can choose a different dataset or cameras now.",
    });
  }, [activeInputDir, resetPipeline, setCurrentVideo, setVideos, toast]);

  const datasetCameraVideos = videos.filter((v) => Boolean(v.cameraId));
  const allSelected =
    datasetCameraVideos.length > 0 && selectedIds.size === datasetCameraVideos.length;

  const toggleSelect = useCallback((id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const toggleSelectAll = useCallback(() => {
    setSelectedIds((prev) =>
      prev.size === datasetCameraVideos.length
        ? new Set()
        : new Set(datasetCameraVideos.map((v) => v.id))
    );
  }, [datasetCameraVideos]);

  // Stage 0 only: ingest the selected cameras and CREATE the run. Detection,
  // features, indexing, and association are NOT cascaded - each runs from its
  // own stage page against this run. Nothing downstream auto-starts.
  const handleStartRun = useCallback(async () => {
    if (!activeInputDir) return;
    const chosen = datasetCameraVideos.filter((v) => selectedIds.has(v.id));
    const cameras = chosen.map((v) => v.cameraId).filter(Boolean) as string[];
    if (cameras.length < 1) return;

    setIsStartingMtmc(true);
    // Fresh run: clear any prior run state, then capture this run's input context
    // so every downstream stage page can run incrementally against the same run.
    resetPipeline();
    // Also drop the previous run's tracking selection + timeline tracks (the selection
    // now persists across reloads) so this fresh run doesn't inherit a stale pick.
    useDetectionStore.getState().deselectAll();
    useTimelineStore.getState().resetAfterUpstreamEdit();
    setRunInput({
      inputDir: activeInputDir,
      cameras,
      name: activeDataset ?? "dataset",
      smoke: smokeRun,
    });
    // Restrict the workspace to ONLY the selected cameras so every downstream
    // stage (detection viewer, camera switcher, etc.) shows/uses exactly what's
    // being processed - not every camera in the dataset folder. Mirrors what
    // loading an existing run does (useLoadRun also scopes to the run's cameras).
    setVideos(chosen);
    setCurrentVideo(chosen[0] ?? null);
    try {
      const result = await runPipelineStage({
        pipelineStage: 0,
        uiStage: 0,
        // Surface the mode in the live progress message so it's unambiguous
        // whether the quick test (first 10 frames) is actually in effect.
        label: smokeRun ? "ingestion (quick test * first 10 frames/camera)" : "ingestion (full video)",
      });
      if (result === "completed") {
        toast({
          title: "Ingestion complete",
          description: `${smokeRun ? "Quick test (10 frames/camera)" : "Full video"} ingested for ${cameras.length} ${cameras.length === 1 ? "camera" : "cameras"}. Open Detection to run tracking.`,
          variant: "success",
        });
      }
    } finally {
      setIsStartingMtmc(false);
    }
  }, [
    activeInputDir,
    activeDataset,
    datasetCameraVideos,
    selectedIds,
    smokeRun,
    resetPipeline,
    setRunInput,
    setVideos,
    setCurrentVideo,
    runPipelineStage,
  ]);

  return (
    <div className="min-h-0 flex-1 overflow-x-hidden overflow-y-auto p-4 sm:p-6">
      <ErrorBanner title="Upload issue" message={loadError ?? uploadError} />

      {inputLocked ? (
        <div className="mt-4 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-warning/30 bg-warning/10 p-3">
          <div className="flex items-center gap-2 text-sm">
            <Lock className="h-4 w-4 shrink-0 text-warning" />
            <span className="text-foreground">
              Run active on {activeDataset ? <strong>{activeDataset}</strong> : "this input"} (run {runId}).
              {" "}Reset the pipeline to choose a different dataset or cameras.
            </span>
          </div>
          <Button type="button" variant="outline" size="sm" onClick={() => void handleResetInput()} className="gap-1.5">
            <RotateCcw className="h-3.5 w-3.5" />
            Reset &amp; change input
          </Button>
        </div>
      ) : null}

      <div className={cn(inputLocked && "pointer-events-none select-none opacity-60")}>
      <div className="mt-4">
        <DatasetPicker
          onLoaded={(name, count, inputDir) => {
            setActiveDataset(count > 0 ? name : null);
            setActiveInputDir(count > 0 ? inputDir : null);
            setSelectedIds(new Set());
            // Keep the vehicle/person model family in sync with the loaded data.
            const app = inferAppDataset(name);
            if (count > 0 && app) setSelectedDataset(app);
          }}
        />
      </div>

      <div className="mt-4 flex items-center gap-3">
        <div className="h-px flex-1 bg-border" />
        <span className="text-xs uppercase tracking-wide text-muted-foreground">or upload your own</span>
        <div className="h-px flex-1 bg-border" />
      </div>

      <div className="mt-4 grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Upload className="h-5 w-5" />
              Upload Video
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div
              className={cn(
                "dropzone relative flex min-h-[300px] flex-col items-center justify-center gap-4 p-8 text-center",
                isDragging && "dropzone-active"
              )}
              onDragOver={(event) => {
                event.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={(event) => {
                event.preventDefault();
                setIsDragging(false);
              }}
              onDrop={handleDrop}
            >
              <div className="flex h-16 w-16 items-center justify-center rounded-full bg-muted">
                <FileVideo className="h-8 w-8 text-muted-foreground" />
              </div>
              <div>
                <p className="text-lg font-medium">Drag and drop video files here</p>
                <p className="text-sm text-muted-foreground">or browse from your computer</p>
              </div>
              <Button type="button" variant="outline" onClick={() => fileInputRef.current?.click()} aria-label="Browse video files">
                <Folder className="mr-2 h-4 w-4" />
                Browse Files
              </Button>
              <p className="text-xs text-muted-foreground">Supports MP4, AVI, MKV, MOV (Max 2GB)</p>
              <input ref={fileInputRef} type="file" accept="video/*" multiple className="hidden" onChange={handleFileSelect} />
            </div>

            {Object.entries(uploadProgress).length > 0 ? (
              <div className="mt-4 space-y-2">
                {Object.entries(uploadProgress).map(([fileId, progress]) => (
                  <div key={fileId} className="space-y-1">
                    <div className="flex justify-between gap-3 text-sm">
                      <span className="truncate">{fileId.split("-")[0]}</span>
                      <span>{Math.round(progress)}%</span>
                    </div>
                    <Progress value={progress} className="h-2" />
                  </div>
                ))}
              </div>
            ) : null}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between gap-3 text-base">
              <span className="flex items-center gap-2">
                <FileVideo className="h-5 w-5" />
                {activeDataset ? `Video Gallery - ${activeDataset}` : "Video Gallery"}
              </span>
              <Badge variant="secondary">{videos.length} videos</Badge>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {datasetCameraVideos.length > 0 ? (
              <div className="mb-3 flex flex-wrap items-center justify-between gap-2 rounded-md border bg-muted/30 p-2">
                <label className="flex cursor-pointer items-center gap-2 text-sm">
                  <Checkbox checked={allSelected} onCheckedChange={toggleSelectAll} />
                  Select cameras ({selectedIds.size}/{datasetCameraVideos.length})
                </label>
                <div className="flex items-center gap-3">
                  <label
                    className="flex cursor-pointer items-center gap-1.5 text-xs text-muted-foreground"
                    title="Process only the first 10 frames per camera - a fast sanity check that runs without a GPU."
                  >
                    <Checkbox checked={smokeRun} onCheckedChange={(v) => setSmokeRun(Boolean(v))} />
                    Quick test (10 frames)
                  </label>
                  <Button
                    type="button"
                    size="sm"
                    disabled={selectedIds.size < 1 || isStartingMtmc || stage0Status === "running"}
                    onClick={() => void handleStartRun()}
                    aria-label={`Start run and ingest ${selectedIds.size} selected cameras`}
                  >
                    {isStartingMtmc || stage0Status === "running" ? (
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    ) : (
                      <Play className="mr-2 h-4 w-4" />
                    )}
                    {isStartingMtmc || stage0Status === "running"
                      ? "Ingesting..."
                      : `Start run * ingest ${selectedIds.size} ${selectedIds.size === 1 ? "camera" : "cameras"}`}
                  </Button>
                </div>
              </div>
            ) : null}
            <ScrollArea className="h-[400px]">
              {isLoadingVideos ? (
                <div className="flex h-[300px] flex-col items-center justify-center gap-2 text-muted-foreground">
                  <Loader2 className="h-8 w-8 animate-spin" />
                  <p>Loading your uploaded videos...</p>
                </div>
              ) : videos.length === 0 ? (
                <div className="flex h-[300px] flex-col items-center justify-center gap-2 px-6 text-center text-muted-foreground">
                  <AlertCircle className="h-8 w-8" />
                  <p>No videos loaded</p>
                  <p className="text-sm">Pick a dataset with "Load videos" above, or upload your own.</p>
                </div>
              ) : (
                <div className="grid gap-3">
                  {videos.map((video) => (
                    <VideoCard
                      key={video.id}
                      video={video}
                      isSelected={currentVideo?.id === video.id}
                      onSelect={() => setCurrentVideo(video)}
                      selectable={Boolean(video.cameraId)}
                      checked={selectedIds.has(video.id)}
                      onToggle={() => toggleSelect(video.id)}
                    />
                  ))}
                </div>
              )}
            </ScrollArea>
          </CardContent>
        </Card>
      </div>
      </div>

      <div className="mt-6 space-y-4">
        {enableKaggleImport ? (
          <DisclosurePanel title="Advanced" description="Import prepared Kaggle outputs for demo and recovery flows.">
            <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
              <p className="text-sm text-muted-foreground">
                Upload a Kaggle output zip containing stage folders, for example stage1, stage4, stage5, or stage6.
              </p>
              <Button type="button" variant="outline" onClick={() => artifactsInputRef.current?.click()} disabled={isImportingArtifacts} aria-label="Import Kaggle artifact zip">
                {isImportingArtifacts ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <FileArchive className="mr-2 h-4 w-4" />}
                Import Artifact Zip
              </Button>
            </div>
            {isImportingArtifacts ? (
              <div className="mt-3 space-y-1">
                <div className="flex justify-between text-sm">
                  <span>Uploading and importing artifacts...</span>
                  <span>{Math.round(artifactImportProgress)}%</span>
                </div>
                <Progress value={artifactImportProgress} className="h-2" />
              </div>
            ) : null}
            <input ref={artifactsInputRef} type="file" accept=".zip,application/zip" className="hidden" onChange={handleImportArtifactsSelect} />
          </DisclosurePanel>
        ) : null}

        <DisclosurePanel title="Debug" tier="debug" description="Dataset compatibility notes for model behavior.">
          <div className="space-y-3 text-sm text-muted-foreground">
            <p>
              Best results use datasets compatible with trained models from{" "}
              <a href="https://www.kaggle.com/datasets/mrkdagods/mtmc-weights" target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 font-medium text-primary hover:underline">
                mrkdagods/mtmc-weights
                <ExternalLink className="h-3 w-3" />
              </a>
              .
            </p>
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="rounded-md border border-success/30 bg-success/10 p-3">
                <p className="mb-1 text-xs font-medium text-success">Recommended</p>
                <p className="text-xs">AI City Challenge, VeRi-776, and CityFlow benchmark footage.</p>
              </div>
              <div className="rounded-md border border-warning/30 bg-warning/10 p-3">
                <p className="mb-1 text-xs font-medium text-warning">Limited support</p>
                <p className="text-xs">Random videos, non-traffic scenes, and low-resolution sources.</p>
              </div>
            </div>
          </div>
        </DisclosurePanel>
      </div>
    </div>
  );
}

export function UploadStageActions() {
  const { setCurrentStage } = useSessionStore();
  const stages = usePipelineStore((s) => s.stages);
  // Navigation only - Stage 1 detection is started from the Detection page's own
  // Run button. Stays disabled until ingestion (stage 0) has actually finished,
  // not merely when the run id is allocated at ingestion start.
  const ingestDone = toStageStatus(stages.find((s) => s.stage === 0)) === "done";

  return (
    <Button
      type="button"
      onClick={() => setCurrentStage(1)}
      disabled={!ingestDone}
      title={!ingestDone ? "Waiting for ingestion to finish" : undefined}
      aria-label="Go to Detection stage"
    >
      Go to Detection
    </Button>
  );
}

function VideoThumbnail({ video }: { video: VideoFile }) {
  const [failed, setFailed] = useState(false);
  const src = video.thumbnail ?? getFrameUrl(video.id, 0);
  if (failed || !src) {
    return (
      <div className="flex h-full w-full items-center justify-center">
        <FileVideo className="h-6 w-6 text-muted-foreground" />
      </div>
    );
  }
  return (
    <img
      src={src}
      alt={video.name}
      loading="lazy"
      className="h-full w-full object-cover"
      onError={() => setFailed(true)}
    />
  );
}

function VideoCard({
  video,
  isSelected,
  onSelect,
  selectable = false,
  checked = false,
  onToggle,
}: {
  video: VideoFile;
  isSelected: boolean;
  onSelect: () => void;
  selectable?: boolean;
  checked?: boolean;
  onToggle?: () => void;
}) {
  return (
    <div
      className={cn(
        "flex w-full items-center gap-3 rounded-lg border p-3 transition-colors hover:bg-accent",
        isSelected && "border-primary bg-primary/5",
        checked && "border-primary/60 bg-primary/5"
      )}
    >
      {selectable ? (
        <Checkbox
          checked={checked}
          onCheckedChange={() => onToggle?.()}
          aria-label={`Select camera ${video.cameraId ?? video.name}`}
          className="flex-shrink-0"
        />
      ) : null}
      <button
        type="button"
        className="flex min-w-0 flex-1 items-center gap-3 text-left focus-visible:outline-none"
        onClick={onSelect}
        aria-label={`Preview ${video.name}`}
        aria-pressed={isSelected}
      >
        <div className="relative h-16 w-24 flex-shrink-0 overflow-hidden rounded-md bg-muted">
          <VideoThumbnail video={video} />
          <div className="absolute bottom-1 right-1 rounded bg-black/70 px-1 text-[10px] text-white">{formatDuration(video.duration)}</div>
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <p className="truncate font-medium">{video.name}</p>
            {video.cameraId ? (
              <Badge variant="secondary" className="flex-shrink-0 text-[10px]">
                {video.cameraId}
              </Badge>
            ) : null}
          </div>
          <p className="text-sm text-muted-foreground">
            {video.width}x{video.height} @ {video.fps}fps
          </p>
          <p className="text-xs text-muted-foreground">{formatBytes(video.size)}</p>
        </div>
        {isSelected ? <CheckCircle2 className="h-5 w-5 flex-shrink-0 text-primary" /> : null}
      </button>
    </div>
  );
}
