"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AlertCircle, CheckCircle2, ExternalLink, FileArchive, FileVideo, Folder, Loader2, Upload } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { DisclosurePanel, ErrorBanner, toStageStatus } from "@/components/pipeline";
import { useToast } from "@/hooks/use-toast";
import { getFrameUrl, getVideos, importKaggleRunArtifacts, uploadVideo } from "@/lib/api";
import { cn, formatBytes, formatDuration } from "@/lib/utils";
import { usePipelineStore, useSessionStore, useVideoStore } from "@/store";
import type { VideoFile } from "@/types";

function inferCameraId(video: VideoFile): string {
  // Prefer the real camera id (e.g. WILDTRACK "C1".."C7"); fall back to the
  // CityFlow S##_c### pattern, then a default.
  if (video.cameraId && video.cameraId.trim()) return video.cameraId.trim();
  const candidate = `${video.name} ${video.path}`;
  const match = candidate.match(/S\d{2}_c\d{3}/i);
  return (match?.[0] ?? "S02_c008").toUpperCase();
}

/**
 * Upload = the PROBE step. You upload (or pick a previously uploaded) video that
 * contains the subject you want to search for. Detection/Selection then run on
 * THIS video only; the dataset you search *within* is chosen later at Inference.
 */
export function UploadStage() {
  const { videos, setVideos, addVideo, setCurrentVideo, currentVideo } = useVideoStore();
  const { setRunId, updateStageProgress } = usePipelineStore();
  const { toast } = useToast();

  const [isDragging, setIsDragging] = useState(false);
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
          // Probe flow shows genuine user uploads only. Dataset footage is never
          // ingested here anymore - it's the search target, chosen at Inference.
          const uploadsOnly = response.data.filter((v) =>
            v.path.replace(/\\/g, "/").includes("/uploads/")
          );
          setVideos(uploadsOnly);
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
      // Capture the input element before the await: the native event's
      // currentTarget is cleared once the event finishes dispatching, so
      // reading it after an `await` throws "Cannot set properties of null".
      const input = event.currentTarget;
      const files = Array.from(input.files || []);
      if (files.length > 0) await handleFiles(files);
      input.value = "";
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

  return (
    <div className="min-h-0 flex-1 overflow-x-hidden overflow-y-auto p-4 sm:p-6">
      <ErrorBanner title="Upload issue" message={loadError ?? uploadError} />

      <div className="mt-2 rounded-lg border border-primary/20 bg-primary/5 p-3 text-sm text-muted-foreground">
        <span className="font-medium text-foreground">Step 1 - Upload your probe video.</span>{" "}
        Upload the clip containing the subject (vehicle or person) you want to find.
        Detection and Selection run on this video only. You&apos;ll choose the dataset
        to search <em>within</em> later, at the Inference step.
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
                <p className="text-lg font-medium">Drag and drop your probe video here</p>
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
                Your Uploaded Videos
              </span>
              <Badge variant="secondary">{videos.length} videos</Badge>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="mb-3 text-xs text-muted-foreground">
              Click a video to make it the active probe for Detection.
            </p>
            <ScrollArea className="h-[400px]">
              {isLoadingVideos ? (
                <div className="flex h-[300px] flex-col items-center justify-center gap-2 text-muted-foreground">
                  <Loader2 className="h-8 w-8 animate-spin" />
                  <p>Loading your uploaded videos...</p>
                </div>
              ) : videos.length === 0 ? (
                <div className="flex h-[300px] flex-col items-center justify-center gap-2 px-6 text-center text-muted-foreground">
                  <AlertCircle className="h-8 w-8" />
                  <p>No videos uploaded yet</p>
                  <p className="text-sm">Drop or browse a probe video on the left to get started.</p>
                </div>
              ) : (
                <div className="grid gap-3">
                  {videos.map((video) => (
                    <VideoCard
                      key={video.id}
                      video={video}
                      isSelected={currentVideo?.id === video.id}
                      onSelect={() => setCurrentVideo(video)}
                    />
                  ))}
                </div>
              )}
            </ScrollArea>
          </CardContent>
        </Card>
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
  const currentVideo = useVideoStore((s) => s.currentVideo);
  const stages = usePipelineStore((s) => s.stages);
  // Probe flow: Detection runs on the selected video and creates the run itself.
  // Enable navigation as soon as a probe video is chosen. If Stage 1 already ran,
  // it stays enabled.
  const stage1Done = toStageStatus(stages.find((s) => s.stage === 1)) === "done";
  const canProceed = Boolean(currentVideo) || stage1Done;

  return (
    <Button
      type="button"
      onClick={() => setCurrentStage(1)}
      disabled={!canProceed}
      title={!canProceed ? "Upload or select a probe video first" : undefined}
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
}: {
  video: VideoFile;
  isSelected: boolean;
  onSelect: () => void;
}) {
  return (
    <div
      className={cn(
        "flex w-full items-center gap-3 rounded-lg border p-3 transition-colors hover:bg-accent",
        isSelected && "border-primary bg-primary/5"
      )}
    >
      <button
        type="button"
        className="flex min-w-0 flex-1 items-center gap-3 text-left focus-visible:outline-none"
        onClick={onSelect}
        aria-label={`Use ${video.name} as probe`}
        aria-pressed={isSelected}
      >
        <div className="relative h-16 w-24 flex-shrink-0 overflow-hidden rounded-md bg-muted">
          <VideoThumbnail video={video} />
          <div className="absolute bottom-1 right-1 rounded bg-black/70 px-1 text-[10px] text-white">{formatDuration(video.duration)}</div>
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <p className="truncate font-medium">{video.name}</p>
            {isSelected ? (
              <Badge variant="secondary" className="flex-shrink-0 text-[10px]">
                probe
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
