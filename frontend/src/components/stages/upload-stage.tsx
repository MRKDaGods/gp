"use client";

import { useCallback, useEffect, useState, useRef } from "react";
import {
  Upload,
  Loader2,
  FileVideo,
  FileArchive,
  X,
  Play,
  Folder,
  AlertCircle,
  CheckCircle2,
  Info,
  ExternalLink,
} from "lucide-react";
import { cn, formatBytes, formatDuration } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { useVideoStore, useSessionStore, usePipelineStore } from "@/store";
import { getVideos, importKaggleRunArtifacts, runStage, uploadVideo } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";
import type { VideoFile } from "@/types";

export function UploadStage() {
  const { videos, setVideos, addVideo, setCurrentVideo, currentVideo } = useVideoStore();
  const { setCurrentStage } = useSessionStore();
  const { setRunId, setIsRunning, updateStageProgress } = usePipelineStore();
  const { toast } = useToast();

  const [isDragging, setIsDragging] = useState(false);
  const [isLoadingVideos, setIsLoadingVideos] = useState(true);
  const [uploadProgress, setUploadProgress] = useState<Record<string, number>>({});
  const [uploadingFiles, setUploadingFiles] = useState<Set<string>>(new Set());
  const [artifactImportProgress, setArtifactImportProgress] = useState(0);
  const [isImportingArtifacts, setIsImportingArtifacts] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const artifactsInputRef = useRef<HTMLInputElement>(null);
  const enableKaggleImport = process.env.NEXT_PUBLIC_ENABLE_KAGGLE_IMPORT !== "false";

  useEffect(() => {
    const loadVideos = async () => {
      setIsLoadingVideos(true);
      try {
        const response = await getVideos();
        if (response.success && response.data) {
          setVideos(response.data);
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        toast({
          title: "Failed to load videos",
          description: `Could not fetch videos from backend: ${msg}`,
          variant: "destructive",
        });
      } finally {
        setIsLoadingVideos(false);
      }
    };

    loadVideos();
  }, [setVideos, toast]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);

      const files = Array.from(e.dataTransfer.files).filter((file) =>
        file.type.startsWith("video/")
      );

      if (files.length === 0) {
        toast({
          title: "Invalid files",
          description: "Please drop video files only",
          variant: "destructive",
        });
        return;
      }

      await handleFiles(files);
    },
    [toast]
  );

  const handleFileSelect = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(e.target.files || []);
      if (files.length > 0) {
        await handleFiles(files);
      }
    },
    []
  );

  const handleFiles = async (files: File[]) => {
    for (const file of files) {
      const fileId = `${file.name}-${Date.now()}`;
      setUploadingFiles((prev) => new Set(prev).add(fileId));
      setUploadProgress((prev) => ({ ...prev, [fileId]: 0 }));

      try {
        const response = await uploadVideo(file, (progress) => {
          setUploadProgress((prev) => ({ ...prev, [fileId]: progress }));
        });

        if (response.success && response.data) {
          addVideo(response.data);
          toast({
            title: "Upload complete",
            description: `${file.name} has been uploaded successfully`,
            variant: "success",
          });
        }
      } catch (error) {
        const msg = error instanceof Error ? error.message : String(error);
        toast({
          title: "Upload failed",
          description: `Failed to upload ${file.name}: ${msg}`,
          variant: "destructive",
        });
      } finally {
        setUploadingFiles((prev) => {
          const next = new Set(prev);
          next.delete(fileId);
          return next;
        });
        setUploadProgress((prev) => {
          const { [fileId]: _, ...rest } = prev;
          return rest;
        });
      }
    }
  };

  const inferCameraId = (video: VideoFile): string => {
    const candidate = `${video.name} ${video.path}`;
    const match = candidate.match(/S\d{2}_c\d{3}/i);
    return (match?.[0] ?? "S02_c008").toUpperCase();
  };

  const handleSelectAndProceed = async (video: VideoFile) => {
    const cameraId = inferCameraId(video);

    setCurrentVideo(video);
    setRunId(null);
    setIsRunning(true);
    updateStageProgress(1, {
      status: "running",
      progress: 0,
      message: `Queued Stage 1 for ${video.name}`,
    });

    try {
      const response = await runStage(1, {
        videoId: video.id,
        cameraId,
      });

      const runId = (response.data as any)?.runId ?? (response.data as any)?.id ?? null;
      if (runId) {
        setRunId(runId);
      }

      toast({
        title: "Stage 1 started",
        description: `Running detection/tracking on "${video.name}"`,
        variant: "success",
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setIsRunning(false);
      updateStageProgress(1, {
        status: "error",
        progress: 100,
        message: `Failed to start Stage 1: ${msg}`,
      });
      toast({
        title: "Failed to start Stage 1",
        description: msg,
        variant: "destructive",
      });
      return;
    }

    setCurrentStage(1);
  };

  const handleImportArtifactsSelect = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      if (!file.name.toLowerCase().endsWith(".zip")) {
        toast({
          title: "Invalid artifact file",
          description: "Please select a .zip file exported from Kaggle outputs.",
          variant: "destructive",
        });
        return;
      }

      const videoForLink = currentVideo ?? videos[0] ?? null;
      const cameraId = videoForLink ? inferCameraId(videoForLink) : undefined;

      try {
        setIsImportingArtifacts(true);
        setArtifactImportProgress(0);

        const response = await importKaggleRunArtifacts(
          file,
          {
            videoId: videoForLink?.id,
            cameraId,
          },
          setArtifactImportProgress
        );

        const importedRunId = (response.data as any)?.runId ?? (response.data as any)?.id ?? null;
        if (importedRunId) {
          setRunId(importedRunId);
          updateStageProgress(6, {
            status: "completed",
            progress: 100,
            message: "Kaggle artifacts imported",
          });
        }

        if (videoForLink) {
          setCurrentVideo(videoForLink);
        }

        toast({
          title: "Kaggle artifacts imported",
          description: importedRunId
            ? `Run ${importedRunId.slice(0, 8)} is ready for timeline/output.`
            : "Artifacts imported successfully.",
          variant: "success",
        });
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        toast({
          title: "Artifact import failed",
          description: `Import failed: ${msg}`,
          variant: "destructive",
        });
      } finally {
        setIsImportingArtifacts(false);
        setTimeout(() => setArtifactImportProgress(0), 500);
      }
    },
    [currentVideo, setCurrentVideo, setRunId, toast, updateStageProgress, videos]
  );

  // Quick-start mode - pick first available real video
  const handleDemoMode = () => {
    const candidate = videos[0];
    if (!candidate) {
      toast({
        title: "No videos available",
        description: "Upload a video first or add CityFlowV2 videos to the uploads folder.",
        variant: "destructive",
      });
      return;
    }

    setCurrentVideo(candidate);
    setCurrentStage(1);
    toast({
      title: "Using real video",
      description: `Loaded ${candidate.name} for YOLOv8 + Deep OC-SORT detection.`,
      variant: "success",
    });
  };

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <header className="flex h-14 items-center justify-between border-b px-6">
        <div>
          <h1 className="text-lg font-semibold">Upload Video</h1>
          <p className="text-sm text-muted-foreground">
            Upload surveillance footage for analysis
          </p>
        </div>
        <Button variant="outline" onClick={handleDemoMode}>
          <Play className="mr-2 h-4 w-4" />
          Demo Mode
        </Button>
      </header>

      {/* Content */}
      <div className="flex-1 overflow-auto p-6">
        {/* Dataset compatibility notice */}
        <Alert className="mb-6 border-blue-500/50 bg-blue-500/10">
          <Info className="h-4 w-4 text-blue-500" />
          <AlertTitle>Use Compatible Dataset Videos</AlertTitle>
          <AlertDescription>
            <div className="mt-2 space-y-2 text-sm">
              <p>
                For best results, use videos from datasets compatible with trained models from{" "}
                <a
                  href="https://www.kaggle.com/datasets/mrkdagods/mtmc-weights"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="font-medium text-blue-400 hover:underline inline-flex items-center gap-1"
                >
                  mrkdagods/mtmc-weights
                  <ExternalLink className="h-3 w-3" />
                </a>
              </p>
              <div className="grid grid-cols-2 gap-3 mt-3">
                <div className="p-2 rounded border border-green-500/30 bg-green-500/10">
                  <p className="font-medium text-green-400 text-xs mb-1">✓ Recommended</p>
                  <ul className="text-xs space-y-0.5">
                    <li>• AI City Challenge 2023</li>
                    <li>• VeRi-776 Dataset</li>
                    <li>• CityFlow Benchmark</li>
                  </ul>
                </div>
                <div className="p-2 rounded border border-yellow-500/30 bg-yellow-500/10">
                  <p className="font-medium text-yellow-400 text-xs mb-1">⚠ Limited Support</p>
                  <ul className="text-xs space-y-0.5">
                    <li>• Random videos</li>
                    <li>• Non-traffic scenes</li>
                    <li>• Low resolution</li>
                  </ul>
                </div>
              </div>
              <div className="mt-3 text-xs">
                <a
                  href="https://www.aicitychallenge.org/2023-data-and-evaluation/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-400 hover:underline"
                >
                  → Download AI City Challenge Dataset
                </a>
              </div>
            </div>
          </AlertDescription>
        </Alert>

        {enableKaggleImport && (
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileArchive className="h-5 w-5" />
              Import Kaggle Artifacts (Demo Fast Path)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
              <p className="text-sm text-muted-foreground">
                Upload a Kaggle output zip containing stage folders (for example stage1, stage4, stage5, stage6) to use immediately in this demo.
              </p>
              <Button
                variant="outline"
                onClick={() => artifactsInputRef.current?.click()}
                disabled={isImportingArtifacts}
              >
                {isImportingArtifacts ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <FileArchive className="mr-2 h-4 w-4" />
                )}
                Import Artifact Zip
              </Button>
            </div>

            {isImportingArtifacts && (
              <div className="mt-3 space-y-1">
                <div className="flex justify-between text-sm">
                  <span>Uploading and importing artifacts...</span>
                  <span>{Math.round(artifactImportProgress)}%</span>
                </div>
                <Progress value={artifactImportProgress} className="h-2" />
              </div>
            )}

            <input
              ref={artifactsInputRef}
              type="file"
              accept=".zip,application/zip"
              className="hidden"
              onChange={handleImportArtifactsSelect}
            />
          </CardContent>
        </Card>
        )}

        <div className="grid gap-6 lg:grid-cols-2">
          {/* Upload area */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
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
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                <div className="flex h-16 w-16 items-center justify-center rounded-full bg-muted">
                  <FileVideo className="h-8 w-8 text-muted-foreground" />
                </div>
                <div>
                  <p className="text-lg font-medium">
                    Drag and drop video files here
                  </p>
                  <p className="text-sm text-muted-foreground">
                    or click to browse from your computer
                  </p>
                </div>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <Folder className="mr-2 h-4 w-4" />
                    Browse Files
                  </Button>
                </div>
                <p className="text-xs text-muted-foreground">
                  Supports MP4, AVI, MKV, MOV (Max 2GB)
                </p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="video/*"
                  multiple
                  className="hidden"
                  onChange={handleFileSelect}
                />
              </div>

              {/* Upload progress */}
              {Object.entries(uploadProgress).length > 0 && (
                <div className="mt-4 space-y-2">
                  {Object.entries(uploadProgress).map(([fileId, progress]) => (
                    <div key={fileId} className="space-y-1">
                      <div className="flex justify-between text-sm">
                        <span className="truncate">{fileId.split("-")[0]}</span>
                        <span>{Math.round(progress)}%</span>
                      </div>
                      <Progress value={progress} className="h-2" />
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Video gallery */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span className="flex items-center gap-2">
                  <FileVideo className="h-5 w-5" />
                  Video Gallery
                </span>
                <Badge variant="secondary">{videos.length} videos</Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[400px]">
                {isLoadingVideos ? (
                  <div className="flex h-[300px] flex-col items-center justify-center gap-2 text-muted-foreground">
                    <Loader2 className="h-8 w-8 animate-spin" />
                    <p>Loading local videos...</p>
                    <p className="text-sm">Scanning uploads and CityFlow directories</p>
                  </div>
                ) : videos.length === 0 ? (
                  <div className="flex h-[300px] flex-col items-center justify-center gap-2 text-muted-foreground">
                    <AlertCircle className="h-8 w-8" />
                    <p>No videos uploaded yet</p>
                    <p className="text-sm">
                      Upload videos to start tracking
                    </p>
                  </div>
                ) : (
                  <div className="grid gap-3">
                    {videos.map((video) => (
                      <VideoCard
                        key={video.id}
                        video={video}
                        isSelected={currentVideo?.id === video.id}
                        onSelect={() => void handleSelectAndProceed(video)}
                      />
                    ))}
                  </div>
                )}
              </ScrollArea>
            </CardContent>
          </Card>
        </div>

        {/* Quick start info */}
        <Card className="mt-6">
          <CardContent className="p-6">
            <div className="grid gap-4 md:grid-cols-3">
              <InfoBlock
                number="1"
                title="Upload Video"
                description="Upload surveillance footage from cameras"
              />
              <InfoBlock
                number="2"
                title="Auto-Detection"
                description="YOLO detects all vehicles automatically"
              />
              <InfoBlock
                number="3"
                title="Select & Track"
                description="Select objects to track across cameras"
              />
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
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
        "flex items-center gap-3 rounded-lg border p-3 transition-colors cursor-pointer hover:bg-accent",
        isSelected && "border-primary bg-primary/5"
      )}
      onClick={onSelect}
    >
      <div className="relative h-16 w-24 flex-shrink-0 overflow-hidden rounded-md bg-muted">
        {video.thumbnail ? (
          <img
            src={video.thumbnail}
            alt={video.name}
            className="h-full w-full object-cover"
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center">
            <FileVideo className="h-6 w-6 text-muted-foreground" />
          </div>
        )}
        <div className="absolute bottom-1 right-1 rounded bg-black/70 px-1 text-[10px] text-white">
          {formatDuration(video.duration)}
        </div>
      </div>
      <div className="flex-1 min-w-0">
        <p className="truncate font-medium">{video.name}</p>
        <p className="text-sm text-muted-foreground">
          {video.width}x{video.height} @ {video.fps}fps
        </p>
        <p className="text-xs text-muted-foreground">{formatBytes(video.size)}</p>
      </div>
      {isSelected && (
        <CheckCircle2 className="h-5 w-5 flex-shrink-0 text-primary" />
      )}
    </div>
  );
}

function InfoBlock({
  number,
  title,
  description,
}: {
  number: string;
  title: string;
  description: string;
}) {
  return (
    <div className="flex items-start gap-3">
      <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground font-semibold">
        {number}
      </div>
      <div>
        <p className="font-medium">{title}</p>
        <p className="text-sm text-muted-foreground">{description}</p>
      </div>
    </div>
  );
}
