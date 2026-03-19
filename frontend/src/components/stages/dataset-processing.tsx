"use client";

import { useEffect, useState, useCallback } from "react";
import {
  FolderOpen,
  Camera,
  Play,
  Loader2,
  CheckCircle2,
  RefreshCw,
  Video,
  XCircle,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { usePipelineStore } from "@/store";
import {
  getDatasets,
  processDataset,
  getPipelineStatus,
  type DatasetFolder,
} from "@/lib/api";

export function DatasetProcessing() {
  const [datasets, setDatasets] = useState<DatasetFolder[]>([]);
  const [loading, setLoading] = useState(true);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [processingFolder, setProcessingFolder] = useState<string | null>(null);
  const [runProgress, setRunProgress] = useState<Record<string, any>>({});

  const { setRunId, setIsRunning } = usePipelineStore();

  const fetchDatasets = useCallback(async () => {
    try {
      const resp: any = await getDatasets();
      const data = resp?.data ?? resp;
      setDatasets(Array.isArray(data) ? data : []);
    } catch (err) {
      setDatasets([]);
      setFetchError(String(err instanceof Error ? err.message : err || "Failed to load datasets"));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchDatasets();
  }, [fetchDatasets]);

  // Poll progress for any processing datasets
  useEffect(() => {
    const processingIds = datasets
      .filter((d) => d.isProcessing && d.runId)
      .map((d) => d.runId!);

    if (processingFolder) {
      const runId = `dataset_precompute_${processingFolder.toLowerCase()}`;
      if (!processingIds.includes(runId)) processingIds.push(runId);
    }

    if (processingIds.length === 0) return;

    const poll = async () => {
      for (const id of processingIds) {
        try {
          const resp: any = await getPipelineStatus(id);
          const data = resp?.data ?? resp;
          setRunProgress((prev) => ({ ...prev, [id]: data }));

          if (data?.status === "completed" || data?.status === "error") {
            setProcessingFolder(null);
            void fetchDatasets(); // refresh list
          }
        } catch (err) {
          // log but keep polling — transient network hiccup
          console.warn(`Status poll failed for ${id}:`, err);
        }
      }
    };

    const interval = setInterval(poll, 2000);
    void poll();
    return () => clearInterval(interval);
  }, [datasets, processingFolder, fetchDatasets]);

  const handleProcess = async (folder: string) => {
    setProcessingFolder(folder);
    const runId = `dataset_precompute_${folder.toLowerCase()}`;
    setRunId(runId);
    setIsRunning(true);

    try {
      await processDataset(folder);
      void fetchDatasets();
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setProcessingFolder(null);
      setIsRunning(false);
      setFetchError(`Failed to start dataset processing: ${msg}`);
    }
  };

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <ScrollArea className="h-full">
      <div className="p-6 space-y-6 max-w-4xl mx-auto">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold tracking-tight">
              Dataset Processing
            </h2>
            <p className="text-muted-foreground mt-1">
              Select a dataset folder to process all cameras through the full
              pipeline (stages 0–4).
            </p>
          </div>
          <Button variant="outline" size="sm" onClick={fetchDatasets}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>

        {fetchError && (
          <Card className="border-destructive">
            <CardContent className="flex items-center gap-3 py-4">
              <XCircle className="h-5 w-5 text-destructive flex-shrink-0" />
              <div>
                <p className="font-medium text-destructive">Failed to load datasets</p>
                <p className="text-sm text-muted-foreground">{fetchError}</p>
              </div>
            </CardContent>
          </Card>
        )}

        {!fetchError && datasets.length === 0 ? (
          <Card>
            <CardContent className="flex flex-col items-center justify-center py-12">
              <FolderOpen className="h-12 w-12 text-muted-foreground mb-4" />
              <p className="text-muted-foreground">
                No dataset folders found in{" "}
                <code className="text-xs bg-muted px-1 py-0.5 rounded">
                  dataset/
                </code>
              </p>
            </CardContent>
          </Card>
        ) : (
          <div className="grid gap-4">
            {datasets.map((ds) => {
              const runId = `dataset_precompute_${ds.name.toLowerCase()}`;
              const progress = runProgress[runId];
              const isProcessing = ds.isProcessing || processingFolder === ds.name;

              return (
                <Card key={ds.name}>
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <CardTitle className="flex items-center gap-2 text-lg">
                        <FolderOpen className="h-5 w-5" />
                        {ds.name}
                      </CardTitle>
                      <div className="flex items-center gap-2">
                        {ds.alreadyProcessed && !isProcessing && (
                          <Badge
                            variant="secondary"
                            className="bg-green-500/10 text-green-500 border-green-500/30"
                          >
                            <CheckCircle2 className="h-3 w-3 mr-1" />
                            Processed
                          </Badge>
                        )}
                        {isProcessing && (
                          <Badge
                            variant="secondary"
                            className="bg-primary/10 text-primary border-primary/30"
                          >
                            <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                            Processing
                          </Badge>
                        )}
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {/* Camera info */}
                    <div className="flex flex-wrap gap-2">
                      {ds.cameras.map((cam) => (
                        <div
                          key={cam.id}
                          className="flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs"
                        >
                          {cam.hasVideo ? (
                            <Video className="h-3 w-3 text-green-500" />
                          ) : (
                            <Camera className="h-3 w-3 text-muted-foreground" />
                          )}
                          {cam.id}
                        </div>
                      ))}
                    </div>

                    <div className="text-sm text-muted-foreground">
                      {ds.cameraCount} camera{ds.cameraCount !== 1 ? "s" : ""},{" "}
                      {ds.videosFound} video{ds.videosFound !== 1 ? "s" : ""}{" "}
                      found
                    </div>

                    {/* Progress when processing */}
                    {isProcessing && progress && (
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-muted-foreground">
                            {progress.message || "Processing..."}
                          </span>
                          <span className="font-mono text-xs">
                            {Math.round(progress.progress ?? 0)}%
                          </span>
                        </div>
                        <Progress value={progress.progress ?? 0} className="h-2" />
                        {progress.totalStages > 1 && (
                          <div className="text-xs text-muted-foreground">
                            Stage {(progress.completedStages ?? 0) + 1} of{" "}
                            {progress.totalStages}
                            {progress.currentStageName &&
                              ` — ${progress.currentStageName}`}
                          </div>
                        )}
                      </div>
                    )}

                    {/* Action button */}
                    {!isProcessing && (
                      <Button
                        onClick={() => void handleProcess(ds.name)}
                        disabled={ds.videosFound === 0}
                        className="w-full"
                        variant={ds.alreadyProcessed ? "outline" : "default"}
                      >
                        <Play className="h-4 w-4 mr-2" />
                        {ds.alreadyProcessed
                          ? "Reprocess Dataset"
                          : "Process Dataset"}
                      </Button>
                    )}
                  </CardContent>
                </Card>
              );
            })}
          </div>
        )}
      </div>
    </ScrollArea>
  );
}
