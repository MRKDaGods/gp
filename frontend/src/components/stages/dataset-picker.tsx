"use client";

import { useCallback, useEffect, useState } from "react";
import {
  ChevronRight,
  Database,
  FileVideo,
  Folder,
  FolderOpen,
  Home,
  Loader2,
  RefreshCw,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ErrorBanner } from "@/components/pipeline";
import { useToast } from "@/hooks/use-toast";
import {
  type AvailableDataset,
  type BrowseResult,
  type DatasetCamera,
  browseDatasetFolder,
  getAvailableDatasets,
  getDatasetVideos,
} from "@/lib/api";
import { useVideoStore } from "@/store";

function CameraChips({ cameras }: { cameras: DatasetCamera[] }) {
  if (!cameras.length) return null;
  const shown = cameras.slice(0, 10);
  return (
    <div className="flex flex-wrap gap-1">
      {shown.map((c) => (
        <span
          key={c.id}
          className="rounded bg-muted px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground"
        >
          {c.id}
        </span>
      ))}
      {cameras.length > shown.length ? (
        <span className="text-[10px] text-muted-foreground">+{cameras.length - shown.length}</span>
      ) : null}
    </div>
  );
}

function AvailabilityBadge({ d }: { d: AvailableDataset }) {
  if (d.available) {
    return (
      <Badge variant="outline" className="border-success/30 text-success">
        ready
      </Badge>
    );
  }
  if (d.layout === "empty") {
    return (
      <Badge variant="outline" className="border-warning/30 text-warning">
        no videos
      </Badge>
    );
  }
  return (
    <Badge variant="outline" className="border-destructive/30 text-destructive">
      not downloaded
    </Badge>
  );
}

const LAYOUT_LABEL: Record<AvailableDataset["layout"], string> = {
  per_camera: "one folder per camera",
  flat: "one file per camera",
  empty: "no videos found",
  missing: "folder missing",
};

/** Human-readable fps: native rate, noting the sampling rate when it differs. */
function fpsLabel(d: AvailableDataset): string {
  const src = d.sourceFps ?? null;
  const sample = d.sampleFps ?? null;
  if (src == null && sample == null) return "";
  if (src == null) return `samples @ ${sample} fps`;
  if (sample != null && Math.abs(sample - src) > 0.01) {
    return `${src} fps source · samples @ ${sample} fps`;
  }
  return `${src} fps`;
}

/** Lets the user choose Stage 0 input from a curated dataset list or a folder browser. */
export function DatasetPicker({
  onLoaded,
}: {
  onLoaded?: (name: string, count: number, inputDir: string) => void;
}) {
  const { toast } = useToast();
  const { setVideos, setCurrentVideo } = useVideoStore();

  const [loadingInput, setLoadingInput] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onLoad = useCallback(
    async (inputDir: string, name: string) => {
      setLoadingInput(inputDir);
      setError(null);
      try {
        const res = await getDatasetVideos(inputDir);
        const vids = res.data ?? [];
        setVideos(vids);
        if (vids.length) setCurrentVideo(vids[0]);
        onLoaded?.(name, vids.length, inputDir);
        toast({
          title: vids.length ? "Videos loaded" : "No videos found",
          description: vids.length
            ? `Loaded ${vids.length} camera video${vids.length === 1 ? "" : "s"} from ${name}. Pick the cameras you want below.`
            : `No videos were found in ${name}.`,
          variant: vids.length ? "success" : "destructive",
        });
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        setError(`Could not load videos from ${name}: ${msg}`);
        toast({ title: "Failed to load videos", description: msg, variant: "destructive" });
      } finally {
        setLoadingInput(null);
      }
    },
    [setVideos, setCurrentVideo, onLoaded, toast]
  );

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Database className="h-5 w-5" />
          Choose Existing Dataset
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ErrorBanner title="Dataset issue" message={error} />
        <Tabs defaultValue="datasets" className="mt-2">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="datasets">
              <Database className="mr-2 h-4 w-4" />
              Datasets
            </TabsTrigger>
            <TabsTrigger value="browse">
              <FolderOpen className="mr-2 h-4 w-4" />
              Browse folder
            </TabsTrigger>
          </TabsList>
          <TabsContent value="datasets">
            <CuratedDatasets onLoad={onLoad} loadingInput={loadingInput} />
          </TabsContent>
          <TabsContent value="browse">
            <FolderBrowser onLoad={onLoad} loadingInput={loadingInput} />
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}

function CuratedDatasets({
  onLoad,
  loadingInput,
}: {
  onLoad: (inputDir: string, name: string) => void;
  loadingInput: string | null;
}) {
  const [datasets, setDatasets] = useState<AvailableDataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getAvailableDatasets();
      if (res.success && res.data) setDatasets(res.data);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  if (loading) {
    return (
      <div className="flex h-40 items-center justify-center text-muted-foreground">
        <Loader2 className="mr-2 h-5 w-5 animate-spin" /> Loading datasets...
      </div>
    );
  }
  if (error) return <ErrorBanner title="Could not load datasets" message={error} />;
  if (!datasets.length) {
    return (
      <p className="py-8 text-center text-sm text-muted-foreground">
        No dataset configs found under configs/datasets/.
      </p>
    );
  }

  return (
    <ScrollArea className="h-[340px] pr-2">
      <div className="grid gap-3">
        {datasets.map((d) => (
          <div key={d.name} className="rounded-lg border p-3">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <p className="font-medium">{d.name}</p>
                  <AvailabilityBadge d={d} />
                  {d.taskType ? (
                    <Badge variant="secondary" className="text-[10px]">
                      {d.taskType}
                    </Badge>
                  ) : null}
                </div>
                <p className="mt-0.5 truncate text-xs text-muted-foreground">{d.inputDir}</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  {d.cameraCount} cameras · {LAYOUT_LABEL[d.layout]}
                  {fpsLabel(d) ? ` · ${fpsLabel(d)}` : ""}
                  {d.width && d.height ? ` · ${d.width}×${d.height}` : ""}
                </p>
                <div className="mt-2">
                  <CameraChips cameras={d.cameras} />
                </div>
              </div>
              <Button
                type="button"
                size="sm"
                className="flex-shrink-0"
                disabled={!d.available || loadingInput === d.inputDir}
                onClick={() => onLoad(d.inputDir, d.name)}
                aria-label={`Load ${d.name} videos into the gallery`}
              >
                {loadingInput === d.inputDir ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <FileVideo className="mr-2 h-4 w-4" />
                )}
                Load videos
              </Button>
            </div>
          </div>
        ))}
      </div>
    </ScrollArea>
  );
}

function FolderBrowser({
  onLoad,
  loadingInput,
}: {
  onLoad: (inputDir: string, name: string) => void;
  loadingInput: string | null;
}) {
  const [path, setPath] = useState("");
  const [result, setResult] = useState<BrowseResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (p: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = await browseDatasetFolder(p);
      if (res.success && res.data) setResult(res.data);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load(path);
  }, [path, load]);

  const crumbs = path ? path.split("/") : [];

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <div className="flex min-w-0 flex-1 items-center gap-1 overflow-x-auto rounded-md border bg-muted/40 px-2 py-1 text-xs">
          <button
            type="button"
            className="flex items-center gap-1 font-medium hover:text-primary"
            onClick={() => setPath("")}
          >
            <Home className="h-3.5 w-3.5" />
            data/raw
          </button>
          {crumbs.map((c, i) => (
            <span key={i} className="flex items-center gap-1">
              <ChevronRight className="h-3 w-3 text-muted-foreground" />
              <button
                type="button"
                className="hover:text-primary"
                onClick={() => setPath(crumbs.slice(0, i + 1).join("/"))}
              >
                {c}
              </button>
            </span>
          ))}
        </div>
        <Button type="button" variant="outline" size="icon" onClick={() => void load(path)} aria-label="Refresh">
          <RefreshCw className="h-4 w-4" />
        </Button>
      </div>

      {error ? <ErrorBanner title="Browse error" message={error} /> : null}

      {result?.datasetLike ? (
        <div className="flex items-center justify-between gap-3 rounded-md border border-success/30 bg-success/10 p-3">
          <div className="min-w-0">
            <p className="text-sm font-medium text-success">
              This folder looks like a dataset ({result.layout}, {result.cameras.length} cameras)
            </p>
            <div className="mt-1">
              <CameraChips cameras={result.cameras} />
            </div>
          </div>
          <Button
            type="button"
            size="sm"
            className="flex-shrink-0"
            disabled={loadingInput === result.inputDir}
            onClick={() => onLoad(result.inputDir, result.path || "data/raw")}
            aria-label="Load this folder's videos into the gallery"
          >
            {loadingInput === result.inputDir ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <FileVideo className="mr-2 h-4 w-4" />
            )}
            Load videos
          </Button>
        </div>
      ) : null}

      <ScrollArea className="h-[260px] rounded-md border">
        {loading ? (
          <div className="flex h-40 items-center justify-center text-muted-foreground">
            <Loader2 className="mr-2 h-5 w-5 animate-spin" /> Loading...
          </div>
        ) : !result || result.entries.length === 0 ? (
          <p className="py-8 text-center text-sm text-muted-foreground">Empty folder.</p>
        ) : (
          <div className="divide-y">
            {result.entries.map((e) => (
              <div key={e.path} className="flex items-center gap-2 px-3 py-2 text-sm">
                {e.type === "dir" ? (
                  <button
                    type="button"
                    className="flex min-w-0 flex-1 items-center gap-2 text-left hover:text-primary"
                    onClick={() => setPath(e.path)}
                  >
                    <Folder className="h-4 w-4 flex-shrink-0 text-muted-foreground" />
                    <span className="truncate">{e.name}</span>
                  </button>
                ) : (
                  <span className="flex min-w-0 flex-1 items-center gap-2 text-muted-foreground">
                    <FileVideo className="h-4 w-4 flex-shrink-0" />
                    <span className="truncate">{e.name}</span>
                  </span>
                )}
              </div>
            ))}
          </div>
        )}
      </ScrollArea>
    </div>
  );
}
