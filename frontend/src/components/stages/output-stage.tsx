"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Film,
  Grid,
  Map as MapIcon,
  Download,
  Play,
  Pause,
  BarChart3,
  Clock,
  Camera,
  Gauge,
  Maximize2,
  Car,
  Truck,
  Bus,
  CheckCircle2,
  TrendingUp,
  Route,
} from "lucide-react";
import { getCameraColor, formatDuration } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Label } from "@/components/ui/label";
import { usePipelineStore, useVideoStore } from "@/store";
import {
  exportTrajectories,
  generateSummaryVideo,
  getTracklets,
  getTrajectories,
} from "@/lib/api";
import type { GlobalTrajectory } from "@/types";

const outputPalette = ["#22c55e", "#3b82f6", "#f97316", "#e11d48", "#06b6d4", "#8b5cf6", "#f59e0b"];

interface OutputTrajectory {
  id: number;
  vehicleId: string;
  cameras: string[];
  duration: number;
  vehicleType: string;
  confidence: number;
  color: string;
}

function trajectoryFromGlobal(item: GlobalTrajectory, index: number): OutputTrajectory {
  const cameras = item.cameraSequence?.length
    ? item.cameraSequence
    : item.timeline?.map((entry) => entry.cameraId) ?? [];

  return {
    id: Number(item.globalId ?? index + 1),
    vehicleId: `G-${String(item.globalId ?? index + 1).padStart(4, "0")}`,
    cameras: Array.from(new Set(cameras)),
    duration: item.totalDuration ?? Math.max(0, (item.timeSpan?.[1] ?? 0) - (item.timeSpan?.[0] ?? 0)),
    vehicleType: item.className || "sedan",
    confidence: item.confidence ?? 0.8,
    color: outputPalette[index % outputPalette.length],
  };
}

function trajectoryFromTrackletSummary(item: any, index: number): OutputTrajectory {
  return {
    id: Number(item.id ?? index + 1),
    vehicleId: `T-${String(item.id ?? index + 1).padStart(4, "0")}`,
    cameras: [String(item.cameraId ?? "unknown")],
    duration: Number(item.duration ?? 0),
    vehicleType: String(item.className ?? "sedan"),
    confidence: Number(item.confidence ?? 0.8),
    color: outputPalette[index % outputPalette.length],
  };
}

export function OutputStage() {
  const [activeTab, setActiveTab] = useState("video");
  const [gridSize, setGridSize] = useState(3);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackProgress, setPlaybackProgress] = useState(33);
  const [trajectories, setTrajectories] = useState<OutputTrajectory[]>([]);
  const [dataSource, setDataSource] = useState<"real" | "none">("none");
  const [error, setError] = useState<string | null>(null);
  const [exportFormat, setExportFormat] = useState<"mp4" | "json" | "csv">("json");
  const [isExporting, setIsExporting] = useState(false);

  const { runId } = usePipelineStore();
  const { currentVideo } = useVideoStore();
  const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";
  const backendOrigin = apiBase.endsWith("/api") ? apiBase.slice(0, -4) : apiBase;
  const streamUrl = currentVideo ? `${apiBase}/videos/stream/${currentVideo.id}` : null;

  const toAbsoluteUrl = (url: string) => {
    if (url.startsWith("http://") || url.startsWith("https://")) return url;
    if (url.startsWith("/")) return `${backendOrigin}${url}`;
    return `${backendOrigin}/${url}`;
  };

  const handleDownloadVideo = async () => {
    if (isExporting) return;

    if (!runId) {
      if (streamUrl) {
        window.open(streamUrl, "_blank", "noopener,noreferrer");
      }
      return;
    }

    try {
      setIsExporting(true);
      const response = await generateSummaryVideo(runId);
      const videoUrl = response.data?.videoUrl;
      if (videoUrl) {
        window.open(toAbsoluteUrl(videoUrl), "_blank", "noopener,noreferrer");
      }
    } finally {
      setIsExporting(false);
    }
  };

  const handleExportTracklets = async () => {
    if (!runId || isExporting) return;

    const fmt = exportFormat === "mp4" ? "json" : exportFormat;
    try {
      setIsExporting(true);
      const response = await exportTrajectories(runId, fmt);
      const downloadUrl = response.data?.downloadUrl;
      if (downloadUrl) {
        window.open(toAbsoluteUrl(downloadUrl), "_blank", "noopener,noreferrer");
      }
    } finally {
      setIsExporting(false);
    }
  };

  useEffect(() => {
    let cancelled = false;

    const loadOutputData = async () => {
      if (!currentVideo) {
        setTrajectories([]);
        setDataSource("none");
        return;
      }

      try {
        if (runId) {
          const trajResponse = await getTrajectories(runId);
          const globalTrajectories = Array.isArray(trajResponse.data) ? trajResponse.data : [];
          if (!cancelled && globalTrajectories.length > 0) {
            setTrajectories(globalTrajectories.map(trajectoryFromGlobal));
            setDataSource("real");
            return;
          }
        }

        const trackletResponse = await getTracklets(undefined, currentVideo.id);
        const summary = Array.isArray(trackletResponse.data) ? trackletResponse.data : [];
        if (!cancelled && summary.length > 0) {
          setTrajectories(summary.map(trajectoryFromTrackletSummary));
          setDataSource("real");
          return;
        }
      } catch (err) {
        if (!cancelled) {
          setError(String(err instanceof Error ? err.message : err || "Failed to load trajectory data"));
          setTrajectories([]);
          setDataSource("none");
        }
        return;
      }

      if (!cancelled) {
        setTrajectories([]);
        setDataSource("none");
      }
    };

    void loadOutputData();

    return () => {
      cancelled = true;
    };
  }, [currentVideo, runId]);

  const outputStats = useMemo(() => {
    const cameras = new Set<string>();
    let crossCamera = 0;
    trajectories.forEach((traj) => {
      traj.cameras.forEach((cam) => cameras.add(cam));
      if (traj.cameras.length > 1) crossCamera += 1;
    });

    const meanConfidence =
      trajectories.length > 0
        ? trajectories.reduce((sum, traj) => sum + traj.confidence, 0) / trajectories.length
        : 0;

    const totalFrames = Math.max(
      0,
      Math.round((currentVideo?.duration ?? 0) * Math.max(currentVideo?.fps ?? 10, 1))
    );

    return {
      camerasAnalyzed: cameras.size,
      uniqueVehicles: trajectories.length,
      crossCameraMatches: crossCamera,
      reIdAccuracy: meanConfidence,
      totalFrames,
      processingTime: Math.max(1, Math.round((currentVideo?.duration ?? 60) * 0.35)),
    };
  }, [currentVideo, dataSource, trajectories]);

  const camerasForGrid = useMemo(() => {
    const ids = Array.from(new Set(trajectories.flatMap((traj) => traj.cameras)));
    if (ids.length === 0) return [];

    return ids.map((id) => ({
      id,
      scene: id.split("_")[0] ?? "Custom",
      name: id,
      location: "Camera",
    }));
  }, [trajectories]);

  const cameraDistribution = useMemo(() => {
    const counts = new Map<string, number>();
    trajectories.forEach((traj) => {
      traj.cameras.forEach((cam) => {
        counts.set(cam, (counts.get(cam) ?? 0) + 1);
      });
    });

    const entries = Array.from(counts.entries());
    const total = entries.reduce((sum, [, count]) => sum + count, 0);

    if (entries.length === 0) {
      return [];
    }

    return entries
      .map(([id, count]) => ({ id, percentage: total > 0 ? (count / total) * 100 : 0 }))
      .sort((a, b) => b.percentage - a.percentage);
  }, [trajectories]);

  const vehicleBreakdown = useMemo(() => {
    if (trajectories.length === 0) {
      return [
        { name: "Sedan/Car", count: 0, color: "#22c55e", icon: Car },
        { name: "SUV", count: 0, color: "#3b82f6", icon: Car },
        { name: "Truck", count: 0, color: "#8b5cf6", icon: Truck },
        { name: "Bus", count: 0, color: "#f97316", icon: Bus },
      ];
    }

    const counts = new Map<string, number>();
    trajectories.forEach((traj) => {
      const type = traj.vehicleType.toLowerCase();
      if (type.includes("truck")) {
        counts.set("Truck", (counts.get("Truck") ?? 0) + 1);
      } else if (type.includes("bus")) {
        counts.set("Bus", (counts.get("Bus") ?? 0) + 1);
      } else if (type.includes("suv")) {
        counts.set("SUV", (counts.get("SUV") ?? 0) + 1);
      } else {
        counts.set("Sedan/Car", (counts.get("Sedan/Car") ?? 0) + 1);
      }
    });

    return [
      { name: "Sedan/Car", count: counts.get("Sedan/Car") ?? 0, color: "#22c55e", icon: Car },
      { name: "SUV", count: counts.get("SUV") ?? 0, color: "#3b82f6", icon: Car },
      { name: "Truck", count: counts.get("Truck") ?? 0, color: "#8b5cf6", icon: Truck },
      { name: "Bus", count: counts.get("Bus") ?? 0, color: "#f97316", icon: Bus },
    ];
  }, [trajectories]);

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden">
      {/* Header */}
      <header className="flex shrink-0 flex-col gap-3 border-b px-4 py-3 sm:flex-row sm:items-center sm:justify-between sm:px-6">
        <div className="min-w-0">
          <h1 className="text-lg font-semibold">Stage 6: Results & Export</h1>
          <p className="text-sm text-muted-foreground">
            CityFlowV2 tracking results visualization
          </p>
        </div>
        <div className="flex flex-wrap items-center justify-end gap-2">
          <Badge variant={dataSource === "real" ? "secondary" : "destructive"}>
            {dataSource === "real" ? "Real Artifacts" : "No Data"}
          </Badge>
          <Badge variant="outline" className="bg-green-500/10 text-green-500 border-green-500/30">
            <CheckCircle2 className="h-3 w-3 mr-1" />
            Pipeline Complete
          </Badge>
          <Button className="shrink-0" variant="outline" onClick={handleExportTracklets} disabled={!runId || isExporting}>
            <Download className="mr-2 h-4 w-4" />
            Export Results
          </Button>
        </div>
      </header>

      {/* Error banner */}
      {error && (
        <div className="flex shrink-0 items-start gap-3 overflow-x-auto border-b border-destructive/30 bg-destructive/10 px-4 py-3 sm:px-6">
          <Route className="mt-0.5 h-5 w-5 shrink-0 text-destructive" />
          <div className="min-w-0">
            <p className="text-sm font-medium text-destructive">Failed to load output data</p>
            <p className="break-words text-xs text-muted-foreground">{error}</p>
          </div>
        </div>
      )}

      {/* Main content with tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
        <div className="shrink-0 overflow-x-auto border-b px-4 sm:px-6">
          <TabsList className="inline-flex h-auto min-h-12 w-max max-w-full flex-wrap gap-1 py-2">
            <TabsTrigger value="video" className="gap-2">
              <Film className="h-4 w-4" />
              Summarized Video
            </TabsTrigger>
            <TabsTrigger value="grid" className="gap-2">
              <Grid className="h-4 w-4" />
              Camera Grid
            </TabsTrigger>
            <TabsTrigger value="map" className="gap-2" disabled>
              <MapIcon className="h-4 w-4" />
              Map (Coming Soon)
            </TabsTrigger>
            <TabsTrigger value="stats" className="gap-2">
              <BarChart3 className="h-4 w-4" />
              Statistics
            </TabsTrigger>
          </TabsList>
        </div>

        {/* Video tab */}
        <TabsContent value="video" className="m-0 flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden lg:flex-row">
          <div className="min-h-0 min-w-0 flex-1 overflow-y-auto overflow-x-hidden p-4">
            {/* Video player */}
            <div className="relative aspect-video bg-slate-900 rounded-lg overflow-hidden mb-4 border border-border">
              {streamUrl ? (
                <video
                  key={streamUrl}
                  src={streamUrl}
                  className="h-full w-full object-cover"
                  autoPlay
                  muted
                  loop
                  playsInline
                  controls={false}
                />
              ) : (
                <div className="absolute inset-0 bg-gradient-to-b from-slate-700 via-slate-800 to-slate-900">
                  <svg className="absolute inset-0 w-full h-full opacity-30" preserveAspectRatio="none">
                    <line x1="50%" y1="30%" x2="15%" y2="100%" stroke="white" strokeWidth="2" />
                    <line x1="50%" y1="30%" x2="85%" y2="100%" stroke="white" strokeWidth="2" />
                    <line x1="50%" y1="30%" x2="50%" y2="100%" stroke="#fbbf24" strokeWidth="2" strokeDasharray="10,8" />
                  </svg>
                </div>
              )}

              <div className="absolute inset-0 pointer-events-none">
                {trajectories.slice(0, 3).map((traj, idx) => (
                  <div
                    key={traj.id}
                    className="absolute flex items-center justify-center animate-pulse"
                    style={{
                      top: `${40 + idx * 15}%`,
                      left: `${25 + idx * 20}%`,
                    }}
                  >
                    <div
                      className="w-16 h-10 rounded border-2 flex items-center justify-center"
                      style={{ borderColor: traj.color, backgroundColor: `${traj.color}33` }}
                    >
                      {traj.vehicleType === "sedan" || traj.vehicleType === "SUV" ? (
                        <Car className="h-5 w-5 text-white" />
                      ) : traj.vehicleType === "truck" ? (
                        <Truck className="h-5 w-5 text-white" />
                      ) : (
                        <Bus className="h-5 w-5 text-white" />
                      )}
                    </div>
                    <Badge
                      variant="secondary"
                      className="absolute -top-5 text-[9px] whitespace-nowrap"
                      style={{ backgroundColor: traj.color, color: "white" }}
                    >
                      {traj.vehicleId}
                    </Badge>
                  </div>
                ))}

                {/* Camera info overlay */}
                <div className="absolute top-3 left-3 flex items-center gap-2">
                  <div className="flex items-center gap-1.5 bg-black/60 rounded px-2 py-1">
                    <div className="h-2 w-2 rounded-full bg-red-500 animate-pulse" />
                    <span className="text-white text-xs font-mono">Multi-Camera Summary</span>
                  </div>
                </div>

                {/* Tracking info */}
                <div className="absolute top-3 right-3 bg-black/60 rounded px-2 py-1">
                  <span className="text-white/80 text-xs font-mono">
                    {outputStats.uniqueVehicles} Vehicles Tracked
                  </span>
                </div>
              </div>

              {/* Controls overlay */}
              <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4">
                <div className="flex items-center gap-4">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="text-white hover:text-white hover:bg-white/20"
                    onClick={() => setIsPlaying(!isPlaying)}
                  >
                    {isPlaying ? <Pause className="h-5 w-5" /> : <Play className="h-5 w-5" />}
                  </Button>
                  <div className="flex-1 h-1.5 bg-white/30 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-primary rounded-full transition-all"
                      style={{ width: `${playbackProgress}%` }}
                    />
                  </div>
                  <span className="text-white text-sm font-mono">0:45 / 2:07</span>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="text-white hover:text-white hover:bg-white/20"
                  >
                    <Maximize2 className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </div>

            {/* Trajectory list */}
            <Card>
              <CardHeader className="py-3">
                <CardTitle className="text-base flex items-center gap-2">
                  <Route className="h-4 w-4" />
                  Tracked Vehicle Trajectories
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {trajectories.map((traj) => (
                    <TrajectoryItem key={traj.id} trajectory={traj} />
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Sidebar */}
          <aside className="w-full shrink-0 border-t border-border bg-muted/20 p-4 lg:w-72 lg:border-l lg:border-t-0">
            <h3 className="mb-4 font-semibold">Export Settings</h3>
            <div className="space-y-6">
              <div className="space-y-2">
                <Label>Playback Speed</Label>
                <select className="w-full bg-background border rounded px-3 py-2 text-sm">
                  <option value="1">1x (Normal)</option>
                  <option value="2">2x (Fast)</option>
                  <option value="4">4x (Very Fast)</option>
                  <option value="0.5">0.5x (Slow)</option>
                </select>
              </div>
              <div className="space-y-2">
                <Label>Export Format</Label>
                <select
                  className="w-full bg-background border rounded px-3 py-2 text-sm"
                  value={exportFormat}
                  onChange={(e) => setExportFormat(e.target.value as "mp4" | "json" | "csv")}
                >
                  <option value="mp4">MP4 Video</option>
                  <option value="json">JSON Tracklets</option>
                  <option value="csv">CSV Export</option>
                </select>
              </div>
              <Separator />
              <div className="space-y-3">
                <Button className="w-full" onClick={handleDownloadVideo} disabled={isExporting}>
                  <Download className="mr-2 h-4 w-4" />
                  Download Video
                </Button>
                <Button
                  variant="outline"
                  className="w-full"
                  onClick={handleExportTracklets}
                  disabled={!runId || isExporting}
                >
                  <Download className="mr-2 h-4 w-4" />
                  Export Tracklets
                </Button>
              </div>
            </div>
          </aside>
        </TabsContent>

        {/* Grid tab */}
        <TabsContent value="grid" className="m-0 flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden lg:flex-row">
          <div className="min-h-0 min-w-0 flex-1 overflow-x-auto overflow-y-auto p-4">
            <div
              className="grid min-w-0 gap-2"
              style={{ gridTemplateColumns: `repeat(${gridSize}, minmax(0, 1fr))` }}
            >
              {camerasForGrid.slice(0, gridSize * Math.ceil(camerasForGrid.length / gridSize)).map((cam, i) => (
                <GridCell key={cam.id} camera={cam} index={i} />
              ))}
            </div>
          </div>

          {/* Sidebar */}
          <aside className="w-full shrink-0 border-t border-border bg-muted/20 p-4 lg:w-72 lg:border-l lg:border-t-0">
            <h3 className="mb-4 font-semibold">Grid Settings</h3>
            <div className="space-y-6">
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label>Columns</Label>
                  <span className="text-sm text-muted-foreground">{gridSize}</span>
                </div>
                <Slider
                  value={[gridSize]}
                  min={1}
                  max={4}
                  step={1}
                  onValueChange={(v) => setGridSize(v[0])}
                />
              </div>
              <div className="space-y-2">
                <Label>Sort By</Label>
                <select className="w-full bg-background border rounded px-3 py-2 text-sm">
                  <option value="scene">Scene</option>
                  <option value="timestamp">Timestamp</option>
                  <option value="detections">Detection Count</option>
                </select>
              </div>
              <Separator />
              <div className="text-sm text-muted-foreground">
                Showing {camerasForGrid.length} cameras from stage artifacts
              </div>
            </div>
          </aside>
        </TabsContent>

        {/* Map tab (future) */}
        <TabsContent value="map" className="m-0 flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
          <div className="min-h-0 flex-1 overflow-auto p-4">
            <Card className="flex h-full min-h-[200px] items-center justify-center">
              <div className="text-center">
                <MapIcon className="h-16 w-16 mx-auto mb-4 text-muted-foreground/50" />
                <h3 className="text-lg font-semibold mb-2">Map View Coming Soon</h3>
                <p className="text-muted-foreground max-w-md">
                  Interactive 2D map with vehicle paths, heatmaps, and spatiotemporal
                  constraints will be available when GPS data is acquired.
                </p>
              </div>
            </Card>
          </div>
        </TabsContent>

        {/* Stats tab */}
        <TabsContent value="stats" className="m-0 min-h-0 flex-1 overflow-x-hidden overflow-y-auto p-4 sm:p-6">
          <div className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard
              icon={Camera}
              label="Cameras"
              value={String(outputStats.camerasAnalyzed)}
              subtext="CityFlowV2 cameras"
            />
            <StatCard
              icon={Car}
              label="Vehicles"
              value={String(outputStats.uniqueVehicles)}
              subtext="unique tracked"
            />
            <StatCard
              icon={TrendingUp}
              label="Cross-Camera"
              value={String(outputStats.crossCameraMatches)}
              subtext="associations"
            />
            <StatCard
              icon={Gauge}
              label="Accuracy"
              value={`${(outputStats.reIdAccuracy * 100).toFixed(0)}%`}
              subtext="ReID confidence"
            />
          </div>

          <div className="grid grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Camera className="h-4 w-4" />
                  Camera Detection Distribution
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {cameraDistribution.map((entry) => {
                    const percentage = entry.percentage;
                    return (
                      <div key={entry.id} className="flex items-center gap-3">
                        <div
                          className="h-3 w-3 rounded-full flex-shrink-0"
                          style={{ backgroundColor: getCameraColor(entry.id) }}
                        />
                        <span className="flex-1 text-sm font-mono">{entry.id}</span>
                        <div className="w-32 h-2 bg-muted rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all"
                            style={{
                              width: `${percentage}%`,
                              backgroundColor: getCameraColor(entry.id),
                            }}
                          />
                        </div>
                        <span className="text-sm text-muted-foreground w-10 text-right">
                          {percentage.toFixed(0)}%
                        </span>
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Car className="h-4 w-4" />
                  Vehicle Type Breakdown
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {vehicleBreakdown.map((cls) => (
                    <div key={cls.name} className="flex items-center gap-3">
                      <div
                        className="h-8 w-8 rounded flex items-center justify-center"
                        style={{ backgroundColor: `${cls.color}20` }}
                      >
                        <cls.icon className="h-4 w-4" style={{ color: cls.color }} />
                      </div>
                      <span className="flex-1 text-sm">{cls.name}</span>
                      <Badge variant="secondary">{cls.count}</Badge>
                    </div>
                  ))}
                </div>

                <Separator className="my-4" />

                <div className="grid grid-cols-2 gap-4 text-center">
                  <div>
                    <p className="text-2xl font-bold">{outputStats.totalFrames.toLocaleString()}</p>
                    <p className="text-xs text-muted-foreground">Total Frames</p>
                  </div>
                  <div>
                    <p className="text-2xl font-bold">{outputStats.processingTime}s</p>
                    <p className="text-xs text-muted-foreground">Processing Time</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}

function TrajectoryItem({
  trajectory,
}: {
  trajectory: {
    id: number;
    vehicleId: string;
    cameras: string[];
    duration: number;
    vehicleType: string;
    confidence: number;
    color: string;
  };
}) {
  return (
    <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50 hover:bg-muted transition-colors">
      <div
        className="h-10 w-10 rounded flex items-center justify-center"
        style={{ backgroundColor: `${trajectory.color}20`, color: trajectory.color }}
      >
        {trajectory.vehicleType === "sedan" || trajectory.vehicleType === "SUV" ? (
          <Car className="h-5 w-5" />
        ) : trajectory.vehicleType === "truck" ? (
          <Truck className="h-5 w-5" />
        ) : (
          <Bus className="h-5 w-5" />
        )}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="font-medium">{trajectory.vehicleId}</span>
          <Badge variant="secondary" className="text-[10px]">
            {(trajectory.confidence * 100).toFixed(0)}%
          </Badge>
        </div>
        <div className="flex items-center gap-1 mt-1">
          {trajectory.cameras.map((cam, i) => (
            <div key={`${cam}-${i}`} className="flex items-center">
              <div
                className="h-2 w-2 rounded-full"
                style={{ backgroundColor: getCameraColor(cam) }}
              />
              {i < trajectory.cameras.length - 1 && (
                <span className="text-muted-foreground mx-1">→</span>
              )}
            </div>
          ))}
          <span className="text-xs text-muted-foreground ml-2">
            {trajectory.cameras.length} cameras
          </span>
        </div>
      </div>
      <div className="text-right">
        <p className="text-sm font-mono">{formatDuration(trajectory.duration)}</p>
        <p className="text-xs text-muted-foreground">duration</p>
      </div>
    </div>
  );
}

function GridCell({ camera, index }: { camera: { id: string; name: string; location: string; scene: string }; index: number }) {
  const detectionCount = [12, 8, 15, 6, 9, 4][index % 6];

  return (
    <div className="relative aspect-video rounded-lg overflow-hidden border border-border group cursor-pointer hover:ring-2 hover:ring-primary transition-all">
      {/* Simulated camera feed */}
      <div className="absolute inset-0 bg-gradient-to-b from-slate-700 via-slate-800 to-slate-900">
        {/* Road markings */}
        <svg className="absolute inset-0 w-full h-full opacity-20" preserveAspectRatio="none">
          <line x1="50%" y1="25%" x2="20%" y2="100%" stroke="white" strokeWidth="1" />
          <line x1="50%" y1="25%" x2="80%" y2="100%" stroke="white" strokeWidth="1" />
        </svg>

        {/* Vehicle indicator */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/4">
          <div
            className="w-8 h-5 rounded border flex items-center justify-center"
            style={{ borderColor: getCameraColor(camera.id), backgroundColor: `${getCameraColor(camera.id)}33` }}
          >
            <Car className="h-3 w-3 text-white" />
          </div>
        </div>
      </div>

      {/* Camera info overlay - top */}
      <div className="absolute top-0 left-0 right-0 p-2 bg-gradient-to-b from-black/70 to-transparent">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <div className="h-1.5 w-1.5 rounded-full bg-green-500 animate-pulse" />
            <span className="text-white text-[10px] font-mono">{camera.id}</span>
          </div>
          <Badge variant="secondary" className="text-[9px] h-4">
            {detectionCount} det
          </Badge>
        </div>
      </div>

      {/* Hover overlay */}
      <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
        <div className="text-center text-white">
          <p className="text-sm font-medium">{camera.location}</p>
          <p className="text-xs text-white/60">{camera.scene}</p>
        </div>
      </div>
    </div>
  );
}

function StatCard({
  icon: Icon,
  label,
  value,
  subtext,
}: {
  icon: React.ElementType;
  label: string;
  value: string;
  subtext: string;
}) {
  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center">
            <Icon className="h-5 w-5 text-primary" />
          </div>
          <div>
            <p className="text-2xl font-bold">{value}</p>
            <p className="text-xs text-muted-foreground">{subtext}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
