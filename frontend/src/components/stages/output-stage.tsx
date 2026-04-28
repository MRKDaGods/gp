"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { LatLngTuple } from "leaflet";
import dynamic from "next/dynamic";
import {
  Download,
  Play,
  Pause,
  Camera,
  Maximize2,
  Car,
  Truck,
  Bus,
  CheckCircle2,
  Loader2,
  TrendingUp,
  Route,
  Gauge,
} from "lucide-react";
import { getCameraColor, formatDuration } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Separator } from "@/components/ui/separator";
import { Label } from "@/components/ui/label";
import { usePipelineStore, useTimelineStore, useVideoStore } from "@/store";
import type { CameraMapCoordinateEntry } from "@/lib/api";
import type { TimelineTrack } from "@/types";
import {
  exportTrajectories,
  generateSummaryVideo,
  getDatasets,
  getMatchedSummary,
  getTracklets,
  getTrajectories,
  getVideo,
} from "@/lib/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8004/api";
const outputPalette = ["#22c55e", "#3b82f6", "#f97316", "#e11d48", "#06b6d4", "#8b5cf6", "#f59e0b"];

const VehiclePathMap = dynamic(() => import("@/components/maps/vehicle-path-map"), { ssr: false });

type CameraCoord = { lat: number; lng: number; label: string };

type MapPathPoint = CameraCoord & { cameraId: string; sequence: number; tooltip: string };

type CoordRegistry = Record<string, CameraMapCoordinateEntry>;

function normalizeCameraId(raw: string): string | null {
  const text = String(raw ?? "").trim();
  if (!text) return null;
  const matches = text.match(/\d+/g);
  if (!matches || matches.length === 0) return text;
  const num = Number(matches[matches.length - 1]);
  if (!Number.isFinite(num)) return text;
  return String(num);
}

function formatCameraLabel(raw: string): string {
  const normalized = normalizeCameraId(raw);
  if (!normalized) return String(raw);
  if (/^\d+$/.test(normalized)) return `Camera ${normalized}`;
  return normalized;
}

function resolveCameraCoord(registry: CoordRegistry, cameraId: string): CameraCoord | null {
  const raw = String(cameraId ?? "").trim();
  if (!raw) return null;
  const normalized = normalizeCameraId(cameraId);
  const entry =
    (normalized ? registry[normalized] : undefined) ?? registry[raw] ?? undefined;
  if (!entry || !Number.isFinite(entry.lat) || !Number.isFinite(entry.lng)) return null;
  return {
    lat: entry.lat,
    lng: entry.lng,
    label:
      typeof entry.label === "string" && entry.label.trim()
        ? entry.label.trim()
        : formatCameraLabel(cameraId),
  };
}

function coordRegistryEntries(registry: CoordRegistry): Array<{ cameraId: string; lat: number; lng: number; label: string }> {
  return Object.entries(registry).map(([cameraId, c]) => ({
    cameraId,
    lat: c.lat,
    lng: c.lng,
    label:
      typeof c.label === "string" && c.label.trim()
        ? c.label.trim()
        : formatCameraLabel(cameraId),
  }));
}

function normalizeCameraList(list: unknown[]): string[] {
  return list.map((item) => String(item).trim()).filter(Boolean);
}

function extractCameraSequence(item: Record<string, unknown>): string[] {
  const camSeq = item.cameraSequence ?? item.camera_sequence;
  if (Array.isArray(camSeq) && camSeq.length) return normalizeCameraList(camSeq);

  const timeline = item.timeline as Array<Record<string, unknown>> | undefined;
  if (Array.isArray(timeline) && timeline.length) {
    const sorted = [...timeline].sort((a, b) => {
      const aStart = Number(a.start ?? a.start_time ?? a.startTime ?? 0);
      const bStart = Number(b.start ?? b.start_time ?? b.startTime ?? 0);
      return aStart - bStart;
    });
    return normalizeCameraList(sorted.map((entry) => entry.cameraId ?? entry.camera_id ?? ""));
  }

  const tracklets = item.tracklets as Array<Record<string, unknown>> | undefined;
  if (Array.isArray(tracklets) && tracklets.length) {
    const sorted = [...tracklets].sort((a, b) => {
      const aStart = Number(a.startTime ?? a.start_time ?? 0);
      const bStart = Number(b.startTime ?? b.start_time ?? 0);
      return aStart - bStart;
    });
    return normalizeCameraList(sorted.map((t) => t.cameraId ?? t.camera_id ?? ""));
  }

  const visited = item.cameras_visited;
  if (Array.isArray(visited) && visited.length) return normalizeCameraList(visited);

  return [];
}

function computeCenter(points: Array<{ lat: number; lng: number }>): LatLngTuple {
  if (!points.length) return [0, 0];
  const totals = points.reduce(
    (acc, point) => ({ lat: acc.lat + point.lat, lng: acc.lng + point.lng }),
    { lat: 0, lng: 0 }
  );
  return [totals.lat / points.length, totals.lng / points.length];
}

interface OutputTrajectory {
  id: number;
  vehicleId: string;
  cameras: string[];
  cameraSequence: string[];
  duration: number;
  vehicleType: string;
  confidence: number;
  color: string;
}

function cameraKeyForMatch(cam: string): string {
  return normalizeCameraId(cam) ?? String(cam).trim();
}

/** True if this timeline row is the same vehicle identity as traj (global id or probe tracklet). */
function trajectoryOwnsRow(traj: OutputTrajectory, row: TimelineTrack): boolean {
  if (row.globalId != null && Number.isFinite(Number(row.globalId))) {
    return Number(row.globalId) === traj.id;
  }
  if (traj.vehicleId.startsWith("T-") && traj.cameras.length >= 1) {
    const rc = cameraKeyForMatch(row.cameraId);
    const tc = cameraKeyForMatch(traj.cameras[0]);
    return Number(row.trackletId) === traj.id && rc === tc;
  }
  return false;
}

/**
 * Timeline has one checkbox per camera clip. Trim each trajectory's camera path to
 * **confirmed** segments only so unchecking a camera removes it from output (path, map, counts).
 */
function applyTimelineSelectionToTrajectories(
  all: OutputTrajectory[],
  tracks: TimelineTrack[],
  timelineFilterEngaged: boolean
): OutputTrajectory[] {
  if (!timelineFilterEngaged || tracks.length === 0) return all;

  const confirmedRows = tracks.filter((t) => t.confirmed);
  if (confirmedRows.length === 0) return [];

  const out: OutputTrajectory[] = [];

  for (const traj of all) {
    const rowsForTraj = confirmedRows.filter((row) => trajectoryOwnsRow(traj, row));
    if (rowsForTraj.length === 0) continue;

    const allowedKeys = new Set(rowsForTraj.map((row) => cameraKeyForMatch(row.cameraId)));

    const newSequence = traj.cameraSequence.filter((c) => allowedKeys.has(cameraKeyForMatch(c)));
    if (newSequence.length === 0) continue;

    out.push({
      ...traj,
      cameraSequence: newSequence,
      cameras: Array.from(new Set(newSequence)),
    });
  }

  return out;
}

/** Clip keys for stitched summary video API — must match matched/summary.json (camera + track). */
function buildIncludeClipsForSummaryApi(
  tracks: TimelineTrack[],
  engaged: boolean
): { camera_id: string; track_id: number }[] | undefined {
  if (!engaged || tracks.length === 0) return undefined;
  const rows = tracks.filter((t) => t.confirmed);
  if (rows.length === 0) return [];
  return rows.map((r) => ({
    camera_id: String(r.cameraId ?? ""),
    track_id: Number(r.trackletId),
  }));
}

function trajectoryFromGlobal(item: Record<string, unknown>, index: number): OutputTrajectory {
  const gid = Number(item.global_id ?? item.globalId ?? index + 1);

  const cameraSequence = extractCameraSequence(item);
  const cameras = Array.from(new Set(cameraSequence));

  const span = (item.timeSpan ?? item.time_span) as [number, number] | undefined;
  let duration = Number(item.totalDuration ?? item.total_duration ?? NaN);
  if (!Number.isFinite(duration) && span && span.length >= 2) {
    duration = Math.max(0, Number(span[1]) - Number(span[0]));
  }
  if (!Number.isFinite(duration) && Array.isArray(item.tracklets)) {
    const tl = item.tracklets as { start_time?: number; end_time?: number; startTime?: number; endTime?: number }[];
    const times = tl.flatMap((t) =>
      [t.start_time, t.end_time, t.startTime, t.endTime].filter((x): x is number => typeof x === "number")
    );
    if (times.length) duration = Math.max(...times) - Math.min(...times);
  }
  if (!Number.isFinite(duration)) duration = 0;

  const t0 = item.tracklets as { class_name?: string; className?: string }[] | undefined;
  const vehicleType = String(
    item.className ?? item.class ?? item.class_name ?? t0?.[0]?.className ?? t0?.[0]?.class_name ?? "sedan"
  ).toLowerCase();

  const confidence = Number(item.confidence ?? 0.8);

  return {
    id: gid,
    vehicleId: `G-${String(gid).padStart(4, "0")}`,
    cameras,
    cameraSequence,
    duration,
    vehicleType,
    confidence: Number.isFinite(confidence) ? confidence : 0.8,
    color: outputPalette[index % outputPalette.length],
  };
}

function trajectoryFromTrackletSummary(item: any, index: number): OutputTrajectory {
  const cameraId = String(item.cameraId ?? item.camera_id ?? "unknown");
  return {
    id: Number(item.id ?? index + 1),
    vehicleId: `T-${String(item.id ?? index + 1).padStart(4, "0")}`,
    cameras: [cameraId],
    cameraSequence: [cameraId],
    duration: Number(item.duration ?? 0),
    vehicleType: String(item.className ?? "sedan").toLowerCase(),
    confidence: Number(item.confidence ?? 0.8),
    color: outputPalette[index % outputPalette.length],
  };
}

function trajectoriesFromMatchedSummary(summary: any): OutputTrajectory[] {
  const clips = Array.isArray(summary?.clips)
    ? summary.clips.filter((c: any) => c && c.ok !== false)
    : [];
  if (clips.length === 0) return [];

  const grouped = new Map<number, any[]>();
  clips.forEach((clip: any, idx: number) => {
    const gid = Number(clip.global_id ?? clip.globalId ?? idx + 1);
    if (!Number.isFinite(gid)) return;
    const list = grouped.get(gid) ?? [];
    list.push(clip);
    grouped.set(gid, list);
  });

  const groups = Array.from(grouped.entries()).sort((a, b) => a[0] - b[0]);

  return groups.map(([gid, groupClips], index) => {
    const ordered = [...groupClips].sort((a, b) => {
      const aStart = Number(a.start_time_s ?? a.startTime ?? 0);
      const bStart = Number(b.start_time_s ?? b.startTime ?? 0);
      return aStart - bStart;
    });

    const cameraSequence = normalizeCameraList(
      ordered.map((clip: any) => clip.camera_id ?? clip.cameraId ?? "")
    );
    const cameras = Array.from(new Set(cameraSequence));

    const starts = ordered.map((clip: any) => Number(clip.start_time_s ?? clip.startTime ?? 0));
    const ends = ordered.map((clip: any) => Number(clip.end_time_s ?? clip.endTime ?? 0));
    const minStart = starts.length > 0 ? Math.min(...starts) : 0;
    const maxEnd = ends.length > 0 ? Math.max(...ends) : minStart;
    const duration = Math.max(0, maxEnd - minStart);

    const confs = ordered
      .map((clip: any) => Number(clip.confidence ?? 0))
      .filter((c: number) => Number.isFinite(c));
    const confidence = confs.length > 0
      ? confs.reduce((sum: number, v: number) => sum + v, 0) / confs.length
      : 0.8;

    return {
      id: gid,
      vehicleId: `G-${String(gid).padStart(4, "0")}`,
      cameras,
      cameraSequence,
      duration,
      vehicleType: "vehicle",
      confidence,
      color: outputPalette[index % outputPalette.length],
    };
  });
}

function formatClock(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function OutputStage() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTimeSec, setCurrentTimeSec] = useState(0);
  const [durationSec, setDurationSec] = useState(0);
  const [videoLoadError, setVideoLoadError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [trajectories, setTrajectories] = useState<OutputTrajectory[]>([]);
  const [selectedTrajectoryId, setSelectedTrajectoryId] = useState<number | null>(null);
  const [dataSource, setDataSource] = useState<"real" | "none">("none");
  const [error, setError] = useState<string | null>(null);
  const [exportFormat, setExportFormat] = useState<"mp4" | "json" | "csv">("mp4");
  const [isExporting, setIsExporting] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(2);
  const [hydratedLatestRunId, setHydratedLatestRunId] = useState<string | null>(null);
  const [outputFetchState, setOutputFetchState] = useState<"idle" | "loading" | "ready">("idle");
  const [summaryVideoUrl, setSummaryVideoUrl] = useState<string | null>(null);
  const [matchedSummary, setMatchedSummary] = useState<any>(null);

  const {
    runId,
    galleryRunId,
    mapCameraCoordinates,
    setMapCameraCoordinates,
  } = usePipelineStore();
  const { currentVideo } = useVideoStore();
  const { tracks: timelineTracks, timelineClipFilterEngaged } = useTimelineStore();

  const displayTrajectories = useMemo(
    () =>
      applyTimelineSelectionToTrajectories(
        trajectories,
        timelineTracks,
        timelineClipFilterEngaged
      ),
    [trajectories, timelineTracks, timelineClipFilterEngaged]
  );

  const summaryVideoPayload = useMemo(():
    | { includeClips: { camera_id: string; track_id: number }[] }
    | undefined => {
    const clips = buildIncludeClipsForSummaryApi(timelineTracks, timelineClipFilterEngaged);
    if (clips === undefined) return undefined;
    return { includeClips: clips };
  }, [timelineTracks, timelineClipFilterEngaged]);

  const summaryVideoRequestKey = useMemo(
    () => JSON.stringify(summaryVideoPayload ?? null),
    [summaryVideoPayload]
  );

  const effectiveRunId = useMemo(
    () => runId ?? currentVideo?.latestRunId ?? hydratedLatestRunId ?? null,
    [runId, currentVideo?.latestRunId, hydratedLatestRunId]
  );

  const backendOrigin = API_BASE.endsWith("/api") ? API_BASE.slice(0, -4) : API_BASE;
  const toAbsoluteUrl = (url: string) => {
    if (url.startsWith("http://") || url.startsWith("https://")) return url;
    if (url.startsWith("/")) return `${backendOrigin}${url}`;
    return `${backendOrigin}/${url}`;
  };

  const rawStreamUrl = currentVideo ? `${API_BASE}/videos/stream/${currentVideo.id}` : null;
  const streamUrl = summaryVideoUrl ?? rawStreamUrl;

  useEffect(() => {
    if (mapCameraCoordinates && Object.keys(mapCameraCoordinates).length > 0) return;

    let cancelled = false;

    void (async () => {
      try {
        const resp = await getDatasets();
        if (cancelled) return;
        const list = Array.isArray(resp.data) ? resp.data : [];

        if (galleryRunId) {
          const dsByGallery = list.find((d) => d.galleryRunId === galleryRunId);
          const cg = dsByGallery?.cameraCoordinates;
          if (cg && Object.keys(cg).length > 0) {
            setMapCameraCoordinates(cg);
            return;
          }
        }

        const m = effectiveRunId?.match(/^dataset_precompute_(.+)$/i);
        if (m) {
          const key = m[1];
          const dsByRun = list.find((d) => d.name.toLowerCase() === key.toLowerCase());
          const cr = dsByRun?.cameraCoordinates;
          if (cr && Object.keys(cr).length > 0) {
            setMapCameraCoordinates(cr);
          }
        }
      } catch {
        /* ignore */
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [galleryRunId, effectiveRunId, mapCameraCoordinates, setMapCameraCoordinates]);

  // -- Playback helpers ---------------------------------------------------

  const playbackProgressPct = useMemo(() => {
    if (!durationSec || durationSec <= 0) return 0;
    return Math.min(100, Math.max(0, (currentTimeSec / durationSec) * 100));
  }, [currentTimeSec, durationSec]);

  const onVideoTimeUpdate = useCallback(() => {
    const v = videoRef.current;
    if (v) setCurrentTimeSec(v.currentTime);
  }, []);

  const onVideoLoaded = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    setDurationSec(Number.isFinite(v.duration) ? v.duration : 0);
    setVideoLoadError(null);
  }, []);

  const togglePlayback = useCallback(async () => {
    const v = videoRef.current;
    if (!v) return;
    try {
      if (v.paused) await v.play();
      else v.pause();
    } catch {
      setVideoLoadError("Playback was blocked or failed.");
    }
  }, []);

  const seekToPercent = useCallback(
    (pct: number) => {
      const v = videoRef.current;
      if (!v || !durationSec) return;
      const t = (pct / 100) * durationSec;
      v.currentTime = t;
      setCurrentTimeSec(t);
    },
    [durationSec]
  );

  useEffect(() => {
    setVideoLoadError(null);
    setCurrentTimeSec(0);
    setDurationSec(0);
  }, [streamUrl]);

  useEffect(() => {
    const v = videoRef.current;
    if (v) v.playbackRate = playbackSpeed;
  }, [playbackSpeed, streamUrl]);

  // -- Export handlers -----------------------------------------------------

  const handleDownloadVideo = async () => {
    if (isExporting) return;
    if (!effectiveRunId) {
      if (streamUrl) window.open(streamUrl, "_blank", "noopener,noreferrer");
      return;
    }
    try {
      setIsExporting(true);
      const response = await generateSummaryVideo(effectiveRunId, summaryVideoPayload);
      const videoUrl = response.data?.videoUrl;
      if (videoUrl) window.open(toAbsoluteUrl(videoUrl), "_blank", "noopener,noreferrer");
      else if (streamUrl) window.open(streamUrl, "_blank", "noopener,noreferrer");
    } finally {
      setIsExporting(false);
    }
  };

  const handleExportTracklets = async () => {
    if (!effectiveRunId || isExporting) return;
    const fmt = exportFormat === "mp4" ? "json" : exportFormat;
    try {
      setIsExporting(true);
      const response = await exportTrajectories(effectiveRunId, fmt);
      const downloadUrl = response.data?.downloadUrl;
      if (downloadUrl) window.open(toAbsoluteUrl(downloadUrl), "_blank", "noopener,noreferrer");
    } finally {
      setIsExporting(false);
    }
  };

  // -- Data loading --------------------------------------------------------

  useEffect(() => {
    const ac = new AbortController();

    const loadOutputData = async () => {
      if (!currentVideo) {
        setTrajectories([]);
        setSummaryVideoUrl(null);
        setMatchedSummary(null);
        setDataSource("none");
        setError(null);
        setHydratedLatestRunId(null);
        setOutputFetchState("idle");
        return;
      }

      setOutputFetchState("loading");
      setError(null);

      try {
        let eff = runId ?? currentVideo.latestRunId ?? null;
        if (!eff) {
          const res = await getVideo(currentVideo.id);
          if (ac.signal.aborted) return;
          eff = res.data?.latestRunId ?? null;
          setHydratedLatestRunId(eff);
        } else {
          setHydratedLatestRunId(null);
        }
        if (ac.signal.aborted) return;

        if (eff) {
          setSummaryVideoUrl(null);

          const [msRes, trajRes] = await Promise.allSettled([
            getMatchedSummary(eff),
            getTrajectories(eff),
          ]);
          if (ac.signal.aborted) return;

          void generateSummaryVideo(eff, summaryVideoPayload)
            .then((resp) => {
              if (ac.signal.aborted) return;
              const url = resp.data?.videoUrl;
              setSummaryVideoUrl(url ? toAbsoluteUrl(url) : null);
            })
            .catch(() => {
              if (!ac.signal.aborted) setSummaryVideoUrl(null);
            });

          let matchedSummaryData: any = null;
          if (msRes.status === "fulfilled") {
            matchedSummaryData = msRes.value;
            setMatchedSummary(msRes.value);
          } else {
            setMatchedSummary(null);
          }

          if (trajRes.status === "fulfilled") {
            const raw = trajRes.value.data;
            const globalTrajectories = Array.isArray(raw) ? raw : [];
            if (globalTrajectories.length > 0) {
              setTrajectories(
                globalTrajectories.map((row, i) =>
                  trajectoryFromGlobal(row as unknown as Record<string, unknown>, i)
                )
              );
              setDataSource("real");
              return;
            }
          }

          if (matchedSummaryData) {
            const summaryTrajectories = trajectoriesFromMatchedSummary(matchedSummaryData);
            if (summaryTrajectories.length > 0) {
              setTrajectories(summaryTrajectories);
              setDataSource("real");
              return;
            }
          }
        }

        if (ac.signal.aborted) return;

        const trackletResponse = await getTracklets(undefined, currentVideo.id);
        if (ac.signal.aborted) return;
        const summary = Array.isArray(trackletResponse.data) ? trackletResponse.data : [];
        if (summary.length > 0) {
          setTrajectories(summary.map(trajectoryFromTrackletSummary));
          setDataSource("real");
          return;
        }
      } catch (err) {
        if (ac.signal.aborted) return;
        setError(String(err instanceof Error ? err.message : err || "Failed to load trajectory data"));
        setTrajectories([]);
        setDataSource("none");
        return;
      }

      if (!ac.signal.aborted) {
        setTrajectories([]);
        setDataSource("none");
      }
    };

    void loadOutputData().finally(() => {
      if (ac.signal.aborted) return;
      setOutputFetchState(currentVideo ? "ready" : "idle");
    });

    return () => { ac.abort(); };
  }, [currentVideo?.id, runId, currentVideo?.latestRunId, summaryVideoRequestKey]);

  useEffect(() => {
    if (displayTrajectories.length === 0) {
      if (selectedTrajectoryId !== null) setSelectedTrajectoryId(null);
      return;
    }
    const hasSelected =
      selectedTrajectoryId != null &&
      displayTrajectories.some((t) => t.id === selectedTrajectoryId);
    if (!hasSelected) setSelectedTrajectoryId(displayTrajectories[0].id);
  }, [displayTrajectories, selectedTrajectoryId]);

  // -- Derived stats -------------------------------------------------------

  const outputStats = useMemo(() => {
    const ms = matchedSummary;
    const hasMatched = ms && typeof ms.totalCameras === "number";

    const camerasFromTrajectories = new Set<string>();
    let crossCameraFromTrajectories = 0;
    displayTrajectories.forEach((traj) => {
      traj.cameras.forEach((cam) => camerasFromTrajectories.add(cam));
      if (traj.cameras.length > 1) crossCameraFromTrajectories += 1;
    });

    const camerasAnalyzed = hasMatched
      ? Math.max(ms.totalCameras, camerasFromTrajectories.size)
      : camerasFromTrajectories.size;

    const uniqueVehicles = hasMatched
      ? Math.max(ms.totalMatchedTracklets ?? 0, displayTrajectories.length)
      : displayTrajectories.length;

    const crossCameraMatches = hasMatched
      ? ms.totalMatchedTrajectories ?? 0
      : crossCameraFromTrajectories;

    const clips = Array.isArray(ms?.clips) ? ms.clips : [];
    const clipConfidences = clips.map((c: any) => Number(c.confidence ?? 0)).filter((c: number) => c > 0);
    const avgConfidence = clipConfidences.length > 0
      ? clipConfidences.reduce((a: number, b: number) => a + b, 0) / clipConfidences.length
      : displayTrajectories.length > 0
        ? displayTrajectories.reduce((sum, traj) => sum + traj.confidence, 0) / displayTrajectories.length
        : 0;

    return { camerasAnalyzed, uniqueVehicles, crossCameraMatches, avgConfidence };
  }, [displayTrajectories, matchedSummary]);

  const selectedTrajectory = useMemo(
    () => displayTrajectories.find((t) => t.id === selectedTrajectoryId) ?? null,
    [displayTrajectories, selectedTrajectoryId]
  );

  const coordRegistry = useMemo(
    () => ({ ...(mapCameraCoordinates ?? {}) }) as CoordRegistry,
    [mapCameraCoordinates]
  );

  const hasMapLayer = Object.keys(coordRegistry).length > 0;

  const registryCameraPoints = useMemo(
    () => coordRegistryEntries(coordRegistry),
    [coordRegistry]
  );

  const pathPoints = useMemo<MapPathPoint[]>(() => {
    if (!selectedTrajectory) return [];
    return selectedTrajectory.cameraSequence
      .map((cameraId, index) => {
        const coord = resolveCameraCoord(coordRegistry, cameraId);
        if (!coord) return null;
        return {
          cameraId,
          sequence: index + 1,
          tooltip: `${coord.label} (#${index + 1})`,
          ...coord,
        };
      })
      .filter((point): point is MapPathPoint => point != null);
  }, [selectedTrajectory, coordRegistry]);

  const pathLatLngs = useMemo<LatLngTuple[]>(
    () => pathPoints.map((point) => [point.lat, point.lng]),
    [pathPoints]
  );

  const missingCameraIds = useMemo(() => {
    if (!selectedTrajectory) return [];
    const missing = selectedTrajectory.cameraSequence.filter(
      (cameraId) => !resolveCameraCoord(coordRegistry, cameraId)
    );
    return Array.from(
      new Set(
        missing
          .map((c) => normalizeCameraId(c))
          .filter((id): id is string => Boolean(id))
      )
    );
  }, [selectedTrajectory, coordRegistry]);

  const mapFitPoints = useMemo(
    () =>
      pathPoints.length > 0
        ? pathPoints
        : hasMapLayer
          ? registryCameraPoints
          : [],
    [pathPoints, hasMapLayer, registryCameraPoints]
  );

  const mapCenter = useMemo(() => {
    const pts =
      pathPoints.length > 0 ? pathPoints : hasMapLayer ? registryCameraPoints : [];
    return computeCenter(pts);
  }, [pathPoints, hasMapLayer, registryCameraPoints]);

  const pathSummary = useMemo(() => {
    if (!selectedTrajectory || selectedTrajectory.cameraSequence.length === 0) return "";
    return selectedTrajectory.cameraSequence.map(formatCameraLabel).join(" -> ");
  }, [selectedTrajectory]);

  // -- Render --------------------------------------------------------------

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
      {/* Header */}
      <header className="flex shrink-0 flex-col gap-3 border-b px-4 py-3 sm:flex-row sm:items-center sm:justify-between sm:px-6">
        <div className="min-w-0">
          <h1 className="text-lg font-semibold">Results & Export</h1>
          <p className="text-sm text-muted-foreground">
            Multi-camera tracking summary
          </p>
        </div>
        <div className="flex flex-wrap items-center justify-end gap-2">
          {outputFetchState === "loading" && (
            <Badge variant="secondary" className="gap-1.5">
              <Loader2 className="h-3 w-3 animate-spin" aria-hidden />
              Loading…
            </Badge>
          )}
          {dataSource === "real" && outputFetchState === "ready" && (
            <Badge variant="outline" className="bg-green-500/10 text-green-500 border-green-500/30">
              <CheckCircle2 className="h-3 w-3 mr-1" />
              Ready
            </Badge>
          )}
        </div>
      </header>

      {/* Error banner */}
      {error && (
        <div className="flex shrink-0 items-start gap-3 border-b border-destructive/30 bg-destructive/10 px-4 py-3 sm:px-6">
          <Route className="mt-0.5 h-5 w-5 shrink-0 text-destructive" />
          <div className="min-w-0">
            <p className="text-sm font-medium text-destructive">Failed to load output data</p>
            <p className="break-words text-xs text-muted-foreground">{error}</p>
          </div>
        </div>
      )}

      {/* Two-panel content */}
      <div className="flex min-h-0 flex-1 flex-col overflow-hidden lg:flex-row">

        {/* Left panel: video + trajectory list */}
        <div className="min-w-0 flex-1 overflow-x-hidden overflow-y-auto p-4 space-y-4">
          {/* Video player */}
          <div
            className="relative w-full overflow-hidden rounded-lg border border-border bg-slate-900"
            style={{ height: "calc(100vh - 10rem)" }}
          >
            {streamUrl ? (
              <video
                ref={videoRef}
                key={streamUrl}
                src={streamUrl}
                className="pointer-events-none absolute inset-0 h-full w-full object-contain"
                autoPlay
                muted
                loop
                playsInline
                controls={false}
                onTimeUpdate={onVideoTimeUpdate}
                onLoadedMetadata={onVideoLoaded}
                onPlay={() => setIsPlaying(true)}
                onPause={() => setIsPlaying(false)}
                onError={() =>
                  setVideoLoadError("Could not load video stream.")
                }
              />
            ) : (
              <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-b from-slate-700 via-slate-800 to-slate-900">
                <p className="text-sm text-white/40">No video available</p>
              </div>
            )}

            {videoLoadError && (
              <div className="absolute bottom-16 left-3 right-3 rounded-md bg-red-950/90 px-3 py-2 text-center text-xs text-red-100">
                {videoLoadError}
              </div>
            )}

            <div className="pointer-events-none absolute top-3 right-3 rounded bg-black/60 px-2 py-1">
              <span className="text-xs font-mono text-white/80">
                {outputStats.uniqueVehicles} Tracked
              </span>
            </div>

            {/* Controls */}
            <div className="pointer-events-auto absolute bottom-0 left-0 right-0 z-10 bg-gradient-to-t from-black/80 to-transparent p-3">
              <div className="flex items-center gap-3">
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-white hover:text-white hover:bg-white/20"
                  onClick={() => void togglePlayback()}
                >
                  {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                </Button>
                <Slider
                  className="flex-1"
                  value={[playbackProgressPct]}
                  max={100}
                  step={0.1}
                  onValueChange={(v) => seekToPercent(v[0])}
                  disabled={!durationSec}
                />
                <span className="shrink-0 text-xs font-mono tabular-nums text-white">
                  {formatClock(currentTimeSec)} / {formatClock(durationSec)}
                </span>
                <span className="shrink-0 rounded bg-white/10 px-1.5 py-0.5 text-[10px] font-mono text-white/70">
                  {playbackSpeed}x
                </span>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-white hover:text-white hover:bg-white/20"
                  onClick={() => {
                    const v = videoRef.current;
                    if (!v) return;
                    if (!document.fullscreenElement) void v.parentElement?.requestFullscreen?.();
                    else void document.exitFullscreen();
                  }}
                >
                  <Maximize2 className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </div>

          {hasMapLayer && (
            <Card>
              <CardHeader className="space-y-1">
                <CardTitle className="text-sm">Vehicle Path Map</CardTitle>
                <p className="text-xs text-muted-foreground">
                  Select a tracked vehicle to draw its camera-to-camera path using this dataset&apos;s
                  geographic camera positions (<code className="text-[10px]">camera_coordinates.json</code>
                  ).
                </p>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="space-y-2">
                  <Label className="text-xs">Tracked vehicle</Label>
                  <select
                    className="w-full rounded border bg-background px-3 py-2 text-sm"
                    value={selectedTrajectoryId ?? ""}
                    onChange={(e) => {
                      const nextId = Number(e.target.value);
                      setSelectedTrajectoryId(Number.isFinite(nextId) ? nextId : null);
                    }}
                    disabled={displayTrajectories.length === 0}
                  >
                    {displayTrajectories.length === 0 && <option value="">No trajectories</option>}
                    {displayTrajectories.map((trajectory) => (
                      <option key={trajectory.id} value={trajectory.id}>
                        {trajectory.vehicleId} ({trajectory.cameras.length} cams)
                      </option>
                    ))}
                  </select>
                  {selectedTrajectory && pathSummary && (
                    <p className="text-xs text-muted-foreground break-words">
                      Path: {pathSummary}
                    </p>
                  )}
                  {selectedTrajectory && !pathSummary && (
                    <p className="text-xs text-muted-foreground">No camera sequence available for this track.</p>
                  )}
                  {missingCameraIds.length > 0 && (
                    <p className="text-xs text-amber-500">
                      Missing coordinates: {missingCameraIds.map((id) => `Camera ${id}`).join(", ")}
                    </p>
                  )}
                </div>
                <div className="relative h-72 overflow-hidden rounded-lg border bg-background">
                  <VehiclePathMap
                    center={mapCenter}
                    fitPoints={mapFitPoints}
                    cameraPoints={registryCameraPoints}
                    pathLatLngs={pathLatLngs}
                    pathPoints={pathPoints}
                    pathColor={selectedTrajectory?.color}
                  />
                  {!selectedTrajectory && (
                    <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-background/60">
                      <span className="rounded bg-background/80 px-3 py-1 text-xs text-muted-foreground">
                        No trajectories to display.
                      </span>
                    </div>
                  )}
                  {selectedTrajectory && pathPoints.length === 0 && (
                    <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-background/60">
                      <span className="rounded bg-background/80 px-3 py-1 text-xs text-muted-foreground">
                        No coordinates available for this path.
                      </span>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}

        </div>

        {/* Right sidebar */}
        <aside className="w-full shrink-0 overflow-y-auto border-t border-border bg-muted/20 p-4 lg:w-80 lg:border-l lg:border-t-0">

          {/* Quick stats */}
          <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Summary
          </h3>
          <div className="grid grid-cols-2 gap-3">
            <StatRow icon={Camera} label="Cameras" value={String(outputStats.camerasAnalyzed)} />
            <StatRow icon={Car} label="Vehicles" value={String(outputStats.uniqueVehicles)} />
            <StatRow icon={TrendingUp} label="Cross-cam" value={String(outputStats.crossCameraMatches)} />
            <StatRow icon={Gauge} label="Avg. Conf." value={`${(outputStats.avgConfidence * 100).toFixed(0)}%`} />
          </div>

          <Separator className="my-5" />

          {/* Playback speed */}
          <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Playback Speed — {playbackSpeed}x
          </h3>
          <Slider
            value={[playbackSpeed]}
            min={1}
            max={16}
            step={1}
            onValueChange={(v) => setPlaybackSpeed(v[0])}
          />

          <Separator className="my-5" />

          {/* Export */}
          <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Export
          </h3>
          <div className="space-y-3">
            <div className="space-y-1.5">
              <Label className="text-xs">Format</Label>
              <select
                className="w-full rounded border bg-background px-3 py-2 text-sm"
                value={exportFormat}
                onChange={(e) => setExportFormat(e.target.value as "mp4" | "json" | "csv")}
              >
                <option value="mp4">MP4 Video</option>
                <option value="json">JSON Tracklets</option>
                <option value="csv">CSV Export</option>
              </select>
            </div>
            <Button className="w-full" onClick={handleDownloadVideo} disabled={isExporting}>
              <Download className="mr-2 h-4 w-4" />
              Download Video
            </Button>
            <Button
              variant="outline"
              className="w-full"
              onClick={handleExportTracklets}
              disabled={!effectiveRunId || isExporting || outputFetchState === "loading"}
            >
              <Download className="mr-2 h-4 w-4" />
              Export Tracklets
            </Button>
          </div>
        </aside>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function TrajectoryItem({ trajectory }: { trajectory: OutputTrajectory }) {
  const vt = trajectory.vehicleType;
  const Icon = vt.includes("truck") ? Truck : vt.includes("bus") ? Bus : Car;

  return (
    <div className="flex items-center gap-3 rounded-lg bg-muted/50 p-3 transition-colors hover:bg-muted">
      <div
        className="flex h-9 w-9 items-center justify-center rounded"
        style={{ backgroundColor: `${trajectory.color}20`, color: trajectory.color }}
      >
        <Icon className="h-4 w-4" />
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">{trajectory.vehicleId}</span>
          <Badge variant="secondary" className="text-[10px]">
            {(trajectory.confidence * 100).toFixed(0)}%
          </Badge>
        </div>
        <div className="mt-0.5 flex items-center gap-1">
          {trajectory.cameras.map((cam, i) => (
            <div key={`${cam}-${i}`} className="flex items-center">
              <div className="h-2 w-2 rounded-full" style={{ backgroundColor: getCameraColor(cam) }} />
              {i < trajectory.cameras.length - 1 && <span className="mx-0.5 text-muted-foreground">→</span>}
            </div>
          ))}
          <span className="ml-1.5 text-[10px] text-muted-foreground">{trajectory.cameras.length} cam</span>
        </div>
      </div>
      <span className="shrink-0 text-xs font-mono text-muted-foreground">{formatDuration(trajectory.duration)}</span>
    </div>
  );
}

function StatRow({ icon: Icon, label, value }: { icon: React.ElementType; label: string; value: string }) {
  return (
    <div className="flex items-center gap-2.5 rounded-md bg-muted/40 px-3 py-2">
      <Icon className="h-4 w-4 shrink-0 text-primary" />
      <div className="min-w-0">
        <p className="text-lg font-bold leading-tight">{value}</p>
        <p className="text-[10px] text-muted-foreground">{label}</p>
      </div>
    </div>
  );
}
