"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  ZoomIn,
  ZoomOut,
  Check,
  X,
  ChevronDown,
  Layers,
  Camera,
  ArrowRight,
  Car,
  RefreshCw,
} from "lucide-react";
import { cn, formatDuration, getCameraColor } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  useTimelineStore,
  usePipelineStore,
  useSessionStore,
  useVideoStore,
  useDetectionStore,
} from "@/store";
import { getTracklets, getTrajectories, runStage, getPipelineStatus, queryTimeline, getMatchedSummary } from "@/lib/api";
import type { TimelineTrack, TrajectorySegment } from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

function shouldUseRunCropsForCamera(runId: string | undefined, cameraId: string): boolean {
  if (!runId) return false;

  // Dataset precompute runs are scene-scoped (e.g. dataset_precompute_s01).
  // If the lane camera is from a different scene, run-stage0 crops will 404.
  const m = runId.match(/^dataset_precompute_(s\d{2})$/i);
  if (m) {
    const runScene = m[1].toUpperCase();
    const camScene = cameraId.split("_")[0]?.toUpperCase() ?? "";
    return runScene === camScene;
  }

  return true;
}


export function TimelineStage() {
  const {
    tracks,
    setTracks,
    zoom,
    setZoom,
    selectedTrackId,
    selectTrack,
    confirmTrack,
    unconfirmTrack,
    removeTrack,
  } = useTimelineStore();
  const { runId, galleryRunId, updateStageProgress, stages } = usePipelineStore();
  const { setCurrentStage } = useSessionStore();
  const { currentVideo } = useVideoStore();
  const { selectedIds } = useDetectionStore();

  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [selectedLaneId, setSelectedLaneId] = useState<string | null>(null);
  const [splitCount, setSplitCount] = useState(6);
  const [showProgress, setShowProgress] = useState(true);
  const [triggerReload, setTriggerReload] = useState(0);
  const [matchedFallbackActive, setMatchedFallbackActive] = useState(false);
  const timelineRef = useRef<HTMLDivElement>(null);

  type CameraLaneSegment = TrajectorySegment & {
    trajectoryId: string;
    globalId?: number;
    confidence?: number;
    className?: string;
    confirmed?: boolean;
  };

  type CameraLane = {
    id: string;
    cameraId: string;
    label: string;
    startTime: number;
    endTime: number;
    segments: CameraLaneSegment[];
  };

  const parseSelectedTrackId = (rawId: string): number | null => {
    const direct = Number(rawId);
    if (Number.isFinite(direct)) return direct;
    const m = rawId.match(/^det-(\d+)-/);
    if (!m) return null;
    const parsed = Number(m[1]);
    return Number.isFinite(parsed) ? parsed : null;
  };

  const parseEvidenceTrackletKey = (raw: string): string | null => {
    // Expected format: "(camera_id, track_id)"
    const m = raw.match(/^\((.+?),\s*(\d+)\)$/);
    if (!m) return null;
    return `${normalizeCameraId(String(m[1]))}:${Number(m[2])}`;
  };

  const normalizeCameraId = (cameraId: string): string => {
    // Stage 4 query mode prefixes uploaded cameras with `query_`.
    // Normalize here so selected keys from Stage 2 still match Stage 4 outputs.
    return cameraId.startsWith("query_") ? cameraId.slice(6) : cameraId;
  };
  const scoreTrajectoryForQuery = (trajectory: any, selectedTrackKeys: Set<string>): number => {
    const evidence = Array.isArray(trajectory?.evidence) ? trajectory.evidence : [];
    let best = -1;

    // Prefer explicit pairwise evidence similarity when present.
    for (const ev of evidence) {
      const aKey = parseEvidenceTrackletKey(String(ev?.tracklet_a ?? ev?.trackletA ?? ""));
      const bKey = parseEvidenceTrackletKey(String(ev?.tracklet_b ?? ev?.trackletB ?? ""));
      const sim = Number(ev?.similarity ?? 0);
      if (!Number.isFinite(sim)) continue;
      const touchesQuery = (aKey && selectedTrackKeys.has(aKey)) || (bKey && selectedTrackKeys.has(bKey));
      if (touchesQuery) best = Math.max(best, sim);
    }

    // Fallback: if evidence is absent, use trajectory confidence.
    if (best < 0) {
      const conf = Number(trajectory?.confidence ?? 0);
      if (Number.isFinite(conf)) best = conf;
    }

    return Math.max(best, 0);
  };

  // Compute the trimmed timeline window: throw away silence before first detection
  // and after last detection. Each camera's activity is a [start, end] interval;
  // we take the union span so the ruler covers exactly the vehicle's presence.
  const { timelineStart, timelineEnd, totalDuration } = (() => {
    if (tracks.length === 0) {
      const dur = Math.max(currentVideo?.duration ?? 1, 1);
      return { timelineStart: 0, timelineEnd: dur, totalDuration: dur };
    }
    let minStart = Infinity;
    let maxEnd = -Infinity;
    tracks.forEach((track) => {
      const segs = track.segments && track.segments.length > 0
        ? track.segments
        : [{ start: track.startTime, end: track.endTime }];
      segs.forEach((s) => {
        if (s.start < minStart) minStart = s.start;
        if (s.end > maxEnd) maxEnd = s.end;
      });
    });
    if (!isFinite(minStart)) minStart = 0;
    if (!isFinite(maxEnd)) maxEnd = minStart + 1;
    const dur = Math.max(maxEnd - minStart, 1);
    return { timelineStart: minStart, timelineEnd: maxEnd, totalDuration: dur };
  })();


  const cameraLanes: CameraLane[] = (() => {
    const laneMap = new Map<string, CameraLaneSegment[]>();

    tracks.forEach((track) => {
      const segs: TrajectorySegment[] = track.segments && track.segments.length > 0
        ? track.segments
        : [{
            cameraId: track.cameraId,
            trackId: track.trackletId,
            start: track.startTime,
            end: track.endTime,
            color: getCameraColor(track.cameraId),
            representativeFrame: track.representativeFrame,
            representativeBbox: track.representativeBbox,
          }];

      segs.forEach((seg) => {
        const list = laneMap.get(seg.cameraId) ?? [];
        list.push({
          ...seg,
          trajectoryId: track.id,
          globalId: track.globalId,
          confidence: track.confidence,
          className: track.className,
          confirmed: track.confirmed,
        });
        laneMap.set(seg.cameraId, list);
      });
    });

    const lanes: CameraLane[] = [];
    laneMap.forEach((segments, cameraId) => {
      const sorted = [...segments].sort((a, b) => a.start - b.start);
      const start = Math.min(...sorted.map((s) => s.start));
      const end = Math.max(...sorted.map((s) => s.end));
      lanes.push({
        id: `lane-${cameraId}`,
        cameraId,
        label: cameraId,
        startTime: start,
        endTime: end,
        segments: sorted,
      });
    });

    lanes.sort((a, b) => {
      if (a.startTime !== b.startTime) return a.startTime - b.startTime;
      return a.cameraId.localeCompare(b.cameraId);
    });

    return lanes;
  })();

  // Cameras available for the preview grid
  // Build from unique cameras across all tracks (including segment cameras)
  const camerasForPreview = (() => {
    return cameraLanes.map((lane) => ({
      id: lane.cameraId,
      scene: lane.cameraId.split("_")[0] ?? "Unknown",
      name: lane.cameraId,
      location: "Camera",
    }));
  })();

  const buildTracksFromSummary = (summary: any[]): TimelineTrack[] => {
    if (!currentVideo || summary.length === 0) return [];
    const fps = Math.max(currentVideo.fps || 10, 1);

    return summary.map((item: any, index: number) => {
      const startFrame = Number(item.startFrame ?? 0);
      const endFrame = Number(item.endFrame ?? startFrame);
      const startTime = startFrame / fps;
      const endTime = endFrame / fps;

      return {
        id: `real-${item.cameraId}-${item.id}-${index}`,
        cameraId: String(item.cameraId ?? "unknown"),
        trackletId: Number(item.id ?? index),
        startTime,
        endTime: Math.max(endTime, startTime + 0.1),
        selected: false,
        confirmed: index === 0,
        representativeFrame: item.representativeFrame,
        representativeBbox: item.representativeBbox,
        sampleFrames: item.sampleFrames,
      };
    });
  };

  /**
   * Build timeline tracks from the matched/summary.json fallback.
   * Used when normal trajectory rendering fails or returns empty rows.
   */
  const buildTracksFromMatchedSummary = (summary: any): TimelineTrack[] => {
    const clips: any[] = Array.isArray(summary?.clips) ? summary.clips.filter((c: any) => c.ok) : [];
    if (clips.length === 0) return [];

    // Group by global_id → one trajectory row per global vehicle
    const byGid = new Map<number, any[]>();
    for (const clip of clips) {
      const gid = Number(clip.global_id ?? 0);
      byGid.set(gid, [...(byGid.get(gid) ?? []), clip]);
    }

    const rows: TimelineTrack[] = [];
    byGid.forEach((clipList, gid) => {
      const segments = clipList.map((clip: any) => ({
        cameraId: String(clip.camera_id),
        trackId: Number(clip.track_id),
        globalId: gid,
        start: Number(clip.start_time_s ?? 0),
        end: Number(clip.end_time_s ?? (Number(clip.start_time_s ?? 0) + Number(clip.duration_s ?? 0.1))),
        color: getCameraColor(String(clip.camera_id)),
      }));
      const startTime = Math.min(...segments.map((s) => s.start));
      const endTime = Math.max(...segments.map((s) => s.end));
      rows.push({
        id: `fallback-${gid}`,
        cameraId: segments[0]?.cameraId ?? "unknown",
        trackletId: gid,
        globalId: gid,
        startTime,
        endTime: Math.max(endTime, startTime + 0.1),
        selected: false,
        confirmed: true,
        segments,
        label: `G-${String(gid).padStart(4, "0")} · ${segments.length} cam${segments.length !== 1 ? "s" : ""}`,
        confidence: Number(clipList[0]?.confidence ?? 0),
      });
    });
    return rows;
  };

  /**
   * Notebook-aligned: one row per global trajectory.
   * Each row carries a `segments[]` array — one colored block per camera,
   * matching the `gp-stage-4.ipynb` multi-camera timeline visualization.
   */
  const buildTracksFromTrajectories = (
    trajectories: any[],
    selectedTrackKeys?: Set<string>
  ): TimelineTrack[] => {
    if (!Array.isArray(trajectories) || trajectories.length === 0) return [];

    // If caller provided a selection filter but it resolved to no keys,
    // treat as "no match" instead of showing every trajectory.
    if (selectedTrackKeys !== undefined && selectedTrackKeys.size === 0) return [];

    // Filter to selected identities using camera_id + track_id, because track_id
    // alone is not globally unique across cameras.
    const filtered = selectedTrackKeys !== undefined
      ? trajectories.filter((traj: any) => {
          const tracklets = Array.isArray(traj.tracklets) ? traj.tracklets : [];
          return tracklets.some((t: any) => {
            const cam = normalizeCameraId(String(t.camera_id ?? t.cameraId ?? ""));
            const tid = Number(t.track_id ?? t.trackId ?? -1);
            return selectedTrackKeys.has(`${cam}:${tid}`);
          });
        })
      : trajectories;

    const rows: TimelineTrack[] = [];

    // Query-centric ordering: highest feature similarity (or confidence fallback) first.
    const sortedFiltered = [...filtered].sort((a: any, b: any) => {
      const sa = selectedTrackKeys && selectedTrackKeys.size > 0
        ? scoreTrajectoryForQuery(a, selectedTrackKeys)
        : Number(a?.confidence ?? 0);
      const sb = selectedTrackKeys && selectedTrackKeys.size > 0
        ? scoreTrajectoryForQuery(b, selectedTrackKeys)
        : Number(b?.confidence ?? 0);
      return sb - sa;
    });

    sortedFiltered.forEach((trajectory: any, trajectoryIndex: number) => {
      const globalId = Number(trajectory.globalId ?? trajectory.global_id ?? trajectoryIndex + 1);

      // Backend writes snake_case timeline entries from global_trajectories.py:
      // { camera_id, track_id, start, end, duration_s, num_frames, mean_confidence }
      const timeline: any[] = Array.isArray(trajectory.timeline) ? trajectory.timeline : [];
      const tracklets: any[] = Array.isArray(trajectory.tracklets) ? trajectory.tracklets : [];

      if (timeline.length === 0) return; // nothing to render

      // Build one segment per camera appearance, sorted by start time (already sorted by backend)
      const segments = timeline.map((entry: any) => {
        const cameraId = String(entry.camera_id ?? entry.cameraId ?? "unknown");
        const trackId = entry.track_id ?? entry.trackId;
        const start = Number(entry.start ?? 0);
        const end = Number(entry.end ?? start + 0.1);

        // Find representative frame + bbox from the matching tracklet
        let representativeFrame: number | undefined;
        let representativeBbox: number[] | undefined;
        const matchedTracklet = tracklets.find(
          (t: any) =>
            (t.track_id ?? t.trackId) === trackId &&
            normalizeCameraId(String(t.camera_id ?? t.cameraId)) === normalizeCameraId(cameraId)
        );
        if (matchedTracklet) {
          const frames: any[] = Array.isArray(matchedTracklet.frames) ? matchedTracklet.frames : [];
          const midFrame = frames[Math.floor(frames.length / 2)];
          if (midFrame) {
            representativeFrame = Number(midFrame.frame_id ?? midFrame.frameId ?? 0);
            representativeBbox = Array.isArray(midFrame.bbox) ? midFrame.bbox : undefined;
          }
        }

        return {
          cameraId,
          trackId: Number(trackId ?? 0),
          globalId,
          start,
          end: Math.max(end, start + 0.1),
          color: getCameraColor(cameraId),
          representativeFrame,
          representativeBbox,
        };
      });

      // Row spans the full trajectory extent
      const rowStart = Math.min(...segments.map((s) => s.start));
      const rowEnd = Math.max(...segments.map((s) => s.end));

      // Prefer the segment that actually matches the user's Stage 2 selection;
      // otherwise fall back to the first segment.
      const selectedSegment = segments.find((seg) =>
        selectedTrackKeys?.has(`${normalizeCameraId(seg.cameraId)}:${seg.trackId}`)
      );
      const primarySegment = selectedSegment ?? segments[0];
      const primaryCamera = primarySegment?.cameraId ?? "unknown";

      // Determine dominant class from trajectory (backend may include class_name)
      const className: string = trajectory.class_name ?? trajectory.className ?? "vehicle";
      const nCams = new Set(segments.map((s) => s.cameraId)).size;
      const confidence: number = selectedTrackKeys && selectedTrackKeys.size > 0
        ? scoreTrajectoryForQuery(trajectory, selectedTrackKeys)
        : (typeof trajectory.confidence === "number" ? trajectory.confidence : 1);

      rows.push({
        id: `traj-${globalId}`,
        cameraId: primaryCamera,
        trackletId: globalId,
        globalId,
        startTime: rowStart,
        endTime: rowEnd,
        selected: false,
        confirmed: true,
        representativeFrame: primarySegment?.representativeFrame,
        representativeBbox: primarySegment?.representativeBbox,
        // Notebook-style multi-camera data
        segments,
        label: `G-${String(globalId).padStart(4, "0")} · ${nCams} cam${nCams !== 1 ? "s" : ""} · ${className}`,
        confidence,
        className,
      });
    });

    return rows;
  };

  // Stage 4: run real association, then load tracks
  useEffect(() => {
    let cancelled = false;

    const loadTracks = async () => {
      if (!currentVideo) return;
        try {
        let attemptedAssociation = false;
        const selectedTrackIds = Array.from(selectedIds).map((v) => String(v));

        console.groupCollapsed("[Stage4][Timeline] loadTracks");
        console.info("context", {
          runId,
          videoId: currentVideo.id,
          selectedTrackletCount: selectedTrackIds.length,
          selectedTrackIds,
        });

        // Query mode: resolve selected tracklets to matched trajectories on backend.
        // Use galleryRunId (precomputed dataset) when available; fall back to probe runId.
        const effectiveGalleryRunId = galleryRunId ?? runId;
        if (effectiveGalleryRunId && selectedIds.size > 0) {
          attemptedAssociation = true;
          const q1 = await queryTimeline(effectiveGalleryRunId, currentVideo.id, selectedTrackIds);
          if (cancelled) return;

          const q1Data: any = q1.data ?? {};
          const q1Traj = Array.isArray(q1Data.trajectories) ? q1Data.trajectories : [];
          const q1Selected = Array.isArray(q1Data.selectedTracklets) ? q1Data.selectedTracklets : [];

          console.info("queryTimeline#1", {
            mode: q1Data.mode,
            message: q1Data.message,
            stage4Available: q1Data.stage4Available,
            diagnostics: q1Data.diagnostics,
            trajectories: q1Traj.length,
            selectedTracklets: q1Selected.length,
          });

          if (q1Traj.length > 0) {
            const rows = buildTracksFromTrajectories(q1Traj);
            if (rows.length > 0) {
              setMatchedFallbackActive(false);
              setTracks(rows);
              updateStageProgress(4, { status: "completed", progress: 100, message: String(q1Data.message ?? "Association loaded (query-matched)") });
              console.info("decision", "matched trajectories rendered", { rows: rows.length });
              console.groupEnd();
              return;
            }
            // Trajectories returned but buildTracksFromTrajectories yielded nothing
            // (likely corrupted timeline entries). Fall through to matched summary fallback.
            console.warn("decision", "trajectories returned but building rows failed, trying matched summary fallback");
          }

          // Matched summary fallback: try outputs/{probeRunId}/matched/summary.json
          if (runId) {
            try {
              const summaryResp = await getMatchedSummary(runId);
              if (cancelled) return;
              const fallbackRows = buildTracksFromMatchedSummary(summaryResp);
              if (fallbackRows.length > 0) {
                setMatchedFallbackActive(true);
                setTracks(fallbackRows);
                updateStageProgress(4, { status: "completed", progress: 100, message: "Showing pre-exported matched clips (fallback)" });
                console.info("decision", "matched summary fallback rendered", { rows: fallbackRows.length });
                console.groupEnd();
                return;
              }
            } catch (_) { /* summary not available, continue */ }
          }

          // If stage4 artifacts are missing for this run, execute stage4 then query again.
          if (!q1Data.stage4Available) {
            updateStageProgress(4, { status: "running", progress: 5, message: "Running cross-camera association..." });
            const stageResp = await runStage(4, { runId: effectiveGalleryRunId, videoId: currentVideo.id });
            if (cancelled) return;
            const stage4RunId = (stageResp.data as any)?.runId ?? runId;

            let done = false;
            while (!done && !cancelled) {
              await new Promise((r) => setTimeout(r, 1500));
              if (cancelled) return;
              const statusResp = await getPipelineStatus(stage4RunId);
              if (cancelled) return;
              const statusData: any = statusResp.data;
              const status = statusData?.status;
              const progress = Number(statusData?.progress ?? 0);
              const message = String(statusData?.message ?? "Running...");
              updateStageProgress(4, { progress, message });
              if (status === "completed" || status === "error") done = true;
              if (status === "error") {
                const errMsg = String(statusData?.error ?? "Stage 4 association failed");
                updateStageProgress(4, { status: "error", message: errMsg });
                break;
              }
            }

            if (!cancelled) {
              const q2 = await queryTimeline(stage4RunId ?? effectiveGalleryRunId, currentVideo.id, selectedTrackIds);
              if (cancelled) return;
              const q2Data: any = q2.data ?? {};
              const q2Traj = Array.isArray(q2Data.trajectories) ? q2Data.trajectories : [];
              const q2Selected = Array.isArray(q2Data.selectedTracklets) ? q2Data.selectedTracklets : [];

              console.info("queryTimeline#2", {
                mode: q2Data.mode,
                message: q2Data.message,
                stage4Available: q2Data.stage4Available,
                diagnostics: q2Data.diagnostics,
                trajectories: q2Traj.length,
                selectedTracklets: q2Selected.length,
              });

              if (q2Traj.length > 0) {
                const rows = buildTracksFromTrajectories(q2Traj);
                setTracks(rows);
                updateStageProgress(4, { status: "completed", progress: 100, message: String(q2Data.message ?? "Association complete (query-matched)") });
                console.info("decision", "matched trajectories rendered after stage4", { rows: rows.length });
                console.groupEnd();
                return;
              }

              if (q2Selected.length > 0) {
                const fallbackTracks = buildTracksFromSummary(q2Selected);
                setTracks(fallbackTracks);
                updateStageProgress(4, {
                  status: "completed",
                  progress: 100,
                  message: String(q2Data.message ?? "No cross-camera match found; showing selected single-camera tracklets"),
                });
                console.info("decision", "selected single-camera fallback rendered after stage4", {
                  rows: fallbackTracks.length,
                });
                console.groupEnd();
                return;
              }
            }
          }

          // stage4 exists but no match; show selected single-camera tracklets if available.
          if (q1Selected.length > 0) {
            const fallbackTracks = buildTracksFromSummary(q1Selected);
            setTracks(fallbackTracks);
            updateStageProgress(4, {
              status: "completed",
              progress: 100,
              message: String(q1Data.message ?? "No cross-camera match found; showing selected single-camera tracklets"),
            });
            console.info("decision", "selected single-camera fallback rendered", { rows: fallbackTracks.length });
            console.groupEnd();
            return;
          }

          // Critical: if query mode was requested but backend returned neither
          // matches nor selected fallback, do NOT fall through to the non-query
          // path that loads all trajectories.
          setTracks([]);
          updateStageProgress(4, {
            status: "completed",
            progress: 100,
            message: String(q1Data.message ?? "Selected query could not be resolved for this run/video context"),
          });
          console.warn("decision", "query unresolved; blocked non-query fallback to avoid showing all trajectories", {
            diagnostics: q1Data.diagnostics,
          });
          console.groupEnd();
          return;
        }

        // If we already have stage4 trajectory artifacts and no explicit query selection,
        // load them directly.
        if (runId) {
          attemptedAssociation = true;
          const trajectoryResponse = await getTrajectories(runId);
          if (cancelled) return;

          const trajectoryRows = buildTracksFromTrajectories(
            Array.isArray(trajectoryResponse.data) ? trajectoryResponse.data : []
          );
          if (trajectoryRows.length > 0) {
            setTracks(trajectoryRows);
            updateStageProgress(4, { status: "completed", progress: 100, message: "Association loaded (query-matched)" });
            console.info("decision", "non-query stage4 trajectories rendered", { rows: trajectoryRows.length });
            console.groupEnd();
            return;
          }

          // No stage4 artifacts yet — run stage 4 now
          updateStageProgress(4, { status: "running", progress: 5, message: "Running cross-camera association..." });
          const stageResp = await runStage(4, { runId, videoId: currentVideo.id });
          if (cancelled) return;
          const stage4RunId = (stageResp.data as any)?.runId ?? runId;

          // Poll until done
          let done = false;
          while (!done && !cancelled) {
            await new Promise((r) => setTimeout(r, 1500));
            if (cancelled) return;
            const statusResp = await getPipelineStatus(stage4RunId);
            if (cancelled) return;
            const statusData: any = statusResp.data;
            const status = statusData?.status;
            const progress = Number(statusData?.progress ?? 0);
            const message = String(statusData?.message ?? "Running...");
            updateStageProgress(4, { progress, message });
            if (status === "completed" || status === "error") done = true;
            if (status === "error") {
              const errMsg = String(statusData?.error ?? "Stage 4 association failed");
              updateStageProgress(4, { status: "error", message: errMsg });
              console.warn("[Stage 4] Association failed, falling back to Stage 1 tracklets:", errMsg);
              // Fall through to load stage1 tracklets
              break;
            }
          }

          if (!cancelled) {
            const traj2 = await getTrajectories(stage4RunId);
            if (cancelled) return;
            const rows2 = buildTracksFromTrajectories(
              Array.isArray(traj2.data) ? traj2.data : []
            );
            if (rows2.length > 0) {
              setTracks(rows2);
              updateStageProgress(4, { status: "completed", progress: 100, message: "Association complete (query-matched)" });
              console.info("decision", "non-query stage4 trajectories rendered after rerun", { rows: rows2.length });
              console.groupEnd();
              return;
            }
          }
        }

        // Only suppress fallback when association was actually attempted.
        // If we ran association but found nothing, just fallback anyway so the timeline isn't completely empty and broken.
        if (attemptedAssociation && selectedIds.size > 0) {
          // Keep strict query behavior (do not show unrelated trajectories), but
          // avoid a blank timeline by falling back to only the selected stage-1 tracklets.
          const fallbackResp = await getTracklets(undefined, currentVideo.id);
          if (cancelled) return;
          let fallbackSummary = Array.isArray(fallbackResp.data) ? fallbackResp.data : [];

          const selectedTrackNums = new Set<number>();
          selectedIds.forEach((raw) => {
            const parsed = parseSelectedTrackId(String(raw));
            if (parsed !== null) selectedTrackNums.add(parsed);
          });
          fallbackSummary = fallbackSummary.filter((item: any) => selectedTrackNums.has(Number(item.id)));

          const fallbackTracks = buildTracksFromSummary(fallbackSummary);
          if (fallbackTracks.length > 0) {
            setTracks(fallbackTracks);
            updateStageProgress(4, {
              status: "completed",
              progress: 100,
              message: "No cross-camera match found; showing selected single-camera tracklets",
            });
            console.info("decision", "selected stage1 fallback rendered", { rows: fallbackTracks.length });
          } else {
            setTracks([]);
            updateStageProgress(4, {
              status: "completed",
              progress: 100,
              message: "No association match found for selected query tracklet(s)",
            });
            console.warn("decision", "no selected stage1 fallback available");
          }
          console.groupEnd();
          return;
        }

        // No query selection: fallback to stage1 tracklets (single-camera view)
        const response = await getTracklets(undefined, currentVideo.id);
        if (cancelled) return;
        let summary = Array.isArray(response.data) ? response.data : [];

        // Filter to only selected tracklets from Stage 2
        if (selectedIds.size > 0) {
          const selectedTrackNums = new Set<number>();
          selectedIds.forEach((raw) => {
            const parsed = parseSelectedTrackId(String(raw));
            if (parsed !== null) selectedTrackNums.add(parsed);
          });
          summary = summary.filter((item: any) => selectedTrackNums.has(Number(item.id)));
        }
        const realTracks = buildTracksFromSummary(summary);
        if (realTracks.length > 0) {
          setTracks(realTracks);
          updateStageProgress(4, { status: "completed", progress: 100, message: "Showing stage 1 tracklets" });
          console.info("decision", "no-query stage1 summary rendered", { rows: realTracks.length });
        }
        console.groupEnd();
      } catch (err) {
        if (!cancelled) {
          updateStageProgress(4, { status: "error", progress: 0, message: String(err) });
          console.error("[Stage4][Timeline] loadTracks error", err);
        }
        console.groupEnd();
      }
    };

    void loadTracks();

    return () => {
      cancelled = true;
    };
  }, [currentVideo, runId, selectedIds, setTracks, triggerReload]);

  // Playback simulation — advance at correct FPS derived from video metadata
  useEffect(() => {
    if (!isPlaying) return;
    const fps = Math.max(currentVideo?.fps ?? 10, 1);
    // Advance by 1 video frame per tick (capped at ~30fps for smooth UI)
    const tickFps = Math.min(fps, 30);
    const increment = 1 / tickFps;
    const interval = setInterval(() => {
      // currentTime is an offset from timelineStart (0 = first detection)
      setCurrentTime((t) => (t + increment >= totalDuration ? 0 : t + increment));
    }, 1000 / tickFps);
    return () => clearInterval(interval);
  }, [isPlaying, totalDuration, currentVideo?.fps]);

  useEffect(() => {
    setCurrentTime((t) => Math.min(t, totalDuration));
  }, [totalDuration]);


  const stage4Progress = stages.find((s) => s.stage === 4);

  // timeToPixel maps an ABSOLUTE timestamp (seconds from dataset start) to
  // a pixel offset on the trimmed ruler (0 px = timelineStart).
  const timeToPixel = useCallback(
    (time: number) => {
      const baseWidth = 1200;
      return ((time - timelineStart) / totalDuration) * baseWidth * zoom;
    },
    [zoom, totalDuration, timelineStart]
  );

  // offsetToTime: inverse — pixel offset → absolute time
  const offsetToTime = useCallback(
    (px: number) => {
      const baseWidth = 1200;
      return (px / (baseWidth * zoom)) * totalDuration + timelineStart;
    },
    [zoom, totalDuration, timelineStart]
  );

  const handleTrackClick = (trackId: string) => {
    selectTrack(trackId === selectedTrackId ? null : trackId);
  };

  const handleConfirmToggle = (trackId: string, isConfirmed: boolean) => {
    if (isConfirmed) {
      unconfirmTrack(trackId);
    } else {
      confirmTrack(trackId);
    }
  };

  const handleProceed = () => {
    setCurrentStage(5);
  };

  const handleRerunAssociation = async () => {
    if (!currentVideo || !runId) return;
    setTracks([]);
    updateStageProgress(4, { status: "running", progress: 5, message: "Manually re-running association..." });
    try {
      await runStage(4, { runId, videoId: currentVideo.id });
      let done = false;
      while (!done) {
        await new Promise((r) => setTimeout(r, 1500));
        const statusData = await getPipelineStatus(runId);
        const st = statusData.data as any;
        const progress = st?.stageProgress?.[4] ?? 0;
        const message = st?.stageMessages?.[4] ?? "Running...";
        const status = st?.stageStatus?.[4] ?? "running";
        
        updateStageProgress(4, { progress, message, status });
        if (status === "completed" || status === "error") {
          done = true;
        }
      }
      // trigger a refresh
      setTriggerReload(n => n + 1);
    } catch (e) {
      updateStageProgress(4, { status: "error", progress: 0, message: String(e) });
    }
  };

  const confirmedCount = tracks.filter((t) => t.confirmed).length;
  const timelineDataSource = tracks.some((t) => t.id.startsWith("real-") || t.id.startsWith("traj-")) ? "real" : "demo";

  // For summary badge: how many trajectories are shown vs total selected tracklets
  const shownTrajectories = tracks.length;
  const shownCameraLanes = cameraLanes.length;
  const selectedTrackletCount = selectedIds.size;

  // Dynamic time ruler tick interval: keep ~10–15 ticks on screen regardless of duration
  const rulerTickInterval = totalDuration <= 30 ? 5 : totalDuration <= 120 ? 10 : totalDuration <= 600 ? 30 : 60;

  const visibleCameras = camerasForPreview.slice(0, splitCount);

  // For the selected trajectory, find which cameras are active at currentTime.
  // Also determine "past" cameras (ended) and "next" cameras (not started yet)
  // to mirror the notebook visualization.
  const absCurrentTime = timelineStart + currentTime;
  const activeCamerasForGrid = visibleCameras.map((cam) => {
    const lane = cameraLanes.find((l) => l.cameraId === cam.id);
    const laneSegments = lane?.segments ?? [];
    const activeSegment = laneSegments.find((s) => absCurrentTime >= s.start && absCurrentTime <= s.end);
    const isActive = Boolean(activeSegment);
    const isPast = !activeSegment && laneSegments.length > 0 && laneSegments.every((s) => absCurrentTime > s.end);
    const isNext = !activeSegment && laneSegments.length > 0 && laneSegments.every((s) => absCurrentTime < s.start);
    // Find a representative frame+bbox for this slot
    const representativeFrame = activeSegment?.representativeFrame;
    const representativeBbox = activeSegment?.representativeBbox;
    const trackForPreview = isActive || isPast || isNext
      ? {
          ...cam,
          representativeFrame,
          representativeBbox,
          cameraId: cam.id,
          color: activeSegment?.color ?? laneSegments[0]?.color,
        }
      : undefined;
    // Primary segment: active now, OR first segment in lane (for NEXT), OR last segment (for PAST)
    const primarySeg = activeSegment
      ?? (isNext ? laneSegments[0] : undefined)
      ?? (isPast ? laneSegments[laneSegments.length - 1] : undefined);
    return { ...cam, activeTrack: isActive ? trackForPreview : undefined, isPast, isNext, segment: activeSegment, primarySeg };
  });

  // DEBUG: log what each camera cell will receive
  if (activeCamerasForGrid.length > 0) {
    console.log("[Timeline] activeCamerasForGrid:", activeCamerasForGrid.map(c => ({
      id: c.id,
      isActive: !!c.activeTrack,
      isPast: c.isPast,
      isNext: c.isNext,
      primarySeg: c.primarySeg ? {
        cameraId: c.primarySeg.cameraId,
        trackId: c.primarySeg.trackId,
        globalId: c.primarySeg.globalId,
        start: c.primarySeg.start,
        end: c.primarySeg.end,
      } : null,
    })));
    console.log("[Timeline] runId =", runId);
  }

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <header className="flex h-14 items-center justify-between border-b px-6">
        <div>
          <h1 className="text-lg font-semibold">Stage 4: Cross-Camera Timeline</h1>
          <p className="text-sm text-muted-foreground">
            DeepOCSORT tracklet association across CityFlow cameras
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant={timelineDataSource === "real" ? "secondary" : "destructive"}>
            {timelineDataSource === "real" ? "Real Artifacts" : "No Data"}
          </Badge>
          {matchedFallbackActive && (
            <Badge variant="outline" className="border-yellow-500/60 text-yellow-400 bg-yellow-500/10">
              ⚠ Fallback: pre-exported matched clips
            </Badge>
          )}
          <Badge variant="secondary">{shownTrajectories} trajectories</Badge>
          <Badge variant="secondary">{shownCameraLanes} camera rows</Badge>
          {selectedTrackletCount > 0 && (
            <Badge variant="outline" className="border-blue-500/30 text-blue-400 bg-blue-500/10">
              {selectedTrackletCount} selected tracklets
            </Badge>
          )}
          <Badge variant="outline" className="bg-green-500/10 text-green-500 border-green-500/30">
            {confirmedCount} confirmed
          </Badge>
          <Button
              className="mr-2"
              variant="outline"
              disabled={false}
              onClick={handleRerunAssociation}
            >
              <RefreshCw className={cn("mr-2 h-4 w-4", false && "animate-spin")} />
              Rerun Association
          </Button>
          <Button onClick={handleProceed}>
            Continue to Refinement
            <ArrowRight className="ml-2 h-4 w-4" />
          </Button>
        </div>
      </header>

      {/* Main content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left panel */}
        <aside className="w-64 border-r flex flex-col bg-muted/20">
          <div className="p-4 border-b">
            <h3 className="font-semibold flex items-center gap-2">
              <Layers className="h-4 w-4" />
              Association Progress
            </h3>
          </div>
          <div className="p-4">
            {stage4Progress?.status === "running" ? (
              <div className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span className="text-xs">{stage4Progress.message}</span>
                  <span className="font-mono">{stage4Progress.progress}%</span>
                </div>
                <Progress value={stage4Progress.progress} className="h-2" />
              </div>
            ) : (
              <div className="flex items-center gap-2 text-green-500">
                <Check className="h-4 w-4" />
                <span className="text-sm">Association complete</span>
              </div>
            )}
          </div>

          <Separator />

          {/* Split screen control */}
          <div className="p-4">
            <h4 className="text-sm font-medium mb-3">Camera Grid</h4>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Cameras</span>
                <span className="text-sm font-medium">{splitCount}</span>
              </div>
              <Slider
                value={[splitCount]}
                min={1}
                max={6}
                step={1}
                onValueChange={(v) => setSplitCount(v[0])}
              />
            </div>
          </div>

          <Separator />

          {/* Tracklet list */}
          <div className="flex-1 overflow-auto p-4">
            <h4 className="text-sm font-medium mb-1">Trajectories</h4>
            {tracks.length === 0 ? (
              <p className="text-xs text-muted-foreground mt-2">
                {selectedTrackletCount > 0
                  ? (stage4Progress?.message || "No trajectories match selected tracklets. Run Stage 4 to associate them.")
                  : "No tracklet data yet."}
              </p>
            ) : (
              <div className="space-y-2">
                {tracks.map((track) => (
                  <TrackletItem
                    key={track.id}
                    track={track}
                    isSelected={selectedTrackId === track.id}
                    onClick={() => handleTrackClick(track.id)}
                    onConfirm={() => handleConfirmToggle(track.id, track.confirmed)}
                    onRemove={() => removeTrack(track.id)}
                  />
                ))}
              </div>
            )}
          </div>
        </aside>

        {/* Main timeline area */}
        <div className="flex-1 flex flex-col">
          {/* Video preview area - CityFlow camera grid */}
          <div className="h-56 border-b bg-slate-900 p-2">
            <div
              className="grid gap-1 h-full"
              style={{
                // Dynamic grid: 1→1×1, 2→2×1, 3→2×2, 4→2×2, 5→3×2, 6→3×2
                gridTemplateColumns: `repeat(${Math.ceil(Math.sqrt(splitCount))}, 1fr)`,
                gridTemplateRows: `repeat(${Math.ceil(splitCount / Math.ceil(Math.sqrt(splitCount)))}, 1fr)`,
              }}
            >
              {activeCamerasForGrid.map((cam) => (
                <CameraPreview
                  key={cam.id}
                  camera={cam}
                  isActive={!!cam.activeTrack}
                  isPast={cam.isPast}
                  isNext={cam.isNext}
                  currentTime={currentTime}
                  absCurrentTime={absCurrentTime}
                  isPlaying={isPlaying}
                  primarySeg={cam.primarySeg}
                  probeRunId={runId ?? undefined}
                  videoId={currentVideo?.id}
                  runId={runId ?? undefined}
                />
              ))}
            </div>
          </div>

          {/* Timeline controls */}
          <div className="flex items-center gap-4 p-4 border-b bg-muted/30">
            <div className="flex items-center gap-1">
              <Button variant="ghost" size="icon" onClick={() => setCurrentTime(0)}>
                <SkipBack className="h-4 w-4" />
              </Button>
              <Button
                variant="default"
                size="icon"
                onClick={() => setIsPlaying(!isPlaying)}
              >
                {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              </Button>
              <Button variant="ghost" size="icon" onClick={() => setCurrentTime(totalDuration)}>
                <SkipForward className="h-4 w-4" />
              </Button>
            </div>

            <div className="text-sm font-mono text-muted-foreground">
              {tracks.length > 0
                ? <>{formatDuration(timelineStart + currentTime)} / {formatDuration(timelineEnd)}<span className="ml-2 text-xs opacity-50">(+{formatDuration(currentTime)})</span></>
                : <span className="opacity-40">Loading…</span>
              }
            </div>

            <div className="flex-1">
              <Slider
                value={[currentTime]}
                max={totalDuration}
                step={0.5}
                onValueChange={(v) => setCurrentTime(v[0])}
              />
            </div>

            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setZoom(Math.max(0.5, zoom - 0.25))}
              >
                <ZoomOut className="h-4 w-4" />
              </Button>
              <span className="text-sm w-14 text-center font-mono">{(zoom * 100).toFixed(0)}%</span>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setZoom(Math.min(4, zoom + 0.25))}
              >
                <ZoomIn className="h-4 w-4" />
              </Button>
            </div>
          </div>

          {/* Timeline tracks */}
          <div className="flex-1 overflow-hidden bg-muted/10">
            <ScrollArea className="h-full">
              <div ref={timelineRef} className="p-4 min-w-max">
                {/* Time ruler */}
                <div className="h-6 mb-2 relative border-b border-muted">
                  {Array.from({ length: Math.ceil(totalDuration / rulerTickInterval) + 1 }).map((_, i) => {
                    const absTime = timelineStart + i * rulerTickInterval;
                    return (
                      <div
                        key={i}
                        className="absolute flex flex-col items-center"
                        style={{ left: timeToPixel(absTime) }}
                      >
                        <div className="h-3 w-px bg-muted-foreground/30" />
                        <span className="text-[10px] text-muted-foreground font-mono">
                          {formatDuration(absTime)}
                        </span>
                      </div>
                    );
                  })}
                  {/* Playhead on ruler — currentTime is offset from timelineStart */}
                  <div
                    className="absolute top-0 h-3 w-0.5 bg-red-500"
                    style={{ left: timeToPixel(timelineStart + currentTime) }}
                  />
                </div>

                {/* Track rows */}
                <div className="space-y-1">
                  {cameraLanes.map((lane) => (
                    <TimelineRow
                      key={lane.id}
                      lane={lane}
                      totalDuration={totalDuration}
                      timelineEnd={timelineEnd}
                      isSelected={selectedLaneId === lane.id}
                      onClick={() => setSelectedLaneId(selectedLaneId === lane.id ? null : lane.id)}
                      timeToPixel={timeToPixel}
                      currentTime={currentTime}
                      timelineStart={timelineStart}
                      videoId={currentVideo?.id}
                      runId={runId ?? undefined}
                    />
                  ))}
                </div>
              </div>
              <ScrollBar orientation="horizontal" />
            </ScrollArea>
          </div>
        </div>
      </div>
    </div>
  );
}

function CameraPreview({
  camera,
  isActive,
  isPast,
  isNext,
  currentTime,
  absCurrentTime,
  isPlaying,
  primarySeg,
  probeRunId,
  videoId,
  runId,
}: {
  camera: { id: string; name: string; location: string; activeTrack?: any };
  isActive: boolean;
  isPast?: boolean;
  isNext?: boolean;
  currentTime: number;
  absCurrentTime: number;
  isPlaying: boolean;
  primarySeg?: { globalId?: number; cameraId: string; trackId: number; start: number; end: number };
  probeRunId?: string;
  videoId?: string;
  runId?: string;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);

  // Derive clip URL directly from segment metadata — no async state needed.
  // Pattern: outputs/{probeRunId}/matched/global_{gid}_cam_{cameraId}_track_{tid}.mp4
  const clipUrl = (() => {
    if (!probeRunId || !primarySeg) {
      console.log(`[CameraPreview] ${camera.id}: no clip — probeRunId=${probeRunId} primarySeg=`, primarySeg);
      return null;
    }
    const { globalId, cameraId, trackId } = primarySeg;
    if (globalId == null || trackId == null) {
      console.log(`[CameraPreview] ${camera.id}: no clip — globalId=${globalId} trackId=${trackId}`);
      return null;
    }
    const filename = `global_${globalId}_cam_${cameraId}_track_${trackId}.mp4`;
    const url = `${API_BASE}/runs/${probeRunId}/matched_clips/${filename}`;
    console.log(`[CameraPreview] ${camera.id}: clipUrl=`, url);
    return url;
  })();

  const clipStartSec = primarySeg?.start ?? 0;

  // Seek to correct position once metadata is loaded (can't seek before this)
  useEffect(() => {
    const v = videoRef.current;
    if (!v || !clipUrl) return;
    const onCanPlay = () => {
      const seekTo = Math.max(0, absCurrentTime - clipStartSec);
      v.currentTime = seekTo;
    };
    v.addEventListener("canplay", onCanPlay, { once: true });
    return () => v.removeEventListener("canplay", onCanPlay);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clipUrl]); // only re-register when URL changes

  // Keep video in sync with timeline playhead scrubbing
  useEffect(() => {
    const v = videoRef.current;
    if (!v || !clipUrl) return;
    const seekTo = Math.max(0, absCurrentTime - clipStartSec);
    if (Math.abs(v.currentTime - seekTo) > 0.15) {
      v.currentTime = seekTo;
    }
  }, [absCurrentTime, clipUrl, clipStartSec]);

  // Play/pause in sync with timeline
  useEffect(() => {
    const v = videoRef.current;
    if (!v || !clipUrl) return;
    if (isPlaying) {
      v.play().catch(() => {});
    } else {
      v.pause();
    }
  }, [isPlaying, clipUrl]);
  // Build a crop URL from the active track's representative frame
  const cropUrl = (() => {
    if (!isActive || !camera.activeTrack) return null;
    const t = camera.activeTrack;
    const bbox = t.representativeBbox;
    const frameId = t.representativeFrame;
    if (frameId == null) return null;
    const bboxParams = (bbox && bbox.length === 4)
      ? `x1=${bbox[0]}&y1=${bbox[1]}&x2=${bbox[2]}&y2=${bbox[3]}`
      : "x1=0&y1=0&x2=9999&y2=9999";
    if (runId && shouldUseRunCropsForCamera(runId, t.cameraId)) {
      return `${API_BASE}/crops/run/${runId}?cameraId=${t.cameraId}&frameId=${frameId}&${bboxParams}`;
    }
    if (videoId) {
      return `${API_BASE}/crops/${videoId}?frameId=${frameId}&${bboxParams}`;
    }
    return null;
  })();

  const ringClass = clipUrl
    ? isActive
      ? "ring-2 ring-green-500"
      : isPast
      ? "ring-1 ring-orange-500/60 opacity-70"
      : isNext
      ? "ring-1 ring-blue-400/60 opacity-70"
      : "opacity-50"
    : isActive
    ? "ring-2 ring-green-500"
    : isPast
    ? "ring-1 ring-orange-400/50 opacity-50"
    : isNext
    ? "ring-1 ring-blue-400/50 opacity-40"
    : "opacity-30";

  const statusLabel = isActive ? null : isPast ? "PAST" : isNext ? "NEXT" : null;
  const statusColor = isPast ? "text-orange-400" : "text-blue-400";

  return (
    <div className={cn("relative rounded overflow-hidden transition-all", ringClass)}>
      {/* Camera feed — video clip, static crop, or placeholder */}
      <div className="absolute inset-0 bg-gradient-to-b from-slate-700 via-slate-800 to-slate-900">
        {clipUrl ? (
          <video
            ref={videoRef}
            src={clipUrl}
            className="absolute inset-0 w-full h-full object-cover"
            muted
            playsInline
            preload="auto"
            onError={(e) => console.error(`[CameraPreview] ${camera.id}: video error`, (e.target as HTMLVideoElement).error)}
            onLoadedData={() => console.log(`[CameraPreview] ${camera.id}: video loaded OK`)}
          />
        ) : cropUrl ? (
          <img
            src={cropUrl}
            alt={camera.id}
            className="absolute inset-0 w-full h-full object-cover"
          />
        ) : (
          <>
            <svg className="absolute inset-0 w-full h-full opacity-20" preserveAspectRatio="none">
              <line x1="50%" y1="30%" x2="20%" y2="100%" stroke="white" strokeWidth="1" />
              <line x1="50%" y1="30%" x2="80%" y2="100%" stroke="white" strokeWidth="1" />
            </svg>
            {isActive && camera.activeTrack && (
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">
                <div
                  className="w-10 h-6 rounded border-2 flex items-center justify-center"
                  style={{ borderColor: camera.activeTrack.color || "#22c55e", backgroundColor: `${camera.activeTrack.color || "#22c55e"}33` }}
                >
                  <Car className="h-4 w-4 text-white" />
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {/* Camera info overlay */}
      <div className="absolute top-0 left-0 right-0 p-1 bg-black/60">
        <div className="flex items-center gap-1">
          <div className={cn("h-1.5 w-1.5 rounded-full", isActive ? "bg-green-500 animate-pulse" : isPast ? "bg-orange-400" : isNext ? "bg-blue-400" : "bg-gray-500")} />
          <span className="text-white text-[10px] font-mono">{camera.id}</span>
          {statusLabel && (
            <span className={cn("text-[9px] font-bold ml-auto", statusColor)}>{statusLabel}</span>
          )}
        </div>
      </div>

      {/* Timestamp */}
      <div className="absolute bottom-0 left-0 right-0 p-1 bg-black/60">
        <span className="text-white/70 text-[9px] font-mono">
          {formatDuration(currentTime)}
        </span>
      </div>
    </div>
  );
}
interface TrackletItemProps {
  track: TimelineTrack;
  isSelected: boolean;
  onClick: () => void;
  onConfirm: () => void;
  onRemove: () => void;
}

function TrackletItem({ track, isSelected, onClick, onConfirm, onRemove }: TrackletItemProps) {
  const nCams = track.segments ? new Set(track.segments.map((s) => s.cameraId)).size : 1;
  const primaryColor = track.segments?.[0]?.color ?? getCameraColor(track.cameraId);

  return (
    <div
      className={cn(
        "p-2 rounded-lg border cursor-pointer transition-all",
        isSelected && "border-primary bg-primary/5",
        track.confirmed && !isSelected && "border-green-500/50 bg-green-500/5"
      )}
      onClick={onClick}
    >
      <div className="flex items-center gap-2">
        {/* mini camera-strip: up to 3 colored dots per camera */}
        <div className="flex gap-0.5 flex-shrink-0">
          {track.segments
            ? Array.from(new Set(track.segments.map((s) => s.cameraId))).slice(0, 4).map((cid) => (
                <div key={cid} className="h-3 w-1.5 rounded-full" style={{ backgroundColor: getCameraColor(cid) }} />
              ))
            : <div className="h-3 w-3 rounded-full" style={{ backgroundColor: primaryColor }} />}
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-xs font-medium truncate">{track.label ?? track.cameraId}</p>
          <p className="text-[10px] text-muted-foreground">
            {formatDuration(track.startTime)} → {formatDuration(track.endTime)}
            {nCams > 1 && <span className="ml-1 text-blue-400">· {nCams} cams</span>}
          </p>
          {typeof track.confidence === "number" && track.confidence > 0 && (
            <p className="text-[9px] text-muted-foreground">
              confidence: {(track.confidence * 100).toFixed(0)}%
            </p>
          )}
        </div>
        <div className="flex gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6"
            onClick={(e) => {
              e.stopPropagation();
              onConfirm();
            }}
          >
            <Check className={cn("h-3 w-3", track.confirmed ? "text-green-500" : "text-muted-foreground")} />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6"
            onClick={(e) => {
              e.stopPropagation();
              onRemove();
            }}
          >
            <X className="h-3 w-3 text-muted-foreground hover:text-red-500" />
          </Button>
        </div>
      </div>
    </div>
  );
}

interface TimelineRowProps {
  lane: {
    id: string;
    cameraId: string;
    label: string;
    segments: Array<TrajectorySegment & {
      trajectoryId: string;
      globalId?: number;
      confidence?: number;
      className?: string;
      confirmed?: boolean;
    }>;
  };
  totalDuration: number;
  timelineEnd: number;
  isSelected: boolean;
  onClick: () => void;
  timeToPixel: (time: number) => number;
  currentTime: number;
  timelineStart: number;
  videoId?: string;
  runId?: string;
}

/**
 * Camera-lane TimelineRow.
 * Renders ONE row per camera, with all matched trajectory segments on that lane.
 */
function TimelineRow({ lane, totalDuration, timelineEnd, isSelected, onClick, timeToPixel, currentTime, timelineStart, videoId, runId }: TimelineRowProps) {
  // currentTime is offset from timelineStart; convert to absolute for segment comparisons
  const absoluteTime = timelineStart + currentTime;
  const isCurrentlyActive = lane.segments.some((seg) => absoluteTime >= seg.start && absoluteTime <= seg.end);
  const segments = lane.segments;

  return (
    <div
      className={cn(
        "h-10 relative rounded border bg-muted/20 cursor-pointer transition-all select-none",
        isSelected && "border-primary ring-1 ring-primary",
        segments.some((s) => s.confirmed) && !isSelected && "border-green-500/30",
        isCurrentlyActive && "bg-primary/5"
      )}
      style={{ width: timeToPixel(timelineEnd) }}
      onClick={onClick}
      title={lane.label}
    >
      {/* Per-camera colored segments — notebook-style: colored blocks arranged along the timeline */}
      {segments.map((seg, i) => {
        const segLeft = timeToPixel(seg.start);
        const segWidth = Math.max(timeToPixel(seg.end) - timeToPixel(seg.start), 4);
        const isSegActive = absoluteTime >= seg.start && absoluteTime <= seg.end;

        // Build crop URL for this segment's representative frame
        const cropUrl = (() => {
          const bbox = seg.representativeBbox;
          const frameId = seg.representativeFrame;
          if (frameId == null) return null;
          const bboxParams = (bbox && bbox.length === 4)
            ? `x1=${bbox[0]}&y1=${bbox[1]}&x2=${bbox[2]}&y2=${bbox[3]}`
            : "x1=0&y1=0&x2=9999&y2=9999";
          if (runId && shouldUseRunCropsForCamera(runId, seg.cameraId)) {
            return `${API_BASE}/crops/run/${runId}?cameraId=${seg.cameraId}&frameId=${frameId}&${bboxParams}`;
          }
          if (videoId) return `${API_BASE}/crops/${videoId}?frameId=${frameId}&${bboxParams}`;
          return null;
        })();

        return (
          <div
            key={`${seg.cameraId}-${seg.trackId}-${seg.trajectoryId}-${i}`}
            className={cn(
              "absolute top-1 bottom-1 rounded overflow-hidden transition-all",
              isSegActive && "ring-1 ring-white/60",
              seg.confirmed && "ring-1 ring-green-500/50"
            )}
            style={{
              left: segLeft,
              width: segWidth,
              backgroundColor: seg.color,
              opacity: isSegActive ? 1 : 0.75,
            }}
            title={`Cam ${seg.cameraId} | G-${String(seg.globalId ?? 0).padStart(4, "0")} | ${formatDuration(seg.start)}–${formatDuration(seg.end)}`}
          >
            {/* Representative crop image */}
            {cropUrl && (
              <img
                src={cropUrl}
                alt={seg.cameraId}
                className="absolute inset-0 w-full h-full object-cover opacity-70"
                loading="lazy"
              />
            )}
            {/* Camera ID label inside segment (only if segment is wide enough) */}
            {segWidth > 32 && (
              <span className="absolute bottom-0.5 left-0.5 text-[8px] font-mono text-white/90 leading-none bg-black/30 px-0.5 rounded-sm">
                {`G${seg.globalId ?? "?"}`}
              </span>
            )}
          </div>
        );
      })}

      {/* Playhead indicator on this row — use absolute time */}
      {isCurrentlyActive && (
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-red-500 z-20 pointer-events-none"
          style={{ left: timeToPixel(absoluteTime) }}
        />
      )}

      {/* Lane label badge (right edge) */}
      <div className="absolute right-1 top-1/2 -translate-y-1/2 text-[8px] text-white/70 font-mono z-10 pointer-events-none bg-black/20 rounded px-1">
        {lane.cameraId}
      </div>

      {/* Confidence badge (highest segment confidence in lane, subtle) */}
      {(() => {
        const confs = segments.map((s) => Number(s.confidence ?? 0)).filter((v) => Number.isFinite(v) && v > 0);
        const best = confs.length > 0 ? Math.max(...confs) : 0;
        return best > 0 ? (
        <div className="absolute right-1 top-1/2 -translate-y-1/2 text-[8px] text-white/60 font-mono z-10 pointer-events-none">
            {(best * 100).toFixed(0)}%
        </div>
        ) : null;
      })()}
    </div>
  );
}
