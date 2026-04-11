"use client";

import {
  useState,
  useCallback,
  useRef,
  useEffect,
  useMemo,
} from "react";
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
  Loader2,
} from "lucide-react";
import { cn, formatDuration, getCameraColor } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
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
import {
  getTracklets,
  getTrajectories,
  runStage,
  getPipelineStatus,
  queryTimeline,
  getMatchedSummary,
  getMatchedAlternatives,
  getMatchedAlternativeClipUrl,
  getTrackletSequence,
  getRunFullFrameUrl,
  type MatchedAlternative,
  type TrackletSequenceFrame,
} from "@/lib/api";
import type { TimelineTrack, TrajectorySegment } from "@/types";
import { TrackletFrameView } from "@/components/ui/double-buffered-img";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

/** Playhead & ruler updates/sec while playing — match typical display cadence for video-like motion. */
const TIMELINE_PLAYHEAD_FPS = 30;
/** Tracklet full-frame picks/sec while playing (lower than playhead to limit image decode load). */
const TRACKLET_PICK_FPS = 15;
const TRACKLET_PICK_BUCKET_SEC = 1 / TRACKLET_PICK_FPS;

function shouldUseRunCropsForCamera(runId: string | undefined, _cameraId: string): boolean {
  if (!runId) return false;
  // Dataset precompute runs are scene-scoped; lane labels are often `c006` without a scene prefix,
  // so comparing scene from the camera id string falsely disabled all run crops.
  return true;
}

/** Any segment (wall-clock) contains video time `t`. */
function trackIsActiveAtVideoTime(track: TimelineTrack, videoTime: number): boolean {
  const segs =
    track.segments && track.segments.length > 0
      ? track.segments
      : [{ start: track.startTime, end: track.endTime }];
  return segs.some((s) => videoTime >= s.start && videoTime <= s.end);
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
  const { selectedTrackIds: selectedTrackIdSet } = useDetectionStore();

  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [selectedLaneId, setSelectedLaneId] = useState<string | null>(null);
  const [splitCount, setSplitCount] = useState<number | null>(null);
  const [showProgress, setShowProgress] = useState(true);
  const [triggerReload, setTriggerReload] = useState(0);
  const [matchedFallbackActive, setMatchedFallbackActive] = useState(false);
  const [tracksLoading, setTracksLoading] = useState(false);
  const [topAlternatives, setTopAlternatives] = useState<MatchedAlternative[]>([]);
  const [alternativesLoading, setAlternativesLoading] = useState(false);
  const [alternativesError, setAlternativesError] = useState<string | null>(null);
  const [alternativesCameraCount, setAlternativesCameraCount] = useState(0);
  const [alternativeHistoryByTrackId, setAlternativeHistoryByTrackId] = useState<Record<string, MatchedAlternative[]>>({});
  const [currentAlternativeByTrackId, setCurrentAlternativeByTrackId] = useState<Record<string, MatchedAlternative>>({});
  const [playingTrackletsOnly, setPlayingTrackletsOnly] = useState(false);
  const timelineRef = useRef<HTMLDivElement>(null);
  const trajectoryListRef = useRef<HTMLDivElement>(null);

  type CameraLaneSegment = TrajectorySegment & {
    trajectoryId: string;
    globalId?: number;
    confidence?: number;
    className?: string;
    confirmed?: boolean;
  };

  type CameraLaneSegmentWithSum = CameraLaneSegment & { sumStart: number; sumEnd: number };

  type CameraLane = {
    id: string;
    cameraId: string;
    label: string;
    startTime: number;
    endTime: number;
    segments: CameraLaneSegmentWithSum[];
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

  // Ruler 0…T where T is the sum of every tracklet segment duration. Segments are placed
  // end-to-end in wall-clock order (per global sort by segment start). Playhead value is
  // that combined time; sumOffsetToVideoTime maps it back to source video time for previews.
  const {
    cameraLanes,
    totalDuration,
    timelineStart,
    timelineEnd,
    sumOffsetToVideoTime,
  } = useMemo(() => {
    const videoDur =
      currentVideo?.duration && currentVideo.duration > 0 ? currentVideo.duration : null;

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

    const lanesDraft: Array<{
      id: string;
      cameraId: string;
      label: string;
      startTime: number;
      endTime: number;
      segments: CameraLaneSegment[];
    }> = [];
    laneMap.forEach((segments, cameraId) => {
      const sorted = [...segments].sort((a, b) => a.start - b.start);
      const start = Math.min(...sorted.map((s) => s.start));
      const end = Math.max(...sorted.map((s) => s.end));
      lanesDraft.push({
        id: `lane-${cameraId}`,
        cameraId,
        label: cameraId,
        startTime: start,
        endTime: end,
        segments: sorted,
      });
    });

    lanesDraft.sort((a, b) => {
      if (a.startTime !== b.startTime) return a.startTime - b.startTime;
      return a.cameraId.localeCompare(b.cameraId);
    });

    if (tracks.length === 0) {
      const dur = Math.max(videoDur ?? 1, 1);
      return {
        cameraLanes: [] as CameraLane[],
        totalDuration: dur,
        timelineStart: 0,
        timelineEnd: dur,
        sumOffsetToVideoTime: (offset: number) => offset,
      };
    }

    const flat: { seg: CameraLaneSegment }[] = [];
    lanesDraft.forEach((lane) => {
      lane.segments.forEach((seg) => {
        flat.push({ seg });
      });
    });
    flat.sort((a, b) => {
      const ds = a.seg.start - b.seg.start;
      if (ds !== 0) return ds;
      const de = a.seg.end - b.seg.end;
      if (de !== 0) return de;
      const ka = `${a.seg.trajectoryId}\0${a.seg.trackId}`;
      const kb = `${b.seg.trajectoryId}\0${b.seg.trackId}`;
      return ka.localeCompare(kb);
    });

    const sumRanges: { sumStart: number; sumEnd: number; seg: CameraLaneSegment }[] = [];
    let cum = 0;
    for (const { seg } of flat) {
      const len = Math.max(seg.end - seg.start, 1e-6);
      sumRanges.push({ sumStart: cum, sumEnd: cum + len, seg });
      cum += len;
    }
    const totalDur = Math.max(cum, 1);

    const keyToSum = new Map<string, { sumStart: number; sumEnd: number }>();
    for (const r of sumRanges) {
      const s = r.seg;
      const key = `${s.cameraId}|${s.trajectoryId}|${s.trackId}|${s.start}|${s.end}`;
      keyToSum.set(key, { sumStart: r.sumStart, sumEnd: r.sumEnd });
    }

    const lanesWithSum: CameraLane[] = lanesDraft.map((lane) => ({
      ...lane,
      segments: lane.segments.map((seg) => {
        const key = `${seg.cameraId}|${seg.trajectoryId}|${seg.trackId}|${seg.start}|${seg.end}`;
        const sum = keyToSum.get(key)!;
        return { ...seg, sumStart: sum.sumStart, sumEnd: sum.sumEnd };
      }),
    }));

    const sumOffsetToVideoTime = (offset: number): number => {
      if (sumRanges.length === 0) return 0;
      const o = Math.min(Math.max(offset, 0), totalDur);
      if (o >= totalDur - 1e-9) return sumRanges[sumRanges.length - 1].seg.end;
      for (const r of sumRanges) {
        if (o >= r.sumStart && o < r.sumEnd) {
          const slen = r.sumEnd - r.sumStart;
          const wallLen = Math.max(r.seg.end - r.seg.start, 1e-6);
          const local = slen > 0 ? (o - r.sumStart) / slen : 0;
          return r.seg.start + local * wallLen;
        }
      }
      return sumRanges[sumRanges.length - 1].seg.end;
    };

    return {
      cameraLanes: lanesWithSum,
      totalDuration: totalDur,
      timelineStart: 0,
      timelineEnd: totalDur,
      sumOffsetToVideoTime,
    };
  }, [tracks, currentVideo?.duration]);

  const allCamerasForPreview = useMemo(
    () =>
      cameraLanes.map((lane) => ({
        id: lane.cameraId,
        scene: lane.cameraId.split("_")[0] ?? "Unknown",
        name: lane.cameraId,
        location: "Camera",
      })),
    [cameraLanes]
  );

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

    const rows: TimelineTrack[] = [];
    for (const clip of clips) {
      const gid = Number(clip.global_id ?? 0);
      const cam = String(clip.camera_id);
      const tid = Number(clip.track_id);
      const start = Number(clip.start_time_s ?? 0);
      const end = Number(clip.end_time_s ?? (start + Number(clip.duration_s ?? 0.1)));
      const seg = {
        cameraId: cam,
        trackId: tid,
        globalId: gid,
        start,
        end,
        color: getCameraColor(cam),
      };
      rows.push({
        id: `fallback-${gid}-${cam}-${tid}`,
        cameraId: cam,
        trackletId: tid,
        globalId: gid,
        startTime: start,
        endTime: Math.max(end, start + 0.1),
        selected: false,
        confirmed: false,
        segments: [seg],
        label: `G-${String(gid).padStart(4, "0")} · ${cam} · track ${tid}`,
        confidence: Number(clip.confidence ?? 0),
      });
    }
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

      // Determine dominant class from trajectory (backend may include class_name)
      const className: string = trajectory.class_name ?? trajectory.className ?? "vehicle";
      const confidence: number = selectedTrackKeys && selectedTrackKeys.size > 0
        ? scoreTrajectoryForQuery(trajectory, selectedTrackKeys)
        : (typeof trajectory.confidence === "number" ? trajectory.confidence : 1);

      // One row per camera segment so the user can confirm/reject each individually
      segments.forEach((seg) => {
        rows.push({
          id: `traj-${globalId}-${seg.cameraId}-${seg.trackId}`,
          cameraId: seg.cameraId,
          trackletId: seg.trackId,
          globalId,
          startTime: seg.start,
          endTime: seg.end,
          selected: false,
          confirmed: false,
          representativeFrame: seg.representativeFrame,
          representativeBbox: seg.representativeBbox,
          segments: [seg],
          label: `G-${String(globalId).padStart(4, "0")} · ${seg.cameraId} · track ${seg.trackId}`,
          confidence,
          className,
        });
      });
    });

    return rows;
  };

  // Stage 4: run real association, then load tracks
  useEffect(() => {
    let cancelled = false;
    if (!currentVideo) {
      setTracksLoading(false);
      return;
    }
    setTracksLoading(true);

    const loadTracks = async () => {
        try {
        let attemptedAssociation = false;
        const selectedTrackIdsArr = Array.from(selectedTrackIdSet).map((v) => String(v));

        console.groupCollapsed("[Stage4][Timeline] loadTracks");
        console.info("context", {
          runId,
          videoId: currentVideo.id,
          selectedTrackletCount: selectedTrackIdsArr.length,
          selectedTrackIds: selectedTrackIdsArr,
        });

        // Query mode: resolve selected tracklets to matched trajectories on backend.
        // Use galleryRunId (precomputed dataset) when available; fall back to probe runId.
        const effectiveGalleryRunId = galleryRunId ?? runId;
        if (effectiveGalleryRunId && selectedTrackIdSet.size > 0) {
          attemptedAssociation = true;
          const q1 = await queryTimeline(effectiveGalleryRunId, currentVideo.id, selectedTrackIdsArr);
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
              const q2 = await queryTimeline(stage4RunId ?? effectiveGalleryRunId, currentVideo.id, selectedTrackIdsArr);
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
        if (attemptedAssociation && selectedTrackIdSet.size > 0) {
          // Keep strict query behavior (do not show unrelated trajectories), but
          // avoid a blank timeline by falling back to only the selected stage-1 tracklets.
          const fallbackResp = await getTracklets(undefined, currentVideo.id);
          if (cancelled) return;
          let fallbackSummary = Array.isArray(fallbackResp.data) ? fallbackResp.data : [];

          const selectedTrackNums = new Set<number>();
          selectedTrackIdSet.forEach((trackId) => {
            selectedTrackNums.add(trackId); 
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
        if (selectedTrackIdSet.size > 0) {
          const selectedTrackNums = new Set<number>();
          selectedTrackIdSet.forEach((trackId) => {
            selectedTrackNums.add(trackId);
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
      } finally {
        if (!cancelled) setTracksLoading(false);
      }
    };

    void loadTracks();

    return () => {
      cancelled = true;
      setTracksLoading(false);
    };
  }, [currentVideo, runId, selectedTrackIdSet, setTracks, triggerReload]);

  // Playback: fixed UI rate so we don't re-render the whole stage at source video FPS.
  useEffect(() => {
    if (!isPlaying) return;
    const increment = 1 / TIMELINE_PLAYHEAD_FPS;
    const interval = setInterval(() => {
      setCurrentTime((t) => (t + increment >= totalDuration ? 0 : t + increment));
    }, 1000 / TIMELINE_PLAYHEAD_FPS);
    return () => clearInterval(interval);
  }, [isPlaying, totalDuration]);

  useEffect(() => {
    setCurrentTime((t) => Math.min(t, totalDuration));
  }, [totalDuration]);

  const selectedTrack = useMemo(
    () => tracks.find((track) => track.id === selectedTrackId) ?? null,
    [tracks, selectedTrackId]
  );

  const buildAlternativeFromTrack = useCallback(
    (track: TimelineTrack, source?: MatchedAlternative): MatchedAlternative => ({
      previewUrl: source?.previewUrl
        ?? (
          runId && track.globalId != null
            ? `${API_BASE}/runs/${encodeURIComponent(runId)}/matched_clips/${encodeURIComponent(
                `global_${track.globalId}_cam_${String(track.cameraId).replace(/[/\\]/g, "_")}_track_${track.trackletId}.mp4`
              )}`
            : undefined
        ),
      rank: 0,
      globalId: track.globalId ?? null,
      cameraId: track.cameraId,
      trackId: track.trackletId,
      score: Number(track.confidence ?? 0),
      confidence: Number(track.confidence ?? 0),
      numCameras: track.segments ? new Set(track.segments.map((s) => s.cameraId)).size : 1,
      className: track.className,
      startTime: track.startTime,
      endTime: track.endTime,
      representativeFrame: track.representativeFrame,
      representativeBbox: track.representativeBbox,
      label: track.label,
      clipPath: source?.clipPath ?? "",
      ok: true,
      message: "Pinned previous main tracklet",
    }),
    [runId]
  );

  const mergeWithHistoryAlternatives = useCallback(
    (
      list: MatchedAlternative[],
      selected: TimelineTrack,
      history: MatchedAlternative[]
    ): MatchedAlternative[] => {
      const selectedKey = `${selected.cameraId}:${selected.trackletId}`;
      const keyToIndex = new Map<string, number>();
      const merged: MatchedAlternative[] = [];

      const mediaScore = (item: MatchedAlternative): number => {
        return item.previewUrl || item.clipPath ? 1 : 0;
      };

      const pushIfValid = (item: MatchedAlternative) => {
        const key = `${item.cameraId}:${item.trackId}`;
        if (key === selectedKey) return;

        const existingIdx = keyToIndex.get(key);
        if (existingIdx == null) {
          keyToIndex.set(key, merged.length);
          merged.push(item);
          return;
        }

        const existing = merged[existingIdx];
        if (mediaScore(item) > mediaScore(existing)) {
          merged[existingIdx] = item;
        }
      };

      history.forEach(pushIfValid);
      list.forEach(pushIfValid);

      return merged.slice(0, 5).map((a, i) => ({ ...a, rank: i + 1 }));
    },
    []
  );

  useEffect(() => {
    if (!runId || !selectedTrack) {
      setTopAlternatives([]);
      setAlternativesCameraCount(0);
      setAlternativesError(null);
      setAlternativesLoading(false);
      return;
    }

    let cancelled = false;
    setAlternativesLoading(true);
    setAlternativesError(null);

    void (async () => {
      try {
        const history = selectedTrackId ? (alternativeHistoryByTrackId[selectedTrackId] ?? []) : [];
        const payload = await getMatchedAlternatives(runId, {
          topK: 5,
          anchorCameraId: selectedTrack.cameraId,
          anchorTrackId: selectedTrack.trackletId,
          excludeGlobalId: selectedTrack.globalId,
          excludeCameraId: selectedTrack.cameraId,
          excludeTrackId: selectedTrack.trackletId,
        });
        if (cancelled) return;
        setAlternativesCameraCount(payload.totalCameras);

        const playable = payload.alternatives
          .filter((item) => item.ok && Boolean(item.clipPath))
          .slice(0, 5);
        setTopAlternatives(mergeWithHistoryAlternatives(playable, selectedTrack, history));
      } catch (err: any) {
        if (cancelled) return;
        const history = selectedTrackId ? (alternativeHistoryByTrackId[selectedTrackId] ?? []) : [];
        setTopAlternatives(mergeWithHistoryAlternatives([], selectedTrack, history));
        const msg = String(err?.message ?? "");
        if (msg.includes("404")) {
          setAlternativesCameraCount(0);
          setAlternativesError("No alternatives exported for this run yet.");
        } else {
          setAlternativesCameraCount(0);
          setAlternativesError("Failed to load alternatives for selected tracklet.");
        }
      } finally {
        if (!cancelled) setAlternativesLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [runId, selectedTrack, selectedTrackId, alternativeHistoryByTrackId, mergeWithHistoryAlternatives]);

  const handleApplyAlternative = useCallback(
    (alt: MatchedAlternative) => {
      if (!selectedTrack || !selectedTrackId) return;

      const selectedKey = `${selectedTrack.cameraId}:${selectedTrack.trackletId}`;
      const sourceForCurrent =
        currentAlternativeByTrackId[selectedTrackId]
        ?? topAlternatives.find((x) => `${x.cameraId}:${x.trackId}` === selectedKey);

      setAlternativeHistoryByTrackId((prev) => {
        const current = prev[selectedTrackId] ?? [];
        const candidate = buildAlternativeFromTrack(selectedTrack, sourceForCurrent);
        const candidateKey = `${candidate.cameraId}:${candidate.trackId}`;
        const deduped = [candidate, ...current.filter((x) => `${x.cameraId}:${x.trackId}` !== candidateKey)];
        return {
          ...prev,
          [selectedTrackId]: deduped.slice(0, 8),
        };
      });

      setCurrentAlternativeByTrackId((prev) => ({
        ...prev,
        [selectedTrackId]: alt,
      }));

      const start = Number.isFinite(alt.startTime) ? Number(alt.startTime) : selectedTrack.startTime;
      const endCandidate = Number.isFinite(alt.endTime) ? Number(alt.endTime) : selectedTrack.endTime;
      const end = Math.max(endCandidate, start + 0.1);

      const seg = {
        cameraId: alt.cameraId,
        trackId: alt.trackId,
        globalId: alt.globalId ?? undefined,
        start,
        end,
        color: getCameraColor(alt.cameraId),
        representativeFrame: alt.representativeFrame,
        representativeBbox: alt.representativeBbox,
      };

      const updated = tracks.map((t) => {
        if (t.id !== selectedTrackId) return t;
        const fallbackLabel = `G-${String(alt.globalId ?? 0).padStart(4, "0")} \u00b7 ${alt.cameraId} \u00b7 track ${alt.trackId}`;
        return {
          ...t,
          cameraId: alt.cameraId,
          trackletId: alt.trackId,
          globalId: alt.globalId ?? undefined,
          startTime: start,
          endTime: end,
          segments: [seg],
          representativeFrame: alt.representativeFrame,
          representativeBbox: alt.representativeBbox,
          label: alt.label ?? fallbackLabel,
          className: alt.className ?? t.className,
          confidence: Number.isFinite(alt.score) ? alt.score : t.confidence,
        };
      });

      setTracks(updated);
    },
    [buildAlternativeFromTrack, currentAlternativeByTrackId, selectedTrack, selectedTrackId, setTracks, topAlternatives, tracks]
  );


  const stage4Progress = stages.find((s) => s.stage === 4);

  // timeToPixel maps ruler time (combined tracklet duration offset, or wall time when no tracks)
  // to horizontal pixels (0 px = timelineStart).
  const timeToPixel = useCallback(
    (time: number) => {
      const baseWidth = 1200;
      return ((time - timelineStart) / totalDuration) * baseWidth * zoom;
    },
    [zoom, totalDuration, timelineStart]
  );

  // offsetToTime: inverse — pixel offset → ruler time
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
    setTracksLoading(true);
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
      setTriggerReload((n) => n + 1);
    } catch (e) {
      updateStageProgress(4, { status: "error", progress: 0, message: String(e) });
      setTracksLoading(false);
    }
  };

  const confirmedCount = tracks.filter((t) => t.confirmed).length;
  const timelineDataSource = tracks.some((t) => t.id.startsWith("real-") || t.id.startsWith("traj-")) ? "real" : "demo";

  // For summary badge: how many trajectories are shown vs total selected tracklets
  const shownTrajectories = tracks.length;
  const shownCameraLanes = cameraLanes.length;
  const selectedTrackletCount = selectedTrackIdSet.size;

  // Dynamic time ruler tick interval: keep ~10–15 ticks on screen regardless of duration
  const rulerTickInterval = totalDuration <= 30 ? 5 : totalDuration <= 120 ? 10 : totalDuration <= 600 ? 30 : 60;
  const rulerTickCount = Math.ceil(totalDuration / rulerTickInterval) + 1;
  const rulerPlayheadLeft = timeToPixel(timelineStart + currentTime);

  // For the selected trajectory, find which cameras are active at currentTime.
  // Also determine "past" cameras (ended) and "next" cameras (not started yet)
  // to mirror the notebook visualization.
  const absCurrentTime =
    tracks.length > 0 ? sumOffsetToVideoTime(currentTime) : timelineStart + currentTime;
  const trackletPickTime = useMemo(() => {
    const step = TRACKLET_PICK_BUCKET_SEC;
    return Math.round(absCurrentTime / step) * step;
  }, [absCurrentTime]);

  const activeAtPlayheadIds = useMemo(() => {
    const ids = new Set<string>();
    for (const t of tracks) {
      if (trackIsActiveAtVideoTime(t, absCurrentTime)) ids.add(t.id);
    }
    return ids;
  }, [tracks, absCurrentTime]);

  const activeAtPlayheadSignature = useMemo(
    () => [...activeAtPlayheadIds].sort().join("|"),
    [activeAtPlayheadIds]
  );

  const trajectoryListTracks = useMemo(() => {
    if (!playingTrackletsOnly) return tracks;
    return tracks.filter((t) => activeAtPlayheadIds.has(t.id));
  }, [tracks, playingTrackletsOnly, activeAtPlayheadIds]);

  useEffect(() => {
    if (!playingTrackletsOnly || trajectoryListTracks.length === 0) return;
    const firstId = trajectoryListTracks[0].id;
    const root = trajectoryListRef.current;
    if (!root) return;
    const el = Array.from(root.querySelectorAll<HTMLElement>("[data-track-id]")).find(
      (node) => node.dataset.trackId === firstId
    );
    el?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [playingTrackletsOnly, activeAtPlayheadSignature, trajectoryListTracks]);

  /** When set, preview playback uses only segments from trajectories active at the playhead. */
  const playbackFilterActive = playingTrackletsOnly && activeAtPlayheadIds.size > 0;

  const camerasForPreview = useMemo(() => {
    if (!playbackFilterActive) return allCamerasForPreview;
    const activeCamIds = new Set<string>();
    for (const lane of cameraLanes) {
      const segs = lane.segments.filter((s) => activeAtPlayheadIds.has(s.trajectoryId));
      if (segs.some((s) => absCurrentTime >= s.start && absCurrentTime <= s.end)) {
        activeCamIds.add(lane.cameraId);
      }
    }
    const filtered = allCamerasForPreview.filter((c) => activeCamIds.has(c.id));
    return filtered.length > 0 ? filtered : allCamerasForPreview;
  }, [
    playbackFilterActive,
    allCamerasForPreview,
    cameraLanes,
    activeAtPlayheadIds,
    absCurrentTime,
  ]);

  const effectiveSplitCount = playingTrackletsOnly
    ? Math.min(camerasForPreview.length || 1, 8)
    : splitCount ?? Math.min(allCamerasForPreview.length || 1, 8);

  const visibleCameras = camerasForPreview.slice(0, effectiveSplitCount);

  const activeCamerasForGrid = visibleCameras.map((cam) => {
    const lane = cameraLanes.find((l) => l.cameraId === cam.id);
    const allSegs = lane?.segments ?? [];
    const laneSegments = playbackFilterActive
      ? allSegs.filter((s) => activeAtPlayheadIds.has(s.trajectoryId))
      : allSegs;
    const activeSegment = laneSegments.find((s) => absCurrentTime >= s.start && absCurrentTime <= s.end);
    const isPast = !activeSegment && laneSegments.length > 0 && laneSegments.every((s) => absCurrentTime > s.end);
    const isNext = !activeSegment && laneSegments.length > 0 && laneSegments.every((s) => absCurrentTime < s.start);
    // Primary segment: active now, OR first segment in lane (for NEXT), OR last segment (for PAST)
    const primarySeg = activeSegment
      ?? (isNext ? laneSegments[0] : undefined)
      ?? (isPast ? laneSegments[laneSegments.length - 1] : undefined);
    // Rep frame/bbox must come from primarySeg so PAST/NEXT cells can still show crops when video fails.
    const representativeFrame = primarySeg?.representativeFrame;
    const representativeBbox = primarySeg?.representativeBbox;
    const trackForPreview = primarySeg
      ? {
          ...cam,
          representativeFrame,
          representativeBbox,
          cameraId: cam.id,
          color: activeSegment?.color ?? primarySeg.color ?? laneSegments[0]?.color,
        }
      : undefined;
    return {
      ...cam,
      activeTrack: trackForPreview,
      isPast,
      isNext,
      segment: activeSegment,
      primarySeg,
    };
  });

  // Where stage0 frames live for dataset cameras (probe run often has no stage0 for gallery cams).
  const cropRunId = galleryRunId ?? runId ?? undefined;

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden">
      {/* Header */}
      <header className="flex shrink-0 flex-col gap-3 border-b px-4 py-3 sm:flex-row sm:items-start sm:justify-between sm:px-6">
        <div className="min-w-0">
          <h1 className="text-lg font-semibold">Stage 4: Cross-Camera Timeline</h1>
          <p className="text-sm text-muted-foreground">
            DeepOCSORT tracklet association across CityFlow cameras
          </p>
        </div>
        <div className="flex flex-shrink-0 flex-wrap items-center justify-end gap-2">
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
              className="shrink-0"
              variant="outline"
              disabled={tracksLoading}
              onClick={handleRerunAssociation}
            >
              <RefreshCw className={cn("mr-2 h-4 w-4", tracksLoading && "animate-spin")} />
              Rerun Association
          </Button>
          <Button className="shrink-0" onClick={handleProceed}>
            Continue to Refinement
            <ArrowRight className="ml-2 h-4 w-4" />
          </Button>
        </div>
      </header>

      {/* Main content */}
      <div className="flex min-h-0 min-w-0 flex-1 overflow-hidden">
        {/* Left panel */}
        <aside className="flex min-h-0 w-64 min-w-0 max-w-[40vw] shrink-0 flex-col overflow-hidden border-r bg-muted/20 sm:max-w-none">
          <div className="p-4 border-b">
            <h3 className="font-semibold flex items-center gap-2">
              <Layers className="h-4 w-4" />
              Association Progress
            </h3>
          </div>
          <div className="p-4">
            {stage4Progress?.status === "running" ? (
              <div className="space-y-3">
                <div className="flex min-w-0 justify-between gap-2 text-sm">
                  <span className="min-w-0 break-words text-xs">{stage4Progress.message}</span>
                  <span className="shrink-0 font-mono">{stage4Progress.progress}%</span>
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
            {playbackFilterActive && (
              <p className="text-[10px] text-muted-foreground mb-2 leading-snug">
                Playback: {camerasForPreview.length} camera{camerasForPreview.length !== 1 ? "s" : ""} with
                active tracklets now
              </p>
            )}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Cameras</span>
                <span className="text-sm font-medium">{effectiveSplitCount}</span>
              </div>
              {playingTrackletsOnly ? (
                <p className="text-[10px] text-muted-foreground leading-snug">
                  Grid size follows active cameras (slider off in adaptive mode).
                </p>
              ) : null}
              <Slider
                value={[Math.min(effectiveSplitCount, 8)]}
                min={1}
                max={8}
                step={1}
                disabled={playingTrackletsOnly}
                onValueChange={(v) => setSplitCount(v[0])}
              />
            </div>
          </div>

          <Separator />

          {/* Tracklet list */}
          <div
            ref={trajectoryListRef}
            className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden p-4"
          >
            <div className="mb-2 flex items-start gap-2">
              <Checkbox
                id="timeline-playing-only"
                className="mt-0.5"
                checked={playingTrackletsOnly}
                disabled={tracks.length === 0}
                onCheckedChange={(v) => setPlayingTrackletsOnly(v === true)}
              />
              <Label
                htmlFor="timeline-playing-only"
                className="text-xs font-normal leading-snug text-muted-foreground cursor-pointer"
              >
                Adaptive mode: trajectory list and preview grid follow the playhead — only identities
                and cameras with live segments at the current video time (updates while scrubbing or playing).
              </Label>
            </div>
            <div className="mb-2 flex flex-wrap items-center gap-2">
              <h4 className="text-sm font-medium">Trajectories</h4>
              {tracks.length > 0 && playingTrackletsOnly && (
                <Badge variant="outline" className="text-[10px] font-normal tabular-nums">
                  {trajectoryListTracks.length} at playhead
                </Badge>
              )}
            </div>
            {tracksLoading ? (
              <div className="mt-3 flex items-center gap-2 text-muted-foreground">
                <Loader2 className="h-4 w-4 shrink-0 animate-spin" />
                <span className="text-xs">Loading trajectories and previews…</span>
              </div>
            ) : tracks.length === 0 ? (
              <p className="text-xs text-muted-foreground mt-2">
                {selectedTrackletCount > 0
                  ? (stage4Progress?.message || "No trajectories match selected tracklets. Run Stage 4 to associate them.")
                  : "No tracklet data yet."}
              </p>
            ) : playingTrackletsOnly && trajectoryListTracks.length === 0 ? (
              <p className="text-xs text-amber-600/90 dark:text-amber-400/90 mt-2">
                No trajectory spans the current video time. Move the playhead to a segment.
              </p>
            ) : (
              <div className="space-y-2">
                {trajectoryListTracks.map((track) => (
                  <TrackletItem
                    key={track.id}
                    track={track}
                    isSelected={selectedTrackId === track.id}
                    isActiveAtPlayhead={
                      !playingTrackletsOnly && activeAtPlayheadIds.has(track.id)
                    }
                    onClick={() => handleTrackClick(track.id)}
                    onConfirm={() => handleConfirmToggle(track.id, track.confirmed)}
                    onRemove={() => removeTrack(track.id)}
                  />
                ))}
              </div>
            )}

            <Separator className="my-3" />
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-medium">Top 5 Alternatives</h4>
              {selectedTrack ? (
                <Badge variant="outline" className="text-[10px]">
                  {selectedTrack.cameraId} &middot; #{selectedTrack.trackletId} &middot; {alternativesCameraCount || "-"} cams
                </Badge>
              ) : null}
            </div>

            {!selectedTrack ? (
              <p className="mt-2 text-xs text-muted-foreground">
                Select a trajectory to load alternatives from matched/top5_alternatives.
              </p>
            ) : alternativesLoading ? (
              <div className="mt-2 flex items-center gap-2 text-muted-foreground">
                <Loader2 className="h-4 w-4 shrink-0 animate-spin" />
                <span className="text-xs">Loading top alternatives&hellip;</span>
              </div>
            ) : alternativesError ? (
              <p className="mt-2 text-xs text-muted-foreground">{alternativesError}</p>
            ) : topAlternatives.length === 0 ? (
              <p className="mt-2 text-xs text-muted-foreground">
                No alternative clips were found for this selection.
              </p>
            ) : (
              <div className="mt-2 space-y-2">
                {topAlternatives.map((alt) => (
                  <AlternativeTrackletItem
                    key={`${alt.rank}-${alt.cameraId}-${alt.trackId}-${alt.clipPath}`}
                    alternative={alt}
                    videoUrl={
                      alt.previewUrl
                        ? alt.previewUrl
                        : runId && alt.clipPath
                        ? getMatchedAlternativeClipUrl(runId, alt.clipPath)
                        : ""
                    }
                    onUse={() => handleApplyAlternative(alt)}
                  />
                ))}
              </div>
            )}
          </div>
        </aside>

        {/* Main timeline area */}
        <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
          {/* Video preview area — bounded height so timeline + header fit in viewport */}
          <div
            className="relative shrink-0 border-b bg-slate-900 p-2"
            style={{ height: "clamp(200px, min(42vh, 50dvh), 560px)" }}
          >
            <div
              className="grid h-full min-h-0 min-w-0 gap-1"
              style={{
                gridTemplateColumns: `repeat(${Math.ceil(Math.sqrt(effectiveSplitCount))}, minmax(0, 1fr))`,
                gridTemplateRows: `repeat(${Math.ceil(effectiveSplitCount / Math.ceil(Math.sqrt(effectiveSplitCount)))}, minmax(0, 1fr))`,
              }}
            >
              {activeCamerasForGrid.map((cam) => (
                <CameraPreview
                  key={cam.id}
                  camera={cam}
                  isActive={Boolean(cam.segment)}
                  isPast={cam.isPast}
                  isNext={cam.isNext}
                  absCurrentTime={absCurrentTime}
                  trackletPickTime={trackletPickTime}
                  isPlaying={isPlaying}
                  primarySeg={cam.primarySeg}
                  probeRunId={runId ?? undefined}
                  videoId={currentVideo?.id}
                  cropRunId={cropRunId}
                />
              ))}
            </div>
            {tracksLoading && (
              <div
                className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 rounded-md bg-slate-950/85 px-4"
                role="status"
                aria-live="polite"
                aria-label="Loading timeline previews"
              >
                <Loader2 className="h-10 w-10 animate-spin text-primary" />
                <p className="text-center text-sm text-muted-foreground">
                  Loading camera previews…
                </p>
              </div>
            )}
          </div>

          {/* Timeline controls */}
          <div className="flex shrink-0 flex-wrap items-end gap-3 border-b border-border/50 bg-background/80 p-3 backdrop-blur-sm sm:gap-4 sm:p-4">
            <div className="flex shrink-0 items-center gap-1">
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

            <div className="min-w-[140px] flex-1 basis-[min(100%,280px)]">
              <div className="mb-1 flex items-baseline justify-between gap-2">
                <span className="text-[11px] tabular-nums tracking-wide text-muted-foreground/70">
                  {tracks.length > 0 ? (
                    <>
                      <span className="text-foreground/85">{formatDuration(timelineStart + currentTime)}</span>
                      <span className="text-muted-foreground/35"> · </span>
                      {formatDuration(timelineEnd)}
                    </>
                  ) : (
                    "…"
                  )}
                </span>
              </div>
              <Slider
                tone="muted"
                value={[currentTime]}
                max={totalDuration}
                step={0.5}
                onValueChange={(v) => setCurrentTime(v[0])}
                title={
                  tracks.length > 0
                    ? `Combined ${formatDuration(timelineStart + currentTime)} (video ${formatDuration(absCurrentTime)})`
                    : undefined
                }
              />
            </div>

            <div className="flex shrink-0 items-center gap-2">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setZoom(Math.max(0.5, zoom - 0.25))}
              >
                <ZoomOut className="h-4 w-4" />
              </Button>
              <span className="w-12 shrink-0 text-center font-mono text-sm sm:w-14">{(zoom * 100).toFixed(0)}%</span>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setZoom(Math.min(4, zoom + 0.25))}
              >
                <ZoomIn className="h-4 w-4" />
              </Button>
            </div>
          </div>

          {/* Timeline: fixed camera labels + scrollable tracks (clean NLE-style) */}
          <div className="flex min-h-0 min-w-0 flex-1 overflow-hidden border-t border-border/50 bg-background">
            <div
              className="flex w-[5.5rem] shrink-0 flex-col border-r border-border/60 bg-muted/25"
              aria-label="Camera lanes"
            >
              <div className="flex h-10 shrink-0 items-end border-b border-border/60 pb-1 pl-2.5">
                <span className="text-[9px] font-semibold uppercase tracking-wider text-muted-foreground">
                  Camera
                </span>
              </div>
              {cameraLanes.map((lane) => {
                const confs = lane.segments
                  .map((s) => Number(s.confidence ?? 0))
                  .filter((v) => Number.isFinite(v) && v > 0);
                const best = confs.length > 0 ? Math.max(...confs) : 0;
                const sel = selectedLaneId === lane.id;
                return (
                  <button
                    key={lane.id}
                    type="button"
                    className={cn(
                      "flex h-10 shrink-0 flex-col items-stretch justify-center border-b border-border/50 px-2.5 text-left transition-colors",
                      "hover:bg-muted/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-inset",
                      sel && "bg-primary/12"
                    )}
                    onClick={() => setSelectedLaneId(sel ? null : lane.id)}
                    title={lane.label}
                  >
                    <span className="truncate font-mono text-[11px] font-semibold leading-tight text-foreground">
                      {lane.cameraId}
                    </span>
                    {best > 0 && (
                      <span className="text-[9px] tabular-nums text-muted-foreground">
                        {Math.round(best * 100)}% match
                      </span>
                    )}
                  </button>
                );
              })}
            </div>

            <ScrollArea className="h-full min-h-0 min-w-0 flex-1">
              <div ref={timelineRef} className="min-w-max pb-3 pl-2 pr-4 pt-2">
                <div
                  className="relative mb-0 h-10 overflow-visible border-b border-border/30 bg-muted/5"
                  style={{ width: timeToPixel(timelineEnd) }}
                >
                  {Array.from({ length: rulerTickCount }).map((_, i) => {
                    const absTime = timelineStart + i * rulerTickInterval;
                    return (
                      <div
                        key={i}
                        className="absolute bottom-0 flex -translate-x-1/2 flex-col items-center"
                        style={{ left: timeToPixel(absTime) }}
                      >
                        <span className="mb-0.5 select-none text-[9px] tabular-nums tracking-tight text-muted-foreground/45">
                          {formatDuration(absTime)}
                        </span>
                        <div className="h-1.5 w-px bg-border/80" />
                      </div>
                    );
                  })}
                  <div
                    className="pointer-events-none absolute inset-y-0 z-30 w-px -translate-x-1/2 bg-foreground/32"
                    style={{ left: rulerPlayheadLeft }}
                    aria-hidden
                  />
                </div>

                <div className="flex flex-col">
                  {cameraLanes.map((lane) => (
                    <TimelineRow
                      key={lane.id}
                      lane={lane}
                      timelineEnd={timelineEnd}
                      isSelected={selectedLaneId === lane.id}
                      onClick={() => setSelectedLaneId(selectedLaneId === lane.id ? null : lane.id)}
                      timeToPixel={timeToPixel}
                      currentTime={currentTime}
                      playheadVideoTime={absCurrentTime}
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

/**
 * Keeps preview tiles from turning into ultra-wide strips: 16:9 frame centered in the cell,
 * typical for CityFlow / traffic footage (letterboxed in the grid slot).
 */
function TimelinePreviewAspectShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-b from-slate-900 via-slate-950 to-black p-1 sm:p-1.5">
      <div
        className={cn(
          "flex min-h-0 min-w-0 items-center justify-center overflow-hidden rounded-md",
          "border border-white/10 bg-black shadow-inner",
          "aspect-video h-full w-auto max-h-full max-w-full"
        )}
      >
        <div className="relative flex h-full min-h-0 w-full min-w-0 items-center justify-center">
          {children}
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
  absCurrentTime,
  trackletPickTime,
  isPlaying,
  primarySeg,
  probeRunId,
  videoId,
  cropRunId,
}: {
  camera: { id: string; name: string; location: string; activeTrack?: any };
  isActive: boolean;
  isPast?: boolean;
  isNext?: boolean;
  absCurrentTime: number;
  /** Quantized time for tracklet frame index — reduces full-frame URL churn vs playhead. */
  trackletPickTime: number;
  isPlaying: boolean;
  primarySeg?: { globalId?: number; cameraId: string; trackId: number; start: number; end: number };
  probeRunId?: string;
  videoId?: string;
  /** Run id whose stage0/ holds frames (gallery precompute), not necessarily the probe run */
  cropRunId?: string;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const seekWallRef = useRef(absCurrentTime);
  const clipStartWallRef = useRef(primarySeg?.start ?? 0);
  const [clipFailed, setClipFailed] = useState(false);
  /** While paused, follow the playhead after a short debounce so we don't hammer seek/decode (black flashes). */
  const [stableScrubTime, setStableScrubTime] = useState(absCurrentTime);
  const [trackSeq, setTrackSeq] = useState<{
    key: string;
    width: number;
    height: number;
    frames: TrackletSequenceFrame[];
  } | null>(null);

  // Derive clip URL directly from segment metadata — no async state needed.
  // Pattern: outputs/{probeRunId}/matched/global_{gid}_cam_{cameraId}_track_{tid}.mp4
  const clipUrl = (() => {
    if (!probeRunId || !primarySeg) return null;
    const { globalId, cameraId, trackId } = primarySeg;
    if (globalId == null || trackId == null) return null;
    const safeCam = String(cameraId).replace(/[/\\]/g, "_");
    const filename = `global_${globalId}_cam_${safeCam}_track_${trackId}.mp4`;
    return `${API_BASE}/runs/${probeRunId}/matched_clips/${filename}`;
  })();

  const trackSeqKey =
    primarySeg?.cameraId != null && primarySeg.trackId != null
      ? `${String(primarySeg.cameraId)}|${Number(primarySeg.trackId)}`
      : "";

  const segmentIdentityKey = `${trackSeqKey}|${primarySeg?.globalId ?? ""}|${clipUrl ?? ""}`;
  const prevSegmentIdentityRef = useRef<string>("");

  useEffect(() => {
    if (!cropRunId || !primarySeg?.cameraId || primarySeg.trackId == null) {
      setTrackSeq(null);
      return;
    }
    const key = `${String(primarySeg.cameraId)}|${Number(primarySeg.trackId)}`;
    let cancelled = false;
    setTrackSeq((prev) => (prev?.key === key ? prev : null));
    void (async () => {
      try {
        const data = await getTrackletSequence(
          cropRunId,
          String(primarySeg.cameraId),
          Number(primarySeg.trackId),
          120
        );
        if (cancelled) return;
        if (data?.frames?.length) {
          setTrackSeq({
            key,
            width: data.width,
            height: data.height,
            frames: data.frames,
          });
        } else {
          setTrackSeq(null);
        }
      } catch {
        if (!cancelled) setTrackSeq(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [cropRunId, primarySeg?.cameraId, primarySeg?.trackId]);

  useEffect(() => {
    if (segmentIdentityKey !== prevSegmentIdentityRef.current) {
      prevSegmentIdentityRef.current = segmentIdentityKey;
      setStableScrubTime(absCurrentTime);
    }
  }, [segmentIdentityKey, absCurrentTime]);

  useEffect(() => {
    if (isPlaying) {
      setStableScrubTime(absCurrentTime);
      return;
    }
    const id = window.setTimeout(() => setStableScrubTime(absCurrentTime), 90);
    return () => window.clearTimeout(id);
  }, [absCurrentTime, isPlaying]);

  seekWallRef.current = isPlaying ? absCurrentTime : stableScrubTime;

  useEffect(() => {
    setClipFailed(false);
  }, [clipUrl]);

  const clipStartSec = primarySeg?.start ?? 0;
  clipStartWallRef.current = clipStartSec;

  // Seek once metadata is ready for a new clip only (deps: clip only — wall time read from refs).
  useEffect(() => {
    const v = videoRef.current;
    if (!v || !clipUrl) return;
    const onCanPlay = () => {
      const vv = videoRef.current;
      if (!vv) return;
      vv.currentTime = Math.max(0, seekWallRef.current - clipStartWallRef.current);
    };
    v.addEventListener("canplay", onCanPlay, { once: true });
    return () => v.removeEventListener("canplay", onCanPlay);
  }, [clipUrl]);

  // Playing: only fix large drift. Paused: seek matches debounced scrub time (avoids rapid seeks → black frames).
  useEffect(() => {
    const v = videoRef.current;
    if (!v || !clipUrl) return;
    if (isPlaying) {
      const seekTo = Math.max(0, absCurrentTime - clipStartSec);
      if (Math.abs(v.currentTime - seekTo) > 1.25) v.currentTime = seekTo;
      return;
    }
    const seekTo = Math.max(0, stableScrubTime - clipStartSec);
    if (Math.abs(v.currentTime - seekTo) < 0.04) return;
    v.currentTime = seekTo;
  }, [isPlaying, absCurrentTime, stableScrubTime, clipUrl, clipStartSec]);

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
  // Build a crop URL from the track's representative frame (gallery run stage0, or upload video)
  const cropUrl = (() => {
    if (!camera.activeTrack) return null;
    const t = camera.activeTrack;
    const bbox = t.representativeBbox;
    const frameId = t.representativeFrame;
    if (frameId == null) return null;
    const bboxParams = (bbox && bbox.length === 4)
      ? `x1=${bbox[0]}&y1=${bbox[1]}&x2=${bbox[2]}&y2=${bbox[3]}`
      : "x1=0&y1=0&x2=9999&y2=9999";
    if (cropRunId && shouldUseRunCropsForCamera(cropRunId, t.cameraId)) {
      return `${API_BASE}/crops/run/${cropRunId}?cameraId=${encodeURIComponent(t.cameraId)}&frameId=${frameId}&${bboxParams}`;
    }
    if (videoId) {
      return `${API_BASE}/crops/${videoId}?frameId=${frameId}&${bboxParams}`;
    }
    return null;
  })();

  const trackletFramePick = (() => {
    if (!trackSeq?.frames?.length || !primarySeg || !trackSeqKey || trackSeq.key !== trackSeqKey) {
      return null;
    }
    const segStart = primarySeg.start;
    const segEnd = Math.max(primarySeg.end, segStart + 1e-3);
    const tPick = isPlaying ? trackletPickTime : stableScrubTime;
    const u = Math.min(1, Math.max(0, (tPick - segStart) / (segEnd - segStart)));
    const frames = trackSeq.frames;
    let best = frames[0];
    let bestD = 1;
    for (const f of frames) {
      const d = Math.abs(f.timeRel - u);
      if (d < bestD) {
        bestD = d;
        best = f;
      }
    }
    return { frame: best };
  })();

  const trackletFullSrc =
    trackletFramePick && cropRunId
      ? getRunFullFrameUrl(
          cropRunId,
          String(primarySeg!.cameraId),
          trackletFramePick.frame.frameId
        )
      : null;

  const showTrackletFrames = Boolean(trackletFullSrc && trackletFramePick);
  const showVideoClip = Boolean(clipUrl && !clipFailed && !showTrackletFrames);
  const showCropOnly = Boolean(cropUrl && !showTrackletFrames && !showVideoClip);

  const ringClass = showTrackletFrames
    ? isActive
      ? "ring-2 ring-green-500"
      : isPast
      ? "ring-1 ring-orange-500/60 opacity-70"
      : isNext
      ? "ring-1 ring-blue-400/60 opacity-70"
      : "opacity-50"
    : clipUrl && !clipFailed
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
    <div
      className={cn(
        "relative h-full min-h-0 w-full min-w-0 overflow-hidden rounded",
        ringClass
      )}
    >
      {/* Camera feed — 16:9 shell + object-contain so tiles never look panoramic / stretched. */}
      <TimelinePreviewAspectShell>
        {showTrackletFrames ? (
          <TrackletFrameView
            src={trackletFullSrc!}
            bbox={trackletFramePick!.frame.bbox}
          />
        ) : showVideoClip ? (
          <video
            key={clipUrl}
            ref={videoRef}
            src={clipUrl!}
            poster={cropUrl ?? undefined}
            className="max-h-full max-w-full object-contain"
            muted
            playsInline
            preload="auto"
            onError={() => setClipFailed(true)}
          />
        ) : showCropOnly ? (
          <img
            src={cropUrl!}
            alt={camera.id}
            className="max-h-full max-w-full object-contain"
            draggable={false}
          />
        ) : (
          <div className="relative h-full min-h-0 w-full min-w-0">
            <svg className="pointer-events-none absolute inset-0 h-full w-full opacity-20" preserveAspectRatio="none">
              <line x1="50%" y1="30%" x2="20%" y2="100%" stroke="white" strokeWidth="1" />
              <line x1="50%" y1="30%" x2="80%" y2="100%" stroke="white" strokeWidth="1" />
            </svg>
            {isActive && camera.activeTrack && (
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">
                <div
                  className="flex h-6 w-10 items-center justify-center rounded border-2"
                  style={{ borderColor: camera.activeTrack.color || "#22c55e", backgroundColor: `${camera.activeTrack.color || "#22c55e"}33` }}
                >
                  <Car className="h-4 w-4 text-white" />
                </div>
              </div>
            )}
          </div>
        )}
      </TimelinePreviewAspectShell>

      {/* Camera info overlay */}
      <div className="absolute top-0 left-0 right-0 p-1 bg-black/60">
        <div className="flex items-center gap-1">
          <div
            className={cn(
              "h-1.5 w-1.5 rounded-full",
              isActive
                ? isPlaying
                  ? "bg-green-500"
                  : "bg-green-500 animate-pulse"
                : isPast
                  ? "bg-orange-400"
                  : isNext
                    ? "bg-blue-400"
                    : "bg-gray-500"
            )}
          />
          <span className="text-white text-[10px] font-mono">{camera.id}</span>
          {statusLabel && (
            <span className={cn("text-[9px] font-bold ml-auto", statusColor)}>{statusLabel}</span>
          )}
        </div>
      </div>

      {/* Timestamp (source video time; ruler uses combined tracklet duration) */}
      <div className="absolute bottom-0 left-0 right-0 p-1 bg-black/60">
        <span className="text-white/70 text-[9px] font-mono">
          {formatDuration(absCurrentTime)}
        </span>
      </div>
    </div>
  );
}
interface TrackletItemProps {
  track: TimelineTrack;
  isSelected: boolean;
  /** Segments cover current video playhead (wall-clock). */
  isActiveAtPlayhead?: boolean;
  onClick: () => void;
  onConfirm: () => void;
  onRemove: () => void;
}

function TrackletItem({
  track,
  isSelected,
  isActiveAtPlayhead = false,
  onClick,
  onConfirm,
  onRemove,
}: TrackletItemProps) {
  const nCams = track.segments ? new Set(track.segments.map((s) => s.cameraId)).size : 1;
  const primaryColor = track.segments?.[0]?.color ?? getCameraColor(track.cameraId);

  return (
    <div
      data-track-id={track.id}
      className={cn(
        "p-2 rounded-lg border cursor-pointer transition-all",
        isSelected && "border-primary bg-primary/5",
        track.confirmed && !isSelected && "border-green-500/50 bg-green-500/5",
        isActiveAtPlayhead && !isSelected && "border-l-2 border-l-emerald-500 bg-emerald-500/5"
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

function AlternativeTrackletItem({
  alternative,
  videoUrl,
  onUse,
}: {
  alternative: MatchedAlternative;
  videoUrl: string;
  onUse: () => void;
}) {
  return (
    <div className="rounded-md border border-border/60 bg-muted/20 p-2">
      <div className="mb-1 flex items-center justify-between gap-2">
        <span className="text-[10px] font-semibold text-blue-400">ALT #{alternative.rank}</span>
        <span className="text-[10px] tabular-nums text-muted-foreground">
          score {(alternative.score * 100).toFixed(1)}%
        </span>
      </div>

      {videoUrl ? (
        <video
          src={videoUrl}
          className="mb-2 h-20 w-full rounded object-cover"
          controls
          muted
          playsInline
          preload="metadata"
        />
      ) : null}

      <div className="space-y-0.5 text-[10px] text-muted-foreground">
        <p className="font-mono text-foreground/90">
          {alternative.cameraId} &middot; track {alternative.trackId}
        </p>
        <p>
          global {alternative.globalId ?? "?"} &middot; {Math.max(1, alternative.numCameras)} cams
        </p>
      </div>

      <Button
        variant="outline"
        size="sm"
        className="mt-2 h-6 w-full text-[10px]"
        onClick={onUse}
      >
        Use In Timeline
      </Button>
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
      sumStart: number;
      sumEnd: number;
    }>;
  };
  timelineEnd: number;
  isSelected: boolean;
  onClick: () => void;
  timeToPixel: (time: number) => number;
  /** Ruler offset (sum of tracklet durations up to the playhead). */
  currentTime: number;
  /** Mapped source video time for highlighting segments vs wall-clock. */
  playheadVideoTime: number;
}

/**
 * Single scrollable track row (camera label lives in the fixed left column).
 * Capsule segments only — no stretched thumbnails.
 */
function TimelineRow({
  lane,
  timelineEnd,
  isSelected,
  onClick,
  timeToPixel,
  currentTime,
  playheadVideoTime,
}: TimelineRowProps) {
  const isCurrentlyActive = lane.segments.some(
    (seg) => playheadVideoTime >= seg.start && playheadVideoTime <= seg.end
  );
  const segments = lane.segments;

  return (
    <button
      type="button"
      className={cn(
        "relative h-10 cursor-pointer border-b border-border/40 text-left transition-colors",
        "hover:bg-muted/30 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-inset",
        isSelected && "bg-primary/8",
        isCurrentlyActive && "bg-muted/20"
      )}
      style={{ width: timeToPixel(timelineEnd) }}
      onClick={onClick}
      title={lane.label}
    >
      <div className="pointer-events-none absolute inset-x-0 top-1/2 h-6 -translate-y-1/2 bg-muted/40" />
      {segments.map((seg, i) => {
        const segLeft = timeToPixel(seg.sumStart);
        const segWidth = Math.max(timeToPixel(seg.sumEnd) - timeToPixel(seg.sumStart), 3);
        const isSegActive =
          playheadVideoTime >= seg.start && playheadVideoTime <= seg.end;

        return (
          <div
            key={`${seg.cameraId}-${seg.trackId}-${seg.trajectoryId}-${i}`}
            className={cn(
              "absolute top-1/2 h-2.5 -translate-y-1/2 rounded-full border transition-shadow",
              "border-black/20 shadow-sm",
              isSegActive && "z-[5] h-3 shadow-md ring-2 ring-white/50",
              seg.confirmed && "ring-1 ring-green-500/80 ring-offset-1 ring-offset-background"
            )}
            style={{
              left: segLeft,
              width: segWidth,
              backgroundColor: seg.color,
              opacity: isSegActive ? 1 : 0.72,
            }}
            title={`G-${String(seg.globalId ?? 0).padStart(4, "0")} · ${formatDuration(seg.start)} → ${formatDuration(seg.end)}`}
          />
        );
      })}

      <div
        className="pointer-events-none absolute inset-y-0 z-20 w-px -translate-x-1/2 bg-foreground/22"
        style={{ left: timeToPixel(currentTime) }}
        aria-hidden
      />
    </button>
  );
}
