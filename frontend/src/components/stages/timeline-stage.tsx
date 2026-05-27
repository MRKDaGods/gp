"use client";

import {
  useState,
  useCallback,
  useRef,
  useEffect,
  useMemo,
} from "react";
import {
  ArrowRight,
  RefreshCw,
} from "lucide-react";
import { cn, formatDuration, formatNetworkFailure, getCameraColor } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { DisclosurePanel, ErrorBanner, PlaybackControls, StageProgressCard, toStageStatus } from "@/components/pipeline";
import { ExecutionTargetToggle } from "@/components/pipeline/run/ExecutionTargetToggle";
import {
  useTimelineStore,
  usePipelineStore,
  useSessionStore,
  useVideoStore,
  useDetectionStore,
  useStageExecutionStore,
} from "@/store";
import { useKaggleCredentialsStore } from "@/lib/kaggle-credentials-store";
import {
  getTracklets,
  getTrajectories,
  runStage,
  getPipelineStatus,
  cancelPipeline,
  queryTimeline,
  getMatchedSummary,
  getMatchedAlternatives,
  type MatchedAlternative,
} from "@/lib/api";
import type { TimelineTrack, TrajectorySegment } from "@/types";
import { AlternativesSheet } from "./timeline/AlternativesSheet";
import { NLETimeline } from "./timeline/NLETimeline";
import { TimelineAdvancedControls } from "./timeline/TimelineAdvancedControls";
import { TimelineDebugPanel } from "./timeline/TimelineDebugPanel";
import { TimelineVideoGrid } from "./timeline/TimelineVideoGrid";
import { TrackletRail } from "./timeline/TrackletRail";
import type {
  TimelineCameraLane as CameraLane,
  TimelineCameraLaneSegment as CameraLaneSegment,
  TimelineCameraLaneSegmentWithSum as CameraLaneSegmentWithSum,
} from "./timeline/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8004/api";

/** Playhead & ruler updates/sec while playing — lower = less React work (grid + memos). */
const TIMELINE_PLAYHEAD_FPS = 12;
/** Tracklet full-frame picks/sec while playing (lower than playhead to limit image decode load). */
const TRACKLET_PICK_FPS = 15;
const TRACKLET_PICK_BUCKET_SEC = 1 / TRACKLET_PICK_FPS;
const TIMELINE_SHOW_ALTERNATIVES_EVENT = "mtmc:timeline:show-alternatives";
const TIMELINE_RERUN_ASSOCIATION_EVENT = "mtmc:timeline:rerun-association";

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
    applyTracksReplaceKeepingMeta,
    zoom,
    setZoom,
    selectedTrackId,
    selectTrack,
    confirmTrack,
    unconfirmTrack,
    removeTrack,
  } = useTimelineStore();
  const {
    runId,
    galleryRunId,
    setRunId,
    updateStageProgress,
    stages,
    downstreamInvalidateGeneration,
  } = usePipelineStore();
  const { currentStage } = useSessionStore();
  const { currentVideo } = useVideoStore();
  const { selectedTrackIds: selectedTrackIdSet } = useDetectionStore();
  const getStageExecutionTarget = useStageExecutionStore((state) => state.getStageExecutionTarget);

  const stage4KaggleRequest = useCallback(() => {
    if (getStageExecutionTarget(4) !== "kaggle") return {};
    const credentials = useKaggleCredentialsStore.getState().credentials;
    return { kaggle: { target: "kaggle" as const, username: credentials?.username, key: credentials?.key } };
  }, [getStageExecutionTarget]);

  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [selectedLaneId, setSelectedLaneId] = useState<string | null>(null);
  const [splitCount, setSplitCount] = useState<number | null>(null);
  const [triggerReload, setTriggerReload] = useState(0);
  const [, setMatchedFallbackActive] = useState(false);
  const [tracksLoading, setTracksLoading] = useState(false);
  const [topAlternatives, setTopAlternatives] = useState<MatchedAlternative[]>([]);
  const [alternativesLoading, setAlternativesLoading] = useState(false);
  const [alternativesError, setAlternativesError] = useState<string | null>(null);
  const [alternativesCameraCount, setAlternativesCameraCount] = useState(0);
  const [alternativesOpen, setAlternativesOpen] = useState(false);
  const [alternativeHistoryByTrackId, setAlternativeHistoryByTrackId] = useState<Record<string, MatchedAlternative[]>>({});
  const [currentAlternativeByTrackId, setCurrentAlternativeByTrackId] = useState<Record<string, MatchedAlternative>>({});
  const [playingTrackletsOnly, setPlayingTrackletsOnly] = useState(false);
  const timelineRef = useRef<HTMLDivElement>(null);
  const trajectoryListRef = useRef<HTMLDivElement>(null);
  /** Monotonic id so stale async loads cannot overwrite newer results. */
  const loadTracksSeqRef = useRef(0);
  /** Skip redundant timeline reload when revisiting Stage 4 with same inputs + tracks already populated. */
  const timelineHandledStampRef = useRef<string | null>(null);
  /** Last timeline inputs key (video/runs/selection); used to detect association-context changes. */
  const prevTimelineLoadKeyRef = useRef<string>("");
  /** Dedupe probe-side exports across repeated timeline queries within one loader run / navigation. */
  const lastExportedQueryKeyRef = useRef<string | null>(null);
  /** One automatic Stage 4 rerun per video when query returns empty (avoids forcing users to click "Rerun association"). */
  const autoAssociationRefreshForVideoRef = useRef<string | null>(null);

  const timelineLoadKey = useMemo(() => {
    const ids = Array.from(selectedTrackIdSet).sort((a, b) => a - b);
    return [currentVideo?.id ?? "", runId ?? "", galleryRunId ?? "", ...ids.map(String)].join("\u0001");
  }, [currentVideo?.id, runId, galleryRunId, selectedTrackIdSet]);

  useEffect(() => {
    autoAssociationRefreshForVideoRef.current = null;
  }, [currentVideo?.id]);

  /** Let "Rerun association" (`triggerReload`) attempt auto Stage-4 refresh again on the next load. */
  useEffect(() => {
    autoAssociationRefreshForVideoRef.current = null;
  }, [triggerReload]);

  /** Backend-resolved probe run (session `runId` can be stale vs disk outputs). */
  const [resolvedProbeRunId, setResolvedProbeRunId] = useState<string | null>(null);

  useEffect(() => {
    setResolvedProbeRunId(null);
  }, [currentVideo?.id]);

  const probeRunIdForMedia = useMemo(
    () => galleryRunId ?? resolvedProbeRunId ?? runId ?? undefined,
    [galleryRunId, resolvedProbeRunId, runId]
  );

  /** Update local resolved probe from timeline diagnostics. Do not call setRunId here — that would
   *  change the loadTracks effect deps mid-await and cancel the in-flight loader (blank timeline). */
  const applyTimelineResolvedProbe = useCallback(
    (diagnostics: Record<string, unknown> | null | undefined): string | null => {
      const d = diagnostics ?? {};
      const raw = d.resolvedProbeRunId ?? d.selectedTrackletsSourceRun;
      if (raw == null || String(raw).trim() === "") return null;
      const id = String(raw);
      setResolvedProbeRunId(id);
      return id;
    },
    []
  );

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

  // Stage 4: run real association, then load tracks.
  // When the tab is inactive (user is on another stage), do not fetch — avoids hammering the API during Selection edits.
  useEffect(() => {
    let cancelled = false;

    if (currentStage !== 4) {
      return;
    }

    if (!currentVideo) {
      setTracksLoading(false);
      return;
    }

    const loadKeyChanged = prevTimelineLoadKeyRef.current !== timelineLoadKey;
    if (loadKeyChanged) {
      prevTimelineLoadKeyRef.current = timelineLoadKey;
      lastExportedQueryKeyRef.current = null;
      timelineHandledStampRef.current = null;
      setTracks([]);
      setMatchedFallbackActive(false);
    }

    const stamp = `${timelineLoadKey}\0${triggerReload}\0${downstreamInvalidateGeneration}`;

    if (
      !loadKeyChanged
      && timelineHandledStampRef.current === stamp
    ) {
      setTracksLoading(false);
      return;
    }

    setTracksLoading(true);
    if (selectedTrackIdSet.size > 0) {
      updateStageProgress(4, {
        status: "running",
        progress: 5,
        message: "Running cross-camera association…",
      });
    }

    const loadTracks = async () => {
      const seq = ++loadTracksSeqRef.current;
      /** Deferred until loader finishes — avoids effect re-entry + cancelled mid-flight when runId syncs from diagnostics */
      let pendingRunIdFromDiagnostics: string | null = null;
      let loadErrored = false;
      try {
        let attemptedAssociation = false;
        let finalTracksSet = false;
        const selectedTrackIdsArr = Array.from(selectedTrackIdSet).map((v) => String(v));
        const sortedSel = selectedTrackIdsArr.slice().sort().join(",");
        const exportDedupeKey = (probeKey: string) =>
          `${probeKey}|${galleryRunId ?? ""}|${currentVideo.id}|${sortedSel}`;
        const callQueryTimeline = async (probeHint: string | undefined) => {
          const dk = exportDedupeKey(probeHint ?? runId ?? "");
          const skipExports = lastExportedQueryKeyRef.current === dk;
          const res = await queryTimeline(probeHint ?? runId ?? undefined, currentVideo.id, selectedTrackIdsArr, {
            galleryRunId: galleryRunId ?? undefined,
            skipExports,
          });
          if (!cancelled && seq === loadTracksSeqRef.current && !skipExports) {
            lastExportedQueryKeyRef.current = dk;
          }
          return res;
        };

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
          const q1 = await callQueryTimeline(runId ?? undefined);
          if (cancelled || seq !== loadTracksSeqRef.current) return;

          const q1Data: any = q1.data ?? {};
          const d1 = q1Data.diagnostics;
          const resolvedFromDiag1 = applyTimelineResolvedProbe(d1);
          if (resolvedFromDiag1) pendingRunIdFromDiagnostics = resolvedFromDiag1;
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
              finalTracksSet = true;
              if (seq !== loadTracksSeqRef.current) return;
              applyTracksReplaceKeepingMeta(rows);
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
          const summaryFetchRunId = resolvedFromDiag1 ?? runId;
          if (summaryFetchRunId) {
            try {
              const summaryResp = await getMatchedSummary(summaryFetchRunId);
              if (cancelled) return;
              const fallbackRows = buildTracksFromMatchedSummary(summaryResp);
              if (fallbackRows.length > 0) {
                setMatchedFallbackActive(true);
                finalTracksSet = true;
                applyTracksReplaceKeepingMeta(fallbackRows);
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
            const stageResp = await runStage(4, { runId: effectiveGalleryRunId, videoId: currentVideo.id, ...stage4KaggleRequest() });
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
              const q2 = await callQueryTimeline(
                typeof stage4RunId === "string" ? stage4RunId : undefined
              );
              if (cancelled || seq !== loadTracksSeqRef.current) return;
              const q2Data: any = q2.data ?? {};
              const pr2 = applyTimelineResolvedProbe(q2Data.diagnostics);
              if (pr2) pendingRunIdFromDiagnostics = pr2;
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
                finalTracksSet = true;
                if (seq !== loadTracksSeqRef.current) return;
                applyTracksReplaceKeepingMeta(rows);
                updateStageProgress(4, { status: "completed", progress: 100, message: String(q2Data.message ?? "Association complete (query-matched)") });
                console.info("decision", "matched trajectories rendered after stage4", { rows: rows.length });
                console.groupEnd();
                return;
              }

              if (q2Selected.length > 0) {
                const fallbackTracks = buildTracksFromSummary(q2Selected);
                finalTracksSet = true;
                if (seq !== loadTracksSeqRef.current) return;
                applyTracksReplaceKeepingMeta(fallbackTracks);
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

          // Stage-4 artifacts exist (`stage4Available`) but matcher returned nothing — often stale vs current probe/embeddings.
          // Run association once automatically (same as pressing "Rerun association") so the first Timeline visit succeeds.
          if (
            q1Data.stage4Available &&
            q1Traj.length === 0 &&
            q1Selected.length === 0 &&
            autoAssociationRefreshForVideoRef.current !== currentVideo.id
          ) {
            autoAssociationRefreshForVideoRef.current = currentVideo.id;
            updateStageProgress(4, {
              status: "running",
              progress: 5,
              message: "Refreshing cross-camera association…",
            });
            const stageResp = await runStage(4, { runId: effectiveGalleryRunId, videoId: currentVideo.id, ...stage4KaggleRequest() });
            if (cancelled || seq !== loadTracksSeqRef.current) return;
            const refreshStage4RunId = (stageResp.data as any)?.runId ?? runId;

            let refreshDone = false;
            while (!refreshDone && !cancelled) {
              await new Promise((r) => setTimeout(r, 1500));
              if (cancelled || seq !== loadTracksSeqRef.current) return;
              const refreshStatusResp = await getPipelineStatus(refreshStage4RunId);
              if (cancelled || seq !== loadTracksSeqRef.current) return;
              const refreshStatusData: any = refreshStatusResp.data;
              const refreshStatus = refreshStatusData?.status;
              const refreshProgress = Number(refreshStatusData?.progress ?? 0);
              const refreshMessage = String(refreshStatusData?.message ?? "Running...");
              updateStageProgress(4, { progress: refreshProgress, message: refreshMessage });
              if (refreshStatus === "completed" || refreshStatus === "error") refreshDone = true;
              if (refreshStatus === "error") {
                const errMsg = String(refreshStatusData?.error ?? "Stage 4 association failed");
                updateStageProgress(4, { status: "error", message: errMsg });
                break;
              }
            }

            if (!cancelled && seq === loadTracksSeqRef.current) {
              const qRefresh = await callQueryTimeline(
                typeof refreshStage4RunId === "string" ? refreshStage4RunId : undefined
              );
              if (cancelled || seq !== loadTracksSeqRef.current) return;
              const qRefreshData: any = qRefresh.data ?? {};
              const prR = applyTimelineResolvedProbe(qRefreshData.diagnostics);
              if (prR) pendingRunIdFromDiagnostics = prR;
              const qRefreshTraj = Array.isArray(qRefreshData.trajectories) ? qRefreshData.trajectories : [];
              const qRefreshSelected = Array.isArray(qRefreshData.selectedTracklets)
                ? qRefreshData.selectedTracklets
                : [];

              if (qRefreshTraj.length > 0) {
                const refreshRows = buildTracksFromTrajectories(qRefreshTraj);
                if (refreshRows.length > 0) {
                  setMatchedFallbackActive(false);
                  finalTracksSet = true;
                  if (seq !== loadTracksSeqRef.current) return;
                  applyTracksReplaceKeepingMeta(refreshRows);
                  updateStageProgress(4, {
                    status: "completed",
                    progress: 100,
                    message: String(qRefreshData.message ?? "Association loaded (query-matched)"),
                  });
                  console.info("decision", "matched trajectories after auto association refresh", {
                    rows: refreshRows.length,
                  });
                  console.groupEnd();
                  return;
                }
              }
              if (qRefreshSelected.length > 0) {
                const refreshFallback = buildTracksFromSummary(qRefreshSelected);
                finalTracksSet = true;
                if (seq !== loadTracksSeqRef.current) return;
                applyTracksReplaceKeepingMeta(refreshFallback);
                updateStageProgress(4, {
                  status: "completed",
                  progress: 100,
                  message: String(
                    qRefreshData.message ?? "No cross-camera match found; showing selected single-camera tracklets"
                  ),
                });
                console.info("decision", "selected single-camera fallback after auto association refresh", {
                  rows: refreshFallback.length,
                });
                console.groupEnd();
                return;
              }
            }
          }

          // stage4 exists but no match; show selected single-camera tracklets if available.
          if (q1Selected.length > 0) {
            const fallbackTracks = buildTracksFromSummary(q1Selected);
            finalTracksSet = true;
            if (seq !== loadTracksSeqRef.current) return;
            applyTracksReplaceKeepingMeta(fallbackTracks);
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
          finalTracksSet = true;
          if (seq !== loadTracksSeqRef.current) return;
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
            finalTracksSet = true;
            applyTracksReplaceKeepingMeta(trajectoryRows);
            updateStageProgress(4, { status: "completed", progress: 100, message: "Association loaded (query-matched)" });
            console.info("decision", "non-query stage4 trajectories rendered", { rows: trajectoryRows.length });
            console.groupEnd();
            return;
          }

          // No stage4 artifacts yet — run stage 4 now
          updateStageProgress(4, { status: "running", progress: 5, message: "Running cross-camera association..." });
          const stageResp = await runStage(4, { runId, videoId: currentVideo.id, ...stage4KaggleRequest() });
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
              finalTracksSet = true;
              applyTracksReplaceKeepingMeta(rows2);
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
            finalTracksSet = true;
            applyTracksReplaceKeepingMeta(fallbackTracks);
            updateStageProgress(4, {
              status: "completed",
              progress: 100,
              message: "No cross-camera match found; showing selected single-camera tracklets",
            });
            console.info("decision", "selected stage1 fallback rendered", { rows: fallbackTracks.length });
          } else {
            finalTracksSet = true;
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
          finalTracksSet = true;
          applyTracksReplaceKeepingMeta(realTracks);
          updateStageProgress(4, { status: "completed", progress: 100, message: "Showing stage 1 tracklets" });
          console.info("decision", "no-query stage1 summary rendered", { rows: realTracks.length });
        }
        console.groupEnd();
      } catch (err) {
        loadErrored = true;
        if (!cancelled) {
          updateStageProgress(4, {
            status: "error",
            progress: 0,
            message: formatNetworkFailure(err),
          });
          console.error("[Stage4][Timeline] loadTracks error", err);
        }
        console.groupEnd();
      } finally {
        if (
          !cancelled
          && seq === loadTracksSeqRef.current
          && pendingRunIdFromDiagnostics
          && !galleryRunId
        ) {
          setRunId(pendingRunIdFromDiagnostics);
        }
        // Only the latest load invocation may clear loading — avoids Strict Mode / runId churn
        // flipping loading off mid-fetch or leaving it stuck when an older async completes later.
        if (seq === loadTracksSeqRef.current) {
          setTracksLoading(false);
          if (!loadErrored) {
            timelineHandledStampRef.current = stamp;
          }
        }
      }
    };

    void loadTracks();

    return () => {
      cancelled = true;
    };
  }, [
    currentStage,
    currentVideo,
    timelineLoadKey,
    runId,
    galleryRunId,
    downstreamInvalidateGeneration,
    selectedTrackIdSet,
    setTracks,
    applyTracksReplaceKeepingMeta,
    triggerReload,
    updateStageProgress,
    applyTimelineResolvedProbe,
    setRunId,
  ]);

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
          probeRunIdForMedia && track.globalId != null
            ? `${API_BASE}/runs/${encodeURIComponent(probeRunIdForMedia)}/matched_clips/${encodeURIComponent(
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
    [probeRunIdForMedia]
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
    const altRunId = probeRunIdForMedia ?? runId ?? null;
    if (!altRunId || !selectedTrack) {
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
        const payload = await getMatchedAlternatives(altRunId, {
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
  }, [
    probeRunIdForMedia,
    runId,
    selectedTrack,
    selectedTrackId,
    alternativeHistoryByTrackId,
    mergeWithHistoryAlternatives,
  ]);

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

      applyTracksReplaceKeepingMeta(updated);
    },
    [buildAlternativeFromTrack, currentAlternativeByTrackId, selectedTrack, selectedTrackId, applyTracksReplaceKeepingMeta, topAlternatives, tracks]
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

  const handleRerunAssociation = async () => {
    if (!currentVideo) return;
    const associationRunId = galleryRunId ?? runId;
    if (!associationRunId) return;
    timelineHandledStampRef.current = null;
    lastExportedQueryKeyRef.current = null;
    setTracks([]);
    setTracksLoading(true);
    updateStageProgress(4, { status: "running", progress: 5, message: "Manually re-running association..." });
    const POLL_MS = 1500;
    const MAX_POLLS = 800;

    try {
      const stageResp = await runStage(4, { runId: associationRunId, videoId: currentVideo.id, ...stage4KaggleRequest() });
      const pollRunId = String((stageResp.data as any)?.runId ?? associationRunId);

      let done = false;
      for (let i = 0; !done && i < MAX_POLLS; i += 1) {
        await new Promise((r) => setTimeout(r, POLL_MS));
        let statusData: Record<string, unknown> | undefined;
        try {
          const statusResp = await getPipelineStatus(pollRunId);
          statusData = (statusResp.data ?? undefined) as Record<string, unknown> | undefined;
        } catch (err) {
          updateStageProgress(4, {
            status: "error",
            progress: 0,
            message: formatNetworkFailure(err),
          });
          done = true;
          break;
        }

        const status = String(statusData?.status ?? "running");
        const progress = Number(statusData?.progress ?? 0);
        const message = String(statusData?.message ?? "Running cross-camera association...");
        updateStageProgress(4, { status: "running", progress, message });

        if (status === "completed" || status === "error") {
          if (status === "error") {
            const errMsg = String(statusData?.error ?? "Stage 4 association failed");
            updateStageProgress(4, { status: "error", message: errMsg });
          } else {
            updateStageProgress(4, { status: "completed", progress: 100, message });
          }
          done = true;
        }
      }

      if (!done) {
        updateStageProgress(4, {
          status: "error",
          progress: 0,
          message: "Association timed out while waiting for the server.",
        });
      }

      setTriggerReload((n) => n + 1);
    } catch (e) {
      updateStageProgress(4, {
        status: "error",
        progress: 0,
        message: formatNetworkFailure(e),
      });
    } finally {
      setTracksLoading(false);
    }
  };

  useEffect(() => {
    const openAlternatives = () => setAlternativesOpen(true);
    const rerunAssociation = () => {
      void handleRerunAssociation();
    };

    window.addEventListener(TIMELINE_SHOW_ALTERNATIVES_EVENT, openAlternatives);
    window.addEventListener(TIMELINE_RERUN_ASSOCIATION_EVENT, rerunAssociation);
    return () => {
      window.removeEventListener(TIMELINE_SHOW_ALTERNATIVES_EVENT, openAlternatives);
      window.removeEventListener(TIMELINE_RERUN_ASSOCIATION_EVENT, rerunAssociation);
    };
  }, [handleRerunAssociation]);

  const confirmedCount = useMemo(
    () => tracks.filter((t) => t.confirmed).length,
    [tracks]
  );
  const timelineDataSource = tracks.some((t) => t.id.startsWith("real-") || t.id.startsWith("traj-")) ? "real" : "demo";

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

  /** Bucketed video time for trajectory/lane hit tests (~10 Hz) so list + grid do not recompute every playhead tick. */
  const coarsePlayheadVideoTime = useMemo(() => {
    const step = 0.1;
    return Math.round(absCurrentTime / step) * step;
  }, [absCurrentTime]);

  const trackletPickTime = useMemo(() => {
    const step = TRACKLET_PICK_BUCKET_SEC;
    return Math.round(absCurrentTime / step) * step;
  }, [absCurrentTime]);

  const activeAtPlayheadIds = useMemo(() => {
    const ids = new Set<string>();
    for (const t of tracks) {
      if (trackIsActiveAtVideoTime(t, coarsePlayheadVideoTime)) ids.add(t.id);
    }
    return ids;
  }, [tracks, coarsePlayheadVideoTime]);

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
      if (segs.some((s) => coarsePlayheadVideoTime >= s.start && coarsePlayheadVideoTime <= s.end)) {
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
    coarsePlayheadVideoTime,
  ]);

  const effectiveSplitCount = playingTrackletsOnly
    ? Math.min(camerasForPreview.length || 1, 8)
    : splitCount ?? Math.min(allCamerasForPreview.length || 1, 8);

  const visibleCameras = camerasForPreview.slice(0, effectiveSplitCount);

  const activeCamerasForGrid = useMemo(() => {
    const t = coarsePlayheadVideoTime;
    return visibleCameras.map((cam) => {
      const lane = cameraLanes.find((l) => l.cameraId === cam.id);
      const allSegs = lane?.segments ?? [];
      const laneSegments = playbackFilterActive
        ? allSegs.filter((s) => activeAtPlayheadIds.has(s.trajectoryId))
        : allSegs;
      const activeSegment = laneSegments.find((s) => t >= s.start && t <= s.end);
      const isPast = !activeSegment && laneSegments.length > 0 && laneSegments.every((s) => t > s.end);
      const isNext = !activeSegment && laneSegments.length > 0 && laneSegments.every((s) => t < s.start);
      const primarySeg = activeSegment
        ?? (isNext ? laneSegments[0] : undefined)
        ?? (isPast ? laneSegments[laneSegments.length - 1] : undefined);
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
  }, [
    visibleCameras,
    cameraLanes,
    playbackFilterActive,
    activeAtPlayheadIds,
    coarsePlayheadVideoTime,
  ]);

  // Where stage0 / matched clips live: prefer gallery, then backend-resolved probe (session runId can be stale).
  const cropRunId = probeRunIdForMedia;

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden">
      <div className="min-h-0 min-w-0 flex-1 overflow-y-auto p-4 sm:p-6">
        <div className="mx-auto flex max-w-7xl flex-col gap-4">
          <ErrorBanner
            title="Association failed"
            message={stage4Progress?.status === "error" ? stage4Progress.message || "Association failed" : null}
          />
          {stage4Progress?.status === "running" ? (
            <StageProgressCard
              title="Association progress"
              status={toStageStatus(stage4Progress)}
              progress={stage4Progress.progress}
              message={stage4Progress.message}
            />
          ) : null}

          <TimelineVideoGrid
            cameras={activeCamerasForGrid}
            splitCount={effectiveSplitCount}
            tracksLoading={tracksLoading}
            currentVideoId={currentVideo?.id}
            currentVideoTime={coarsePlayheadVideoTime}
            trackletPickTime={trackletPickTime}
            isPlaying={isPlaying}
            probeRunId={probeRunIdForMedia ?? undefined}
            cropRunId={cropRunId}
          />

          <PlaybackControls
            isPlaying={isPlaying}
            currentFrame={Math.round(currentTime * 2)}
            totalFrames={Math.max(1, Math.round(totalDuration * 2) + 1)}
            speedOptions={[]}
            onPlayPause={() => setIsPlaying(!isPlaying)}
            onStepBack={() => setCurrentTime(0)}
            onStepForward={() => setCurrentTime(totalDuration)}
            onFrameChange={(frame) => setCurrentTime(Math.min(totalDuration, frame / 2))}
          />
          <div className="-mt-3 flex flex-wrap items-center justify-between gap-2 px-1 text-[11px] tabular-nums tracking-wide text-muted-foreground/70">
            <span>
              {tracks.length > 0 ? (
                <>
                  <span className="text-foreground/85">{formatDuration(timelineStart + currentTime)}</span>
                  <span className="text-muted-foreground/35"> / </span>
                  {formatDuration(timelineEnd)}
                </>
              ) : (
                "..."
              )}
            </span>
            {tracks.length > 0 ? (
              <span>video {formatDuration(absCurrentTime)}</span>
            ) : null}
          </div>

          <div className="grid min-h-[360px] gap-4 lg:grid-cols-[minmax(0,1fr)_300px]">
            <div className="flex min-h-[320px] min-w-0 overflow-hidden rounded-md border bg-card">
              <NLETimeline
                timelineRef={timelineRef}
                cameraLanes={cameraLanes}
                selectedLaneId={selectedLaneId}
                timelineStart={timelineStart}
                timelineEnd={timelineEnd}
                currentTime={currentTime}
                playheadVideoTime={absCurrentTime}
                rulerTickCount={rulerTickCount}
                rulerTickInterval={rulerTickInterval}
                rulerPlayheadLeft={rulerPlayheadLeft}
                timeToPixel={timeToPixel}
                onLaneClick={(laneId) => setSelectedLaneId(selectedLaneId === laneId ? null : laneId)}
              />
            </div>
            <aside className="flex min-h-[320px] min-w-0 flex-col overflow-hidden rounded-md border bg-muted/20">
              <TrackletRail
                listRef={trajectoryListRef}
                tracks={tracks}
                visibleTracks={trajectoryListTracks}
                tracksLoading={tracksLoading}
                selectedTrackId={selectedTrackId}
                selectedTrackletCount={selectedTrackletCount}
                stage4Progress={stage4Progress}
                playingTrackletsOnly={playingTrackletsOnly}
                activeAtPlayheadIds={activeAtPlayheadIds}
                onSelectTrack={handleTrackClick}
                onConfirmToggle={handleConfirmToggle}
                onRemoveTrack={removeTrack}
              />
            </aside>
          </div>

          <DisclosurePanel title="Advanced" description="Camera grid density, timeline zoom, and playhead-follow behavior.">
            <TimelineAdvancedControls
              effectiveSplitCount={effectiveSplitCount}
              cameraCount={camerasForPreview.length}
              playbackFilterActive={playbackFilterActive}
              playingTrackletsOnly={playingTrackletsOnly}
              zoom={zoom}
              tracksCount={tracks.length}
              onSplitCountChange={setSplitCount}
              onPlayingTrackletsOnlyChange={setPlayingTrackletsOnly}
              onZoomChange={setZoom}
            />
          </DisclosurePanel>

          <DisclosurePanel title="Debug" tier="debug" description="Raw Stage 4 counters, resolved run identifiers, and association status.">
            <TimelineDebugPanel
              runId={runId}
              galleryRunId={galleryRunId}
              resolvedProbeRunId={resolvedProbeRunId}
              currentStage={currentStage}
              triggerReload={triggerReload}
              downstreamInvalidateGeneration={downstreamInvalidateGeneration}
              tracksCount={tracks.length}
              confirmedCount={confirmedCount}
              selectedTrackletCount={selectedTrackletCount}
              cameraLaneCount={shownCameraLanes}
              alternativesCount={topAlternatives.length}
              alternativesCameraCount={alternativesCameraCount}
              timelineDataSource={timelineDataSource}
              stage4Progress={stage4Progress}
            />
          </DisclosurePanel>

          <AlternativesSheet
            open={alternativesOpen}
            onOpenChange={setAlternativesOpen}
            selectedTrack={selectedTrack}
            alternatives={topAlternatives}
            alternativesLoading={alternativesLoading}
            alternativesError={alternativesError}
            alternativesCameraCount={alternativesCameraCount}
            probeRunId={probeRunIdForMedia}
            onApplyAlternative={handleApplyAlternative}
          />
        </div>
      </div>
    </div>
  );
}

export function TimelineStageActions() {
  const { runId, galleryRunId, stages, updateStageProgress } = usePipelineStore();
  const { setCurrentStage } = useSessionStore();
  const stage4Progress = stages.find((stage) => stage.stage === 4);
  const isRunning = stage4Progress?.status === "running";
  const cancelRunId = galleryRunId ?? runId;

  const dispatchTimelineEvent = (eventName: string) => {
    window.dispatchEvent(new Event(eventName));
  };

  const handleCancel = async () => {
    if (!cancelRunId) {
      updateStageProgress(4, { status: "idle", progress: 0, message: "Stage 4 cancelled" });
      return;
    }

    try {
      await cancelPipeline(cancelRunId);
    } finally {
      updateStageProgress(4, { status: "idle", progress: 0, message: "Stage 4 cancelled" });
    }
  };

  return (
    <>
      <ExecutionTargetToggle stage={4} variant="compact" />
      {isRunning ? (
        <Button type="button" variant="outline" onClick={() => void handleCancel()} aria-label="Cancel Stage 4 association run">
          Cancel
        </Button>
      ) : null}
      <Button
        type="button"
        variant="outline"
        disabled={isRunning}
        aria-label="Run Stage 4 association"
        onClick={() => dispatchTimelineEvent(TIMELINE_RERUN_ASSOCIATION_EVENT)}
      >
        <RefreshCw className={cn("mr-2 h-4 w-4", isRunning && "animate-spin")} />
        Run Association
      </Button>
      <Button type="button" variant="outline" onClick={() => dispatchTimelineEvent(TIMELINE_SHOW_ALTERNATIVES_EVENT)} aria-label="Show timeline alternatives">
        Alternatives
      </Button>
      <Button type="button" onClick={() => setCurrentStage(5)}>
        Continue to Refinement
        <ArrowRight className="ml-2 h-4 w-4" />
      </Button>
    </>
  );
}
