"use client";

import { useCallback } from "react";

import { useToast } from "@/hooks/use-toast";
import { getDatasetVideos, getRunDetail, type RunStageMap } from "@/lib/api";
import { usePipelineStore, useSessionStore, useVideoStore } from "@/store";
import type { StageNumber, VideoFile } from "@/types";

/** Re-open an existing run from disk: restore run id, input context, per-stage
 *  status, and the camera footage, then navigate to a sensible stage. */
export function useLoadRun() {
  const setRunId = usePipelineStore((s) => s.setRunId);
  const setRunInput = usePipelineStore((s) => s.setRunInput);
  const resetPipeline = usePipelineStore((s) => s.reset);
  const updateStageProgress = usePipelineStore((s) => s.updateStageProgress);
  const setPipelineStage = usePipelineStore((s) => s.setCurrentStage);
  const setVideos = useVideoStore((s) => s.setVideos);
  const setCurrentVideo = useVideoStore((s) => s.setCurrentVideo);
  const setSessionStage = useSessionStore((s) => s.setCurrentStage);
  const { toast } = useToast();

  return useCallback(
    async (runId: string): Promise<boolean> => {
      let detail;
      try {
        const res = await getRunDetail(runId);
        detail = res.data;
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        toast({ title: "Couldn't load run", description: msg, variant: "destructive" });
        return false;
      }
      if (!detail) {
        toast({ title: "Couldn't load run", description: `Run ${runId} not found`, variant: "destructive" });
        return false;
      }

      // Start from a clean slate, then restore this run's identity + input.
      resetPipeline();
      setRunId(detail.runId);
      if (detail.inputDir) {
        setRunInput({
          inputDir: detail.inputDir,
          cameras: detail.cameras ?? [],
          name: detail.name ?? "dataset",
          smoke: Boolean(detail.smoke),
        });
      }

      // Restore per-stage status from which stages produced output on disk.
      for (let i = 0; i <= 6; i += 1) {
        const present = detail.stages[`stage${i}` as keyof RunStageMap];
        updateStageProgress(i as StageNumber, present
          ? { status: "completed", progress: 100, message: "Loaded from disk" }
          : { status: "idle", progress: 0, message: "" });
      }

      // Restore the camera footage. Prefer freshly-probed records (full metadata
      // for the scrubber); fall back to the light records stored in run_context.
      let videos: VideoFile[] = [];
      if (detail.inputDir) {
        try {
          const vres = await getDatasetVideos(detail.inputDir);
          videos = vres.data ?? [];
        } catch {
          videos = [];
        }
      }
      if (videos.length === 0 && detail.videos.length > 0) {
        videos = detail.videos.map((v) => ({
          id: v.id,
          name: v.name,
          path: v.path,
          size: 0,
          duration: 0,
          fps: 0,
          width: 0,
          height: 0,
          uploadedAt: "",
          cameraId: v.cameraId,
          latestRunId: detail!.runId,
        }));
      }
      // Restrict to the run's cameras when known.
      const camSet = new Set((detail.cameras ?? []).map((c) => c.toLowerCase()));
      const runVideos = camSet.size > 0
        ? videos.filter((v) => v.cameraId && camSet.has(v.cameraId.toLowerCase()))
        : videos;
      const finalVideos = runVideos.length > 0 ? runVideos : videos;
      setVideos(finalVideos);
      setCurrentVideo(finalVideos[0] ?? null);

      // Land on the furthest meaningful stage: Detection if tracking ran, else Upload.
      // Land on the furthest useful UI stage for what's on disk:
      // assoc → Timeline, index/features → Inference, tracking/ingestion → Detection.
      const s = detail.stages;
      const target: StageNumber = s.stage4 ? 4 : (s.stage3 || s.stage2) ? 3 : (s.stage1 || s.stage0) ? 1 : 0;
      setSessionStage(target);
      setPipelineStage(target);

      toast({
        title: `Run ${detail.runId} opened`,
        description: detail.name ? `${detail.name} — ${finalVideos.length} cameras` : `${finalVideos.length} cameras restored`,
        variant: "success",
      });
      return true;
    },
    [resetPipeline, setRunId, setRunInput, updateStageProgress, setVideos, setCurrentVideo, setSessionStage, setPipelineStage, toast]
  );
}
