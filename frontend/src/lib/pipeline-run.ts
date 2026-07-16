/**
 * Shared helpers for the PROBE flow: running a single pipeline stage against an
 * uploaded video (via /api/pipeline/run-stage/{stage}) and polling it to
 * completion. This is distinct from the dataset flow (useRunPipelineStage),
 * which runs a stage against a run created from a dataset/ folder via
 * /api/datasets/run.
 */
import { ApiError, getPipelineStatus } from "@/lib/api";
import type { StageNumber } from "@/types";

/** Derive a camera id for API calls: prefer the video's real camera id (e.g.
 * WILDTRACK "C1".."C7"), else the CityFlow S##_c### pattern, else a default. */
export function inferCameraId(video: { name: string; path: string; cameraId?: string | null } | null): string {
  if (!video) return "S02_c008";
  if (video.cameraId && video.cameraId.trim()) return video.cameraId.trim();
  const candidate = `${video.name} ${video.path}`;
  const match = candidate.match(/S\d{2}_c\d{3}/i);
  return (match?.[0] ?? "S02_c008").toUpperCase();
}

/** Poll a single-stage run (started via runStage) until it reaches a terminal state. */
export async function pollStageStatus(
  activeRunId: string,
  stage: StageNumber,
  updateStageProgress: (stage: StageNumber, progress: any) => void
): Promise<void> {
  // eslint-disable-next-line no-constant-condition
  while (true) {
    const statusResponse = await getPipelineStatus(activeRunId);
    const statusData: any = statusResponse.data;
    const status = String(statusData?.status ?? "running");
    const progress = Number(statusData?.progress ?? 0);
    const message = String(statusData?.message ?? `Stage ${stage} running...`);

    updateStageProgress(stage, { status: "running", progress, message });

    if (status === "completed") {
      updateStageProgress(stage, { status: "completed", progress: 100, message: `Stage ${stage} complete` });
      return;
    }

    if (status === "cancelled") {
      updateStageProgress(stage, { status: "idle", progress: 0, message: `Stage ${stage} cancelled` });
      return;
    }

    if (status === "error") {
      throw new Error(String(statusData?.error ?? `Stage ${stage} failed`));
    }

    await new Promise((resolve) => setTimeout(resolve, 1200));
  }
}

export function getRunStageErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 401) return "Kaggle credentials missing or invalid. Configure them in the sidebar settings.";
    if (error.status === 429) return "Both Kaggle slots busy - try again later";
    if (error.status === 400) {
      const data = error.data as { detail?: unknown; message?: unknown } | undefined;
      return String(data?.detail ?? data?.message ?? error.message);
    }
    if (error.status === 500) return "Kaggle dispatch failed. Falling back to local? Check backend logs.";
  }

  return error instanceof Error ? error.message : "Pipeline stage failed";
}
