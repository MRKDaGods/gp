import type { TrajectorySegment } from "@/types";

export type TimelineCameraLaneSegment = TrajectorySegment & {
  trajectoryId: string;
  globalId?: number;
  confidence?: number;
  className?: string;
  confirmed?: boolean;
};

export type TimelineCameraLaneSegmentWithSum = TimelineCameraLaneSegment & {
  sumStart: number;
  sumEnd: number;
};

export type TimelineCameraLane = {
  id: string;
  cameraId: string;
  label: string;
  startTime: number;
  endTime: number;
  segments: TimelineCameraLaneSegmentWithSum[];
};

export type TimelinePreviewCamera = {
  id: string;
  name: string;
  location: string;
  activeTrack?: any;
  isPast?: boolean;
  isNext?: boolean;
  segment?: TimelineCameraLaneSegmentWithSum;
  primarySeg?: {
    globalId?: number;
    cameraId: string;
    trackId: number;
    start: number;
    end: number;
  };
};
