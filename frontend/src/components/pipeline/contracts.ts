import type { ContractBannerProps } from "./shell/ContractBanner";

export type StageContract = Pick<ContractBannerProps, "stage" | "title" | "needs" | "produces" | "helpText">;

export const STAGE_CONTRACTS: StageContract[] = [
  {
    stage: 0,
    title: "Upload",
    needs: [],
    produces: [{ label: "Video" }, { label: "runId" }],
    helpText: "Import a video or prepared run artifact. This creates the run context used by every downstream stage.",
  },
  {
    stage: 1,
    title: "Detection",
    needs: [{ label: "Video", stage: 0 }],
    produces: [{ label: "Detections" }, { label: "Tracklets" }],
    helpText: "Detect objects and build single-camera tracklets. These tracks become the selectable candidates for cross-camera search.",
  },
  {
    stage: 2,
    title: "Selection",
    needs: [{ label: "Tracklets", stage: 1 }],
    produces: [{ label: "Selected track IDs" }],
    helpText: "Choose the tracklets that should seed the ReID search. Selected IDs are passed into inference.",
  },
  {
    stage: 3,
    title: "Inference",
    needs: [{ label: "Selected tracks", stage: 2 }, { label: "Video", stage: 0 }],
    produces: [{ label: "ReID embeddings" }, { label: "FAISS index" }],
    helpText: "Extract ReID features and build the searchable index. Model and fusion controls remain inside the existing stage UI for now.",
  },
  {
    stage: 4,
    title: "Timeline",
    needs: [{ label: "Embeddings", stage: 3 }, { label: "Index", stage: 3 }],
    produces: [{ label: "Confirmed cross-camera tracks" }],
    helpText: "Associate indexed tracklets across cameras and review the resulting timeline before refinement.",
  },
  {
    stage: 5,
    title: "Refinement",
    needs: [{ label: "Confirmed tracks", stage: 4 }],
    produces: [{ label: "Refined tracks" }],
    helpText: "Select stronger references and re-search when the confirmed trajectory needs cleanup.",
  },
  {
    stage: 6,
    title: "Output",
    needs: [{ label: "Refined tracks", stage: 5 }],
    produces: [{ label: "Summary video" }, { label: "Exports" }],
    helpText: "Render the final summary video and export trajectories for review or handoff.",
  },
];

export function stageContract(stage: number): StageContract {
  return STAGE_CONTRACTS.find((contract) => contract.stage === stage) ?? STAGE_CONTRACTS[0];
}
