// CityFlow Demo Data - Hardcoded for working visual demonstration
// Based on AI City Challenge CityFlowV2 dataset structure

export const CITYFLOW_CAMERAS = [
  { id: "S01_c001", scene: "S01", name: "Camera 001", location: "Intersection A - North" },
  { id: "S01_c002", scene: "S01", name: "Camera 002", location: "Intersection A - East" },
  { id: "S01_c003", scene: "S01", name: "Camera 003", location: "Intersection A - South" },
  { id: "S02_c006", scene: "S02", name: "Camera 006", location: "Intersection B - North" },
  { id: "S02_c007", scene: "S02", name: "Camera 007", location: "Intersection B - West" },
  { id: "S02_c008", scene: "S02", name: "Camera 008", location: "Intersection B - South" },
];

export const DEMO_VIDEO = {
  id: "cityflow-demo",
  name: "CityFlowV2_S01_Query.mp4",
  path: "/demo/cityflow",
  size: 52428800,
  duration: 127,
  fps: 10,
  width: 1920,
  height: 1080,
  uploadedAt: new Date().toISOString(),
};

// Detection results - simulating YOLO output
export const DEMO_DETECTIONS = [
  {
    id: "det-001",
    frameNumber: 45,
    timestamp: 4.5,
    bbox: { x: 450, y: 320, width: 180, height: 120 },
    confidence: 0.94,
    classId: 2, // car
    className: "car",
    color: "white",
    trackId: "T001",
  },
  {
    id: "det-002",
    frameNumber: 45,
    timestamp: 4.5,
    bbox: { x: 820, y: 380, width: 200, height: 140 },
    confidence: 0.91,
    classId: 2, // car
    className: "car",
    color: "black",
    trackId: "T002",
  },
  {
    id: "det-003",
    frameNumber: 45,
    timestamp: 4.5,
    bbox: { x: 1200, y: 290, width: 160, height: 100 },
    confidence: 0.88,
    classId: 2, // car
    className: "car",
    color: "red",
    trackId: "T003",
  },
  {
    id: "det-004",
    frameNumber: 45,
    timestamp: 4.5,
    bbox: { x: 150, y: 450, width: 220, height: 160 },
    confidence: 0.96,
    classId: 7, // truck
    className: "truck",
    color: "blue",
    trackId: "T004",
  },
  {
    id: "det-005",
    frameNumber: 45,
    timestamp: 4.5,
    bbox: { x: 650, y: 520, width: 190, height: 130 },
    confidence: 0.89,
    classId: 2, // car
    className: "car",
    color: "silver",
    trackId: "T005",
  },
  {
    id: "det-006",
    frameNumber: 45,
    timestamp: 4.5,
    bbox: { x: 1050, y: 480, width: 170, height: 110 },
    confidence: 0.92,
    classId: 2, // car
    className: "car",
    color: "green",
    trackId: "T006",
  },
];

// Timeline tracklets - vehicles tracked across cameras
export const DEMO_TRACKLETS = [
  {
    id: "track-001",
    vehicleId: "V-2847",
    color: "#ef4444", // red
    vehicleType: "sedan",
    vehicleColor: "white",
    confidence: 0.94,
    appearances: [
      { cameraId: "S01_c001", startTime: 0, endTime: 8, startFrame: 0, endFrame: 80 },
      { cameraId: "S01_c002", startTime: 12, endTime: 22, startFrame: 120, endFrame: 220 },
      { cameraId: "S01_c003", startTime: 28, endTime: 35, startFrame: 280, endFrame: 350 },
    ],
  },
  {
    id: "track-002",
    vehicleId: "V-1923",
    color: "#22c55e", // green
    vehicleType: "SUV",
    vehicleColor: "black",
    confidence: 0.91,
    appearances: [
      { cameraId: "S01_c003", startTime: 5, endTime: 15, startFrame: 50, endFrame: 150 },
      { cameraId: "S01_c001", startTime: 20, endTime: 28, startFrame: 200, endFrame: 280 },
      { cameraId: "S01_c002", startTime: 35, endTime: 45, startFrame: 350, endFrame: 450 },
    ],
  },
  {
    id: "track-003",
    vehicleId: "V-4512",
    color: "#3b82f6", // blue
    vehicleType: "truck",
    vehicleColor: "blue",
    confidence: 0.88,
    appearances: [
      { cameraId: "S02_c006", startTime: 10, endTime: 25, startFrame: 100, endFrame: 250 },
      { cameraId: "S02_c007", startTime: 32, endTime: 48, startFrame: 320, endFrame: 480 },
    ],
  },
  {
    id: "track-004",
    vehicleId: "V-3391",
    color: "#f59e0b", // amber
    vehicleType: "sedan",
    vehicleColor: "red",
    confidence: 0.93,
    appearances: [
      { cameraId: "S01_c002", startTime: 2, endTime: 10, startFrame: 20, endFrame: 100 },
      { cameraId: "S01_c003", startTime: 18, endTime: 30, startFrame: 180, endFrame: 300 },
      { cameraId: "S02_c008", startTime: 55, endTime: 70, startFrame: 550, endFrame: 700 },
    ],
  },
  {
    id: "track-005",
    vehicleId: "V-7824",
    color: "#8b5cf6", // purple
    vehicleType: "van",
    vehicleColor: "silver",
    confidence: 0.86,
    appearances: [
      { cameraId: "S02_c007", startTime: 0, endTime: 12, startFrame: 0, endFrame: 120 },
      { cameraId: "S02_c008", startTime: 18, endTime: 32, startFrame: 180, endFrame: 320 },
      { cameraId: "S02_c006", startTime: 40, endTime: 55, startFrame: 400, endFrame: 550 },
    ],
  },
];

// Output statistics
export const DEMO_STATS = {
  totalVehicles: 127,
  uniqueVehicles: 45,
  avgTrackLength: 28.4,
  crossCameraMatches: 38,
  avgConfidence: 0.89,
  processingTime: 45.2,
  camerasAnalyzed: 6,
  totalFrames: 12700,
  detectionRate: 0.94,
  reIdAccuracy: 0.87,
};

// For generating synthetic camera frames with vehicles
export function generateCameraFrame(cameraId: string, frameNum: number) {
  const camera = CITYFLOW_CAMERAS.find(c => c.id === cameraId);
  return {
    cameraId,
    cameraName: camera?.name || cameraId,
    location: camera?.location || "Unknown",
    frameNumber: frameNum,
    timestamp: new Date().toISOString(),
    vehicles: DEMO_DETECTIONS.slice(0, Math.floor(Math.random() * 4) + 2),
  };
}
