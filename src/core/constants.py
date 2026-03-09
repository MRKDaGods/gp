"""Shared constants for the MTMC tracking pipeline."""

# COCO class IDs used in detection
COCO_PERSON = 0
COCO_CAR = 2
COCO_BUS = 5
COCO_TRUCK = 7

# Mapping from COCO class ID to human-readable name
CLASS_NAMES = {
    COCO_PERSON: "person",
    COCO_CAR: "car",
    COCO_BUS: "bus",
    COCO_TRUCK: "truck",
}

# Class groups for ReID model selection
PERSON_CLASSES = {COCO_PERSON}
VEHICLE_CLASSES = {COCO_CAR, COCO_BUS, COCO_TRUCK}

# Default file names used across the pipeline
FRAME_MANIFEST_FILE = "frames_manifest.json"
TRACKLETS_FILE_PATTERN = "tracklets_{camera_id}.json"
EMBEDDINGS_FILE = "embeddings.npy"
EMBEDDING_INDEX_FILE = "embedding_index.json"
HSV_FEATURES_FILE = "hsv_features.npy"
FAISS_INDEX_FILE = "faiss_index.bin"
METADATA_DB_FILE = "metadata.db"
GLOBAL_TRAJECTORIES_FILE = "global_trajectories.json"
EVALUATION_REPORT_FILE = "evaluation_report.json"

# Default model file names
YOLO_WEIGHTS = "yolo26m.pt"
PERSON_REID_WEIGHTS = "person_transreid_vit_base_market1501.pth"
VEHICLE_REID_WEIGHTS = "vehicle_transreid_vit_base_veri776.pth"
PCA_TRANSFORM_FILE = "pca_transform.pkl"
