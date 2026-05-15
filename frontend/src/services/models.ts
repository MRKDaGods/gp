const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8004/api";

export type ModelTaskType = "mtmc_vehicle" | "mtmc_person" | "single_cam_reid" | "detector_only";
export type ModelStatus = "production" | "research" | "dead_end" | "reference";
export type DatasetName = "cityflowv2" | "wildtrack" | "veri776" | "custom";

export interface ModelMetricSource {
  kind: "kernel_summary" | "local_json" | "log_line" | "docs";
  path: string;
  kernel?: string | null;
  line_ref?: string | null;
}

export interface ModelMetric {
  name: string;
  value: number;
  verified: boolean;
  source: ModelMetricSource;
  note?: string | null;
}

export interface HostedCheckpoint {
  kaggle_dataset: string;
  member: string;
}

export interface CheckpointRef {
  role: string;
  local_path: string;
  expected_sha256?: string | null;
  hosted: HostedCheckpoint[];
  source_training_kernel?: string | null;
  size_bytes?: number | null;
  on_disk: boolean;
}

export interface ModelRequirements {
  gpu_required: boolean;
  min_vram_gb: number;
  data_dependencies: string[];
}

export interface ModelProvenance {
  created_at: string;
  created_by_kernel?: string | null;
  verified_by: string;
}

export interface ModelTombstone {
  reason: string;
  superseded_by_id?: string | null;
}

export interface ModelEntry {
  id: string;
  name: string;
  task_type: ModelTaskType;
  dataset: DatasetName;
  description: string;
  metrics: ModelMetric[];
  pipeline_config: string | null;
  model_overrides: string[];
  checkpoint_refs: CheckpointRef[];
  requirements: ModelRequirements;
  status: ModelStatus;
  runnable_locally: boolean;
  notebook_or_kernel_ref: string | null;
  provenance: ModelProvenance;
  tombstone: ModelTombstone | null;
  missing_checkpoints: string[];
}

export interface ModelFilters {
  task_type?: ModelTaskType;
  status?: ModelStatus;
  include_dead_ends?: boolean;
}

interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

async function fetchRegistry<T>(endpoint: string): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    headers: { "Content-Type": "application/json" },
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    const message = errorData?.detail || errorData?.message || `HTTP ${response.status}`;
    throw new Error(String(message));
  }

  const parsed = (await response.json()) as ApiResponse<T>;
  if (!parsed.success) {
    throw new Error(parsed.error || parsed.message || "Model registry request failed");
  }

  return parsed.data as T;
}

export async function fetchModels(filters: ModelFilters = {}): Promise<ModelEntry[]> {
  const params = new URLSearchParams();
  if (filters.task_type) params.set("task_type", filters.task_type);
  if (filters.status) params.set("status", filters.status);
  if (filters.include_dead_ends) params.set("include_dead_ends", "true");

  const query = params.toString();
  return fetchRegistry<ModelEntry[]>(`/models${query ? `?${query}` : ""}`);
}

export async function fetchModel(id: string): Promise<ModelEntry> {
  return fetchRegistry<ModelEntry>(`/models/${encodeURIComponent(id)}`);
}