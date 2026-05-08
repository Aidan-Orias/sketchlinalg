export type DatasetSummary = {
  name: string;
  task: string;
  train_rows: number;
  test_rows: number;
  features: number;
  train_nnz: number;
  test_nnz: number;
  train_density: number;
  test_density: number;
  target_mean: number;
  target_std: number;
  target_min: number;
  target_max: number;
};

export type BenchmarkOptions = {
  sketch_dimension_min: number;
  sketch_dimension_max: number;
  sketch_dimension_step: number;
  default_sketch_dimension: number;
  alpha_values: number[];
  default_alpha: number;
  default_repeats: number;
  default_seed: number;
};

export type BenchmarkRequest = {
  sketch_dimension: number;
  alpha: number;
  repeats: number;
  seed: number | null;
};

export type BenchmarkTimings = {
  sketch_construction: number;
  sketched_fit: number;
  regular_fit: number;
  total_sketched: number;
  total_regular: number;
};

export type BenchmarkMetrics = {
  fit_speedup: number;
  total_speedup: number;
  regular_rmse: number;
  sketched_rmse: number;
  rmse_ratio: number;
};

export type BenchmarkResult = {
  sketch_dimension: number;
  alpha: number;
  repeats: number;
  timings: BenchmarkTimings;
  metrics: BenchmarkMetrics;
};

export type BenchmarkJobResponse = {
  job_id: string;
  status: "queued" | "running" | "complete" | "error";
};

export type BenchmarkPhase =
  | "queued"
  | "load_dataset"
  | "fit_regular"
  | "construct_sketch"
  | "fit_sketched"
  | "complete"
  | "error";

export type BenchmarkProgressEvent = {
  type: "progress" | "complete" | "error";
  job_id: string;
  phase: BenchmarkPhase;
  progress: number;
  elapsed_seconds?: number | null;
  message?: string | null;
  result?: BenchmarkResult | null;
  error?: string | null;
};
