/** Mirrors `history.json` / `TrainingHistory` aliases from the Python API. */
export interface TrainingHistoryJson {
  history_train_loss: number[];
  history_val_loss: number[];
  history_train_acc: number[];
  history_val_acc: number[];
  history_train_precision: number[];
  history_val_precision: number[];
  history_train_recall: number[];
  history_val_recall: number[];
  history_train_f1: number[];
  history_val_f1: number[];
  history_test_loss?: number[];
  history_test_accuracy?: number[];
  history_test_precision?: number[];
  history_test_recall?: number[];
  history_test_f1?: number[];
  elapsed_seconds?: number;
  epochs_ran?: number;
}

export interface RunConfigJson {
  train_path: string;
  val_ratio: number;
  layers: number[];
  epochs: number;
  learning_rate: number;
  seed: number;
  batch_size: number;
  optimizer: string;
  patience: number;
  test_paths?: string[];
  parent_dir?: string;
}

export interface RunListItem {
  id: string;
  relative_path: string;
  has_history: boolean;
  has_run_config: boolean;
  epochs_ran: number | null;
  elapsed_seconds: number | null;
  final_train_loss: number | null;
  final_val_loss: number | null;
  history_mtime_ms: number | null;
  config_train_path: string | null;
  config_layers_str: string | null;
  config_epochs: number | null;
  config_learning_rate: number | null;
  config_seed: number | null;
  config_batch_size: number | null;
  config_optimizer: string | null;
  config_patience: number | null;
}

export interface RunListResponse {
  runs_root: string;
  runs: RunListItem[];
}

export type TestEvalResult =
  | { loss: number; accuracy: number; precision: number; recall: number; f1: number }
  | { error: string };

export interface EvaluateTestResponse {
  test_path: string;
  results: Record<string, TestEvalResult>;
}

export interface DatasetListResponse {
  datasets_root: string;
  files: string[];
}

export interface PrepareDatasetResponse {
  source: string;
  output: string;
}

export interface SplitDatasetResponse {
  train_path: string;
  test_path: string;
  folder: string;
}

export interface RunDetailResponse {
  run_path: string;
  history: TrainingHistoryJson;
  run_config: RunConfigJson | null;
}

export type SseEvent =
  | {
      type: "batch";
      epoch: number;
      batch_index: number;
      n_batches: number;
      loss: number;
      grad_norm_per_layer: number[];
      weight_delta_norm_per_layer: number[];
    }
  | {
      type: "epoch";
      epoch: number;
      train: {
        loss: number;
        accuracy: number;
        precision: number;
        recall: number;
        f1: number;
      };
      val: {
        loss: number;
        accuracy: number;
        precision: number;
        recall: number;
        f1: number;
      } | null;
    }
  | {
      type: "done";
      elapsed_seconds: number;
      epochs_ran: number;
      history: TrainingHistoryJson;
    }
  | {
      type: "test_eval";
      loss: number;
      accuracy: number;
      precision: number;
      recall: number;
      f1: number;
      test_path: string;
      run_dir: string;
    }
  | { type: "error"; message: string };

/** best_summary-shaped object streamed during grid search */
export type BestSummaryPayload = Record<string, unknown>;

export type GridSseEvent =
  | {
      type: "grid_started";
      parent_dir: string;
      total: number;
      grid_mode: string;
      full_grid_combos: number;
    }
  | {
      type: "grid_combo_done";
      index: number;
      total: number;
      run_name: string;
      leaderboard: BestSummaryPayload;
    }
  | {
      type: "grid_final";
      parent_dir: string;
      best_run_dir: string;
      summary: BestSummaryPayload;
    }
  | { type: "done" }
  | { type: "error"; message: string };
