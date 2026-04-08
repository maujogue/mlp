# Visualizer HTTP API contract

Base URL: `http://127.0.0.1:8765` (default for `mlp-visualizer`). All JSON responses use UTF-8.

## Run discovery (replay)

### `GET /api/health`

Returns `{ "ok": true }`.

### `GET /api/runs`

Query parameters:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `root` | string | `temp` | Directory to scan (relative to process CWD or absolute). |

Response: `RunListResponse`

```json
{
  "runs_root": "/abs/path/temp",
  "runs": [
    {
      "id": "train-data_layers-24-24-12_epochs-10_...",
      "relative_path": "train-data_layers-24-24-12_epochs-10_...",
      "has_history": true,
      "has_run_config": true,
      "epochs_ran": 10,
      "elapsed_seconds": 1.23,
      "final_train_loss": 0.45,
      "final_val_loss": 0.52,
      "history_mtime_ms": 1735689600000,
      "config_train_path": "/abs/path/train.csv",
      "config_layers_str": "24-24-12",
      "config_epochs": 70,
      "config_learning_rate": 0.01,
      "config_seed": 42,
      "config_batch_size": 32,
      "config_optimizer": "rmsprop",
      "config_patience": 10
    }
  ]
}
```

- **`id`**: Same as `relative_path` under `root` (POSIX, no leading `./`). Use URL path segment encoding when calling detail endpoints.
- Metrics `final_*` are taken from the last epoch in `history.json` when present.
- **`history_mtime_ms`**: `history.json` modification time as Unix milliseconds (`null` if unavailable).
- **`config_*`**: Populated from `run_config.json` when present; otherwise `null`. Use for filtering experiments in the UI.

---

## Datasets (prepare / split / list)

Paths are resolved relative to the visualizer process **current working directory** (project root when started normally). The API rejects paths that escape that directory.

### `GET /api/datasets`

Query:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `root` | string | `datasets` | Directory to scan recursively for `*.csv`. |

Response:

```json
{
  "datasets_root": "/abs/path/to/project/datasets",
  "files": ["data_prepared.csv", "some_subdir/train.csv"]
}
```

- **`files`**: Paths are POSIX, relative to `datasets_root`, sorted. Use with `root` from the query to build train/test paths (e.g. `datasets/foo/train.csv` when `root` is `datasets` and the relative file is `foo/train.csv`).

### `POST /api/datasets/prepare`

JSON body (`PrepareDatasetRequest`):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source` | string | yes | Raw CSV path (relative or absolute). |
| `output` | string | no | If omitted, writes `{stem}_prepared{suffix}` next to `source` (same as CLI). |

Response: `{ "source": "<resolved>", "output": "<resolved>" }`

Runs the same pipeline as `mlp-prepare-dataset` (load, fix columns / missing values, write CSV).

### `POST /api/datasets/split`

JSON body (`SplitDatasetRequest`):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prepared_path` | string | yes | Prepared CSV (relative or absolute). |
| `test_size` | number | no | Default `0.2`; must be in `(0, 1)`. |

Response: `{ "train_path": "<abs>", "test_path": "<abs>", "folder": "<abs parent of train/test>" }`

Writes `train.csv` and `test.csv` under `datasets/<prepared_stem>/` relative to CWD (same as `mlp-split`).

### `POST /api/runs/evaluate-test`

JSON body:

| Field | Type | Description |
|-------|------|-------------|
| `root` | string | Same as `/api/runs` `root`. |
| `test_path` | string | CSV path (relative to CWD or absolute). |
| `run_paths` | string[] | Run ids (relative paths under `root`), max 400 per request. |

Response: `{ "test_path": "<resolved>", "results": { "<run_id>": { "loss", "accuracy", "precision", "recall", "f1" } | { "error": "..." } } }`

Each run must contain `model.pkl` and `scaler.pkl` (saved training). `loss` is binary cross-entropy on the test set.

### `GET /api/runs/{run_path:path}`

Path is the run’s path **relative to** `runs_root` (e.g. `subdir/my_run_20250101-120000`).

Query: `root` (same as `/api/runs`).

Response: `RunDetailResponse`

```json
{
  "run_path": "temp/...",
  "history": { ... },
  "run_config": { ... }
}
```

- **`history`**: Matches on-disk `history.json` after Pydantic validation (`TrainingHistory` aliases: `history_train_loss`, etc.).
- **`run_config`**: Matches `run_config.json` (`TrainingRunConfig`). Omitted if file is missing.

### `GET /api/best-summary`

Query:

| Name | Type | Description |
|------|------|-------------|
| `parent` | string | Directory containing `best_summary.json` (relative or absolute). |

Response: raw JSON object from `best_summary.json`, or 404 if missing.

---

## Live training + SSE

### `POST /api/live/train`

JSON body (`LiveTrainRequest`):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `train_path` | string | yes | CSV path (relative or absolute). |
| `val_ratio` | number | no | Default `0.2`. |
| `layers` | int[] | no | Default `[24, 24, 12]`. |
| `epochs` | int | no | Default `70`. |
| `learning_rate` | number | no | Default `0.01`. |
| `seed` | int | no | Default `42`. |
| `batch_size` | int | no | Default `0` (full batch). |
| `optimizer` | `"sgd"` \| `"rmsprop"` | no | Default `rmsprop`. |
| `patience` | int | no | Default `0`. |
| `parent_dir` | string | no | Run parent dir; default `temp`. |
| `telemetry_sample_every_n_batches` | int | no | `1` = every batch; `N` = every Nth batch only. |
| `eval_test_path` | string | no | If set, saves the run to disk and evaluates on this CSV; SSE includes `test_eval` before `done`. |
| `lesson_mode` | bool | no | Default `false`. When `true`, records a micro-step replay during training and defers the SSE `done` until after `fit` completes. |
| `lesson_max_micro_steps` | int | no | Default `25000` (min `500`, max `500000`). If the estimated replay size exceeds this, the API returns **400** — reduce epochs, use a larger batch, or raise the cap. |

Response: `{ "session_id": "<uuid>" }`

Training runs in a background thread. Events are read via SSE.

### `GET /api/live/stream/{session_id}`

`Content-Type: text/event-stream`

Each event is one SSE message: `data: <json>\n\n`

Event types (`type` field inside JSON):

| `type` | Payload |
|--------|---------|
| `batch` | `epoch`, `batch_index`, `n_batches`, `loss`, `grad_norm_per_layer` (float[]), `weight_delta_norm_per_layer` (float[]) |
| `epoch` | `epoch`, `train` (TrainingMetrics-like dict), `val` (optional same shape) |
| `test_eval` | `loss`, `accuracy`, `precision`, `recall`, `f1`, `test_path`, `run_dir` (only when `eval_test_path` was set; sent before `done`) |
| `done` | `elapsed_seconds`, `epochs_ran`, optional `history` (epoch arrays summary). With `lesson_mode`: may include `lesson_manifest` and `lesson_steps` inline when the serialized payload is small; otherwise `lesson_replay_run_dir` (absolute path under the server CWD) so the client can load replay files via `GET /api/live/lesson-replay`. |
| `error` | `message` (string) |

When `lesson_mode` is **true** (or when `eval_test_path` is set), **`done` is emitted only after training finishes** (not immediately at thread start), so the client should wait for `done` before treating the session as complete.

### `GET /api/live/lesson-replay`

Query:

| Name | Type | Description |
|------|------|-------------|
| `run_dir` | string | Run directory that contains a `lesson_replay/` folder with `replay_manifest.json` and `replay_steps.jsonl` (path relative to process CWD or absolute, must stay under CWD). |

Response: `{ "lesson_manifest": { ... }, "lesson_steps": [ ... ] }` — same shapes as the optional inline fields on SSE `done`.

---

## Grid search (best) + SSE

### `POST /api/live/best`

JSON body (`LiveBestRequest`):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `train_path` | string | yes | CSV path (relative or absolute). |
| `val_ratio` | number | no | Default `0.2`. |
| `epochs` | int | no | Default `70`. |
| `seed` | int | no | Default `42`. |
| `parent_dir` | string | no | Directory under which `best_<stem>_<timestamp>/` is created; default `temp`. |
| `test_paths` | string[] | no | Optional test CSVs; after all grid runs, models are evaluated and ranking uses test metrics when set. |
| `grid_mode` | `"quick"` \| `"full"` | no | Default `quick` (~12 combos). `full` runs the full CLI grid (~600 combos). |

Response: `{ "session_id": "<uuid>" }`

### `GET /api/live/best/stream/{session_id}`

Same framing as live training SSE. Event types:

| `type` | Payload |
|--------|---------|
| `grid_started` | `parent_dir`, `total`, `grid_mode`, `full_grid_combos` |
| `grid_combo_done` | `index`, `total`, `run_name`, `leaderboard` (same shape as `best_summary.json` for completed runs so far) |
| `grid_final` | `parent_dir`, `best_run_dir`, `summary` (full `best_summary.json` object) |
| `done` | (empty object) |
| `error` | `message` |

---

## Lesson loss slice (optional experiment API)

### `POST /api/lesson/loss-slice`

JSON body: `root`, `replay_path`, `step_index`, `param_i`, `param_j`, `grid_half_extent`, `grid_n`.

- Set `replay_path` to a **folder name** under `root` (default `lesson_replays`), or to an **absolute path** to a directory that already contains `replay_manifest.json` / steps (or a run directory whose `lesson_replay/` subfolder does), resolved under the server’s current working directory.

Evaluates mean training loss on the manifest’s toy points (when `toy_points` are present in the manifest) while varying two scalars in the flattened weight vector; other parameters stay fixed at the snapshot implied by micro-steps up to `step_index`. For **tabular** replays from live `lesson_mode`, toy points may be absent — this endpoint is mainly useful when the manifest includes toy geometry.

Response: `{ "param_i", "param_j", "center_i", "center_j", "x_axis", "y_axis", "z", "n_params" }` where `z` is a 2D grid of loss values.

---

## CORS

The server enables CORS for all origins in development so the Vite dev server can call the API.
