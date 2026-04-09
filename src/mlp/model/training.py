import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..data.data_engineering import (
    fit_scaler_on_train_and_transform_train_val,
    fix_dataset,
    split_features_labels,
    split_train_validation,
)
from ..utils.constants import FEATURE_COLUMNS
from ..utils.loader import build_run_dir, load_dataset
from .mlp_classifier import LESSON_REPLAY_ANCHOR_TRAIN_INDEX, MLPClassifier
from .plots import save_learning_curves
from .schemas import TrainingHistory, TrainingRunConfig
from .telemetry import TrainingTelemetryOptions
from .serialization import (
    save_model,
    save_run_config,
    save_training_history,
)

from mlp.visualizer.replay_schemas import ReplayManifest
from mlp.visualizer.replay_writer import (
    LessonReplayWriter,
    replay_step_to_jsonable,
    tabular_explanation,
)

INLINE_LESSON_MAX_BYTES = 3_500_000


def _load_and_prepare_train_val_arrays(
    train_path: Path,
    val_ratio: float,
    run_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load train_path, fix and split into train/val, scale, and return (X_train, y_train, X_val, y_val).
    """
    df = load_dataset(train_path)
    if "label" not in df.columns or not all(c in df.columns for c in FEATURE_COLUMNS):
        df = fix_dataset(df)
    train_df, val_df = split_train_validation(df, val_ratio)
    scaler_path: Path = run_dir / Path("scaler.pkl")
    train_df, val_df = fit_scaler_on_train_and_transform_train_val(
        train_df, val_df, scaler_path
    )
    X_train, y_train = split_features_labels(train_df)
    X_val, y_val = split_features_labels(val_df)
    X_train = X_train.to_numpy(dtype=np.float64)
    y_train = y_train.astype(np.int64).values
    X_val = X_val.to_numpy(dtype=np.float64)
    y_val = y_val.astype(np.int64).values
    return X_train, y_train, X_val, y_val


def train_cmd(
    run_config: TrainingRunConfig,
) -> Path:
    # batch_size=0 means full dataset per step (full-batch gradient descent)
    # patience=0 means early stopping disabled
    # val_ratio: fraction of train_path to use as validation (train is split into train/val)
    run_dir: Path = build_run_dir(run_config)
    curves_dir: Path = run_dir / "figures"
    run_training(
        run_dir,
        run_config,
        telemetry=None,
        save_artifacts=True,
    )
    print(f"Training figures saved to {curves_dir}")
    return run_dir


def run_training(
    run_dir: Path,
    run_config: TrainingRunConfig,
    *,
    telemetry: TrainingTelemetryOptions | None = None,
    save_artifacts: bool = True,
    after_save: Callable[[], None] | None = None,
    lesson_mode: bool = False,
) -> TrainingHistory:
    """Load data, train model, optionally persist artifacts into ``run_dir``."""
    curves_dir: Path = run_dir / "figures"
    model_path: Path = run_dir / "model.pkl"

    X_train, y_train, X_val, y_val = _load_and_prepare_train_val_arrays(
        train_path=run_config.train_path,
        val_ratio=run_config.val_ratio,
        run_dir=run_dir,
    )

    model = MLPClassifier(
        n_features=len(FEATURE_COLUMNS),
        hidden_layers=run_config.layers,
        output_size=2,
        seed=run_config.seed,
    )

    lesson_writer: LessonReplayWriter | None = None
    lesson_final_manifest: ReplayManifest | None = None
    if lesson_mode:
        n_train_samples = int(len(X_train))
        resolved_lesson_anchor = int(
            min(LESSON_REPLAY_ANCHOR_TRAIN_INDEX, max(0, n_train_samples - 1)),
        )
        lesson_dir = run_dir / "lesson_replay"
        lesson_writer = LessonReplayWriter(
            lesson_dir,
            run_id=run_dir.name,
            explain=tabular_explanation,
        )
        lesson_writer.emit_raw(
            {
                "phase": "init",
                "toc_id": "init",
                "epoch": 0,
                "batch": 0,
                "sample_in_batch": 0,
                "lesson_anchor_train_index": resolved_lesson_anchor,
                "lesson_trace_this_batch": None,
                "explanation": tabular_explanation(
                    {
                        "phase": "init",
                        "lesson_anchor_train_index": resolved_lesson_anchor,
                    },
                ),
            },
        )

    start_time = time.perf_counter()
    history: TrainingHistory = model.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        run_config=run_config,
        telemetry=telemetry,
        lesson_hook=lesson_writer.emit_raw if lesson_writer else None,
        on_lesson_batch_end=lesson_writer.note_batch_completed
        if lesson_writer
        else None,
        on_lesson_epoch_end=lesson_writer.end_epoch if lesson_writer else None,
    )

    elapsed_seconds = time.perf_counter() - start_time
    if lesson_writer is not None:
        manifest = ReplayManifest(
            run_id=run_dir.name,
            input_dim=len(FEATURE_COLUMNS),
            layer_sizes=list(run_config.layers),
            n_classes=2,
            activation=model._activation,
            optimizer=run_config.optimizer,
            learning_rate=float(run_config.learning_rate),
            rmsprop_decay=0.99 if run_config.optimizer == "rmsprop" else None,
            rmsprop_eps=1e-8 if run_config.optimizer == "rmsprop" else None,
            viz_mode="tabular",
            viz_note=(
                "Each example has many numeric inputs (scaled); the picture is the network, "
                f"not the spreadsheet. Forward/loss/backward steps follow training row "
                f"{resolved_lesson_anchor} whenever it appears in the minibatch; other batches "
                "still train but only emit optimizer/batch-end in the replay."
            ),
            toy_points=None,
            n_epochs=run_config.epochs,
            total_micro_steps=0,
            batches_per_epoch=[],
            lesson_anchor_train_index=resolved_lesson_anchor,
        )
        lesson_final_manifest = lesson_writer.write(manifest)

    if save_artifacts:
        save_model(model, model_path)
        save_learning_curves(history, curves_dir)
        save_training_history(run_dir, history, elapsed_seconds)
        save_run_config(run_dir, run_config)
    if after_save is not None:
        after_save()
    if (
        telemetry is not None
        and telemetry.callback is not None
        and telemetry.defer_fit_done_callback
    ):
        done_payload: dict = {
            "elapsed_seconds": model.last_fit_seconds,
            "epochs_ran": len(history.train_loss),
            "history": history.model_dump(by_alias=True),
        }
        if lesson_writer is not None and lesson_final_manifest is not None:
            m_dict = lesson_final_manifest.model_dump(mode="json")
            steps_list = [replay_step_to_jsonable(s) for s in lesson_writer.steps]
            try:
                blob = json.dumps({"m": m_dict, "s": steps_list}).encode("utf-8")
                if len(blob) <= INLINE_LESSON_MAX_BYTES:
                    done_payload["lesson_manifest"] = m_dict
                    done_payload["lesson_steps"] = steps_list
                else:
                    done_payload["lesson_replay_run_dir"] = str(run_dir.resolve())
            except (TypeError, ValueError):
                done_payload["lesson_replay_run_dir"] = str(run_dir.resolve())
        telemetry.callback("done", done_payload)
    return history
