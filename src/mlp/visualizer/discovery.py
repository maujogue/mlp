"""Discover training runs (folders containing history.json) under a root directory."""

from __future__ import annotations

import json
from pathlib import Path

from mlp.model.serialization import load_run_config

from .api_schemas import RunListItem


def resolve_runs_root(root: str | Path, *, cwd: Path | None = None) -> Path:
    base = cwd or Path.cwd()
    p = Path(root)
    return p if p.is_absolute() else (base / p).resolve()


def find_run_directories(runs_root: Path) -> list[Path]:
    if not runs_root.is_dir():
        return []
    out: list[Path] = []
    for hist in sorted(runs_root.rglob("history.json")):
        run_dir = hist.parent
        out.append(run_dir)
    return out


def _summarize_history_json(history_path: Path) -> dict:
    with open(history_path, encoding="utf-8") as f:
        d = json.load(f)
    train_loss = d.get("history_train_loss") or []
    val_loss = d.get("history_val_loss") or []
    epochs_ran = d.get("epochs_ran")
    if epochs_ran is None:
        epochs_ran = len(train_loss)
    return {
        "epochs_ran": int(epochs_ran) if epochs_ran is not None else 0,
        "elapsed_seconds": d.get("elapsed_seconds"),
        "final_train_loss": float(train_loss[-1]) if train_loss else None,
        "final_val_loss": float(val_loss[-1]) if val_loss else None,
    }


def list_runs(runs_root: Path) -> list[RunListItem]:
    items: list[RunListItem] = []
    root_resolved = runs_root.resolve()
    for run_dir in find_run_directories(runs_root):
        hist = run_dir / "history.json"
        cfg = run_dir / "run_config.json"
        try:
            rel = run_dir.resolve().relative_to(root_resolved)
        except ValueError:
            # Run outside resolved root (symlinks); skip or use as posix relpath
            continue
        rel_posix = rel.as_posix()
        summary = _summarize_history_json(hist)
        try:
            mtime_ms = int(hist.stat().st_mtime * 1000)
        except OSError:
            mtime_ms = None
        config_train_path: str | None = None
        config_layers_str: str | None = None
        config_epochs: int | None = None
        config_learning_rate: float | None = None
        config_seed: int | None = None
        config_batch_size: int | None = None
        config_optimizer: str | None = None
        config_patience: int | None = None
        if cfg.is_file():
            try:
                rc = load_run_config(run_dir)
                config_train_path = str(rc.train_path)
                config_layers_str = "-".join(str(x) for x in rc.layers)
                config_epochs = rc.epochs
                config_learning_rate = rc.learning_rate
                config_seed = rc.seed
                config_batch_size = rc.batch_size
                config_optimizer = rc.optimizer
                config_patience = rc.patience
            except (OSError, ValueError, TypeError):
                pass
        items.append(
            RunListItem(
                id=rel_posix,
                relative_path=rel_posix,
                has_history=True,
                has_run_config=cfg.exists(),
                epochs_ran=summary["epochs_ran"],
                elapsed_seconds=float(summary["elapsed_seconds"])
                if summary["elapsed_seconds"] is not None
                else None,
                final_train_loss=summary["final_train_loss"],
                final_val_loss=summary["final_val_loss"],
                history_mtime_ms=mtime_ms,
                config_train_path=config_train_path,
                config_layers_str=config_layers_str,
                config_epochs=config_epochs,
                config_learning_rate=config_learning_rate,
                config_seed=config_seed,
                config_batch_size=config_batch_size,
                config_optimizer=config_optimizer,
                config_patience=config_patience,
            )
        )
    return items
