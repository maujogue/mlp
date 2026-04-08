"""FastAPI server for run replay and live training SSE."""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import uuid
from typing import Any, cast
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware import Middleware

from mlp.model.compare import BEST_GRID, VISUALIZER_QUICK_GRID, run_best_search
from mlp.model.evaluation import evaluate_model_on_dataset
from mlp.model.schemas import TrainingHistory
from mlp.model.serialization import load_run_config, load_training_history
from mlp.model.telemetry import TrainingTelemetryOptions
from mlp.model.training import run_training
from mlp.utils.loader import build_run_dir

from mlp.data.data_engineering import prepare_dataset_cmd, split_cmd

from .api_schemas import (
    DatasetListResponse,
    EvaluateTestRequest,
    EvaluateTestResponse,
    LiveBestRequest,
    LiveBestResponse,
    LiveTrainRequest,
    LiveTrainResponse,
    PrepareDatasetRequest,
    PrepareDatasetResponse,
    RunDetailResponse,
    RunListResponse,
    SplitDatasetRequest,
    SplitDatasetResponse,
    training_run_config_for_best_search,
    training_run_config_from_live,
)
from .datasets_util import list_csv_under_datasets, resolve_datasets_root, resolve_under_cwd
from .discovery import list_runs, resolve_runs_root


def create_app(*, static_dir: Path | None = None) -> FastAPI:
    app = FastAPI(
        title="MLP Visualizer",
        version="0.1.0",
        middleware=[
            Middleware(
                cast(Any, CORSMiddleware),
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            ),
        ],
    )

    live_sessions: dict[str, queue.Queue[str]] = {}
    live_best_sessions: dict[str, queue.Queue[str]] = {}
    live_lock = threading.Lock()

    @app.get("/api/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/api/datasets", response_model=DatasetListResponse)
    def api_datasets_list(
        root: str = Query(default="datasets", description="Directory to scan for *.csv"),
    ) -> DatasetListResponse:
        d = resolve_datasets_root(root)
        files = list_csv_under_datasets(d)
        return DatasetListResponse(datasets_root=str(d.resolve()), files=files)

    @app.post("/api/datasets/prepare", response_model=PrepareDatasetResponse)
    def api_datasets_prepare(body: PrepareDatasetRequest) -> PrepareDatasetResponse:
        src = resolve_under_cwd(body.source)
        if not src.is_file():
            raise HTTPException(status_code=400, detail="source is not a file")
        if body.output is None:
            out = src.parent / f"{src.stem}_prepared{src.suffix}"
        else:
            out = resolve_under_cwd(body.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        prepare_dataset_cmd(src, out)
        return PrepareDatasetResponse(
            source=str(src.resolve()),
            output=str(out.resolve()),
        )

    @app.post("/api/datasets/split", response_model=SplitDatasetResponse)
    def api_datasets_split(body: SplitDatasetRequest) -> SplitDatasetResponse:
        p = resolve_under_cwd(body.prepared_path)
        if not p.is_file():
            raise HTTPException(status_code=400, detail="prepared_path is not a file")
        split_cmd(p, body.test_size)
        stem = p.stem
        folder = Path("datasets") / stem
        tr = folder / "train.csv"
        te = folder / "test.csv"
        cwd = Path.cwd().resolve()
        return SplitDatasetResponse(
            train_path=str((cwd / tr).resolve()),
            test_path=str((cwd / te).resolve()),
            folder=str((cwd / folder).resolve()),
        )

    @app.get("/api/runs", response_model=RunListResponse)
    def api_runs(
        root: str = Query(default="temp"),
    ) -> RunListResponse:
        runs_root = resolve_runs_root(root)
        runs = list_runs(runs_root)
        return RunListResponse(runs_root=str(runs_root), runs=runs)

    def _safe_run_path(runs_root: Path, rel_path: str) -> Path:
        root_res = runs_root.resolve()
        target = (root_res / rel_path).resolve()
        try:
            target.relative_to(root_res)
        except ValueError as e:
            raise HTTPException(status_code=400, detail="Invalid run path") from e
        if not target.is_dir():
            raise HTTPException(status_code=404, detail="Run folder not found")
        return target

    @app.get("/api/runs/{run_path:path}", response_model=RunDetailResponse)
    def api_run_detail(
        run_path: str,
        root: str = Query(default="temp"),
    ) -> RunDetailResponse:
        runs_root = resolve_runs_root(root)
        run_dir = _safe_run_path(runs_root, run_path)
        history = load_training_history(run_dir)
        rc = None
        cfg_path = run_dir / "run_config.json"
        if cfg_path.exists():
            rc = load_run_config(run_dir)
        return RunDetailResponse(
            run_path=str(run_dir.resolve()),
            history=history,
            run_config=rc,
        )

    _MAX_EVAL_RUNS = 1500

    @app.post("/api/runs/evaluate-test", response_model=EvaluateTestResponse)
    def api_runs_evaluate_test(body: EvaluateTestRequest) -> EvaluateTestResponse:
        if len(body.run_paths) > _MAX_EVAL_RUNS:
            raise HTTPException(
                status_code=400,
                detail=f"At most {_MAX_EVAL_RUNS} runs per request",
            )
        runs_root = resolve_runs_root(body.root)
        test_path = body.test_path
        if not test_path.is_absolute():
            test_path = (Path.cwd() / test_path).resolve()
        else:
            test_path = test_path.resolve()
        if not test_path.is_file():
            raise HTTPException(status_code=400, detail="test_path is not a file")
        results: dict[str, Any] = {}
        for rel in body.run_paths:
            try:
                run_dir = _safe_run_path(runs_root, rel)
                if not (run_dir / "model.pkl").is_file():
                    results[rel] = {"error": "model.pkl missing in run folder"}
                    continue
                m = evaluate_model_on_dataset(run_dir, test_path)
                results[rel] = m.model_dump()
            except Exception as e:  # noqa: BLE001
                results[rel] = {"error": str(e)}
        return EvaluateTestResponse(test_path=str(test_path), results=results)

    @app.get("/api/best-summary")
    def api_best_summary(
        parent: str = Query(..., description="Directory containing best_summary.json"),
    ) -> dict:
        parent_path = resolve_runs_root(parent)
        summary_path = parent_path / "best_summary.json"
        if not summary_path.is_file():
            raise HTTPException(status_code=404, detail="best_summary.json not found")
        with open(summary_path, encoding="utf-8") as f:
            return json.load(f)

    @app.post("/api/live/train", response_model=LiveTrainResponse)
    def api_live_train(body: LiveTrainRequest) -> LiveTrainResponse:
        session_id = str(uuid.uuid4())
        q: queue.Queue[str] = queue.Queue()
        with live_lock:
            live_sessions[session_id] = q

        def emit(event_type: str, payload: dict) -> None:
            msg = json.dumps({"type": event_type, **payload})
            q.put(msg)

        def worker() -> None:
            try:
                run_cfg = training_run_config_from_live(body)
                run_dir = build_run_dir(run_cfg)
                save = body.eval_test_path is not None
                telemetry = TrainingTelemetryOptions(
                    callback=emit,
                    sample_every_n_batches=body.telemetry_sample_every_n_batches,
                    defer_fit_done_callback=save,
                )
                def after_save() -> None:
                    if body.eval_test_path is None:
                        return
                    tp = body.eval_test_path
                    tp = tp.resolve() if tp.is_absolute() else (Path.cwd() / tp).resolve()
                    metrics = evaluate_model_on_dataset(run_dir, tp)
                    emit(
                        "test_eval",
                        {
                            **metrics.model_dump(),
                            "test_path": str(tp),
                            "run_dir": str(run_dir.resolve()),
                        },
                    )

                run_training(
                    run_dir,
                    run_cfg,
                    telemetry=telemetry,
                    save_artifacts=save,
                    after_save=after_save if save else None,
                )
            except Exception as e:  # noqa: BLE001 — surface to client
                emit("error", {"message": str(e)})
            finally:

                def _cleanup_if_unused() -> None:
                    threading.Event().wait(300.0)
                    with live_lock:
                        live_sessions.pop(session_id, None)

                threading.Thread(target=_cleanup_if_unused, daemon=True).start()

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        return LiveTrainResponse(session_id=session_id)

    @app.post("/api/live/best", response_model=LiveBestResponse)
    def api_live_best(body: LiveBestRequest) -> LiveBestResponse:
        session_id = str(uuid.uuid4())
        q: queue.Queue[str] = queue.Queue()
        with live_lock:
            live_best_sessions[session_id] = q

        def emit(event_type: str, payload: dict[str, Any]) -> None:
            q.put(json.dumps({"type": event_type, **payload}))

        def worker() -> None:
            try:
                run_cfg = training_run_config_for_best_search(body)
                grid = BEST_GRID if body.grid_mode == "full" else VISUALIZER_QUICK_GRID
                started_sent = False

                def on_combo(
                    i: int,
                    total: int,
                    run_dir: Path,
                    history: TrainingHistory,
                    snap: dict[str, Any],
                ) -> None:
                    nonlocal started_sent
                    if not started_sent:
                        emit(
                            "grid_started",
                            {
                                "parent_dir": str(run_dir.parent.resolve()),
                                "total": total,
                                "grid_mode": body.grid_mode,
                                "full_grid_combos": len(BEST_GRID),
                            },
                        )
                        started_sent = True
                    emit(
                        "grid_combo_done",
                        {
                            "index": i,
                            "total": total,
                            "run_name": run_dir.name,
                            "leaderboard": snap,
                        },
                    )

                best_run_dir = run_best_search(
                    run_cfg,
                    grid=grid,
                    on_combo_complete=on_combo,
                )
                parent_dir = best_run_dir.parent.resolve()
                with open(parent_dir / "best_summary.json", encoding="utf-8") as f:
                    summary_final = json.load(f)
                emit(
                    "grid_final",
                    {
                        "parent_dir": str(parent_dir),
                        "best_run_dir": str(best_run_dir.resolve()),
                        "summary": summary_final,
                    },
                )
                emit("done", {})
            except Exception as e:  # noqa: BLE001
                emit("error", {"message": str(e)})
            finally:

                def _cleanup_best() -> None:
                    threading.Event().wait(300.0)
                    with live_lock:
                        live_best_sessions.pop(session_id, None)

                threading.Thread(target=_cleanup_best, daemon=True).start()

        threading.Thread(target=worker, daemon=True).start()
        return LiveBestResponse(session_id=session_id)

    @app.get("/api/live/best/stream/{session_id}")
    async def api_live_best_stream(session_id: str) -> StreamingResponse:
        with live_lock:
            q = live_best_sessions.get(session_id)
        if q is None:
            raise HTTPException(status_code=404, detail="Unknown or ended session")

        async def gen_best() -> AsyncIterator[bytes]:
            try:
                while True:
                    raw = await asyncio.to_thread(_queue_get, q, 0.5)
                    if raw is None:
                        yield b": keepalive\n\n"
                        continue
                    yield f"data: {raw}\n\n".encode()
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("type") in ("done", "error"):
                        break
            finally:
                with live_lock:
                    live_best_sessions.pop(session_id, None)

        return StreamingResponse(
            gen_best(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/live/stream/{session_id}")
    async def api_live_stream(session_id: str) -> StreamingResponse:
        with live_lock:
            q = live_sessions.get(session_id)
        if q is None:
            raise HTTPException(status_code=404, detail="Unknown or ended session")

        async def gen() -> AsyncIterator[bytes]:
            try:
                while True:
                    raw = await asyncio.to_thread(_queue_get, q, 0.5)
                    if raw is None:
                        yield b": keepalive\n\n"
                        continue
                    yield f"data: {raw}\n\n".encode()
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("type") in ("done", "error"):
                        break
            finally:
                with live_lock:
                    live_sessions.pop(session_id, None)

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    if static_dir is not None and static_dir.is_dir():
        app.mount(
            "/",
            StaticFiles(directory=str(static_dir), html=True),
            name="static",
        )

    return app


def _queue_get(q: queue.Queue[str], timeout: float) -> str | None:
    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        return None


def run_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    static_dir: Path | None = None,
) -> None:
    import uvicorn

    app = create_app(static_dir=static_dir)
    uvicorn.run(app, host=host, port=port, log_level="info")
