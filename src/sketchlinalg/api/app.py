from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect

from sketchlinalg.api.jobs import job_store
from sketchlinalg.api.schemas import (
    BenchmarkJobResponse,
    BenchmarkJobSnapshot,
    BenchmarkOptions,
    BenchmarkRequest,
    DatasetSummary,
    model_to_dict,
)
from sketchlinalg.datasets import PROJECT_ROOT, e2006_summary


app = FastAPI(title="sketchlinalg benchmark API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/dataset-summary", response_model=DatasetSummary)
def dataset_summary() -> dict:
    try:
        return e2006_summary()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/benchmark/options", response_model=BenchmarkOptions)
def benchmark_options() -> BenchmarkOptions:
    return BenchmarkOptions()


@app.post("/api/benchmark/start", response_model=BenchmarkJobResponse)
async def start_benchmark(request: BenchmarkRequest) -> BenchmarkJobResponse:
    job = job_store.start(request)
    return BenchmarkJobResponse(job_id=job.job_id, status=job.status)


@app.get("/api/benchmark/{job_id}", response_model=BenchmarkJobSnapshot)
def benchmark_snapshot(job_id: str) -> BenchmarkJobSnapshot:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown benchmark job")
    return job.snapshot()


@app.websocket("/api/benchmark/{job_id}/progress")
async def benchmark_progress(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    job = job_store.get(job_id)
    if job is None:
        await websocket.send_json(
            {
                "type": "error",
                "job_id": job_id,
                "phase": "error",
                "progress": 1,
                "message": "Unknown benchmark job",
                "error": "Unknown benchmark job",
            }
        )
        await websocket.close()
        return

    history, queue = job_store.subscribe(job)
    try:
        for event in history:
            await websocket.send_json(model_to_dict(event))
        if job.status in {"complete", "error"}:
            await websocket.close()
            return

        while True:
            event = await queue.get()
            await websocket.send_json(model_to_dict(event))
            if event.type in {"complete", "error"}:
                await websocket.close()
                return
    except WebSocketDisconnect:
        return
    finally:
        job_store.unsubscribe(job, queue)


FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

if FRONTEND_DIST.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=FRONTEND_DIST / "assets"),
        name="assets",
    )

    @app.get("/")
    def frontend_index() -> FileResponse:
        return FileResponse(FRONTEND_DIST / "index.html")

    @app.get("/{path:path}")
    def frontend_fallback(path: str) -> FileResponse:
        requested = (FRONTEND_DIST / Path(path)).resolve()
        if requested.is_relative_to(FRONTEND_DIST) and requested.is_file():
            return FileResponse(requested)
        return FileResponse(FRONTEND_DIST / "index.html")
