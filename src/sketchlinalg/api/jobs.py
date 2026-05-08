from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import time
from uuid import uuid4

from sketchlinalg.api.benchmark_runner import run_count_sketch_benchmark
from sketchlinalg.api.schemas import (
    BenchmarkJobSnapshot,
    BenchmarkProgressEvent,
    BenchmarkRequest,
    BenchmarkResult,
)


@dataclass
class BenchmarkJob:
    job_id: str
    request: BenchmarkRequest
    status: str = "queued"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    result: BenchmarkResult | None = None
    error: str | None = None
    events: list[BenchmarkProgressEvent] = field(default_factory=list)
    subscribers: set[asyncio.Queue] = field(default_factory=set)

    def publish(self, event: BenchmarkProgressEvent) -> None:
        self.events.append(event)
        for queue in list(self.subscribers):
            queue.put_nowait(event)

    def snapshot(self) -> BenchmarkJobSnapshot:
        return BenchmarkJobSnapshot(
            job_id=self.job_id,
            status=self.status,
            result=self.result,
            error=self.error,
        )


class BenchmarkJobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, BenchmarkJob] = {}

    def get(self, job_id: str) -> BenchmarkJob | None:
        return self._jobs.get(job_id)

    def start(self, request: BenchmarkRequest) -> BenchmarkJob:
        loop = asyncio.get_running_loop()
        job = BenchmarkJob(job_id=uuid4().hex, request=request)
        self._jobs[job.job_id] = job
        job.publish(
            BenchmarkProgressEvent(
                type="progress",
                job_id=job.job_id,
                phase="queued",
                progress=0,
                message="Benchmark queued",
            )
        )
        asyncio.create_task(self._run(job, loop))
        return job

    async def _run(self, job: BenchmarkJob, loop: asyncio.AbstractEventLoop) -> None:
        job.status = "running"
        started_at = time.perf_counter()

        def publish_from_thread(event: BenchmarkProgressEvent) -> None:
            loop.call_soon_threadsafe(job.publish, event)

        try:
            result = await asyncio.to_thread(
                run_count_sketch_benchmark,
                job.job_id,
                job.request,
                publish_from_thread,
            )
        except Exception as exc:  # pragma: no cover - surfaced through the API.
            job.status = "error"
            job.error = str(exc)
            job.publish(
                BenchmarkProgressEvent(
                    type="error",
                    job_id=job.job_id,
                    phase="error",
                    progress=1,
                    message="Benchmark failed",
                    error=str(exc),
                )
            )
            return

        job.status = "complete"
        job.result = result
        job.publish(
            BenchmarkProgressEvent(
                type="complete",
                job_id=job.job_id,
                phase="complete",
                progress=1,
                elapsed_seconds=float(time.perf_counter() - started_at),
                message="Benchmark complete",
                result=result,
            )
        )

    def subscribe(
            self, job: BenchmarkJob
    ) -> tuple[list[BenchmarkProgressEvent], asyncio.Queue]:
        queue: asyncio.Queue = asyncio.Queue()
        job.subscribers.add(queue)
        return list(job.events), queue

    def unsubscribe(self, job: BenchmarkJob, queue: asyncio.Queue) -> None:
        job.subscribers.discard(queue)


job_store = BenchmarkJobStore()
