"""One in-memory queue, one worker, injectable local analyzer."""

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path

from contracts.schemas import JobResponse, Violation

Analyzer = Callable[[Path], list[Violation]]
logger = logging.getLogger(__name__)


class JobQueue:
    def __init__(self, analyzer: Analyzer):
        self.analyzer = analyzer
        self.jobs: dict[str, JobResponse] = {}
        self.queue: asyncio.Queue[tuple[str, Path] | None] = asyncio.Queue()

    def submit(self, job_id: str, path: Path) -> None:
        self.jobs[job_id] = JobResponse(job_id=job_id, status="processing")
        self.queue.put_nowait((job_id, path))

    async def run(self) -> None:
        while True:
            item = await self.queue.get()
            try:
                if item is None:
                    return
                job_id, path = item
                try:
                    violations = await asyncio.to_thread(self.analyzer, path)
                    result = JobResponse(
                        job_id=job_id,
                        status="done",
                        verdict="violation" if violations else "no_violation",
                        violations=violations,
                    )
                except Exception:
                    logger.exception("Analyzer failed for job %s", job_id)
                    result = JobResponse(job_id=job_id, status="failed")
                finally:
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        logger.exception("Upload cleanup failed for job %s", job_id)
                        result = JobResponse(job_id=job_id, status="failed")
                self.jobs[job_id] = result
            finally:
                self.queue.task_done()
