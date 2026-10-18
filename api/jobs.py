"""One in-memory queue, one worker, injectable local analyzer."""

import asyncio
import logging
import multiprocessing
from collections.abc import Callable
from multiprocessing.connection import Connection
from pathlib import Path
from time import monotonic
from typing import Generic, TypeVar

from contracts.schemas import JobResponse, Violation

Analyzer = Callable[[Path], list[Violation]]
AnalyzerLoader = Callable[[], Analyzer]
# Inherit uvicorn's configured handler so INFO timings appear in serving logs.
logger = logging.getLogger("uvicorn.error.jobs")
Result = TypeVar("Result")


def _serve_worker(
    operation: Callable[[Path], Result] | None,
    loader: Callable[[], Callable[[Path], Result]] | None,
    connection: Connection,
) -> None:
    """Keep models warm between jobs; spawn avoids inheriting a CUDA context."""
    try:
        try:
            if loader is not None:
                operation = loader()
            if operation is None:
                raise RuntimeError("No local worker operation configured")
        except Exception as exc:
            connection.send((False, str(exc)))
            return
        connection.send((True, None))
        while True:
            path = connection.recv()
            try:
                result = operation(path)
                connection.send((True, result))
            except Exception:
                logger.exception("Worker failed for job %s", path.stem)
                connection.send((False, None))
    except EOFError:
        pass
    finally:
        connection.close()


class LocalProcess(Generic[Result]):
    """A terminable worker for native video/model code; operations must be picklable."""

    def __init__(
        self,
        operation: Callable[[Path], Result] | None,
        loader: Callable[[], Callable[[Path], Result]] | None = None,
    ):
        self.operation = operation
        self.loader = loader
        self.process = None
        self.connection = None
        self.ready = False

    def stop(self) -> None:
        self.ready = False
        if self.process is not None:
            if self.process.is_alive():
                self.process.kill()
            self.process.join()
            self.process.close()
            self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    async def start(self, timeout_sec: float) -> None:
        if self.process is None:
            context = multiprocessing.get_context("spawn")
            self.connection, child = context.Pipe()
            process = context.Process(target=_serve_worker, args=(self.operation, self.loader, child))
            try:
                process.start()
                self.process = process
            except Exception:
                self.connection.close()
                self.connection = None
                process.close()
                raise
            finally:
                child.close()
            succeeded, error = await self._reply(self.connection.recv, timeout_sec)
            if not succeeded:
                await asyncio.to_thread(self.stop)
                raise RuntimeError(error)
            self.ready = True

    async def run(self, path: Path, timeout_sec: float, *, load_timeout_sec: float = 180) -> Result:
        # A cold model reload must not consume the following video's analysis budget.
        await self.start(load_timeout_sec)
        connection = self.connection

        def exchange():
            connection.send(path)
            return connection.recv()

        succeeded, result = await self._reply(exchange, timeout_sec)
        if not succeeded:
            raise RuntimeError("Worker failed")
        return result

    async def _reply(self, receive: Callable, timeout_sec: float):
        reply = asyncio.create_task(asyncio.to_thread(receive))
        try:
            return await asyncio.wait_for(asyncio.shield(reply), timeout=timeout_sec)
        except BaseException:
            # Stop native code before deleting its input or starting another GPU job.
            await asyncio.to_thread(self.stop)
            await asyncio.gather(reply, return_exceptions=True)
            raise


class JobQueue:
    def __init__(
        self,
        analyzer: Analyzer | None,
        timeout_sec: float = 55,
        loader: AnalyzerLoader | None = None,
        load_timeout_sec: float = 180,
    ):
        self.worker = LocalProcess(analyzer, loader)
        self.timeout_sec = timeout_sec
        self.load_timeout_sec = load_timeout_sec
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
                started = monotonic()
                try:
                    violations = await self.worker.run(
                        path, self.timeout_sec, load_timeout_sec=self.load_timeout_sec
                    )
                    result = JobResponse(
                        job_id=job_id,
                        status="done",
                        verdict="violation" if violations else "no_violation",
                        violations=violations,
                    )
                except TimeoutError:
                    logger.warning("Analyzer timed out for job %s", job_id)
                    result = JobResponse(job_id=job_id, status="failed")
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
                logger.info(
                    "Job %s status=%s duration_sec=%.3f", job_id, result.status, monotonic() - started
                )
                if not self.worker.ready:
                    # Publish the failed job first, then warm its replacement even if idle.
                    try:
                        await self.worker.start(self.load_timeout_sec)
                    except Exception:
                        logger.exception("Worker reload failed after job %s", job_id)
            finally:
                self.queue.task_done()
