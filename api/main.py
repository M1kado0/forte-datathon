"""Video API and same-origin web app. Run one uvicorn process for the in-memory queue."""

import asyncio
import os
import secrets
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from api.analyzer import analyze_stub, load_analyzer
from api.jobs import Analyzer, JobQueue
from contracts.schemas import AnalyzeResponse, Health, JobResponse, RuleType
from src.config import ROOT, VERSION, WEB_DIST
from src.rules.describe import rule_catalog


def create_app(*, analyzer: Analyzer | None = None, api_key: str | None = None) -> FastAPI:
    load_dotenv(ROOT / ".env")
    key = api_key if api_key is not None else os.getenv("API_KEY", "")
    if analyzer is None:
        mode = os.getenv("FORTE_ANALYZER", "real")
        if mode not in {"real", "stub"}:
            raise ValueError("FORTE_ANALYZER must be 'real' or 'stub'")
        if mode == "stub":
            analyzer = analyze_stub

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model_loaded = False
        active_analyzer = analyzer
        if active_analyzer is None:
            active_analyzer = await run_in_threadpool(load_analyzer)
            app.state.model_loaded = True
        with TemporaryDirectory(prefix="forte-upload-") as directory:
            app.state.upload_dir = Path(directory)
            app.state.jobs = JobQueue(active_analyzer)
            worker = asyncio.create_task(app.state.jobs.run())
            try:
                yield
            finally:
                app.state.jobs.queue.put_nowait(None)
                await worker
                app.state.model_loaded = False

    app = FastAPI(title="Forte: проверка кассовой дисциплины", version=VERSION, lifespan=lifespan)

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        if request.url.path == "/api/v1" or request.url.path.startswith("/api/v1/"):
            supplied = request.headers.get("X-API-Key", "")
            if not key or not secrets.compare_digest(supplied.encode(), key.encode()):
                return JSONResponse(status_code=401, content={"detail": "Неверный API-ключ"})
        return await call_next(request)

    @app.get("/api/health", response_model=Health)
    def health():
        return Health(version=VERSION, model_loaded=app.state.model_loaded)

    router = APIRouter()

    @router.post("/analyze", response_model=AnalyzeResponse)
    async def analyze(request: Request, file: UploadFile):
        path = None
        submitted = False
        try:
            if Path(file.filename or "").suffix.lower() != ".mov":
                raise HTTPException(422, "Загрузите видео в формате .mov")
            job_id = uuid4().hex
            path = request.app.state.upload_dir / f"{job_id}.mov"
            with path.open("wb") as target:
                await run_in_threadpool(shutil.copyfileobj, file.file, target)
            request.app.state.jobs.submit(job_id, path)
            submitted = True
            return AnalyzeResponse(job_id=job_id)
        finally:
            await file.close()
            if path is not None and not submitted:
                path.unlink(missing_ok=True)

    @router.get("/jobs/{job_id}", response_model=JobResponse)
    def job(request: Request, job_id: str):
        result = request.app.state.jobs.jobs.get(job_id)
        if result is None:
            raise HTTPException(404, "Задача не найдена")
        return result

    app.include_router(router, prefix="/api/v1")
    app.include_router(router, prefix="/api/ui")

    @app.get("/api/ui/rules", response_model=list[RuleType])
    def rules():
        """Display metadata only; the jury's job response stays unchanged."""
        return rule_catalog()

    # API misses must never become a successful SPA response.
    @app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    def missing_api(path: str):
        raise HTTPException(404, "Маршрут не найден")

    if (WEB_DIST / "index.html").exists():
        if (WEB_DIST / "assets").is_dir():
            app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            candidate = (WEB_DIST / path).resolve()
            if not candidate.is_relative_to(WEB_DIST.resolve()):
                raise HTTPException(404)
            return FileResponse(candidate if path and candidate.is_file() else WEB_DIST / "index.html")

    return app


app = create_app()
