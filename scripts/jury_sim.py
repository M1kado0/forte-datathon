"""Jury simulation: `uv run python -m scripts.jury_sim --url $PUBLIC_URL`.

Phase 1 mimics the jury: N parallel uploads of real clips, p95 of time-to-result, schema check of every
answer against contracts/. Phase 2 is the stress test: M parallel ~145 s videos built locally with ffmpeg
(needs the `ffmpeg` binary) from our clips, all of them must finish inside the budget. Exit code is 1 when
any criterion fails.
"""

import argparse
import asyncio
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from dotenv import load_dotenv
from pydantic import ValidationError

from contracts.schemas import JobResponse
from src.config import ROOT, VIDEOS_DIR

P95_LIMIT_SEC = 240.0  # brief: result within 5 min per video, keep a margin
STRESS_LIMIT_SEC = 300.0
CLIP_LIMIT_SEC = 45.0  # docs/PLAN.md T10: one 145-s clip
POLL_SEC = 1.0


@dataclass
class Outcome:
    clip: str
    seconds: float
    ok: bool
    problem: str = ""


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile; the max for small samples, which is what a strict jury would see."""
    if not values:
        raise ValueError("percentile of an empty sample")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def schema_problem(payload: dict) -> str:
    """Empty string when the answer is a valid, finished job response; otherwise a short reason."""
    try:
        job = JobResponse.model_validate(payload)
    except ValidationError as error:
        return f"schema: {error.errors()[0]['msg']}"
    return "" if job.status == "done" else f"status {job.status}"


def pick_clips(directory: Path, count: int) -> list[Path]:
    clips = sorted(directory.glob("*.mov"))
    if not clips:
        raise SystemExit(f"Нет .mov в {directory}")
    return [clips[i % len(clips)] for i in range(count)]


def make_long_clip(source: Path, target: Path, seconds: int) -> None:
    """Loop a local clip to `seconds` without re-encoding."""
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-stream_loop",
            "-1",
            "-i",
            str(source),
            "-t",
            str(seconds),
            "-c",
            "copy",
            str(target),
        ],
        check=True,
    )


async def run_one(client: httpx.AsyncClient, clip: Path, timeout: float) -> Outcome:
    started = time.monotonic()

    def outcome(ok: bool, problem: str = "") -> Outcome:
        return Outcome(clip.name, time.monotonic() - started, ok, problem)

    try:
        with clip.open("rb") as handle:
            sent = await client.post(
                "/api/v1/analyze", files={"file": (clip.name, handle, "video/quicktime")}
            )
        sent.raise_for_status()
        job_id = sent.json()["job_id"]
        while time.monotonic() - started < timeout:
            polled = await client.get(f"/api/v1/jobs/{job_id}")
            polled.raise_for_status()
            payload = polled.json()
            if payload.get("status") != "processing":
                problem = schema_problem(payload)
                return outcome(not problem, problem)
            await asyncio.sleep(POLL_SEC)
        return outcome(False, f"timeout after {timeout:.0f} s")
    except (httpx.HTTPError, KeyError, ValueError) as error:
        return outcome(False, f"{type(error).__name__}: {error}")


async def run_batch(
    client: httpx.AsyncClient, clips: list[Path], timeout: float
) -> tuple[list[Outcome], float]:
    started = time.monotonic()
    outcomes = await asyncio.gather(*(run_one(client, clip, timeout) for clip in clips))
    return list(outcomes), time.monotonic() - started


async def check_access(client: httpx.AsyncClient) -> list[str]:
    errors = []
    health = await client.get("/api/health")
    try:
        payload = health.json()
    except ValueError:  # an HTML page from the tunnel instead of JSON
        payload = None
    health_ok = health.status_code == 200 and isinstance(payload, dict) and payload.get("status") == "ok"
    if not health_ok:
        errors.append(f"/api/health вернул {health.status_code} и не JSON со status ok")
    elif payload.get("model_loaded") is not True:
        errors.append("/api/health: требуется model_loaded: true; реальная модель не готова")
    async with httpx.AsyncClient(base_url=str(client.base_url), timeout=30) as anonymous:
        unauthorized = await anonymous.get("/api/v1/jobs/none")
    if unauthorized.status_code != 401:
        errors.append(f"без ключа ожидался 401, получен {unauthorized.status_code}")
    return errors


def failed(outcomes: list[Outcome]) -> list[str]:
    return [f"{item.clip}: {item.problem}" for item in outcomes if not item.ok]


def report(title: str, outcomes: list[Outcome]) -> None:
    for item in outcomes:
        print(f"  {'ok  ' if item.ok else 'FAIL'} {item.seconds:7.1f} s  {item.clip}  {item.problem}")
    print(f"{title}: ошибок {sum(not item.ok for item in outcomes)} из {len(outcomes)}")


async def main(args: argparse.Namespace) -> int:
    load_dotenv(ROOT / ".env")
    key = args.api_key or os.getenv("API_KEY", "")
    if not key:
        raise SystemExit("Нет ключа: --api-key или API_KEY в .env")
    if args.stress and not shutil.which("ffmpeg"):
        raise SystemExit("Для стресса нужен ffmpeg (sudo apt install ffmpeg); без него: --stress 0")
    url = args.url.rstrip("/")
    failures: list[str] = []

    async with httpx.AsyncClient(
        base_url=url, headers={"X-API-Key": key}, timeout=httpx.Timeout(120.0)
    ) as client:
        try:
            failures += await check_access(client)
        except httpx.HTTPError as error:
            raise SystemExit(f"Сервис недоступен по {url}: {type(error).__name__}: {error}") from None

        if failures:
            print("НЕ ПРОШЛО:\n  " + "\n  ".join(failures))
            return 1

        print(f"Жюри: {args.parallel} параллельных задач")
        jury, jury_wall = await run_batch(client, pick_clips(args.videos, args.parallel), args.timeout)
        report("Жюри", jury)
        p95 = percentile([item.seconds for item in jury], 0.95)
        schema_errors = sum(item.problem.startswith("schema") for item in jury)
        print(
            f"p95 = {p95:.1f} s (порог {P95_LIMIT_SEC:.0f}; при {len(jury)} задачах p95 = максимум), "
            f"вся пачка за {jury_wall:.1f} s, ошибок схемы = {schema_errors}"
        )
        if p95 >= P95_LIMIT_SEC:
            failures.append(f"p95 {p95:.1f} s >= {P95_LIMIT_SEC:.0f} s")
        failures += failed(jury)

        if args.stress:
            print(f"Стресс: {args.stress} параллельных видео по {args.stress_seconds} с")
            with tempfile.TemporaryDirectory(prefix="forte-stress-") as directory:
                longs = []
                try:
                    for index, source in enumerate(pick_clips(args.videos, args.stress)):
                        target = Path(directory) / f"stress-{index}.mov"
                        make_long_clip(source, target, args.stress_seconds)
                        longs.append(target)
                except subprocess.CalledProcessError as error:
                    raise SystemExit(f"ffmpeg не собрал длинный клип (код {error.returncode})") from None
                stress, wall = await run_batch(client, longs, STRESS_LIMIT_SEC + 60)
            report("Стресс", stress)
            slowest = max(item.seconds for item in stress)
            print(
                f"время стресса = {wall:.1f} s (порог {STRESS_LIMIT_SEC:.0f}), "
                f"самый долгий клип = {slowest:.1f} s (порог {CLIP_LIMIT_SEC:.0f})"
            )
            if wall >= STRESS_LIMIT_SEC:
                failures.append(f"стресс {wall:.1f} s >= {STRESS_LIMIT_SEC:.0f} s")
            if slowest >= CLIP_LIMIT_SEC:
                failures.append(
                    f"клип {args.stress_seconds} с занял {slowest:.1f} s >= {CLIP_LIMIT_SEC:.0f} s"
                )
            failures += failed(stress)

    print("ПРОШЛО" if not failures else "НЕ ПРОШЛО:\n  " + "\n  ".join(failures))
    return 1 if failures else 0


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True, help="публичный адрес сервиса, например $PUBLIC_URL")
    parser.add_argument("--api-key", help="по умолчанию API_KEY из .env")
    parser.add_argument("--videos", type=Path, default=VIDEOS_DIR)
    parser.add_argument("--parallel", type=int, default=5)
    parser.add_argument("--stress", type=int, default=8, help="0 отключает стресс")
    parser.add_argument("--stress-seconds", type=int, default=145)
    parser.add_argument("--timeout", type=float, default=300.0, help="потолок ожидания одной задачи, с")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse())))
