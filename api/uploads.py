"""Bound upload bodies and verify locally that a MOV contains decodable video."""

from pathlib import Path

import cv2
from starlette.datastructures import Headers
from starlette.formparsers import MultiPartException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class UploadLimitMiddleware:
    """Count actual bytes before the multipart parser can spool an unlimited body."""

    def __init__(self, app: ASGIApp, max_body_bytes: int):
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"].rstrip("/") not in (
            "/api/v1/analyze",
            "/api/ui/analyze",
        ):
            await self.app(scope, receive, send)
            return
        error = JSONResponse({"detail": "Превышен допустимый размер загрузки"}, status_code=413)
        length = Headers(scope=scope).get("content-length", "")
        if length.isdecimal() and int(length) > self.max_body_bytes:
            await error(scope, receive, send)
            return
        received = 0
        exceeded = False

        async def limited_receive() -> Message:
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    exceeded = True
                    # Starlette closes partially spooled files for this exception.
                    raise MultiPartException("Превышен допустимый размер загрузки")
            return message

        async def limited_send(message: Message) -> None:
            if exceeded:
                # The multipart parser maps its exception to 400; retain our 413.
                if message["type"] == "http.response.start":
                    await error(scope, receive, send)
            else:
                await send(message)

        await self.app(scope, limited_receive, limited_send)


def is_video(path: Path) -> bool:
    # Restrict the backend: a renamed image must not pass as a one-frame video.
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    try:
        if not capture.isOpened() or capture.get(cv2.CAP_PROP_FPS) <= 0:
            return False
        ok, frame = capture.read()
        return bool(ok and frame is not None and frame.size)
    except cv2.error:
        return False
    finally:
        capture.release()
