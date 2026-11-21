"""Common request metadata and event creation for provider observations."""

import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
import re
import time
from typing import Any
from uuid import uuid4

from ._telemetry import count, field, read_usage, text

Record = Callable[[dict[str, Any]], None]
_ERROR_CODE = re.compile(r"[A-Za-z0-9_.-]+", re.ASCII)


class Observation:
    request_id_header: str | None = None

    def __init__(self, endpoint: str, model: str, context: dict[str, Any], record: Record, *, provider: str) -> None:
        self.endpoint = endpoint
        self._provider = provider
        self._model = model
        self._context = context
        self._record = record
        self._started = time.monotonic()
        self._occurred_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        self._usage: Any = None
        self._request_id: str | None = None
        self._service_tier: str | None = None
        self.done = False

    def _metadata(self, result: Any) -> None:
        self._model = text(field(result, "model"), 256) or self._model
        self._request_id = self._request_id or text(field(result, "_request_id"), 256)
        self._service_tier = text(field(result, "service_tier"), 64) or self._service_tier
        usage = field(result, "usage")
        if usage is not None:
            self._usage = usage

    def result(self, value: Any) -> None:
        if not self.done:
            self._final_result(value)

    def _final_result(self, value: Any) -> None:
        self._metadata(value)
        status = field(value, "status")
        if status == "cancelled":
            self._emit("cancelled", use_usage=True)
        elif status in ("failed", "incomplete"):
            self._emit("error", field(value, "error"), use_usage=True)
        else:
            self._emit("success", use_usage=True)

    def _read_request_id(self, value: Any) -> None:
        headers = field(value, "headers")
        if self.request_id_header and headers is not None and not self._request_id:
            self._request_id = text(headers.get(self.request_id_header), 256)

    def stream_headers(self, stream: Any) -> None:
        if self.request_id_header:
            self._read_request_id(field(stream, "response"))
            self._read_request_id(field(field(stream, "_raw_stream"), "response"))

    def raw(self, value: Any) -> None:
        self._read_request_id(value)
        self.finish("success")

    def chunk(self, value: Any) -> None:
        if not self.done:
            self._metadata(value)

    def completed(self) -> None:
        self.finish("error")

    def failed(self, error: BaseException) -> None:
        cancelled = isinstance(error, (asyncio.CancelledError, GeneratorExit)) or type(error).__name__ in (
            "APICancelledError", "APIUserAbortError", "APIRequestAbortedError",
        )
        self.finish("cancelled" if cancelled else "error", error)

    def finish(self, status: str, reason: Any = None, use_usage: bool = False) -> None:
        self._emit(status, reason, use_usage)

    def _event_id(self) -> str:
        return str(uuid4())

    def _http_status(self, reason: Any) -> int | None:
        return count(field(reason, "status_code") or field(reason, "status"))

    def _billing(self) -> dict[str, str]:
        return {"serviceTier": self._service_tier} if self._service_tier else {}

    def _normalized_usage(self) -> dict[str, Any] | None:
        return read_usage(self.endpoint, self._usage)

    def _emit(self, status: str, reason: Any = None, use_usage: bool = False) -> None:
        if self.done:
            return
        self.done = True
        code = text(field(reason, "code") or field(reason, "type"), 128)
        http_status = self._http_status(reason)
        request_id = self._request_id or text(field(reason, "request_id"), 256)
        event = {
            **self._context,
            "id": self._event_id(),
            "type": "llm.request",
            "provider": self._provider,
            "model": self._model,
            "occurredAt": self._occurred_at,
            "durationMs": round(max(0, time.monotonic() - self._started) * 1000),
            "status": status,
            "usage": self._normalized_usage() if use_usage else None,
        }
        if request_id:
            event["providerRequestId"] = request_id
        if billing := self._billing():
            event["billing"] = billing
        error: dict[str, Any] = {}
        if code and _ERROR_CODE.fullmatch(code):
            error["code"] = code
        if http_status is not None and 100 <= http_status <= 599:
            error["httpStatus"] = http_status
        if error:
            event["error"] = error
        self._record(event)
