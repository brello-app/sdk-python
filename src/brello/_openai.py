"""Interpret OpenAI responses and correlate background requests across calls."""

from collections import OrderedDict
import logging
from threading import Lock
import time
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from ._observation import Observation, Record
from ._telemetry import field, text

_PENDING_STATUSES = ("queued", "in_progress")
_PENDING_LIMIT = 4096
_PENDING_TTL = 10 * 60
_COMPLETED_LIMIT = 1024
_LOG = logging.getLogger("brello")


class BackgroundResponses:
    def __init__(self) -> None:
        self._pending: OrderedDict[str, tuple[float, OpenAIObservation]] = OrderedDict()
        self._completed: OrderedDict[str, None] = OrderedDict()
        self._lock = Lock()

    def _expire(self, now: float) -> int:
        expired = 0
        while self._pending and next(iter(self._pending.values()))[0] <= now - _PENDING_TTL:
            self._pending.popitem(last=False)
            expired += 1
        return expired

    def is_pending(self, response_id: str) -> bool:
        with self._lock:
            expired = self._expire(time.monotonic())
            pending = response_id in self._pending
        if expired:
            _LOG.debug("Expired %d pending OpenAI responses", expired)
        return pending

    def update(self, response_id: str, current: "OpenAIObservation", value: Any) -> None:
        evicted = False
        with self._lock:
            now = time.monotonic()
            expired = self._expire(now)
            if response_id in self._completed:
                current.done = True
                self._completed.move_to_end(response_id)
            else:
                tracked = self._pending.get(response_id)
                original = tracked[1] if tracked is not None else current
                if field(value, "status") in _PENDING_STATUSES:
                    self._pending[response_id] = (now, original)
                    self._pending.move_to_end(response_id)
                    original._metadata(value)
                    if len(self._pending) > _PENDING_LIMIT:
                        self._pending.popitem(last=False)
                        evicted = True
                else:
                    original._request_id = original._request_id or current._request_id
                    original._final_result(value)
                    self._pending.pop(response_id, None)
                    self._completed[response_id] = None
                    if len(self._completed) > _COMPLETED_LIMIT:
                        self._completed.popitem(last=False)
                    current.done = True
        if expired:
            _LOG.debug("Expired %d pending OpenAI responses", expired)
        if evicted:
            _LOG.debug("Evicted a pending OpenAI response at the tracking limit")


class OpenAIObservation(Observation):
    request_id_header = "x-request-id"

    def __init__(self, endpoint: str, model: str, context: dict[str, Any], record: Record,
                 *, background: BackgroundResponses | None, response_id: str | None) -> None:
        super().__init__(endpoint, model, context, record, provider="openai")
        self._background = background
        self._response_id = response_id

    def result(self, value: Any) -> None:
        if self.done:
            return
        if self._background is not None:
            self._response_id = text(field(value, "id"), 256) or self._response_id
            if self._response_id:
                self._background.update(self._response_id, self, value)
                return
        self._final_result(value)

    def chunk(self, value: Any) -> None:
        if self.done:
            return
        kind = field(value, "type")
        if self.endpoint != "responses":
            self._metadata(field(value, "chunk") if kind == "chunk" else value)
        elif kind in ("response.created", "response.queued", "response.in_progress"):
            response = field(value, "response")
            if self._background is not None:
                self.result(response)
            else:
                self._metadata(response)
        elif kind in ("response.completed", "response.incomplete", "response.failed", "response.cancelled"):
            self.result(field(value, "response"))
        elif kind == "error":
            self.finish("error", value)

    def completed(self) -> None:
        if self.endpoint == "chat":
            self.finish("success", use_usage=True)
        else:
            super().completed()

    def finish(self, status: str, reason: Any = None, use_usage: bool = False) -> None:
        # Ending a connection or a poll does not end a known background response.
        if self._background is None or self._response_id is None:
            super().finish(status, reason, use_usage)

    def _event_id(self) -> str:
        if self._response_id:
            return str(uuid5(NAMESPACE_URL, f"https://api.openai.com/v1/responses/{self._response_id}"))
        return super()._event_id()
