"""Bounded, single-worker telemetry delivery shared by sync and async clients."""

import base64
from collections import Counter, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
import hashlib
import hmac
import http.client
import json
import logging
import random
import threading
import time
from typing import Any
from urllib.parse import urlsplit

_BATCH_SIZE = 20
_FLUSH_INTERVAL = 2.0
_MAX_EVENTS = 500
_MAX_BUFFER_BYTES = 2 * 1024 * 1024
_MAX_EVENT_BYTES = 64_000
_MAX_BATCH_BYTES = 256_000
_MAX_RESPONSE_BYTES = 64_000
_MAX_AGE = 60.0
_MAX_ATTEMPTS = 5
_REQUEST_TIMEOUT = 3.0
_LOG = logging.getLogger("brello")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8", "backslashreplace")


@dataclass
class Entry:
    event: dict[str, Any]
    size: int
    queued: float
    failures: int = 0


@dataclass(frozen=True)
class Reply:
    status: int
    retry_after: str | None = None
    data: Any = None


def _request(base_url: str, key: str, path: str, body: bytes | None, debug: bool) -> Reply:
    url = urlsplit(base_url)
    connection_type = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(url.hostname, url.port, timeout=_REQUEST_TIMEOUT)
    headers = {"Authorization": "Bearer " + key}
    if body is not None:
        headers["Content-Type"] = "application/json"
    try:
        connection.request("GET" if body is None else "POST", url.path + path, body=body, headers=headers)
        with connection.getresponse() as response:
            data = None
            if (path == "config" and 200 <= response.status < 300) or (debug and response.status in (202, 422)):
                deadline = time.monotonic() + _REQUEST_TIMEOUT
                raw = bytearray()
                try:
                    while time.monotonic() < deadline and len(raw) <= _MAX_RESPONSE_BYTES:
                        chunk = response.read1(min(8192, _MAX_RESPONSE_BYTES + 1 - len(raw)))
                        if not chunk:
                            data = json.loads(raw)
                            break
                        raw.extend(chunk)
                except (OSError, http.client.HTTPException, ValueError, RecursionError):
                    # A receipt body failure cannot change the submission status.
                    pass
            return Reply(response.status, response.getheader("Retry-After"), data)
    finally:
        connection.close()


def _object(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _retry_after(value: str | None) -> float:
    if not value:
        return 0.0
    value = value.strip()
    try:
        if value.isascii() and value.isdigit():
            return float(value)
        return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
    except (ValueError, OverflowError, TypeError, OSError):
        return 0.0


class Transport:
    def __init__(
        self, key: str, base_url: str, *, debug: bool, debug_include_context: bool,
        hash_end_user_id: bool, version: str,
        request: Callable[[str, str, str, bytes | None, bool], Reply] = _request,
    ) -> None:
        self._key = key
        self._base_url = base_url
        self._debug = debug
        self._debug_include_context = debug_include_context
        self._hash_end_user_id = hash_end_user_id
        self._version = version
        self._request = request
        self._condition = threading.Condition()
        self._pending: deque[Entry] = deque()
        self._bytes = 0
        self._in_flight = 0
        self._worker: threading.Thread | None = None
        self._flush_deadlines: Counter[float] = Counter()
        self._disabled = False
        self._retry_at = 0.0
        self._circuit_until = 0.0
        self._circuit_delay = 0.0
        self._failures = 0
        self._next_config_at = 0.0
        self._events_enabled: bool | None = None
        self._collection: Mapping[str, Any] = {}
        self._identity: Mapping[str, Any] = {}
        self._drops: Counter[str] = Counter()
        self._next_drop_log = 0.0

    def enqueue(self, event: dict[str, Any]) -> None:
        size = len(_json_bytes(event))
        with self._condition:
            if self._disabled:
                return
            self._expire()
            if size > _MAX_EVENT_BYTES:
                self._drops["oversized"] += 1
                return
            while self._pending and self._full(size):
                self._bytes -= self._pending.popleft().size
                self._drops["buffer_full"] += 1
            if self._full(size):
                self._drops["buffer_full"] += 1
                return
            self._pending.append(Entry(event, size, time.monotonic()))
            self._bytes += size
            if self._worker is None:
                worker = threading.Thread(target=self._run, name="brello-delivery", daemon=True)
                worker.start()
                self._worker = worker
            self._condition.notify_all()

    def flush(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        with self._condition:
            self._flush_deadlines[deadline] += 1
            self._condition.notify_all()
            try:
                while not self._disabled and (self._pending or self._in_flight):
                    self._expire()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or (not self._in_flight and self._next_send() >= deadline):
                        break
                    if not self._pending and not self._in_flight:
                        break
                    self._condition.wait(remaining)
            finally:
                self._flush_deadlines[deadline] -= 1
                if not self._flush_deadlines[deadline]:
                    del self._flush_deadlines[deadline]
                self._condition.notify_all()

    def _full(self, size: int) -> bool:
        return len(self._pending) + self._in_flight >= _MAX_EVENTS or self._bytes + size > _MAX_BUFFER_BYTES

    def _next_send(self) -> float:
        return max(self._retry_at, self._circuit_until)

    def _expire(self) -> None:
        cutoff = time.monotonic() - _MAX_AGE
        while self._pending and self._pending[0].queued <= cutoff:
            self._bytes -= self._pending.popleft().size
            self._drops["expired"] += 1

    def _take_batch(self) -> list[Entry]:
        with self._condition:
            while not self._disabled:
                self._expire()
                if not self._pending:
                    self._worker = None
                    return []
                first = self._pending[0].queued
                now = time.monotonic()
                immediate = len(self._pending) >= _BATCH_SIZE or max(self._flush_deadlines, default=0) > now
                due = max(self._next_send(), now if immediate else first + _FLUSH_INTERVAL)
                if due <= now:
                    batch = [self._pending.popleft() for _ in range(min(len(self._pending), _BATCH_SIZE))]
                    self._in_flight = len(batch)
                    return batch
                self._condition.wait(min(due, first + _MAX_AGE) - now)
            self._worker = None
            return []

    def _run(self) -> None:
        while batch := self._take_batch():
            retry: list[Entry] = []
            try:
                self._log_drops()
                if not self._circuit_until:
                    self._refresh_config()
                if not self._disabled and self._events_enabled is not False:
                    retry = self._send(batch)
            except Exception:
                self._log(logging.DEBUG, "Telemetry delivery failed")
            finally:
                with self._condition:
                    self._in_flight = 0
                    self._bytes -= sum(entry.size for entry in batch)
                    if not self._disabled:
                        self._pending.extendleft(reversed(retry))
                        self._bytes += sum(entry.size for entry in retry)
                    self._expire()
                    self._condition.notify_all()

    def _refresh_config(self) -> None:
        now = time.monotonic()
        if now < self._next_config_at:
            return
        self._collection = {}
        self._identity = {}
        self._next_config_at = now + 60
        try:
            reply = self._request(self._base_url, self._key, "config", None, self._debug)
        except Exception:
            self._log(logging.DEBUG, "SDK configuration unavailable")
            return
        if reply.status == 401:
            self._disable()
        elif 200 <= reply.status < 300:
            data = _object(_object(reply.data).get("data"))
            self._collection = _object(data.get("collection"))
            self._identity = _object(_object(data.get("identity")).get("endUser"))
            events_enabled = self._collection.get("eventsEnabled")
            if isinstance(events_enabled, bool):
                self._events_enabled = events_enabled
            interval = data.get("refreshAfterSeconds")
            if type(interval) in (int, float) and 0 <= interval <= 86_400:
                self._next_config_at = time.monotonic() + max(1, interval)

    def _event_payload(self, event: dict[str, Any]) -> dict[str, Any]:
        payload = event.copy()
        end_user = payload.pop("endUser", None)
        attributes = payload.pop("attributes", None)
        collection = self._collection if time.monotonic() < self._next_config_at else {}
        if collection.get("attributesEnabled") is True and attributes:
            payload["attributes"] = attributes
        if collection.get("endUserIdEnabled") is not True or not end_user:
            return payload
        if not self._hash_end_user_id:
            payload["endUser"] = end_user
            return payload
        identity = self._identity
        secret, domain = identity.get("secret"), identity.get("domain")
        identifier = end_user.get("id")
        if (
            identifier and isinstance(secret, str) and len(secret) == 43 and isinstance(domain, str)
            and identity.get("algorithm") == "hmac-sha256"
            and identity.get("secretEncoding") == "base64url" and identity.get("keyVersion") == "1"
        ):
            try:
                key = base64.b64decode(secret + "=", altchars=b"-_", validate=True)
                digest = hmac.new(key, (domain + "\0" + identifier).encode(), hashlib.sha256).hexdigest()
            except (ValueError, UnicodeError):
                return payload
            payload["endUser"] = {"idHash": digest, "algorithm": "hmac-sha256", "keyVersion": "1"}
        return payload

    def _send(self, batch: list[Entry]) -> list[Entry]:
        if self._disabled:
            return []
        now = time.monotonic()
        active = [entry for entry in batch if now - entry.queued < _MAX_AGE]
        if len(active) < len(batch):
            self._count_drop("expired", len(batch) - len(active))
        if not active or self._next_send() > now:
            return active
        events = [self._event_payload(entry.event) for entry in active]
        body = _json_bytes({
            "schemaVersion": "1",
            "sdk": {"name": "brello-sdk", "version": self._version, "runtime": "python"},
            "events": events,
        })
        if len(body) > _MAX_BATCH_BYTES:
            return self._split(active)
        if self._debug:
            visible = events if self._debug_include_context else [
                {key: "[masked]" if key in ("endUser", "attributes") else value
                 for key, value in event.items()} for event in events
            ]
            self._log(logging.DEBUG, "Sending events", visible)
        try:
            reply = self._request(self._base_url, self._key, "events", body, self._debug)
        except Exception:
            return self._defer(active)
        if reply.status == 429 or reply.status >= 500:
            return self._defer(active, reply)
        self._reset_circuit()
        if reply.status == 401:
            self._disable()
        elif reply.status == 413:
            return self._split(active)
        elif reply.status in (202, 422):
            rejected = _object(_object(reply.data).get("data")).get("rejectedCount")
            if type(rejected) is int and rejected > 0:
                self._log(logging.DEBUG, "Events rejected", rejected)
        else:
            self._log(logging.DEBUG, "Batch rejected", reply.status)
        return []

    def _defer(self, batch: list[Entry], reply: Reply | None = None) -> list[Entry]:
        now = time.monotonic()
        status = reply.status if reply else None
        if status == 429:
            self._failures = 0
        else:
            self._failures += 1
            if self._failures >= 3 or self._circuit_until:
                previous = self._circuit_delay
                self._circuit_delay = min(previous * 2, 300) if previous else 30
                delay = self._circuit_delay * random.uniform(0.8, 1.0) if previous else 30
                self._circuit_until = now + delay
                self._log(logging.DEBUG, "circuit_open", round(delay, 2))
        backoff = min(0.5 * 2 ** batch[0].failures, 8) * random.uniform(0.5, 1.0)
        self._retry_at = now + max(backoff, _retry_after(reply.retry_after if reply else None))
        retry = []
        for entry in batch:
            entry.failures += 1
            if entry.failures < _MAX_ATTEMPTS:
                retry.append(entry)
        if len(retry) < len(batch):
            self._count_drop("attempts_exhausted", len(batch) - len(retry))
        return retry

    def _reset_circuit(self) -> None:
        self._failures = 0
        self._circuit_until = self._circuit_delay = self._retry_at = 0

    def _split(self, batch: list[Entry]) -> list[Entry]:
        if len(batch) == 1:
            self._count_drop("oversized", 1)
            return []
        midpoint = (len(batch) + 1) // 2
        return self._send(batch[:midpoint]) + self._send(batch[midpoint:])

    def _disable(self) -> None:
        with self._condition:
            self._disabled = True
            self._bytes -= sum(entry.size for entry in self._pending)
            self._pending.clear()
            self._condition.notify_all()
        self._log(logging.WARNING, "The Brello project key was rejected. Event delivery is paused.")

    def _count_drop(self, reason: str, count: int) -> None:
        with self._condition:
            self._drops[reason] += count

    def _log_drops(self) -> None:
        if not self._debug:
            return
        with self._condition:
            now = time.monotonic()
            if now < self._next_drop_log or not self._drops:
                return
            counts = dict(self._drops)
            self._drops.clear()
            self._next_drop_log = now + 5
        self._log(logging.DEBUG, "Events dropped", counts)

    def _log(self, level: int, message: str, detail: Any = "") -> None:
        if level != logging.DEBUG or self._debug:
            try:
                _LOG.log(logging.INFO if level == logging.DEBUG else level, "%s %s", message, detail)
            except Exception:
                pass
