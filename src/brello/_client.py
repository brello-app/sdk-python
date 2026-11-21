"""Brello client and task-local event context."""

import asyncio
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import logging
import math
import os
import re
from typing import Any, TypeVar
from urllib.parse import urlsplit, urlunsplit
from zlib import crc32

from ._google import GoogleProvider
from ._telemetry import safe_context, text
from ._transport import Transport
from ._wrap import wrap_client

T = TypeVar("T")
_DEFAULT_BASE_URL = "https://api.brello.app/v1"
_ISSUED = re.compile(r"(brl_(live|test)_|brello_)[A-Za-z0-9_-]{43}")
_CHECKSUM = re.compile(r"(brl_(live|test)_[A-Za-z0-9_-]{43})_([0-9a-f]{8})")
_LOG = logging.getLogger("brello")

try:
    _VERSION = version("brello-sdk")
except PackageNotFoundError:
    _VERSION = "0.0.1"


@dataclass(frozen=True)
class BrelloError:
    code: str
    message: str


def _key(value: str | None) -> tuple[str, str] | None:
    if not isinstance(value, str):
        return None
    if match := _ISSUED.fullmatch(value):
        return value, match[2] or "live"
    if match := _CHECKSUM.fullmatch(value):
        if format(crc32(match[1].encode("ascii")), "08x") == match[3]:
            return match[1], match[2]
    return None


def _api_url(value: str) -> str | None:
    try:
        url = urlsplit(value)
        local = url.hostname in ("localhost", "127.0.0.1", "::1")
        if (
            not url.hostname or not (url.scheme == "https" or (url.scheme == "http" and local))
            or url.username is not None or url.password is not None or url.query or url.fragment
            or any(char.isspace() for char in value)
        ):
            return None
        _ = url.port
        return urlunsplit((url.scheme, url.netloc, (url.path.rstrip("/") or "/v1") + "/", "", ""))
    except (TypeError, ValueError):
        return None


def _warn(message: str) -> None:
    try:
        _LOG.warning(message)
    except Exception:
        pass


def _context_fields(end_user: Mapping[str, str] | bool | None, feature: str | None,
                    attributes: Mapping[str, str | int | float | bool] | bool | None) -> dict[str, Any]:
    return {key: value for key, value in (("endUser", end_user), ("feature", feature), ("attributes", attributes))
            if value is not None}


class Brello:
    """Wrap a provider client and send telemetry in the background."""

    def __init__(
        self, api_key: str | None = None, *, api_base_url: str | None = None,
        environment: str | None = None, service_name: str | None = None, enabled: bool | None = None,
        debug: bool | None = None, debug_include_context: bool = False, hash_end_user_id: bool = False,
    ) -> None:
        candidate = os.getenv("BRELLO_API_KEY") if api_key is None else api_key
        key = _key(candidate)
        self.mode = key[1] if key else None
        missing = candidate in (None, "")
        self.api_key_status = "valid" if key else "missing" if missing else "invalid"
        self.error = None if key else (
            BrelloError("API_KEY_MISSING", "Set BRELLO_API_KEY or pass api_key.") if missing
            else BrelloError("API_KEY_INVALID", "Use a Brello project ingestion key.")
        )
        self._context: ContextVar[dict[str, Any] | None] = ContextVar("brello_context", default=None)
        self._service_name = text(os.getenv("BRELLO_SERVICE_NAME") if service_name is None else service_name, 64)
        self._transport: Transport | None = None
        if enabled is False or (enabled is None and os.getenv("BRELLO_ENABLED", "").lower() in ("false", "0")):
            return
        if not key:
            if not missing:
                _warn("The Brello project key format is invalid. Event delivery is disabled.")
            return
        expected = os.getenv("BRELLO_ENV") if environment is None else environment
        expected = {"prod": "live", "staging": "test"}.get(expected, expected)
        if expected is not None and expected != self.mode:
            _warn("BRELLO_ENV does not match the Brello project key. Event delivery is disabled.")
            return
        configured = os.getenv("BRELLO_API_BASE_URL", "").strip() or _DEFAULT_BASE_URL
        url = _api_url(configured if api_base_url is None else api_base_url)
        if not url:
            _warn("The custom Brello API base URL is invalid. Event delivery is disabled.")
            return
        self._transport = Transport(
            key[0], url, debug=os.getenv("BRELLO_DEBUG", "").lower() in ("true", "1") if debug is None else debug,
            debug_include_context=debug_include_context, hash_end_user_id=hash_end_user_id, version=_VERSION,
        )

    def wrap(
        self, client: T, *, end_user: Mapping[str, str] | bool | None = None, feature: str | None = None,
        attributes: Mapping[str, str | int | float | bool] | bool | None = None,
        provider: GoogleProvider | None = None,
    ) -> T:
        """Keep the wrapped provider client's type and normal method signatures."""
        transport = self._transport
        if not transport:
            return client
        defaults = _context_fields(end_user, feature, attributes)

        def context() -> dict[str, Any]:
            values = {**defaults, **(self._context.get() or {})}
            tags = values.get("attributes")
            if self._service_name and tags is not False:
                values["attributes"] = {"serviceName": self._service_name, **(tags if isinstance(tags, Mapping) else {})}
            return safe_context(values)

        return wrap_client(client, context, transport.enqueue, provider=provider)

    @contextmanager
    def with_context(
        self, *, end_user: Mapping[str, str] | bool | None = None, feature: str | None = None,
        attributes: Mapping[str, str | int | float | bool] | bool | None = None,
    ) -> Iterator[None]:
        """Apply event metadata to provider calls inside this context."""
        token = self._context.set({**(self._context.get() or {}), **_context_fields(end_user, feature, attributes)})
        try:
            yield
        finally:
            self._context.reset(token)

    def flush(self, timeout: float = 5.0) -> None:
        """Wait at most timeout seconds for eligible buffered delivery."""
        if self._transport and math.isfinite(timeout) and timeout > 0:
            self._transport.flush(min(timeout, 60))

    async def aflush(self, timeout: float = 5.0) -> None:
        """Flush without blocking the asyncio event loop."""
        if self._transport:
            await asyncio.to_thread(self.flush, timeout)
