"""Observe supported provider calls without replacing their protocol or consuming their bodies."""

import asyncio
from collections.abc import Callable
import inspect
from typing import Any, TypeVar, cast

from ._anthropic import AnthropicObservation, message_text
from ._google import METHODS as GOOGLE_METHODS, GoogleObservation, GoogleProvider
from ._observation import Observation, Record
from ._openai import BackgroundResponses, OpenAIObservation
from ._telemetry import text

T = TypeVar("T")
_Context = Callable[[], dict[str, Any]]
_FINAL_METHOD = {
    "messages": "get_final_message",
    "chat": "get_final_completion",
    "responses": "get_final_response",
}
_RAW_MODES = ("with_raw_response", "with_streaming_response")


def _safe(operation: Callable[..., T], *args: Any, fallback: Any = None) -> T:
    try:
        return operation(*args)
    except Exception:
        return fallback


def _invoke(method: Callable[..., Any], observation: Observation, convert: Callable[[Any], Any],
            *args: Any, **kwargs: Any) -> Any:
    try:
        value = method(*args, **kwargs)
    except (Exception, asyncio.CancelledError) as error:
        _safe(observation.failed, error)
        raise
    if not inspect.isawaitable(value):
        return _safe(convert, value, fallback=value)

    async def observe_awaitable() -> Any:
        try:
            result = await value
        except (Exception, asyncio.CancelledError) as error:
            _safe(observation.failed, error)
            raise
        return _safe(convert, result, fallback=result)

    return observe_awaitable()


class _Stream:
    def __init__(self, stream: Any, observation: Observation) -> None:
        self._stream = stream
        self._observation = observation
        self._iterator: Any = None
        _safe(observation.stream_headers, stream)

    def __iter__(self) -> "_Stream":
        return self

    def __next__(self) -> Any:
        try:
            if self._iterator is None:
                self._iterator = iter(self._stream)
            value = next(self._iterator)
        except StopIteration:
            _safe(self._observation.completed)
            raise
        except Exception as error:
            _safe(self._observation.failed, error)
            raise
        _safe(self._observation.chunk, value)
        return value

    def __aiter__(self) -> "_Stream":
        return self

    async def __anext__(self) -> Any:
        try:
            if self._iterator is None:
                self._iterator = aiter(self._stream)
            value = await anext(self._iterator)
        except StopAsyncIteration:
            _safe(self._observation.completed)
            raise
        except (Exception, asyncio.CancelledError) as error:
            _safe(self._observation.failed, error)
            raise
        _safe(self._observation.chunk, value)
        return value

    def __enter__(self) -> "_Stream":
        return _invoke(self._stream.__enter__, self._observation, self._entered)

    async def __aenter__(self) -> "_Stream":
        return await _invoke(self._stream.__aenter__, self._observation, self._entered)

    def _entered(self, stream: Any) -> "_Stream":
        if stream is self._stream:
            return self
        return _Stream(stream, self._observation)

    def __exit__(self, *args: Any) -> Any:
        self._exit(args[1])
        return self._stream.__exit__(*args)

    async def __aexit__(self, *args: Any) -> Any:
        self._exit(args[1])
        return await self._stream.__aexit__(*args)

    def _exit(self, error: BaseException | None = None) -> None:
        if error is None:
            _safe(self._observation.finish, "cancelled")
        else:
            _safe(self._observation.failed, error)

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._stream, name)
        observation = self._observation
        if observation.endpoint == "messages":
            # The provider's helper can return a partial message after a clean EOF.
            # Consume through the proxy so only message_stop records success.
            if name == "text_stream":
                return self._anthropic_text(hasattr(value, "__aiter__"))
            if name in ("get_final_message", "get_final_text", "until_done"):
                return self._anthropic_helper(value)
        if name in _FINAL_METHOD.values():
            def accept_final(result: Any) -> Any:
                observation.result(result)
                return result

            return lambda *args, **kwargs: _invoke(value, observation, accept_final, *args, **kwargs)
        if name in ("until_done", "get_final_text"):
            def finish(*args: Any, **kwargs: Any) -> Any:
                result = _invoke(value, observation, lambda result: result, *args, **kwargs)
                if inspect.isawaitable(result):
                    async def complete() -> Any:
                        returned = await result
                        await self._capture_final_async()
                        return returned
                    return complete()
                self._capture_final()
                return result

            return finish
        if name in ("close", "aclose"):
            def close(*args: Any, **kwargs: Any) -> Any:
                self._exit()
                return value(*args, **kwargs)

            return close
        if name == "text_stream":
            return self._watch_text(value)
        return value

    def _anthropic_text(self, asynchronous: bool) -> Any:
        if asynchronous:
            async def iterate_async() -> Any:
                async for event in self:
                    if (value := message_text(event)) is not None:
                        yield value
            return iterate_async()

        def iterate() -> Any:
            for event in self:
                if (value := message_text(event)) is not None:
                    yield value
        return iterate()

    def _anthropic_helper(self, method: Callable[..., Any]) -> Callable[..., Any]:
        if hasattr(self._stream, "__aiter__"):
            async def run_async(*args: Any, **kwargs: Any) -> Any:
                if not self._observation.done:
                    async for _ in self:
                        pass
                return await method(*args, **kwargs)
            return run_async

        def run(*args: Any, **kwargs: Any) -> Any:
            if not self._observation.done:
                for _ in self:
                    pass
            return method(*args, **kwargs)
        return run

    def _capture_final(self) -> None:
        if self._observation.done:
            return
        final_name = _FINAL_METHOD.get(self._observation.endpoint)
        try:
            if final_name:
                self._observation.result(getattr(self._stream, final_name)())
        except Exception:
            _safe(self._observation.completed)

    async def _capture_final_async(self) -> None:
        if self._observation.done:
            return
        final_name = _FINAL_METHOD.get(self._observation.endpoint)
        try:
            if final_name:
                self._observation.result(await getattr(self._stream, final_name)())
        except Exception:
            _safe(self._observation.completed)

    def _watch_text(self, value: Any) -> Any:
        if hasattr(value, "__aiter__"):
            async def iterate() -> Any:
                try:
                    async for item in value:
                        yield item
                except (Exception, asyncio.CancelledError, GeneratorExit) as error:
                    _safe(self._observation.failed, error)
                    raise
                await self._capture_final_async()
            return iterate()

        def iterate() -> Any:
            try:
                yield from value
            except (Exception, GeneratorExit) as error:
                _safe(self._observation.failed, error)
                raise
            self._capture_final()
        return iterate()


class _RawManager:
    def __init__(self, manager: Any, observation: Observation) -> None:
        self._manager = manager
        self._observation = observation

    def __enter__(self) -> Any:
        return _invoke(self._manager.__enter__, self._observation, self._raw)

    async def __aenter__(self) -> Any:
        return await _invoke(self._manager.__aenter__, self._observation, self._raw)

    def __exit__(self, *args: Any) -> Any:
        return self._manager.__exit__(*args)

    async def __aexit__(self, *args: Any) -> Any:
        return await self._manager.__aexit__(*args)

    def _raw(self, value: Any) -> Any:
        self._observation.raw(value)
        return value

    def __getattr__(self, name: str) -> Any:
        return getattr(self._manager, name)


class _Resource:
    def __init__(self, resource: Any, endpoint: str, context: _Context, record: Record,
                 mode: str | None = None, provider: GoogleProvider | None = None,
                 background: BackgroundResponses | None = None) -> None:
        self._resource = resource
        self._endpoint = endpoint
        self._context = context
        self._record = record
        self._mode = mode
        self._provider = provider
        self._background = background

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._resource, name)
        if self._provider:
            endpoint = GOOGLE_METHODS.get(name)
        elif name in _RAW_MODES:
            return _Resource(value, self._endpoint, self._context, self._record, name)
        else:
            supported = (name in ("create", "stream")
                         or (name == "parse" and self._endpoint in ("responses", "chat", "messages"))
                         or (name == "retrieve" and self._background is not None and self._mode is None))
            endpoint = self._endpoint if supported else None
        if endpoint is None or not callable(value):
            return value

        def call(*args: Any, **kwargs: Any) -> Any:
            def prepare() -> Observation | None:
                model = text(kwargs.get("model"), 256)
                background = None
                response_id = None
                if self._background is not None and self._mode is None:
                    if name in ("retrieve", "stream"):
                        response_id = text(kwargs.get("response_id", args[0] if args else None), 256)
                    if name == "retrieve" and (not response_id or not self._background.is_pending(response_id)):
                        return None
                    if response_id or kwargs.get("background") is True:
                        background = self._background
                if model is None and (endpoint != "responses" or self._mode is not None):
                    return None
                model = model or "unknown"
                context = self._context()
                if self._provider:
                    return GoogleObservation(endpoint, model, context, self._record, provider=self._provider)
                if endpoint == "messages":
                    return AnthropicObservation(model, context, self._record, fast=kwargs.get("speed") == "fast")
                return OpenAIObservation(endpoint, model, context, self._record,
                                         background=background, response_id=response_id)

            observation = _safe(prepare)
            if observation is None:
                return value(*args, **kwargs)

            def convert(result: Any) -> Any:
                if self._mode == "with_streaming_response":
                    return _RawManager(result, observation)
                if self._mode == "with_raw_response":
                    observation.raw(result)
                elif name in ("stream", "generate_content_stream") or kwargs.get("stream") is True:
                    return _Stream(result, observation)
                else:
                    observation.result(result)
                return result

            return _invoke(value, observation, convert, *args, **kwargs)

        return call


class _Client:
    def __init__(self, client: Any, context: _Context, record: Record, section: str = "",
                 mode: str | None = None, provider: GoogleProvider | None = None,
                 background: BackgroundResponses | None = None) -> None:
        self._client = client
        self._context = context
        self._record = record
        self._section = section
        self._mode = mode
        self._provider = provider
        self._background = background if background is not None else BackgroundResponses()

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._client, name)
        if self._provider:
            if name == "models":
                return _Resource(value, "google-models", self._context, self._record, provider=self._provider)
            if name == "aio":
                return _Client(value, self._context, self._record, provider=self._provider,
                               background=self._background)
            return value
        if self._section == "beta":
            if name == "messages":
                return _Resource(value, "messages", self._context, self._record, self._mode)
            return value
        if name == "beta":
            return _Client(value, self._context, self._record, "beta", self._mode, background=self._background)
        if name == "chat":
            return _Client(value, self._context, self._record, "chat", self._mode, background=self._background)
        if name in _RAW_MODES:
            return _Client(value, self._context, self._record, self._section, name, background=self._background)
        if name == "completions" and self._section == "chat":
            return _Resource(value, "chat", self._context, self._record, self._mode)
        if name in ("responses", "embeddings", "messages"):
            return _Resource(value, name, self._context, self._record, self._mode,
                             background=self._background if name == "responses" else None)
        return value

    def __enter__(self) -> "_Client":
        value = self._client.__enter__()
        return self if value is self._client else _Client(
            value, self._context, self._record, self._section, self._mode, self._provider, self._background,
        )

    def __exit__(self, *args: Any) -> Any:
        return self._client.__exit__(*args)

    async def __aenter__(self) -> "_Client":
        value = await self._client.__aenter__()
        return self if value is self._client else _Client(
            value, self._context, self._record, self._section, self._mode, self._provider, self._background,
        )

    async def __aexit__(self, *args: Any) -> Any:
        return await self._client.__aexit__(*args)


def wrap_client(client: T, context: _Context, record: Record, *, provider: GoogleProvider | None = None) -> T:
    if provider not in ("google_ai_studio", "google_vertex_ai"):
        provider = None
    return cast(T, _Client(client, context, record, provider=provider))
