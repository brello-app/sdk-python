"""Normalize only the metadata used by the ingestion contract."""

import re
from sys import float_info
from collections.abc import Mapping
from typing import Any

_MAX_COUNT = (1 << 53) - 1
_ATTRIBUTE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", re.ASCII)
_CACHE_FIELDS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


def field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


def text(value: Any, limit: int) -> str | None:
    if isinstance(value, str) and value and len(value.encode("utf-16-le", "surrogatepass")) <= limit * 2:
        return value
    return None


def count(value: Any) -> int | None:
    return value if type(value) is int and 0 <= value <= _MAX_COUNT else None


def merge_message_usage(previous: Any, update: Any) -> dict[str, Any]:
    usage: dict[str, Any] = {}
    for name in _CACHE_FIELDS + ("cache_creation",):
        current = field(update, name)
        usage[name] = field(previous, name) if current is None else current
    return usage


def read_usage(endpoint: str, value: Any) -> dict[str, Any] | None:
    if endpoint == "embeddings":
        input_tokens = count(field(value, "prompt_tokens"))
        return None if input_tokens is None else {"inputTokens": input_tokens, "outputTokens": 0}
    if endpoint == "messages":
        uncached = count(field(value, "input_tokens"))
        output_tokens = count(field(value, "output_tokens"))
        if uncached is None or output_tokens is None:
            return None
        cache_read = count(field(value, "cache_read_input_tokens"))
        cache_write = count(field(value, "cache_creation_input_tokens"))
        creation = field(value, "cache_creation")
        writes = {
            name: amount for name, source in (
                ("fiveMinutes", "ephemeral_5m_input_tokens"), ("oneHour", "ephemeral_1h_input_tokens")
            ) if (amount := count(field(creation, source))) is not None
        }
        total_writes = sum(writes.values())
        if cache_write is not None and cache_write != total_writes:
            return None
        input_tokens = uncached + (cache_read or 0) + total_writes
        if input_tokens > _MAX_COUNT:
            return None
        result: dict[str, Any] = {"inputTokens": input_tokens, "outputTokens": output_tokens}
        if cache_read is not None:
            result["cacheReadInputTokens"] = cache_read
        if writes:
            result["cacheCreationInputTokens"] = writes
        return result

    responses = endpoint == "responses"
    input_tokens = count(field(value, "input_tokens" if responses else "prompt_tokens"))
    output_tokens = count(field(value, "output_tokens" if responses else "completion_tokens"))
    details = field(value, "input_tokens_details" if responses else "prompt_tokens_details")
    cache_read = count(field(details, "cached_tokens"))
    if (input_tokens is None or output_tokens is None or (cache_read or 0) > input_tokens
            or (count(field(details, "cache_write_tokens")) or 0) > 0):
        return None
    result = {"inputTokens": input_tokens, "outputTokens": output_tokens}
    if cache_read is not None:
        result["cacheReadInputTokens"] = cache_read
    return result


def safe_context(context: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if feature := text(context.get("feature"), 64):
        result["feature"] = feature
    user = context.get("endUser")
    if isinstance(user, Mapping):
        identifier = text(user.get("id"), 256)
        name = user.get("name")
        display_name = text(name.strip(), 200) if isinstance(name, str) else None
        identity = {key: value for key, value in (("id", identifier), ("name", display_name))
                    if value and value.strip() and "\0" not in value}
        if identity:
            result["endUser"] = identity
    attributes = context.get("attributes")
    if isinstance(attributes, Mapping):
        accepted: dict[str, Any] = {}
        for key, value in attributes.items():
            if not isinstance(key, str) or not _ATTRIBUTE_NAME.fullmatch(key):
                continue
            if isinstance(value, str):
                if value and text(value, 256) is None:
                    continue
            elif type(value) in (int, float):
                if not -float_info.max <= value <= float_info.max:
                    continue
            elif type(value) is not bool:
                continue
            accepted[key] = value
            if len(accepted) == 32:
                break
        if accepted:
            result["attributes"] = accepted
    return result
