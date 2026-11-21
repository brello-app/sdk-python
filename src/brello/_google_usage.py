"""Normalize Google Gen AI metadata for both Google backends."""

from collections.abc import Sequence
import re
from typing import Any

from ._telemetry import count, field

_MODEL_PREFIX = re.compile(r"^(?:google/|(?:(?:projects/[^/]+/locations/[^/]+/)?publishers/google/)?models/)(?=[^/]+$)")
_TRAFFIC_TIERS = {
    "ON_DEMAND_FLEX": "flex",
    "ON_DEMAND_OFFPEAK": "offpeak",
    "ON_DEMAND_PRIORITY": "priority",
    "PROVISIONED_THROUGHPUT": "provisioned_throughput",
}


def model_id(model: str) -> str:
    return _MODEL_PREFIX.sub("", model)


def _optional_count(value: Any) -> int | None:
    return 0 if value is None else count(value)


def generation_usage(usage: Any) -> dict[str, int] | None:
    prompt = count(field(usage, "prompt_token_count"))
    candidates = field(usage, "candidates_token_count")
    if candidates is None:
        candidates = field(usage, "response_token_count")
    output = _optional_count(candidates)
    tool = _optional_count(field(usage, "tool_use_prompt_token_count"))
    thoughts = _optional_count(field(usage, "thoughts_token_count"))
    cached_value = field(usage, "cached_content_token_count")
    cached = count(cached_value)
    total_value = field(usage, "total_token_count")
    total = count(total_value)
    if (prompt is None or output is None or tool is None or thoughts is None
            or (cached_value is not None and cached is None)
            or (candidates is None and total is None)):
        return None

    input_tokens = prompt + tool
    output_tokens = output + thoughts
    combined = input_tokens + output_tokens
    if count(combined) is None or (cached or 0) > input_tokens or (total_value is not None and total != combined):
        return None
    result = {"inputTokens": input_tokens, "outputTokens": output_tokens}
    if cached is not None:
        result["cacheReadInputTokens"] = cached
    return result


def combine_generation_rounds(rounds: list[dict[str, int] | None]) -> dict[str, int] | None:
    totals = {"inputTokens": 0, "outputTokens": 0}
    for usage in rounds:
        if usage is None:
            return None
        for key, value in usage.items():
            totals[key] = totals.get(key, 0) + value
    return totals if count(totals["inputTokens"] + totals["outputTokens"]) is not None else None


def embedding_usage(response: Any) -> dict[str, int] | None:
    usage = field(response, "usage_metadata")
    reported = field(usage, "prompt_token_count")
    if reported is None:
        reported = field(usage, "total_token_count")
    if (tokens := count(reported)) is not None:
        return {"inputTokens": tokens, "outputTokens": 0}

    embeddings = field(response, "embeddings")
    if not isinstance(embeddings, Sequence) or isinstance(embeddings, (str, bytes)) or not embeddings:
        return None
    input_tokens = 0
    for embedding in embeddings:
        tokens = field(field(embedding, "statistics"), "token_count")
        # Google may decode embedding statistics as floats, including exact integers.
        if isinstance(tokens, float) and tokens.is_integer():
            tokens = int(tokens)
        validated = count(tokens)
        if validated is None:
            return None
        input_tokens += validated
    return None if count(input_tokens) is None else {"inputTokens": input_tokens, "outputTokens": 0}


def service_tier(usage: Any) -> str | None:
    traffic = field(usage, "traffic_type")
    if isinstance(traffic, str) and traffic in _TRAFFIC_TIERS:
        return _TRAFFIC_TIERS[traffic]
    tier = field(usage, "service_tier")
    if tier in ("standard", "flex", "priority"):
        return tier
    return "standard" if traffic == "ON_DEMAND" else None
