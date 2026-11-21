"""Interpret Anthropic Messages responses and terminal stream events."""

from typing import Any

from ._observation import Observation, Record
from ._telemetry import field, merge_message_usage, text


def message_text(event: Any) -> str | None:
    delta = field(event, "delta")
    if field(event, "type") == "content_block_delta" and field(delta, "type") == "text_delta":
        return field(delta, "text")
    return None


class AnthropicObservation(Observation):
    request_id_header = "request-id"

    def __init__(self, model: str, context: dict[str, Any], record: Record, *, fast: bool) -> None:
        super().__init__("messages", model, context, record, provider="anthropic")
        self._fast = fast

    def _metadata(self, result: Any) -> None:
        super()._metadata(result)
        usage = field(result, "usage")
        self._service_tier = text(field(usage, "service_tier"), 64) or self._service_tier
        speed = field(usage, "speed")
        if speed in ("standard", "fast"):
            self._fast = speed == "fast"

    def chunk(self, value: Any) -> None:
        if self.done:
            return
        kind = field(value, "type")
        if kind == "message_start":
            self._metadata(field(value, "message"))
        elif kind == "message_delta":
            self._usage = merge_message_usage(self._usage, field(value, "usage"))
        elif kind == "message_stop":
            self.result(field(value, "message"))
        elif kind == "error":
            self.finish("error", field(value, "error"))

    def _billing(self) -> dict[str, str]:
        billing = super()._billing()
        if self._fast:
            billing["mode"] = "unknown"
        return billing
