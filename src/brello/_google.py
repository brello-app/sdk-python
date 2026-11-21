"""Interpret Google Gen AI responses and multiple streamed tool rounds."""

from typing import Any, Literal

from ._google_usage import combine_generation_rounds, embedding_usage, generation_usage, model_id, service_tier
from ._observation import Observation, Record
from ._telemetry import count, field, text

GoogleProvider = Literal["google_ai_studio", "google_vertex_ai"]
METHODS = {
    "generate_content": "google-generate",
    "generate_content_stream": "google-generate",
    "embed_content": "google-embedding",
}


class GoogleObservation(Observation):
    def __init__(self, endpoint: str, model: str, context: dict[str, Any], record: Record,
                 *, provider: GoogleProvider) -> None:
        super().__init__(endpoint, model_id(model), context, record, provider=provider)
        self._history_size = 0
        self._rounds: list[dict[str, int] | None] = []
        self._nonstandard_tier: str | None = None

    def _metadata(self, result: Any) -> None:
        if model := text(field(result, "model_version"), 256):
            self._model = model_id(model)
        self._request_id = self._request_id or text(field(result, "response_id"), 256)
        usage = field(result, "usage_metadata")
        self._service_tier = service_tier(usage) or self._service_tier
        if self.endpoint == "google-embedding":
            self._usage = embedding_usage(result)
        elif usage is not None:
            self._usage = generation_usage(usage)

    def chunk(self, value: Any) -> None:
        if self.done:
            return
        # Google grows AFC history between tool rounds. Within one round,
        # stream snapshots can already include usage from continuation hops.
        history = field(value, "automatic_function_calling_history")
        if isinstance(history, list) and len(history) > self._history_size:
            self._history_size = len(history)
            self._rounds.append(self._usage)
            self._usage = None
            if self._service_tier != "standard":
                self._nonstandard_tier = self._service_tier or self._nonstandard_tier
            self._service_tier = None
        self._metadata(value)

    def completed(self) -> None:
        if self.done:
            return
        if self._rounds:
            self._usage = combine_generation_rounds([*self._rounds, self._usage])
            self._request_id = None
            self._service_tier = self._nonstandard_tier or self._service_tier
        completed = self.endpoint == "google-generate"
        self.finish("success" if completed else "error", use_usage=completed)

    def _http_status(self, reason: Any) -> int | None:
        status = super()._http_status(reason)
        return count(field(reason, "code")) if status is None else status

    def _normalized_usage(self) -> dict[str, int] | None:
        return self._usage
