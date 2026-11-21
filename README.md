# Brello Python SDK

Wrap an existing OpenAI, Anthropic, or Google Gen AI client to send usage metadata to Brello. The package supports Python 3.10 and newer. Install a local checkout with `python -m pip install .`. Once published, the distribution name is `brello-sdk` and the import is `brello`.

## Connect a provider

Set `BRELLO_API_KEY` to your project's ingestion key. Set the provider's own credentials as usual.

~~~python
from brello import Brello
from openai import OpenAI
from anthropic import Anthropic

brello = Brello()
openai = brello.wrap(OpenAI())
anthropic = brello.wrap(Anthropic())
~~~

Use the same `Brello` instance throughout your process. The SDK supports `OpenAI`, `AsyncOpenAI`, `Anthropic`, and `AsyncAnthropic`. Keep your existing calls, awaits, and stream context managers.

| Provider | Observed calls |
| --- | --- |
| OpenAI | `responses.create`, `responses.parse`, `responses.stream`, `chat.completions.create`, `chat.completions.parse`, `chat.completions.stream`, `embeddings.create` |
| Anthropic | `messages.create`, `messages.parse`, `messages.stream` and their `beta.messages` equivalents |
| Google AI Studio and Vertex AI | `models.generate_content`, `models.generate_content_stream`, `models.embed_content` |

For OpenAI Chat Completions streaming, set `stream_options={"include_usage": True}` when you need the final token counts. The wrapper sends `usage: null` if counts are unavailable. Raw-response calls keep the body available to your application and send basic request metadata without usage. The SDK reads selected fields such as model, token counts, and request ID. Prompts, output text, tool arguments, and embedding vectors stay in your application. Brello estimates costs server-side.

For OpenAI background Responses, poll with `responses.retrieve` on the same wrapped client or resume the stream with `responses.stream(response_id=...)` to record terminal usage. Each wrapper tracks up to 4,096 pending responses and expires entries after 10 minutes without a status update. Token-based estimates exclude separate OpenAI built-in tool fees.

### Google AI Studio and Vertex AI

Install `google-genai` in your application and select the provider when wrapping each client.

~~~python
import os

from google import genai

studio = brello.wrap(genai.Client(api_key=os.environ["GEMINI_API_KEY"]), provider="google_ai_studio")
vertex = brello.wrap(
    genai.Client(vertexai=True, project=os.environ["GOOGLE_CLOUD_PROJECT"], location=os.environ["GOOGLE_CLOUD_LOCATION"]),
    provider="google_vertex_ai",
)
~~~

Use `client.models` for sync calls or `client.aio.models` for async calls. For async generation streams, await `generate_content_stream()` before iterating. Consume streams to completion for final token counts. If you stop early, close the stream with `close()` or `await aclose()` to record the cancellation. Brello records one event per wrapped method call and uses `usage: null` when usable counts are unavailable. Automatic Google billing imports are not yet available.

With [automatic function calling](https://googleapis.github.io/python-genai/#automatic-python-function-support), non-streaming generation exposes only the last tool round's usage. Brello combines streamed tool rounds when Google includes function call history and usable counts for each round. Keep `ignore_call_history` at its default to capture those round boundaries.

## Add context

~~~python
import os

with brello.with_context(end_user={"id": "user_123"}, feature="summarize", attributes={"plan": "pro"}):
    result = openai.responses.create(
        model=os.environ["OPENAI_MODEL"],
        input="Summarize the release notes.",
    )
~~~

The context follows the current async task. You can set defaults on `wrap()`. Pass `end_user=False` or `attributes=False` to suppress those fields. Brello sends those fields when the project permits collection. To send a project-scoped HMAC, set `hash_end_user_id=True`. The SDK omits the user identifier until the server provides hashing configuration.

## Configure delivery

| Constructor option | Environment variable | Behavior |
| --- | --- | --- |
| `api_key` | `BRELLO_API_KEY` | Project ingestion key |
| `api_base_url` | `BRELLO_API_BASE_URL` | Optional self-hosted URL. Defaults to `https://api.brello.app/v1` |
| `environment` | `BRELLO_ENV` | `prod` or `live` expects a live key. `staging` or `test` expects a test key |
| `service_name` | `BRELLO_SERVICE_NAME` | Sets the `serviceName` attribute when attribute collection is enabled |
| `enabled` | `BRELLO_ENABLED` | Use `false` or `0` in the environment to disable delivery |
| `debug` | `BRELLO_DEBUG` | Use `true` or `1` and enable INFO logging for the `brello` logger to inspect delivery |
| `debug_include_context` | | Includes user and attribute values in debug logs. Use with local test data |

Explicit options take precedence. On self-hosted deployments, include the `/v1` prefix in your URL. Use HTTPS outside loopback development.

The background worker batches 20 events or about two seconds of activity. It buffers up to 500 events or 2 MiB of serialized events, including the in-flight batch. It evicts older pending events when full and discards events after 60 seconds. Retries reuse UUIDs and respect `Retry-After`. After three consecutive ingestion failures caused by network errors, timeouts, or HTTP 5xx, delivery pauses for 30 seconds. The next batch checks recovery. Further failures increase the pause up to five minutes. Provider calls continue during delivery failures.

At shutdown or before a short-lived script exits, call `brello.flush()` or `await brello.aflush()`. The default wait is five seconds. You can set a different timeout in seconds. Flush may return earlier when delivery is paused past its deadline.

## Develop and release

~~~sh
python -m pip install '.[test]' build twine
python -m unittest discover -s tests -v
python -m build
python -m twine check dist/*
~~~

Use this directory as the Python GitHub repository root. CI tests Python 3.10 and 3.14 and checks the wheel and source distribution.

Set up a PyPI Trusted Publisher with the new GitHub repository, workflow `publish.yml`, and environment `pypi`. Publish a GitHub release tagged with the package version, such as `v0.0.1`, to trigger publishing after the release checks pass.
