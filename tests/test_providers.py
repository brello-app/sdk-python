"""Exercise the official clients without contacting providers."""

import importlib.util
import json
import os
import unittest

from test_sdk import Capture, sdk_with_request

HAS_PROVIDERS = all(importlib.util.find_spec(name) for name in ("openai", "anthropic", "httpx"))
HAS_GOOGLE = (importlib.util.find_spec("httpx") is not None and importlib.util.find_spec("google") is not None
              and importlib.util.find_spec("google.genai") is not None)
if os.getenv("CI") == "true" and not (HAS_PROVIDERS and HAS_GOOGLE):
    raise RuntimeError("Install the SDK test extra before running CI.")

if HAS_PROVIDERS:
    import anthropic
    import openai

if HAS_PROVIDERS or HAS_GOOGLE:
    import httpx

if HAS_GOOGLE:
    from google import genai
    from google.oauth2.credentials import Credentials


@unittest.skipUnless(HAS_GOOGLE, "Install the SDK test extra to test the official Google client.")
class GoogleIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_studio_generation_and_vertex_embeddings(self):
        def generated(output_tokens):
            return {
                "responseId": "google-response", "modelVersion": "gemini-2.5-flash",
                "candidates": [{"content": {"role": "model", "parts": [{"text": "hello"}]},
                                "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": output_tokens,
                                  "totalTokenCount": 8 + output_tokens},
            }

        def provider(request):
            if request.url.path.endswith(":predict"):
                self.assertEqual(request.headers.get("Authorization"), "Bearer vertex-test-token")
                return httpx.Response(200, json={"predictions": [{"embeddings": {
                    "values": [0.25], "statistics": {"token_count": 4, "truncated": False},
                }}]})
            if request.url.path.endswith(":streamGenerateContent"):
                events = "".join(f"data: {json.dumps(generated(tokens))}\n\n" for tokens in (1, 3))
                return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=events)
            self.assertTrue(request.url.path.endswith(":generateContent"), request.url.path)
            return httpx.Response(200, json=generated(2))

        transport = httpx.MockTransport(provider)
        capture = Capture()
        with sdk_with_request(capture) as sdk:
            studio_options = genai.types.HttpOptions(
                httpx_client=httpx.Client(transport=transport),
                httpx_async_client=httpx.AsyncClient(transport=transport),
            )
            with sdk.wrap(genai.Client(api_key="fixture", http_options=studio_options),
                          provider="google_ai_studio") as studio:
                args = {"model": "gemini-2.5-flash", "contents": "private"}
                self.assertEqual(studio.models.generate_content(**args).text, "hello")
                async with studio.aio as aio:
                    stream = await aio.models.generate_content_stream(**args)
                    self.assertEqual([chunk.text async for chunk in stream], ["hello", "hello"])

            vertex_options = genai.types.HttpOptions(httpx_client=httpx.Client(transport=transport))
            with sdk.wrap(genai.Client(vertexai=True, project="fixture", location="us-central1",
                                       credentials=Credentials(token="vertex-test-token"),
                                       http_options=vertex_options),
                          provider="google_vertex_ai") as vertex:
                result = vertex.models.embed_content(model="google/text-embedding-005", contents="private")
                self.assertEqual(result.embeddings[0].values, [0.25])
            await sdk.aflush()

        self.assertEqual([(event["provider"], event["model"], event["usage"]) for event in capture.events], [
            ("google_ai_studio", "gemini-2.5-flash", {"inputTokens": 8, "outputTokens": 2}),
            ("google_ai_studio", "gemini-2.5-flash", {"inputTokens": 8, "outputTokens": 3}),
            ("google_vertex_ai", "text-embedding-005", {"inputTokens": 4, "outputTokens": 0}),
        ])


@unittest.skipUnless(HAS_PROVIDERS, "Install the SDK test extra to test official provider clients.")
class OpenAIIntegrationTests(unittest.TestCase):
    def test_chat_stream_records_openai_http_request_id(self):
        chunk = {"id": "chatcmpl_fixture", "object": "chat.completion.chunk", "created": 0,
                 "model": "gpt-example", "choices": [],
                 "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}}

        def provider(_request):
            return httpx.Response(200, headers={"Content-Type": "text/event-stream", "x-request-id": "req_chat"},
                                  content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n")

        capture = Capture()
        with sdk_with_request(capture) as sdk, httpx.Client(transport=httpx.MockTransport(provider)) as http:
            with sdk.wrap(openai.OpenAI(api_key="fixture", http_client=http)) as client:
                with client.chat.completions.create(
                    model="gpt-example", messages=[{"role": "user", "content": "secret"}],
                    stream=True, stream_options={"include_usage": True},
                ) as stream:
                    self.assertEqual([item.usage.total_tokens for item in stream], [14])
            sdk.flush()

        event, = capture.events
        self.assertEqual(event["providerRequestId"], "req_chat")
        self.assertEqual(event["usage"], {"inputTokens": 10, "outputTokens": 4})

    def test_regular_embeddings_and_raw_response(self):
        def provider(request):
            if request.url.path.endswith("/embeddings"):
                return httpx.Response(200, headers={"x-request-id": "req_embedding"}, json={
                    "object": "list", "model": "text-embedding-3-small",
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                    "usage": {"prompt_tokens": 5, "total_tokens": 5},
                })
            return httpx.Response(200, headers={"x-request-id": "req_response"}, json={
                "id": "resp_example", "object": "response", "created_at": 0,
                "model": "gpt-example", "status": "completed", "output": [],
                "usage": {
                    "input_tokens": 10, "output_tokens": 4, "total_tokens": 14,
                    "input_tokens_details": {"cached_tokens": 2},
                    "output_tokens_details": {"reasoning_tokens": 1},
                },
            })

        capture = Capture()
        with sdk_with_request(capture) as sdk, httpx.Client(transport=httpx.MockTransport(provider)) as http:
            with sdk.wrap(openai.OpenAI(api_key="fixture", base_url="https://provider.example/v1", http_client=http)) as client:
                self.assertEqual(client.responses.create(model="gpt-example", input="secret").id, "resp_example")
                self.assertEqual(client.embeddings.create(
                    model="text-embedding-3-small", input="secret", encoding_format="float",
                ).data[0].embedding, [0.1, 0.2])
                with client.with_streaming_response.responses.create(model="gpt-example", input="secret") as raw:
                    self.assertEqual(raw.headers.get("x-request-id"), "req_response")
                    self.assertEqual(raw.parse().id, "resp_example")
            sdk.flush()

        self.assertEqual([row["usage"] for row in capture.events], [
            {"inputTokens": 10, "outputTokens": 4, "cacheReadInputTokens": 2},
            {"inputTokens": 5, "outputTokens": 0},
            None,
        ])
        self.assertEqual(capture.events[-1]["providerRequestId"], "req_response")


@unittest.skipUnless(HAS_PROVIDERS, "Install the SDK test extra to test official provider clients.")
class AnthropicIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_stream_helper(self):
        events = [
            {"type": "message_start", "message": {
                "id": "msg_example", "type": "message", "role": "assistant", "content": [],
                "model": "claude-example", "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": 8, "output_tokens": 1, "cache_read_input_tokens": 4},
            }},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
             "usage": {"output_tokens": 3}},
            {"type": "message_stop"},
        ]
        data = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)

        def provider(_request):
            return httpx.Response(200, headers={"Content-Type": "text/event-stream", "request-id": "req_message"},
                                  content=data.encode())

        capture = Capture()
        with sdk_with_request(capture) as sdk:
            async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as http:
                async with sdk.wrap(anthropic.AsyncAnthropic(
                    api_key="fixture", base_url="https://provider.example", http_client=http,
                )) as client:
                    async with client.messages.stream(
                        model="claude-example", max_tokens=256, messages=[{"role": "user", "content": "secret"}],
                    ) as stream:
                        self.assertEqual([text async for text in stream.text_stream], ["hello"])
                        self.assertEqual((await stream.get_final_message()).id, "msg_example")
            await sdk.aflush()

        row, = capture.events
        self.assertEqual(row["usage"], {"inputTokens": 12, "outputTokens": 3, "cacheReadInputTokens": 4})
        self.assertEqual(row["providerRequestId"], "req_message")
        self.assertNotIn("hello", str(row))
        self.assertNotIn("secret", str(row))
