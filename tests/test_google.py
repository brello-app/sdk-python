import asyncio
from types import SimpleNamespace
import unittest

from test_sdk import Capture, sdk_with_request


class GoogleTests(unittest.IsolatedAsyncioTestCase):
    async def test_sync_and_async_models_send_distinct_provider_ids(self):
        capture = Capture()
        output_reads = []

        class Result:
            model_version = "models/gemini-example"
            response_id = "response-1"
            usage_metadata = SimpleNamespace(
                prompt_token_count=20, tool_use_prompt_token_count=3, candidates_token_count=4,
                thoughts_token_count=2, cached_content_token_count=7, total_token_count=29,
            )

            @property
            def candidates(self):
                output_reads.append(True)
                return ["private output"]

        result = Result()

        async def generate(**_):
            return result

        models = SimpleNamespace(generate_content=lambda **_: result, count_tokens=lambda **_: 5)
        original = SimpleNamespace(models=models, aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
        with sdk_with_request(capture) as sdk:
            studio = sdk.wrap(original, provider="google_ai_studio")
            vertex = sdk.wrap(original, provider="google_vertex_ai")
            self.assertIs(sdk.wrap(original).models, models)
            self.assertEqual(studio.models.count_tokens(model="gemini-example"), 5)
            with sdk.with_context(feature="summary"):
                self.assertIs(studio.models.generate_content(model="gemini-example", contents="private prompt"), result)
            self.assertIs(await vertex.aio.models.generate_content(model="gemini-example"), result)
            await sdk.aflush()

        first, second = capture.events
        self.assertEqual([first["provider"], second["provider"]], ["google_ai_studio", "google_vertex_ai"])
        self.assertEqual(first["model"], "gemini-example")
        self.assertEqual(first["providerRequestId"], "response-1")
        self.assertEqual(first["feature"], "summary")
        self.assertEqual(first["usage"], {"inputTokens": 23, "outputTokens": 6, "cacheReadInputTokens": 7})
        self.assertEqual(first["usage"], second["usage"])
        self.assertNotIn("private prompt", str(capture.events))
        self.assertEqual(output_reads, [])

    async def test_lazy_streams_record_final_usage_cancellation_and_errors_once(self):
        capture = Capture()
        chunks = [
            {"usage_metadata": {"prompt_token_count": 8, "candidates_token_count": 2}},
            {"usage_metadata": {"prompt_token_count": 8, "candidates_token_count": 5, "total_token_count": 13,
                                "traffic_type": "PROVISIONED_THROUGHPUT"}},
            {},
        ]
        advanced = []

        def stream(**_):
            advanced.append(True)
            yield from chunks

        failure = RuntimeError("private stream failure")
        failure.code = 503

        async def astream(*, fail=False, **_):
            async def iterate():
                yield chunks[0]
                if fail:
                    raise failure
                await asyncio.sleep(0)
                yield chunks[1]
            return iterate()

        with sdk_with_request(capture) as sdk:
            client = sdk.wrap(SimpleNamespace(
                models=SimpleNamespace(generate_content_stream=stream),
                aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=astream)),
            ), provider="google_vertex_ai")
            model = "projects/test/locations/global/publishers/google/models/gemini-example"
            sync = client.models.generate_content_stream(model=model)
            self.assertEqual(advanced, [])
            self.assertEqual(list(sync), chunks)
            self.assertEqual(list(sync), [])
            cancelled = await client.aio.models.generate_content_stream(model=model)
            self.assertIs(await anext(cancelled), chunks[0])
            await cancelled.aclose()
            broken = await client.aio.models.generate_content_stream(model=model, fail=True)
            self.assertIs(await anext(broken), chunks[0])
            with self.assertRaises(RuntimeError) as caught:
                await anext(broken)
            self.assertIs(caught.exception, failure)
            finished = await client.aio.models.generate_content_stream(model=model)
            self.assertEqual([chunk async for chunk in finished], chunks[:2])
            await sdk.aflush()

        first, cancelled, error, last = capture.events
        self.assertEqual([event["status"] for event in capture.events], ["success", "cancelled", "error", "success"])
        self.assertEqual(first["model"], "gemini-example")
        self.assertEqual(first["usage"], {"inputTokens": 8, "outputTokens": 5})
        self.assertEqual(first["billing"], {"serviceTier": "provisioned_throughput"})
        self.assertEqual(last["usage"], first["usage"])
        self.assertIsNone(cancelled["usage"])
        self.assertEqual(error["error"], {"httpStatus": 503})
        self.assertNotIn("private stream failure", str(capture.events))

    def test_embeddings_use_provider_counts_and_leave_vectors_untouched(self):
        capture = Capture()
        vector_reads = []

        class Embedding:
            statistics = SimpleNamespace(token_count=4.0)

            @property
            def values(self):
                vector_reads.append(True)
                return [0.25]

        embedding = Embedding()
        results = [
            {"usage_metadata": {"prompt_token_count": 5}, "embeddings": [embedding]},
            {"embeddings": [embedding, embedding]},
            {"embeddings": [embedding, {}]},
        ]
        with sdk_with_request(capture) as sdk:
            client = sdk.wrap(SimpleNamespace(models=SimpleNamespace(
                embed_content=lambda *, index, **_: results[index],
            )), provider="google_vertex_ai")
            for index, result in enumerate(results):
                self.assertIs(client.models.embed_content(model="google/gemini-embedding-example", index=index), result)
            sdk.flush()

        self.assertEqual([event["usage"] for event in capture.events], [
            {"inputTokens": 5, "outputTokens": 0}, {"inputTokens": 8, "outputTokens": 0}, None,
        ])
        self.assertEqual({event["model"] for event in capture.events}, {"gemini-embedding-example"})
        self.assertEqual(vector_reads, [])

    def test_stream_adds_tool_rounds_without_counting_continuation_snapshots_twice(self):
        def chunk(request_id, input_tokens, output_tokens, cached, history=None, traffic="ON_DEMAND"):
            return SimpleNamespace(
                response_id=request_id, automatic_function_calling_history=history,
                usage_metadata=SimpleNamespace(prompt_token_count=input_tokens,
                    candidates_token_count=output_tokens, cached_content_token_count=cached, traffic_type=traffic),
            )

        history = [object(), object()]
        chunks = [
            chunk("first", 8, 1, 2),
            chunk("continuation", 10, 2, 3, traffic="ON_DEMAND_FLEX"),
            chunk("second", 20, 3, 4, history),
            chunk("second", 20, 5, 4, history),
            chunk("third", 24, 6, 8, [*history, object(), object()]),
        ]
        capture = Capture()
        with sdk_with_request(capture) as sdk:
            client = sdk.wrap(SimpleNamespace(models=SimpleNamespace(
                generate_content_stream=lambda **_: iter(chunks),
            )), provider="google_ai_studio")
            self.assertEqual(list(client.models.generate_content_stream(model="gemini-example")), chunks)
            sdk.flush()

        event, = capture.events
        self.assertEqual(event["usage"], {"inputTokens": 54, "outputTokens": 13, "cacheReadInputTokens": 15})
        self.assertEqual(event["billing"], {"serviceTier": "flex"})
        self.assertNotIn("providerRequestId", event)

    def test_stream_does_not_report_partial_totals_when_a_tool_round_lacks_usage(self):
        chunks = [
            {"usage_metadata": {"prompt_token_count": 8, "candidates_token_count": 2}},
            {"automatic_function_calling_history": [object(), object()]},
        ]
        capture = Capture()
        with sdk_with_request(capture) as sdk:
            client = sdk.wrap(SimpleNamespace(models=SimpleNamespace(
                generate_content_stream=lambda **_: iter(chunks),
            )), provider="google_ai_studio")
            self.assertEqual(list(client.models.generate_content_stream(model="gemini-example")), chunks)
            sdk.flush()

        event, = capture.events
        self.assertIsNone(event["usage"])
