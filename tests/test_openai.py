from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from test_sdk import Capture, sdk_with_request


def response(response_id="resp_fixture", status="completed"):
    usage = NS(input_tokens=10, output_tokens=4) if status == "completed" else None
    return NS(id=response_id, status=status, model="gpt-example", usage=usage)


class OpenAITests(unittest.IsolatedAsyncioTestCase):
    async def test_completed_responses_do_not_evict_pending_context(self):
        capture = Capture()
        with sdk_with_request(capture) as sdk, patch("brello._openai._COMPLETED_LIMIT", 1):
            resource = NS(create=lambda *, identifier, **kwargs: response(identifier, "queued"), retrieve=response)
            client = sdk.wrap(NS(responses=resource))
            with sdk.with_context(feature="original"):
                client.responses.create(model="gpt-example", background=True, identifier="resp_original")
            for identifier in ("resp_first", "resp_second"):
                client.responses.create(model="gpt-example", background=True, identifier=identifier)
                client.responses.retrieve(identifier)
            client.responses.retrieve("resp_original")
            await sdk.aflush()

        self.assertEqual(len(capture.events), 3)
        self.assertEqual(capture.events[-1]["feature"], "original")

    async def test_pending_response_ttl_and_capacity_are_bounded(self):
        capture = Capture()
        clock = NS(monotonic=lambda: clock.now, now=0)
        with (sdk_with_request(capture) as sdk,
              patch("brello._openai.time", clock), patch("brello._openai._PENDING_LIMIT", 2),
              self.assertLogs("brello", level="DEBUG") as diagnostics):
            resource = NS(create=lambda *, identifier, **kwargs: response(identifier, "queued"), retrieve=response)
            client = sdk.wrap(NS(responses=resource))
            for identifier in ("resp_evicted", "resp_expired", "resp_finished"):
                client.responses.create(model="gpt-example", background=True, identifier=identifier)
            client.responses.retrieve("resp_evicted")
            client.responses.retrieve("resp_finished")
            clock.now = 600
            client.responses.retrieve("resp_expired")
            await sdk.aflush()

        self.assertEqual(len(capture.events), 1)
        self.assertTrue(any("tracking limit" in message for message in diagnostics.output))
        self.assertTrue(any("Expired 1" in message for message in diagnostics.output))
        self.assertNotIn("resp_", str(diagnostics.output))

    async def test_structured_outputs_observe_sync_and_async_parse(self):
        parsed_response = response()
        parsed_chat = NS(model="gpt-example", usage=NS(prompt_tokens=8, completion_tokens=2))

        async def parse_chat(**kwargs):
            return parsed_chat

        capture = Capture()
        with sdk_with_request(capture) as sdk:
            client = sdk.wrap(NS(responses=NS(parse=lambda **kwargs: parsed_response),
                                 chat=NS(completions=NS(parse=parse_chat))))
            self.assertIs(client.responses.parse(model="gpt-example"), parsed_response)
            self.assertIs(await client.chat.completions.parse(model="gpt-example"), parsed_chat)
            await sdk.aflush()

        self.assertEqual([event["usage"] for event in capture.events], [
            {"inputTokens": 10, "outputTokens": 4}, {"inputTokens": 8, "outputTokens": 2},
        ])

    async def test_background_polls_emit_once_with_the_creation_context(self):
        queued = response(status="queued")
        pending = response(status="in_progress")
        terminal = response()
        capture = Capture()
        with sdk_with_request(capture) as sdk:
            resource = NS(create=lambda **kwargs: queued,
                          retrieve=lambda response_id: pending if resource.pending else terminal,
                          pending=True)
            client = sdk.wrap(NS(responses=resource))
            with sdk.with_context(feature="original"):
                self.assertIs(client.responses.create(model="gpt-example", background=True), queued)
            self.assertIs(client.responses.retrieve(queued.id), pending)
            await sdk.aflush()
            self.assertEqual(capture.events, [])

            resource.pending = False
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(client.responses.retrieve, [queued.id, queued.id]))
            self.assertEqual(results, [terminal, terminal])
            self.assertIs(client.responses.retrieve(response_id="resp_untracked"), terminal)
            await sdk.aflush()

        event, = capture.events
        self.assertEqual(event["feature"], "original")
        self.assertEqual(event["usage"], {"inputTokens": 10, "outputTokens": 4})

    async def test_background_resume_uses_final_usage_and_stream_header(self):
        queued = response(status="queued")
        terminal = response()

        class Stream:
            _raw_stream = NS(response=NS(headers={"x-request-id": "req_resume"}))

            def __init__(self, *events):
                self.events = events

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def __aiter__(self):
                for event in self.events:
                    yield event

        def stream(**kwargs):
            if kwargs.get("response_id"):
                return Stream(NS(type="response.completed", response=terminal))
            return Stream(NS(type="response.created", response=queued))

        capture = Capture()
        with sdk_with_request(capture) as sdk:
            upstream = NS(responses=NS(stream=stream, retrieve=lambda response_id: terminal))
            client = sdk.wrap(upstream)
            with sdk.with_context(feature="original"):
                async with client.responses.stream(model="gpt-example", background=True) as initial:
                    self.assertIs((await anext(initial)).response, queued)
            for _ in range(2):
                async with client.responses.stream(response_id=queued.id, starting_after=1) as resumed:
                    self.assertEqual([event.response async for event in resumed], [terminal])
            client.responses.retrieve(queued.id)
            await sdk.aflush()

            # A resume on another wrapper produces the same server deduplication key.
            async with sdk.wrap(upstream).responses.stream(response_id=queued.id) as resumed:
                self.assertIs((await anext(resumed)).response, terminal)
            await sdk.aflush()

        event, replay = capture.events
        self.assertEqual(event["id"], replay["id"])
        self.assertEqual(event["feature"], "original")
        self.assertEqual(event["model"], "gpt-example")
        self.assertEqual(event["providerRequestId"], "req_resume")
        self.assertEqual(event["usage"], {"inputTokens": 10, "outputTokens": 4})
