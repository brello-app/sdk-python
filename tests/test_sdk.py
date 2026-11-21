import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import os
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from brello import Brello
from brello._transport import Reply, Transport, _request

KEY = "brl_test_" + "A" * 43


@contextmanager
def sdk_with_request(request, **options):
    def factory(*args, **kwargs):
        return Transport(*args, **kwargs, request=request)
    with patch("brello._client.Transport", side_effect=factory):
        yield Brello(KEY, environment="test", enabled=True, **options)


class Capture:
    def __init__(self, statuses=()):
        self.requests = []
        self.batches = []
        self.statuses = iter(statuses)

    def __call__(self, url, key, path, body, debug):
        self.requests.append((url, key, path))
        if path == "config":
            return Reply(200, data={"data": {
                "collection": {"eventsEnabled": True, "endUserIdEnabled": True, "attributesEnabled": True},
                "identity": {"endUser": {
                    "algorithm": "hmac-sha256", "keyVersion": "1", "secretEncoding": "base64url",
                    "domain": "brello:end-user:v1", "secret": "B" * 43,
                }},
            }})
        self.batches.append(json.loads(body)["events"])
        return Reply(next(self.statuses, 202))

    @property
    def events(self):
        return [event for batch in self.batches for event in batch]


class SdkTests(unittest.TestCase):
    def test_disabled_collection_survives_refresh_failure_until_explicitly_enabled(self):
        capture = Capture()
        settings = iter([
            Reply(200, data={"data": {"collection": {"eventsEnabled": False}}}),
            Reply(503),
            Reply(200, data={"data": {"collection": {}}}),
            Reply(200, data={"data": {"collection": {"eventsEnabled": True, "endUserIdEnabled": True}}}),
            Reply(503),
        ])

        def request(*args):
            return next(settings) if args[2] == "config" else capture(*args)

        with sdk_with_request(request) as sdk:
            client = sdk.wrap(SimpleNamespace(responses=SimpleNamespace(create=lambda **_: {"model": "test"})),
                              end_user={"id": "user"})
            for count in (0, 0, 0, 1, 2):
                sdk._transport._next_config_at = 0
                client.responses.create(model="test")
                sdk.flush()
                self.assertEqual(len(capture.events), count)

        self.assertEqual(capture.events[0]["endUser"], {"id": "user"})
        self.assertNotIn("endUser", capture.events[1])

    def test_http_transport_closes_failed_bodies_and_preserves_accepted_status(self):
        with patch("brello._transport.http.client.HTTPSConnection") as create:
            connection = create.return_value
            response = MagicMock()
            connection.getresponse.return_value.__enter__.return_value = response
            response.status = 503
            response.getheader.return_value = "10"
            self.assertEqual(
                _request("https://api.brello.app/v1/", KEY, "config", None, False),
                Reply(503, "10"),
            )
            response.read1.assert_not_called()
            connection.close.assert_called_once()
            self.assertEqual(connection.request.call_args.args[:2], ("GET", "/v1/config"))

            response.status = 202
            response.getheader.return_value = None
            response.read1.side_effect = TimeoutError
            self.assertEqual(_request("https://api.brello.app/v1/", KEY, "events", b"{}", True), Reply(202))
            self.assertEqual(connection.close.call_count, 2)

    def test_default_config_context_and_payload_privacy(self):
        capture = Capture()
        class Result:
            model = "example"
            usage = {"input_tokens": 14, "output_tokens": 7, "input_tokens_details": {"cached_tokens": 3}}
            _request_id = "req_example"
            @property
            def output(self):
                raise AssertionError("SDK must not inspect model output")

        with patch.dict(os.environ, {"BRELLO_API_BASE_URL": ""}):
            with sdk_with_request(capture, hash_end_user_id=True, service_name="support") as sdk:
                instance = Result()
                client = sdk.wrap(SimpleNamespace(responses=SimpleNamespace(create=lambda **_: instance)))
                with sdk.with_context(end_user={"id": "real-user", "name": "Jane"}, feature="summary"):
                    self.assertIs(client.responses.create(model="example", input="private prompt"), instance)
                sdk.flush()
        self.assertEqual(capture.requests, [
            ("https://api.brello.app/v1/", KEY, "config"), ("https://api.brello.app/v1/", KEY, "events"),
        ])
        event, = capture.events
        self.assertEqual(event["usage"], {"inputTokens": 14, "outputTokens": 7, "cacheReadInputTokens": 3})
        self.assertEqual(event["providerRequestId"], "req_example")
        self.assertEqual(event["feature"], "summary")
        self.assertEqual(event["attributes"], {"serviceName": "support"})
        self.assertRegex(event["endUser"]["idHash"], "^[a-f0-9]{64}$")
        self.assertNotIn("real-user", str(event))
        self.assertNotIn("private prompt", str(event))

    def test_anthropic_stream_counts_cumulative_usage_once_and_retries_with_same_id(self):
        capture = Capture([429, 202])
        chunks = [
            {"type": "message_start", "message": {"model": "claude-example", "usage": {
                "input_tokens": 10, "output_tokens": 1, "cache_read_input_tokens": 8,
                "cache_creation_input_tokens": 5,
                "cache_creation": {"ephemeral_5m_input_tokens": 3, "ephemeral_1h_input_tokens": 2},
            }}},
            {"type": "message_delta", "usage": {"output_tokens": 4}},
            {"type": "message_delta", "usage": {"output_tokens": 6}},
            {"type": "message_stop"},
        ]

        class Manager:
            def __enter__(self):
                return iter(chunks)

            def __exit__(self, *_):
                return None

        with sdk_with_request(capture, api_base_url="https://private.example") as sdk:
            client = sdk.wrap(SimpleNamespace(messages=SimpleNamespace(stream=lambda **_: Manager())))
            with client.messages.stream(model="claude-example") as stream:
                self.assertEqual(list(stream), chunks)
            sdk.flush()
        first, second = capture.events
        self.assertEqual(first, second)
        self.assertEqual(first["usage"], {
            "inputTokens": 23, "outputTokens": 6, "cacheReadInputTokens": 8,
            "cacheCreationInputTokens": {"fiveMinutes": 3, "oneHour": 2},
        })
        self.assertEqual(capture.requests[0][0], "https://private.example/v1/")

    def test_raw_response_context_preserves_the_response_without_reading_body(self):
        capture = Capture()
        raw = SimpleNamespace(headers={"x-request-id": "req_raw"})

        class Manager:
            def __enter__(self):
                return raw

            def __exit__(self, *_):
                return None

        wrapped = SimpleNamespace(with_streaming_response=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: Manager()))
        ))
        with sdk_with_request(capture) as sdk:
            client = sdk.wrap(wrapped)
            with client.with_streaming_response.chat.completions.create(model="gpt-example") as response:
                self.assertIs(response, raw)
            sdk.flush()

        row, = capture.events
        self.assertIsNone(row["usage"])
        self.assertEqual(row["providerRequestId"], "req_raw")

    def test_circuit_opens_and_flush_does_not_wait_for_the_cooldown(self):
        capture = Capture([503] * 3)
        with sdk_with_request(capture) as sdk:
            client = sdk.wrap(SimpleNamespace(responses=SimpleNamespace(create=lambda **_: {"model": "example"})))
            client.responses.create(model="example")
            started = time.monotonic()
            sdk.flush(4)
            self.assertLess(time.monotonic() - started, 3.5)
            self.assertEqual(len(capture.events), 3)
            self.assertEqual(len({item["id"] for item in capture.events}), 1)
            self.assertEqual(client.responses.create(model="example"), {"model": "example"})
            sdk.flush(0.1)
            self.assertEqual(len(capture.events), 3)

    def test_auth_failure_keeps_original_response_and_stops_delivery(self):
        received = []
        def reject(_url, _key, path, _body, _debug):
            received.append(path)
            return Reply(401)
        with sdk_with_request(reject) as sdk:
            instance = {"model": "example"}
            client = sdk.wrap(SimpleNamespace(responses=SimpleNamespace(create=lambda **_: instance)))
            self.assertIs(client.responses.create(model="example"), instance)
            sdk.flush()
            self.assertIs(client.responses.create(model="example"), instance)
            sdk.flush()
        self.assertEqual(received, ["config"])

    def test_concurrent_flush_remains_bounded_during_stalled_delivery(self):
        capture = Capture()
        release = threading.Event()
        entered = threading.Event()
        def request(*args):
            if args[2] == "events" and not release.is_set():
                entered.set()
                release.wait(2)
            return capture(*args)

        with sdk_with_request(request) as sdk:
            client = sdk.wrap(SimpleNamespace(responses=SimpleNamespace(create=lambda **_: {"model": "example"})))
            client.responses.create(model="example")
            try:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    flushes = [executor.submit(sdk.flush, 0.05) for _ in range(2)]
                    self.assertTrue(entered.wait(0.5))
                    for flush in flushes:
                        flush.result(timeout=0.5)
                started = time.monotonic()
                client.responses.create(model="example")
                self.assertLess(time.monotonic() - started, 0.25)
            finally:
                release.set()
            sdk.flush()
        self.assertEqual(len(capture.events), 2)


class AsyncSdkTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_context_isolated_and_stream_errors_keep_the_original_exception(self):
        capture = Capture()
        failure = RuntimeError("private provider error")

        async def fail_stream():
            yield {"model": "gpt-test", "choices": []}
            raise failure

        async def create(**kwargs):
            await asyncio.sleep(0)
            if kwargs.get("stream"):
                return fail_stream()
            return {"model": "gpt-test"}

        with sdk_with_request(capture) as sdk:
            client = sdk.wrap(SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
            async def call(label):
                with sdk.with_context(feature=label):
                    await client.chat.completions.create(model="gpt-test")
            await asyncio.gather(call("first"), call("second"))
            stream = await client.chat.completions.create(model="gpt-test", stream=True)
            self.assertEqual((await anext(stream))["model"], "gpt-test")
            with self.assertRaises(RuntimeError) as caught:
                await anext(stream)
            self.assertIs(caught.exception, failure)
            await sdk.aflush()
        self.assertEqual({event.get("feature") for event in capture.events}, {"first", "second", None})
        self.assertIn("error", [event["status"] for event in capture.events])
        self.assertNotIn("private provider error", str(capture.events))
