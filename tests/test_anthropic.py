from types import SimpleNamespace as NS
import unittest

from test_sdk import Capture, sdk_with_request


class AnthropicTests(unittest.IsolatedAsyncioTestCase):
    async def test_structured_and_beta_calls_record_actual_billing_metadata(self):
        def message(tier, speed):
            return NS(model="claude-example", usage=NS(input_tokens=10, output_tokens=3,
                                                      service_tier=tier, speed=speed))

        priority = message("priority", "standard")
        fast = message("standard", "fast")
        standard = message("standard", "standard")

        async def create(**kwargs):
            return fast

        capture = Capture()
        with sdk_with_request(capture) as sdk:
            client = sdk.wrap(NS(messages=NS(parse=lambda **kwargs: priority),
                                 beta=NS(messages=NS(create=create, parse=lambda **kwargs: standard))))
            self.assertIs(client.messages.parse(model="claude-example"), priority)
            self.assertIs(await client.beta.messages.create(model="claude-example"), fast)
            self.assertIs(client.beta.messages.parse(model="claude-example", speed="fast"), standard)
            await sdk.aflush()

        self.assertEqual([event["billing"] for event in capture.events], [
            {"serviceTier": "priority"}, {"serviceTier": "standard", "mode": "unknown"},
            {"serviceTier": "standard"},
        ])
        self.assertEqual([event["usage"] for event in capture.events], [
            {"inputTokens": 10, "outputTokens": 3},
        ] * 3)

    async def test_text_and_final_message_helpers_require_a_terminal_stream_event(self):
        message = NS(model="claude-example", usage=NS(input_tokens=12, output_tokens=1))
        start = NS(type="message_start", message=message)
        text = NS(type="content_block_delta", delta=NS(type="text_delta", text="partial"))
        delta = NS(type="message_delta", usage=NS(output_tokens=5))
        stop = NS(type="message_stop")

        class Stream:
            response = NS(headers={"request-id": "req_anthropic"})

            def __init__(self, completed):
                self.events = iter([start, text, delta, stop] if completed else [start, text, delta])

            async def __aiter__(self):
                for event in self.events:
                    yield event

            @property
            def text_stream(self):
                async def iterate():
                    async for event in self:
                        if event.type == "content_block_delta":
                            yield event.delta.text
                return iterate()

            async def get_final_message(self):
                async for _ in self:
                    pass
                return message

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        capture = Capture()
        with sdk_with_request(capture) as sdk:
            client = sdk.wrap(NS(beta=NS(messages=NS(stream=lambda *, completed, **kwargs: Stream(completed)))))
            async with client.beta.messages.stream(model="claude-example", completed=False) as stream:
                self.assertEqual([value async for value in stream.text_stream], ["partial"])
                self.assertIs(await stream.get_final_message(), message)
            async with client.beta.messages.stream(model="claude-example", completed=True) as stream:
                self.assertIs(await stream.get_final_message(), message)
            await sdk.aflush()

        error, success = capture.events
        self.assertEqual((error["status"], error["usage"]), ("error", None))
        self.assertEqual(success["status"], "success")
        self.assertEqual(success["usage"], {"inputTokens": 12, "outputTokens": 5})
        self.assertEqual({event["providerRequestId"] for event in capture.events}, {"req_anthropic"})
