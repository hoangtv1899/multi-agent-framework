"""No test reaches the model gateway unless asked (conftest.no_model_gateway).

    pytest tests/test_no_gateway.py -v

A step that is given no scripted client makes a real one. With the gateway's
key in the environment its call would be real and paid, and a test could pass
all the same. The root conftest puts a stand-in in the place of the gateway's
client for every test of a plain run: it can be built, as the agents build
theirs, and a request through it fails the test.
"""
import pytest


def test_a_request_through_a_real_client_fails_and_is_counted(no_model_gateway, monkeypatch):
    if no_model_gateway is None:
        pytest.skip("the guard is off: --runllm was given, or openai is not installed")
    monkeypatch.setenv("PNNL_API_KEY", "not-a-key")
    from agents.llm_agent import SimpleLLMClient
    client = SimpleLLMClient(model="a-model")          # built as the agents build theirs
    with pytest.raises(AssertionError, match="asked the real model gateway"):
        client.ask([{"role": "user", "content": "hello"}])
    assert no_model_gateway == ["a-model"]
    no_model_gateway.clear()                           # this test meant to reach it


def test_a_scripted_client_is_left_alone(no_model_gateway):
    class Scripted:
        def ask(self, messages, **kw):
            return "a reply"
    assert Scripted().ask([]) == "a reply"
    assert not no_model_gateway
