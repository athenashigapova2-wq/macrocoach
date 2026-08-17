"""Offline checks for transient retries and read/write retry boundaries."""

import sys
from pathlib import Path
from unittest.mock import patch

import httpx
from langchain_core.tools import StructuredTool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.specialists import _invoke_tool  # noqa: E402
from app.resilience import is_transient_error, retry_transient  # noqa: E402
from app.services import agent_traces  # noqa: E402


def _state() -> dict:
    return {
        "user_id": "user-id",
        "run_id": None,
        "locale": "ru",
        "messages": [],
        "route": "nutrition",
    }


def assert_retry_schedule() -> None:
    attempts = 0

    def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ConnectError("temporary network failure")
        return "ok"

    with (
        patch("app.resilience.time.sleep") as sleep,
        patch("app.resilience.random.uniform", side_effect=[0.1, 0.2]),
    ):
        assert retry_transient(flaky, operation_name="test.read") == "ok"

    assert attempts == 3
    assert [call.args[0] for call in sleep.call_args_list] == [0.6, 1.2]


def assert_non_transient_failure_is_not_retried() -> None:
    attempts = 0

    def invalid() -> None:
        nonlocal attempts
        attempts += 1
        raise ValueError("invalid request")

    try:
        retry_transient(invalid, operation_name="test.invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("A permanent failure must propagate")
    assert attempts == 1


def assert_status_classification() -> None:
    request = httpx.Request("GET", "https://example.test")
    assert is_transient_error(
        httpx.HTTPStatusError(
            "rate limited",
            request=request,
            response=httpx.Response(429, request=request),
        )
    )
    assert not is_transient_error(
        httpx.HTTPStatusError(
            "unauthorized",
            request=request,
            response=httpx.Response(401, request=request),
        )
    )


def assert_llm_is_retried() -> None:
    class FlakyLLM:
        attempts = 0

        def invoke(self, messages):
            self.attempts += 1
            if self.attempts < 3:
                raise httpx.ReadTimeout("provider timeout")
            return "answer"

    llm = FlakyLLM()
    with patch("app.resilience.time.sleep"):
        assert agent_traces.invoke_llm(
            llm,
            ["hello"],
            run_id=None,
            node_name="router",
            purpose="test",
            model_tier="small",
        ) == "answer"
    assert llm.attempts == 3


def assert_only_read_tools_are_retried() -> None:
    read_attempts = 0

    def read_tool() -> dict:
        nonlocal read_attempts
        read_attempts += 1
        if read_attempts < 3:
            raise httpx.ConnectError("read failed")
        return {"status": "ok"}

    read = StructuredTool.from_function(
        func=read_tool,
        name="read_tool",
        description="Read test data.",
        metadata={"read_only": True},
    )
    call = {"id": "read-call", "name": "read_tool", "args": {}}
    with patch("app.resilience.time.sleep"):
        assert _invoke_tool(_state(), call, {read.name: read}) == {"status": "ok"}
    assert read_attempts == 3

    write_attempts = 0

    def write_tool() -> dict:
        nonlocal write_attempts
        write_attempts += 1
        raise httpx.ConnectError("write outcome is unknown")

    write = StructuredTool.from_function(
        func=write_tool,
        name="write_tool",
        description="Write test data.",
        metadata={"read_only": False},
    )
    call = {"id": "write-call", "name": "write_tool", "args": {}}
    try:
        _invoke_tool(_state(), call, {write.name: write})
    except httpx.ConnectError:
        pass
    else:
        raise AssertionError("A write failure must propagate without retry")
    assert write_attempts == 1


if __name__ == "__main__":
    assert_retry_schedule()
    assert_non_transient_failure_is_not_retried()
    assert_status_classification()
    assert_llm_is_retried()
    assert_only_read_tools_are_retried()
    print("Retry policy checks passed")
