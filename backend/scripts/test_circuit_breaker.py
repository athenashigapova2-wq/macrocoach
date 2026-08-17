"""Offline checks for the shared Redis-backed LLM circuit breaker."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
from redis.exceptions import RedisError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.circuit_breaker import (  # noqa: E402
    CircuitOpenError,
    _ACQUIRE_SCRIPT,
    _FAILURE_SCRIPT,
    call_with_circuit_breaker,
)
from app.config import settings  # noqa: E402
from app.resilience import is_transient_error  # noqa: E402


class FakeRedis:
    def __init__(self, *results) -> None:
        self.results = list(results)
        self.eval_calls: list[tuple[str, int, tuple]] = []

    def eval(self, script: str, numkeys: int, *args):
        self.eval_calls.append((script, numkeys, args))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def check_open_circuit_short_circuits() -> None:
    client = FakeRedis([0, "open", 2_500])
    operation = Mock(return_value="must not run")
    with patch("app.circuit_breaker.redis_client", return_value=client):
        try:
            call_with_circuit_breaker(
                operation,
                circuit_name="gigachat",
                operation_name="llm.test",
            )
        except CircuitOpenError as error:
            assert error.retry_after_seconds == 2.5
            assert not is_transient_error(error)
        else:
            raise AssertionError("An open circuit must reject the provider call")
    operation.assert_not_called()


def check_transient_failure_is_recorded_after_retries() -> None:
    client = FakeRedis([1, "closed", 0], ["closed", 1])
    operation = Mock(side_effect=httpx.ConnectError("provider unavailable"))
    with (
        patch("app.circuit_breaker.redis_client", return_value=client),
        patch.object(settings, "safe_retry_max_attempts", 1),
    ):
        try:
            call_with_circuit_breaker(
                operation,
                circuit_name="gigachat",
                operation_name="llm.test",
            )
        except httpx.ConnectError:
            pass
        else:
            raise AssertionError("The provider error must propagate")
    operation.assert_called_once()
    assert len(client.eval_calls) == 2
    assert client.eval_calls[1][0] == _FAILURE_SCRIPT


def check_half_open_allows_one_successful_probe() -> None:
    client = FakeRedis([1, "half_open", 0], "closed")
    with patch("app.circuit_breaker.redis_client", return_value=client):
        result = call_with_circuit_breaker(
            lambda: "answer",
            circuit_name="gigachat",
            operation_name="llm.test",
        )
    assert result == "answer"
    acquire_token = client.eval_calls[0][2][1]
    success_state = client.eval_calls[1][2][1]
    success_token = client.eval_calls[1][2][2]
    assert acquire_token
    assert success_state == "half_open"
    assert success_token == acquire_token


def check_non_transient_probe_releases_breaker() -> None:
    client = FakeRedis([1, "half_open", 0], "closed")
    with patch("app.circuit_breaker.redis_client", return_value=client):
        try:
            call_with_circuit_breaker(
                lambda: (_ for _ in ()).throw(ValueError("invalid request")),
                circuit_name="gigachat",
                operation_name="llm.test",
            )
        except ValueError:
            pass
        else:
            raise AssertionError("The non-transient error must propagate")
    assert len(client.eval_calls) == 2
    assert client.eval_calls[1][2][1] == "half_open"


def check_redis_failure_is_fail_open() -> None:
    client = FakeRedis(RedisError("redis unavailable"))
    operation = Mock(return_value="answer")
    with patch("app.circuit_breaker.redis_client", return_value=client):
        assert call_with_circuit_breaker(
            operation,
            circuit_name="gigachat",
            operation_name="llm.test",
        ) == "answer"
    operation.assert_called_once()
    assert len(client.eval_calls) == 1


def check_atomic_scripts_use_redis_time() -> None:
    assert 'redis.call("TIME")' in _ACQUIRE_SCRIPT
    assert '"state", "half_open"' in _ACQUIRE_SCRIPT
    assert 'redis.call("HINCRBY"' in _FAILURE_SCRIPT
    assert '"state", "open"' in _FAILURE_SCRIPT


if __name__ == "__main__":
    with patch.object(settings, "llm_circuit_breaker_enabled", True):
        check_open_circuit_short_circuits()
        check_transient_failure_is_recorded_after_retries()
        check_half_open_allows_one_successful_probe()
        check_non_transient_probe_releases_breaker()
        check_redis_failure_is_fail_open()
        check_atomic_scripts_use_redis_time()
    print("Circuit breaker checks passed")
