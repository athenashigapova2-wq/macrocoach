"""Redis-backed circuit breaker shared by all API and worker processes."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, TypeVar
from uuid import uuid4

from redis import Redis
from redis.exceptions import RedisError

from app.config import settings
from app.resilience import is_transient_error, retry_transient

logger = logging.getLogger(__name__)

T = TypeVar("T")
CircuitState = Literal["closed", "open", "half_open", "bypassed"]
_KEY_PREFIX = "athena:circuit-breaker:"

# Redis TIME keeps all workers on one clock. The state transition and probe lease
# are handled in one script so only one worker can enter half-open at a time.
_ACQUIRE_SCRIPT = r"""
local key = KEYS[1]
local token = ARGV[1]
local recovery_ms = tonumber(ARGV[2])
local lease_ms = tonumber(ARGV[3])
local ttl_ms = tonumber(ARGV[4])
local redis_time = redis.call("TIME")
local now_ms = tonumber(redis_time[1]) * 1000 + math.floor(tonumber(redis_time[2]) / 1000)
local state = redis.call("HGET", key, "state")

if not state or state == "closed" then
    return {1, "closed", 0}
end

if state == "open" then
    local opened_at_ms = tonumber(redis.call("HGET", key, "opened_at_ms") or "0")
    local remaining_ms = recovery_ms - (now_ms - opened_at_ms)
    if remaining_ms > 0 then
        return {0, "open", remaining_ms}
    end
    redis.call(
        "HSET", key,
        "state", "half_open",
        "probe_token", token,
        "probe_expires_at_ms", now_ms + lease_ms
    )
    redis.call("PEXPIRE", key, ttl_ms)
    return {1, "half_open", 0}
end

if state == "half_open" then
    local probe_expires_at_ms = tonumber(redis.call("HGET", key, "probe_expires_at_ms") or "0")
    if probe_expires_at_ms <= now_ms then
        redis.call(
            "HSET", key,
            "probe_token", token,
            "probe_expires_at_ms", now_ms + lease_ms
        )
        redis.call("PEXPIRE", key, ttl_ms)
        return {1, "half_open", 0}
    end
    return {0, "half_open", probe_expires_at_ms - now_ms}
end

return {1, "closed", 0}
"""

_SUCCESS_SCRIPT = r"""
local key = KEYS[1]
local permit_state = ARGV[1]
local token = ARGV[2]
local state = redis.call("HGET", key, "state")

if not state then
    return "closed"
end

if permit_state == "half_open" then
    local current_token = redis.call("HGET", key, "probe_token")
    if state == "half_open" and current_token == token then
        redis.call("DEL", key)
        return "closed"
    end
    return state
end

-- A late success from a request admitted while closed must not close a circuit
-- that another concurrent request has already opened.
if state == "closed" then
    redis.call("DEL", key)
    return "closed"
end

return state
"""

_FAILURE_SCRIPT = r"""
local key = KEYS[1]
local threshold = tonumber(ARGV[1])
local ttl_ms = tonumber(ARGV[2])
local permit_state = ARGV[3]
local token = ARGV[4]
local redis_time = redis.call("TIME")
local now_ms = tonumber(redis_time[1]) * 1000 + math.floor(tonumber(redis_time[2]) / 1000)
local state = redis.call("HGET", key, "state") or "closed"

if permit_state == "half_open" then
    local current_token = redis.call("HGET", key, "probe_token")
    if state == "half_open" and current_token == token then
        redis.call(
            "HSET", key,
            "state", "open",
            "failures", threshold,
            "opened_at_ms", now_ms
        )
        redis.call("HDEL", key, "probe_token", "probe_expires_at_ms")
        redis.call("PEXPIRE", key, ttl_ms)
        return {"open", threshold}
    end
    return {state, tonumber(redis.call("HGET", key, "failures") or "0")}
end

-- Ignore late failures once another request has already opened the circuit.
if state ~= "closed" then
    return {state, tonumber(redis.call("HGET", key, "failures") or "0")}
end

local failures = redis.call("HINCRBY", key, "failures", 1)
redis.call("HSET", key, "state", "closed")
if failures >= threshold then
    state = "open"
    redis.call("HSET", key, "state", state, "opened_at_ms", now_ms)
    redis.call("HDEL", key, "probe_token", "probe_expires_at_ms")
end
redis.call("PEXPIRE", key, ttl_ms)
return {state, failures}
"""


class CircuitOpenError(RuntimeError):
    """Raised without contacting the provider while its circuit is open."""

    def __init__(self, name: str, retry_after_seconds: float) -> None:
        self.name = name
        self.retry_after_seconds = max(0.0, retry_after_seconds)
        super().__init__(
            f"Circuit {name!r} is open; retry after "
            f"{self.retry_after_seconds:.2f}s"
        )


@dataclass(frozen=True)
class CircuitPermit:
    """Permission returned before a provider call."""

    name: str
    state: CircuitState
    token: str = ""


@lru_cache(maxsize=1)
def redis_client() -> Redis:
    return Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=2,
        health_check_interval=30,
    )


def _key(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9:_-]+", "-", name.strip().lower()).strip("-")
    if not normalized:
        raise ValueError("Circuit breaker name must not be empty")
    return f"{_KEY_PREFIX}{normalized}"


def _state_ttl_ms() -> int:
    minimum = (
        settings.llm_circuit_breaker_recovery_timeout_seconds
        + settings.llm_circuit_breaker_half_open_lease_seconds
    )
    return int(max(settings.llm_circuit_breaker_state_ttl_seconds, minimum) * 1_000)


def acquire_circuit(name: str) -> CircuitPermit:
    """Atomically admit a normal call or the single half-open probe."""
    if not settings.llm_circuit_breaker_enabled:
        return CircuitPermit(name=name, state="bypassed")

    token = uuid4().hex
    try:
        result = redis_client().eval(
            _ACQUIRE_SCRIPT,
            1,
            _key(name),
            token,
            int(settings.llm_circuit_breaker_recovery_timeout_seconds * 1_000),
            int(settings.llm_circuit_breaker_half_open_lease_seconds * 1_000),
            _state_ttl_ms(),
        )
    except RedisError as error:
        logger.warning(
            "Circuit breaker Redis read failed; allowing %s call: %s",
            name,
            type(error).__name__,
        )
        return CircuitPermit(name=name, state="bypassed")

    allowed = bool(int(result[0]))
    state = str(result[1])
    retry_after_seconds = float(result[2]) / 1_000
    if not allowed:
        raise CircuitOpenError(name, retry_after_seconds)
    return CircuitPermit(
        name=name,
        state="half_open" if state == "half_open" else "closed",
        token=token if state == "half_open" else "",
    )


def record_circuit_success(permit: CircuitPermit) -> None:
    """Reset closed-state failures or close a successful half-open probe."""
    if permit.state == "bypassed":
        return
    try:
        state = redis_client().eval(
            _SUCCESS_SCRIPT,
            1,
            _key(permit.name),
            permit.state,
            permit.token,
        )
    except RedisError as error:
        logger.warning(
            "Circuit breaker Redis success update failed for %s: %s",
            permit.name,
            type(error).__name__,
        )
        return
    if permit.state == "half_open" and state == "closed":
        logger.info("Circuit breaker %s closed after successful probe", permit.name)


def record_circuit_failure(permit: CircuitPermit) -> None:
    """Record one logical transient failure after retries are exhausted."""
    if permit.state == "bypassed":
        return
    try:
        state, failures = redis_client().eval(
            _FAILURE_SCRIPT,
            1,
            _key(permit.name),
            settings.llm_circuit_breaker_failure_threshold,
            _state_ttl_ms(),
            permit.state,
            permit.token,
        )
    except RedisError as error:
        logger.warning(
            "Circuit breaker Redis failure update failed for %s: %s",
            permit.name,
            type(error).__name__,
        )
        return
    if state == "open":
        logger.warning(
            "Circuit breaker %s is open after %s transient failures",
            permit.name,
            failures,
        )


def call_with_circuit_breaker(
    operation: Callable[[], T],
    *,
    circuit_name: str,
    operation_name: str,
) -> T:
    """Run one safe operation with retries behind a shared circuit breaker."""
    permit = acquire_circuit(circuit_name)
    try:
        result = retry_transient(operation, operation_name=operation_name)
    except Exception as error:
        if is_transient_error(error):
            record_circuit_failure(permit)
        else:
            # A non-transient response proves provider reachability and must not
            # count as an availability failure. It also releases a half-open probe.
            record_circuit_success(permit)
        raise
    record_circuit_success(permit)
    return result
