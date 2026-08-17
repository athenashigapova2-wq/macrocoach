"""Persistence helpers for agent-run observability in Supabase."""

from datetime import datetime, timezone
from time import perf_counter
from typing import Any
from uuid import uuid4

from app.circuit_breaker import call_with_circuit_breaker
from app.config import settings
from app.model_routing import ModelSelection, model_name_for_tier
from app.services.supabase import get_supabase

MODEL_PROVIDER = "gigachat"


def _model_name(model_tier: str = "main") -> str:
    return model_name_for_tier(model_tier)


def elapsed_ms(started_at: float) -> int:
    """Return elapsed monotonic time in whole milliseconds."""
    return max(0, round((perf_counter() - started_at) * 1_000))


def _completed_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_agent_run(
    user_id: str,
    input_text: str,
    conversation_id: str | None = None,
) -> str:
    """Create a started run and return its database id."""
    response = (
        get_supabase()
        .table("agent_runs")
        .insert(
            {
                "user_id": user_id,
                "route": "general",
                "model_provider": MODEL_PROVIDER,
                "model_name": _model_name(),
                "input_text": input_text,
                "conversation_id": conversation_id,
                "status": "started",
                "resolution_mode": "main_llm",
                "baseline_version": settings.agent_baseline_version,
            }
        )
        .execute()
    )
    if not response.data:
        raise RuntimeError("Supabase не вернул созданный agent_run")
    return str(response.data[0]["id"])


def succeed_agent_run(
    run_id: str,
    user_id: str,
    route: str,
    output_text: str,
    latency_ms: int,
    resolution_mode: str = "main_llm",
) -> None:
    """Mark one user-owned run as successfully completed."""
    _update_owned_run(
        run_id,
        user_id,
        {
            "route": route,
            "output_text": output_text,
            "status": "succeeded",
            "latency_ms": latency_ms,
            "resolution_mode": resolution_mode,
            "completed_at": _completed_at(),
        },
    )


def fail_agent_run(
    run_id: str,
    user_id: str,
    error: Exception,
    latency_ms: int,
) -> None:
    """Mark one user-owned run as failed without storing a traceback."""
    error_message = f"{type(error).__name__}: {error}"[:1_000]
    _update_owned_run(
        run_id,
        user_id,
        {
            "status": "failed",
            "error_message": error_message,
            "latency_ms": latency_ms,
            "completed_at": _completed_at(),
        },
    )


def create_tool_call(
    run_id: str,
    tool_name: str,
    tool_args: dict[str, Any],
    tool_step: int = 1,
) -> str:
    """Create a started tool-call trace linked to its parent agent run."""
    response = (
        get_supabase()
        .table("agent_tool_calls")
        .insert(
            {
                "run_id": run_id,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "tool_step": tool_step,
                "status": "started",
            }
        )
        .execute()
    )
    if not response.data:
        raise RuntimeError("Supabase не вернул созданный agent_tool_call")
    return str(response.data[0]["id"])


def create_llm_call(
    run_id: str,
    node_name: str,
    purpose: str,
    model_tier: str,
    model_name: str | None = None,
    *,
    invocation_id: str | None = None,
    attempt_number: int = 1,
    model_selection: ModelSelection | None = None,
    retry_reason: str | None = None,
) -> str:
    """Create one row for one actual provider attempt."""
    selection_payload = _model_selection_payload(
        model_selection=model_selection,
        model_tier=model_tier,
        model_name=model_name,
    )
    response = (
        get_supabase()
        .table("agent_llm_calls")
        .insert(
            {
                "run_id": run_id,
                "node_name": node_name,
                "purpose": purpose,
                "model_provider": MODEL_PROVIDER,
                **selection_payload,
                "invocation_id": invocation_id or str(uuid4()),
                "attempt_number": attempt_number,
                "retry_reason": retry_reason,
                "status": "started",
            }
        )
        .execute()
    )
    if not response.data:
        raise RuntimeError("Supabase не вернул созданный agent_llm_call")
    return str(response.data[0]["id"])


def _model_selection_payload(
    *,
    model_selection: ModelSelection | None,
    model_tier: str,
    model_name: str | None,
) -> dict[str, Any]:
    if model_selection is not None:
        return {
            "model_name": model_selection.model_name,
            "model_tier": model_selection.model_tier,
            "requested_model_tier": model_selection.requested_model_tier,
            "routing_rule": model_selection.matched_rule,
            "selection_reason": model_selection.selection_reason,
            "is_fallback": model_selection.is_fallback,
            "fallback_reason": model_selection.fallback_reason,
        }
    return {
        "model_name": model_name or _model_name(model_tier),
        "model_tier": model_tier,
        "requested_model_tier": model_tier,
        "routing_rule": "legacy",
        "selection_reason": "model supplied without routing metadata",
        "is_fallback": False,
        "fallback_reason": None,
    }


def token_usage(message: Any) -> dict[str, int | bool]:
    """Normalize LangChain provider token metadata without guessing missing values."""
    usage = getattr(message, "usage_metadata", None) or {}
    response_metadata = getattr(message, "response_metadata", {})
    response_usage = response_metadata.get("token_usage") or response_metadata.get("usage") or {}
    input_tokens = int(
        usage.get("input_tokens")
        or response_usage.get("input_tokens")
        or response_usage.get("prompt_tokens")
        or 0
    )
    output_tokens = int(
        usage.get("output_tokens")
        or response_usage.get("output_tokens")
        or response_usage.get("completion_tokens")
        or 0
    )
    cached = int(
        usage.get("input_token_details", {}).get("cache_read")
        or response_usage.get("cached_tokens")
        or 0
    )
    total = int(usage.get("total_tokens") or response_usage.get("total_tokens") or input_tokens + output_tokens)
    return {
        "token_usage_available": bool(usage or response_usage),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached,
        "total_tokens": total,
    }


def succeed_llm_call(llm_call_id: str, run_id: str, message: Any, latency_ms: int) -> None:
    _update_run_llm_call(
        llm_call_id,
        run_id,
        {
            **token_usage(message),
            "status": "succeeded",
            "latency_ms": latency_ms,
            "completed_at": _completed_at(),
        },
    )


def fail_llm_call(llm_call_id: str, run_id: str, error: Exception, latency_ms: int) -> None:
    _update_run_llm_call(
        llm_call_id,
        run_id,
        {
            "status": "failed",
            "error_message": f"{type(error).__name__}: {error}"[:1_000],
            "latency_ms": latency_ms,
            "completed_at": _completed_at(),
        },
    )


def invoke_llm(
    llm: Any,
    messages: list[Any],
    *,
    run_id: str | None,
    node_name: str,
    purpose: str,
    model_tier: str,
    model_name: str | None = None,
    model_selection: ModelSelection | None = None,
) -> Any:
    """Invoke an LLM and persist every actual provider attempt."""
    invocation_id = str(uuid4())
    attempt_number = 0
    retry_reason: str | None = None

    def invoke_attempt() -> Any:
        nonlocal attempt_number, retry_reason
        attempt_number += 1
        if run_id is None:
            return llm.invoke(messages)

        llm_call_id = create_llm_call(
            run_id,
            node_name,
            purpose,
            model_tier,
            model_name=model_name,
            invocation_id=invocation_id,
            attempt_number=attempt_number,
            model_selection=model_selection,
            retry_reason=retry_reason,
        )
        started_at = perf_counter()
        try:
            message = llm.invoke(messages)
        except Exception as error:
            fail_llm_call(
                llm_call_id,
                run_id,
                error,
                elapsed_ms(started_at),
            )
            retry_reason = _retry_reason(error)
            raise
        succeed_llm_call(
            llm_call_id,
            run_id,
            message,
            elapsed_ms(started_at),
        )
        return message

    return call_with_circuit_breaker(
        invoke_attempt,
        circuit_name=MODEL_PROVIDER,
        operation_name=f"llm.{node_name}.{purpose}",
    )


def _retry_reason(error: BaseException) -> str:
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    if status is not None:
        return f"{type(error).__name__}:http_{status}"
    return type(error).__name__


def succeed_tool_call(
    tool_call_id: str,
    run_id: str,
    tool_result: Any,
    latency_ms: int,
) -> None:
    """Mark a tool call as succeeded and store its structured result."""
    _update_run_tool_call(
        tool_call_id,
        run_id,
        {
            "tool_result": tool_result,
            "status": "succeeded",
            "latency_ms": latency_ms,
            "completed_at": _completed_at(),
        },
    )


def fail_tool_call(
    tool_call_id: str,
    run_id: str,
    error: Exception,
    latency_ms: int,
) -> None:
    """Mark a tool call as failed without persisting a traceback."""
    error_message = f"{type(error).__name__}: {error}"[:1_000]
    _update_run_tool_call(
        tool_call_id,
        run_id,
        {
            "status": "failed",
            "error_message": error_message,
            "latency_ms": latency_ms,
            "completed_at": _completed_at(),
        },
    )


def _update_owned_run(run_id: str, user_id: str, values: dict[str, Any]) -> None:
    """Update by both id and user_id because the server client bypasses RLS."""
    (
        get_supabase()
        .table("agent_runs")
        .update(values)
        .eq("id", run_id)
        .eq("user_id", user_id)
        .execute()
    )


def _update_run_tool_call(
    tool_call_id: str,
    run_id: str,
    values: dict[str, Any],
) -> None:
    """Scope tool-call updates to both the call and its trusted parent run."""
    (
        get_supabase()
        .table("agent_tool_calls")
        .update(values)
        .eq("id", tool_call_id)
        .eq("run_id", run_id)
        .execute()
    )


def _update_run_llm_call(
    llm_call_id: str,
    run_id: str,
    values: dict[str, Any],
) -> None:
    (
        get_supabase()
        .table("agent_llm_calls")
        .update(values)
        .eq("id", llm_call_id)
        .eq("run_id", run_id)
        .execute()
    )
