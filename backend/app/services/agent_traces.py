"""Persistence helpers for agent-run observability in Supabase."""

from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from app.config import settings
from app.resilience import retry_transient
from app.services.supabase import get_supabase

MODEL_PROVIDER = "gigachat"


def _model_name(model_tier: str = "main") -> str:
    if model_tier == "small" and settings.llm_router_model:
        return settings.llm_router_model
    return settings.gigachat_model


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
) -> str:
    """Create one provider call so token accounting is auditable per invocation."""
    response = (
        get_supabase()
        .table("agent_llm_calls")
        .insert(
            {
                "run_id": run_id,
                "node_name": node_name,
                "purpose": purpose,
                "model_provider": MODEL_PROVIDER,
                "model_name": _model_name(model_tier),
                "model_tier": model_tier,
                "status": "started",
            }
        )
        .execute()
    )
    if not response.data:
        raise RuntimeError("Supabase не вернул созданный agent_llm_call")
    return str(response.data[0]["id"])


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
) -> Any:
    """Invoke an LLM and persist its lifecycle when a traced run is available."""
    invoke = lambda: retry_transient(
        lambda: llm.invoke(messages),
        operation_name=f"llm.{node_name}.{purpose}",
    )
    if run_id is None:
        return invoke()
    llm_call_id = create_llm_call(run_id, node_name, purpose, model_tier)
    started_at = perf_counter()
    try:
        message = invoke()
    except Exception as exc:
        fail_llm_call(llm_call_id, run_id, exc, elapsed_ms(started_at))
        raise
    succeed_llm_call(llm_call_id, run_id, message, elapsed_ms(started_at))
    return message


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
