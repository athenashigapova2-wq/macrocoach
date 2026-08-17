"""Offline checks for Supabase agent-run payloads and ownership filters."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.tools import StructuredTool
from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.specialists import _invoke_tool  # noqa: E402
from app.services import agent_traces  # noqa: E402


def main() -> None:
    query = Mock()
    query.table.return_value = query
    query.insert.return_value = query
    query.update.return_value = query
    query.eq.return_value = query
    query.execute.return_value = SimpleNamespace(data=[{"id": "run-id"}])

    with patch("app.services.agent_traces.get_supabase", return_value=query):
        run_id = agent_traces.create_agent_run("user-id", "Что я сегодня съела?")
        assert run_id == "run-id"
        insert_payload = query.insert.call_args.args[0]
        assert insert_payload["user_id"] == "user-id"
        assert insert_payload["status"] == "started"
        assert insert_payload["input_text"] == "Что я сегодня съела?"
        assert insert_payload["model_provider"] == "gigachat"
        assert insert_payload["baseline_version"] == "baseline-v1"
        assert insert_payload["resolution_mode"] == "main_llm"

        query.eq.reset_mock()
        agent_traces.succeed_agent_run(
            run_id="run-id",
            user_id="user-id",
            route="nutrition",
            output_text="Ответ",
            latency_ms=120,
        )

    filters = [call.args for call in query.eq.call_args_list]
    assert ("id", "run-id") in filters
    assert ("user_id", "user-id") in filters
    update_payload = query.update.call_args.args[0]
    assert update_payload["status"] == "succeeded"
    assert update_payload["route"] == "nutrition"
    assert update_payload["latency_ms"] == 120
    assert update_payload["resolution_mode"] == "main_llm"

    query.execute.return_value = SimpleNamespace(data=[{"id": "tool-call-id"}])
    with patch("app.services.agent_traces.get_supabase", return_value=query):
        tool_call_id = agent_traces.create_tool_call(
            run_id="run-id",
            tool_name="get_daily_intake",
            tool_args={"day": "2026-08-07"},
        )
        assert tool_call_id == "tool-call-id"
        tool_insert = query.insert.call_args.args[0]
        assert tool_insert["run_id"] == "run-id"
        assert tool_insert["tool_name"] == "get_daily_intake"
        assert tool_insert["status"] == "started"
        assert tool_insert["tool_step"] == 1

        query.eq.reset_mock()
        agent_traces.succeed_tool_call(
            tool_call_id="tool-call-id",
            run_id="run-id",
            tool_result={"calories": 1_500},
            latency_ms=35,
        )

    filters = [call.args for call in query.eq.call_args_list]
    assert ("id", "tool-call-id") in filters
    assert ("run_id", "run-id") in filters
    tool_update = query.update.call_args.args[0]
    assert tool_update["status"] == "succeeded"
    assert tool_update["tool_result"] == {"calories": 1_500}

    llm_message = AIMessage(
        content="Готово",
        usage_metadata={
            "input_tokens": 120,
            "output_tokens": 30,
            "total_tokens": 150,
            "input_token_details": {"cache_read": 20},
        },
    )
    query.execute.return_value = SimpleNamespace(data=[{"id": "llm-call-id"}])
    with patch("app.services.agent_traces.get_supabase", return_value=query):
        llm_call_id = agent_traces.create_llm_call(
            run_id="run-id",
            node_name="router",
            purpose="route_classification",
            model_tier="small",
        )
        assert llm_call_id == "llm-call-id"
        llm_insert = query.insert.call_args.args[0]
        assert llm_insert["node_name"] == "router"
        assert llm_insert["model_tier"] == "small"
        assert llm_insert["model_provider"] == "gigachat"

        query.eq.reset_mock()
        agent_traces.succeed_llm_call(
            llm_call_id="llm-call-id",
            run_id="run-id",
            message=llm_message,
            latency_ms=80,
        )

    llm_update = query.update.call_args.args[0]
    assert llm_update["token_usage_available"] is True
    assert llm_update["input_tokens"] == 120
    assert llm_update["output_tokens"] == 30
    assert llm_update["cached_input_tokens"] == 20
    assert llm_update["total_tokens"] == 150
    filters = [call.args for call in query.eq.call_args_list]
    assert ("id", "llm-call-id") in filters
    assert ("run_id", "run-id") in filters

    def echo_food(name: str) -> dict:
        return {"name": name}

    tool = StructuredTool.from_function(
        func=echo_food,
        name="echo_food",
        description="Return a food name for an offline trace test.",
    )
    state = {
        "user_id": "user-id",
        "run_id": "run-id",
        "locale": "ru",
        "messages": [],
        "route": "nutrition",
    }
    call = {"id": "llm-call-id", "name": "echo_food", "args": {"name": "рис"}}
    with (
        patch(
            "app.agents.specialists.agent_traces.create_tool_call",
            return_value="trace-id",
        ),
        patch("app.agents.specialists.agent_traces.succeed_tool_call") as succeed,
    ):
        result = _invoke_tool(state, call, {tool.name: tool})

    assert result == {"name": "рис"}
    succeed.assert_called_once()
    assert succeed.call_args.kwargs["tool_call_id"] == "trace-id"
    assert succeed.call_args.kwargs["run_id"] == "run-id"
    assert succeed.call_args.kwargs["tool_result"] == {"name": "рис"}

    def broken_tool(query: str) -> dict:
        raise RuntimeError(f"cannot process {query}")

    broken = StructuredTool.from_function(
        func=broken_tool,
        name="broken_tool",
        description="Fail for an offline trace test.",
    )
    failed_call = {
        "id": "failed-llm-call-id",
        "name": "broken_tool",
        "args": {"query": "test"},
    }
    with (
        patch(
            "app.agents.specialists.agent_traces.create_tool_call",
            return_value="failed-trace-id",
        ),
        patch("app.agents.specialists.agent_traces.fail_tool_call") as fail,
    ):
        try:
            _invoke_tool(state, failed_call, {broken.name: broken})
        except RuntimeError:
            pass
        else:
            raise AssertionError("A failing tool must propagate its exception")

    fail.assert_called_once()
    assert fail.call_args.kwargs["tool_call_id"] == "failed-trace-id"
    assert fail.call_args.kwargs["run_id"] == "run-id"
    print("Agent trace checks passed")


if __name__ == "__main__":
    main()
