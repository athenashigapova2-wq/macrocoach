"""Offline checks for agent service extraction and the Celery task boundary."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.agent_chat import run_agent_chat  # noqa: E402
from app.services.agent_jobs import get_agent_job  # noqa: E402
from app.workers.tasks import run_agent_chat_task  # noqa: E402


def check_agent_service() -> None:
    with (
        patch(
            "app.services.agent_chat.agent_conversations.prepare_conversation",
            return_value=("conversation-id", []),
        ),
        patch("app.services.agent_chat.agent_conversations.save_turn") as save_turn,
        patch("app.services.agent_chat.agent_traces.create_agent_run", return_value="run-id"),
        patch("app.services.agent_chat.agent_traces.succeed_agent_run") as succeed_run,
        patch("app.services.agent_chat.agent_graph.run_agent_turn_details") as run_turn,
    ):
        run_turn.return_value = {
            "answer": "Done",
            "route": "nutrition",
            "resolution_mode": "main_llm",
        }
        result = run_agent_chat(
            user_id="user-id",
            message="Add breakfast",
            locale="en",
            conversation_id=None,
        )
    assert result == {
        "answer": "Done",
        "route": "nutrition",
        "conversation_id": "conversation-id",
    }
    save_turn.assert_called_once_with("conversation-id", "Add breakfast", "Done")
    succeed_run.assert_called_once()


def check_worker_task() -> None:
    result = {"answer": "Done", "route": "general", "conversation_id": "conversation-id"}
    with (
        patch("app.workers.tasks.mark_job_running") as running,
        patch("app.workers.tasks.run_agent_chat", return_value=result),
        patch("app.workers.tasks.mark_job_succeeded") as succeeded,
    ):
        run_agent_chat_task.run(
            job_id="job-id",
            user_id="user-id",
            message="Hello",
            locale="en",
            conversation_id=None,
        )
    running.assert_called_once_with("job-id")
    succeeded.assert_called_once_with("job-id", result)

    with (
        patch("app.workers.tasks.mark_job_running"),
        patch("app.workers.tasks.run_agent_chat", side_effect=RuntimeError("offline")),
        patch("app.workers.tasks.mark_job_failed") as failed,
        patch("app.workers.tasks.logger.exception"),
    ):
        try:
            run_agent_chat_task.run(
                job_id="failed-job-id",
                user_id="user-id",
                message="Hello",
                locale="en",
                conversation_id=None,
            )
        except RuntimeError as exc:
            assert str(exc) == "offline"
        else:
            raise AssertionError("A failed worker task must remain failed in Celery")
    failed.assert_called_once_with("failed-job-id", "Агент временно недоступен")


def check_job_ownership() -> None:
    client = MagicMock()
    client.hgetall.return_value = {
        "user_id": "owner-id",
        "status": "succeeded",
        "result": (
            '{"answer":"Done","route":"general",'
            '"conversation_id":"conversation-id"}'
        ),
    }
    with patch("app.services.agent_jobs.redis_client", return_value=client):
        assert get_agent_job("job-id", "other-user") is None
        owned = get_agent_job("job-id", "owner-id")
    assert owned == {
        "job_id": "job-id",
        "status": "succeeded",
        "answer": "Done",
        "route": "general",
        "conversation_id": "conversation-id",
    }


def main() -> None:
    check_agent_service()
    check_worker_task()
    check_job_ownership()
    print("Agent worker checks passed")


if __name__ == "__main__":
    main()
