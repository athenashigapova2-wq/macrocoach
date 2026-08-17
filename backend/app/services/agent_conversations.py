"""Persist user-owned chat history around one agent turn."""

from typing import Any

from app.resilience import retry_transient
from app.services.supabase import get_supabase


def prepare_conversation(
    user_id: str,
    conversation_id: str | None,
    message: str,
    locale: str,
) -> tuple[str, list[dict[str, str]]]:
    client = get_supabase()
    if conversation_id:
        query = (
            client.table("agent_conversations")
            .select("id")
            .eq("id", conversation_id)
            .eq("user_id", user_id)
            .limit(1)
        )
        response = retry_transient(
            query.execute,
            operation_name="supabase.read.agent_conversation",
        )
        if not response.data:
            raise ValueError("Conversation not found")
    else:
        response = (
            client.table("agent_conversations")
            .insert(
                {
                    "user_id": user_id,
                    "agent_name": "athena_multi_agent",
                    "title": message[:48],
                    "metadata": {"locale": locale},
                }
            )
            .execute()
        )
        if not response.data:
            raise RuntimeError("Supabase did not return the created conversation")
        conversation_id = str(response.data[0]["id"])

    history_query = (
        client.table("agent_messages")
        .select("role, content")
        .eq("conversation_id", conversation_id)
        .order("created_at", desc=True)
        .limit(20)
    )
    history_response = retry_transient(
        history_query.execute,
        operation_name="supabase.read.agent_messages",
    )
    rows: list[dict[str, Any]] = history_response.data or []
    history = [
        {"role": str(row["role"]), "content": str(row["content"])}
        for row in reversed(rows)
        if row.get("role") in {"user", "assistant"}
    ]
    return conversation_id, history


def save_turn(conversation_id: str, message: str, answer: str) -> None:
    response = (
        get_supabase()
        .table("agent_messages")
        .insert(
            [
                {"conversation_id": conversation_id, "role": "user", "content": message},
                {"conversation_id": conversation_id, "role": "assistant", "content": answer},
            ]
        )
        .execute()
    )
    if response.data is None:
        raise RuntimeError("Supabase did not persist the agent turn")
