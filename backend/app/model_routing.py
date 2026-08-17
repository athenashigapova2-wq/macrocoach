"""Configurable GigaChat-only model routing policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.config import settings

ModelTier = Literal["small", "main"]


@dataclass(frozen=True)
class ModelSelection:
    """Auditable result of one model-routing decision."""

    provider: Literal["gigachat"]
    requested_model_tier: ModelTier
    model_tier: ModelTier
    model_name: str
    matched_rule: str
    selection_reason: str
    is_fallback: bool
    fallback_reason: str | None


def model_name_for_tier(model_tier: str) -> str:
    """Resolve a configured tier to an actual GigaChat model name."""
    if model_tier == "small":
        return settings.llm_router_model.strip() or settings.gigachat_model
    if model_tier == "main":
        return settings.gigachat_model
    raise ValueError(f"Unsupported model tier: {model_tier}")


def select_model(
    *,
    node_name: str,
    purpose: str,
    default_tier: ModelTier = "main",
) -> ModelSelection:
    """Select a model using exact, node, purpose, and global rules in order."""
    node = node_name.strip().lower()
    operation = purpose.strip().lower()
    if not node or not operation:
        raise ValueError("node_name and purpose must not be empty")

    requested_tier = default_tier
    matched_rule = "default"
    if settings.llm_model_routing_enabled:
        policy = settings.llm_model_routing_policy
        candidates = (
            f"{node}.{operation}",
            f"{node}.*",
            f"*.{operation}",
            "*",
        )
        for rule in candidates:
            configured_tier = policy.get(rule)
            if configured_tier is not None:
                requested_tier = configured_tier
                matched_rule = rule
                break

    if matched_rule != "default":
        selection_reason = (
            f"matched routing rule {matched_rule!r}, selecting tier "
            f"{requested_tier!r}"
        )
    elif settings.llm_model_routing_enabled:
        selection_reason = (
            f"no routing rule matched; using call default tier {requested_tier!r}"
        )
    else:
        selection_reason = (
            f"model routing disabled; using call default tier {requested_tier!r}"
        )

    actual_tier = requested_tier
    is_fallback = False
    fallback_reason = None
    if requested_tier == "small" and not settings.llm_router_model.strip():
        actual_tier = "main"
        is_fallback = True
        fallback_reason = (
            "LLM_ROUTER_MODEL is empty; using GIGACHAT_MODEL for requested "
            "small tier"
        )

    return ModelSelection(
        provider="gigachat",
        requested_model_tier=requested_tier,
        model_tier=actual_tier,
        model_name=model_name_for_tier(actual_tier),
        matched_rule=matched_rule,
        selection_reason=selection_reason,
        is_fallback=is_fallback,
        fallback_reason=fallback_reason,
    )
