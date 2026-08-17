"""Offline checks for configurable GigaChat model-routing policy."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings, settings  # noqa: E402
from app.llm import get_routed_llm  # noqa: E402
from app.model_routing import select_model  # noqa: E402


def check_rule_precedence() -> None:
    policy = {
        "nutrition.food_translation": "small",
        "nutrition.*": "main",
        "*.answer": "small",
        "*": "main",
    }
    with (
        patch.object(settings, "llm_model_routing_enabled", True),
        patch.object(settings, "llm_model_routing_policy", policy),
        patch.object(settings, "llm_router_model", "GigaChat-Lite"),
        patch.object(settings, "gigachat_model", "GigaChat-Main"),
    ):
        exact = select_model(
            node_name="nutrition",
            purpose="food_translation",
        )
        assert exact.model_tier == "small"
        assert exact.requested_model_tier == "small"
        assert exact.model_name == "GigaChat-Lite"
        assert exact.matched_rule == "nutrition.food_translation"
        assert "matched routing rule" in exact.selection_reason
        assert exact.is_fallback is False
        assert exact.fallback_reason is None

        node_wildcard = select_model(
            node_name="nutrition",
            purpose="tool_planning_or_answer",
        )
        assert node_wildcard.model_tier == "main"
        assert node_wildcard.matched_rule == "nutrition.*"

        purpose_wildcard = select_model(
            node_name="general",
            purpose="answer",
        )
        assert purpose_wildcard.model_tier == "small"
        assert purpose_wildcard.matched_rule == "*.answer"

        global_default = select_model(
            node_name="general",
            purpose="summarize",
        )
        assert global_default.model_tier == "main"
        assert global_default.matched_rule == "*"


def check_disabled_policy_uses_call_default() -> None:
    with (
        patch.object(settings, "llm_model_routing_enabled", False),
        patch.object(settings, "llm_model_routing_policy", {"*": "main"}),
        patch.object(settings, "llm_router_model", "GigaChat-Lite"),
    ):
        selection = select_model(
            node_name="router",
            purpose="route_classification",
            default_tier="small",
        )
    assert selection.model_tier == "small"
    assert selection.requested_model_tier == "small"
    assert selection.model_name == "GigaChat-Lite"
    assert selection.matched_rule == "default"


def check_small_tier_falls_back_to_main_model() -> None:
    with (
        patch.object(settings, "llm_model_routing_enabled", True),
        patch.object(settings, "llm_model_routing_policy", {"*": "small"}),
        patch.object(settings, "llm_router_model", ""),
        patch.object(settings, "gigachat_model", "GigaChat-Main"),
    ):
        selection = select_model(node_name="router", purpose="test")
    assert selection.requested_model_tier == "small"
    assert selection.model_tier == "main"
    assert selection.model_name == "GigaChat-Main"
    assert selection.is_fallback is True
    assert "LLM_ROUTER_MODEL is empty" in str(selection.fallback_reason)


def check_routed_client_uses_selected_model() -> None:
    sentinel = object()
    with (
        patch.object(settings, "llm_model_routing_enabled", True),
        patch.object(
            settings,
            "llm_model_routing_policy",
            {"router.route_classification": "small", "*": "main"},
        ),
        patch.object(settings, "llm_router_model", "GigaChat-Lite"),
        patch("app.llm._get_gigachat", return_value=sentinel) as get_model,
    ):
        llm, selection = get_routed_llm(
            node_name="router",
            purpose="route_classification",
            temperature=0.0,
        )
    assert llm is sentinel
    assert selection.model_name == "GigaChat-Lite"
    get_model.assert_called_once_with("GigaChat-Lite", temperature=0.0)


def check_invalid_policy_is_rejected() -> None:
    invalid_policies = (
        {"router": "small"},
        {"Router.answer": "main"},
        {"router.answer.extra": "main"},
        {"router.answer": "unknown"},
    )
    for policy in invalid_policies:
        try:
            Settings(_env_file=None, llm_model_routing_policy=policy)
        except ValidationError:
            pass
        else:
            raise AssertionError(f"Invalid routing policy was accepted: {policy}")


def check_policy_json_is_loaded_from_environment() -> None:
    raw_policy = '{"router.route_classification":"small","*":"main"}'
    with patch.dict(os.environ, {"LLM_MODEL_ROUTING_POLICY": raw_policy}):
        configured = Settings(_env_file=None)
    assert configured.llm_model_routing_policy == {
        "router.route_classification": "small",
        "*": "main",
    }


if __name__ == "__main__":
    check_rule_precedence()
    check_disabled_policy_uses_call_default()
    check_small_tier_falls_back_to_main_model()
    check_routed_client_uses_selected_model()
    check_invalid_policy_is_rejected()
    check_policy_json_is_loaded_from_environment()
    print("Model routing checks passed")
