"""GigaChat models used by Athena agents.

Agents depend on BaseChatModel while credentials and model selection stay here.
"""

from functools import lru_cache

from langchain_core.language_models import BaseChatModel

from app.config import settings
from app.model_routing import ModelSelection, ModelTier, model_name_for_tier, select_model


@lru_cache(maxsize=8)
def _get_gigachat(model: str, *, temperature: float | None = None) -> BaseChatModel:
    from langchain_gigachat import GigaChat

    if not settings.gigachat_auth_key:
        raise RuntimeError("GIGACHAT_AUTH_KEY must be set")
    kwargs = {
        "credentials": settings.gigachat_auth_key,
        "scope": settings.gigachat_scope,
        "model": model,
        "verify_ssl_certs": False,  # TODO: заменить на ca_bundle_file перед продакшеном
        "profanity_check": False,
        "timeout": 60,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    return GigaChat(**kwargs)


def get_llm() -> BaseChatModel:
    """Основная модель: диалог, вызов инструментов."""
    return _get_gigachat(settings.gigachat_model)


@lru_cache(maxsize=1)
def get_router_llm() -> BaseChatModel:
    """Лёгкая модель для роутера: одна классификация, нужна скорость."""
    return _get_gigachat(model_name_for_tier("small"), temperature=0.0)


def get_routed_llm(
    *,
    node_name: str,
    purpose: str,
    default_tier: ModelTier = "main",
    temperature: float | None = None,
) -> tuple[BaseChatModel, ModelSelection]:
    """Return both the selected model and the decision recorded in traces."""
    selection = select_model(
        node_name=node_name,
        purpose=purpose,
        default_tier=default_tier,
    )
    return (
        _get_gigachat(selection.model_name, temperature=temperature),
        selection,
    )
