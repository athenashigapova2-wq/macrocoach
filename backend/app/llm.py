"""GigaChat models used by Athena agents.

Agents depend on BaseChatModel while credentials and model selection stay here.
"""

from functools import lru_cache

from langchain_core.language_models import BaseChatModel

from app.config import settings


@lru_cache(maxsize=1)
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
    model = settings.llm_router_model or settings.gigachat_model
    return _get_gigachat(model, temperature=0.0)
