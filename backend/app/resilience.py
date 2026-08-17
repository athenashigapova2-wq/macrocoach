"""Retry helpers for operations that are safe to execute more than once."""

from __future__ import annotations

import logging
import random
import ssl
import time
from collections.abc import Callable, Iterator
from typing import TypeVar

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_TRANSIENT_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
T = TypeVar("T")


def _exception_chain(error: BaseException) -> Iterator[BaseException]:
    """Yield an exception and its causes without looping over malformed chains."""
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _status_code(error: BaseException) -> int | None:
    status = getattr(error, "status_code", None)
    if status is None:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def is_transient_error(error: BaseException) -> bool:
    """Return whether retrying this failure is likely to succeed later."""
    for item in _exception_chain(error):
        if isinstance(
            item,
            (
                TimeoutError,
                ConnectionError,
                ssl.SSLError,
                httpx.TimeoutException,
                httpx.NetworkError,
            ),
        ):
            return True
        if _status_code(item) in _TRANSIENT_STATUS_CODES:
            return True
    return False


def retry_transient(
    operation: Callable[[], T],
    *,
    operation_name: str,
) -> T:
    """Retry a declared-safe operation with capped exponential backoff and jitter."""
    max_attempts = settings.safe_retry_max_attempts
    attempt = 1
    while True:
        try:
            return operation()
        except Exception as error:
            if attempt >= max_attempts or not is_transient_error(error):
                raise

            backoff = min(
                settings.safe_retry_max_delay_seconds,
                settings.safe_retry_base_delay_seconds * (2 ** (attempt - 1)),
            )
            jitter = random.uniform(0.0, backoff * settings.safe_retry_jitter_ratio)
            delay = min(settings.safe_retry_max_delay_seconds, backoff + jitter)
            logger.warning(
                "Transient %s failure (%s); retrying in %.2fs (attempt %d/%d)",
                operation_name,
                type(error).__name__,
                delay,
                attempt + 1,
                max_attempts,
            )
            time.sleep(delay)
            attempt += 1
