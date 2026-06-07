"""Manual structured-output validation retries that keep model history bounded."""

from __future__ import annotations

import re
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior

from localagent_settings import get_runtime_settings
from .observability import _rt, observable_run

# Total calls, not retries: one initial call plus at most two fresh retries.
MAX_MANUAL_OUTPUT_ATTEMPTS = 3
MAX_RETRY_ERROR_CHARS = 240
SELF_REVIEW_BOUNDARY_RE = re.compile(
    r"(?im)^\s*(?:"
    r"wait(?:[,.:]|\s+-)|"
    r"final check:|"
    r"i need to (?:ensure|check|verify)\b|"
    r"(?:okay|ok),?\s+(?:i will|let me)\b"
    r")"
)


def _attempt_limit(attempts: int | None = None) -> int:
    configured = (
        get_runtime_settings().structured_output_attempts
        if attempts is None
        else attempts
    )
    return max(1, min(configured, MAX_MANUAL_OUTPUT_ATTEMPTS))


def _merge_model_settings(
    kwargs: dict[str, Any],
    *,
    max_tokens: int | None,
) -> dict[str, Any]:
    settings = get_runtime_settings()
    run_kwargs = dict(kwargs)
    existing = run_kwargs.pop("model_settings", None) or {}
    merged = dict(existing)
    if max_tokens and "max_tokens" not in merged:
        merged["max_tokens"] = max_tokens
    if settings.model_request_timeout_seconds and "timeout" not in merged:
        merged["timeout"] = settings.model_request_timeout_seconds
    if settings.disable_model_thinking:
        extra_body = merged.get("extra_body")
        if extra_body is None or isinstance(extra_body, dict):
            normalized_extra = dict(extra_body or {})
            chat_template_kwargs = dict(
                normalized_extra.get("chat_template_kwargs") or {}
            )
            chat_template_kwargs.setdefault("enable_thinking", False)
            normalized_extra["chat_template_kwargs"] = chat_template_kwargs
            merged["extra_body"] = normalized_extra
    if merged:
        run_kwargs["model_settings"] = merged
    return run_kwargs


def structured_model_settings(kwargs: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return run kwargs with the default structured-output token cap applied."""
    settings = get_runtime_settings()
    return _merge_model_settings(
        kwargs or {},
        max_tokens=settings.structured_output_max_tokens,
    )


def answer_model_settings(kwargs: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return run kwargs with the default final-answer token cap applied."""
    settings = get_runtime_settings()
    return _merge_model_settings(
        kwargs or {},
        max_tokens=settings.answer_output_max_tokens,
    )


def clean_text_answer(text: str) -> str:
    """Trim small-model self-review that continues after a usable answer."""
    answer = text.strip()
    for match in SELF_REVIEW_BOUNDARY_RE.finditer(answer):
        prefix = answer[: match.start()].rstrip()
        if len(prefix) >= 80:
            return prefix
    return answer


def _compact_error(exc: BaseException) -> str:
    text = str(exc).strip()
    if len(text) <= MAX_RETRY_ERROR_CHARS:
        return text
    return text[: MAX_RETRY_ERROR_CHARS - 3].rstrip() + "..."


def _validation_retry_prompt(
    *,
    original_prompt: str,
    output_name: str,
    error: BaseException,
) -> str:
    return (
        "The invalid response is intentionally omitted.\n"
        f"Return only a valid {output_name}; follow the system schema exactly "
        "and do not explain.\n\n"
        f"Task:\n{original_prompt}\n\n"
        f"Validation summary: {_compact_error(error)}"
    )


async def observable_run_with_manual_validation_retries(
    agent: Agent[Any, Any],
    prompt: str,
    *,
    output_type: type | tuple[type, ...],
    output_name: str,
    label: str,
    indent: int = 0,
    message_history: list[Any] | None = None,
    attempts: int | None = None,
    **kwargs: Any,
) -> Any:
    """Run a structured agent without replaying failed model completions.

    PydanticAI's built-in output retry keeps the invalid model response in the
    next request's conversation history. With small local context windows, one
    runaway invalid completion can make the next retry exceed context. This
    wrapper keeps retry attempts fresh: original history plus a compact repair
    prompt, never the invalid completion.
    """
    max_attempts = _attempt_limit(attempts)
    current_prompt = prompt
    last_error: BaseException | None = None

    for attempt in range(max_attempts):
        try:
            result = await observable_run(
                agent,
                current_prompt,
                label=label,
                indent=indent,
                message_history=message_history,
                **structured_model_settings(kwargs),
            )
        except UnexpectedModelBehavior as exc:
            last_error = exc
        else:
            if isinstance(result.output, output_type):
                return result
            last_error = RuntimeError(
                f"{output_name} returned unexpected output type: "
                f"{type(result.output).__name__}"
            )

        if attempt >= max_attempts - 1:
            break

        _rt(
            f"[{label}] output validation retry {attempt + 1}/{max_attempts - 1}",
            "yellow",
            indent,
        )
        current_prompt = _validation_retry_prompt(
            original_prompt=prompt,
            output_name=output_name,
            error=last_error or RuntimeError("unknown validation error"),
        )

    assert last_error is not None
    raise last_error
