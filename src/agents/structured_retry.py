from __future__ import annotations

from typing import Any

from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior

from localagent_settings import get_runtime_settings
from .observability import _rt, observable_run

MAX_MANUAL_OUTPUT_ATTEMPTS = 3
MAX_RETRY_ERROR_CHARS = 1200


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
        "The previous model response failed runtime output validation. The "
        "invalid response is intentionally omitted from this retry.\n\n"
        f"Expected output: {output_name}\n\n"
        "Validation error summary:\n"
        f"{_compact_error(error)}\n\n"
        "Original task:\n"
        f"{original_prompt}\n\n"
        "Retry from the original task. Return only the required final output "
        "for the expected schema; do not explain the validation failure."
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
