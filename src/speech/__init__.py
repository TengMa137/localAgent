"""Local speech provider integrations."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .qwen3 import (
        ASRResult,
        Qwen3ASRConfig,
        Qwen3ASRProvider,
        Qwen3TTSConfig,
        Qwen3TTSProvider,
        TTSResult,
    )

__all__ = [
    "ASRResult",
    "Qwen3ASRConfig",
    "Qwen3ASRProvider",
    "Qwen3TTSConfig",
    "Qwen3TTSProvider",
    "TTSResult",
]


def __getattr__(name: str) -> object:
    if name in __all__:
        from . import qwen3

        return getattr(qwen3, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
