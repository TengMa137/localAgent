"""Chunked terminal TTS playback for assistant replies.

This module does not build one growing WAV file. It turns reply text into
independent text chunks, synthesizes each chunk as its own audio response, and
plays those audio responses in order.

Chunking rules:
- Complete sentences are preferred whenever punctuation is available.
- If a sentence gets long, optional phrase boundaries such as commas,
  semicolons, or colons can become chunk points after
  ``phrase_boundary_chars`` characters.
- If no sentence or phrase boundary is available, the chunker falls back to a
  word-boundary split at the current max chunk size.
- ``initial_max_chunk_chars`` can make only the first chunk shorter so playback
  starts sooner; later chunks use ``max_chunk_chars``.

Playback pipeline:
1. The chunker emits prepared text chunks.
2. The synthesis worker sends one chunk at a time to the configured TTS
   provider and pushes the returned audio bytes into an audio queue.
3. The playback worker writes each audio response to a temporary file and calls
   the local player command. Synthesis of the next chunk can overlap playback
   of the previous chunk, but playback itself remains sequential.

For the current terminal runner, text is passed in after the assistant reply is
complete. The same chunker also supports delta input, so a future streaming
model path can feed tokens earlier through ``speak_delta``.
"""

from __future__ import annotations

import asyncio
import mimetypes
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from .qwen3 import Qwen3TTSProvider


_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")
_PHRASE_BOUNDARY_RE = re.compile(r"(?<=[,;:])\s+")
_WHITESPACE_RE = re.compile(r"\s+")


class TTSProvider(Protocol):
    async def synthesize(self, text: str, **kwargs) -> object:
        ...


@dataclass(frozen=True)
class StreamingTTSConfig:
    """Runtime controls for terminal TTS chunking and playback."""

    min_chunk_chars: int = int(os.getenv("LOCALAGENT_TTS_MIN_CHARS", "80"))
    max_chunk_chars: int = int(os.getenv("LOCALAGENT_TTS_MAX_CHARS", "260"))
    min_sentence_chars: int = int(os.getenv("LOCALAGENT_TTS_MIN_SENTENCE_CHARS", "1"))
    initial_max_chunk_chars: int = int(os.getenv("LOCALAGENT_TTS_INITIAL_MAX_CHARS", "0"))
    phrase_boundary_chars: int = int(os.getenv("LOCALAGENT_TTS_PHRASE_BOUNDARY_CHARS", "0"))
    player_command: str = os.getenv("LOCALAGENT_TTS_PLAYER", "")


@dataclass(frozen=True)
class SynthesizedAudio:
    audio_bytes: bytes
    mime_type: str


class TextChunker:
    """Turn text deltas into TTS-sized chunks without splitting every token."""

    def __init__(
        self,
        *,
        min_chars: int,
        max_chars: int,
        min_sentence_chars: int = 1,
        initial_max_chars: int = 0,
        phrase_boundary_chars: int = 0,
    ) -> None:
        self.min_chars = max(1, min_chars)
        self.max_chars = max(self.min_chars, max_chars)
        self.min_sentence_chars = max(1, min_sentence_chars)
        self.initial_max_chars = (
            max(self.min_chars, initial_max_chars) if initial_max_chars > 0 else 0
        )
        self.phrase_boundary_chars = max(0, phrase_boundary_chars)
        self._buffer = ""
        self._chunks_emitted = 0

    def feed(self, text: str) -> list[str]:
        if not text:
            return []
        self._buffer += text
        return self._pop_ready(final=False)

    def flush(self) -> list[str]:
        return self._pop_ready(final=True)

    def _pop_ready(self, *, final: bool) -> list[str]:
        chunks: list[str] = []
        while True:
            clean = self._buffer.lstrip()
            if clean != self._buffer:
                self._buffer = clean
            if not self._buffer:
                return chunks

            boundary = self._find_boundary()
            if boundary is not None:
                chunks.append(self._take(boundary))
                continue

            max_chars = self._current_max_chars()
            if len(self._buffer) >= max_chars:
                chunks.append(self._take(self._split_at_word()))
                continue

            if final:
                chunks.append(self._take(len(self._buffer)))
                continue

            return chunks

    def _find_boundary(self) -> int | None:
        for match in _SENTENCE_BOUNDARY_RE.finditer(self._buffer):
            end = match.end()
            if end >= self.min_sentence_chars:
                return end
        if self.phrase_boundary_chars:
            for match in _PHRASE_BOUNDARY_RE.finditer(self._buffer):
                end = match.end()
                if end >= self.phrase_boundary_chars:
                    return end
        return None

    def _split_at_word(self) -> int:
        window = self._buffer[: self._current_max_chars()]
        index = window.rfind(" ")
        if index >= self.min_chars:
            return index + 1
        return self._current_max_chars()

    def _take(self, count: int) -> str:
        chunk = self._buffer[:count]
        self._buffer = self._buffer[count:]
        self._chunks_emitted += 1
        return _prepare_tts_text(chunk)

    def _current_max_chars(self) -> int:
        if self._chunks_emitted == 0 and self.initial_max_chars:
            return min(self.initial_max_chars, self.max_chars)
        return self.max_chars


def _prepare_tts_text(text: str) -> str:
    """Make markdown-heavy terminal output less awkward for speech."""

    text = re.sub(r"```.*?```", " code block omitted. ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("*", "").replace("_", "")
    return _WHITESPACE_RE.sub(" ", text).strip()


class StreamingTTSPlayer:
    """Synthesize queued text chunks and play them sequentially."""

    def __init__(
        self,
        *,
        provider: TTSProvider | None = None,
        config: StreamingTTSConfig | None = None,
    ) -> None:
        base_config = config or StreamingTTSConfig()
        self.config = (
            _with_qwen3_chunking_defaults(base_config)
            if provider is None
            else base_config
        )
        self.provider = provider or Qwen3TTSProvider()
        self._chunker = TextChunker(
            min_chars=self.config.min_chunk_chars,
            max_chars=self.config.max_chunk_chars,
            min_sentence_chars=self.config.min_sentence_chars,
            initial_max_chars=self.config.initial_max_chunk_chars,
            phrase_boundary_chars=self.config.phrase_boundary_chars,
        )
        self._text_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._audio_queue: asyncio.Queue[SynthesizedAudio | None] = asyncio.Queue()
        self._synthesis_worker: asyncio.Task[None] | None = None
        self._playback_worker: asyncio.Task[None] | None = None
        self._warned = False

    async def speak_text(self, text: str) -> None:
        await self.speak_delta(text)
        for chunk in self._chunker.flush():
            await self._enqueue(chunk)

    async def speak_delta(self, text: str) -> None:
        for chunk in self._chunker.feed(text):
            await self._enqueue(chunk)

    async def drain(self) -> None:
        await self._text_queue.join()
        await self._audio_queue.join()

    async def close(self) -> None:
        if self._synthesis_worker is None and self._playback_worker is None:
            return
        self._ensure_workers()
        await self._text_queue.put(None)
        if self._synthesis_worker is not None:
            await self._synthesis_worker
        if self._playback_worker is not None:
            await self._playback_worker
        self._synthesis_worker = None
        self._playback_worker = None

    async def _enqueue(self, chunk: str) -> None:
        if not chunk:
            return
        self._ensure_workers()
        await self._text_queue.put(chunk)

    def _ensure_workers(self) -> None:
        if self._synthesis_worker is None or self._synthesis_worker.done():
            self._synthesis_worker = asyncio.create_task(self._run_synthesis())
        if self._playback_worker is None or self._playback_worker.done():
            self._playback_worker = asyncio.create_task(self._run_playback())

    async def _run_synthesis(self) -> None:
        while True:
            chunk = await self._text_queue.get()
            try:
                if chunk is None:
                    await self._audio_queue.put(None)
                    return
                await self._synthesize(chunk)
            finally:
                self._text_queue.task_done()

    async def _run_playback(self) -> None:
        while True:
            item = await self._audio_queue.get()
            try:
                if item is None:
                    return
                await asyncio.to_thread(
                    _play_audio_bytes,
                    item.audio_bytes,
                    item.mime_type,
                    self.config.player_command,
                )
            except Exception as exc:
                self._warn_once(f"[tts warn] {exc}")
            finally:
                self._audio_queue.task_done()

    async def _synthesize(self, text: str) -> None:
        try:
            result = await self.provider.synthesize(text)
            audio_bytes = getattr(result, "audio_bytes")
            mime_type = getattr(result, "mime_type", "audio/wav")
            await self._audio_queue.put(SynthesizedAudio(audio_bytes, mime_type))
        except Exception as exc:
            self._warn_once(f"[tts warn] {exc}")

    def _warn_once(self, message: str) -> None:
        if self._warned:
            return
        self._warned = True
        print(message, file=sys.stderr, flush=True)


def _play_audio_bytes(
    audio_bytes: bytes,
    mime_type: str,
    player_command: str,
) -> None:
    command = _resolve_player_command(player_command)
    suffix = mimetypes.guess_extension(mime_type) or ".wav"
    handle = tempfile.NamedTemporaryFile(
        prefix="localagent-tts-play-",
        suffix=suffix,
        delete=False,
    )
    path = Path(handle.name)
    try:
        handle.write(audio_bytes)
        handle.close()
        completed = subprocess.run(
            [*command, str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        try:
            handle.close()
        except Exception:
            pass
        path.unlink(missing_ok=True)

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"TTS playback failed: {detail or command[0]}")


def _resolve_player_command(player_command: str) -> list[str]:
    if player_command.strip():
        return shlex.split(player_command)

    for candidate in (
        ("afplay",),
        ("aplay", "-q"),
        ("paplay",),
        ("ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"),
    ):
        if shutil.which(candidate[0]):
            return list(candidate)

    raise RuntimeError(
        "no audio player found; set LOCALAGENT_TTS_PLAYER or pass --tts-player"
    )


def _with_qwen3_chunking_defaults(config: StreamingTTSConfig) -> StreamingTTSConfig:
    updates = {}
    if "LOCALAGENT_TTS_MIN_CHARS" not in os.environ:
        updates["min_chunk_chars"] = 50
    if "LOCALAGENT_TTS_MAX_CHARS" not in os.environ:
        updates["max_chunk_chars"] = 180
    if "LOCALAGENT_TTS_MIN_SENTENCE_CHARS" not in os.environ:
        updates["min_sentence_chars"] = 24
    if "LOCALAGENT_TTS_INITIAL_MAX_CHARS" not in os.environ:
        updates["initial_max_chunk_chars"] = 120
    if "LOCALAGENT_TTS_PHRASE_BOUNDARY_CHARS" not in os.environ:
        updates["phrase_boundary_chars"] = 90
    return replace(config, **updates) if updates else config
