import asyncio
import time
from types import SimpleNamespace

import pytest

from speech.qwen3 import Qwen3TTSProvider
from speech.stream_tts import (
    StreamingTTSConfig,
    StreamingTTSPlayer,
    TextChunker,
)


def test_text_chunker_emits_sentence_then_flushes_remainder():
    chunker = TextChunker(min_chars=5, max_chars=40)

    assert chunker.feed("Hello there. Still working") == ["Hello there."]
    assert chunker.flush() == ["Still working"]


def test_text_chunker_emits_short_complete_sentence_before_long_sentence():
    chunker = TextChunker(min_chars=80, max_chars=120)

    chunks = chunker.feed(
        "OK. This second sentence is intentionally much longer so playback "
        "does not wait for the short opener to be combined with it."
    )

    assert chunks[0] == "OK."
    assert chunks[1].startswith("This second sentence")


def test_text_chunker_splits_long_text_at_word_boundary():
    chunker = TextChunker(min_chars=8, max_chars=18)

    chunks = chunker.feed("This response contains many words without punctuation")

    assert chunks == ["This response", "contains many", "words without"]
    assert chunker.flush() == ["punctuation"]


def test_text_chunker_uses_shorter_initial_chunk_for_tts_startup():
    chunker = TextChunker(min_chars=8, max_chars=36, initial_max_chars=18)

    chunks = chunker.feed("This response contains many words without punctuation")

    assert chunks[:2] == ["This response", "contains many words without"]


def test_text_chunker_can_split_long_sentence_at_phrase_boundary():
    chunker = TextChunker(
        min_chars=10,
        max_chars=80,
        min_sentence_chars=20,
        phrase_boundary_chars=24,
    )

    chunks = chunker.feed("This first clause is ready, but the sentence keeps going")

    assert chunks == ["This first clause is ready,"]


def test_streaming_tts_player_defaults_to_qwen_provider():
    player = StreamingTTSPlayer()

    assert isinstance(player.provider, Qwen3TTSProvider)


def test_streaming_tts_player_uses_qwen_realtime_chunking_defaults():
    player = StreamingTTSPlayer(config=StreamingTTSConfig())

    assert player.config.min_chunk_chars == 50
    assert player.config.max_chunk_chars == 180
    assert player.config.min_sentence_chars == 24
    assert player.config.initial_max_chunk_chars == 120
    assert player.config.phrase_boundary_chars == 90


@pytest.mark.asyncio
async def test_streaming_tts_player_synthesizes_and_plays_chunks(monkeypatch):
    synthesized = []
    played = []

    class FakeProvider:
        async def synthesize(self, text, **kwargs):
            del kwargs
            synthesized.append(text)
            return SimpleNamespace(audio_bytes=f"wav:{text}".encode(), mime_type="audio/wav")

    def fake_play(audio_bytes, mime_type, player_command):
        played.append((audio_bytes, mime_type, player_command))

    monkeypatch.setattr("speech.stream_tts._play_audio_bytes", fake_play)
    player = StreamingTTSPlayer(
        provider=FakeProvider(),
        config=StreamingTTSConfig(
            min_chunk_chars=5,
            max_chunk_chars=40,
            player_command="test-player",
        ),
    )

    await player.speak_text("First sentence. Second sentence.")
    await player.drain()
    await player.close()

    assert synthesized == ["First sentence.", "Second sentence."]
    assert played == [
        (b"wav:First sentence.", "audio/wav", "test-player"),
        (b"wav:Second sentence.", "audio/wav", "test-player"),
    ]


@pytest.mark.asyncio
async def test_streaming_tts_player_synthesizes_next_chunk_during_playback(monkeypatch):
    events = []

    class FakeProvider:
        async def synthesize(self, text, **kwargs):
            del kwargs
            await asyncio.sleep(0.01)
            events.append((f"synth:{text}", time.monotonic()))
            return SimpleNamespace(audio_bytes=f"wav:{text}".encode(), mime_type="audio/wav")

    def fake_play(audio_bytes, mime_type, player_command):
        del mime_type, player_command
        label = audio_bytes.decode().removeprefix("wav:")
        events.append((f"play_start:{label}", time.monotonic()))
        if label == "First sentence.":
            time.sleep(0.1)
        events.append((f"play_end:{label}", time.monotonic()))

    monkeypatch.setattr("speech.stream_tts._play_audio_bytes", fake_play)
    player = StreamingTTSPlayer(
        provider=FakeProvider(),
        config=StreamingTTSConfig(
            min_chunk_chars=5,
            max_chunk_chars=80,
            player_command="test-player",
        ),
    )

    await player.speak_text("First sentence. Second sentence.")
    await player.drain()
    await player.close()

    event_times = dict(events)
    assert event_times["synth:Second sentence."] < event_times["play_end:First sentence."]
