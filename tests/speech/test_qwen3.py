import base64
import io
import json
import subprocess
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from speech.qwen3 import (
    Qwen3ASRConfig,
    Qwen3ASRProvider,
    Qwen3TTSConfig,
    Qwen3TTSProvider,
    Qwen3TTSError,
    _parse_audio_device,
    _run_tts,
    parse_asr_output,
)


def test_parse_qwen3_asr_tagged_output():
    language, text = parse_asr_output(
        "language English<asr_text>This is a local transcript."
    )

    assert language == "English"
    assert text == "This is a local transcript."


def test_parse_qwen3_asr_empty_audio():
    language, text = parse_asr_output("language None<asr_text>")

    assert language == ""
    assert text == ""


def test_parse_qwen3_asr_plain_text_when_no_tag():
    language, text = parse_asr_output("plain transcript")

    assert language == ""
    assert text == "plain transcript"


def test_forced_language_treats_raw_output_as_text():
    language, text = parse_asr_output("bonjour", user_language="french")

    assert language == "French"
    assert text == "bonjour"


def test_forced_language_still_strips_qwen_tags():
    language, text = parse_asr_output(
        "language English<asr_text>Hello there.",
        user_language="English",
    )

    assert language == "English"
    assert text == "Hello there."


def test_transcription_body_uses_openai_audio_endpoint_fields():
    _boundary, body = Qwen3ASRProvider()._build_transcription_body(
        b"wav",
        filename="sample.wav",
        mime_type="audio/wav",
        language="English",
    )

    assert b'name="model"' in body
    assert b"./models/qwen3-asr-1.7b-q8_0.gguf" in body
    assert b'name="language"' in body
    assert b"English" in body
    assert b'name="file"; filename="sample.wav"' in body
    assert b"Content-Type: audio/wav" in body
    assert b"wav" in body


def test_default_configs_use_preferred_crispasr_servers():
    asr = Qwen3ASRConfig()
    tts = Qwen3TTSConfig()

    assert asr.base_url == "http://localhost:8081/v1"
    assert asr.backend == "qwen3"
    assert asr.model == "./models/qwen3-asr-1.7b-q8_0.gguf"
    assert tts.base_url == "http://localhost:8082/v1"
    assert tts.backend == "qwen3-tts-customvoice"
    assert tts.model == "./models/qwen3-tts-12hz-0.6b-customvoice-q8_0.gguf"
    assert tts.codec_model == "./models/qwen3-tts-tokenizer-12hz.gguf"
    assert tts.voice == "vivian"


def test_asr_command_uses_crispasr_qwen_defaults():
    provider = Qwen3ASRProvider(
        Qwen3ASRConfig(
            base_url="",
            cli_path="/bin/crispasr",
            backend="qwen3",
            model="auto",
            temperature=0.0,
            max_tokens=128,
            threads=2,
            vad=True,
        )
    )

    cmd = provider._build_transcription_command(
        Path("/tmp/sample.wav"),
        output_base=Path("/tmp/out"),
        language="en",
    )

    assert cmd == [
        "/bin/crispasr",
        "--backend",
        "qwen3",
        "-m",
        "auto",
        "-f",
        "/tmp/sample.wav",
        "-of",
        "/tmp/out",
        "-np",
        "-oj",
        "-l",
        "en",
        "--vad",
        "-t",
        "2",
        "-tp",
        "0.0",
        "-n",
        "128",
    ]


def test_transcribe_base64_accepts_existing_data_url(monkeypatch):
    captured = {}

    def fake_transcribe_bytes(audio_bytes, filename, mime_type, language):
        captured["audio_bytes"] = audio_bytes
        captured["filename"] = filename
        captured["mime_type"] = mime_type
        captured["language"] = language

    provider = Qwen3ASRProvider()
    monkeypatch.setattr(provider, "_transcribe_bytes_sync", fake_transcribe_bytes)
    audio = base64.b64encode(b"wav").decode("ascii")

    import asyncio

    asyncio.run(
        provider.transcribe_base64(
            f"data:audio/wav;base64,{audio}",
            mime_type="audio/wav",
            language="English",
        )
    )

    assert captured["audio_bytes"] == b"wav"
    assert captured["filename"] == "audio.wav"
    assert captured["mime_type"] == "audio/wav"
    assert captured["language"] == "English"


def test_asr_http_uses_openai_compatible_transcriptions_endpoint(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"text": "hello from asr"}'

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["data"] = request.data
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("speech.qwen3.urllib.request.urlopen", fake_urlopen)
    provider = Qwen3ASRProvider(
        Qwen3ASRConfig(
            base_url="http://localhost:8081/v1",
            model="./models/qwen3-asr-1.7b-q8_0.gguf",
            timeout_seconds=12,
        )
    )

    result = provider._transcribe_bytes_sync(
        b"wav",
        filename="sample.wav",
        mime_type="audio/wav",
        language="en",
    )

    assert captured["url"] == "http://localhost:8081/v1/audio/transcriptions"
    assert captured["timeout"] == 12
    assert b'name="file"; filename="sample.wav"' in captured["data"]
    assert b'name="model"' in captured["data"]
    assert b"./models/qwen3-asr-1.7b-q8_0.gguf" in captured["data"]
    assert b'name="response_format"' in captured["data"]
    assert result.text == "hello from asr"
    assert result.provider == "crispasr-qwen3-asr"


def test_tts_command_uses_qwen3_tts_cli_fields():
    provider = Qwen3TTSProvider(
        Qwen3TTSConfig(
            base_url="",
            cli_path="/bin/crispasr",
            backend="qwen3-tts-customvoice",
            model="auto",
            reference_text="reference words",
            temperature=0.0,
            threads=2,
        )
    )

    cmd = provider._build_synthesis_command(
        "hello",
        output_path=Path("/tmp/out.wav"),
        reference_audio_path=Path("/tmp/ref.wav"),
    )

    assert cmd == [
        "/bin/crispasr",
        "--backend",
        "qwen3-tts-customvoice",
        "-m",
        "auto",
        "--tts",
        "hello",
        "--tts-output",
        "/tmp/out.wav",
        "--voice",
        "/tmp/ref.wav",
        "--ref-text",
        "reference words",
        "--codec-model",
        "./models/qwen3-tts-tokenizer-12hz.gguf",
        "--temperature",
        "0.0",
        "-t",
        "2",
    ]


def test_tts_synthesize_reads_cli_output(monkeypatch):
    provider = Qwen3TTSProvider(
        Qwen3TTSConfig(
            base_url="",
            cli_path="/bin/crispasr",
            backend="qwen3-tts-customvoice",
            model="auto",
        )
    )

    def fake_run(cmd, **kwargs):
        del kwargs
        Path(cmd[cmd.index("--tts-output") + 1]).write_bytes(b"RIFF wav")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("speech.qwen3.subprocess.run", fake_run)

    result = provider._synthesize_sync("hello", reference_audio_path=None)

    assert result.audio_bytes == b"RIFF wav"
    assert result.audio_base64 == base64.b64encode(b"RIFF wav").decode("ascii")
    assert result.mime_type == "audio/wav"
    assert result.provider == "crispasr-qwen3-tts"


def test_tts_http_uses_openai_compatible_speech_endpoint(monkeypatch):
    captured = {}

    class FakeHeaders:
        def get_content_type(self):
            return "audio/wav"

    class FakeResponse:
        headers = FakeHeaders()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b"RIFF wav"

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("speech.qwen3.urllib.request.urlopen", fake_urlopen)
    provider = Qwen3TTSProvider(
        Qwen3TTSConfig(
            base_url="http://localhost:8082/v1",
            model="./models/qwen3-tts-12hz-0.6b-customvoice-q8_0.gguf",
            voice="vivian",
            timeout_seconds=34,
        )
    )

    result = provider._synthesize_sync(
        "hello",
        reference_audio_path=None,
        instructions="speak plainly",
        speed=1.2,
    )

    assert captured["url"] == "http://localhost:8082/v1/audio/speech"
    assert captured["timeout"] == 34
    assert captured["payload"] == {
        "model": "./models/qwen3-tts-12hz-0.6b-customvoice-q8_0.gguf",
        "input": "hello",
        "response_format": "wav",
        "voice": "vivian",
        "instructions": "speak plainly",
        "speed": 1.2,
    }
    assert result.audio_bytes == b"RIFF wav"
    assert result.mime_type == "audio/wav"
    assert result.provider == "crispasr-qwen3-tts"


def test_tts_http_empty_audio_error_includes_voice_context(monkeypatch):
    class FakeHeaders:
        def get_content_type(self):
            return "application/json"

    class FakeResponse:
        headers = FakeHeaders()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"voices":[{"name":"vivian"},{"name":"ryan"}]}'

    def fake_urlopen(request, timeout):
        del timeout
        if request.full_url.endswith("/voices"):
            return FakeResponse()
        body = (
            b'{"error":{"message":"synthesis failed '
            b'(backend returned empty audio)","code":"synthesis_failed"}}'
        )
        raise urllib.error.HTTPError(
            request.full_url,
            500,
            "Internal Server Error",
            {},
            io.BytesIO(body),
        )

    monkeypatch.setattr("speech.qwen3.urllib.request.urlopen", fake_urlopen)
    provider = Qwen3TTSProvider(
        Qwen3TTSConfig(
            base_url="http://localhost:8082/v1",
            voice="nonexistent",
        )
    )

    with pytest.raises(Qwen3TTSError) as excinfo:
        provider._synthesize_sync("hello", reference_audio_path=None)

    message = str(excinfo.value)
    assert "synthesis_failed" in message
    assert "request voice='nonexistent'" in message
    assert "server voices=vivian, ryan" in message
    assert "LOCALAGENT_TTS_VOICE" in message


def test_tts_cli_writes_output_file(monkeypatch, tmp_path):
    async def fake_synthesize(self, text, *, reference_audio_path=None):
        del self
        assert text == "hello"
        assert reference_audio_path is None
        return SimpleNamespace(audio_bytes=b"RIFF wav")

    monkeypatch.setattr(Qwen3TTSProvider, "synthesize", fake_synthesize)
    output_path = tmp_path / "out.wav"

    import asyncio

    asyncio.run(
        _run_tts(
            SimpleNamespace(
                text="hello",
                output=output_path,
                out_dir=tmp_path,
                reference_audio=None,
            )
        )
    )

    assert output_path.read_bytes() == b"RIFF wav"


def test_parse_audio_device_preserves_named_devices():
    assert _parse_audio_device(None) is None
    assert _parse_audio_device("2") == 2
    assert _parse_audio_device("MacBook Microphone") == "MacBook Microphone"
