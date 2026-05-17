import base64
import subprocess
from types import SimpleNamespace
from pathlib import Path

from speech.qwen3 import (
    Qwen3ASRProvider,
    Qwen3TTSConfig,
    Qwen3TTSProvider,
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
    assert b"Qwen3-ASR-1.7B-GGUF" in body
    assert b'name="language"' in body
    assert b"English" in body
    assert b'name="file"; filename="sample.wav"' in body
    assert b"Content-Type: audio/wav" in body
    assert b"wav" in body


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


def test_tts_command_uses_qwen3_tts_cli_fields():
    provider = Qwen3TTSProvider(
        Qwen3TTSConfig(
            cli_path="/bin/qwen3-tts-cli",
            model_dir="/models/qwen3-tts",
            temperature=0.0,
            top_k=1,
            top_p=0.8,
            max_tokens=128,
            repetition_penalty=1.1,
            threads=2,
        )
    )

    cmd = provider._build_synthesis_command(
        "hello",
        output_path=Path("/tmp/out.wav"),
        reference_audio_path=Path("/tmp/ref.wav"),
    )

    assert cmd[:7] == [
        "/bin/qwen3-tts-cli",
        "-m",
        "/models/qwen3-tts",
        "-t",
        "hello",
        "-o",
        "/tmp/out.wav",
    ]
    assert cmd[7:] == [
        "-r",
        "/tmp/ref.wav",
        "--temperature",
        "0.0",
        "--top-k",
        "1",
        "--top-p",
        "0.8",
        "--max-tokens",
        "128",
        "--repetition-penalty",
        "1.1",
        "-j",
        "2",
    ]


def test_tts_synthesize_reads_cli_output(monkeypatch):
    provider = Qwen3TTSProvider(
        Qwen3TTSConfig(cli_path="/bin/qwen3-tts-cli", model_dir="/models/qwen3-tts")
    )

    def fake_run(cmd, **kwargs):
        del kwargs
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"RIFF wav")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("speech.qwen3.subprocess.run", fake_run)

    result = provider._synthesize_sync("hello", reference_audio_path=None)

    assert result.audio_bytes == b"RIFF wav"
    assert result.audio_base64 == base64.b64encode(b"RIFF wav").decode("ascii")
    assert result.mime_type == "audio/wav"
    assert result.provider == "qwen3-tts-cpp"


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
