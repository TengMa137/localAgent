"""Compact experimental Qwen3 speech providers."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


_ASR_TEXT_TAG = "<asr_text>"
_LANG_PREFIX = "language "


@dataclass(frozen=True)
class ASRResult:
    """Normalized speech-to-text result from any ASR backend."""

    text: str
    language: str = ""
    raw_text: str = ""
    provider: str = ""


@dataclass(frozen=True)
class TTSResult:
    """Normalized text-to-speech result from any TTS backend."""

    audio_bytes: bytes
    mime_type: str = "audio/wav"
    provider: str = ""

    @property
    def audio_base64(self) -> str:
        return base64.b64encode(self.audio_bytes).decode("ascii")


@dataclass(frozen=True)
class Qwen3ASRConfig:
    """Runtime config for a Qwen3-ASR llama.cpp-compatible server."""

    base_url: str = os.getenv("LOCALAGENT_ASR_BASE_URL", "http://localhost:8081/v1")
    model: str = os.getenv("LOCALAGENT_ASR_MODEL", "Qwen3-ASR-1.7B-GGUF")
    api_key: str = os.getenv("LOCALAGENT_ASR_API_KEY", "no-key")
    timeout_seconds: float = float(os.getenv("LOCALAGENT_ASR_TIMEOUT_SECONDS", "300"))
    max_tokens: int = int(os.getenv("LOCALAGENT_ASR_MAX_TOKENS", "512"))
    temperature: float = float(os.getenv("LOCALAGENT_ASR_TEMPERATURE", "0.01"))


class Qwen3ASRError(RuntimeError):
    """Raised when the local Qwen3-ASR backend cannot produce a transcript."""


def _optional_float(name: str) -> float | None:
    value = os.getenv(name)
    return float(value) if value else None


def _optional_int(name: str) -> int | None:
    value = os.getenv(name)
    return int(value) if value else None


@dataclass(frozen=True)
class Qwen3TTSConfig:
    """Runtime config for qwen3-tts.cpp CLI synthesis."""

    cli_path: str = os.getenv("LOCALAGENT_TTS_CLI", "qwen3-tts-cli")
    model_dir: str = os.getenv("LOCALAGENT_TTS_MODEL_DIR", "models")
    timeout_seconds: float = float(os.getenv("LOCALAGENT_TTS_TIMEOUT_SECONDS", "600"))
    temperature: float | None = _optional_float("LOCALAGENT_TTS_TEMPERATURE")
    top_k: int | None = _optional_int("LOCALAGENT_TTS_TOP_K")
    top_p: float | None = _optional_float("LOCALAGENT_TTS_TOP_P")
    max_tokens: int | None = _optional_int("LOCALAGENT_TTS_MAX_TOKENS")
    repetition_penalty: float | None = _optional_float(
        "LOCALAGENT_TTS_REPETITION_PENALTY"
    )
    threads: int | None = _optional_int("LOCALAGENT_TTS_THREADS")


class Qwen3TTSError(RuntimeError):
    """Raised when qwen3-tts.cpp cannot produce audio."""


class Qwen3ASRProvider:
    """Transcribe audio through a local Qwen3-ASR GGUF server.

    Start the server with something like:

        llama-server -hf ggml-org/Qwen3-ASR-1.7B-GGUF:Q8_0 --port 8081

    The provider sends audio to the OpenAI-style audio transcription endpoint.
    This keeps the rest of the agent code text-only.
    """

    provider_name = "qwen3-asr-llamacpp"

    def __init__(self, config: Qwen3ASRConfig | None = None) -> None:
        self.config = config or Qwen3ASRConfig()

    async def transcribe_file(
        self,
        audio_path: str | Path,
        *,
        language: str | None = None,
    ) -> ASRResult:
        path = Path(audio_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return await asyncio.to_thread(self._transcribe_file_sync, path, language)

    async def transcribe_base64(
        self,
        audio_base64: str,
        *,
        mime_type: str = "audio/wav",
        language: str | None = None,
    ) -> ASRResult:
        audio_base64 = _strip_data_url_prefix(audio_base64)
        audio_bytes = base64.b64decode(audio_base64)
        return await asyncio.to_thread(
            self._transcribe_bytes_sync,
            audio_bytes,
            "audio.wav",
            mime_type,
            language,
        )

    def _transcribe_file_sync(self, path: Path, language: str | None) -> ASRResult:
        mime_type = mimetypes.guess_type(path.name)[0] or "audio/wav"
        return self._transcribe_bytes_sync(path.read_bytes(), path.name, mime_type, language)

    def _transcribe_bytes_sync(
        self,
        audio_bytes: bytes,
        filename: str,
        mime_type: str,
        language: str | None,
    ) -> ASRResult:
        boundary, body = self._build_transcription_body(
            audio_bytes,
            filename=filename,
            mime_type=mime_type,
            language=language,
        )
        url = self.config.base_url.rstrip("/") + "/audio/transcriptions"
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise Qwen3ASRError(f"Qwen3-ASR HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise Qwen3ASRError(
                f"Could not reach Qwen3-ASR server at {url}: {exc.reason}"
            ) from exc

        raw = _extract_transcription_text(json.loads(body))
        parsed_language, text = parse_asr_output(raw, user_language=language)
        return ASRResult(
            text=text,
            language=parsed_language,
            raw_text=raw,
            provider=self.provider_name,
        )

    def _build_transcription_body(
        self,
        audio_bytes: bytes,
        *,
        filename: str,
        mime_type: str,
        language: str | None,
    ) -> tuple[str, bytes]:
        boundary = "----localagent-qwen3-asr"
        parts = [
            _multipart_field(boundary, "model", self.config.model),
            _multipart_field(boundary, "temperature", str(self.config.temperature)),
            _multipart_field(boundary, "max_tokens", str(self.config.max_tokens)),
        ]
        if language:
            parts.append(_multipart_field(boundary, "language", language))
        parts.append(
            _multipart_file(
                boundary,
                "file",
                filename=filename,
                content_type=mime_type,
                content=audio_bytes,
            )
        )
        parts.append(f"--{boundary}--\r\n".encode("utf-8"))
        return boundary, b"".join(parts)


class Qwen3TTSProvider:
    """Synthesize speech through the qwen3-tts.cpp command-line binary.

    Build https://github.com/predict-woo/qwen3-tts.cpp separately, then either
    put `qwen3-tts-cli` on PATH or set LOCALAGENT_TTS_CLI. The model directory
    should contain the converted GGUF artifacts from that project.
    """

    provider_name = "qwen3-tts-cpp"

    def __init__(self, config: Qwen3TTSConfig | None = None) -> None:
        self.config = config or Qwen3TTSConfig()

    async def synthesize(
        self,
        text: str,
        *,
        reference_audio_path: str | Path | None = None,
    ) -> TTSResult:
        return await asyncio.to_thread(
            self._synthesize_sync,
            text,
            Path(reference_audio_path).expanduser().resolve()
            if reference_audio_path
            else None,
        )

    async def synthesize_with_reference_base64(
        self,
        text: str,
        reference_audio_base64: str,
        *,
        reference_mime_type: str = "audio/wav",
    ) -> TTSResult:
        suffix = mimetypes.guess_extension(reference_mime_type) or ".wav"
        reference_audio_base64 = _strip_data_url_prefix(reference_audio_base64)
        reference_path: Path | None = None
        try:
            handle = tempfile.NamedTemporaryFile(
                prefix="localagent-tts-reference-",
                suffix=suffix,
                delete=False,
            )
            reference_path = Path(handle.name)
            handle.close()
            reference_path.write_bytes(base64.b64decode(reference_audio_base64))
            return await self.synthesize(text, reference_audio_path=reference_path)
        finally:
            if reference_path is not None:
                reference_path.unlink(missing_ok=True)

    def _synthesize_sync(
        self,
        text: str,
        reference_audio_path: Path | None,
    ) -> TTSResult:
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("text is required")
        if reference_audio_path is not None and not reference_audio_path.exists():
            raise FileNotFoundError(reference_audio_path)

        handle = tempfile.NamedTemporaryFile(
            prefix="localagent-tts-",
            suffix=".wav",
            delete=False,
        )
        output_path = Path(handle.name)
        handle.close()

        try:
            completed = subprocess.run(
                self._build_synthesis_command(
                    clean_text,
                    output_path=output_path,
                    reference_audio_path=reference_audio_path,
                ),
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            output_path.unlink(missing_ok=True)
            raise Qwen3TTSError(
                f"Could not find qwen3-tts.cpp CLI: {self.config.cli_path}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            output_path.unlink(missing_ok=True)
            raise Qwen3TTSError("qwen3-tts.cpp synthesis timed out") from exc

        if completed.returncode != 0:
            output_path.unlink(missing_ok=True)
            detail = (completed.stderr or completed.stdout or "").strip()
            raise Qwen3TTSError(f"qwen3-tts.cpp failed: {detail}")

        try:
            audio_bytes = output_path.read_bytes()
        finally:
            output_path.unlink(missing_ok=True)
        if not audio_bytes:
            raise Qwen3TTSError("qwen3-tts.cpp produced an empty WAV file")
        return TTSResult(
            audio_bytes=audio_bytes,
            mime_type="audio/wav",
            provider=self.provider_name,
        )

    def _build_synthesis_command(
        self,
        text: str,
        *,
        output_path: Path,
        reference_audio_path: Path | None = None,
    ) -> list[str]:
        cmd = [
            self.config.cli_path,
            "-m",
            self.config.model_dir,
            "-t",
            text,
            "-o",
            str(output_path),
        ]
        if reference_audio_path is not None:
            cmd.extend(["-r", str(reference_audio_path)])
        _append_optional(cmd, "--temperature", self.config.temperature)
        _append_optional(cmd, "--top-k", self.config.top_k)
        _append_optional(cmd, "--top-p", self.config.top_p)
        _append_optional(cmd, "--max-tokens", self.config.max_tokens)
        _append_optional(cmd, "--repetition-penalty", self.config.repetition_penalty)
        if self.config.threads is not None:
            cmd.extend(["-j", str(self.config.threads)])
        return cmd


def parse_asr_output(raw: str | None, user_language: str | None = None) -> tuple[str, str]:
    """Parse Qwen3-ASR raw output into ``(language, text)``.

    Qwen3-ASR usually returns ``language English<asr_text>...``. If a caller
    forced a language, upstream treats the result as plain transcription text.
    """

    if raw is None:
        return "", ""
    value = _fix_repetitions(str(raw).strip())
    if not value:
        return "", ""

    if _ASR_TEXT_TAG not in value:
        language = _normalize_language_name(user_language) if user_language else ""
        return language, value

    meta, text = value.split(_ASR_TEXT_TAG, 1)
    if "language none" in meta.lower() and not text.strip():
        return "", ""

    language = _normalize_language_name(user_language) if user_language else ""
    for line in meta.splitlines():
        clean = line.strip()
        if clean.lower().startswith(_LANG_PREFIX):
            if not language:
                language = _normalize_language_name(clean[len(_LANG_PREFIX) :].strip())
            break

    return language, text.strip()


def _extract_transcription_text(payload: dict[str, object]) -> str:
    text = payload.get("text")
    if isinstance(text, str):
        return text
    try:
        choices = payload["choices"]
        if isinstance(choices, list):
            message = choices[0]["message"]
            content = message["content"]
            return str(content)
    except (KeyError, IndexError, TypeError) as exc:
        raise Qwen3ASRError(f"Unexpected ASR response payload: {payload!r}") from exc
    raise Qwen3ASRError(f"Unexpected ASR response payload: {payload!r}")


def _multipart_field(boundary: str, name: str, value: str) -> bytes:
    return (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
        f"{value}\r\n"
    ).encode("utf-8")


def _multipart_file(
    boundary: str,
    name: str,
    *,
    filename: str,
    content_type: str,
    content: bytes,
) -> bytes:
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8")
    return header + content + b"\r\n"


def _strip_data_url_prefix(value: str) -> str:
    text = value.strip()
    if text.startswith("data:") and "," in text:
        return text.split(",", 1)[1]
    return text


def _normalize_language_name(language: str) -> str:
    language = str(language).strip()
    if not language:
        return ""
    return language[:1].upper() + language[1:].lower()


def _fix_repetitions(text: str, threshold: int = 20) -> str:
    """Small copy of Qwen's repetition cleanup for pathological decodes."""

    if not text:
        return text

    def replace_char_repeat(match: re.Match[str]) -> str:
        return match.group(1)

    text = re.sub(rf"(.)\1{{{threshold},}}", replace_char_repeat, text)

    for width in range(1, 21):
        pattern = re.compile(rf"(.{{{width}}})\1{{{threshold},}}")
        text = pattern.sub(lambda match: match.group(1), text)
    return text


def _append_optional(cmd: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def _timestamped_wav_path(out_dir: Path, prefix: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns()}.wav"


async def _run_asr_file(args: argparse.Namespace) -> None:
    result = await Qwen3ASRProvider().transcribe_file(
        args.audio_path,
        language=args.language,
    )
    _print_asr_result(result)


async def _run_asr_mic(args: argparse.Namespace) -> None:
    from .terminal_voice import TerminalRecorder

    recorder = TerminalRecorder(
        sample_rate=args.sample_rate,
        device=_parse_audio_device(args.device),
        normalize_audio=not args.no_normalize,
    )
    print(f"Recording microphone for {args.seconds:.1f}s...")
    recorder.start()
    await asyncio.sleep(args.seconds)
    recording = recorder.stop()
    if recording is None:
        print("No audio recorded")
        return

    saved_path = _timestamped_wav_path(args.out_dir, "mic")
    try:
        shutil.copy2(recording.path, saved_path)
        print(f"Saved mic audio: {saved_path}")
        print(
            f"Transcribing {recording.elapsed:.1f}s "
            f"(input_peak={recording.input_peak:.1%}, "
            f"gain={recording.gain:.1f}x, "
            f"peak={recording.peak:.1%}, rms={recording.rms:.1%})..."
        )
        result = await Qwen3ASRProvider().transcribe_file(
            saved_path,
            language=args.language,
        )
        _print_asr_result(result)
    finally:
        recording.path.unlink(missing_ok=True)


async def _run_tts(args: argparse.Namespace) -> None:
    output_path = args.output or _timestamped_wav_path(args.out_dir, "tts")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = await Qwen3TTSProvider().synthesize(
        args.text,
        reference_audio_path=args.reference_audio,
    )
    output_path.write_bytes(result.audio_bytes)
    print(f"Saved TTS audio: {output_path}")


def _print_asr_result(result: ASRResult) -> None:
    if result.language:
        print(f"[{result.language}] {result.text}")
    else:
        print(result.text)


def _parse_audio_device(value: str | None) -> int | str | None:
    if value is None:
        return None
    clean = value.strip()
    return int(clean) if clean.isdecimal() else clean


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Test local Qwen3 ASR/TTS providers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    asr_file = subparsers.add_parser("asr-file", help="Transcribe an audio file")
    asr_file.add_argument("audio_path", help="Path to a wav/mp3/m4a/flac audio file")
    asr_file.add_argument("--language", help="Optional forced language, e.g. English")
    asr_file.set_defaults(handler=_run_asr_file)

    asr_mic = subparsers.add_parser("asr-mic", help="Record mic audio and transcribe it")
    asr_mic.add_argument("--seconds", type=float, default=5.0, help="Recording length")
    asr_mic.add_argument("--out-dir", type=Path, default=Path("./tmp"))
    asr_mic.add_argument("--language", help="Optional forced language, e.g. English")
    asr_mic.add_argument("--device", help="Input device index or name")
    asr_mic.add_argument("--sample-rate", type=int, default=16_000)
    asr_mic.add_argument("--no-normalize", action="store_true")
    asr_mic.set_defaults(handler=_run_asr_mic)

    tts = subparsers.add_parser("tts", help="Synthesize text to a WAV file")
    tts.add_argument("text", help="Text to synthesize")
    tts.add_argument("--out-dir", type=Path, default=Path("./tmp"))
    tts.add_argument("--output", type=Path, help="Exact output WAV path")
    tts.add_argument("--reference-audio", type=Path, help="Optional voice reference WAV")
    tts.set_defaults(handler=_run_tts)

    args = parser.parse_args()

    await args.handler(args)


if __name__ == "__main__":
    asyncio.run(_main())
