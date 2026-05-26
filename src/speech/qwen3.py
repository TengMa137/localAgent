"""CrispASR-backed Qwen speech providers."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import mimetypes
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from localagent_settings import SpeechSettings


_ASR_TEXT_TAG = "<asr_text>"
_LANG_PREFIX = "language "


def _optional_text(value: object | None) -> str:
    return value.strip() if isinstance(value, str) else ""


def _speech_settings() -> SpeechSettings:
    return SpeechSettings()


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
    """Runtime config for CrispASR Qwen3-ASR.

    The default assumes a persistent CrispASR server loaded from
    ``./models/qwen3-asr-1.7b-q8_0.gguf``.
    Set ``base_url`` to an empty string only for ad hoc CLI fallback.
    """

    base_url: str = field(default_factory=lambda: _speech_settings().asr_base_url)
    cli_path: str = field(default_factory=lambda: _speech_settings().speech_cli)
    backend: str = field(default_factory=lambda: _speech_settings().asr_backend)
    model: str = field(default_factory=lambda: _speech_settings().asr_model)
    api_key: str = field(default_factory=lambda: _speech_settings().asr_api_key)
    timeout_seconds: float = field(
        default_factory=lambda: _speech_settings().asr_timeout_seconds
    )
    max_tokens: int = field(default_factory=lambda: _speech_settings().asr_max_tokens)
    temperature: float = field(
        default_factory=lambda: _speech_settings().asr_temperature
    )
    response_format: str = field(
        default_factory=lambda: _speech_settings().asr_response_format
    )
    language: str = field(default_factory=lambda: _speech_settings().asr_language)
    threads: int | None = field(default_factory=lambda: _speech_settings().asr_threads)
    vad: bool = field(default_factory=lambda: _speech_settings().asr_vad)
    output_json: bool = field(
        default_factory=lambda: _speech_settings().asr_output_json
    )


class Qwen3ASRError(RuntimeError):
    """Raised when CrispASR cannot produce a transcript."""


@dataclass(frozen=True)
class Qwen3TTSConfig:
    """Runtime config for CrispASR Qwen3-TTS."""

    base_url: str = field(default_factory=lambda: _speech_settings().tts_base_url)
    cli_path: str = field(default_factory=lambda: _speech_settings().tts_cli_path)
    backend: str = field(default_factory=lambda: _speech_settings().tts_backend)
    model: str = field(default_factory=lambda: _speech_settings().tts_model)
    api_key: str = field(default_factory=lambda: _speech_settings().tts_api_key)
    timeout_seconds: float = field(
        default_factory=lambda: _speech_settings().tts_timeout_seconds
    )
    voice: str = field(default_factory=lambda: _speech_settings().tts_voice)
    voice_dir: str = field(default_factory=lambda: _speech_settings().tts_voice_dir)
    reference_text: str = field(default_factory=lambda: _speech_settings().tts_ref_text)
    codec_model: str = field(default_factory=lambda: _speech_settings().tts_codec_model)
    language: str = field(default_factory=lambda: _speech_settings().tts_language)
    instructions: str = field(
        default_factory=lambda: _speech_settings().tts_instructions
    )
    response_format: str = field(
        default_factory=lambda: _speech_settings().tts_response_format
    )
    temperature: float | None = field(
        default_factory=lambda: _speech_settings().tts_temperature
    )
    speed: float | None = field(default_factory=lambda: _speech_settings().tts_speed)
    threads: int | None = field(default_factory=lambda: _speech_settings().tts_threads)


class Qwen3TTSError(RuntimeError):
    """Raised when CrispASR cannot produce audio."""


class Qwen3ASRProvider:
    """Transcribe audio through CrispASR with the Qwen3 ASR backend.

    Preferred server mode:

        crispasr --server --backend qwen3 -m auto --port 8081

    For ad hoc use, set ``LOCALAGENT_ASR_BASE_URL=`` and optionally override
    ``LOCALAGENT_ASR_MODEL=auto`` so the CLI path can use CrispASR's downloader.
    """

    provider_name = "crispasr-qwen3-asr"

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
        return await self.transcribe_bytes(
            audio_bytes,
            filename=f"audio{mimetypes.guess_extension(mime_type) or '.wav'}",
            mime_type=mime_type,
            language=language,
        )

    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "audio.wav",
        mime_type: str = "audio/wav",
        language: str | None = None,
    ) -> ASRResult:
        if not self.config.base_url.strip():
            suffix = mimetypes.guess_extension(mime_type) or ".wav"
            temp_path: Path | None = None
            try:
                handle = tempfile.NamedTemporaryFile(
                    prefix="localagent-asr-upload-",
                    suffix=suffix,
                    delete=False,
                )
                temp_path = Path(handle.name)
                handle.close()
                temp_path.write_bytes(audio_bytes)
                return await self.transcribe_file(temp_path, language=language)
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
        return await asyncio.to_thread(
            self._transcribe_bytes_sync,
            audio_bytes,
            filename,
            mime_type,
            language,
        )

    def _transcribe_file_sync(self, path: Path, language: str | None) -> ASRResult:
        if self.config.base_url.strip():
            mime_type = mimetypes.guess_type(path.name)[0] or "audio/wav"
            return self._transcribe_bytes_sync(
                path.read_bytes(), path.name, mime_type, language
            )
        return self._transcribe_cli_sync(path, language)

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
            with urllib.request.urlopen(
                request, timeout=self.config.timeout_seconds
            ) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise Qwen3ASRError(f"CrispASR ASR HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise Qwen3ASRError(
                f"Could not reach CrispASR ASR server at {url}: {exc.reason}"
            ) from exc

        try:
            raw = _extract_transcription_text(json.loads(body))
        except json.JSONDecodeError:
            raw = body.strip()
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
            _multipart_field(boundary, "response_format", self.config.response_format),
            _multipart_field(boundary, "temperature", str(self.config.temperature)),
        ]
        request_language = _optional_text(language) or self.config.language
        if request_language:
            parts.append(_multipart_field(boundary, "language", request_language))
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

    def _transcribe_cli_sync(self, path: Path, language: str | None) -> ASRResult:
        handle = tempfile.NamedTemporaryFile(
            prefix="localagent-asr-",
            delete=False,
        )
        output_base = Path(handle.name)
        handle.close()
        output_base.unlink(missing_ok=True)
        json_path = output_base.with_suffix(".json")
        txt_path = output_base.with_suffix(".txt")
        cmd = self._build_transcription_command(
            path,
            output_base=output_base,
            language=language,
        )

        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            raise Qwen3ASRError(
                f"Could not find CrispASR CLI: {self.config.cli_path}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise Qwen3ASRError("CrispASR ASR timed out") from exc

        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise Qwen3ASRError(f"CrispASR ASR failed: {detail}")

        raw = ""
        try:
            if json_path.exists():
                raw = _extract_crispasr_json_text(json.loads(json_path.read_text()))
            elif txt_path.exists():
                raw = txt_path.read_text().strip()
            else:
                raw = _clean_cli_transcript(completed.stdout)
        finally:
            output_base.unlink(missing_ok=True)
            json_path.unlink(missing_ok=True)
            txt_path.unlink(missing_ok=True)

        parsed_language, text = parse_asr_output(raw, user_language=language)
        return ASRResult(
            text=text,
            language=parsed_language,
            raw_text=raw,
            provider=self.provider_name,
        )

    def _build_transcription_command(
        self,
        audio_path: Path,
        *,
        output_base: Path,
        language: str | None,
    ) -> list[str]:
        cmd = [
            self.config.cli_path,
            "--backend",
            self.config.backend,
            "-m",
            self.config.model,
            "-f",
            str(audio_path),
            "-of",
            str(output_base),
            "-np",
        ]
        if self.config.output_json:
            cmd.append("-oj")
        else:
            cmd.append("-otxt")
        request_language = _optional_text(language) or self.config.language
        if request_language:
            cmd.extend(["-l", request_language])
        if self.config.vad:
            cmd.append("--vad")
        if self.config.threads is not None:
            cmd.extend(["-t", str(self.config.threads)])
        _append_optional(cmd, "-tp", self.config.temperature)
        _append_optional(cmd, "-n", self.config.max_tokens)
        return cmd


class Qwen3TTSProvider:
    """Synthesize speech through CrispASR's Qwen3-TTS backend.

    The default path calls a persistent CrispASR TTS server loaded from
    ``./models/qwen3-tts-12hz-0.6b-customvoice-q8_0.gguf``. Set
    ``LOCALAGENT_TTS_BASE_URL=`` only for ad hoc CLI fallback.
    """

    provider_name = "crispasr-qwen3-tts"

    def __init__(self, config: Qwen3TTSConfig | None = None) -> None:
        self.config = config or Qwen3TTSConfig()

    async def synthesize(
        self,
        text: str,
        *,
        reference_audio_path: str | Path | None = None,
        reference_text: str | None = None,
        voice: str | None = None,
        instructions: str | None = None,
        speed: float | None = None,
    ) -> TTSResult:
        return await asyncio.to_thread(
            self._synthesize_sync,
            text,
            Path(reference_audio_path).expanduser().resolve()
            if reference_audio_path
            else None,
            reference_text,
            voice,
            instructions,
            speed,
        )

    async def synthesize_with_reference_base64(
        self,
        text: str,
        reference_audio_base64: str,
        *,
        reference_mime_type: str = "audio/wav",
        reference_text: str | None = None,
        speed: float | None = None,
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
            return await self.synthesize(
                text,
                reference_audio_path=reference_path,
                reference_text=reference_text,
                speed=speed,
            )
        finally:
            if reference_path is not None:
                reference_path.unlink(missing_ok=True)

    def _synthesize_sync(
        self,
        text: str,
        reference_audio_path: Path | None,
        reference_text: str | None = None,
        voice: str | None = None,
        instructions: str | None = None,
        speed: float | None = None,
    ) -> TTSResult:
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("text is required")
        if reference_audio_path is not None and not reference_audio_path.exists():
            raise FileNotFoundError(reference_audio_path)
        if self.config.base_url.strip():
            return self._synthesize_http_sync(
                clean_text,
                reference_audio_path=reference_audio_path,
                reference_text=reference_text,
                voice=voice,
                instructions=instructions,
                speed=speed,
            )

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
                    reference_text=reference_text,
                    voice=voice,
                    instructions=instructions,
                    speed=speed,
                ),
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            output_path.unlink(missing_ok=True)
            raise Qwen3TTSError(
                f"Could not find CrispASR CLI: {self.config.cli_path}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            output_path.unlink(missing_ok=True)
            raise Qwen3TTSError("CrispASR TTS timed out") from exc

        if completed.returncode != 0:
            output_path.unlink(missing_ok=True)
            detail = (completed.stderr or completed.stdout or "").strip()
            raise Qwen3TTSError(f"CrispASR TTS failed: {detail}")

        try:
            audio_bytes = output_path.read_bytes()
        finally:
            output_path.unlink(missing_ok=True)
        if not audio_bytes:
            raise Qwen3TTSError("CrispASR TTS produced an empty WAV file")
        return TTSResult(
            audio_bytes=audio_bytes,
            mime_type="audio/wav",
            provider=self.provider_name,
        )

    def _synthesize_http_sync(
        self,
        text: str,
        *,
        reference_audio_path: Path | None,
        reference_text: str | None,
        voice: str | None,
        instructions: str | None,
        speed: float | None,
    ) -> TTSResult:
        request_voice = self._resolve_voice(reference_audio_path, voice)
        request_instructions = _optional_text(instructions) or self.config.instructions
        payload: dict[str, Any] = {
            "model": self.config.model,
            "input": text,
            "response_format": self.config.response_format,
        }
        if request_voice:
            payload["voice"] = request_voice
        if request_instructions:
            payload["instructions"] = request_instructions
        if self.config.language:
            payload["language"] = self.config.language
        request_speed = speed if speed is not None else self.config.speed
        if request_speed is not None:
            payload["speed"] = request_speed
        url = self.config.base_url.rstrip("/") + "/audio/speech"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.timeout_seconds
            ) as resp:
                audio_bytes = resp.read()
                mime_type = resp.headers.get_content_type() or "audio/wav"
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            hint = self._tts_http_error_hint(
                url=url,
                detail=detail,
                request_voice=request_voice,
            )
            raise Qwen3TTSError(
                f"CrispASR TTS HTTP {exc.code}: {detail}{hint}"
            ) from exc
        except urllib.error.URLError as exc:
            raise Qwen3TTSError(
                f"Could not reach CrispASR TTS server at {url}: {exc.reason}"
            ) from exc

        if not audio_bytes:
            raise Qwen3TTSError("CrispASR TTS server returned empty audio")
        return TTSResult(
            audio_bytes=audio_bytes,
            mime_type=mime_type,
            provider=self.provider_name,
        )

    def _tts_http_error_hint(
        self,
        *,
        url: str,
        detail: str,
        request_voice: str,
    ) -> str:
        if "synthesis_failed" not in detail and "empty audio" not in detail:
            return ""

        context = (
            f"request voice={request_voice!r}"
            if request_voice
            else "request used server startup voice"
        )
        voices = self._fetch_server_voice_names(url)
        if voices:
            context += f"; server voices={', '.join(voices[:12])}"
            if len(voices) > 12:
                context += f", ... (+{len(voices) - 12} more)"
        elif voices == []:
            context += "; server reported no named voices"
        context += "; check LOCALAGENT_TTS_VOICE against the CrispASR server backend"
        return f" ({context})"

    def _fetch_server_voice_names(self, speech_url: str) -> list[str] | None:
        voices_url = speech_url.rsplit("/", 1)[0] + "/voices"
        request = urllib.request.Request(
            voices_url,
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=min(max(self.config.timeout_seconds, 0.1), 2.0),
            ) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

        raw_voices = payload.get("voices") if isinstance(payload, dict) else None
        if not isinstance(raw_voices, list):
            return None

        names: list[str] = []
        for item in raw_voices:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                name = item["name"].strip()
                if name:
                    names.append(name)
            elif isinstance(item, str) and item.strip():
                names.append(item.strip())
        return names

    def _build_synthesis_command(
        self,
        text: str,
        *,
        output_path: Path,
        reference_audio_path: Path | None = None,
        reference_text: str | None = None,
        voice: str | None = None,
        instructions: str | None = None,
        speed: float | None = None,
    ) -> list[str]:
        cmd = [
            self.config.cli_path,
            "--backend",
            self.config.backend,
            "-m",
            self.config.model,
            "--tts",
            text,
            "--tts-output",
            str(output_path),
        ]
        request_voice = self._resolve_voice(reference_audio_path, voice)
        if request_voice:
            cmd.extend(["--voice", request_voice])
        ref_text = _optional_text(reference_text) or self.config.reference_text
        if ref_text:
            cmd.extend(["--ref-text", ref_text])
        if self.config.voice_dir:
            cmd.extend(["--voice-dir", self.config.voice_dir])
        if self.config.codec_model:
            cmd.extend(["--codec-model", self.config.codec_model])
        request_instructions = _optional_text(instructions) or self.config.instructions
        if request_instructions:
            cmd.extend(["--instruct", request_instructions])
        if self.config.language:
            cmd.extend(["-l", self.config.language])
        request_speed = speed if speed is not None else self.config.speed
        _append_optional(cmd, "--speed", request_speed)
        _append_optional(cmd, "--temperature", self.config.temperature)
        if self.config.threads is not None:
            cmd.extend(["-t", str(self.config.threads)])
        return cmd

    def _resolve_voice(
        self,
        reference_audio_path: Path | None,
        voice: str | None,
    ) -> str:
        if reference_audio_path is not None:
            return str(reference_audio_path)
        if isinstance(voice, str):
            return voice.strip()
        return self.config.voice.strip()


def parse_asr_output(
    raw: str | None, user_language: str | None = None
) -> tuple[str, str]:
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


def _extract_crispasr_json_text(payload: dict[str, object]) -> str:
    text = payload.get("text")
    if isinstance(text, str):
        return text.strip()

    transcription = payload.get("transcription")
    if isinstance(transcription, list):
        parts: list[str] = []
        for item in transcription:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"].strip())
        return " ".join(part for part in parts if part).strip()

    return _clean_cli_transcript(json.dumps(payload, ensure_ascii=False))


def _clean_cli_transcript(stdout: str) -> str:
    lines = []
    for line in stdout.splitlines():
        clean = line.strip()
        if not clean:
            continue
        if clean.startswith("[") and "-->" in clean:
            clean = clean.split("]", 1)[-1].strip()
        if clean.lower().startswith(("crispasr:", "whisper_", "ggml_")):
            continue
        lines.append(clean)
    return " ".join(lines).strip()


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

    asr_mic = subparsers.add_parser(
        "asr-mic", help="Record mic audio and transcribe it"
    )
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
    tts.add_argument(
        "--reference-audio", type=Path, help="Optional voice reference WAV"
    )
    tts.set_defaults(handler=_run_tts)

    args = parser.parse_args()

    await args.handler(args)


if __name__ == "__main__":
    asyncio.run(_main())
