"""Terminal Enter-to-talk voice input for the local agent."""

from __future__ import annotations

import asyncio
import array
import select
import shutil
import sys
import termios
import tempfile
import threading
import time
import tty
import wave
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from .qwen3 import Qwen3ASRProvider


TextHandler = Callable[[str], Awaitable[None]]


class VoiceRuntimeError(RuntimeError):
    """Raised when terminal voice input cannot be started."""


@dataclass(frozen=True)
class RecordedAudio:
    path: Path
    elapsed: float
    byte_count: int
    peak: float
    rms: float
    gain: float
    input_peak: float
    input_rms: float


class TerminalRecorder:
    """Record microphone PCM between two terminal prompts."""

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        channels: int = 1,
        min_seconds: float = 0.7,
        device: int | str | None = None,
        normalize_audio: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.min_seconds = min_seconds
        self.device = device
        self.normalize_audio = normalize_audio
        self._sounddevice = _import_required(
            "sounddevice",
            "Install terminal voice dependencies first: uv add sounddevice",
        )
        self._frames: list[bytes] = []
        self._lock = threading.Lock()
        self._stream: Any = None
        self._started_at = 0.0
        self._level = 0.0

    @property
    def elapsed_seconds(self) -> float:
        if self._stream is None:
            return 0.0
        return time.monotonic() - self._started_at

    @property
    def level(self) -> float:
        with self._lock:
            return self._level

    def start(self) -> None:
        if self._stream is not None:
            return
        self._frames.clear()
        self._level = 0.0
        self._started_at = time.monotonic()

        def callback(indata: bytes, frames: int, time_info: Any, status: Any) -> None:
            del frames, time_info
            if status:
                print(f"\n[voice warn] {status}")
            chunk = bytes(indata)
            level = _audio_level(chunk)
            with self._lock:
                self._frames.append(chunk)
                self._level = max(level, self._level * 0.82)

        self._stream = self._sounddevice.RawInputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            device=self.device,
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> RecordedAudio | None:
        if self._stream is None:
            return None

        elapsed = time.monotonic() - self._started_at
        stream = self._stream
        self._stream = None
        stream.stop()
        stream.close()

        with self._lock:
            frames = b"".join(self._frames)
            self._frames.clear()

        if not frames or elapsed < self.min_seconds:
            return None

        input_peak, input_rms = _audio_stats(frames)
        gain = 1.0
        output_frames = frames
        if self.normalize_audio:
            output_frames, gain = _normalize_pcm16(frames, input_peak)
        peak, rms = _audio_stats(output_frames)

        handle = tempfile.NamedTemporaryFile(
            prefix="localagent-voice-",
            suffix=".wav",
            delete=False,
        )
        path = Path(handle.name)
        handle.close()

        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(self.channels)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(output_frames)
        return RecordedAudio(
            path=path,
            elapsed=elapsed,
            byte_count=len(frames),
            peak=peak,
            rms=rms,
            gain=gain,
            input_peak=input_peak,
            input_rms=input_rms,
        )


async def run_enter_to_talk(
    handle_text: TextHandler,
    *,
    language: str | None = None,
    device: int | str | None = None,
    normalize_audio: bool = True,
    save_audio_dir: Path | None = None,
    save_failed_audio_dir: Path | None = None,
) -> None:
    """Record one utterance per Enter start/stop cycle."""

    recorder = TerminalRecorder(device=device, normalize_audio=normalize_audio)
    asr = Qwen3ASRProvider()
    print("Voice mode: Enter starts recording, Enter stops. Press q to quit.")
    if save_audio_dir is not None:
        save_audio_dir.mkdir(parents=True, exist_ok=True)
        print(f"Debug audio will be saved under {save_audio_dir}")
    if save_failed_audio_dir is not None:
        save_failed_audio_dir.mkdir(parents=True, exist_ok=True)
        print(f"Failed ASR audio will be saved under {save_failed_audio_dir}")

    with _raw_terminal():
        while True:
            print("\rVoice > ", end="", flush=True)
            key = await _read_key()
            if key.lower() == "q":
                print()
                return
            if key not in {"\r", "\n"}:
                continue

            recorder.start()

            while True:
                elapsed = recorder.elapsed_seconds
                print(f"\r{_listening_status(elapsed, recorder.level)}", end="", flush=True)
                if await _key_available():
                    key = await _read_key()
                    if key in {"\r", "\n"} and elapsed >= recorder.min_seconds:
                        break
                    if key.lower() == "q":
                        break
                await asyncio.sleep(0.08)

            recording = recorder.stop()
            _clear_status_line()
            if recording is None:
                print("No audio recorded")
                continue

            try:
                saved_path: Path | None = None
                if save_audio_dir is not None:
                    saved_path = save_audio_dir / f"voice-{time.time_ns()}.wav"
                    shutil.copy2(recording.path, saved_path)
                    print(f"Saved ASR audio: {saved_path}")
                print(
                    f"Transcribing {recording.elapsed:.1f}s "
                    f"(input_peak={recording.input_peak:.1%}, "
                    f"gain={recording.gain:.1f}x, "
                    f"peak={recording.peak:.1%}, rms={recording.rms:.1%})..."
                )
                result = await asr.transcribe_file(recording.path, language=language)
                if not result.text:
                    if save_failed_audio_dir is not None and saved_path is None:
                        saved_path = save_failed_audio_dir / f"failed-{time.time_ns()}.wav"
                        shutil.copy2(recording.path, saved_path)
                    print(
                        "No speech detected. Check the input device and macOS "
                        "Microphone permission."
                    )
                    if saved_path is not None:
                        print(f"Saved failed ASR audio: {saved_path}")
                    if recording.input_peak < 0.01:
                        print(
                            "Recorded level is near silence. Try "
                            "`--list-audio-devices` and pass the real mic with "
                            "`--voice-device <index>`."
                        )
                    continue
                print(f"You: {result.text}\n")
                await handle_text(result.text)
            finally:
                recording.path.unlink(missing_ok=True)


def _import_required(module_name: str, install_hint: str) -> ModuleType:
    try:
        return __import__(module_name, fromlist=["*"])
    except ImportError as exc:
        raise VoiceRuntimeError(install_hint) from exc


def _audio_level(chunk: bytes) -> float:
    peak, _rms = _audio_stats(chunk)
    return peak


def _audio_stats(chunk: bytes) -> tuple[float, float]:
    if not chunk:
        return 0.0, 0.0
    samples = array.array("h")
    samples.frombytes(chunk)
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return 0.0, 0.0
    peak = max(abs(sample) for sample in samples)
    rms = (sum(sample * sample for sample in samples) / len(samples)) ** 0.5
    return min(1.0, peak / 32768), min(1.0, rms / 32768)


def _normalize_pcm16(
    chunk: bytes,
    peak: float,
    *,
    target_peak: float = 0.65,
    max_gain: float = 18.0,
) -> tuple[bytes, float]:
    if not chunk or peak <= 0.001:
        return chunk, 1.0

    gain = min(max_gain, max(1.0, target_peak / peak))
    if gain <= 1.01:
        return chunk, 1.0

    samples = array.array("h")
    samples.frombytes(chunk)
    needs_swap = sys.byteorder != "little"
    if needs_swap:
        samples.byteswap()

    for idx, sample in enumerate(samples):
        boosted = int(sample * gain)
        samples[idx] = max(-32768, min(32767, boosted))

    if needs_swap:
        samples.byteswap()
    return samples.tobytes(), gain


def _listening_status(elapsed: float, level: float) -> str:
    width = 30
    active = min(width, int(level * width * 1.8))
    wave = "#" * active + "-" * (width - active)
    return f"Listening {elapsed:04.1f}s [{wave}]  Enter=stop"


def _clear_status_line() -> None:
    print("\r\x1b[2K", end="", flush=True)


class _raw_terminal:
    def __enter__(self) -> None:
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)


async def _read_key() -> str:
    return await asyncio.to_thread(sys.stdin.read, 1)


async def _key_available() -> bool:
    return await asyncio.to_thread(_stdin_ready)


def _stdin_ready() -> bool:
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    return bool(ready)


def list_input_devices() -> None:
    sounddevice = _import_required(
        "sounddevice",
        "Install terminal voice dependencies first: uv add sounddevice",
    )
    devices = sounddevice.query_devices()
    print("Input audio devices:")
    for idx, device in enumerate(devices):
        if int(device.get("max_input_channels", 0)) <= 0:
            continue
        default_marker = ""
        defaults = sounddevice.default.device
        if isinstance(defaults, (list, tuple)) and defaults and defaults[0] == idx:
            default_marker = " (default)"
        name = device.get("name", "unknown")
        channels = device.get("max_input_channels", "?")
        print(f"  {idx}: {name} [{channels} input channels]{default_marker}")
