from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LocalAgentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LOCALAGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )


class AgentRuntimeSettings(LocalAgentSettings):
    model_base_url: str = "http://localhost:8080/v1"
    model_api_key: str = "no-key"
    docs_dir: Path = Path("./user_docs")
    skills_dir: Path = Path("skills")
    skills_mode: Literal["ro", "rw"] = "rw"
    mcp_url: str = "http://localhost:8000/sse"
    memory_dir: Path = Path(".memory")
    memory_enabled: bool = True
    use_regex: bool = True
    structured_output_attempts: int = 3
    structured_output_max_tokens: int = 2048
    answer_output_max_tokens: int = 4096
    disable_model_thinking: bool = True
    model_request_timeout_seconds: float = 180
    approve_tools: str = ""
    max_approval_rounds: int = 3
    log_level: str = ""
    trace: str = ""

    @field_validator("skills_mode", mode="before")
    @classmethod
    def parse_skills_mode(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value


@lru_cache
def get_runtime_settings() -> AgentRuntimeSettings:
    return AgentRuntimeSettings()


class SpeechSettings(LocalAgentSettings):
    speech_cli: str = "crispasr"

    asr_base_url: str = "http://localhost:8081/v1"
    asr_backend: str = "qwen3"
    asr_model: str = "./models/qwen3-asr-1.7b-q8_0.gguf"
    asr_api_key: str = "no-key"
    asr_timeout_seconds: float = 300
    asr_max_tokens: int = 512
    asr_temperature: float = 0
    asr_response_format: str = "json"
    asr_language: str = ""
    asr_threads: int | None = None
    asr_vad: bool = False
    asr_output_json: bool = True

    tts_base_url: str = "http://localhost:8082/v1"
    tts_cli: str | None = None
    tts_backend: str = "qwen3-tts-customvoice"
    tts_model: str = "./models/qwen3-tts-12hz-0.6b-customvoice-q8_0.gguf"
    tts_api_key: str = "no-key"
    tts_timeout_seconds: float = 600
    tts_voice: str = "vivian"
    tts_voice_dir: str = ""
    tts_ref_text: str = ""
    tts_codec_model: str = "./models/qwen3-tts-tokenizer-12hz.gguf"
    tts_language: str = ""
    tts_instructions: str = ""
    tts_response_format: str = "wav"
    tts_temperature: float | None = None
    tts_speed: float | None = None
    tts_threads: int | None = None
    tts_player: str = ""
    tts_min_chars: int | None = None
    tts_max_chars: int | None = None
    tts_min_sentence_chars: int | None = None
    tts_initial_max_chars: int | None = None
    tts_phrase_boundary_chars: int | None = None

    @property
    def tts_cli_path(self) -> str:
        return self.tts_cli or self.speech_cli
