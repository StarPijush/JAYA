"""
Settings Configuration

Centralized settings using Pydantic Settings for environment variable management.
"""
from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: str = "development"
    log_level: str = "INFO"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    debug: bool = True

    # Future LLM settings (optional)
    llm_provider: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_temperature: float = 0.7

    # Future voice settings (optional)
    wake_word: str | None = None
    stt_model: str | None = None
    tts_voice: str | None = None

    # Voice recognition (Phase 1)
    voice_model_size: str = "base"
    voice_language: str = "en"
    voice_device: str = "cpu"
    voice_compute_type: str = "int8"
    voice_record_seconds: float = 5.0
    voice_microphone_device: int | None = None
    voice_model_path: str | None = None

    @field_validator("voice_device")
    @classmethod
    def _validate_voice_device(cls, value: str) -> str:
        allowed = {"cpu", "cuda", "auto"}
        if value not in allowed:
            raise ValueError(
                f"VOICE_DEVICE must be one of {sorted(allowed)}, got '{value}'"
            )
        return value

    @field_validator("voice_record_seconds")
    @classmethod
    def _validate_voice_record_seconds(cls, value: float) -> float:
        if value <= 0:
            raise ValueError(f"VOICE_RECORD_SECONDS must be greater than 0, got {value}")
        if value > 300:
            raise ValueError(f"VOICE_RECORD_SECONDS must be 300 or less, got {value}")
        return value

    @field_validator("voice_microphone_device")
    @classmethod
    def _validate_voice_microphone_device(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError(
                f"VOICE_MICROPHONE_DEVICE must be a non-negative index or unset, got {value}"
            )
        return value

    @field_validator("voice_model_size", "voice_compute_type", "voice_language")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Voice settings must not be empty")
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()