"""
Speech-to-Text Transcription

Wraps a faster-whisper model behind a small interface that returns plain text.
The model is loaded lazily on first use, once per instance, and never during
import.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from core.voice.exceptions import ModelLoadError, TranscriptionError
from core.voice.microphone import TRANSCRIPTION_SAMPLE_RATE, validate_audio

logger = logging.getLogger(__name__)


class WhisperModelProtocol(Protocol):
    """The subset of the faster-whisper model interface this module relies on."""

    def transcribe(self, audio: np.ndarray, language: str | None = None, **kwargs: Any) -> Any:
        ...


def default_model_path() -> Path:
    """
    Resolve the model cache directory outside the repository.

    Uses %LOCALAPPDATA%\\JARVIS\\models\\whisper on Windows so downloaded weights
    never land in the project tree.
    """
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "JARVIS" / "models" / "whisper"
    return Path.home() / ".cache" / "JARVIS" / "models" / "whisper"


class Transcriber:
    """
    Transcribes audio into text using faster-whisper.

    Args:
        model_size: Whisper model identifier, e.g. "base" or "small.en".
        language: Spoken language code, or None to let Whisper detect it.
        device: Inference device, one of "cpu", "cuda", or "auto".
        compute_type: CTranslate2 compute type, e.g. "int8" for CPU.
        model_path: Cache directory for downloaded weights.
        model: Pre-built model, primarily for testing. When supplied, no model
            is loaded and faster-whisper is never imported.
    """

    def __init__(
        self,
        model_size: str = "base",
        language: str | None = "en",
        device: str = "cpu",
        compute_type: str = "int8",
        model_path: str | Path | None = None,
        model: WhisperModelProtocol | None = None,
    ) -> None:
        self.model_size = model_size
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.model_path = Path(model_path) if model_path else default_model_path()
        self._model = model
        self._model_loaded = model is not None

    @property
    def is_loaded(self) -> bool:
        """Whether the underlying model has been initialized."""
        return self._model_loaded

    def _load_model(self) -> WhisperModelProtocol:
        """Load the model once, reusing the instance on later calls."""
        if self._model is not None:
            return self._model

        try:
            self.model_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ModelLoadError(
                f"Could not create the model directory '{self.model_path}': {exc}"
            ) from exc

        logger.info(
            "Loading Whisper model '%s' (device=%s, compute_type=%s) into %s",
            self.model_size,
            self.device,
            self.compute_type,
            self.model_path,
        )

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ModelLoadError(
                "faster-whisper is not installed. Run: uv sync"
            ) from exc

        try:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(self.model_path),
            )
        except Exception as exc:
            raise ModelLoadError(
                f"Could not load Whisper model '{self.model_size}': {exc}. "
                "The first run downloads the model and requires internet access."
            ) from exc

        self._model_loaded = True
        return self._model

    def transcribe(self, audio: np.ndarray, sample_rate: int = TRANSCRIPTION_SAMPLE_RATE) -> str:
        """
        Transcribe mono audio into plain text.

        Returns an empty string when no speech was recognized. Leading and
        trailing whitespace is removed and Whisper segment objects never escape.
        """
        if sample_rate != TRANSCRIPTION_SAMPLE_RATE:
            raise TranscriptionError(
                f"Whisper requires {TRANSCRIPTION_SAMPLE_RATE} Hz audio, got {sample_rate} Hz."
            )

        samples = validate_audio(audio)
        model = self._load_model()

        try:
            segments, _info = model.transcribe(samples, language=self.language)
            text = "".join(segment.text for segment in segments)
        except Exception as exc:
            raise TranscriptionError(f"Transcription failed: {exc}") from exc

        return text.strip()
