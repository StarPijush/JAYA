"""
Voice Service

Combines microphone capture and speech-to-text behind one small interface.
Recording and transcription stay independently usable and independently testable.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from core.voice.microphone import record as record_audio
from core.voice.transcriber import Transcriber

logger = logging.getLogger(__name__)


class VoiceService:
    """
    Records audio and transcribes it into text.

    Args:
        transcriber: Transcriber instance. Created from defaults when omitted.
        recorder: Callable performing the recording. Injectable so tests need
            no microphone hardware.
        record_seconds: Default recording length.
        device_index: Microphone index, or None for the system default.
    """

    def __init__(
        self,
        transcriber: Transcriber | None = None,
        recorder: Callable[..., np.ndarray] | None = None,
        record_seconds: float = 5.0,
        device_index: int | None = None,
    ) -> None:
        self.transcriber = transcriber if transcriber is not None else Transcriber()
        self._recorder = recorder if recorder is not None else record_audio
        self.record_seconds = record_seconds
        self.device_index = device_index

    def record(self, seconds: float | None = None) -> np.ndarray:
        """Record a single session and return audio at the transcription sample rate."""
        duration = self.record_seconds if seconds is None else seconds
        return self._recorder(seconds=duration, device_index=self.device_index)

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe already-captured audio."""
        return self.transcriber.transcribe(audio)

    def listen_and_transcribe(self, seconds: float | None = None) -> str:
        """
        Record one session and return the recognized text.

        Returns an empty string when no speech was recognized.
        """
        audio = self.record(seconds)
        return self.transcribe(audio)

    @classmethod
    def from_settings(cls, **overrides: Any) -> VoiceService:
        """Build a service from application settings, with optional overrides."""
        from core.config.settings import get_settings

        settings = get_settings()
        return cls(
            transcriber=Transcriber(
                model_size=settings.voice_model_size,
                language=settings.voice_language,
                device=settings.voice_device,
                compute_type=settings.voice_compute_type,
                model_path=settings.voice_model_path,
            ),
            record_seconds=settings.voice_record_seconds,
            device_index=settings.voice_microphone_device,
            **overrides,
        )
