"""
Voice Service Tests

Recording and transcription are exercised independently through injected
fakes, so no microphone or model is required.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.voice.exceptions import NoAudioError
from core.voice.service import VoiceService
from core.voice.transcriber import TRANSCRIPTION_SAMPLE_RATE, Transcriber


class FakeSegment:
    def __init__(self, text: str):
        self.text = text


class FakeModel:
    def __init__(self, texts=None):
        self._texts = texts if texts is not None else ["open chrome"]

    def transcribe(self, audio, language=None, **kwargs):
        return iter([FakeSegment(t) for t in self._texts]), object()


def tone(samples: int = 16000, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(samples, dtype=np.float32) / TRANSCRIPTION_SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


class RecordingRecorder:
    """Fake recorder capturing the arguments it was called with."""

    def __init__(self, audio=None, error=None):
        self.audio = tone() if audio is None else audio
        self.error = error
        self.calls: list[dict] = []

    def __call__(self, seconds=None, device_index=None):
        self.calls.append({"seconds": seconds, "device_index": device_index})
        if self.error is not None:
            raise self.error
        return self.audio


def test_listen_and_transcribe_returns_text():
    recorder = RecordingRecorder()
    service = VoiceService(transcriber=Transcriber(model=FakeModel(["open chrome"])), recorder=recorder)

    assert service.listen_and_transcribe() == "open chrome"


def test_record_passes_configured_defaults():
    recorder = RecordingRecorder()
    service = VoiceService(
        transcriber=Transcriber(model=FakeModel()),
        recorder=recorder,
        record_seconds=7.5,
        device_index=3,
    )

    service.record()

    assert recorder.calls == [{"seconds": 7.5, "device_index": 3}]


def test_record_accepts_explicit_duration_override():
    recorder = RecordingRecorder()
    service = VoiceService(
        transcriber=Transcriber(model=FakeModel()),
        recorder=recorder,
        record_seconds=5.0,
    )

    service.record(seconds=2.0)

    assert recorder.calls[0]["seconds"] == 2.0


def test_record_and_transcribe_are_independently_usable():
    recorder = RecordingRecorder()
    service = VoiceService(transcriber=Transcriber(model=FakeModel(["first"])), recorder=recorder)

    audio = service.record()
    assert service.transcribe(audio) == "first"
    assert service.transcribe(audio) == "first"

    assert len(recorder.calls) == 1


def test_record_propagates_microphone_error():
    from core.voice.exceptions import MicrophoneError

    recorder = RecordingRecorder(error=MicrophoneError("permission denied"))
    service = VoiceService(transcriber=Transcriber(model=FakeModel()), recorder=recorder)

    with pytest.raises(MicrophoneError, match="permission denied"):
        service.record()


def test_transcribe_returns_empty_string_when_no_speech():
    service = VoiceService(
        transcriber=Transcriber(model=FakeModel(["  "])), recorder=RecordingRecorder()
    )
    assert service.listen_and_transcribe() == ""


def test_transcribe_rejects_silent_recording():
    service = VoiceService(
        transcriber=Transcriber(model=FakeModel(["text"])),
        recorder=RecordingRecorder(audio=np.zeros(16000, dtype=np.float32)),
    )
    with pytest.raises(NoAudioError):
        service.listen_and_transcribe()


def test_default_transcriber_is_created_when_omitted():
    service = VoiceService(recorder=RecordingRecorder())
    assert isinstance(service.transcriber, Transcriber)
    assert service.transcriber.model_size == "base"


def test_from_settings_maps_configuration(monkeypatch):
    from core.config.settings import Settings

    monkeypatch.setenv(
        "VOICE_MODEL_SIZE", "small"
    )
    monkeypatch.setenv("VOICE_LANGUAGE", "fr")
    monkeypatch.setenv("VOICE_RECORD_SECONDS", "9")
    monkeypatch.setenv("VOICE_MICROPHONE_DEVICE", "2")
    from core.config import settings as settings_module

    monkeypatch.setattr(settings_module, "get_settings", lambda: Settings())

    service = VoiceService.from_settings(recorder=RecordingRecorder())

    assert service.transcriber.model_size == "small"
    assert service.transcriber.language == "fr"
    assert service.record_seconds == 9.0
    assert service.device_index == 2
