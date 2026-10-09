"""
Voice Module

Local microphone capture and speech-to-text. Public interfaces only; see
``core.voice.microphone``, ``core.voice.transcriber``, and ``core.voice.service``
for the implementations.
"""
from core.voice.exceptions import (
    MicrophoneError,
    ModelLoadError,
    NoAudioError,
    TranscriptionError,
    VoiceError,
)
from core.voice.microphone import (
    TRANSCRIPTION_SAMPLE_RATE,
    DeviceInfo,
    list_input_devices,
    record,
    resolve_device,
)
from core.voice.service import VoiceService
from core.voice.transcriber import Transcriber

__all__ = [
    "TRANSCRIPTION_SAMPLE_RATE",
    "DeviceInfo",
    "MicrophoneError",
    "ModelLoadError",
    "NoAudioError",
    "Transcriber",
    "TranscriptionError",
    "VoiceError",
    "VoiceService",
    "list_input_devices",
    "record",
    "resolve_device",
]
