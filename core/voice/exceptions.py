"""
Voice Exceptions

Recoverable error types raised by the voice subsystem. Each one signals a
condition the caller can report to the user and continue from, rather than an
unrecoverable programming fault.
"""


class VoiceError(Exception):
    """Base class for all recoverable voice subsystem errors."""


class MicrophoneError(VoiceError):
    """Microphone is unavailable, busy, inaccessible, or disconnected."""


class NoAudioError(VoiceError):
    """Recording produced no usable audio signal."""


class ModelLoadError(VoiceError):
    """The speech-to-text model could not be initialized."""


class TranscriptionError(VoiceError):
    """The model failed while transcribing audio."""
