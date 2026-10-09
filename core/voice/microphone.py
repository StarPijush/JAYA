"""
Microphone Capture

Short, explicit recording sessions from a selected input device. Audio is
captured as mono float32 and resampled to the 16 kHz rate the Whisper feature
extractor requires before being handed off for transcription.

This module does not implement continuous background listening.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import gcd
from typing import Any

import numpy as np

from core.voice.exceptions import MicrophoneError, NoAudioError

logger = logging.getLogger(__name__)

# Rate the Whisper feature extractor expects. Fixed by the model, not configurable.
TRANSCRIPTION_SAMPLE_RATE = 16000

# Peak amplitude below roughly -80 dBFS. Low enough to keep quiet speech, high
# enough to catch a muted stream or unplugged device returning zeros.
MIN_PEAK_AMPLITUDE = 1e-4

StreamFactory = Callable[..., Any]


@dataclass(frozen=True)
class DeviceInfo:
    """An input device reported by PortAudio."""

    index: int
    name: str
    hostapi: str
    default_sample_rate: float
    max_input_channels: int
    is_default: bool


def _sounddevice() -> Any:
    """Import sounddevice lazily so unit tests need no audio hardware."""
    try:
        import sounddevice as sd
    except Exception as exc:  # pragma: no cover - import failure is environment specific
        raise MicrophoneError(
            "sounddevice could not be imported. PortAudio may be missing or blocked."
        ) from exc
    return sd


def _describe_devices(devices: Sequence[DeviceInfo]) -> str:
    if not devices:
        return "No input devices were detected. Check that a microphone is connected."
    lines = [f"  [{d.index}] {d.name} ({d.default_sample_rate:.0f} Hz)" for d in devices]
    return "Available input devices:\n" + "\n".join(lines)


def list_input_devices() -> list[DeviceInfo]:
    """Return every device that can record audio, using real PortAudio data."""
    sd = _sounddevice()
    try:
        raw_devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        default_index = sd.default.device[0]
    except sd.PortAudioError as exc:
        raise MicrophoneError(f"Could not query audio devices: {exc}") from exc

    if not raw_devices:
        raise MicrophoneError("No audio devices found. Check that a microphone is connected.")

    devices: list[DeviceInfo] = []
    for index, device in enumerate(raw_devices):
        if device.get("max_input_channels", 0) <= 0:
            continue
        hostapi_index = device.get("hostapi", 0)
        hostapi_name = "unknown"
        try:
            hostapi_name = hostapis[hostapi_index]["name"]
        except (IndexError, KeyError, TypeError):
            pass
        devices.append(
            DeviceInfo(
                index=index,
                name=str(device.get("name", f"Device {index}")),
                hostapi=str(hostapi_name),
                default_sample_rate=float(device.get("default_samplerate", 0.0)),
                max_input_channels=int(device.get("max_input_channels", 0)),
                is_default=index == default_index,
            )
        )
    return devices


def resolve_device(device_index: int | None = None) -> DeviceInfo:
    """Resolve a device index to an input device, defaulting to the system default."""
    devices = list_input_devices()

    if not devices:
        raise MicrophoneError("No input devices were detected. " + _describe_devices(devices))

    if device_index is None:
        for device in devices:
            if device.is_default:
                return device
        return devices[0]

    if device_index < 0:
        raise MicrophoneError(
            f"Invalid microphone index {device_index}. " + _describe_devices(devices)
        )

    for device in devices:
        if device.index == device_index:
            if device.max_input_channels < 1:
                raise MicrophoneError(
                    f"Device {device.index} ({device.name}) cannot record audio."
                )
            return device

    raise MicrophoneError(
        f"No input device with index {device_index}. " + _describe_devices(devices)
    )


def _check_settings(sd: Any, device: DeviceInfo, samplerate: float) -> bool:
    """Confirm the device supports mono capture at this rate."""
    try:
        sd.check_input_settings(
            device=device.index, channels=1, dtype="float32", samplerate=samplerate
        )
    except sd.PortAudioError:
        return False
    return True


def resolve_capture_rate(device: DeviceInfo, sd: Any | None = None) -> float:
    """
    Choose the capture sample rate for a device.

    Prefers 16 kHz so the common path needs no resampling at all. Falls back to
    the rate the device reports as its default.
    """
    sd = sd if sd is not None else _sounddevice()

    if _check_settings(sd, device, TRANSCRIPTION_SAMPLE_RATE):
        return float(TRANSCRIPTION_SAMPLE_RATE)

    fallback = device.default_sample_rate
    if fallback > 0 and _check_settings(sd, device, fallback):
        logger.info(
            "Device %d does not support %d Hz; capturing at %.0f Hz and resampling.",
            device.index,
            TRANSCRIPTION_SAMPLE_RATE,
            fallback,
        )
        return float(fallback)

    raise MicrophoneError(
        f"Device {device.index} ({device.name}) does not support mono capture at "
        f"{TRANSCRIPTION_SAMPLE_RATE} Hz or its default rate of {fallback:.0f} Hz."
    )


def resample_audio(audio: np.ndarray, source_rate: float) -> np.ndarray:
    """
    Band-limited resample to the transcription rate using a polyphase FIR filter.

    The up/down factors are derived from the greatest common divisor so rates
    like 48000 -> 16000 become an exact integer 1:3 ratio rather than a float
    approximation.
    """
    if source_rate == TRANSCRIPTION_SAMPLE_RATE:
        return audio

    from scipy.signal import resample_poly

    source = round(source_rate)
    divisor = gcd(source, TRANSCRIPTION_SAMPLE_RATE)
    if divisor <= 0:
        raise NoAudioError(f"Invalid capture rate {source_rate} Hz.")
    up = TRANSCRIPTION_SAMPLE_RATE // divisor
    down = source // divisor

    resampled = resample_poly(audio, up, down)
    return resampled.astype(np.float32, copy=False)


def validate_audio(audio: np.ndarray | None) -> np.ndarray:
    """
    Confirm a recording is usable, returning it as contiguous float32.

    Only rejects audio that cannot contain speech at all. Whisper's own
    no-speech detection decides whether a real signal contains words.
    """
    if audio is None:
        raise NoAudioError("No audio was captured.")

    array = np.asarray(audio)

    if array.size == 0:
        raise NoAudioError("No audio was captured (empty recording).")

    if array.ndim > 2:
        raise NoAudioError(f"Unexpected audio shape {array.shape}; expected mono audio.")
    elif array.ndim == 2:
        if 1 in array.shape:
            array = array.reshape(-1)
        elif min(array.shape) <= 8:
            array = array.mean(axis=1)
        else:
            raise NoAudioError(
                f"Unexpected audio shape {array.shape}; expected mono or multichannel audio."
            )

    array = np.ascontiguousarray(array, dtype=np.float32)

    if not np.all(np.isfinite(array)):
        raise NoAudioError("Recording contained invalid samples (NaN or infinity).")

    if float(np.max(np.abs(array))) < MIN_PEAK_AMPLITUDE:
        raise NoAudioError(
            "No audible signal was detected. Check the microphone connection and input level."
        )

    return array


def _read_frames(stream: Any, frame_count: int) -> np.ndarray:
    """Pull frames from an open stream, tolerating short reads."""
    chunks: list[np.ndarray] = []
    remaining = frame_count
    while remaining > 0:
        data, _overflowed = stream.read(remaining)
        if data is None or len(data) == 0:
            break
        chunks.append(np.asarray(data, dtype=np.float32).reshape(-1))
        remaining -= len(chunks[-1])
    if not chunks:
        raise NoAudioError("No audio was captured from the microphone.")
    return np.concatenate(chunks)


def record(
    seconds: float = 5.0,
    device_index: int | None = None,
    stream_factory: StreamFactory | None = None,
) -> np.ndarray:
    """
    Record a single mono session and return audio at the transcription sample rate.

    Args:
        seconds: Length of the recording.
        device_index: Input device to use, or None for the system default.
        stream_factory: Callable used to open the stream. Defaults to
            ``sounddevice.InputStream``; injectable so tests need no hardware.
    """
    if seconds <= 0:
        raise MicrophoneError(f"Recording duration must be greater than 0, got {seconds}.")

    sd = _sounddevice()
    device = resolve_device(device_index)
    capture_rate = resolve_capture_rate(device, sd)
    open_stream = stream_factory if stream_factory is not None else sd.InputStream

    frame_count = max(1, round(seconds * capture_rate))

    try:
        with open_stream(
            device=device.index,
            channels=1,
            samplerate=capture_rate,
            dtype="float32",
        ) as stream:
            stream.start()
            try:
                audio = _read_frames(stream, frame_count)
            finally:
                stream.stop()
    except MicrophoneError:
        raise
    except sd.PortAudioError as exc:
        raise MicrophoneError(_portaudio_message(exc, device)) from exc
    except OSError as exc:
        raise MicrophoneError(
            f"Microphone {device.index} ({device.name}) was disconnected: {exc}"
        ) from exc

    audio = resample_audio(audio, capture_rate)
    return validate_audio(audio)


def _portaudio_message(exc: Exception, device: DeviceInfo) -> str:
    """Turn a PortAudio error into a message a user can act on."""
    text = str(exc)
    lowered = text.lower()
    if "access denied" in lowered or "permission" in lowered:
        return (
            f"Windows denied microphone access for '{device.name}'. "
            "Enable microphone access in Settings > Privacy & security > Microphone."
        )
    if "unavailable" in lowered or "in use" in lowered or "exclusive" in lowered:
        return (
            f"Microphone '{device.name}' is unavailable or in use by another application. "
            "Close other apps using it and try again."
        )
    if "invalid sample rate" in lowered or "invalid sample" in lowered:
        return f"Microphone '{device.name}' rejected the requested sample rate: {text}"
    return f"Could not record from microphone '{device.name}': {text}"
