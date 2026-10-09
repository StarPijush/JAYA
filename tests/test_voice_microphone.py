"""
Microphone Capture Tests

Uses fakes for PortAudio and synthetic audio. No microphone hardware, no
network access, and no model download is required.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.voice.exceptions import MicrophoneError, NoAudioError
from core.voice.microphone import (
    MIN_PEAK_AMPLITUDE,
    TRANSCRIPTION_SAMPLE_RATE,
    DeviceInfo,
    list_input_devices,
    record,
    resample_audio,
    resolve_capture_rate,
    resolve_device,
    validate_audio,
)


class FakePortAudioError(Exception):
    """Stand-in for sounddevice.PortAudioError."""


class FakeSD:
    """Minimal sounddevice stand-in driven by a device table."""

    PortAudioError = FakePortAudioError

    def __init__(self, raw_devices, hostapis=None, default_device=(0, 0), supported=None):
        self._raw_devices = raw_devices
        self._hostapis = hostapis or [{"name": "Windows WASAPI"}]
        self.default = type("Defaults", (), {"device": default_device})()
        self._supported = supported or {}
        self.checked_settings: list[dict] = []
        self.input_stream_args: dict | None = None

    def query_devices(self):
        return self._raw_devices

    def query_hostapis(self):
        return self._hostapis

    def check_input_settings(self, device=None, channels=None, dtype=None, samplerate=None):
        self.checked_settings.append(
            {"device": device, "channels": channels, "dtype": dtype, "samplerate": samplerate}
        )
        allowed = self._supported.get(device)
        if allowed is not None and samplerate not in allowed:
            raise FakePortAudioError(f"Invalid sample rate: {samplerate}")

    def InputStream(self, **kwargs):
        self.input_stream_args = kwargs
        return FakeStream(self)


class FakeStream:
    """Context-managed stream returning a fixed number of frames."""

    def __init__(self, sd, samples=None, short_read=False, error=None):
        self._sd = sd
        self._samples = np.zeros(1024, dtype=np.float32) if samples is None else samples
        self._short_read = short_read
        self._error = error
        self.started = False
        self.stopped = False

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def start(self):
        if self._error is not None:
            raise self._error
        self.started = True

    def read(self, frames):
        if self._error is not None:
            raise self._error
        count = max(1, frames // 4) if self._short_read else frames
        data = self._samples[:count]
        if len(data) == 0:
            return np.zeros(0, dtype=np.float32), False
        return data.astype(np.float32), False

    def stop(self):
        self.stopped = True


def make_device(index=0, name="Test Mic", rate=48000.0, channels=1, is_default=True):
    return DeviceInfo(
        index=index,
        name=name,
        hostapi="Windows WASAPI",
        default_sample_rate=rate,
        max_input_channels=channels,
        is_default=is_default,
    )


def tone(samples: int, rate: int = TRANSCRIPTION_SAMPLE_RATE, amplitude: float = 0.5) -> np.ndarray:
    """A quiet-ish synthetic speech-like signal, well above the silence floor."""
    t = np.arange(samples, dtype=np.float32) / rate
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


@pytest.fixture
def patch_sounddevice(monkeypatch):
    """Install a fake sounddevice module for the duration of a test."""

    def _install(sd: FakeSD) -> FakeSD:
        module = type("M", (), {})()
        for name in (
            "PortAudioError",
            "query_devices",
            "query_hostapis",
            "check_input_settings",
            "InputStream",
            "default",
        ):
            setattr(module, name, getattr(sd, name))
        monkeypatch.setitem(__import__("sys").modules, "sounddevice", module)
        return sd

    return _install


# --- Device discovery ---


def test_list_input_devices_returns_only_input_devices(patch_sounddevice):
    patch_sounddevice(
        FakeSD(
            [
                {"name": "Speakers", "hostapi": 0, "max_input_channels": 0, "default_samplerate": 44100.0},
                {"name": "Headset Mic", "hostapi": 0, "max_input_channels": 2, "default_samplerate": 48000.0},
                {"name": "Webcam Mic", "hostapi": 1, "max_input_channels": 1, "default_samplerate": 16000.0},
            ],
            hostapis=[{"name": "WASAPI"}, {"name": "WDM"}],
            default_device=(1, 0),
        )
    )

    devices = list_input_devices()

    assert [d.name for d in devices] == ["Headset Mic", "Webcam Mic"]
    assert devices[0].index == 1
    assert devices[0].hostapi == "WASAPI"
    assert devices[1].hostapi == "WDM"
    assert devices[0].is_default is True
    assert devices[1].is_default is False


def test_list_input_devices_raises_when_no_devices(patch_sounddevice):
    patch_sounddevice(FakeSD(None))
    with pytest.raises(MicrophoneError, match="No audio devices found"):
        list_input_devices()


def test_list_input_devices_wraps_portaudio_error(patch_sounddevice):
    class Failing(FakeSD):
        def query_devices(self):
            raise FakePortAudioError("device unavailable")

    patch_sounddevice(Failing([{"name": "x", "max_input_channels": 1}]))
    with pytest.raises(MicrophoneError, match="Could not query audio devices"):
        list_input_devices()


# --- Device resolution ---


def test_resolve_device_defaults_to_system_default(patch_sounddevice):
    patch_sounddevice(
        FakeSD(
            [
                {"name": "A", "hostapi": 0, "max_input_channels": 1, "default_samplerate": 48000.0},
                {"name": "B", "hostapi": 0, "max_input_channels": 1, "default_samplerate": 48000.0},
            ],
            default_device=(1, 0),
        )
    )
    assert resolve_device(None).name == "B"


def test_resolve_device_falls_back_to_first_when_no_default(patch_sounddevice):
    patch_sounddevice(
        FakeSD(
            [{"name": "Only", "hostapi": 0, "max_input_channels": 1, "default_samplerate": 48000.0}],
            default_device=(None, None),
        )
    )
    assert resolve_device(None).name == "Only"


def test_resolve_device_by_index(patch_sounddevice):
    patch_sounddevice(
        FakeSD(
            [
                {"name": "A", "hostapi": 0, "max_input_channels": 1, "default_samplerate": 48000.0},
                {"name": "B", "hostapi": 0, "max_input_channels": 1, "default_samplerate": 48000.0},
            ]
        )
    )
    assert resolve_device(0).name == "A"


def test_resolve_device_rejects_invalid_index_and_lists_devices(patch_sounddevice):
    patch_sounddevice(
        FakeSD(
            [{"name": "A", "hostapi": 0, "max_input_channels": 1, "default_samplerate": 48000.0}]
        )
    )
    with pytest.raises(MicrophoneError) as info:
        resolve_device(7)
    assert "No input device with index 7" in str(info.value)
    assert "Available input devices" in str(info.value)


def test_resolve_device_rejects_negative_index(patch_sounddevice):
    patch_sounddevice(
        FakeSD(
            [{"name": "A", "hostapi": 0, "max_input_channels": 1, "default_samplerate": 48000.0}]
        )
    )
    with pytest.raises(MicrophoneError, match="Invalid microphone index -1"):
        resolve_device(-1)


def test_resolve_device_raises_when_no_input_devices(patch_sounddevice):
    patch_sounddevice(FakeSD([{"name": "Speakers", "max_input_channels": 0}]))
    with pytest.raises(MicrophoneError, match="No input devices were detected"):
        resolve_device(None)


# --- Sample rate handling ---


def test_resolve_capture_rate_prefers_transcription_rate():
    sd = FakeSD([], supported={0: {16000}})
    assert resolve_capture_rate(make_device(rate=44100.0), sd) == float(TRANSCRIPTION_SAMPLE_RATE)


def test_resolve_capture_rate_falls_back_to_device_default():
    sd = FakeSD([], supported={0: {44100}})
    assert resolve_capture_rate(make_device(rate=44100.0), sd) == 44100.0


def test_resolve_capture_rate_raises_when_no_rate_works():
    sd = FakeSD([], supported={0: set()})
    with pytest.raises(MicrophoneError, match="does not support mono capture"):
        resolve_capture_rate(make_device(rate=44100.0), sd)


def test_check_input_settings_requests_mono_float32():
    sd = FakeSD([], supported={0: {16000}})
    resolve_capture_rate(make_device(), sd)
    assert sd.checked_settings[0] == {
        "device": 0,
        "channels": 1,
        "dtype": "float32",
        "samplerate": 16000,
    }


def test_resample_audio_is_identity_at_transcription_rate():
    audio = tone(800)
    assert resample_audio(audio, TRANSCRIPTION_SAMPLE_RATE) is audio


@pytest.mark.parametrize("source_rate", [48000, 44100, 32000, 8000, 22050])
def test_resample_audio_produces_expected_length(source_rate):
    seconds = 1.0
    audio = tone(int(seconds * source_rate), rate=source_rate)
    resampled = resample_audio(audio, source_rate)
    expected = round(seconds * TRANSCRIPTION_SAMPLE_RATE)
    assert abs(len(resampled) - expected) <= 1
    assert resampled.dtype == np.float32


def test_resample_audio_preserves_signal_level():
    audio = tone(48000, rate=48000, amplitude=0.4)
    resampled = resample_audio(audio, 48000)
    assert 0.3 < float(np.max(np.abs(resampled))) < 0.5


def test_resample_audio_attenuates_content_above_new_nyquist():
    """An 11 kHz tone at 48 kHz must be attenuated, where naive decimation would alias it."""
    rate = 48000
    t = np.arange(rate, dtype=np.float32) / rate
    out_of_band = np.sin(2 * np.pi * 11000 * t).astype(np.float32)

    resampled = resample_audio(out_of_band, rate)
    input_rms = float(np.sqrt(np.mean(out_of_band**2)))
    resampled_rms = float(np.sqrt(np.mean(resampled**2)))

    assert resampled_rms < 0.02 * input_rms
    assert float(np.sqrt(np.mean(out_of_band[::3] ** 2))) > 0.5 * input_rms


def test_resample_audio_keeps_content_below_new_nyquist():
    """A 3 kHz speech-range tone must survive resampling at full amplitude."""
    rate = 48000
    t = np.arange(rate, dtype=np.float32) / rate
    in_band = np.sin(2 * np.pi * 3000 * t).astype(np.float32)

    resampled = resample_audio(in_band, rate)
    input_rms = float(np.sqrt(np.mean(in_band**2)))
    resampled_rms = float(np.sqrt(np.mean(resampled**2)))

    assert 0.9 < resampled_rms / input_rms < 1.1


# --- Audio validation ---


def test_validate_audio_rejects_none():
    with pytest.raises(NoAudioError, match="No audio was captured"):
        validate_audio(None)


def test_validate_audio_rejects_empty_array():
    with pytest.raises(NoAudioError, match="empty recording"):
        validate_audio(np.array([], dtype=np.float32))


def test_validate_audio_rejects_pure_silence():
    with pytest.raises(NoAudioError, match="No audible signal"):
        validate_audio(np.zeros(16000, dtype=np.float32))


def test_validate_audio_rejects_non_finite_samples():
    audio = tone(1000)
    audio[10] = np.nan
    with pytest.raises(NoAudioError, match="invalid samples"):
        validate_audio(audio)


def test_validate_accepts_quiet_speech_above_floor():
    quiet = tone(16000, amplitude=MIN_PEAK_AMPLITUDE * 5)
    validated = validate_audio(quiet)
    assert len(validated) == 16000
    assert validated.dtype == np.float32


def test_validate_audio_downmixes_multichannel_to_mono():
    stereo = np.ones((16000, 2), dtype=np.float32) * 0.2
    validated = validate_audio(stereo)
    assert validated.ndim == 1
    assert validated.shape == (16000,)
    assert float(np.max(np.abs(validated))) == pytest.approx(0.2, abs=1e-6)


def test_validate_audio_flattens_singleton_channel_axis():
    validated = validate_audio(np.ones((1, 16000), dtype=np.float32) * 0.3)
    assert validated.shape == (16000,)


def test_validate_audio_rejects_unexpected_shape():
    with pytest.raises(NoAudioError, match="Unexpected audio shape"):
        validate_audio(np.zeros((32, 16000), dtype=np.float32))


def test_validate_audio_returns_contiguous_float32():
    validated = validate_audio(tone(1600, amplitude=0.3).astype(np.float64))
    assert validated.dtype == np.float32
    assert validated.flags["C_CONTIGUOUS"]


# --- Recording ---


def test_record_returns_audio_at_transcription_rate(patch_sounddevice):
    sd = patch_sounddevice(
        FakeSD(
            [{"name": "Mic", "hostapi": 0, "max_input_channels": 1, "default_samplerate": 48000.0}],
            supported={0: {48000}},
        )
    )
    stream = FakeStream(sd, samples=tone(48000, rate=48000))

    audio = record(seconds=1.0, stream_factory=lambda **kw: stream)

    assert abs(len(audio) - 16000) <= 1
    assert audio.dtype == np.float32
    assert stream.started is True
    assert stream.stopped is True


def test_record_opens_mono_float32_stream(patch_sounddevice):
    sd = patch_sounddevice(
        FakeSD(
            [{"name": "Mic", "hostapi": 0, "max_input_channels": 1, "default_samplerate": 16000.0}],
            supported={0: {16000}},
        )
    )
    stream = FakeStream(sd, samples=tone(16000))
    captured: dict = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return stream

    record(seconds=1.0, stream_factory=factory)

    assert captured == {"device": 0, "channels": 1, "dtype": "float32", "samplerate": 16000.0}


def test_record_handles_short_reads(patch_sounddevice):
    sd = patch_sounddevice(
        FakeSD(
            [{"name": "Mic", "hostapi": 0, "max_input_channels": 1, "default_samplerate": 16000.0}],
            supported={0: {16000}},
        )
    )
    stream = FakeStream(sd, samples=tone(16000), short_read=True)

    audio = record(seconds=1.0, stream_factory=lambda **kw: stream)

    assert abs(len(audio) - 16000) <= 1


def test_record_wraps_portaudio_access_denied(patch_sounddevice):
    sd = patch_sounddevice(
        FakeSD(
            [{"name": "Mic", "hostapi": 0, "max_input_channels": 1, "default_samplerate": 16000.0}],
            supported={0: {16000}},
        )
    )
    stream = FakeStream(sd, error=FakePortAudioError("Access denied"))

    with pytest.raises(MicrophoneError, match="Privacy & security > Microphone"):
        record(seconds=1.0, stream_factory=lambda **kw: stream)


def test_record_wraps_portaudio_device_in_use(patch_sounddevice):
    sd = patch_sounddevice(
        FakeSD(
            [{"name": "Mic", "hostapi": 0, "max_input_channels": 1, "default_samplerate": 16000.0}],
            supported={0: {16000}},
        )
    )
    stream = FakeStream(sd, error=FakePortAudioError("Device unavailable"))

    with pytest.raises(MicrophoneError, match="in use by another application"):
        record(seconds=1.0, stream_factory=lambda **kw: stream)


def test_record_wraps_generic_portaudio_error(patch_sounddevice):
    sd = patch_sounddevice(
        FakeSD(
            [{"name": "Mic", "hostapi": 0, "max_input_channels": 1, "default_samplerate": 16000.0}],
            supported={0: {16000}},
        )
    )
    stream = FakeStream(sd, error=FakePortAudioError("some other failure"))

    with pytest.raises(MicrophoneError, match="Could not record from microphone"):
        record(seconds=1.0, stream_factory=lambda **kw: stream)


def test_record_wraps_device_disconnection(patch_sounddevice):
    sd = patch_sounddevice(
        FakeSD(
            [{"name": "Mic", "hostapi": 0, "max_input_channels": 1, "default_samplerate": 16000.0}],
            supported={0: {16000}},
        )
    )
    stream = FakeStream(sd, error=OSError("device removed"))

    with pytest.raises(MicrophoneError, match="was disconnected"):
        record(seconds=1.0, stream_factory=lambda **kw: stream)


def test_record_stops_stream_when_read_fails(patch_sounddevice):
    sd = patch_sounddevice(
        FakeSD(
            [{"name": "Mic", "hostapi": 0, "max_input_channels": 1, "default_samplerate": 16000.0}],
            supported={0: {16000}},
        )
    )

    class ExplodingStream(FakeStream):
        def read(self, frames):
            raise FakePortAudioError("stream failed")

    stream = ExplodingStream(sd, samples=tone(16000))
    stream._error = None

    with pytest.raises(MicrophoneError):
        record(seconds=1.0, stream_factory=lambda **kw: stream)

    assert stream.stopped is True


def test_record_rejects_silence(patch_sounddevice):
    sd = patch_sounddevice(
        FakeSD(
            [{"name": "Mic", "hostapi": 0, "max_input_channels": 1, "default_samplerate": 16000.0}],
            supported={0: {16000}},
        )
    )
    stream = FakeStream(sd, samples=np.zeros(16000, dtype=np.float32))

    with pytest.raises(NoAudioError, match="No audible signal"):
        record(seconds=1.0, stream_factory=lambda **kw: stream)


def test_record_raises_when_stream_returns_nothing(patch_sounddevice):
    sd = patch_sounddevice(
        FakeSD(
            [{"name": "Mic", "hostapi": 0, "max_input_channels": 1, "default_samplerate": 16000.0}],
            supported={0: {16000}},
        )
    )
    stream = FakeStream(sd, samples=np.array([], dtype=np.float32))

    with pytest.raises(NoAudioError, match="No audio was captured"):
        record(seconds=1.0, stream_factory=lambda **kw: stream)


def test_record_rejects_non_positive_duration(patch_sounddevice):
    patch_sounddevice(FakeSD([]))
    with pytest.raises(MicrophoneError, match="must be greater than 0"):
        record(seconds=0.0)


def test_record_uses_requested_device_index(patch_sounddevice):
    sd = patch_sounddevice(
        FakeSD(
            [
                {"name": "A", "hostapi": 0, "max_input_channels": 1, "default_samplerate": 16000.0},
                {"name": "B", "hostapi": 0, "max_input_channels": 1, "default_samplerate": 16000.0},
            ],
            supported={0: {16000}, 1: {16000}},
        )
    )
    stream = FakeStream(sd, samples=tone(16000))
    captured: dict = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return stream

    record(seconds=1.0, device_index=1, stream_factory=factory)

    assert captured["device"] == 1
