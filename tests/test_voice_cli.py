"""
Voice CLI Tests

Exercises argument parsing, device listing, Ctrl+C handling, and error paths.
Nothing here loads Whisper or touches audio hardware.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.voice.__main__ import HEADER, build_parser, main
from core.voice.exceptions import MicrophoneError, NoAudioError
from core.voice.microphone import TRANSCRIPTION_SAMPLE_RATE, DeviceInfo


def tone(samples: int = 16000, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(samples, dtype=np.float32) / TRANSCRIPTION_SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


class FakeSegment:
    def __init__(self, text: str):
        self.text = text


class FakeTranscriber:
    """Mirrors the real Transcriber's public attributes without loading Whisper."""

    def __init__(self, texts, **kwargs):
        self.model_size = kwargs.get("model_size", "base")
        self.language = kwargs.get("language", "en")
        self.device = kwargs.get("device", "cpu")
        self.compute_type = kwargs.get("compute_type", "int8")
        self._texts = texts

    def transcribe(self, audio):
        return "".join(self._texts).strip()


@pytest.fixture
def fake_service(monkeypatch):
    """Replace VoiceService with a controllable fake. Records constructor args."""
    state: dict = {"instances": [], "texts": ["open chrome"], "error": None}

    class FakeVoiceService:
        def __init__(self, transcriber=None, record_seconds=5.0, device_index=None, **kwargs):
            self.transcriber = transcriber
            self.record_seconds = record_seconds
            self.device_index = device_index
            self.record_calls = 0
            state["instances"].append(self)

        def record(self, seconds=None):
            self.record_calls += 1
            if state["error"] is not None:
                raise state["error"]
            return tone()

        def transcribe(self, audio):
            return self.transcriber.transcribe(audio)

    monkeypatch.setattr("core.voice.__main__.VoiceService", FakeVoiceService)

    def fake_transcriber(**kwargs):
        return FakeTranscriber(state["texts"], **kwargs)

    monkeypatch.setattr("core.voice.__main__.Transcriber", fake_transcriber)
    return state


# --- Argument parsing ---


def test_parser_defaults_are_none_so_settings_win():
    args = build_parser().parse_args([])
    assert args.list_devices is False
    for attribute in ("model", "language", "device", "compute_type", "seconds", "microphone"):
        assert getattr(args, attribute) is None


def test_parser_accepts_all_options():
    args = build_parser().parse_args(
        [
            "--list-devices",
            "--model", "small",
            "--language", "es",
            "--device", "cuda",
            "--compute-type", "float16",
            "--seconds", "12",
            "--microphone", "4",
        ]
    )
    assert args.list_devices is True
    assert args.model == "small"
    assert args.language == "es"
    assert args.device == "cuda"
    assert args.compute_type == "float16"
    assert args.seconds == 12.0
    assert args.microphone == 4


def test_parser_rejects_non_numeric_seconds():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--seconds", "abc"])


# --- --list-devices ---


def test_list_devices_prints_real_portaudio_data(monkeypatch, capsys):
    devices = [
        DeviceInfo(0, "Headset Microphone", "Windows WASAPI", 48000.0, 1, True),
        DeviceInfo(2, "Webcam Microphone", "Windows WDM", 16000.0, 1, False),
    ]
    monkeypatch.setattr("core.voice.__main__.list_input_devices", lambda: devices)

    exit_code = main(["--list-devices"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "[0] Headset Microphone" in output
    assert "[2] Webcam Microphone" in output
    assert "(system default)" in output
    assert "48000 Hz" in output


def test_list_devices_does_not_load_whisper(monkeypatch, capsys):
    monkeypatch.setattr("core.voice.__main__.list_input_devices", list)
    monkeypatch.setattr(
        "core.voice.transcriber.Transcriber._load_model",
        lambda self: pytest.fail("Whisper must not load for --list-devices"),
    )

    main(["--list-devices"])

    assert "No input devices were detected." in capsys.readouterr().out


def test_list_devices_reports_when_none_found(monkeypatch, capsys):
    monkeypatch.setattr("core.voice.__main__.list_input_devices", list)

    assert main(["--list-devices"]) == 1
    assert "No input devices were detected." in capsys.readouterr().out


def test_list_devices_reports_microphone_error(monkeypatch, capsys):
    def boom():
        raise MicrophoneError("PortAudio missing")

    monkeypatch.setattr("core.voice.__main__.list_input_devices", boom)

    assert main(["--list-devices"]) == 1
    assert "PortAudio missing" in capsys.readouterr().err


# --- Recording session ---


def test_session_prints_header_and_recognized_text(fake_service, monkeypatch, capsys):
    monkeypatch.setattr("core.voice.__main__._device_label", lambda index: "Default microphone")
    fake_service["texts"] = ["open chrome"]

    exit_code = main([])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert HEADER in output
    assert "Model: base" in output
    assert "Recording for 5 seconds..." in output
    assert "Processing speech..." in output
    assert "Recognized: open chrome" in output


def test_session_records_exactly_once(fake_service, monkeypatch):
    monkeypatch.setattr("core.voice.__main__._device_label", lambda index: "Default microphone")
    main([])
    assert fake_service["instances"][0].record_calls == 1


def test_cli_arguments_override_settings(fake_service, monkeypatch):
    monkeypatch.setattr("core.voice.__main__._device_label", lambda index: "Default microphone")
    monkeypatch.setenv("VOICE_MODEL_SIZE", "tiny")
    monkeypatch.setenv("VOICE_RECORD_SECONDS", "3")

    main(["--model", "large-v3", "--seconds", "9", "--microphone", "2"])

    instance = fake_service["instances"][0]
    assert instance.record_seconds == 9.0
    assert instance.device_index == 2
    assert instance.transcriber.model_size == "large-v3"


def test_session_reports_no_speech(fake_service, monkeypatch, capsys):
    monkeypatch.setattr("core.voice.__main__._device_label", lambda index: "Default microphone")
    fake_service["texts"] = ["   "]

    exit_code = main([])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "No speech recognized." in output
    assert "Recognized:" not in output


def test_session_reports_microphone_error(fake_service, monkeypatch, capsys):
    monkeypatch.setattr("core.voice.__main__._device_label", lambda index: "Default microphone")
    fake_service["error"] = MicrophoneError("Microphone is in use by another application")

    exit_code = main([])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "in use by another application" in captured.err
    assert "Traceback" not in captured.err


def test_session_reports_no_audio_error(fake_service, monkeypatch, capsys):
    monkeypatch.setattr("core.voice.__main__._device_label", lambda index: "Default microphone")
    fake_service["error"] = NoAudioError("No audible signal was detected.")

    assert main([]) == 1
    assert "No audible signal" in capsys.readouterr().err


# --- Keyboard interrupt ---


def test_ctrl_c_during_recording_exits_cleanly(fake_service, monkeypatch, capsys):
    monkeypatch.setattr("core.voice.__main__._device_label", lambda index: "Default microphone")
    fake_service["error"] = KeyboardInterrupt

    exit_code = main([])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Cancelled." in output
    assert "Traceback" not in output


def test_ctrl_c_during_transcription_exits_cleanly(monkeypatch, capsys):
    class InterruptingService:
        def __init__(self, transcriber=None, record_seconds=5.0, device_index=None, **kwargs):
            self.transcriber = transcriber
            self.record_seconds = record_seconds
            self.device_index = device_index

        def record(self, seconds=None):
            return tone()

        def transcribe(self, audio):
            raise KeyboardInterrupt

    monkeypatch.setattr("core.voice.__main__.VoiceService", InterruptingService)
    monkeypatch.setattr("core.voice.__main__._device_label", lambda index: "Default microphone")

    exit_code = main([])

    assert exit_code == 0
    assert "Cancelled." in capsys.readouterr().out


def test_ctrl_c_does_not_print_traceback(fake_service, monkeypatch, capsys):
    monkeypatch.setattr("core.voice.__main__._device_label", lambda index: "Default microphone")
    fake_service["error"] = KeyboardInterrupt

    main([])

    assert "Traceback" not in capsys.readouterr().out
