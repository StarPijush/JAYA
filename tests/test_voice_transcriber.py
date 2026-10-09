"""
Transcriber Tests

The fake model is injected, so faster-whisper is never imported and no weights
are downloaded during a test run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from core.voice.exceptions import ModelLoadError, NoAudioError, TranscriptionError
from core.voice.transcriber import TRANSCRIPTION_SAMPLE_RATE, Transcriber, default_model_path


class FakeSegment:
    def __init__(self, text: str):
        self.text = text


class FakeModel:
    """Stand-in for WhisperModel recording how it was called."""

    def __init__(self, texts=None, error=None, generator_error=None):
        self._texts = texts if texts is not None else ["hello world"]
        self._error = error
        self._generator_error = generator_error
        self.calls: list[dict] = []

    def transcribe(self, audio, language=None, **kwargs):
        self.calls.append({"audio": audio, "language": language, "kwargs": kwargs})
        if self._error is not None:
            raise self._error
        return iter([FakeSegment(t) for t in self._texts]), object()


class ExplodingGenerator:
    def __init__(self, error):
        self._error = error

    def __iter__(self):
        raise self._error
        yield  # pragma: no cover - never reached


class FailingIterationModel:
    def __init__(self, error):
        self._error = error

    def transcribe(self, audio, language=None, **kwargs):
        return ExplodingGenerator(self._error), object()


def tone(samples: int = 8000, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(samples, dtype=np.float32) / TRANSCRIPTION_SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


# --- Construction ---


def test_injected_model_is_used_without_loading():
    model = FakeModel(["injected"])
    transcriber = Transcriber(model=model)
    assert transcriber.is_loaded is True
    assert transcriber.transcribe(tone()) == "injected"
    assert len(model.calls) == 1


def test_model_is_not_loaded_on_construction():
    transcriber = Transcriber(model_size="base")
    assert transcriber.is_loaded is False


def test_constructor_does_not_import_faster_whisper():
    """Constructing with an injected model must not pull in the ML dependency."""
    sys.modules.pop("faster_whisper", None)
    transcriber = Transcriber(model=FakeModel(["ok"]))
    transcriber.transcribe(tone())
    assert "faster_whisper" not in sys.modules


def test_model_loads_once_across_calls():
    model = FakeModel(["text"])
    transcriber = Transcriber(model=model)
    transcriber.transcribe(tone())
    transcriber.transcribe(tone())
    transcriber.transcribe(tone())
    assert len(model.calls) == 3


def test_model_instance_is_reused_not_recreated():
    model = FakeModel(["a", "b"])
    transcriber = Transcriber(model=model)
    transcriber.transcribe(tone())
    transcriber.transcribe(tone())
    assert transcriber._model is model


# --- Transcription output ---


def test_transcribe_joins_multiple_segments():
    transcriber = Transcriber(model=FakeModel(["Hello ", "there ", "friend"]))
    assert transcriber.transcribe(tone()) == "Hello there friend"


def test_transcribe_strips_leading_and_trailing_whitespace():
    transcriber = Transcriber(model=FakeModel(["   spaced text   "]))
    assert transcriber.transcribe(tone()) == "spaced text"


def test_transcribe_returns_empty_string_for_whitespace_only():
    transcriber = Transcriber(model=FakeModel(["   ", "\n", "\t"]))
    assert transcriber.transcribe(tone()) == ""


def test_transcribe_returns_empty_string_for_no_segments():
    transcriber = Transcriber(model=FakeModel([]))
    assert transcriber.transcribe(tone()) == ""


def test_transcribe_returns_plain_string_not_segment_objects():
    transcriber = Transcriber(model=FakeModel(["open chrome"]))
    result = transcriber.transcribe(tone())
    assert type(result) is str
    assert not hasattr(result, "text")


def test_transcribe_passes_configured_language():
    model = FakeModel(["bonjour"])
    Transcriber(language="fr", model=model).transcribe(tone())
    assert model.calls[0]["language"] == "fr"


def test_transcribe_passes_none_language_for_detection():
    model = FakeModel(["text"])
    Transcriber(language=None, model=model).transcribe(tone())
    assert model.calls[0]["language"] is None


def test_transcribe_passes_float32_audio_to_model():
    model = FakeModel(["text"])
    Transcriber(model=model).transcribe(tone())
    audio = model.calls[0]["audio"]
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32


# --- Input validation ---


def test_transcribe_rejects_empty_audio():
    transcriber = Transcriber(model=FakeModel(["text"]))
    with pytest.raises(NoAudioError):
        transcriber.transcribe(np.array([], dtype=np.float32))


def test_transcribe_rejects_silent_audio():
    transcriber = Transcriber(model=FakeModel(["text"]))
    with pytest.raises(NoAudioError):
        transcriber.transcribe(np.zeros(TRANSCRIPTION_SAMPLE_RATE, dtype=np.float32))


def test_transcribe_rejects_non_16khz_audio():
    transcriber = Transcriber(model=FakeModel(["text"]))
    with pytest.raises(TranscriptionError, match="Whisper requires 16000 Hz"):
        transcriber.transcribe(tone(), sample_rate=48000)


def test_wrong_sample_rate_is_rejected_before_model_runs():
    model = FakeModel(["text"])
    with pytest.raises(TranscriptionError):
        Transcriber(model=model).transcribe(tone(), sample_rate=44100)
    assert model.calls == []


# --- Error handling ---


def test_transcribe_wraps_inference_error():
    transcriber = Transcriber(model=FakeModel(error=RuntimeError("cuda out of memory")))
    with pytest.raises(TranscriptionError, match="Transcription failed"):
        transcriber.transcribe(tone())


def test_transcribe_wraps_error_raised_during_segment_iteration():
    transcriber = Transcriber(model=FailingIterationModel(RuntimeError("decode error")))
    with pytest.raises(TranscriptionError, match="decode error"):
        transcriber.transcribe(tone())


def test_model_load_error_is_distinct_from_transcription_error():
    model_error = ModelLoadError("could not load")
    inference_error = TranscriptionError("could not infer")
    assert isinstance(model_error, ModelLoadError)
    assert not isinstance(model_error, TranscriptionError)
    assert isinstance(inference_error, TranscriptionError)
    assert not isinstance(inference_error, ModelLoadError)


def test_missing_faster_whisper_raises_model_load_error(monkeypatch, tmp_path):
    """A None entry in sys.modules makes the import raise ImportError."""
    monkeypatch.setitem(sys.modules, "faster_whisper", None)

    transcriber = Transcriber(model_path=tmp_path)
    with pytest.raises(ModelLoadError, match="faster-whisper is not installed"):
        transcriber.transcribe(tone())


def test_model_initialization_failure_raises_model_load_error(monkeypatch, tmp_path):
    module = type("M", (), {})()

    def boom(*args, **kwargs):
        raise OSError("no network and not cached")

    module.WhisperModel = boom
    monkeypatch.setitem(sys.modules, "faster_whisper", module)

    transcriber = Transcriber(model_path=tmp_path)
    with pytest.raises(ModelLoadError) as info:
        transcriber.transcribe(tone())
    assert "requires internet access" in str(info.value)


def test_model_path_directory_is_created(monkeypatch, tmp_path):
    module = type("M", (), {})()
    module.WhisperModel = lambda *a, **k: FakeModel(["loaded"])
    monkeypatch.setitem(sys.modules, "faster_whisper", module)

    target = tmp_path / "nested" / "whisper"
    transcriber = Transcriber(model_path=target)
    transcriber.transcribe(tone())
    assert target.is_dir()


def test_whisper_model_receives_configured_arguments(monkeypatch, tmp_path):
    captured: dict = {}

    def fake_model(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeModel(["text"])

    module = type("M", (), {})()
    module.WhisperModel = fake_model
    monkeypatch.setitem(sys.modules, "faster_whisper", module)

    Transcriber(
        model_size="small",
        device="cpu",
        compute_type="int8",
        model_path=tmp_path,
    ).transcribe(tone())

    assert captured["args"][0] == "small"
    assert captured["kwargs"]["device"] == "cpu"
    assert captured["kwargs"]["compute_type"] == "int8"
    assert captured["kwargs"]["download_root"] == str(tmp_path)


# --- Model cache location ---


def test_default_model_path_is_outside_repository(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    path = default_model_path()
    assert path == tmp_path / "JARVIS" / "models" / "whisper"
    repo_root = Path(__file__).resolve().parents[1]
    assert repo_root not in path.parents


def test_default_model_path_falls_back_when_localappdata_missing(monkeypatch):
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    path = default_model_path()
    assert path.parts[-4:] == (".cache", "JARVIS", "models", "whisper")


def test_model_path_override_is_respected(tmp_path):
    transcriber = Transcriber(model_path=tmp_path / "custom")
    assert transcriber.model_path == tmp_path / "custom"
