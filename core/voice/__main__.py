"""
Voice CLI Entry Point

Run a single voice recognition session.

    uv run python -m core.voice
    uv run python -m core.voice --list-devices

This entry point is independent of the FastAPI service and never starts it.
"""
from __future__ import annotations

import argparse
import sys

from core.config.logging import setup_logging
from core.config.settings import get_settings
from core.voice.exceptions import VoiceError
from core.voice.microphone import list_input_devices, resolve_device
from core.voice.service import VoiceService
from core.voice.transcriber import Transcriber

HEADER = "JARVIS Voice Recognition"


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser. Exposed so tests can exercise it directly."""
    parser = argparse.ArgumentParser(
        prog="python -m core.voice",
        description="Record one audio sample from the microphone and transcribe it to text.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available microphone devices and exit.",
    )
    parser.add_argument("--model", default=None, help="Whisper model size (default: base).")
    parser.add_argument("--language", default=None, help="Spoken language code (default: en).")
    parser.add_argument("--device", default=None, help="Inference device: cpu, cuda, or auto.")
    parser.add_argument(
        "--compute-type", default=None, help="CTranslate2 compute type (default: int8)."
    )
    parser.add_argument("--seconds", type=float, default=None, help="Recording length in seconds.")
    parser.add_argument(
        "--microphone", type=int, default=None, help="Microphone device index to record from."
    )
    return parser


def _print_devices() -> int:
    """Print every input device PortAudio reports."""
    try:
        devices = list_input_devices()
    except VoiceError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not devices:
        print("No input devices were detected.")
        print("Check that a microphone is connected and enabled in Windows Sound settings.")
        return 1

    print("Available input devices:")
    for device in devices:
        marker = " (system default)" if device.is_default else ""
        print(
            f"  [{device.index}] {device.name} - {device.hostapi}, "
            f"{device.default_sample_rate:.0f} Hz{marker}"
        )
    print()
    print("Select one with: uv run python -m core.voice --microphone <index>")
    print("or set VOICE_MICROPHONE_DEVICE=<index> in your .env file.")
    return 0


def _device_label(index: int | None) -> str:
    try:
        device = resolve_device(index)
    except VoiceError:
        return "Default microphone"
    return f"{device.name} (index {device.index})"


def _build_service(args: argparse.Namespace) -> VoiceService:
    settings = get_settings()

    model_size = args.model or settings.voice_model_size
    language = args.language or settings.voice_language
    device = args.device or settings.voice_device
    compute_type = args.compute_type or settings.voice_compute_type
    seconds = args.seconds if args.seconds is not None else settings.voice_record_seconds
    microphone = args.microphone if args.microphone is not None else settings.voice_microphone_device

    transcriber = Transcriber(
        model_size=model_size,
        language=language,
        device=device,
        compute_type=compute_type,
        model_path=settings.voice_model_path,
    )
    return VoiceService(
        transcriber=transcriber,
        record_seconds=seconds,
        device_index=microphone,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code."""
    setup_logging()
    args = build_parser().parse_args(argv)

    if args.list_devices:
        return _print_devices()

    try:
        service = _build_service(args)
    except VoiceError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(HEADER)
    print("-" * len(HEADER))
    print(f"Model: {service.transcriber.model_size}")
    print(f"Device: {_device_label(service.device_index)}")
    print()

    try:
        print(f"Recording for {service.record_seconds:g} seconds...")
        audio = service.record()
        print("Processing speech...")
        text = service.transcribe(audio)
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 0
    except VoiceError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if text:
        print()
        print(f"Recognized: {text}")
    else:
        print()
        print("No speech recognized. Try again and speak clearly during the recording.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
