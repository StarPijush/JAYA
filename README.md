# JARVIS
Personal AI Assistant for Windows

## Current Status
**Phase 1 — Voice Recognition**

Local microphone-to-text is working. Speech is captured from a selected
microphone and transcribed to text by a locally-run Whisper model.

## Technology
- Python
- FastAPI
- faster-whisper (speech-to-text, CPU by default)
- sounddevice (microphone capture via PortAudio)
- NumPy + SciPy (audio processing and band-limited resampling)
- SQLite (future)
- TypeScript/React (future)
- Windows APIs (future)
- LLM APIs (future)

## Architecture
```
JARVIS/
├── apps/
│   └── desktop/
├── core/
│   ├── agent/
│   ├── voice/
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── microphone.py
│   │   ├── transcriber.py
│   │   ├── service.py
│   │   └── exceptions.py
│   ├── tools/
│   ├── memory/
│   └── config/
├── services/
│   └── api/
├── database/
├── scripts/
├── tests/
├── .gitignore
├── README.md
└── pyproject.toml
```

## Running Locally

### 1. Create virtual environment
```powershell
uv venv
```

### 2. Activate environment (PowerShell)
```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```powershell
uv pip install -e ".[dev]"
```

### 4. Start the API
```powershell
python -m services.api
# or
jarvis-api
```

The API will be available at `http://127.0.0.1:8000`

### 5. Run tests
```powershell
uv run pytest -v
```

### 6. Verify health endpoint
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
```

### 7. Run the mobile web UI (UI-only, no backend connection)
```powershell
cd apps\web
npm install
npm run dev
```

Open `http://localhost:3000` in a mobile viewport (320–430px). See
`apps/web/README.md` for details and verification commands.

## Phase 1 — Voice Recognition

### Requirements

- Windows 10 or 11
- Python 3.11 or newer (developed and tested on 3.14.3)
- A working microphone
- Internet access on the first run only, to download the Whisper model

No GPU is required. No cloud service is used; all transcription happens locally.

### Install dependencies

```powershell
uv sync --extra dev
```

This installs `faster-whisper`, `sounddevice`, `numpy`, and `scipy`.

### List your microphones

```powershell
uv run python -m core.voice --list-devices
```

Output on this machine:

```text
Available input devices:
  [0] Microsoft Sound Mapper - Input - MME, 44100 Hz
  [1] Microphone (Realtek(R) Audio) - MME, 44100 Hz (system default)
  ...
```

This reads live data from PortAudio, so the names and indexes reflect whatever
your machine actually has. Only devices that can record are listed; output-only
devices such as speakers are excluded.

### Run one voice recognition session

```powershell
uv run python -m core.voice
```

```text
JARVIS Voice Recognition
------------------------
Model: base
Device: Microphone (Realtek(R) Audio) (index 1)

Recording for 5 seconds...
Processing speech...

Recognized: open the browser
```

Press `Ctrl+C` at any time to exit cleanly.

The CLI runs a single recording and exits. There is no continuous listening loop.

### Choosing a microphone

The default device is the Windows system default input. If that is the wrong
one, pass an index:

```powershell
uv run python -m core.voice --microphone 9
```

Or set it once in `.env`:

```
VOICE_MICROPHONE_DEVICE=9
```

Use the indexes from `--list-devices`. Note that the same physical microphone
can appear several times under different host APIs (MME, DirectSound, WASAPI,
WDM-KS). WASAPI generally gives the best quality, but any of them will work.

### Options

| Option | Env variable | Default | Purpose |
| --- | --- | --- | --- |
| `--model` | `VOICE_MODEL_SIZE` | `base` | Whisper model size |
| `--language` | `VOICE_LANGUAGE` | `en` | Spoken language code |
| `--device` | `VOICE_DEVICE` | `cpu` | `cpu`, `cuda`, or `auto` |
| `--compute-type` | `VOICE_COMPUTE_TYPE` | `int8` | CTranslate2 quantization |
| `--seconds` | `VOICE_RECORD_SECONDS` | `5` | Recording length |
| `--microphone` | `VOICE_MICROPHONE_DEVICE` | system default | Input device index |
| `--list-devices` | — | — | List microphones and exit |

Command-line flags override `.env` values for that run.

### Changing model size and language

| Model | Approx. download | Notes |
| --- | --- | --- |
| `tiny` | 75 MB | Fastest, lowest accuracy |
| `base` | 145 MB | Default, good balance |
| `small` | 480 MB | Noticeably more accurate |
| `medium` | 1.5 GB | Slow on CPU |
| `large-v3` | 3 GB | Best accuracy, impractical on CPU |

English-only variants (`base.en`, `small.en`) are available and are slightly
faster since no language detection is needed.

```powershell
uv run python -m core.voice --model small
uv run python -m core.voice --language es
```

Larger models give better accuracy on accents and background noise, at a
proportional cost in download size and CPU time.

### Microphone permissions on Windows

Windows blocks microphone access by default for many applications. If you get
an access-denied error:

1. Open **Settings → Privacy & security → Microphone**
2. Enable **Microphone access**
3. Enable **Let desktop apps access your microphone**
4. Confirm the microphone is not muted in the mixer, and that its input level is
   above zero

The voice CLI is a desktop application, so it needs the desktop-apps toggle.

### First-run model download

The Whisper model is downloaded lazily, the first time transcription runs, and
cached outside the repository at:

```text
%LOCALAPPDATA%\JARVIS\models\whisper
```

Nothing is bundled with the project and no model files enter the Git tree.
The `base` model is about 145 MB and takes roughly 100 seconds on a first
download. Later runs are fully offline.

If you would rather keep the cache elsewhere, set:

```
VOICE_MODEL_PATH=C:\some\other\directory
```

On Windows you may see a warning from `huggingface_hub` about symlinks. It is
harmless; caching works without Developer Mode, using slightly more disk space.

### CPU performance expectations

Measured on this machine (AMD Ryzen 3 3250U, 4 threads), `base` model, `int8`:

| Operation | Time |
| --- | --- |
| Model load, first run (includes download) | ~98 s |
| Model load, cached | ~2 s |
| Transcription of 3 s of audio | ~15 s |

So expect transcription to run several times longer than the recording itself.
For more responsive use:

- Use the `tiny` or `base` model
- Record shorter clips
- Keep `VOICE_COMPUTE_TYPE=int8` on CPU

Setting `VOICE_DEVICE=cuda` requires an NVIDIA GPU with CUDA 12 and compatible
drivers. Without one, leave it on `cpu`. Do not set `cpu_threads` above the
default; on this machine it made inference substantially slower.

### Troubleshooting

**`Windows denied microphone access`** — Enable microphone access in
**Settings → Privacy & security → Microphone**, including the desktop-apps
toggle.

**`Microphone ... is unavailable or in use by another application`** — Close
other apps holding the microphone. Recording applications, video call clients,
and games are common culprits. Re-run `--list-devices` to pick a different
device.

**`Error opening InputStream: Unanticipated host error [PaErrorCode -9999]`** —
PortAudio could not open the device. This usually means the microphone is
disabled in Device Manager, is used exclusively by another application, or the
default device points at something unavailable. Try another index from
`--list-devices`, and prefer a WASAPI entry if one exists.

**`No audio devices found`** — No input device is visible to the process.
Confirm a microphone is connected and enabled in Windows Sound settings.

**`No audible signal was detected`** — The recording was silent or effectively
zero. Check the input volume in the Windows mixer, and confirm the right device
is selected. Recordings quieter than about -80 dBFS are rejected as unusable.

**`No speech recognized`** — Audio was captured but no words were detected. Speak
during the whole window and try again, or increase the input level.

**`Could not load Whisper model ... requires internet access`** — The model is
not in the cache and could not be downloaded. Check your network connection.

**`faster-whisper is not installed`** — Run `uv sync --extra dev`.

### Using the voice subsystem from Python

```python
from core.voice import VoiceService

service = VoiceService.from_settings()
text = service.listen_and_transcribe()
print(text)
```

Recording and transcription are separately usable, which also makes them easy
to test:

```python
audio = service.record(seconds=3)
text = service.transcribe(audio)
```

The voice subsystem is independent of the FastAPI service. Running voice
recognition never starts the API.

## Development Philosophy
JARVIS will be built incrementally:

```
Foundation
    ↓
Voice Recognition
    ↓
Command Engine
    ↓
Text-to-Speech
    ↓
Wake Word
    ↓
LLM Agent
    ↓
Windows Tools
    ↓
Browser Automation
    ↓
Memory
    ↓
Desktop UI
    ↓
Advanced Native Capabilities
```

Each phase adds real functionality. No placeholder implementations.