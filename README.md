# AI CRAFT Codebase

AI CRAFT is an ESP32 + Python automation workspace for voice, gesture, and pose-driven control experiments.

## What is in this repo

- `aura-ai-server/`: Main Python logic used in current integrations.
- `platformio/Mic Testing/`: ESP32-S3 microphone client + GPIO control via WebSocket messages.
- `platformio/Testing/`: ESP32-CAM project files (currently includes camera HTTP handler and PSRAM test setup).
- `gesture-control/`: Standalone MediaPipe gesture/pose prototype.
- `voice-control/`: NLP/intent matching prototypes.
- `speech-to-text-stt/`: Minimal Whisper transcription test.
- `servers/`: Experimental WebSocket server code.

## Current behavior (workspace-accurate)

### Audio path

- `aura-ai-server/server.py` connects as a WebSocket client using `WS_URL`.
- Incoming ESP32 audio is handled as 32-bit PCM at 44.1 kHz and resampled to 16 kHz for Whisper.
- Transcription is parsed by `aura-ai-server/spacy_parser.py`.
- `platformio/Mic Testing/src/main.cpp` streams I2S audio and applies GPIO actions from messages like `light:true`.

### Vision path

- `aura-ai-server/vision_bridge.py` combines pose + hand landmark detection.
- `aura-ai-server/vision_server.py` runs the bridge against webcam or ESP32-CAM stream URLs.
- The vision code requires both `pose_landmarker.task` and `hand_landmarker.task` in `aura-ai-server/`.

### ESP32 projects

- `platformio/Mic Testing/` is actively wired for Wi-Fi, WebSocket, I2S mic capture, and appliance pins.
- `platformio/Testing/` currently has:
  - `src/app_httpd.cpp`: MJPEG stream handler.
  - `src/main.cpp`: PSRAM test stub.

## Requirements

- Python 3.8+
- PlatformIO (for ESP32 builds)
- ESP32-S3 + INMP441 (audio path)
- ESP32-CAM (vision stream path)

## Python setup

From repository root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install numpy scipy websockets openai-whisper spacy mediapipe opencv-python sentence-transformers scikit-learn
python -m spacy download en_core_web_sm
```

MediaPipe model files needed by vision modules:

- `aura-ai-server/pose_landmarker.task`
- `aura-ai-server/hand_landmarker.task`

## Run (Python)

Start audio processing:

```bash
cd aura-ai-server
python server.py
```

Start vision processing (separate terminal):

```bash
cd aura-ai-server
python vision_server.py
```

## ESP32 notes

### Mic Testing

Edit `platformio/Mic Testing/src/main.cpp` before upload:

- `ssid`
- `password`
- `websocket_server_host`
- `websocket_server_port`

I2S pin mapping in current code:

- `SCK`: 26
- `WS`: 25
- `SD`: 33

### Testing (ESP32-CAM)

Project files are present, but the current `src/main.cpp` is a PSRAM check program. If you want full camera boot + stream startup in this project, implement/restore camera init + Wi-Fi startup in `src/main.cpp` and call `startCameraServer()` from `src/app_httpd.cpp`.

## Repository tree

```text
.
|-- aura-ai-server/
|-- gesture-control/
|-- peg-audio/
|-- platformio/
|   |-- Mic Testing/
|   `-- Testing/
|-- servers/
|-- speech-to-text-stt/
|-- voice-control/
`-- websocket-connection-esp/
```