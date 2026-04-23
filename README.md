# AI CRAFT Codebase
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/Vitthal-Jauhari/AI-CRAFT-Codebase)

## Overview

AI CRAFT is a comprehensive project for creating an AI-driven smart environment. It integrates multiple AI modalities—voice, gesture, and body pose—to provide intuitive, hands-free control over home appliances. The system is built on a client-server architecture, using ESP32 microcontrollers for sensor data acquisition and a powerful Python backend for AI processing.

This repository contains the complete codebase, including ESP32 firmware for audio and video streaming, and the central "Aura AI Server" that uses state-of-the-art models like Whisper for speech recognition, spaCy for natural language understanding, and MediaPipe for vision-based control.

## Features

*   **Voice Control**: Leverages OpenAI's Whisper model for real-time, high-accuracy speech-to-text transcription. A custom spaCy-based parser interprets commands like "turn on the light" to extract actionable device and state information.
*   **Gesture Recognition**: Utilizes MediaPipe Hand Landmarker to detect specific hand gestures. For example, an open palm can turn a device on, while a closed fist can turn it off, providing immediate and silent control.
*   **Ambient Pose Detection**: Employs MediaPipe Pose Landmarker to understand the user's posture (e.g., sitting, standing, lying down). This allows the system to make context-aware decisions, such as adjusting appliances when a person sits at a desk or lies down to sleep.
*   **Unified Vision Engine**: The `VisionBridge` module intelligently fuses pose and gesture recognition from a single video stream (webcam or ESP32-CAM), emitting unified action commands and preventing conflicting signals.
*   **ESP32 Integration**: Includes PlatformIO projects for ESP32 devices equipped with an INMP441 I2S microphone for high-quality audio capture and an ESP32-CAM for video streaming.
*   **Real-time Communication**: Uses WebSockets for low-latency, bidirectional communication between the ESP32 clients and the central server.

## System Architecture

The system operates on a client-server model:

1.  **ESP32 Clients**:
    *   **Audio Client (`Mic Testing`)**: An ESP32 with an INMP441 I2S microphone captures audio, streams it as raw 32-bit PCM data to the server via a WebSocket, and listens for command messages to control GPIO-connected appliances.
    *   **Video Client (`Testing`)**: An ESP32-CAM board streams an MJPEG video feed over the local network.

2.  **Aura AI Server (Python Backend)**:
    *   **WebSocket Server**: Manages connections from ESP32 clients.
    *   **Audio Processor (`server.py`)**: Receives the raw audio stream, resamples it from 44.1kHz to the 16kHz expected by Whisper, and performs transcription.
    *   **Vision Processor (`vision_server.py` + `vision_bridge.py`)**: Connects to the ESP32-CAM's video stream, processes each frame to detect poses and gestures, and generates control events.
    *   **NLU Parser (`spacy_parser.py`)**: Takes transcribed text from the audio processor, identifies the target device and desired action (e.g., `(light, turn on)`), and formulates a response.
    *   **Command Dispatcher**: Sends the final commands (e.g., `light:true`) back to the Audio Client ESP32 via WebSocket to trigger the physical appliance.

## Repository Structure

```
.
├── aura-ai-server/      # Main Python AI server for voice and vision processing.
├── platformio/          # ESP32 firmware projects.
│   ├── Mic Testing/     # Firmware for ESP32 with I2S microphone and appliance control.
│   └── Testing/         # Firmware for ESP32-CAM video streaming.
├── gesture-control/     # Standalone prototype for MediaPipe hand gesture control.
├── voice-control/       # Prototypes for voice command intent recognition.
├── speech-to-text-stt/  # Simple Whisper transcription script.
└── servers/             # Experimental WebSocket server implementations.
```

## Key Components

### Aura AI Server (`aura-ai-server/`)

This is the heart of the system.
*   **`server.py`**: Connects to the ESP32 audio client via WebSockets. It buffers incoming 32-bit PCM audio, resamples it to 16kHz, and uses a background thread to transcribe it with the Whisper `base` model. It includes logic to detect a wake word before processing a command.
*   **`vision_bridge.py`**: A powerful, unified vision processing class that takes a video source and simultaneously runs MediaPipe's Pose and Hand landmarkers. It intelligently combines the results to infer user intent from both deliberate gestures and ambient posture.
*   **`vision_server.py`**: A simple script to run the `VisionBridge` with an ESP32-CAM stream URL.
*   **`spacy_parser.py`**: A reusable class that uses a spaCy NLP pipeline to parse natural language commands. It handles complex sentences, phrasal verbs, coordinated objects ("turn on the light and fan"), and negation.
*   **`pose_landmarker.task` / `hand_landmarker.task`**: MediaPipe model files required by the vision components.

### ESP32 Firmware (`platformio/`)

*   **`Mic Testing/`**:
    *   **Hardware**: ESP32 with an INMP441 I2S Microphone.
    *   **Pins**: `SCK: 26`, `WS: 25`, `SD: 33`.
    *   **Functionality**: Connects to WiFi and a WebSocket server to continuously stream audio data. It also listens for incoming control messages (e.g., `"light:true"`) to toggle corresponding GPIO pins.
*   **`Testing/`**:
    *   **Hardware**: ESP32-CAM (AI-Thinker model).
    *   **Functionality**: Configures the camera module and starts a web server that provides an MJPEG video stream, serving as the input for the vision server.

## Setup and Installation

#### 1. Prerequisites
- Python 3.8+
- [PlatformIO Core](https://docs.platformio.org/en/latest/core/installation.html) for ESP32 firmware development.
- ESP32 development boards (e.g., ESP32-S3 DevKit-C and an ESP32-CAM).
- An INMP441 I2S microphone.
- Appliances/relays to connect to the ESP32 GPIO pins.

#### 2. Software (Server)
1.  Clone the repository:
    ```bash
    git clone https://github.com/Vitthal-Jauhari/AI-CRAFT-Codebase.git
    cd AI-CRAFT-Codebase
    ```
2.  Install the required Python packages (a virtual environment is recommended):
    ```bash
    pip install numpy scipy websockets whisper-openai spacy mediapipe opencv-python sentence-transformers
    ```
3.  Download the spaCy model for command parsing:
    ```bash
    python -m spacy download en_core_web_sm
    ```
4.  Download the MediaPipe models (`pose_landmarker.task`, `hand_landmarker.task`) from Google's official model repository and place them in the `aura-ai-server/` directory.

#### 3. Hardware (ESP32)

1.  **Audio Client (`Mic Testing`)**:
    *   Open the `platformio/Mic Testing/` project in an IDE like VSCode with the PlatformIO extension.
    *   Modify `src/main.cpp` with your WiFi credentials (`ssid`, `password`) and the IP address of your server (`websocket_server_host`).
    *   Connect the INMP441 microphone to the ESP32 using pins `SCK: 26`, `WS: 25`, `SD: 33`.
    *   Connect your relays/appliances to the control pins defined in the code (e.g., `LIGHT_PIN: 2`).
    *   Build and upload the project to your ESP32.

2.  **Video Client (`Testing`)**:
    *   Open and configure the `platformio/Testing/` project with your WiFi credentials.
    *   Build and upload the project to your ESP32-CAM.
    *   Use the Serial Monitor to find the board's IP address. The video stream will be available at `http://<ESP32_CAM_IP>/stream`.

#### 4. Running the System
1.  Navigate to the server directory and start the voice server:
    ```bash
    cd aura-ai-server
    python server.py
    ```
    The server will listen on port `42069`. Once the audio client ESP32 connects, it will begin processing audio.

2.  In a separate terminal, start the vision server:
    *   Edit `vision_server.py` and set the `source` variable to your ESP32-CAM stream URL.
    *   Run the script:
    ```bash
    python vision_server.py
    ```
    A window will appear showing the video feed with detected landmarks, and the console will print actions triggered by poses or gestures.