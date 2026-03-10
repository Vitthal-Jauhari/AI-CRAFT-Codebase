import asyncio
import subprocess
import os
import whisper
import websockets

PORT = 42069
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_RAW = os.path.join(SCRIPT_DIR, "audio.raw")
AUDIO_WAV = os.path.join(SCRIPT_DIR, "output.wav")
TRANSCRIBE_INTERVAL = 8  # seconds between transcription attempts

model = whisper.load_model("base")


async def transcribe_loop(audio_buffer, last_offset):
    """Periodically convert accumulated audio and transcribe for real-time output."""
    while True:
        await asyncio.sleep(TRANSCRIBE_INTERVAL)

        current_len = len(audio_buffer)
        if current_len == 0 or current_len == last_offset[0]:
            continue

        # Write current buffer to raw file
        with open(AUDIO_RAW, "wb") as f:
            f.write(audio_buffer)

        # Convert raw → wav with ffmpeg
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "s32le",
                    "-ar", "44100",
                    "-ac", "1",
                    "-i", AUDIO_RAW,
                    AUDIO_WAV,
                ],
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print("ffmpeg error:", e.stderr.decode())
            continue

        last_offset[0] = current_len

        # Transcribe with Whisper
        result = model.transcribe(AUDIO_WAV, language="en")
        text = result["text"].strip()
        if text:
            print(f"\r[Transcript] {text}", flush=True)


async def ws_handler(websocket):
    print("ESP32 connected")

    audio_buffer = bytearray()
    last_offset = [0]  # mutable so the transcribe task can update it

    transcribe_task = asyncio.create_task(transcribe_loop(audio_buffer, last_offset))

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                audio_buffer.extend(message)
            else:
                print(f"Received text: {message}")
    except websockets.exceptions.ConnectionClosed:
        print("ESP32 disconnected")
    finally:
        transcribe_task.cancel()

        # Final transcription of all collected audio
        if len(audio_buffer) > 0:
            print("\n--- Final transcription ---")

            with open(AUDIO_RAW, "wb") as f:
                f.write(audio_buffer)

            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-f", "s32le",
                        "-ar", "44100",
                        "-ac", "1",
                        "-i", AUDIO_RAW,
                        AUDIO_WAV,
                    ],
                    capture_output=True,
                    check=True,
                )
                result = model.transcribe(AUDIO_WAV, language="en")
                print("Transcript:")
                print(result["text"])
            except subprocess.CalledProcessError as e:
                print("ffmpeg error:", e.stderr.decode())


async def main():
    async with websockets.serve(ws_handler, "0.0.0.0", PORT):
        print(f"WebSocket server listening on :{PORT}")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
