import asyncio
from concurrent.futures import ThreadPoolExecutor
import websockets

import numpy as np

from scipy.signal import resample

import whisper

from spacy_parser import SpacyParser



# Config
WS_URL = "ws://localhost:42069/ws?hash=001ca698898c60f835261b55cc2ac77b3a5560ba7da32b4292dc508d53e2a82946d6175d18a5760e982ad3f85285b29f3f8320c9be3d0aa5a4cff3ce45b74511"
PING_INTERVAL_SECONDS = 20
PING_TIMEOUT_SECONDS = 20

ESP_SAMPLE_RATE = 44100   # what the ESP32 I2S sends
WHISPER_SAMPLE_RATE = 16000  # what Whisper expects

buffer_lock = asyncio.Lock()

SAMPLE_WIDTH = 4          # 32-bit PCM from ESP32
TRANSCRIBE_INTERVAL = 5   # seconds

executor = ThreadPoolExecutor(max_workers=2)
audio_buffer = bytearray()

async def transcription_loop():
    global audio_buffer
    loop = asyncio.get_event_loop()

    while True:
        await asyncio.sleep(TRANSCRIBE_INTERVAL)

        async with buffer_lock:
            if len(audio_buffer) < ESP_SAMPLE_RATE * SAMPLE_WIDTH:
                continue
            chunk = bytes(audio_buffer)

        try:
            text = await loop.run_in_executor(executor, transcribe_audio, chunk)
            await process_transcription_result(text)
        except Exception as e:
            print(f"[Whisper Error] {e}")











command_buffer = None

recording_command = False





async def process_transcription_result(result: str) -> None:
    """Process the transcribed text result from Whisper."""
    text = result.strip()
    if text:

        global recording_command, command_buffer

        print(f"[Whisper] {text}", flush=True)


        if not recording_command and "Alexa" in text:
            print("[Action] Detected 'Alexa' in transcription. Starting command recording.")
            recording_command = True


            # if first word is not "Alexa" clip text before "Alexa"
            if text.split()[0] != "Alexa":
                print("[Action] Detected 'Alexa' in transcription, but not at start. Clipping text.")
                text = "Alexa "+text[text.find("Alexa"):]
        elif not recording_command:
            print("[Action] Not recording command and 'Alexa' not detected. Ignoring transcription.")
            
            audio_buffer.clear()  # clear buffer to avoid processing stale audio
            

            return


        if recording_command:

            if command_buffer is None:
                command_buffer = text
                return

            if command_buffer is text:

                print(command_buffer)
                print(text)


                return 


            # print(f"[Command Buffer] Updated with command: {command_buffer}")

            if parser is not None:
                action = parser.parse_as_tuples(command_buffer)



                print(command_buffer)
                print(action)

                if len(action) > 0:
                    print(f"[Action] Parsed action: {action}")

                    command_buffer = None
                    text = None

                    audio_buffer.clear()

                    recording_command = False

















def transcribe_audio(pcm_data: bytes) -> str:
    """Convert 32-bit PCM from ESP32, resample 44100→16000, transcribe."""
    # ESP32 sends int32 I2S data (INMP441 24-bit in 32-bit frame)
    audio_np = np.frombuffer(pcm_data, dtype=np.int32).astype(np.float32) / 2147483648.0

    # Resample from 44100 Hz to 16000 Hz for Whisper
    num_samples_16k = int(len(audio_np) * WHISPER_SAMPLE_RATE / ESP_SAMPLE_RATE)
    audio_16k = resample(audio_np, num_samples_16k).astype(np.float32)

    result = model.transcribe(audio_16k, language="en", verbose=False)
    return result["text"]

async def process_audio_chunk(chunk: bytes) -> None:
    """Append incoming audio bytes to the shared buffer."""
    async with buffer_lock:
        audio_buffer.extend(chunk)
    # print(f"[Audio] Buffered {len(chunk)} bytes (total={len(audio_buffer)})")


async def listen_for_audio() -> None:
    """Connect once to the websocket server and listen for incoming audio bytes."""
    print(f"[Client] Connecting to {WS_URL}")

    try:
        async with websockets.connect(
            WS_URL,
            ping_interval=PING_INTERVAL_SECONDS,
            ping_timeout=PING_TIMEOUT_SECONDS,
        ) as ws:
            print(f"[Client] Connected (id={id(ws)})")

            async for message in ws:
                if not isinstance(message, bytes):
                    continue

                await process_audio_chunk(message)
    except (websockets.exceptions.ConnectionClosed, OSError) as error:
        print(f"[Client] Connection closed: {error}")



model = None


parser = None

async def main():
    global model, parser
    print("Loading Whisper model…")
    model = whisper.load_model("base")
   

    parser = SpacyParser()




    asyncio.create_task(transcription_loop())
    await listen_for_audio()



if __name__ == "__main__":

    

    asyncio.run(main())
