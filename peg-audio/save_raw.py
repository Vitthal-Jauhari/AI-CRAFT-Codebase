import asyncio
import os
import websockets

PORT = 42069
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_RAW = os.path.join(SCRIPT_DIR, "audio.raw")


async def ws_handler(websocket):
    print("ESP32 connected")

    with open(AUDIO_RAW, "wb") as f:
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    f.write(message)
                    print(f"Received {len(message)} bytes")
        except websockets.exceptions.ConnectionClosed:
            print("ESP32 disconnected")

    print(f"Audio saved to {AUDIO_RAW}")


async def main():
    async with websockets.serve(ws_handler, "0.0.0.0", PORT):
        print(f"WebSocket server listening on :{PORT}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
