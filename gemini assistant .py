import asyncio
import io
import os
import mss
import pyaudio
from PIL import Image
from google import genai
from google.genai import types

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_RATE = 16000
RECV_RATE = 24000
CHUNK = 1024

MODEL = "gemini-3.8-live"
client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY"),
    http_options={"api_version": "v1beta"},
)

config = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    output_audio_transcription=types.AudioTranscriptionConfig(),
    system_instruction=(
        "Tum ek live AI assistant ho jo user ki screen real-time dekh raha hai. "
        "User ki awaaz aur text ke hisaab se screen dekh kar Hinglish me short aur helpful jawab do."
    ),
)


async def main():
    pya = pyaudio.PyAudio()
    mic_stream = pya.open(
        format=FORMAT, channels=CHANNELS, rate=SEND_RATE, input=True, frames_per_buffer=CHUNK
    )
    spk_stream = pya.open(
        format=FORMAT, channels=CHANNELS, rate=RECV_RATE, output=True
    )

    out_queue = asyncio.Queue(maxsize=10)
    audio_in_queue = asyncio.Queue()

    async with client.aio.live.connect(model=MODEL, config=config) as session:
        print("🟢 Live Assistant Shuru Ho Gaya! Seedhe bolein ya neeche type karein ('q' to quit).")

        async def send_text():
            while True:
                text = await asyncio.to_thread(input, "\nYou (Text) > ")
                if text.strip().lower() == "q":
                    raise asyncio.CancelledError()
                if text.strip():
                    await session.send_client_content(
                        turns={"role": "user", "parts": [{"text": text}]},
                        turn_complete=True,
                    )

        async def listen_mic():
            while True:
                data = await asyncio.to_thread(
                    mic_stream.read, CHUNK, exception_on_overflow=False
                )
                await out_queue.put({"type": "audio", "data": data})

        async def capture_screen():
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                while True:
                    sct_img = sct.grab(monitor)
                    img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                    img.thumbnail((1024, 576))
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=75)
                    await out_queue.put({"type": "video", "data": buf.getvalue()})
                    await asyncio.sleep(1.0)  # Har 1 second me screen frame bheje

        async def send_realtime():
            while True:
                item = await out_queue.get()
                if item["type"] == "audio":
                    await session.send_realtime_input(
                        audio=types.Blob(data=item["data"], mime_type="audio/pcm;rate=16000")
                    )
                elif item["type"] == "video":
                    await session.send_realtime_input(
                        video=types.Blob(data=item["data"], mime_type="image/jpeg")
                    )

        async def receive_ai():
            while True:
                async for response in session.receive():
                    sc = response.server_content
                    if not sc:
                        continue
                    if sc.interrupted:
                        while not audio_in_queue.empty():
                            audio_in_queue.get_nowait()
                    if sc.output_transcription and sc.output_transcription.text:
                        print(sc.output_transcription.text, end="", flush=True)
                    if sc.model_turn:
                        for part in sc.model_turn.parts:
                            if part.inline_data and part.inline_data.data:
                                audio_in_queue.put_nowait(part.inline_data.data)

        async def play_speaker():
            while True:
                audio_data = await audio_in_queue.get()
                await asyncio.to_thread(spk_stream.write, audio_data)

        try:
            await asyncio.gather(
                send_text(),
                listen_mic(),
                capture_screen(),
                send_realtime(),
                receive_ai(),
                play_speaker(),
            )
        except asyncio.CancelledError:
            print("\nBand ho raha hai...")
        finally:
            mic_stream.stop_stream()
            mic_stream.close()
            spk_stream.stop_stream()
            spk_stream.close()
            pya.terminate()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBand ho gaya.")