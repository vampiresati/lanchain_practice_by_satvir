import os
import tempfile
import time
from typing import Optional

import pyttsx3
import requests
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3:4b"

SAMPLE_RATE = 16_000
CHANNELS = 1
MAX_DURATION = 30

# Whisper model choices:
# tiny.en   -> fastest, English only
# base.en   -> better accuracy, English only
# small     -> multilingual, slower
WHISPER_MODEL_NAME = "base.en"


# ---------------------------------------------------------
# Load local models once
# ---------------------------------------------------------

print("Loading local Whisper model...")

whisper_model = WhisperModel(
    WHISPER_MODEL_NAME,
    device="cpu",
    compute_type="int8",
)

print("Initializing local text-to-speech...")

tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 175)
tts_engine.setProperty("volume", 1.0)


# Conversation memory for Ollama
conversation = [
    {
        "role": "system",
        "content": (
            "You are a helpful local voice assistant. "
            "Keep responses short, clear and conversational. "
            "Do not use markdown unless the user asks for it."
        ),
    }
]


def record_audio() -> str:
    """
    Record microphone audio.

    Recording stops when the user presses Enter or when MAX_DURATION
    seconds have elapsed.
    """

    input("\nPress Enter to start recording...")

    print(
        f"Recording for up to {MAX_DURATION} seconds. "
        "Press Enter to stop."
    )

    max_frames = MAX_DURATION * SAMPLE_RATE

    start_time = time.monotonic()

    audio_data = sd.rec(
        max_frames,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
    )

    input()

    elapsed = time.monotonic() - start_time
    sd.stop()

    recorded_frames = min(
        max(int(elapsed * SAMPLE_RATE), 1),
        max_frames,
    )

    audio_data = audio_data[:recorded_frames]

    print(f"Recording stopped after {elapsed:.1f} seconds.")

    temporary_file = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    )
    temporary_file.close()

    sf.write(
        temporary_file.name,
        audio_data,
        SAMPLE_RATE,
    )

    return temporary_file.name


def transcribe(audio_path: str) -> str:
    """
    Transcribe audio locally with faster-whisper.
    """

    print("Transcribing locally...")

    segments, information = whisper_model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
    )

    text_parts = []

    for segment in segments:
        cleaned_text = segment.text.strip()

        if cleaned_text:
            text_parts.append(cleaned_text)

    transcript = " ".join(text_parts).strip()

    if information.language:
        print(
            f"Detected language: {information.language} "
            f"({information.language_probability:.2f})"
        )

    return transcript


def ask_ollama(user_text: str) -> str:
    """
    Send the transcribed message to the local Ollama API.
    """

    conversation.append(
        {
            "role": "user",
            "content": user_text,
        }
    )

    payload = {
        "model": OLLAMA_MODEL,
        "messages": conversation,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 200,
        },
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=180,
        )

        response.raise_for_status()

    except requests.ConnectionError as error:
        conversation.pop()

        raise RuntimeError(
            "Cannot connect to Ollama. Start it with:\n"
            "ollama serve"
        ) from error

    except requests.HTTPError as error:
        conversation.pop()

        try:
            error_details = response.json()
        except ValueError:
            error_details = response.text

        raise RuntimeError(
            f"Ollama returned an error: {error_details}"
        ) from error

    result = response.json()

    reply = (
        result.get("message", {})
        .get("content", "")
        .strip()
    )

    if not reply:
        conversation.pop()

        raise RuntimeError(
            "Ollama returned an empty response."
        )

    conversation.append(
        {
            "role": "assistant",
            "content": reply,
        }
    )

    return reply


def speak(text: str) -> None:
    """
    Speak the response locally using pyttsx3/eSpeak.
    """

    if not text:
        return

    tts_engine.say(text)
    tts_engine.runAndWait()


def get_typed_message() -> Optional[str]:
    """
    Read a message from the keyboard.
    """

    message = input("You: ").strip()

    return message or None


def main() -> None:
    print("\nLocal Voice Assistant")
    print("---------------------")
    print(f"Ollama model: {OLLAMA_MODEL}")
    print(f"Whisper model: {WHISPER_MODEL_NAME}")
    print("Everything runs locally.")
    print()
    print("Commands:")
    print("  Press Enter  -> record your voice")
    print("  Type text    -> send a typed message")
    print("  exit         -> close the assistant")

    while True:
        audio_path: Optional[str] = None

        try:
            typed_input = input(
                "\nPress Enter to speak, or type a message: "
            ).strip()

            if typed_input.lower() in {
                "exit",
                "quit",
                "bye",
            }:
                print("Assistant stopped.")
                break

            if typed_input:
                transcript = typed_input
            else:
                audio_path = record_audio()
                transcript = transcribe(audio_path)

            if not transcript:
                print("I could not detect any speech.")
                continue

            print(f"\nYou: {transcript}")

            reply = ask_ollama(transcript)

            print(f"Assistant: {reply}")

            speak(reply)

        except KeyboardInterrupt:
            print("\nAssistant stopped.")
            break

        except Exception as error:
            print(f"\nError: {error}")

        finally:
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)


if __name__ == "__main__":
    main()