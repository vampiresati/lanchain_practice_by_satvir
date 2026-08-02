import os
import tempfile
import time
from typing import Optional

import soundfile as sf
from faster_whisper import WhisperModel
from langchain.agents import create_agent

try:
    import sounddevice as sd
    SOUNDDEVICE_IMPORT_ERROR = None
except OSError as error:
    sd = None
    SOUNDDEVICE_IMPORT_ERROR = error

try:
    import pyttsx3
    PYTTSX3_IMPORT_ERROR = None
except Exception as error:
    pyttsx3 = None
    PYTTSX3_IMPORT_ERROR = error


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

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

tts_engine = None
if pyttsx3 is not None:
    tts_engine = pyttsx3.init()
    tts_engine.setProperty("rate", 175)
    tts_engine.setProperty("volume", 1.0)

print("Creating LangChain Ollama agent...")

agent = create_agent(
    model=f"ollama:{OLLAMA_MODEL}",
    tools=[],
)


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

    if sd is None:
        raise RuntimeError(
            "Microphone recording is unavailable because PortAudio is not installed."
        ) from SOUNDDEVICE_IMPORT_ERROR

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
    Send the transcribed message to the LangChain Ollama agent.
    """

    conversation.append(
        {
            "role": "user",
            "content": user_text,
        }
    )

    try:
        response = agent.invoke({"messages": conversation})
    except Exception as error:
        conversation.pop()
        raise RuntimeError(
            "LangChain could not get a response from Ollama. "
            "Make sure Ollama is running and the model is available:\n"
            f"  ollama serve\n"
            f"  ollama pull {OLLAMA_MODEL}"
        ) from error

    reply_message = response["messages"][-1]
    reply_content = getattr(reply_message, "content", "")

    if isinstance(reply_content, str):
        reply = reply_content.strip()
    elif isinstance(reply_content, list):
        reply = " ".join(
            part.get("text", "").strip()
            for part in reply_content
            if isinstance(part, dict) and part.get("text")
        ).strip()
    else:
        reply = str(reply_content).strip()

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

    if tts_engine is None:
        print(
            "Text-to-speech is unavailable on this machine. "
            "Response will be shown as text only."
        )
        return

    tts_engine.say(text)
    tts_engine.runAndWait()


def main() -> None:
    print("\nLocal Voice Assistant")
    print("---------------------")
    print(f"Ollama model: {OLLAMA_MODEL}")
    print(f"Whisper model: {WHISPER_MODEL_NAME}")
    print("Everything runs locally.")
    if sd is None:
        print("Microphone input disabled: PortAudio is not installed.")
    if tts_engine is None:
        print("Text-to-speech disabled: pyttsx3 engine is unavailable.")
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
