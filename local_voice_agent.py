import os
from typing import Optional

from langchain.agents import create_agent
from speech.local_text_to_voice import speak, tts_engine
from speech.local_recorder import record_audio, sd
from speech.local_voice_to_text import WHISPER_MODEL_NAME, transcribe

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

OLLAMA_MODEL = "qwen3:4b"

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
