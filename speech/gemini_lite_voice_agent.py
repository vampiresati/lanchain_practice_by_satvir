import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from local_recorder import record_audio, sd
from local_text_to_voice import speak
from local_voice_to_text import transcribe

load_dotenv()

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
SYSTEM_PROMPT = """
You are a teacher talking assistant.
Keep responses short, clear, and conversational.
"""

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0.7,
)

agent = create_agent(
    model=llm,
    system_prompt=SYSTEM_PROMPT,
    tools=[],
)


def think(text: str) -> str:
    """Send user query to Gemini and return the response."""
    response = agent.invoke({"messages": [{"role": "user", "content": text}]})
    reply_message = response["messages"][-1]
    reply_content = getattr(reply_message, "content", "")

    if isinstance(reply_content, str):
        return reply_content.strip()

    if isinstance(reply_content, list):
        return " ".join(
            part.get("text", "").strip()
            for part in reply_content
            if isinstance(part, dict) and part.get("text")
        ).strip()

    return str(reply_content).strip()


def main() -> None:
    if sd is None:
        print("PortAudio is not available, so microphone recording is disabled.")
        transcript = input("Type your message instead: ").strip()
        if not transcript:
            raise ValueError("No input provided.")
    else:
        audio_file = record_audio()
        try:
            transcript = transcribe(audio_file)
            print(f"\nYou said: {transcript}")
        finally:
            os.unlink(audio_file)

    reply = think(transcript)
    if not reply:
        raise RuntimeError("Gemini returned an empty response.")

    print(f"AI: {reply}")
    speak(reply)


if __name__ == "__main__":
    main()
