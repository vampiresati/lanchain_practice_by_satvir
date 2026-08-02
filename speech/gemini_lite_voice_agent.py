import os

from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import create_agent
from speech.local_recorder import record_audio, sd
from speech.local_text_to_voice import speak
from speech.local_voice_to_text import transcribe

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.7,
)
system_prompt = """
You are a teacher talking assistant. 
"""


def think(text: str) -> str:
    """Send user query to the Gemini agent and return the response."""
    agent = create_agent(
        model=llm,
        system_prompt=system_prompt)
    response = agent.invoke({"messages": [{"role": "user", "content": text}]})
    return response["messages"][-1]["content"].strip()


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
    print(f"AI: {reply}")
    speak(reply)
