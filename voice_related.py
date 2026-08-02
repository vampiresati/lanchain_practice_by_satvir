import os
import tempfile
import soundfile as sf
from dotenv import load_dotenv
from google import genai
from google.genai import types

try:
    import sounddevice as sd
    SOUNDDEVICE_IMPORT_ERROR = None
except OSError as error:
    sd = None
    SOUNDDEVICE_IMPORT_ERROR = error

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in the environment.")

client = genai.Client(api_key=GOOGLE_API_KEY)

SAMPLE_RATE = 16000
MAX_DURATION = 30


def record_audio() -> str:
    """Record from microphone, return path to temp WAV file."""
    if sd is None:
        raise RuntimeError(
            "Microphone recording is unavailable because PortAudio is not installed."
        ) from SOUNDDEVICE_IMPORT_ERROR

    input("Press Enter to start recording...")
    print("Recording... Press Enter to stop.")

    audio_data = sd.rec(
        int(MAX_DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float64",
    )

    input()
    sd.stop()
    print("Recording stopped.")

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio_data, SAMPLE_RATE)
    return tmp.name


def transcribe(audio_path: str) -> str:
    """Send audio to Gemini and return the transcript."""
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            "Transcribe this audio. Return only the spoken text.",
            types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
        ],
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=512,
        ),
    )
    return response.text.strip()


def think(text: str) -> str:
    """Send text to Gemini and return the response."""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=text,
        config=types.GenerateContentConfig(
            system_instruction="You are a helpful voice assistant. Keep responses short and conversational.",
            temperature=0.7,
            max_output_tokens=256,
        )
    )
    return response.text.strip()


def speak(text: str):
    """Convert text to speech and play it."""
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["audio"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="charon"
                    )
                )
            ),
        ),
    )

    audio_part = next(
        (
            part
            for part in response.candidates[0].content.parts
            if getattr(part, "inline_data", None)
        ),
        None,
    )
    if not audio_part or not audio_part.inline_data or not audio_part.inline_data.data:
        raise ValueError("Gemini did not return audio data.")

    mime_type = audio_part.inline_data.mime_type or "audio/wav"
    suffix = ".wav" if "wav" in mime_type else ".mp3"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(audio_part.inline_data.data)
    tmp.close()

    if sd is None:
        print(f"Audio playback is unavailable because PortAudio is not installed.")
        print(f"Saved Gemini audio response to: {tmp.name}")
        return

    data, sr = sf.read(tmp.name)
    sd.play(data, sr)
    sd.wait()
    os.unlink(tmp.name)


def main():
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


if __name__ == "__main__":
    main()
