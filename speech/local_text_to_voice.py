try:
    import pyttsx3
except Exception:
    pyttsx3 = None


print("Initializing local text-to-speech...")

tts_engine = None
if pyttsx3 is not None:
    tts_engine = pyttsx3.init()
    tts_engine.setProperty("rate", 175)
    tts_engine.setProperty("volume", 1.0)


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

if __name__ == "__main__":
    speak("Hello, this is a test.")