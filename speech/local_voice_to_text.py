from faster_whisper import WhisperModel

WHISPER_MODEL_NAME = "base.en"

# print("Loading local Whisper model...")

whisper_model = WhisperModel(
    WHISPER_MODEL_NAME,
    device="cpu",
    compute_type="int8",
)


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
if __name__ == "__main__":
    transcript=transcribe("/tmp/tmpvrks9m2b.wav")
    print(transcript)