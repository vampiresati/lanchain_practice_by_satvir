import tempfile
import time

import soundfile as sf

try:
    import sounddevice as sd
    SOUNDDEVICE_IMPORT_ERROR = None
except OSError as error:
    sd = None
    SOUNDDEVICE_IMPORT_ERROR = error

SAMPLE_RATE = 16_000
CHANNELS = 1
MAX_DURATION = 30


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

if __name__ == "__main__":
    path=record_audio()
    print(f"Audio saved to {path}")