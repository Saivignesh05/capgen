
from pathlib import Path

from generator.indic_parler_generator import (
    IndicParlerGenerator
)

from generator.audio_writer import save_audio


ROOT_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = (
    ROOT_DIR / "outputs" / "generated_audio"
)


def main():

    print("=" * 60)
    print("INDIC PARLER TTS GENERATOR TEST")
    print("=" * 60)

    # Load model once
    generator = IndicParlerGenerator(
        device="cpu"
    )

    # Hindi test input
    transcript = "नमस्ते, आप कैसे हैं?"

    description = (
        "Divya speaks in Hindi with a calm, friendly "
        "and natural tone. She speaks at a moderate "
        "speed with clear pronunciation and a medium "
        "pitch. The recording is of very high quality "
        "with no background noise."
    )

    # Generate audio
    audio_result = generator.generate_audio(
        transcript=transcript,
        description=description,
        seed=42
    )

    # Inspect returned audio representation
    print("\nAudioResult details:")

    print(
        "Waveform type:",
        type(audio_result.waveform)
    )

    print(
        "Waveform dtype:",
        audio_result.waveform.dtype
    )

    print(
        "Waveform shape:",
        audio_result.waveform.shape
    )

    print(
        "Sampling rate:",
        audio_result.sample_rate
    )

    print(
        "Duration:",
        audio_result.duration
    )

    # Save WAV file
    audio_path = save_audio(
        audio=audio_result,
        output_dir=OUTPUT_DIR,
        filename="hindi_test_001.wav"
    )

    print("\nFinal audio:", audio_path)

    print("\nTEST COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":

    main()

