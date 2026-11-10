"""
generator – Indic Parler-TTS audio generation package.

Public API
----------
    from generator import IndicParlerGenerator, AudioResult, save_audio

    gen = IndicParlerGenerator()
    result = gen.generate_audio(transcript="...", description="...")
    save_audio(result, output_dir="outputs/generated_audio", filename="out.wav")
"""

from generator.indic_parler_generator import AudioResult, IndicParlerGenerator
from generator.audio_writer import save_audio

__all__ = ["AudioResult", "IndicParlerGenerator", "save_audio"]
