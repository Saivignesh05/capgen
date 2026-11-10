
import soundfile as sf
import numpy as np

from pathlib import Path

from generator.indic_parler_generator import AudioResult


def save_audio(
    audio: AudioResult,
    output_dir,
    filename
) -> str:

    """
    Save generated audio as a standard WAV file.

    Input:
        audio: AudioResult object.

        output_dir: Destination directory.

        filename: WAV filename.

    Output:
        Absolute path of saved WAV.
    """

    if not isinstance(audio, AudioResult):

        raise TypeError(
            "Expected AudioResult object."
        )

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    filename = Path(filename)

    if filename.name != str(filename):

        raise ValueError(
            "Filename must not contain directory paths."
        )

    if filename.suffix.lower() != ".wav":

        filename = filename.with_suffix(".wav")

    output_path = output_dir / filename

    waveform = np.asarray(
        audio.waveform,
        dtype=np.float32
    )

    if not np.isfinite(waveform).all():

        raise ValueError(
            "Waveform contains invalid values."
        )

    sf.write(
        str(output_path),
        waveform,
        audio.sample_rate,
        format="WAV",
        subtype="PCM_16"
    )

    print("\nAudio saved successfully.")

    print("Location:", output_path.resolve())

    return str(output_path.resolve())

