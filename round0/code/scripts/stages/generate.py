"""Stage 3, generator environment: speak each transcript in the voice its description asks for.

Started by scripts/run.py, which decides which samples to make and gives each its seed. It makes
every sample in --tasks, writes it to <run folder>/audio/<clip_id>/<sample>.wav, appends one
AudioRecord per success to --output and one ErrorRecord per failure to --errors.
"""
import argparse
import os

import numpy as np
import soundfile
import torch

from pipeline.audio_types import AudioDescription, AudioTranscript
from pipeline.models.generator import Generator
from pipeline.settings import generator_settings, load_settings
from scripts.records import AudioRecord, ErrorRecord, GenerationTask, append_record, read_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--tasks", required=True, help="JSONL of GenerationTask")
    parser.add_argument("--output", required=True, help="audio.jsonl, in the run folder")
    parser.add_argument("--errors", required=True, help="errors.jsonl")
    args = parser.parse_args()

    tasks = read_records(args.tasks, GenerationTask)
    generator = Generator(generator_settings(load_settings(args.settings)))
    run_folder = os.path.dirname(os.path.abspath(args.output))

    for number, task in enumerate(tasks, 1):
        torch.manual_seed(task.seed)        # CAREFUL R3 (scripts/run.py)
        try:
            audio = generator.generate(
                AudioTranscript(text=task.transcript, language=task.language),
                AudioDescription(text=task.description))
            if audio.waveform.size == 0 or not np.isfinite(audio.waveform).all():
                raise ValueError("empty waveform or NaN/infinite values")
        except Exception as error:      # one bad sample must not stop the others
            append_record(args.errors, ErrorRecord(stage="generate", clip_id=task.clip_id, sample=task.sample, error=repr(error)))
            print(f"[generate {number}/{len(tasks)}] {task.clip_id} #{task.sample}: FAILED")
            continue

        path = os.path.join(run_folder, "audio", task.clip_id, f"{task.sample}.wav")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Write under a temporary name, then rename: a crash never leaves a half-written .wav.
        soundfile.write(path + ".partial", audio.waveform, audio.sample_rate, format="WAV", subtype="PCM_16")     # CAREFUL R4 (scripts/run.py)
        os.replace(path + ".partial", path)
        append_record(args.output, AudioRecord(clip_id=task.clip_id, sample=task.sample, seed=task.seed, file=os.path.relpath(path, run_folder)))
        print(f"[generate {number}/{len(tasks)}] {task.clip_id} #{task.sample}")


if __name__ == "__main__":
    main()
