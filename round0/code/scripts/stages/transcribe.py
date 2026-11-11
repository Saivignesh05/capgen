"""Stage 1, asr-captioner environment: identify the language of each clip and transcribe it.

Started by scripts/run.py, which decides which clips to give it. It processes every clip in --tasks,
appends one TranscriptRecord per success to --output and one ErrorRecord per failure to --errors.
"""
import argparse

from pipeline.audio_types import AudioClip
from pipeline.models.asr import ASR
from pipeline.settings import asr_settings, load_settings
from scripts.records import ClipRecord, ErrorRecord, TranscriptRecord, append_record, read_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--tasks", required=True, help="JSONL of ClipRecord")
    parser.add_argument("--output", required=True, help="transcripts.jsonl")
    parser.add_argument("--errors", required=True, help="errors.jsonl")
    args = parser.parse_args()

    tasks = read_records(args.tasks, ClipRecord)
    asr = ASR(asr_settings(load_settings(args.settings)))

    for number, task in enumerate(tasks, 1):
        try:
            clip = AudioClip(path=task.path)
            language = asr.identify(clip)
            transcript = asr.transcribe(clip, language)
        except Exception as error:      # one bad clip must not stop the others
            append_record(args.errors, ErrorRecord(stage="transcribe", clip_id=task.clip_id, sample=None, error=repr(error)))
            print(f"[transcribe {number}/{len(tasks)}] {task.clip_id}: FAILED")
            continue
        append_record(args.output, TranscriptRecord(clip_id=task.clip_id, language=language, transcript=transcript.text))
        print(f"[transcribe {number}/{len(tasks)}] {task.clip_id}: {language}")


if __name__ == "__main__":
    main()
