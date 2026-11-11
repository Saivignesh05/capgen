"""Stage 2, asr-captioner environment: describe each clip with the captioner.

Started by scripts/run.py, which decides which clips to give it. It processes every clip in --tasks,
appends one DescriptionRecord per success to --output and one ErrorRecord per failure to --errors.
"""
import argparse

from pipeline.audio_types import AudioClip
from pipeline.models.captioner import Captioner
from pipeline.settings import captioner_settings, load_settings
from scripts.records import ClipRecord, DescriptionRecord, ErrorRecord, append_record, read_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--tasks", required=True, help="JSONL of ClipRecord")
    parser.add_argument("--output", required=True, help="descriptions.jsonl")
    parser.add_argument("--errors", required=True, help="errors.jsonl")
    args = parser.parse_args()

    tasks = read_records(args.tasks, ClipRecord)
    captioner = Captioner(captioner_settings(load_settings(args.settings)))

    for number, task in enumerate(tasks, 1):
        try:
            description = captioner.describe(AudioClip(path=task.path))
        except Exception as error:      # one bad clip must not stop the others
            append_record(args.errors, ErrorRecord(stage="describe", clip_id=task.clip_id, sample=None, error=repr(error)))
            print(f"[describe {number}/{len(tasks)}] {task.clip_id}: FAILED")
            continue
        append_record(args.output, DescriptionRecord(clip_id=task.clip_id, description=description.text))
        print(f"[describe {number}/{len(tasks)}] {task.clip_id}")


if __name__ == "__main__":
    main()
