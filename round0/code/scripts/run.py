"""The runner: run the whole pipeline on a folder of clips, as a new run or by resuming one.

Run from the code folder with the asr-captioner environment's Python:

    python -m scripts.run new --settings ../settings.yaml --clips /path/to/devset --runs /path/to/runs --environments /path/to/envs
    python -m scripts.run resume --run latest --runs /path/to/runs --environments /path/to/envs

Every new run gets its own folder, <runs>/<run_id>/:

    run.json             lineage: inputs, code, settings, models, environments, and every attempt
    settings.yaml        the settings the run uses: a copy, so a run never mixes two settings
    clips.jsonl          the input clips with their SHA-256, fixed when the run starts
    transcripts.jsonl    stage 1, ASR:        clip_id, language, transcript
    descriptions.jsonl   stage 2, captioner:  clip_id, description
    audio.jsonl          stage 3, generator:  clip_id, sample, seed, file
    audio/<clip_id>/<sample>.wav
    errors.jsonl         every failure of every attempt

<runs>/latest.json names the newest run. Resuming does only what is missing: clips with no
transcript or description yet, and samples not written yet. Every output line is keyed by clip_id,
so any audio file traces back to its description, its transcript and its input clip.
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys

from pipeline.seeds import sample_seed
from pipeline.settings import load_settings, model_folder
from scripts.records import (AudioRecord, ClipRecord, DescriptionRecord, GenerationTask, TranscriptRecord,
                             read_records, write_records)

CODE_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE_ENVIRONMENT = {"transcribe": "asr-captioner", "describe": "asr-captioner", "generate": "generator"}
FROZEN_KEYS = ["code_commit", "settings_sha256", "environment_locks"]     # must not change within a run. CAREFUL R1


# ---------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------

def now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CODE_FOLDER, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def gpus() -> list[str]:
    """Name and driver of each visible GPU, from nvidia-smi, or [] without it."""
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def environment_locks() -> dict[str, str | None]:
    """SHA-256 of each environment's lock file: which exact packages the stages ran with."""
    locks = {}
    for name in sorted(set(STAGE_ENVIRONMENT.values())):
        path = os.path.join(CODE_FOLDER, "environments", name, "requirements.lock.txt")
        locks[name] = file_sha256(path) if os.path.isfile(path) else None
    return locks


def read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: dict) -> None:
    """Write to a temporary file, then rename, so the file is never half-written."""
    with open(path + ".partial", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(path + ".partial", path)


# ---------------------------------------------------------------------------------------------
# Planning: what is still to do
# ---------------------------------------------------------------------------------------------

def find_clips(folder: str) -> list[ClipRecord]:
    """Every .wav under the folder, with its id (file name without .wav) and SHA-256. CAREFUL R2"""
    clips = {}
    for path in sorted(glob.glob(os.path.join(folder, "**", "*.wav"), recursive=True)):
        clip_id = os.path.splitext(os.path.basename(path))[0]
        if clip_id in clips:
            raise ValueError(f"two clips share the id {clip_id}: {clips[clip_id].path} and {path}")
        clips[clip_id] = ClipRecord(clip_id=clip_id, path=os.path.abspath(path), sha256=file_sha256(path))
    if not clips:
        raise ValueError(f"no .wav files under {folder}")
    return list(clips.values())


def missing_clips(clips: list[ClipRecord], done: list) -> list[ClipRecord]:
    """Clips with no record yet in a stage's output."""
    done_ids = {record.clip_id for record in done}
    return [clip for clip in clips if clip.clip_id not in done_ids]


def generation_tasks(clips: list[ClipRecord], transcripts: list[TranscriptRecord], descriptions: list[DescriptionRecord],
                     done: list[AudioRecord], samples_per_item: int, base_seed: int) -> list[GenerationTask]:
    """Every sample still to make, for clips that have a non-empty transcript and a description. CAREFUL R3"""
    transcript_of = {record.clip_id: record for record in transcripts}
    description_of = {record.clip_id: record for record in descriptions}
    done_samples = {(record.clip_id, record.sample) for record in done}
    tasks = []
    for clip in clips:
        transcript = transcript_of.get(clip.clip_id)
        description = description_of.get(clip.clip_id)
        if transcript is None or description is None or transcript.transcript == "":
            continue
        for sample in range(samples_per_item):
            if (clip.clip_id, sample) not in done_samples:
                tasks.append(GenerationTask(
                    clip_id=clip.clip_id, sample=sample, seed=sample_seed(base_seed, clip.clip_id, sample),
                    language=transcript.language, transcript=transcript.transcript, description=description.description))
    return tasks


# ---------------------------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------------------------

def lineage(run_folder: str) -> dict:
    """What a run's outputs depend on, apart from its inputs."""
    return {
        "code_commit": git_commit(),
        "settings_sha256": file_sha256(os.path.join(run_folder, "settings.yaml")),
        "environment_locks": environment_locks(),
    }


def new_run(settings_path: str, clips_folder: str, runs_folder: str) -> str:
    settings = load_settings(settings_path)     # check the settings before creating anything
    clips = find_clips(clips_folder)

    run_id = datetime.datetime.now().strftime("run-%Y%m%d-%H%M%S")
    run_folder = os.path.join(runs_folder, run_id)
    os.makedirs(run_folder)     # fails if it exists: a run is never overwritten
    shutil.copyfile(settings_path, os.path.join(run_folder, "settings.yaml"))
    write_records(os.path.join(run_folder, "clips.jsonl"), clips)

    write_json(os.path.join(run_folder, "run.json"), {
        "run_id": run_id,
        "created": now(),
        "clips_folder": os.path.abspath(clips_folder),
        "clip_count": len(clips),
        **lineage(run_folder),
        "models": {name: f"{repo.repo_id}@{repo.revision}" for name, repo in settings.downloads.named()},
        "attempts": [],
    })
    write_json(os.path.join(runs_folder, "latest.json"), {"run_id": run_id})
    print(f"== New run {run_id}: {len(clips)} clips")
    return run_folder


def existing_run(run: str, runs_folder: str) -> str:
    run_id = read_json(os.path.join(runs_folder, "latest.json"))["run_id"] if run == "latest" else run
    run_folder = os.path.join(runs_folder, run_id)
    recorded = read_json(os.path.join(run_folder, "run.json"))
    current = lineage(run_folder)
    for key in FROZEN_KEYS:
        if recorded[key] != current[key]:
            raise SystemExit(f"Cannot resume {run_id}: its {key} has changed since it started. Start a new run instead.")
    print(f"== Resuming run {run_id}")
    return run_folder


def run_stage(stage: str, tasks: list, run_folder: str, output_file: str, environments_folder: str, extra_environment: dict) -> None:
    """Give a stage its tasks and run it in its environment's Python. A crash stops the runner. CAREFUL R4"""
    if not tasks:
        print(f"== {stage}: nothing to do")
        return
    print(f"== {stage}: {len(tasks)} to do")
    tasks_path = os.path.join(run_folder, f"tasks_{stage}.jsonl")
    write_records(tasks_path, tasks)
    python = os.path.join(environments_folder, STAGE_ENVIRONMENT[stage], "bin", "python")
    subprocess.run(
        [python, "-m", f"scripts.stages.{stage}",
         "--settings", os.path.join(run_folder, "settings.yaml"),
         "--tasks", tasks_path,
         "--output", os.path.join(run_folder, output_file),
         "--errors", os.path.join(run_folder, "errors.jsonl")],
        cwd=CODE_FOLDER, env={**os.environ, **extra_environment}, check=True)
    os.remove(tasks_path)


def run_stages(run_folder: str, environments_folder: str) -> bool:
    """Do whatever is missing in the run; record the attempt in run.json. True if the run is complete."""
    settings = load_settings(os.path.join(run_folder, "settings.yaml"))
    clips = read_records(os.path.join(run_folder, "clips.jsonl"), ClipRecord)
    transcripts_path = os.path.join(run_folder, "transcripts.jsonl")
    descriptions_path = os.path.join(run_folder, "descriptions.jsonl")
    audio_path = os.path.join(run_folder, "audio.jsonl")
    attempt = {"started": now(), "gpus": gpus()}

    # The ASR's code ships inside its model folder, so its stage needs that folder on PYTHONPATH.
    asr_folder = model_folder(settings, settings.downloads.asr)
    run_stage("transcribe", missing_clips(clips, read_records(transcripts_path, TranscriptRecord)),
              run_folder, "transcripts.jsonl", environments_folder, {"PYTHONPATH": asr_folder})
    run_stage("describe", missing_clips(clips, read_records(descriptions_path, DescriptionRecord)),
              run_folder, "descriptions.jsonl", environments_folder, {})

    transcripts = read_records(transcripts_path, TranscriptRecord)
    descriptions = read_records(descriptions_path, DescriptionRecord)
    run_stage("generate", generation_tasks(clips, transcripts, descriptions, read_records(audio_path, AudioRecord),
                                           settings.generation.samples_per_item, settings.generation.seed),
              run_folder, "audio.jsonl", environments_folder, {})

    no_speech = sum(1 for record in transcripts if record.transcript == "")
    speakable = len(transcripts) - no_speech
    summary = {
        "transcripts": f"{len(transcripts)}/{len(clips)}",
        "descriptions": f"{len(descriptions)}/{len(clips)}",
        "audio_samples": f"{len(read_records(audio_path, AudioRecord))}/{speakable * settings.generation.samples_per_item}",
        "no_speech_clips": no_speech,
    }
    missing_generation = generation_tasks(clips, transcripts, descriptions, read_records(audio_path, AudioRecord),
                                          settings.generation.samples_per_item, settings.generation.seed)
    complete = len(transcripts) == len(clips) and len(descriptions) == len(clips) and not missing_generation

    run_json = read_json(os.path.join(run_folder, "run.json"))
    run_json["attempts"].append({**attempt, "finished": now(), "complete": complete, **summary})
    write_json(os.path.join(run_folder, "run.json"), run_json)
    print(f"== {'Complete' if complete else 'Incomplete (see errors.jsonl, then resume)'}: {summary}")
    return complete


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    new = commands.add_parser("new", help="start a new run")
    new.add_argument("--settings", required=True, help="path to settings.yaml")
    new.add_argument("--clips", required=True, help="folder searched recursively for .wav files")
    resume = commands.add_parser("resume", help="continue a run")
    resume.add_argument("--run", required=True, help='a run id, or "latest"')
    for command in [new, resume]:
        command.add_argument("--runs", required=True, help="folder that holds every run")
        command.add_argument("--environments", required=True, help="folder that holds the asr-captioner and generator environments")
    args = parser.parse_args()

    if args.command == "new":
        run_folder = new_run(args.settings, args.clips, args.runs)
    else:
        run_folder = existing_run(args.run, args.runs)
    complete = run_stages(run_folder, args.environments)
    sys.exit(0 if complete else 1)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------------------------
# CAREFUL codes
#
# R1   A run is resumed only if the code commit, its settings file and the environments' lock
#      files are unchanged since it started, so every output of a run comes from one setup. If any
#      changed, start a new run.
# R2   A clip's id is its file name without .wav: two files with the same name in different
#      subfolders are an error rather than one silently replacing the other. Each clip's SHA-256
#      is recorded, so outputs can be traced to the exact audio.
# R3   Each sample's seed depends only on generation.seed, the clip id and the sample number, so a
#      sample is the same however often the run is resumed (on the same GPU and library versions).
#      Every sample is kept: the competition forbids choosing between generations. Clips whose
#      transcript is empty (no speech) are not spoken; run.json counts them as no_speech_clips.
# R4   A stage skips a failing clip or sample and carries on, and the runner then reports the run
#      as incomplete. If a stage itself crashes (out of memory, killed job), the runner stops;
#      resume continues from the last line each stage wrote. Generated audio is 16-bit WAV at the
#      model's own sample rate (44.1 kHz for Indic Parler-TTS; the competition asks 24 kHz or more).
# ---------------------------------------------------------------------------------------------
