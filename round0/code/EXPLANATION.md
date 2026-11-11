# How the code works

For the steps to reproduce the results, see `../README.md`.

## Layout

```
round0/
├── README.md                   how to reproduce the results
├── settings.yaml               every setting and value, for devset runs
├── settings_eval.yaml          the same, with 10 samples per description for the evaluation outputs
├── checkpoint_hashes.json      SHA-256 of every model checkpoint (written by scripts/hash_checkpoints.py)
└── code/
    ├── EXPLANATION.md          this file
    ├── pipeline/               the Python package
    │   ├── audio_types.py      shared inputs and outputs
    │   ├── settings.py         the classes that check settings.yaml, and load_settings()
    │   ├── seeds.py            how each generated sample's seed is derived
    │   └── models/
    │       ├── captioner.py    Captioner   (asr-captioner environment)
    │       ├── asr.py          ASR         (asr-captioner environment)
    │       └── generator.py    Generator   (generator environment)
    ├── scripts/
    │   ├── setup_envs.sh       build environments, download models, pack (machine with internet)
    │   ├── unpack_envs.sh      unpack environments (cluster)
    │   ├── download.py         download every model at its pinned commit
    │   ├── hash_checkpoints.py write or check checkpoint_hashes.json
    │   ├── run.py              the runner: runs, resuming, lineage
    │   ├── records.py          the record types of a run's files
    │   └── stages/
    │       ├── transcribe.py   stage 1 (asr-captioner): identify language, transcribe
    │       ├── describe.py     stage 2 (asr-captioner): describe
    │       └── generate.py     stage 3 (generator): speak each transcript several times
    ├── environments/
    │   ├── asr-captioner/      environment.yml, requirements.txt, requirements.lock.txt
    │   └── generator/          environment.yml, requirements.txt, requirements.lock.txt
    └── tests/
        ├── helpers.py, conftest.py
        ├── unit/               no models needed
        └── models/             real models
```

Settings live outside `code/` so that what runs (the code) and how it is configured (one YAML file) stay separate. Every script takes the settings file as `--settings`; nothing has a default.

## The models

| Model | Module | In | Out | Pinned repo |
|---|---|---|---|---|
| Captioner | `pipeline.models.captioner` | `AudioClip` | `AudioDescription` (English) | `Qwen/Qwen3-Omni-30B-A3B-Captioner` |
| ASR | `pipeline.models.asr` | `AudioClip` (+ language to transcribe) | `AudioLanguage`, `AudioTranscript` | `bodhan-ai/indic-transcribe-core` |
| Generator | `pipeline.models.generator` | `AudioTranscript` + `AudioDescription` | `AudioOut` | `ai4bharat/indic-parler-tts` (+ `google/flan-t5-large` tokenizer) |

Each module has an interface (`CaptionerInterface`, `ASRInterface`, `GeneratorInterface`) and one implementation, so a later round's model can replace one by implementing the same interface.

`pipeline/__init__.py` and `pipeline/models/__init__.py` import nothing on purpose: the captioner and the generator run in different environments, so importing one model must never import another.

## The runner

`scripts/run.py` runs the whole pipeline on a folder of clips (for example the devset). The captioner and the generator need different versions of `transformers` (see "Two environments"), so the models cannot all run in one Python process. The runner starts each stage as its own process, with its environment's Python, and the stages pass data through files:

```
clips ──► transcribe (ASR) ──────► transcripts.jsonl ──┐
      └─► describe (captioner) ──► descriptions.jsonl ─┴─► generate (generator) ──► audio/<clip_id>/<n>.wav, audio.jsonl
```

All bookkeeping lives in the runner. The stages know nothing about runs: each gets a list of tasks, does them, and appends one line per result to its output file (or one line per failure to `errors.jsonl`). The runner decides what each stage still has to do by reading what is already in the output files.

### Runs

Every `new` run gets its own folder, `<runs>/<run_id>/`, and `<runs>/latest.json` names the newest one:

| File | One line per | Fields |
|---|---|---|
| `clips.jsonl` | input clip, fixed when the run starts | `clip_id`, `path`, `sha256` |
| `transcripts.jsonl` | transcribed clip | `clip_id`, `language`, `transcript` |
| `descriptions.jsonl` | described clip | `clip_id`, `description` (the competition's describe format) |
| `audio.jsonl` | generated sample | `clip_id`, `sample`, `seed`, `file` |
| `errors.jsonl` | failure, in any attempt | `stage`, `clip_id`, `sample`, `error` |

Plus `audio/<clip_id>/<sample>.wav`, `settings.yaml` (a copy of the settings the run uses), and `run.json`.

### Lineage

`run.json` records what the outputs depend on:

- the clips folder and clip count (each clip's SHA-256 is in `clips.jsonl`)
- the code commit, the SHA-256 of the run's `settings.yaml`, and the SHA-256 of each environment's lock file
- each model as `repo_id@revision`
- every attempt: when it started and finished, the GPUs it ran on, and how far it got

Every output line is keyed by `clip_id`, so any audio file traces back to its seed, description, transcript and input clip.

### Resuming

`resume --run latest` (or a run id) does only what is missing: clips with no transcript or description yet, and samples not yet written. Failed clips are tried again; finished ones are never redone. A run is resumed only if its code commit, settings and lock files are unchanged since it started, so all of a run's outputs come from one setup. If anything changed, start a new run.

If a stage crashes outright (out of memory, a killed job), the runner stops. Each stage has already written every result it finished, so resuming continues from there.

### Generation

A clip is spoken once it has both a transcript and a description, `samples_per_item` times (1 in `settings.yaml`, 10 in `settings_eval.yaml`). Clips with an empty transcript (no speech) are not spoken; `run.json` counts them as `no_speech_clips`. Each sample's seed is derived from `generation.seed`, the clip id and the sample number, so a resumed sample is the same as it would have been. Every sample is kept: the competition forbids choosing between generations.

## Reproducibility

| What | How it is fixed |
|---|---|
| Model weights | Each repo pinned to a commit in `settings.yaml`; every file's SHA-256 in `checkpoint_hashes.json`, checked with `scripts/hash_checkpoints.py --check` |
| Python packages | `requirements.lock.txt` per environment: exact versions, used by every build after the first |
| Settings | One file, copied into each run; its SHA-256 is in `run.json` |
| Code | The git commit is in `run.json` |
| Captioner output | Greedy decoding: no randomness |
| Generator output | Each sample's seed is SHA-256(`seed|id|n`), so it depends only on the item and sample number, not on run order or resumes. Every sample is kept: the competition forbids choosing between generations |

What cannot be fixed in software is GPU arithmetic: the same inputs can give slightly different numbers on a different GPU model, driver or GPU count, and with greedy decoding one changed token changes the rest of a description. Results repeat exactly on the same GPU type with the same environment; `run.json` records the GPUs of every attempt.

### Checkpoint hashes

The competition asks for "the SHA-256 hash of every model checkpoint": proof that the exact weights you used are the ones a reviewer runs. A Hugging Face commit says which files to download; a SHA-256 hash proves which bytes are on disk, including after copying them to another machine.

`scripts/hash_checkpoints.py` hashes every file of each model folder, and gives each model one hash: the SHA-256 of its `sha256sum`-style list of `<file hash>  <file name>` lines. Hugging Face's `.cache` bookkeeping and the download marker are not part of a checkpoint and are skipped. `--write` creates `checkpoint_hashes.json` once; `--check` recomputes and compares, and exits with an error on any mismatch.

## Settings

`settings.yaml` has one section per part. `pipeline/settings.py` has a class for each section: unknown or misspelled keys are errors, every key is required, and values cannot change after loading.

- `model_dir`: every model is stored in `model_dir/<repo_id>`, so each repo id is written once and the download folder and the load folder cannot drift apart. This folder must exist.
- `downloads`: each repo with its pinned commit.
- `captioner`, `asr`, `generator`: model options.
- `generation`: samples per item and the base seed.

There are two settings files. `settings.yaml` makes 1 sample per clip, enough for devset runs; `settings_eval.yaml` makes 10, as the competition asks for each evaluation description. Everything else is identical, and `tests/unit/test_settings.py` fails if the two files ever differ in anything but `samples_per_item`. Because each sample's seed depends only on the clip and the sample number, sample 0 is the same in both.

Loading `settings.yaml` does not check that model folders exist, because `scripts/download.py` loads it before they do. Each model's settings (`captioner_settings(settings)` and the others) combine its options with its folder and check the folder only when that model is about to load.

## Two environments

- Qwen3-Omni was added to `transformers` in version 4.57.0.
- `parler-tts` requires `transformers` 4.46.1 exactly.

So there are two conda environments, each defined in its own folder under `environments/`:

- `environment.yml`: the conda part, Python 3.10 and pip.
- `requirements.txt`: what the environment needs, fixing only the versions that must be fixed. The generator's file comes from the verified generator setup, with its fixes for `protobuf` and `sentencepiece`.
- `requirements.lock.txt`: written by the first build with the exact version of every package. Every later build installs from it instead, so everyone gets the same environment.

`torch` is in neither `requirements.txt`: the first build installs it first, from the PyTorch index you give, because the right CUDA build depends on the GPU driver. The ASR's own requirements come from its model folder and are installed with the fixed versions as constraints, so if it ever needs something different, the build stops instead of silently breaking the captioner.

The cluster has no internet, so environments are built elsewhere, packed into one file each with `conda-pack`, copied, and unpacked with `unpack_envs.sh`.

## Downloads

`scripts/download.py` writes a marker file, `.download_complete.json`, into each model's folder after it finishes:

- Marker with the same settings: skipped without contacting Hugging Face.
- Marker with different settings (another revision, say): an error instead of mixing two versions.
- No marker (first run, or a crash part-way): downloads, resuming half-finished files.

## Design rules

- **No defaults.** Every value is in `settings.yaml` or a required command-line argument.
- **Inputs are checked once, where they are created.** An `AudioClip` only exists if its file is readable audio; settings are checked when loaded.
- **Models return raw output; whoever uses it decides what is valid.** An empty transcript is allowed, since a clip may have no speech; the generator, which cannot speak it, raises. Text has surrounding whitespace removed by its type.
- **Values are read from the model where possible.** The captioner reads its sample rate from the model's config; the ASR resamples audio itself; the generator reads its output rate from the model's config.
- **Imports are at the top of each file.** This is why the ASR's model folder goes on `PYTHONPATH` instead of being added inside the code.

## CAREFUL codes

Lines that need care end with a tag such as `# CAREFUL C4`. Each code is explained at the bottom of its file:

| Prefix | File |
|---|---|
| C | `pipeline/models/captioner.py` |
| A | `pipeline/models/asr.py` |
| G | `pipeline/models/generator.py` |
| S | `pipeline/settings.py` |
| D | `scripts/download.py` |
| H | `scripts/hash_checkpoints.py` |
| R | `scripts/run.py` |
| E | `scripts/setup_envs.sh` |

Fields in `pipeline/settings.py` name the file their code is in, for example `CAREFUL C1 (pipeline/models/captioner.py)`.

## Tests

Run from `code/`. Anything a machine cannot run is skipped with a reason (`pytest -rs` prints them). Run the generator tests in the `generator` environment and the rest in `asr-captioner`; a test run in the wrong environment skips.

| Test file | Uses real models? | Skipped when |
|---|---|---|
| `tests/unit/test_audio_types.py` | No | `soundfile` is not installed |
| `tests/unit/test_settings.py` | No | never |
| `tests/unit/test_seeds.py` | No | never |
| `tests/unit/test_run.py` | No | `soundfile` is not installed |
| `tests/unit/test_download.py` | No (the download is faked) | `huggingface_hub` is not installed |
| `tests/unit/test_hash_checkpoints.py` | No (small fake models) | `huggingface_hub` is not installed |
| `tests/models/test_captioner.py` | Yes | `settings.yaml` cannot load, model not downloaded, no CUDA, or less than ~70 GB GPU memory in total (~135 GB in `float32`) |
| `tests/models/test_asr.py` | Yes | `settings.yaml` cannot load, model not downloaded, or its folder not on `PYTHONPATH` |
| `tests/models/test_generator.py` | Yes | `settings.yaml` cannot load, model or tokenizer not downloaded, or `device: cuda` without a GPU |

The real-speech ASR tests use `tests/data/hi.wav` and `tests/data/ta.wav` (short clips you add) and skip without them; the other tests use a generated tone. Each model loads once per test file.
