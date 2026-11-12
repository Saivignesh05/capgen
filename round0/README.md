# Round 0

Round 0 of our Track 2 system: three frozen open models, used as they are, with no training.

| Part | Model (pinned to an exact commit) | What it does |
|---|---|---|
| ASR | `bodhan-ai/indic-transcribe-core` | finds a clip's language (Hindi, Tamil or English) and writes down what was said |
| Captioner | `Qwen/Qwen3-Omni-30B-A3B-Captioner` | writes an English description of how the clip sounds |
| Generator | `ai4bharat/indic-parler-tts` | speaks the transcript in the voice the description asks for |

**Hardware we used:** one node of an HPC cluster with <!-- FILL IN: e.g. 1 × NVIDIA A100 80 GB -->, Linux x86_64, NVIDIA driver for CUDA 12.1 or newer.

Everything is done with one command, `./round0`, run from this folder. Run `./round0` on its own to list what it can do. For how the code works, see `code/EXPLANATION.md`.

---

## What you need

- **A Linux (x86_64) computer with internet** and [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed, with about **120 GB of free disk space** (models about 75 GB, environments about 20 GB).
- **NVIDIA GPUs** to run on (the same computer, or a cluster; see [No internet on the GPU machine](#no-internet-on-the-gpu-machine)):
  - the captioner needs about **70 GB of GPU memory in total** on Ampere or newer GPUs (for example one 80 GB A100 or H100, or two 40 GB A100s);
  - the ASR and the generator need one GPU with about 8 GB.
- **A Hugging Face account** that has accepted the terms of the two gated models, [bodhan-ai/indic-transcribe-core](https://huggingface.co/bodhan-ai/indic-transcribe-core) and [ai4bharat/indic-parler-tts](https://huggingface.co/ai4bharat/indic-parler-tts) (open each page, log in, and accept). You also need a **read access token**: create one at https://huggingface.co/settings/tokens.

## Setup

**1. Say where things go.** Make your own copy of the example paths file:

```bash
cp paths.example.yaml paths.yaml
```

Open `paths.yaml` in a text editor and replace the three example folders with real ones. Use full paths that start with `/`; the folders are created for you.

```yaml
models: /home/you/round0-data/models
environments: /home/you/round0-data/environments
runs: /home/you/round0-data/runs
```

**2. Build the environments and download the models.** This needs internet and takes a while (about 75 GB):

```bash
./round0 setup
```

The first time, it asks for your Hugging Face token: paste it and press Enter (it does not show on screen). If the command stops for any reason, for example a dropped connection, run it again: everything already done is kept. Near the end, pip prints a message starting with `ERROR: pip's dependency resolver...` about `descript-audiotools` and `protobuf`; this is expected and harmless. It ends with `Setup done`.

**3. Check that you have exactly our models.** This compares every model file with `checkpoint_hashes.json` and takes a few minutes:

```bash
./round0 check
```

It must end with `All checkpoints match.`

**4. Check that everything works.** On the GPU machine:

```bash
./round0 test
```

It must end with `All tests passed`. Tests that need something this machine does not have are skipped and say why; the two tests that need your own speech clips (`tests/data/hi.wav`, `tests/data/ta.wav`) are optional.

## Running

Put your clips in a folder: `.wav` files of at most 30 seconds, in any subfolders, with no two files of the same name. Then:

```bash
./round0 run all /path/to/clips
```

For every clip this transcribes it, describes it, and speaks the transcript in the described voice, and prints `Complete` at the end. Each run gets its own folder inside the `runs` folder of `paths.yaml`, named after its start time:

| File in the run folder | What it holds |
|---|---|
| `transcripts.jsonl` | per clip: language and what was said |
| `descriptions.jsonl` | per clip: the description (the competition's describe format) |
| `audio/<clip name>/0.wav` | the generated speech |
| `audio.jsonl` | every generated file with its random seed |
| `run.json` | what the run used (code version, settings, models, GPUs) and every attempt |
| `errors.jsonl` | anything that failed |

Other things you can do:

| Command | What it does |
|---|---|
| `./round0 status latest` | how far the latest run got, and its last errors |
| `./round0 resume all latest` | continue the latest run after a stop or failures; only what is missing is done |
| `./round0 run transcribe /path/to/clips` | run only some stages: `transcribe`, `describe`, `generate`, or several separated by commas |
| `./round0 resume describe,generate latest` | add the other stages to that run later |
| `./round0 run-eval all /path/to/clips` | the same, with 10 generated samples per clip, as the competition asks for the evaluation outputs |

Long runs: start them inside `tmux` (or as a cluster job) so they keep going if your connection drops.

## No internet on the GPU machine

Do the setup on a machine with internet, then move everything over:

1. On the machine with internet: `pip install conda-pack`, then `./round0 pack /path/to/packed` (packs both environments into two files).
2. Copy three things to the GPU machine: this `round0` folder, your models folder, and the packed folder.
3. On the GPU machine: make `paths.yaml` there as in Setup step 1 (point `models` at the copied models; choose `environments` and `runs`), then run `./round0 unpack /path/to/packed`, then `./round0 check` and `./round0 test`.

On a cluster with a job scheduler such as SLURM, put the `./round0` command in your job script, after `cd` to this folder.

## If something goes wrong

| You see | What to do |
|---|---|
| `paths.yaml is missing`, `... is still the example value` or `must be a full path` | Do Setup, step 1. |
| `the environments are not in ...` | Run `./round0 setup`, or fix `environments` in `paths.yaml`. |
| `Read timed out`, `Retrying in 2s` | A slow connection. It retries by itself; if setup stops, run it again. |
| `401`, `403`, `gated`, `restricted` | Accept the terms of both gated models with the account you logged in with. |
| `MISMATCH: <name>` | That model is damaged or incomplete: delete its folder inside your models folder and run `./round0 setup` again. |
| `CUDA out of memory` or `no CUDA GPU` | Not enough GPU memory, or no GPU visible: see "What you need". |
| `Incomplete` after a run | Run `./round0 status latest` to see the errors, fix them (for example a clip over 30 seconds), then `./round0 resume all latest`. |
| `Permission denied` for `./round0` | Run `chmod +x round0` once. |

On the same GPU model, our outputs repeat exactly: the ASR and captioner have no randomness and every generated sample uses a fixed seed. Different GPUs can change a few outputs slightly; compare the `gpus` in your `run.json` with ours.

## For the authors: freezing round 0

Before submitting, with the models downloaded:

1. `./round0 freeze` writes `checkpoint_hashes.json` (the SHA-256 of every model checkpoint, for `freeze.txt`).
2. Make sure `code/environments/asr-captioner/requirements.lock.txt` and `code/environments/generator/requirements.lock.txt` exist and are not empty (the first `./round0 setup` writes them).
3. Commit everything, and record `git rev-parse HEAD`, the run command and `generation.seed` from `settings.yaml` in `freeze.txt`.
4. `./round0 package /path/to/round0.zip` makes the submission zip. It leaves out `paths.yaml`, caches and voice recordings, and refuses to write the zip if `checkpoint_hashes.json` or a lock file is missing, or if any file mentions your home folder.
