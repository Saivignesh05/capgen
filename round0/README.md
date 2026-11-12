# Round 0: how to reproduce our results

This is round 0 of our Track 2 system: three frozen models, used as they are, with no training.

| Part | Model | What it does |
|---|---|---|
| ASR | `bodhan-ai/indic-transcribe-core` | finds a clip's language (Hindi, Tamil or English) and writes down what was said |
| Captioner | `Qwen/Qwen3-Omni-30B-A3B-Captioner` | listens to a clip and writes an English description of how it sounds |
| Generator | `ai4bharat/indic-parler-tts` | speaks a transcript in the voice a description asks for |

This README takes you from nothing to our outputs. Every command is meant to be copied and pasted exactly. Each step says what you should see when it worked, and what to do when it did not. For how the code works inside, see `code/EXPLANATION.md`; you do not need it to reproduce anything.

## Contents

- [Before you start](#before-you-start)
- [Step 1: Unpack the code](#step-1-unpack-the-code)
- [Step 2: Choose your folders](#step-2-choose-your-folders)
- [Step 3: Point the settings at your models folder](#step-3-point-the-settings-at-your-models-folder)
- [Step 4: Build the environments and download the models](#step-4-build-the-environments-and-download-the-models)
- [Step 5: Check that you have exactly our model files](#step-5-check-that-you-have-exactly-our-model-files)
- [Step 6: Tell the code where the environments and runs are](#step-6-tell-the-code-where-the-environments-and-runs-are)
- [Step 7: Check that everything works](#step-7-check-that-everything-works)
- [Step 8: Run the pipeline](#step-8-run-the-pipeline)
- [Step 9: Look at the results](#step-9-look-at-the-results)
- [Step 10: Compare with our results](#step-10-compare-with-our-results)
- [Running on a cluster without internet](#running-on-a-cluster-without-internet)
- [Running as a cluster job (SLURM)](#running-as-a-cluster-job-slurm)
- [Keeping long jobs alive with tmux](#keeping-long-jobs-alive-with-tmux)
- [Troubleshooting](#troubleshooting)
- [Starting over and cleaning up](#starting-over-and-cleaning-up)
- [For us: freezing round 0](#for-us-freezing-round-0)

---

## Before you start

You need four things.

**1. A Linux computer (x86_64) with internet**, with at least 120 GB of free disk space (models about 75 GB, environments about 20 GB, plus room for results), and **conda**. Check that conda is installed:

```bash
conda --version
```

You should see something like `conda 25.x.x`. If you see `command not found`, install Miniconda first, following https://docs.conda.io/en/latest/miniconda.html, then close and reopen the terminal.

Check your free disk space where you plan to put the models (here, your home folder):

```bash
df -h ~
```

The `Avail` column must show at least `120G`.

**2. NVIDIA GPUs** to run the models:

- The captioner needs about **70 GB of GPU memory in total**: for example one 80 GB A100 or H100, or two 40 GB A100s. It needs an Ampere or newer GPU (A100, H100, L40, RTX 30xx/40xx); older GPUs such as the V100 do not work.
- The ASR and the generator each need one GPU with about 8 GB.

This can be the same computer as in point 1, or a separate GPU cluster, even one without internet (see [Running on a cluster without internet](#running-on-a-cluster-without-internet)).

Check the GPUs on the GPU computer:

```bash
nvidia-smi
```

You should see a table listing each GPU, its memory (for example `81920MiB`) and, at the top right, `CUDA Version: 12.x`. If you see `command not found`, that computer has no NVIDIA driver installed.

**3. A Hugging Face account.** Two of the models are gated, so you have to accept their terms before you can download them:

1. Create an account at https://huggingface.co/join (free).
2. Open https://huggingface.co/bodhan-ai/indic-transcribe-core, log in, and accept the terms on that page.
3. Open https://huggingface.co/ai4bharat/indic-parler-tts and accept the terms there too.
4. Create an access token: open https://huggingface.co/settings/tokens, click "Create new token", choose the token type "Read", give it any name, click "Create token", and copy the token (it starts with `hf_`). You will paste it in Step 4d. Keep it private.

**4. Your clips.** A folder of `.wav` files, each at most 30 seconds long, for example the devset. They can be in subfolders (like `devset/hi/` and `devset/ta/`), but no two files may have the same name, because a file's name (without `.wav`) is used as its id.

---

## Step 1: Unpack the code

Put `round0.zip` in the folder where you want to work, go to that folder, and unzip it:

```bash
cd /full/path/to/where/round0.zip/is
unzip round0.zip
```

You now have a folder called `round0`:

```
round0/
├── README.md                this file
├── settings.yaml            all settings, for devset runs (1 audio sample per clip)
├── settings_eval.yaml       the same, for evaluation runs (10 audio samples per clip)
├── paths.yaml               where the environments and runs are on this machine (Step 6)
├── checkpoint_hashes.json   fingerprints of the exact model files we used
└── code/                    the code
```

## Step 2: Choose your folders

You need four folder paths:

| Name | What goes there | Size |
|---|---|---|
| `ROUND0` | the `round0` folder from Step 1 | small |
| `MODELS` | the downloaded models | about 75 GB |
| `PACKED` | the environments, packed into two files for copying | about 10 GB |
| `RUNS` | the results of every run | grows with each run |

Run these lines, **replacing each `/full/path/to/...` with a real folder**. Every path must start with `/`. The folders do not have to exist yet; `mkdir` creates them.

```bash
export ROUND0=/full/path/to/round0
export MODELS=/full/path/to/models
export PACKED=/full/path/to/packed-envs
export RUNS=/full/path/to/runs
mkdir -p "$MODELS" "$PACKED" "$RUNS"
```

For example, to keep everything next to `round0`:

```bash
export ROUND0=$HOME/work/round0
export MODELS=$HOME/work/models
export PACKED=$HOME/work/packed-envs
export RUNS=$HOME/work/runs
mkdir -p "$MODELS" "$PACKED" "$RUNS"
```

These `export` lines only last until you close the terminal. **Every time you open a new terminal, run them again** before continuing. To avoid retyping them, save them in a file once:

```bash
cat > ~/round0-folders.sh <<EOF
export ROUND0=$ROUND0
export MODELS=$MODELS
export PACKED=$PACKED
export RUNS=$RUNS
EOF
```

and in every new terminal, run:

```bash
source ~/round0-folders.sh
```

Check that they are right:

```bash
ls "$ROUND0"
```

You should see `README.md  checkpoint_hashes.json  code  paths.yaml  settings.yaml  settings_eval.yaml`. If you see `No such file or directory`, `ROUND0` is wrong: fix it and run the `export` line again.

## Step 3: Point the settings at your models folder

This writes your models folder into both settings files and changes nothing else:

```bash
cd "$ROUND0"
sed -i "s#^model_dir: .*#model_dir: $MODELS#" settings.yaml settings_eval.yaml
grep "^model_dir" settings.yaml settings_eval.yaml
```

You should see your models folder printed twice, for example:

```
settings.yaml:model_dir: /home/you/work/models
settings_eval.yaml:model_dir: /home/you/work/models
```

**Do not change any other value in these files**: they are part of what you are reproducing. If you open them in an editor and change something by accident, unzip `round0.zip` again and redo this step.

## Step 4: Build the environments and download the models

This step needs internet and takes a long time: about 70 GB of models, and about 10 GB of Python packages. On a slow connection it can take several hours, so consider running it inside tmux (see [Keeping long jobs alive with tmux](#keeping-long-jobs-alive-with-tmux)).

It builds two separate Python environments, because the captioner and the generator need different versions of the same library:

- `asr-captioner`: for the ASR, the captioner, the downloads and most tests
- `generator`: for the generator

### 4a. Give conda and Hugging Face more time on slow connections

```bash
conda config --set remote_read_timeout_secs 300
conda config --set remote_max_retries 10
```

This prints nothing. It is saved in your conda settings, so you only do it once.

### 4b. Install conda-pack

```bash
conda activate base
pip install conda-pack
conda-pack --version
```

The last line should print a version, like `conda-pack 0.8.1`.

### 4c. Check your GPU's CUDA version

On the computer with the GPUs, run `nvidia-smi` and read `CUDA Version` at the top right. It must be **12.1 or higher**: the environments use the CUDA 12.1 build of PyTorch. If it is lower, the NVIDIA driver is too old; ask whoever manages the computer to update it.

### 4d. Run the setup script

```bash
cd "$ROUND0/code"
bash scripts/setup_envs.sh ../settings.yaml "$PACKED" https://download.pytorch.org/whl/cu121
```

It prints a lot. The lines starting with `==` tell you where it is:

```
== Creating environment asr-captioner
== No lock file yet: installing from requirements.txt      (or: Installing the exact versions in ...)
```

**The first time, it stops after building the first environment**, with:

```
Not logged in to Hugging Face. Run this, then run this script again:
    conda run --no-capture-output -n asr-captioner hf auth login
```

That is expected. Log in:

```bash
conda run --no-capture-output -n asr-captioner hf auth login
```

1. When it asks for your token, paste the token from "Before you start" and press Enter. The token does not appear on screen as you paste; that is normal.
2. If it asks whether to add the token as a git credential, type `n` and press Enter.
3. It should end with a message that the login was successful.

Then run the setup script again:

```bash
bash scripts/setup_envs.sh ../settings.yaml "$PACKED" https://download.pytorch.org/whl/cu121
```

This time it goes on to download the models. You see progress bars like:

```
Downloading: Qwen/Qwen3-Omni-30B-A3B-Captioner at a2bd106... -> /home/you/work/models/Qwen/Qwen3-Omni-30B-A3B-Captioner
model-00003-of-00016.safetensors:  45%|████▌     | 1.80G/4.00G [02:10<02:40, 13.7MB/s]
```

Lines like `Read timed out` followed by `Retrying in 2s [Retry 2/5]` are normal on a slow connection: it retries by itself.

Then it builds the generator environment. Near the end of that, pip prints this, starting with the word `ERROR`:

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. ...
descript-audiotools 0.7.4 requires protobuf<5.0.0, but you have protobuf 5.29.6 which is incompatible.
```

**This is expected and harmless**, and the script carries on. (One package declares an outdated limit on `protobuf` that it does not actually need; `code/environments/generator/requirements-overrides.txt` explains.)

When everything is done, it ends with:

```
== Done. Copy ... and the models folder to the cluster, then run scripts/unpack_envs.sh there.
```

**If it stops with any error, run the same command again.** Everything already finished is skipped: built environments are reused, downloaded model files are kept, and half-downloaded files continue where they stopped. If the same error keeps coming back, see [Troubleshooting](#troubleshooting).

Check that the models are all there:

```bash
du -sh "$MODELS"/*/*
```

You should see four folders, roughly:

```
4.6G   .../bodhan-ai/indic-transcribe-core
3.8G   .../ai4bharat/indic-parler-tts
3.2M   .../google/flan-t5-large
66G    .../Qwen/Qwen3-Omni-30B-A3B-Captioner
```

## Step 5: Check that you have exactly our model files

`checkpoint_hashes.json` holds a fingerprint (a SHA-256 hash) of every model file we used. This step computes the same fingerprints for your files and compares them, so you know you are running exactly the same models.

```bash
cd "$ROUND0/code"
conda run --no-capture-output -n asr-captioner python -m scripts.hash_checkpoints --settings ../settings.yaml --check ../checkpoint_hashes.json
```

It reads every model file (about 75 GB), so it takes several minutes. It prints one line per model:

```
Hashing captioner: /home/you/work/models/Qwen/Qwen3-Omni-30B-A3B-Captioner
Hashing asr: ...
Hashing generator: ...
Hashing description_tokenizer: ...
All checkpoints match.
```

The last line must be `All checkpoints match.`

If it prints `MISMATCH: <name>` instead, that model's files are damaged or incomplete:

1. Delete that model's folder. The folder for each name is:

   | Name | Folder |
   |---|---|
   | `captioner` | `$MODELS/Qwen/Qwen3-Omni-30B-A3B-Captioner` |
   | `asr` | `$MODELS/bodhan-ai/indic-transcribe-core` |
   | `generator` | `$MODELS/ai4bharat/indic-parler-tts` |
   | `description_tokenizer` | `$MODELS/google/flan-t5-large` |

   For example: `rm -rf "$MODELS/ai4bharat/indic-parler-tts"`
2. Run Step 4d again, which downloads it again.
3. Repeat this step.

## Step 6: Tell the code where the environments and runs are

`paths.yaml` holds two folders on this machine, so the commands below do not need them typed out every time. Fill it in:

```bash
cd "$ROUND0"
sed -i "s#^environments: .*#environments: $(conda info --base)/envs#; s#^runs: .*#runs: $RUNS#" paths.yaml
grep -v "^#" paths.yaml
```

You should see two lines with full paths, for example:

```
environments: /home/you/miniconda3/envs
runs: /home/you/work/runs
```

`paths.yaml` is not part of the results, so it is fine that it differs between machines. You only do this once per machine.

## Step 7: Check that everything works

The tests load each model and check that it gives sensible output. Run them on the computer with the GPUs:

```bash
cd "$ROUND0"
code/test.sh
```

It runs the tests in two parts, one per environment. Each part prints dots while it runs, then a summary line like:

```
================ 48 passed, 2 skipped in 312.41s (0:05:12) ================
```

At the very end you see `All tests passed`. There must be **no `failed`** and no `error`. The captioner test loads a 30-billion-parameter model, so the first part takes several minutes.

Above each summary line is a list of skipped tests, with their reasons, like:

```
SKIPPED [2] tests/helpers.py:49: no speech clip at .../tests/data/hi.wav (see EXPLANATION.md: Tests)
```

Skips with the reason `no speech clip` are fine: those tests need real speech clips you can add yourself (put one short Hindi clip at `code/tests/data/hi.wav` and one Tamil clip at `code/tests/data/ta.wav`). Any other skip means something is missing on this computer:

| Skip reason | What it means |
|---|---|
| `no CUDA GPU on this computer` | you are not on a GPU computer, or the GPUs are not visible to this shell |
| `needs 70 GB of GPU memory, this computer has 40 GB` | not enough GPU memory for the captioner; ask for more GPUs |
| `cannot load .../settings.yaml` | `model_dir` is wrong; redo Step 3 |
| `... not downloaded (run scripts/download.py)` | a model is missing; redo Step 4d |

## Step 8: Run the pipeline

Start a run on your clips folder:

```bash
cd "$ROUND0"
code/run.sh new --stages transcribe,describe,generate --settings settings.yaml --clips /full/path/to/devset
```

Replace `/full/path/to/devset` with your clips folder. `code/run.sh` reads the environments and runs folders from `paths.yaml` (Step 6). For every clip, the run:

1. **transcribes** it: finds its language and writes down what was said (ASR),
2. **describes** it: writes how it sounds (captioner),
3. **generates** audio: speaks what was said in the voice of the description (generator).

It does all clips through step 1, then all through step 2, then all through step 3. While it runs you see:

```
== New run run-20261112-093000: 60 clips
== transcribe: 60 to do
[transcribe 1/60] clip_001: hi
...
== describe: 60 to do
[describe 1/60] clip_001
...
== generate: 60 to do
[generate 1/60] clip_001 #0
...
== Complete for transcribe, describe, generate: {'transcripts': '60/60', 'descriptions': '60/60', 'audio_samples': '60/60', 'no_speech_clips': 0}
```

`Complete` means everything was produced. The run's results are in a new folder inside your runs folder, named after the time it started, for example `$RUNS/run-20261112-093000/`. Every run gets its own folder; earlier runs are never changed.

`no_speech_clips` counts clips in which the ASR heard no words (for example silence or music). They get a description but no generated audio, since there is nothing to say.

### Running only some stages

`--stages` says which stages to run: any of `transcribe`, `describe` and `generate`, separated by commas (no spaces). For example, to only transcribe every clip:

```bash
cd "$ROUND0"
code/run.sh new --stages transcribe --settings settings.yaml --clips /full/path/to/devset
```

It finds every `.wav` file in the folder and its subfolders, transcribes each one, and ends with:

```
== Complete for transcribe: {'transcripts': '60/60', 'descriptions': '0/60', 'audio_samples': '0/60', 'no_speech_clips': 0}
```

The transcripts are in `transcripts.jsonl` in the new run's folder. To do the other stages later on the same run, resume it with them:

```bash
code/run.sh resume --stages describe,generate --run latest
```

Stages always run in the order transcribe, describe, generate, whatever order you list them in. `generate` only speaks clips that already have both a transcript and a description.

### If it stops, or prints `Incomplete`

A run can stop part-way (for example, the computer was switched off) or finish with:

```
== Incomplete (see errors.jsonl, then resume) for transcribe, describe, generate: {'transcripts': '59/60', ...}
```

First, find your latest run's folder and look at what went wrong:

```bash
export RUN="$RUNS/$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['run_id'])" "$RUNS/latest.json")"
echo "$RUN"
cat "$RUN/errors.jsonl"
```

Each line of `errors.jsonl` names the stage, the clip and the error, for example:

```
{"stage":"transcribe","clip_id":"clip_042","sample":null,"error":"ValueError('/data/devset/hi/clip_042.wav is 34.2s, max_clip_seconds is 30.0s')"}
```

Fix the cause if you can (here, the clip is longer than 30 seconds: replace it with a shorter version under the same name). Then continue the same run:

```bash
cd "$ROUND0"
code/run.sh resume --stages transcribe,describe,generate --run latest
```

Resuming only does what is still missing: clips without a transcript or description, and audio not yet generated. Finished work is never redone. You can resume as many times as you need. To resume an older run instead of the latest, give its name: `--run run-20261112-093000`.

If it says `Cannot resume ... has changed since it started`, the code, the settings or the environments changed after the run began. A run never mixes two setups, so start a new run with the `new` command instead.

### For the evaluation outputs (10 samples per clip)

`settings.yaml` generates 1 audio sample per clip, which is enough to check the pipeline. The competition asks for 10 samples for every evaluation item, so for those, start a run with `settings_eval.yaml` instead:

```bash
cd "$ROUND0"
code/run.sh new --stages transcribe,describe,generate --settings settings_eval.yaml --clips /full/path/to/clips
```

The two files differ only in this number. Each sample's random seed depends only on the clip and the sample number, so sample `0.wav` is identical in both kinds of run.

## Step 9: Look at the results

Set `RUN` to your run's folder (as in Step 8), then list what is there:

```bash
ls "$RUN"
```

```
audio  audio.jsonl  clips.jsonl  descriptions.jsonl  errors.jsonl  run.json  settings.yaml  transcripts.jsonl
```

| File | What it holds |
|---|---|
| `clips.jsonl` | the input clips: name, path, and fingerprint |
| `transcripts.jsonl` | what was said in each clip, and its language |
| `descriptions.jsonl` | the description of each clip (the competition's describe format) |
| `audio/<clip name>/0.wav` | the generated speech for each clip (`0.wav` to `9.wav` with `settings_eval.yaml`) |
| `audio.jsonl` | every generated file with the random seed it used |
| `errors.jsonl` | anything that failed (empty when nothing failed) |
| `settings.yaml` | a copy of the settings this run used |
| `run.json` | what the run used: code version, settings, models, environments, GPUs, and every attempt |

The `.jsonl` files have one entry per line. To read the transcripts and descriptions side by side:

```bash
python3 - "$RUN" <<'EOF'
import json, sys
run = sys.argv[1]
transcripts = {r["clip_id"]: r for r in map(json.loads, open(f"{run}/transcripts.jsonl"))}
for line in open(f"{run}/descriptions.jsonl"):
    d = json.loads(line)
    t = transcripts.get(d["clip_id"], {})
    print(f"== {d['clip_id']} ({t.get('language', '?')})")
    print(f"   said:   {t.get('transcript', '(no transcript)')}")
    print(f"   sounds: {d['description']}")
EOF
```

To listen to a generated file, open it in any audio player, or from the terminal:

```bash
aplay "$RUN/audio/<clip name>/0.wav"      # or: ffplay, vlc, or your file manager
```

To see how the run went over time (every attempt, with how far it got):

```bash
python3 -m json.tool "$RUN/run.json"
```

## Step 10: Compare with our results

Compare your `transcripts.jsonl`, `descriptions.jsonl` and audio with ours, clip by clip (every line starts with the clip's name).

- **Same GPU model as ours:** they should match exactly. The ASR and captioner have no randomness, and every generated sample uses a fixed seed.
- **Different GPU model or driver:** a few outputs can differ slightly, because different GPUs round numbers slightly differently, and one changed word in a description changes the rest of it. Compare the `gpus` and `environment_locks` in your `run.json` with ours to see what differs.

---

## Running on a cluster without internet

If your GPUs are on a cluster without internet, do Steps 1 to 5 on a computer **with** internet (it can have no GPU), then:

1. **Copy three folders to the cluster**: the `round0` folder, your models folder and your packed-environments folder. For example, from the computer with internet:
   ```bash
   rsync -av --progress "$ROUND0" "$MODELS" "$PACKED" yourname@cluster.example.org:/full/path/on/cluster/
   ```
   Replace `yourname@cluster.example.org` with your login and the cluster's address, and `/full/path/on/cluster/` with a folder on the cluster with enough space (most clusters have a large "scratch" or "work" area; ask your administrators). If the copy is interrupted, run the same `rsync` command again: it continues.
2. **Log in to the cluster** and set the folder paths **as they are on the cluster**, plus one more for the unpacked environments:
   ```bash
   export ROUND0=/full/path/on/cluster/round0
   export MODELS=/full/path/on/cluster/models
   export PACKED=/full/path/on/cluster/packed-envs
   export RUNS=/full/path/on/cluster/runs
   export ENVS=/full/path/on/cluster/envs
   mkdir -p "$RUNS" "$ENVS"
   ```
3. **Point the settings at the models folder on the cluster** (Step 3 again):
   ```bash
   cd "$ROUND0"
   sed -i "s#^model_dir: .*#model_dir: $MODELS#" settings.yaml settings_eval.yaml
   grep "^model_dir" settings.yaml settings_eval.yaml
   ```
4. **Unpack the environments**:
   ```bash
   cd "$ROUND0/code"
   bash scripts/unpack_envs.sh "$PACKED" "$ENVS"
   ```
   It prints `== Unpacking asr-captioner into ...`, the same for `generator`, and ends with `== Done. Activate one with: ...`.
5. **Check the model files again**, because copying can damage them:
   ```bash
   "$ENVS/asr-captioner/bin/python" -m scripts.hash_checkpoints --settings ../settings.yaml --check ../checkpoint_hashes.json
   ```
   It must end with `All checkpoints match.`
6. **Tell the code where the environments and runs are on the cluster** (Step 6, but with the unpacked folder instead of conda's):
   ```bash
   cd "$ROUND0"
   sed -i "s#^environments: .*#environments: $ENVS#; s#^runs: .*#runs: $RUNS#" paths.yaml
   grep -v "^#" paths.yaml
   ```
7. **Continue with Steps 7 to 10 on a GPU node.** On most clusters you get a GPU node by submitting a job; see the next section.

## Running as a cluster job (SLURM)

Most GPU clusters run work as jobs through a scheduler, often SLURM. Save this as `round0-job.sh` in your home folder, and change the `#SBATCH` lines and the paths for your cluster (its documentation says which partitions and GPU types exist):

```bash
#!/bin/bash
#SBATCH --job-name=round0
#SBATCH --gres=gpu:1               # GPUs: enough for ~70 GB in total (e.g. gpu:1 on 80 GB GPUs, gpu:2 on 40 GB)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=round0-%j.log     # the log goes here; %j is the job number

set -e
ROUND0=/full/path/on/cluster/round0
CLIPS=/full/path/on/cluster/devset

cd "$ROUND0"
nvidia-smi
if [ "$1" = "new" ]; then
    code/run.sh new --stages transcribe,describe,generate --settings settings.yaml --clips "$CLIPS"
else
    code/run.sh resume --stages transcribe,describe,generate --run latest
fi
```

Start a new run as a job:

```bash
sbatch ~/round0-job.sh new
```

It prints `Submitted batch job 123456`. Check on it, and follow its log:

```bash
squeue -u $USER
tail -f round0-123456.log        # press Ctrl+C to stop following
```

If the job ends before the run is complete (for example, it hit its time limit), submit it again without `new`, which resumes the latest run:

```bash
sbatch ~/round0-job.sh
```

The same job script runs the tests too: replace the `if ... fi` block with `code/test.sh`.

## Keeping long jobs alive with tmux

Downloading the models and running the pipeline take hours. If you run them in a normal terminal, they stop when you close it or lose the connection. `tmux` keeps them running:

```bash
tmux new -s round0
```

You are now inside tmux. Run `source ~/round0-folders.sh` (Step 2) and then your long command. To leave it running, press `Ctrl+B`, let go, then press `D`. You can now close the terminal or log out.

To come back to it later:

```bash
tmux attach -t round0
```

When the command has finished, type `exit` to close tmux. If `tmux` is not installed, install it with your system's package manager (for example `sudo dnf install tmux` or `sudo apt install tmux`).

---

## Troubleshooting

### During setup (Step 4)

| You see | What to do |
|---|---|
| `Read timed out`, `connection broken`, `Retrying in 2s` | Your network is slow. A few retries are normal. If the script stops, run the same command again: finished work is kept. If it keeps failing, try another network. |
| `Not logged in to Hugging Face` | Expected the first time. Log in as in Step 4d and run the script again. |
| `401`, `403`, `gated`, `Access to model ... is restricted` | You have not accepted the terms of one of the two gated models, or you logged in with a different account. Accept them (see "Before you start") and run the script again. |
| `model_dir must be an absolute path` | The path in the settings does not start with `/`. Redo Step 3 with a full path. |
| `No such directory` | The models folder does not exist, or the path in the settings is wrong. Check with `grep "^model_dir" "$ROUND0/settings.yaml"`, create the folder with `mkdir -p`, or redo Step 3. |
| `conda-pack not found` | Do Step 4b. If it is installed, run `conda activate base` first. |
| `ERROR: pip's dependency resolver does not currently take into account ...` together with `descript-audiotools ... protobuf` | Expected and harmless; the script carries on. |
| `ResolutionImpossible` or `conflict` (without the line above) | A real package version clash. Do not change any version numbers; send the full message to us. |
| `No space left on device` | The disk is full. Free space, or move the models folder to a bigger disk (Step 2 and Step 3), then run the script again. |
| `already holds ... with other settings. Delete it to download again.` | A model folder holds a different version than the settings ask for. Delete that folder (Step 5 lists them) and run the script again. |

### During checking and running (Steps 5 to 8)

| You see | What to do |
|---|---|
| `MISMATCH: <name>` | That model's files are damaged or incomplete. Follow Step 5. |
| `CUDA out of memory` | Not enough GPU memory. The captioner needs about 70 GB in total; use more or bigger GPUs, then resume the run. |
| `no CUDA GPU on this computer`, or `torch.cuda.is_available()` is `False` | You are not on a GPU computer, or (on a cluster) your job did not request GPUs. |
| `Set 'environments' in .../paths.yaml to an absolute path` | Do Step 6. |
| `No asr-captioner environment in ...` | `environments` in `paths.yaml` is wrong. Check it with `ls` (it must contain `asr-captioner` and `generator`), then fix it (Step 6). |
| `no .wav files under ...` | The clips folder is wrong, or the files do not end in `.wav`. |
| `two clips share the id ...` | Two files in different subfolders have the same name. Rename one. |
| `is 34.2s, max_clip_seconds is 30.0s` in `errors.jsonl` | That clip is too long. Replace it with a clip of at most 30 seconds under the same name, and resume the run. |
| `Cannot resume ... has changed since it started` | The code, settings or environments changed since the run began. Start a new run. |

If something fails that is not listed here, send us the last 50 lines of the output, the run's `errors.jsonl` and `run.json`.

---

## Starting over and cleaning up

| To | Run |
|---|---|
| Delete one run | `rm -rf "$RUNS/run-20261112-093000"` (if it was the latest, also delete `"$RUNS/latest.json"` before using `--run latest`) |
| Re-download one model | delete its folder (see Step 5), then Step 4d |
| Rebuild one environment | `conda env remove -n generator` (or `asr-captioner`), then Step 4d |
| Free pip's download cache (several GB) | `conda run -n asr-captioner pip cache purge` |
| Remove everything | `conda env remove -n asr-captioner`, `conda env remove -n generator`, then delete the `MODELS`, `PACKED` and `RUNS` folders |

---

## For us: freezing round 0

Done once by us before submitting; you do not need this to reproduce.

1. The first run of `setup_envs.sh` wrote `code/environments/asr-captioner/requirements.lock.txt` and `code/environments/generator/requirements.lock.txt`. Commit both: they make everyone install exactly the same package versions.
2. After downloading, write the model fingerprints, check them, and commit `checkpoint_hashes.json`:
   ```bash
   cd "$ROUND0/code"
   conda run --no-capture-output -n asr-captioner python -m scripts.hash_checkpoints --settings ../settings.yaml --write ../checkpoint_hashes.json
   conda run --no-capture-output -n asr-captioner python -m scripts.hash_checkpoints --settings ../settings.yaml --check ../checkpoint_hashes.json
   ```
   The second command must print `All checkpoints match.` Each model's top-level `sha256` in this file is the checkpoint hash for `freeze.txt`.
3. Record the code version with `git rev-parse HEAD` for `freeze.txt`, together with the generation command (Step 8) and `generation.seed` from the settings.
