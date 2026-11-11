# Round 0: how to reproduce our results

This is round 0 of our Track 2 system: three frozen models, used as they are, with no training.

| Part | Model | What it does |
|---|---|---|
| Captioner | `Qwen/Qwen3-Omni-30B-A3B-Captioner` | listens to a clip and writes an English description of how it sounds |
| ASR | `bodhan-ai/indic-transcribe-core` | finds the clip's language and writes down what was said |
| Generator | `ai4bharat/indic-parler-tts` | speaks a transcript in the voice a description asks for |

Follow the steps below in order. Every command is meant to be copied and pasted exactly. Each step says what you should see when it worked, and what to do when it did not.

For how the code works inside, see `code/EXPLANATION.md`. You do not need it to reproduce the results.

---

## Before you start: what you need

1. **A Linux computer (x86_64) with internet**, at least 120 GB of free disk space (models about 75 GB, environments about 20 GB), and **conda** (Miniconda or Anaconda). Check conda is there:
   ```bash
   conda --version
   ```
   You should see something like `conda 25.x.x`. If you see `command not found`, install Miniconda first: https://docs.conda.io/en/latest/miniconda.html

2. **NVIDIA GPUs** to run the models:
   - The captioner needs about **70 GB of GPU memory in total**: for example one 80 GB A100 or H100, or two 40 GB A100s. Older GPUs such as the V100 do not work.
   - The generator needs one GPU with about 8 GB.

   This can be the same computer as in point 1, or a separate GPU cluster (even one without internet; see "Running on a cluster without internet" at the end).

3. **A Hugging Face account.** Two of the models are gated, so you have to accept their terms before you can download them:
   1. Create an account at https://huggingface.co/join (free).
   2. Open https://huggingface.co/bodhan-ai/indic-transcribe-core, log in, and accept the terms on that page.
   3. Open https://huggingface.co/ai4bharat/indic-parler-tts and accept the terms there too.
   4. Create an access token: open https://huggingface.co/settings/tokens, click "Create new token", choose "Read", and copy the token. You will paste it in Step 4.

---

## Step 1: Unpack the code

Put `round0.zip` where you want to work, then unzip it:

```bash
unzip round0.zip
```

You now have a folder called `round0`. Inside it:

```
round0/
├── README.md                this file
├── settings.yaml            all settings, for devset runs (1 audio sample per clip)
├── settings_eval.yaml       the same, for evaluation runs (10 audio samples per description)
├── checkpoint_hashes.json   fingerprints of the exact model files we used
└── code/                    the code
```

## Step 2: Choose your folders

You need three folders: where the models go, where the packed environments go, and where results go. Run these lines, **replacing each `/full/path/to/...` with a real folder on your disk** (they do not have to exist yet). The models folder needs about 75 GB of space.

```bash
export ROUND0=/full/path/to/round0
export MODELS=/full/path/to/models
export PACKED=/full/path/to/packed-envs
export RUNS=/full/path/to/runs
mkdir -p "$MODELS" "$PACKED" "$RUNS"
```

These `export` lines only last until you close the terminal. **Every time you open a new terminal, run them again** before continuing.

Check them:

```bash
ls "$ROUND0"
```

You should see `README.md  checkpoint_hashes.json  code  settings.yaml  settings_eval.yaml`. If you see `No such file or directory`, `ROUND0` is wrong: fix it and run the `export` line again.

## Step 3: Tell the settings where the models are

This writes your models folder into both settings files. It changes nothing else.

```bash
cd "$ROUND0"
sed -i "s#^model_dir: .*#model_dir: $MODELS#" settings.yaml settings_eval.yaml
grep "^model_dir" settings.yaml settings_eval.yaml
```

You should see your models folder printed twice. **Do not change any other value in these files**: they are part of what you are reproducing.

## Step 4: Build the environments and download the models

This step needs internet and takes a long time (the models alone are about 70 GB).

### 4a. Give conda more time on slow connections

```bash
conda config --set remote_read_timeout_secs 300
conda config --set remote_max_retries 10
```

### 4b. Install conda-pack

```bash
conda activate base
pip install conda-pack
conda-pack --version
```

You should see a version number, like `conda-pack 0.8.x`.

### 4c. Check your GPU's CUDA version

On the computer with the GPUs, run:

```bash
nvidia-smi
```

At the top right you see `CUDA Version: 12.x` (or higher). It must be **12.1 or higher**: the environments use the CUDA 12.1 build of PyTorch. If it is lower, ask whoever manages the computer to update the NVIDIA driver.

### 4d. Run the setup script

```bash
cd "$ROUND0/code"
bash scripts/setup_envs.sh ../settings.yaml "$PACKED" https://download.pytorch.org/whl/cu121
```

The first time, it builds the first environment and then stops with:

```
Not logged in to Hugging Face. Run this, then run this script again:
    conda run --no-capture-output -n asr-captioner hf auth login
```

That is expected. Log in with the token from "Before you start":

```bash
conda run --no-capture-output -n asr-captioner hf auth login
```

Paste the token when it asks (the token does not appear on screen as you paste; that is normal) and press Enter. Answer `n` if it asks about git credentials. Then run the setup script again:

```bash
bash scripts/setup_envs.sh ../settings.yaml "$PACKED" https://download.pytorch.org/whl/cu121
```

It finishes with:

```
== Done. Copy ... and the models folder to the cluster, then run scripts/unpack_envs.sh there.
```

**If it stops with an error, run the same command again.** Finished work is skipped, so it continues where it stopped. If the same error keeps coming back:

| Error mentions | What to do |
|---|---|
| `Read timed out`, `connection broken`, `Retrying in 2s` | Your network is slow. A few retries are normal; if the script stops, run the same command again: downloaded files are kept and half-finished ones continue. If it keeps failing, try another network. |
| `401`, `403`, `gated`, `access` | You have not accepted the terms of one of the two gated models (see "Before you start"), or logged in with the wrong account. |
| `No such directory` | `model_dir` in the settings is wrong. Redo Step 3. |
| `ResolutionImpossible`, `conflict` | A package version clash. Do not change version numbers; send us the full message. |

While the generator environment builds, pip prints this, starting with the word `ERROR`:

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. ...
descript-audiotools 0.7.4 requires protobuf<5.0.0, but you have protobuf 5.29.6 which is incompatible.
```

**This is expected and harmless**, and the script carries on. (One package declares an outdated limit on `protobuf` that it does not actually need; `code/environments/generator/requirements-overrides.txt` explains.)

## Step 5: Check that you have exactly our model files

```bash
cd "$ROUND0/code"
conda run -n asr-captioner python -m scripts.hash_checkpoints --settings ../settings.yaml --check ../checkpoint_hashes.json
```

This reads every model file, so it takes several minutes. It must end with:

```
All checkpoints match.
```

If it prints `MISMATCH: <name>`, that model's files are damaged or incomplete. Delete that model's folder inside your models folder (for example `$MODELS/ai4bharat/indic-parler-tts` for `generator`), run Step 4d again to download it again, then repeat this step.

## Step 6: Check that everything works

Run this on the computer with the GPUs:

```bash
cd "$ROUND0/code"
export ENVS="$(conda info --base)/envs"
PYTHONPATH="$MODELS/bodhan-ai/indic-transcribe-core" "$ENVS/asr-captioner/bin/python" -m pytest -rs --ignore=tests/models/test_generator.py
"$ENVS/generator/bin/python" -m pytest -rs tests/models/test_generator.py
```

Each of the two commands ends with a line like `== 40 passed, 2 skipped in 312.4s ==`. There must be **no `failed`**. The test of the captioner loads a 30B model, so the first command takes several minutes.

`skipped` is fine for the tests that need real speech clips (`no speech clip at .../tests/data/hi.wav`). For any other skip, read its reason in the list printed above that last line: it says what is missing (for example `no CUDA GPU on this computer`).

## Step 7: Run the pipeline

Put the clips you want to process in one folder (for example the devset). The clips must be `.wav` files of at most 30 seconds. They can be in subfolders, but no two files may have the same name.

```bash
cd "$ROUND0/code"
export ENVS="$(conda info --base)/envs"
"$ENVS/asr-captioner/bin/python" -m scripts.run new --settings ../settings.yaml --clips /full/path/to/devset --runs "$RUNS" --environments "$ENVS"
```

Replace `/full/path/to/devset` with your clips folder. For every clip, this:

1. finds its language and writes down what was said (ASR),
2. writes a description of how it sounds (captioner),
3. speaks what was said in the voice of the description (generator).

When it is done, it prints:

```
== Complete: {'transcripts': '60/60', 'descriptions': '60/60', 'audio_samples': '60/60', 'no_speech_clips': 0}
```

The results are in a new folder inside your runs folder, named after the time you started, like `$RUNS/run-20261112-093000/`:

| File | What it holds |
|---|---|
| `transcripts.jsonl` | what was said in each clip, and its language |
| `descriptions.jsonl` | the description of each clip |
| `audio/<clip name>/0.wav` | the generated speech for each clip |
| `audio.jsonl` | every generated file with the random seed it used |
| `run.json` | exactly what was used: code version, settings, models, GPUs |
| `errors.jsonl` | anything that failed (empty when nothing failed) |

To see where your latest run is:

```bash
cat "$RUNS/latest.json"
```

### If it stops or prints `Incomplete`

Look at what went wrong:

```bash
cat "$RUNS/$(python3 -c "import json; print(json.load(open('$RUNS/latest.json'))['run_id'])")/errors.jsonl"
```

Fix the cause if you can (for example a clip longer than 30 seconds), then continue the same run. This only does what is still missing:

```bash
cd "$ROUND0/code"
"$ENVS/asr-captioner/bin/python" -m scripts.run resume --run latest --runs "$RUNS" --environments "$ENVS"
```

If it says `Cannot resume ... has changed since it started`, something in the code or settings changed since the run began. Start a new run with the `new` command above instead.

### For the evaluation outputs

The competition asks for 10 generated samples per evaluation description. Use the same command with `settings_eval.yaml` instead of `settings.yaml`:

```bash
"$ENVS/asr-captioner/bin/python" -m scripts.run new --settings ../settings_eval.yaml --clips /full/path/to/clips --runs "$RUNS" --environments "$ENVS"
```

## Step 8: Compare with our results

Compare your `descriptions.jsonl`, `transcripts.jsonl` and audio with ours.

- **Same GPU type as ours:** they should match exactly. The captioner and ASR have no randomness, and every generated sample uses a fixed seed.
- **Different GPU type or driver:** a few outputs can differ slightly, because GPUs do arithmetic in slightly different ways. Compare the `gpus` and `environment_locks` in your `run.json` with ours to see what differs.

---

## Running on a cluster without internet

If your GPUs are on a cluster without internet, do Steps 1 to 5 on a computer **with** internet, then:

1. Copy three folders to the cluster: the `round0` folder, your models folder, and your packed-environments folder. For example:
   ```bash
   rsync -av --progress "$ROUND0" "$MODELS" "$PACKED" yourname@cluster:/full/path/on/cluster/
   ```
2. Log in to the cluster. Run Step 2 there, with the folder paths **on the cluster**, and also choose where the environments will live:
   ```bash
   export ENVS=/full/path/on/cluster/envs
   ```
3. Run Step 3 there, so the settings point to the models folder on the cluster.
4. Unpack the environments:
   ```bash
   cd "$ROUND0/code"
   bash scripts/unpack_envs.sh "$PACKED" "$ENVS"
   ```
   It ends with `== Done. Activate one with: ...`.
5. Check the model files again, because copying can damage them:
   ```bash
   "$ENVS/asr-captioner/bin/python" -m scripts.hash_checkpoints --settings ../settings.yaml --check ../checkpoint_hashes.json
   ```
   It must end with `All checkpoints match.`
6. Continue with Steps 6 to 8 on a GPU node, **without** the `export ENVS="$(conda info --base)/envs"` lines: your `ENVS` is the folder you unpacked into.

---

## For us: freezing round 0

Done once by us before submitting; you do not need this to reproduce.

1. The first run of `setup_envs.sh` wrote `code/environments/asr-captioner/requirements.lock.txt` and `code/environments/generator/requirements.lock.txt`. Commit both: they make everyone install exactly the same package versions.
2. After downloading, write the model fingerprints and commit the file:
   ```bash
   cd "$ROUND0/code"
   conda run -n asr-captioner python -m scripts.hash_checkpoints --settings ../settings.yaml --write ../checkpoint_hashes.json
   ```
