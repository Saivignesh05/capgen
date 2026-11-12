# How the midterm repo works

This branch (`midterm`) turns the repo into one pipeline. It runs our round 0 system on the devset,
scores what it produced, and writes `results.csv` in the midterm submission format. Everything heavy
runs on the HPC; nothing here is meant to run on a laptop.

**Where to start**

| You want to… | Read |
|---|---|
| understand what the pipeline does and measures | sections 1, 4 |
| run it on an HPC whose login node has internet | section 6, then 8 |
| run it on an HPC without internet | section 7, then 8 |
| fix a failed run | section 9 |
| write the report's "Your choices" part | section 4.2 |
| know what is still missing | section 10 |

---

## 1. What the pipeline does

```
devset/{hi,ta}/clips/*.wav        (30 Hindi + 30 Tamil clips)
        │
        ▼  ROUND 0 (round0/code/scripts/run.py), environments: asr-captioner + generator
   ASR (indic-transcribe) ──► transcript ──┐
   captioner (Qwen3-Omni-30B) ──► description ┴─► generator (Indic Parler-TTS) ──► N audio samples per clip
        │
        ▼  SCORING (midterm/scoring/run.py), environment: scoring
   intelligibility · speaker_diversity · naturalness · caption_generated · judge
        │
        ▼
   results.csv  (+ results_detail.json with 95% confidence intervals)
```

In words: for each devset clip, round 0 writes down what was said, describes how it sounds, and then
speaks the same words again from that description. The scoring stages then measure how good the
descriptions and the new speech are, the way Rime will at evaluation (section 4).

**Two runs instead of one.** Only speaker diversity needs 10 samples per clip, and generating
samples is the slowest step. So by default the devset is run twice and the results are merged:

| Run | Clips | Samples per clip | Generations | Gives |
|---|---|---|---|---|
| A, `main` | all 60 | 1 (`round0/settings.yaml`) | 60 | every measure except speaker diversity |
| B, `diversity` | 10 hi + 10 ta, one per speaker | 10 (`round0/settings_eval.yaml`) | 200 | speaker diversity |

That is about 3 h on one 80 GB GPU, against 5–7 h for 10 samples on all 60 clips (`MODE=full`). The
SLURM job does both runs and the merge (section 8).

## 2. What is where

| Path | What it is |
|---|---|
| `round0/` | The round 0 system (frozen models, no training). `round0/README.md` reproduces it; `round0/code/EXPLANATION.md` explains its code. |
| `round0/settings.yaml`, `round0/settings_eval.yaml` | Round 0 settings: 1 sample per clip, and 10 samples per clip. Otherwise identical. |
| `midterm/slurm/run_midterm.sbatch` | **The job you submit.** Choose what runs with `MODE` (section 8). |
| `midterm/run_midterm.sh` | One run: round 0 on a folder of clips, then scoring. The sbatch file calls it. |
| `midterm/settings.yaml` | Every scoring setting: models and their pinned revisions, thresholds, stages, reference descriptions. |
| `midterm/setup_scoring.sh` | Builds the `scoring` environment and downloads the scoring models. |
| `midterm/scoring/` | The scoring package (section 5). |
| `midterm/environment/` | The `scoring` environment: `environment.yml`, `requirements.txt` (and `requirements.lock.txt` after the first build). |
| `nat.py` | Naturalness scorer (UTMOS, SIGMOS, FAD, PSD), used by the naturalness stage. |
| `score_hi_ta.py`, `normalize_hi_ta.py` | CER/WER for Hindi and Tamil, used by the intelligibility stage. |
| `devset/{hi,ta}/clips/` | The dev clips. The language of a clip is the folder it is in. |
| `devset/{hi,ta}/tool_descriptions.json` | What `features-extractor/` measured on each clip. Second reference for the rubric score. |
| `devset/{hi,ta}/ground_truth_descriptions.jsonl` | Hand-written reference descriptions for the rubric score: 5 Hindi and 5 Tamil clips so far (section 10). |
| `devset_div/` | Created by the job: the 20-clip subset for speaker diversity, with `subset.json` listing it. |
| `features-extractor/` | The acoustic feature extractor that wrote the tool descriptions. Not run by this pipeline. |
| `6621_midterm/test_set.zip` | Our contributed test set. Not used here: it is for the other teams. |

**Folders you choose** (used in every command below; write them down once):

| Name | What goes there | Size |
|---|---|---|
| `$MODELS` | round 0 models (`model_dir` in both round 0 settings files) | ~75 GB |
| `$SCORING_MODELS` | scoring models (`model_dir` in `midterm/settings.yaml`) | ~25 GB |
| `$ENVS` | the three conda environments: `asr-captioner`, `generator`, `scoring` | ~25 GB |
| `$RUNS` | run outputs, one subfolder per run kind (`main`, `diversity`, `full`) | ~1–5 GB |
| `$PACKED` | packed environments, only without internet (section 7) | ~10 GB |

Put them on a large, fast disk (scratch or project space), not your home folder if it has a quota.

## 3. What changed on this branch

- **Captioner fix** (`round0/code/pipeline/models/captioner.py`). With transformers 4.57.6, Qwen3-Omni's
  `generate()` returns a pair `(text, audio)`. The code read `.sequences` from the pair, so every
  clip failed and round 0 produced 0 descriptions. It now unpacks the pair (CAREFUL C10).
- **Checkpoint-hash fix** (`round0/code/scripts/hash_checkpoints.py`). The ASR's code runs from inside
  its model folder, so Python writes `__pycache__` there; `--check` then reported a mismatch after
  any run. `__pycache__` is now skipped (CAREFUL H2).
- **Setup without packing** (`round0/code/scripts/setup_envs.sh`, `midterm/setup_scoring.sh`):
  `PACK=0` builds the environments in place and skips `conda-pack`, for an HPC with internet.
- **`nat.py`**: `utmos_repo` and `xlsr_model` can be local folders, so it works offline. The usage
  example now passes the required `metrics` argument, and the SIGMOS source is corrected
  (microsoft/SIG-Challenge, not DNS-Challenge).
- **New**: `midterm/` (scoring pipeline, SLURM job), the devset ground-truth descriptions, this file.

## 4. The measures

### 4.1 What is in results.csv

`results.csv` has the midterm columns `track,part,language,system,measure,value,n_items`: one row per
number, `hi` and `ta` separately, `track=2`, `system=round_0`. Each value is computed per item (one
devset clip = one description) and then averaged over items, as Rime averages over clips and
descriptions. `n_items` is the number of items behind the value.

`results_detail.json` (next to each run's `results.csv`) has the same numbers with 95% bootstrap
confidence intervals (Rime's Appendix A draws), what each measure means, which direction is better,
why any measure is missing, and `native_level`: the UTMOS and SIGMOS of the real devset clips, the
level real speech gets.

| measure | part | Rime's measure (points) | How we compute it | Better |
|---|---|---|---|---|
| `rubric_score` | captioner | Rubric score (10) | A text LLM (Qwen3-8B) splits the **ground-truth** description into rubric items and grades the round 0 caption, cut at 300 words: (stated − contradicted) / items, floored at 0 | higher |
| `rubric_score_tool` | captioner | (stand-in) | Same, with the **tool descriptions** as the reference: measured facts only (pitch, pauses, SNR, events…) | higher |
| `instruction_relevance` | generator | Instruction relevance (5) | Qwen3-8B splits the instruction into checks; the round 0 captioner describes a generated sample; Qwen3-8B marks each check met / not met / unverifiable. Score = met / checks | higher |
| `intelligibility` | generator | Intelligibility (1) | Character error rate of AI4Bharat IndicConformer (the ASR Rime uses) on each generated sample, against the text the generator was asked to say. Rime's points = 1 − CER | **lower** |
| `speaker_diversity` | generator | Speaker diversity (2) | Distinct speakers among the 10 samples of a description: WavLM-base-plus-sv embeddings, complete-linkage clustering at cosine 0.86 | higher |
| `naturalness_utmos` | generator | (stand-in for CMOS, 2) | UTMOS, 1–5, per sample | higher |
| `naturalness_sigmos_ovrl` | generator | (stand-in) | SIGMOS overall, 1–5, per sample | higher |
| `naturalness_fad` | generator | (stand-in) | Fréchet distance of XLS-R layer-9 embeddings, generated vs native devset clips, per language (`n_items` = samples) | **lower** |
| `naturalness_psd` | generator | (stand-in) | Same with prosody features | **lower** |

### 4.2 Choices where Rime is silent or cannot be copied (for the report)

- **Rubric score.** Rime's rubrics are private and graded by "a fixed grader model". We build rubrics
  from reference descriptions with a text LLM and use equal weights, because Rime's weights (by
  difficulty) are not published. Contradicting a true item is penalised; Rime's separate "false
  items" (invented details) are not modelled. Note that our ground-truth descriptions reuse many
  tool-measured numbers (HNR, jitter, …) that no captioner can state, which lowers `rubric_score`.
- **Instruction relevance.** Rime uses three listeners. We use the captioner as the judge (Rime
  suggests this), so the measure is only as good as the captioner. A property the caption does not
  mention counts as *not met* (conservative); the share of `unverifiable` checks is in `relevance.jsonl`.
  Only `relevance.samples_per_item` samples per item are judged, to save GPU time.
- **Intelligibility.** The reference is the round 0 ASR transcript of the devset clip, because the
  devset has no human transcripts, so that ASR's errors count against the generator. Transcripts
  with Latin letters (code-mixed speech) are skipped and logged, because `score_hi_ta.py` only scores
  native script. We use `cer_ns` (CER with spaces removed, the primary metric of `score_hi_ta.py`).
- **Speaker diversity.** Rime counts only among samples that pass the relevance checks; we count
  among all 10, because relevance is judged on only a few. Rime's speaker model is not named; we use
  WavLM-base-plus-sv at its model-card threshold. It is measured on 20 clips, not 60 (section 1).
- **Naturalness.** Rime's measure is a blind CMOS against other teams' outputs, which cannot be run
  alone. UTMOS, SIGMOS, FAD and PSD are automatic stand-ins. UTMOS was trained on English, so
  compare it with `native_level`, not with published numbers.
- **What the generator is given.** On the devset the generator re-speaks each clip: the text is the
  ASR transcript and the instruction is the round 0 caption. At evaluation, Rime gives text and a
  human-written description instead (`descriptions.csv`); that path is not built yet (section 10).

## 5. The scoring package (`midterm/scoring/`)

| File | Role |
|---|---|
| `run.py` | Orchestrator. Runs each stage in its own process, then `results.py`. Sets the offline environment (`HF_HUB_OFFLINE=1`, pinned caches, `PYTHONPATH`). |
| `stages/intelligibility.py` | IndicConformer → CER per sample → `intelligibility.jsonl` |
| `stages/speaker_diversity.py` | WavLM-SV embeddings → speakers per item → `speaker_diversity.jsonl` |
| `stages/naturalness.py` | `nat.py` → `naturalness.jsonl`, `naturalness_native.jsonl`, `naturalness_corpus.json` |
| `stages/caption_generated.py` | Round 0 captioner on generated samples → `caption_generated.jsonl` (~70 GB GPU memory) |
| `stages/judge.py` | Qwen3-8B → `rubric.jsonl`, `relevance.jsonl`, plus `extractions.jsonl` (cached rubric items) |
| `results.py` | All of the above → `results.csv`, `results_detail.json` |
| `diversity.py` | `subset`: builds `devset_div/`; `merge`: run A's results + run B's speaker diversity → one `results.csv` |
| `check.py` | Smoke test: loads every scoring model and runs it on one clip per language |
| `download.py` | Downloads every scoring model at its pinned revision |
| `models.py`, `llm.py`, `common.py`, `settings.py`, `stage.py` | Shared code: model loaders, judge prompts and parsing, logging, bootstrap, settings |
| `tests/` | Unit tests without models: `python -m pytest scoring/tests` from `midterm/` |

Scoring models (all open, pinned in `midterm/settings.yaml`): Qwen3-8B (judge), IndicConformer-600M
(**gated**: accept its terms on Hugging Face), WavLM-base-plus-sv, wav2vec2-XLS-R-300M, UTMOS
(tarepan/SpeechMOS v1.2.0), SIGMOS (microsoft/SIG-Challenge). The Qwen3-Omni captioner is round 0's own.

---

## 6. Setup on an HPC with internet

Use this when the HPC's login node can reach the internet (`curl -sI https://huggingface.co` prints
`HTTP/2 200`). Everything is built and downloaded on the HPC itself, so nothing is packed or copied.
Done once; about 1–2 h, mostly downloads.

**Where to run each step.** Builds and downloads run on the **login node** (they need internet, not a
GPU). If your HPC limits long jobs on login nodes, run them in an interactive CPU job instead
(`srun --pty bash`) when compute nodes also have internet. The smoke test and the real runs need a
**GPU node**. The pipeline never downloads during a run (it sets `HF_HUB_OFFLINE=1`), so it works
even when GPU nodes have no internet.

**0. Before you start.**
- A Hugging Face account that has accepted the terms of the three gated models:
  [bodhan-ai/indic-transcribe-core](https://huggingface.co/bodhan-ai/indic-transcribe-core),
  [ai4bharat/indic-parler-tts](https://huggingface.co/ai4bharat/indic-parler-tts) and
  [ai4bharat/indic-conformer-600m-multilingual](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual),
  and a "Read" access token from https://huggingface.co/settings/tokens.
- conda on the HPC (often `module load anaconda` or `module load miniconda`; ask your admins).
  `conda --version` must work.
- The CUDA version of the GPU nodes: run `nvidia-smi` on one; the top right must say 12.1 or higher.

**1. Get the code** (login node):
```bash
git clone https://github.com/Saivignesh05/capgen.git
cd capgen && git checkout midterm
export REPO=$(pwd)
```

**2. Choose the folders** (section 2) and write them into the three settings files:
```bash
export MODELS=/scratch/$USER/capgen/models SCORING_MODELS=/scratch/$USER/capgen/scoring-models RUNS=/scratch/$USER/capgen/runs
mkdir -p "$MODELS" "$SCORING_MODELS" "$RUNS"
sed -i "s#^model_dir: .*#model_dir: $MODELS#" round0/settings.yaml round0/settings_eval.yaml
sed -i "s#^model_dir: .*#model_dir: $SCORING_MODELS#" midterm/settings.yaml
grep "^model_dir" round0/settings.yaml round0/settings_eval.yaml midterm/settings.yaml
```
You should see your folders printed three times. **Do not change any other value** in these files.

**3. Round 0 environments and models** (login node). Same as `round0/README.md` Step 4, but with
`PACK=0`, so nothing is packed:
```bash
conda config --set remote_read_timeout_secs 300
conda config --set remote_max_retries 10
cd "$REPO/round0/code"
PACK=0 bash scripts/setup_envs.sh ../settings.yaml - https://download.pytorch.org/whl/cu121
```
The first time it stops with `Not logged in to Hugging Face`: run the `hf auth login` command it
prints, paste your token, then run the `setup_envs.sh` line again. It ends with
`== Done (not packed, PACK=0). The environments are in: <folder>`: that folder is your `$ENVS`.
```bash
export ENVS=<the folder it printed>      # e.g. /home/you/.conda/envs
```

**4. Scoring environment and models** (login node):
```bash
cd "$REPO"
PACK=0 bash midterm/setup_scoring.sh midterm/settings.yaml - https://download.pytorch.org/whl/cu121
```
If it says `Not logged in` (the login of step 3 is normally shared), log in the same way and run it
again. It ends with `done (not packed, PACK=0)`.

**5. Check the model files** (login node, several minutes):
```bash
cd "$REPO/round0/code"
"$ENVS/asr-captioner/bin/python" -m scripts.hash_checkpoints --settings ../settings.yaml --write ../checkpoint_hashes.json
```
The first time this *writes* the hashes (they go into `round0.zip`). Later, `--check` instead of
`--write` must end with `All checkpoints match.`

**6. Commit the lock files** the setup wrote, so every later build is identical:
`round0/code/environments/*/requirements.lock.txt` and `midterm/environment/requirements.lock.txt`
(and `round0/checkpoint_hashes.json`). Then continue with section 8.

## 7. Setup on an HPC without internet

The environments and models are built on a Linux x86_64 machine **with** internet, packed, and copied.

**1. Round 0** (machine with internet): `round0/README.md` Steps 1–5, with the `model_dir` of both
`round0/settings.yaml` and `round0/settings_eval.yaml` set. You get `$PACKED/asr-captioner.tar.gz`,
`$PACKED/generator.tar.gz` and the round 0 models in `$MODELS`.

**2. Scoring** (same machine, repo root, after setting `model_dir` in `midterm/settings.yaml`):
```bash
bash midterm/setup_scoring.sh midterm/settings.yaml "$PACKED" https://download.pytorch.org/whl/cu121
```
You get `$PACKED/scoring.tar.gz` and the scoring models in `$SCORING_MODELS`.

**3. Copy to the HPC**: the repo (`midterm` branch), `$PACKED`, `$MODELS` and `$SCORING_MODELS`:
```bash
rsync -av --progress capgen "$PACKED" "$MODELS" "$SCORING_MODELS" you@hpc:/scratch/you/capgen/
```

**4. On the HPC**: unpack the three environments, point the settings at the copied folders, and
check the model files survived the copy:
```bash
bash round0/code/scripts/unpack_envs.sh "$PACKED" "$ENVS"
sed -i "s#^model_dir: .*#model_dir: $MODELS#" round0/settings.yaml round0/settings_eval.yaml
sed -i "s#^model_dir: .*#model_dir: $SCORING_MODELS#" midterm/settings.yaml
cd round0/code && "$ENVS/asr-captioner/bin/python" -m scripts.hash_checkpoints --settings ../settings.yaml --check ../checkpoint_hashes.json
```
The last command must end with `All checkpoints match.` Then continue with section 8.

---

## 8. Running (both setups)

**1. Smoke test** on a GPU node, a few minutes, before the long job:
```bash
srun --gres=gpu:1 --pty bash          # or however your HPC gives an interactive GPU shell
cd "$REPO/midterm" && "$ENVS/scoring/bin/python" -m scoring.check --settings settings.yaml
```
Every line must say `PASS`, and the transcripts it prints must be real Hindi and Tamil text. Fix any
`FAIL` first (section 9); its message says what is missing.

**2. Edit the job file** `midterm/slurm/run_midterm.sbatch`: the lines marked `EDIT` (`REPO`, `ENVS`,
`RUNS`, the GPU partition, and `--gres=gpu:2` if your GPUs have 40 GB).

**3. Submit** from the repo root:
```bash
sbatch midterm/slurm/run_midterm.sbatch                          # MODE=split: run A + run B + merge
squeue -u $USER                                                  # is it running?
tail -f midterm-<jobid>.out                                      # watch it
```

| MODE | What runs | Time (one 80 GB GPU, rough) | The results.csv to submit |
|---|---|---|---|
| `split` (default) | run A, run B, merge | ~3 h | `$RUNS/results.csv` |
| `main` | run A only | ~1.5–2 h | `$RUNS/main/<run_id>/scores/results.csv` |
| `diversity` | run B only (builds `devset_div/` if needed) | ~1 h | `$RUNS/diversity/<run_id>/scores/results.csv` |
| `full` | 60 clips × 10 samples, every measure, one run | ~5–7 h | `$RUNS/full/<run_id>/scores/results.csv` |

Choose a mode with `sbatch --export=ALL,MODE=main midterm/slurm/run_midterm.sbatch`. To use two GPUs
at once, submit `main` and `diversity` as two jobs and merge afterwards (step 5). The times are
estimates; `scoring_run.json` records the real minutes of every stage.

**4. Resuming.** If a job is killed (time limit, node failure), submit the same mode again with
`RESUME=1`: `sbatch --export=ALL,MODE=split,RESUME=1 midterm/slurm/run_midterm.sbatch`. Every
finished transcript, description, sample and score is kept; only what is missing is done.

**5. Merging by hand** (only when `main` and `diversity` ran as separate jobs):
```bash
cd "$REPO/midterm" && "$ENVS/scoring/bin/python" -m scoring.diversity merge \
    --main "$RUNS/main/<run_id>/scores/results.csv" \
    --diversity "$RUNS/diversity/<run_id>/scores/results.csv" \
    --out "$RUNS/results.csv"
```
`<run_id>` is in `$RUNS/main/latest.json` and `$RUNS/diversity/latest.json`.

**6. Re-scoring only** (e.g. after adding ground-truth descriptions), without redoing round 0:
```bash
RESUME_RUN=latest SKIP_ROUND0=1 SCORING_ONLY=judge bash midterm/run_midterm.sh "$ENVS" "$RUNS/main"
```
Then merge again (step 5).

**What a run writes** (`$RUNS/<kind>/<run_id>/`):

| File | What it holds |
|---|---|
| `transcripts.jsonl`, `descriptions.jsonl` | round 0's transcript and description of every clip |
| `audio/<clip>/<n>.wav`, `audio.jsonl` | the generated samples and their seeds |
| `run.json`, `errors.jsonl` | round 0 lineage and attempts; round 0 failures |
| `scores/*.jsonl` | per-item scores of every stage (section 5) |
| `scores/results.csv`, `scores/results_detail.json` | the numbers |
| `scores/logs/`, `scores/errors.jsonl`, `scores/scoring_run.json` | scoring logs, failures, stage times |

## 9. When something goes wrong

| Where to look | What it has |
|---|---|
| `midterm-<jobid>.out` | Everything the job printed, with a timestamp on each line. **Start here.** |
| `<run>/errors.jsonl` | Round 0 failures (transcribe / describe / generate), one line per failure. |
| `<run>/run.json` | Round 0 lineage and every attempt: GPUs, counts, complete or not. |
| `<run>/scores/logs/<stage>.log` | One log per scoring stage. Each starts with host, GPUs, CUDA, torch and cache paths. |
| `<run>/scores/errors.jsonl` | Every scoring item that failed, with its full traceback. |
| `<run>/scores/scoring_run.json` | Exit code and minutes of each stage, the code commit, model revisions. |
| `<run>/scores/results_detail.json → missing` | Which measures are missing from `results.csv`, and why. |

A stage that fails does not stop the others: `results.csv` is built from what finished, and the
missing measures are listed. Common causes:

| Message | Cause, and what to do |
|---|---|
| `... not found: unpack the environments first` | `ENVS` in the sbatch file is wrong. It must hold `asr-captioner/`, `generator/` and `scoring/`. |
| `set model_dir in ...` | A settings file still has `/path/to/...`: section 6 step 2 (or 7 step 4). |
| `... is not in .../hf. Run scoring/download.py` | A scoring model was not downloaded (or not copied). Redo the scoring setup. |
| `401` / `gated` during setup | A gated model's terms were not accepted, or the wrong account is logged in (`hf auth whoami`). |
| `Read timed out` during setup | Slow network: run the same setup command again; finished downloads are kept. |
| `CUDA out of memory` (describe or caption_generated) | The captioner needs ~70 GB: ask for an 80 GB GPU or `--gres=gpu:2`. |
| `no kernel image is available` / `CUDA error` at start | The GPU driver is older than CUDA 12.1, or the GPU is too old (V100). Use another partition. |
| `psp_eval_dir is not set` (warning) | FAD and PSD are skipped; UTMOS and SIGMOS still run (section 10). |
| `reference is not clean hi script` (warning) | A code-mixed transcript; that clip is not scored for intelligibility (by design). |
| `ASR language differs from the devset folder` | The round 0 ASR picked the wrong language: an ASR error worth reporting. |
| `judge reply was not JSON` | Qwen3-8B answered badly once; it is asked again. If both fail, the item is in `errors.jsonl`. |
| round 0 `Incomplete` | See `<run>/errors.jsonl`. Scoring still runs on what finished. Resubmit with `RESUME=1`. |
| `Cannot resume ... has changed since it started` | The code or settings changed since that run began (e.g. a `git pull`). Submit without `RESUME=1` for a new run. |

## 10. Not done yet

1. **More ground-truth descriptions.** `rubric_score` (the captioner's main measure, 10 points at
   Rime) is computed only on clips that have one: 5 Hindi and 5 Tamil so far. Add more as one line
   per clip (JSON Lines, so line breaks inside the text are written `\n`):
   `{"audio_path": "devset/hi/clips/kashish1.wav", "ground_truth_description": "..."}`.
   Write at the level of Appendix F of the problem statement, and only what you can confirm by ear.
   Then re-score (section 8 step 6).
2. **psp-eval** (for FAD and PSD) is not public that we could find. Whoever wrote `nat.py` should put
   it on the HPC and set `naturalness.psp_eval_dir` in `midterm/settings.yaml`.
3. **Generation from `descriptions.csv`.** Rime evaluates the generator on text + description,
   without audio, with 10 samples each. `round0/code/scripts/run.py` only takes clips. Needed for
   the final submission, not for the midterm devset numbers.
4. **Not run with real models yet.** The scoring code has only been tested without models (unit tests
   and a dry run of the orchestrator). Run `scoring.check` first (section 8 step 1): the
   IndicConformer interface in particular could not be checked, because the model is gated.
5. **Freezing round 0**: `round0/checkpoint_hashes.json` and the `requirements.lock.txt` files are
   written by the setup (section 6 steps 5–6) and must be committed.
