# How the midterm repo works

This branch (`midterm`) turns the repo into one pipeline. It runs the round 0 system on the devset,
scores what it produced, and writes `results.csv` in the midterm format. Everything heavy runs on
the HPC; nothing here is meant to run on a laptop.

```
devset/{hi,ta}/clips/*.wav
        │
        ▼  round 0 (round0/code/scripts/run.py), envs: asr-captioner + generator
   ASR (indic-transcribe) ──► transcript ─┐
   captioner (Qwen3-Omni) ──► description ┴─► generator (Indic Parler-TTS) ──► 10 samples per clip
        │
        ▼  scoring (midterm/scoring/run.py), env: scoring
   intelligibility · speaker_diversity · naturalness · caption_generated · judge
        │
        ▼
   <runs>/<run_id>/scores/results.csv  (+ results_detail.json with confidence intervals)
```

One command does all of it: `bash midterm/run_midterm.sh <envs> <runs>`, usually through
`sbatch midterm/slurm/run_midterm.sbatch`.

---

## 1. What is where

| Path | What it is |
|---|---|
| `round0/` | The round 0 system (frozen models, no training). `round0/README.md` reproduces it; `round0/code/EXPLANATION.md` explains it. |
| `midterm/run_midterm.sh` | The whole pipeline: round 0 on the devset, then scoring, then prints `results.csv`. |
| `midterm/slurm/run_midterm.sbatch` | SLURM job around `run_midterm.sh`. Edit the lines marked `EDIT`. |
| `midterm/settings.yaml` | Every scoring setting: models and their pinned revisions, thresholds, which stages run, reference descriptions. |
| `midterm/setup_scoring.sh` | Builds the `scoring` conda env, downloads the scoring models, packs the env. Machine with internet. |
| `midterm/scoring/` | The scoring package (see section 4). |
| `midterm/environment/` | The `scoring` env: `environment.yml`, `requirements.txt` (and `requirements.lock.txt` after the first build). |
| `nat.py` | Naturalness scorer (UTMOS, SIGMOS, FAD, PSD), used by the naturalness stage. |
| `score_hi_ta.py`, `normalize_hi_ta.py` | CER/WER for Hindi and Tamil, used by the intelligibility stage. |
| `devset/{hi,ta}/clips/` | 30 Hindi and 30 Tamil dev clips. The language of a clip is the folder it is in. |
| `devset/{hi,ta}/tool_descriptions.json` | What `features-extractor/` measured on each clip. Used as a second reference for the rubric. |
| `devset/{hi,ta}/ground_truth_descriptions.jsonl` | Hand-written reference descriptions for the rubric score: 5 Hindi and 5 Tamil clips so far (see section 7). |
| `features-extractor/` | Acoustic feature extraction that wrote the tool descriptions. Not part of this pipeline run. |
| `6621_midterm/test_set.zip` | Our contributed test set. Not used here: it is for the other teams. |

## 2. What changed on this branch

- **Captioner fix** (`round0/code/pipeline/models/captioner.py`). With transformers 4.57.6, Qwen3-Omni's
  `generate()` returns a pair `(text, audio)`. The code read `.sequences` from the pair, so every
  clip failed and round 0 would have produced 0 descriptions. It now unpacks the pair (CAREFUL C10).
- **Checkpoint-hash fix** (`round0/code/scripts/hash_checkpoints.py`). The ASR's code runs from inside
  its model folder, so Python writes `__pycache__` there; `--check` then reported a mismatch after
  any run. `__pycache__` is now skipped (CAREFUL H2).
- **`nat.py`**: `utmos_repo` and `xlsr_model` can now be local folders, so it works offline on the HPC.
  The usage example now passes the required `metrics` argument, and the SIGMOS source is corrected
  (it is in microsoft/SIG-Challenge, not DNS-Challenge).
- **New**: `midterm/` (scoring pipeline) and this file.

## 3. The measures in results.csv

`results.csv` has the midterm columns `track,part,language,system,measure,value,n_items`, one row per
number, for `hi` and `ta` separately, with `system=round_0`. Each value is computed per item (one
devset clip = one description) and then averaged over items, as Rime averages over clips and
descriptions. `results_detail.json` has the same numbers with 95% bootstrap confidence intervals
(Rime's Appendix A hash-based draws), what each measure means, which direction is better, and why
any measure is missing.

| measure | part | Rime's measure (points) | How we compute it | Better |
|---|---|---|---|---|
| `rubric_score` | captioner | Rubric score (10) | A text LLM (Qwen3-8B) splits the **ground-truth** description into rubric items and grades the round 0 caption, cut at 300 words: (stated − contradicted) / items, floored at 0 | higher |
| `rubric_score_tool` | captioner | (proxy) | Same, with the **tool descriptions** as the reference: measured facts only (pitch, pauses, SNR, events…) | higher |
| `instruction_relevance` | generator | Instruction relevance (5) | Qwen3-8B splits the instruction into checks. The round 0 captioner describes a generated sample, and Qwen3-8B marks each check met / not met / unverifiable. Score = met / checks | higher |
| `intelligibility` | generator | Intelligibility (1) | CER of AI4Bharat IndicConformer (the ASR Rime uses) on each generated sample, against the text the generator was asked to say. Rime's points = 1 − CER | **lower** |
| `speaker_diversity` | generator | Speaker diversity (2) | Distinct speakers among the 10 samples of a description, using WavLM-base-plus-sv at cosine 0.86 with complete-linkage clustering | higher |
| `naturalness_utmos` | generator | (proxy for CMOS, 2) | UTMOS, 1–5, per sample | higher |
| `naturalness_sigmos_ovrl` | generator | (proxy) | SIGMOS overall, 1–5, per sample | higher |
| `naturalness_fad` | generator | (proxy) | Fréchet distance of XLS-R layer-9 embeddings, generated vs native devset clips, per language (`n_items` = samples) | **lower** |
| `naturalness_psd` | generator | (proxy) | Same with prosody features | **lower** |

`results_detail.json → native_level` gives the UTMOS and SIGMOS of the real devset clips: the score
that real speech gets, to read the generator's numbers against.

### Choices where Rime is silent or cannot be copied (for the report's "Your choices" section)

- **Rubric score.** Rime's rubrics are private and graded by "a fixed grader model". We build rubrics
  from reference descriptions with a text LLM and use equal weights, because Rime's weights (by
  difficulty) are not published. Contradicting a true item is penalised. Rime's separate "false
  items" (invented details) are not modelled.
- **Instruction relevance.** Rime uses three listeners. We use the captioner as the judge (Rime
  suggests this), so the measure is only as good as the captioner. A property the caption does not
  mention counts as *not met* (conservative); the share of `unverifiable` checks is in `relevance.jsonl`.
  Only `relevance.samples_per_item` (2) samples per item are judged, to save GPU time.
- **Intelligibility.** The reference is the round 0 ASR transcript of the devset clip, because the
  devset has no human transcripts. Errors of that ASR therefore count against the generator.
  Transcripts with Latin letters (code-mixed speech) are skipped and logged, because `score_hi_ta.py`
  only scores native script. We use `cer_ns` (CER with spaces removed, the primary metric of `score_hi_ta.py`).
- **Speaker diversity.** Rime counts only among samples that pass the relevance checks. We count
  among all 10, because relevance is judged on only 2. Rime's speaker model is not named; we use
  WavLM-base-plus-sv at its model-card threshold.
- **Naturalness.** Rime's measure is a blind CMOS against other teams' outputs, which cannot be run
  alone. UTMOS, SIGMOS, FAD and PSD are automatic stand-ins. UTMOS was trained on English, so
  compare it to the native level, not to published numbers.
- **What the generator is given.** On the devset the generator re-speaks each clip: the text is the
  ASR transcript and the instruction is the round 0 caption. At evaluation, Rime gives text and a
  human-written description instead (`descriptions.csv`). That path is not built yet (section 7).

## 4. The scoring package (`midterm/scoring/`)

| File | Role |
|---|---|
| `run.py` | Orchestrator. Runs each stage in its own process, then `results.py`. Sets the offline environment (`HF_HUB_OFFLINE=1`, pinned caches, `PYTHONPATH`). |
| `stages/intelligibility.py` | IndicConformer → CER per sample → `intelligibility.jsonl` |
| `stages/speaker_diversity.py` | WavLM-SV embeddings → speakers per item → `speaker_diversity.jsonl` |
| `stages/naturalness.py` | `nat.py` → `naturalness.jsonl`, `naturalness_native.jsonl`, `naturalness_corpus.json` |
| `stages/caption_generated.py` | Round 0 captioner on generated samples → `caption_generated.jsonl` (about 70 GB of GPU memory) |
| `stages/judge.py` | Qwen3-8B → `rubric.jsonl`, `relevance.jsonl`, plus `extractions.jsonl` (cached rubric items) |
| `results.py` | All of the above → `results.csv`, `results_detail.json` |
| `diversity.py` | Builds the speaker-diversity subset of the devset, and merges its results into the main `results.csv` (section 5) |
| `check.py` | Smoke test: loads every scoring model and runs it on one clip per language |
| `download.py` | Downloads every scoring model at its pinned revision |
| `models.py`, `llm.py`, `common.py`, `settings.py`, `stage.py` | Shared code (model loaders, judge prompts and parsing, logging, bootstrap, settings) |
| `tests/` | Unit tests without models: `python -m pytest scoring/tests` from `midterm/` |

Models (all open, pinned in `midterm/settings.yaml`): Qwen3-8B (judge), IndicConformer-600M
(gated: accept its terms on Hugging Face), WavLM-base-plus-sv, wav2vec2-XLS-R-300M, UTMOS
(tarepan/SpeechMOS v1.2.0), SIGMOS (microsoft/SIG-Challenge). The Qwen3-Omni captioner is round 0's own.

## 5. Running it on the HPC

The HPC has no internet, so the environments and models are prepared on a Linux machine with
internet, then copied. Steps 1–3 happen once.

**1. Round 0 environments and models.** Follow `round0/README.md` Steps 1–5. You get the
`asr-captioner` and `generator` packed envs and the round 0 models.

**2. Scoring environment and models** (same machine, from the repo root):
```bash
# set model_dir in midterm/settings.yaml first (an absolute path, about 25 GB)
bash midterm/setup_scoring.sh midterm/settings.yaml "$PACKED" https://download.pytorch.org/whl/cu121
```
If it stops with "Not logged in", run the `hf auth login` command it prints and run it again. Commit
`midterm/environment/requirements.lock.txt` once it is written.

**3. Copy to the HPC and unpack.** Copy the repo (`git clone` + `git checkout midterm`), the packed
envs, the round 0 models and the scoring models. Then:
```bash
bash round0/code/scripts/unpack_envs.sh "$PACKED" "$ENVS"       # unpacks asr-captioner, generator, scoring
```
Set `model_dir` in `round0/settings_eval.yaml` and in `midterm/settings.yaml` to their folders on the HPC.

**4. Smoke test** (GPU node, a few minutes). Do this before the long job:
```bash
cd midterm && "$ENVS/scoring/bin/python" -m scoring.check --settings settings.yaml
```
Every line must say `PASS`. Also check the transcripts it prints are real Hindi and Tamil text.

**5. The real runs.** Generating 10 samples for every clip is the expensive part, and only speaker
diversity needs 10. So the devset is run twice, each with its own `<runs>` folder:

| Run | Clips | Samples | Generations | Gives |
|---|---|---|---|---|
| A (main) | all 60 | 1 (`round0/settings.yaml`) | 60 | every measure except speaker diversity |
| B (diversity) | 10 hi + 10 ta | 10 (`round0/settings_eval.yaml`) | 200 | speaker diversity |

```bash
# once: the subset (one clip per speaker in turn; writes devset_div/subset.json)
cd midterm && python -m scoring.diversity subset --devset ../devset --out ../devset_div --per-language 10 && cd ..

# run A: in the sbatch file, the last lines become
ROUND0_SETTINGS=$REPO/round0/settings.yaml SCORING_ONLY=intelligibility,naturalness,caption_generated,judge \
    bash "$REPO/midterm/run_midterm.sh" "$ENVS" "$RUNS/main"
# run B: a second job (or after A in the same job)
ROUND0_SETTINGS=$REPO/round0/settings_eval.yaml CLIPS=$REPO/devset_div SCORING_ONLY=speaker_diversity \
    bash "$REPO/midterm/run_midterm.sh" "$ENVS" "$RUNS/diversity"
```
Each job needs about 70 GB of GPU memory (one 80 GB A100/H100 or two 40 GB A100s; V100s do not
work). Rough time: run A 1.5–2 h, run B about 1 h. The first real run gives the true numbers;
`scoring_run.json` records the minutes of every stage. Seeds depend only on the clip and the sample
number, so sample 0 of a clip is the same in both runs.

To do everything in one run instead (about 5–7 h): `sbatch midterm/slurm/run_midterm.sbatch` as it is,
which uses `settings_eval.yaml` (10 samples) on all 60 clips and runs every stage.

**6. Results.** Each job prints its `results.csv` at the end (`<runs>/<run_id>/scores/results.csv`).
Merge the two runs into the file you submit:
```bash
cd midterm && python -m scoring.diversity merge \
    --main "$RUNS/main/<run_id>/scores/results.csv" \
    --diversity "$RUNS/diversity/<run_id>/scores/results.csv" \
    --out ../results.csv
```
The merged file keeps every row of run A and takes speaker diversity from run B.

### Resuming

Everything resumes: every finished item is on disk, and a second run only does what is missing.
- Job killed during round 0 or scoring: `RESUME_RUN=latest` before `bash midterm/run_midterm.sh ...`
  (or add `export RESUME_RUN=latest` in the sbatch file).
- Only re-score an existing run: `RESUME_RUN=<run_id> SKIP_ROUND0=1 bash midterm/run_midterm.sh ...`
- Only some stages: `python -m scoring.run --settings settings.yaml --run <run folder> --only judge`

## 6. When something goes wrong: where to look

| Where | What it has |
|---|---|
| `midterm-<jobid>.out` | Everything the job printed, with a timestamp on each line. Start here. |
| `<run>/errors.jsonl` | Round 0 failures (transcribe / describe / generate), one line per failure. |
| `<run>/run.json` | Round 0 lineage and every attempt: GPUs, counts, complete or not. |
| `<run>/scores/logs/<stage>.log` | One log per scoring stage. Each starts with host, GPUs, CUDA, torch and cache paths. |
| `<run>/scores/errors.jsonl` | Every scoring item that failed, with its full traceback. |
| `<run>/scores/scoring_run.json` | Exit code and minutes of each stage, the code commit, model revisions. |
| `<run>/scores/results_detail.json → missing` | Which measures are missing from `results.csv`, and why. |

A stage that fails does not stop the others. `results.csv` is built from what finished, and the
missing measures are listed. Common causes:

| Message | Cause, and what to do |
|---|---|
| `... is not in .../hf. Run scoring/download.py` | A scoring model was not downloaded or not copied. Redo setup step 2 and copy `model_dir` again. |
| `401` / `gated` during setup | IndicConformer's terms were not accepted on Hugging Face, or the wrong account is logged in. |
| `CUDA out of memory` in `caption_generated` | The captioner needs about 70 GB; ask for a bigger GPU or two GPUs (`--gres=gpu:2`). |
| `psp_eval_dir is not set` (warning) | FAD and PSD are skipped; UTMOS and SIGMOS still run. See section 7. |
| `reference is not clean hi script` (warning) | A code-mixed transcript; that clip is not scored for intelligibility (by design). |
| `ASR language differs from the devset folder` | The round 0 ASR picked the wrong language: an ASR error worth reporting. |
| `judge reply was not JSON` | Qwen3-8B answered badly once; it is asked again. If both fail, the item is in `errors.jsonl`. |
| round 0 `Incomplete` | See `<run>/errors.jsonl`. Scoring still runs on what finished. Resume with `RESUME_RUN=<run_id>`. |

## 7. Not done yet

1. **More ground-truth descriptions.** `rubric_score` (the captioner's main measure, 10 points at
   Rime) is computed on the clips that have one: 5 Hindi and 5 Tamil so far. Add more as one line per
   clip (JSON Lines, so line breaks in the text are written `\n`):
   `{"audio_path": "devset/hi/clips/kashish1.wav", "ground_truth_description": "..."}`,
   at the level of Appendix F of the problem statement. A pretty-printed JSON list in the file is
   accepted too. The text field name is `references[0].text_field` in `midterm/settings.yaml`.
   Re-score with `RESUME_RUN=<run_id> SKIP_ROUND0=1 SCORING_ONLY=judge` (only new clips are graded).
2. **psp-eval** (for FAD and PSD) is not public that we could find. Whoever wrote `nat.py` should put
   it on the HPC and set `naturalness.psp_eval_dir` in `midterm/settings.yaml`.
3. **Generation from `descriptions.csv`.** Rime evaluates the generator on text + description
   without audio, with 10 samples each. `round0/code/scripts/run.py` only takes clips. This is
   needed for the final submission, not for the midterm devset numbers.
4. **Not run yet.** The scoring stages have only been tested without models (unit tests and a dry
   run of the orchestrator). Run `scoring.check` on the HPC first: the IndicConformer interface in
   particular could not be checked, because the model is gated.
5. `checkpoint_hashes.json` and the `requirements.lock.txt` files still have to be written and committed
   (`round0/README.md`, "For us: freezing round 0").
