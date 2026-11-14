# Phase 0: measurement and the first parts of the Claim-Calibrated Loop

This explains the code added on the `Tentative` branch after the mid-term: what each part does,
how to run it, what is checked and what is still open. Track 2 only (captioner and generator,
Hindi and Tamil). Round 0 (`round0/`) is frozen and is not touched by any of this.

Read first: [post_midterm_plan.md](post_midterm_plan.md) (what we are building and why) and
[evaluation_plan.md](evaluation_plan.md) (the spec for the evaluation package).

## 1. What is where

| Folder or file | What it holds |
|---|---|
| `settings/phase0.yaml` | Every value that defines a result: models and revisions, grader options, bootstrap, thresholds, composer and extractor options. The same on every machine. |
| `settings/paths.example.yaml` | Copy to `settings/paths.yaml` and set the two folders of your machine. `paths.yaml` is never committed. |
| `environments/evaluation/` | The conda environment definition for everything below. |
| `evaluation/` | The measurement package: rubric score, grader, generator measures, statistics, `results.csv`, run manifest. |
| `claims/` | The claim model, the composer (the knapsack) and the claim extractor. |
| `training_data/` | The training manifest: one row per training example, and its SHA-256. |
| `scripts/download_models.py` | Downloads the evaluation models at pinned revisions; records and checks their hashes. |
| `scripts/ladder_rung2.py` | Ladder rung 2: round 0 captions re-composed into 1,000 characters, scored against the originals. |
| `tests/` | Unit tests on small inputs with known answers, and model tests that skip when a model is absent. |

The other files in `scripts/` are the team's earlier devset utilities and are unchanged.

### The evaluation package

| File | What it does |
|---|---|
| `settings.py` | Checks the `evaluation` section of the settings file and `paths.yaml`. Unknown keys are errors; nothing has a default. |
| `rubric/schema.py` | `RubricItem` (one statement about a clip, true or false, optionally timed, with a weight) and the grader's three verdicts. Reads a rubric JSONL file. |
| `rubric/score.py` | The clip score from the verdicts, weighted and unweighted. |
| `rubric/grader.py` | Everything about grading that needs no model: the 1,000-character cut, the frozen prompt, strict parsing, the loop over items, the parse-failure rate. |
| `rubric/grader_sarvam.py` | The `sarvamai/sarvam-m` grader: greedy, 16 new tokens, thinking off. |
| `captioner_eval.py` | Grades the descriptions of several systems against a rubric (resumable) and summarises each system with a confidence interval. |
| `generator_eval/interfaces.py` | One interface each for an ASR, a speaker-verification model and a naturalness scorer. |
| `generator_eval/intelligibility.py` | CER of an ASR transcript against the target text; value = 1 − CER, floored at 0. |
| `generator_eval/diversity.py` | Cosine similarity, the greedy distinct-speaker count, speaker diversity over 10 samples, similarity to a target recording. |
| `generator_eval/whisper_asr.py`, `indic_conformer_asr.py`, `wavlm_speaker.py`, `utmos_scorer.py` | The model implementations behind those interfaces. |
| `text_norm.py` | The text normalisation shared by both ASRs (NFC, lowercase, no punctuation, single spaces) and the CER. |
| `stats.py` | Hash-based bootstrap confidence interval, paired comparison of two systems, gap-closed level 0 to 4. |
| `report.py` | The `results.csv` writer: `track, part, language, system, measure, n_items, value`. |
| `manifest.py` | The run manifest: code commit and SHA-256 of the settings, the inputs and the checkpoints. |

### The claims package

| File | What it does |
|---|---|
| `schema.py` | `Claim`: item type (one of the 13 rubric types), statement, optional start and end time, source (model, tool or transform), probability of being right. |
| `composer.py` | Ranks claims by expected value per character and fills at most 1,000 characters. Deterministic, no model call. |
| `extractor.py` | `ClaimExtractorInterface`, and `RuleBasedExtractor`: one claim per sentence of a caption, typed by keyword. |
| `settings.py` | Checks the `claims` section of the settings file. |

## 2. Setup

Build the environment once, on a machine with internet. Use the PyTorch index that matches the
machine's CUDA version.

```bash
conda env create -p /path/to/environments/evaluation -f environments/evaluation/environment.yml
conda activate /path/to/environments/evaluation
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r environments/evaluation/requirements.txt
pip freeze > environments/evaluation/requirements.lock.txt      # commit this file

cp settings/paths.example.yaml settings/paths.yaml              # then edit the two folders
```

Download the models (about 55 GB, mostly the grader), then commit `settings/checkpoint_hashes.json`:

```bash
python -m scripts.download_models download --settings settings/phase0.yaml --paths settings/paths.yaml
```

The IndicConformer repo is gated: accept its terms on Hugging Face and log in (`hf auth login`)
before downloading. The cluster has no internet: copy the environment and the models folder there
as round 0 does, then check the copy:

```bash
python -m scripts.download_models verify --settings settings/phase0.yaml --paths settings/paths.yaml
```

## 3. One command per part

All commands run from the repository root with the evaluation environment.

| What | Command |
|---|---|
| All unit tests (no GPU, no model) | `python -m pytest` |
| The same, showing why model tests were skipped | `python -m pytest -rs` |
| One model test (needs that model downloaded) | `python -m pytest tests/test_models.py -k whisper -rs` |
| Download the models | `python -m scripts.download_models download --settings settings/phase0.yaml --paths settings/paths.yaml` |
| Check the models' hashes | `python -m scripts.download_models verify --settings settings/phase0.yaml --paths settings/paths.yaml` |
| Ladder rung 2 (GPU node) | `python -m scripts.ladder_rung2 --settings settings/phase0.yaml --paths settings/paths.yaml --run <round 0 run folder> --rubric <rubric.jsonl> --name <output name>` |

The library parts have no command of their own; they are called from Python:

```python
from evaluation.rubric.score import clip_score                    # rubric score of one clip
from evaluation.stats import confidence_interval, paired_difference, gap_closed_level
from evaluation.generator_eval.intelligibility import score_sample
from evaluation.generator_eval.diversity import speaker_diversity, similarity_to_target
from evaluation.report import write_results_csv
from claims.composer import compose
from training_data.manifest import write_training_manifest, training_manifest_sha256
```

## 4. How the rubric score is computed

1. The description is cut to its first 1,000 characters, then to at most 300 words.
2. An empty or non-English description gets `NOT_MENTIONED` for every item, which scores 0.
3. The grader answers one question per rubric item with one word: `STATED`, `CONTRADICTED` or
   `NOT_MENTIONED`. For a timed item the prompt gives the item's span widened by 0.25 s each side.
4. The answer is parsed strictly: the first non-empty line must be exactly one of the three words.
   Anything else counts as `NOT_MENTIONED` and is counted in the parse-failure rate.
5. Points per item:

   | Item | STATED | NOT_MENTIONED | CONTRADICTED |
   |---|---|---|---|
   | true | +w | 0 | −w |
   | false | −w | 0 | 0 |

6. Clip score = max(0, sum of points ÷ sum of w over the true items).
7. Per language: the mean over clips. Overall: the mean of Hindi and Tamil.

With every w = 1 this is the doubt-session version, written to `results.csv` as `rubric_score`.
With the items' difficulty weights it is the problem-statement version, `rubric_score_weighted`.

A rubric is a JSONL file with one item per line; see `evaluation/rubric/schema.py` for the fields.
The clip id is `<language>/<name>`, where `<name>` is the round 0 clip id (the file name without
`.wav`). Every field is required; write `null` for the times of an untimed item.

## 5. How the composer works

The rubric pays +1 for a right claim that matches an item and −1 for a wrong one, and the grader
reads only 1,000 characters. So for a claim with probability p of being right:

- expected value = (2p − 1) × prior of its item type
- cost = the length of its sentence plus one separator

The composer drops claims with p below `min_probability`, keeps one of any duplicated sentence,
sorts the rest by value per character, and adds each claim that still fits whole. A claim is never
cut, and the text never exceeds `max_chars`. The chosen claims appear in ranking order.

In Phase 0 every prior is 1 and the rule-based extractor gives every claim the same probability.
The ranking then reduces to "shortest sentences first", which packs the most claims into the
budget. That is exactly what rung 2 tests: the budget and the ordering, with nothing else changed.

## 6. Ladder rung 2

`scripts/ladder_rung2.py` scores two systems on the same clips with the same grader:

| System | Description given to the grader |
|---|---|
| `round_0` | The round 0 caption as it is. The grader reads its first 1,000 characters. |
| `round_0_recomposed` | The same caption, split into claims and re-composed into 1,000 characters. |

It writes `compositions.jsonl`, `gradings.jsonl`, `summary.json`, `results.csv`, `manifest.json`
and `run.log` to `<outputs>/<name>/`. The summary has each system's score per language, the overall
score with its 95% confidence interval, the parse-failure rate, and the paired difference with its
gap-closed level. Grading is resumable: after a time limit, run the same command again.

It cannot give a real number yet: **the dev rubrics do not exist**. Someone has to write rubric
items for the clips (evaluation plan, section 2) before this script measures anything.

## 7. Rules this code follows

- Pydantic models for every setting and every record. No default values anywhere: every setting
  comes from `settings/phase0.yaml`, and every function argument is passed by the caller.
- Imports at the top of each file.
- Models return raw output; the measure that uses it normalises, parses and checks it.
- One interface per kind of model, with the implementation in its own file.
- No cluster names, usernames or personal paths. Machine folders live in `settings/paths.yaml`.
- Models are pinned to a commit, loaded from local folders only, and checked by SHA-256.
- Intelligibility is never measured with the pipeline's own ASR.
- Held-out audio is for reporting only. Nothing in Phase 0 tunes on data: the composer's
  settings are fixed in the settings file. When thresholds or priors are fitted later, they must
  be fitted on the tuning split, and the script that fits them must refuse a held-out rubric.

## 8. What is verified, and what is not

Verified here, on a laptop:

- 66 unit tests pass: rubric score against hand-computed values, the grader's parsing and loop,
  the bootstrap and levels, the composer and extractor, both manifests, CER and speaker counting,
  and rung 2 from start to finish with a fake grader.

Not verified, because the model or a large GPU was not available:

| Part | What is unverified |
|---|---|
| sarvam-m grader | Never loaded. That it fits in bfloat16 on the cluster's GPUs; that `enable_thinking: false` gives a one-word answer within 16 tokens; the parse-failure rate on real descriptions. `tests/test_models.py` has a test for this that runs once the model is downloaded. |
| IndicConformer | Never loaded (gated). Whether its code finds its ONNX files when loaded from a plain folder (CAREFUL I1 in `indic_conformer_asr.py`). |
| `scripts/download_models.py` | Not run end to end (the grader alone is about 48 GB). |
| Rung 2 on real data | Needs the grader, a round 0 run folder and a rubric. |

## 9. Open choices and disagreements

These are places where the two plan documents disagree with each other or with the brief, or
leave a choice open. Each is implemented one way, in one place, and listed here so that it can be
decided, not guessed.

| # | Topic | What the documents say | What the code does |
|---|---|---|---|
| 1 | Rubric weights | Problem statement: difficulty-weighted. Doubt session: unweighted. | Both are computed; unweighted is `rubric_score`. |
| 2 | Word limit | The 1,000-character cut comes from the doubt session; the 300-word limit from the problem statement. Whether both still apply is an open question for Discord. | Both are applied, characters first. |
| 3 | Validity checks | The evaluation plan asks to score 0 a description that is empty, not English, or addresses the grader. | Empty and not-English are checked. "Addresses the grader" is not: it needs a model or a person. |
| 4 | Parsing "strictly" | The plan does not define it. | The first non-empty line must be exactly one of the three words. |
| 5 | Bootstrap counting | The plan gives the hash formula but not whether replicates and draws count from 0 or 1. | Replicates from 1, draws from 0 (as the midterm scoring did). |
| 6 | Bootstrap across languages | "Resample each language separately, then average." The formula has no language in it. | Each language is resampled with the same formula, so two languages with the same number of items get the same positions. |
| 7 | Gap-closed level edges | "1 if the CI includes 0." | An interval whose lower end is exactly 0 counts as including 0. |
| 8 | Speaker diversity count | The evaluation plan: greedy count against kept representatives, among samples that pass relevance. The midterm code: complete-linkage clustering over all samples. | Greedy, over all 10 samples (relevance checks are not built yet). The rule matters: on 100 real generator samples (10 clips × 10) the greedy count gave 1.2 speakers per clip in both languages, where complete linkage gave 1.4 (Hindi) and 2.0 (Tamil). |
| 9 | Speaker measure | Problem statement: diversity. Doubt session: similarity to the target. | Both are built. |
| 10 | Intelligibility ASR | Problem statement: IndicConformer. Doubt session: Whisper. | Both are built, with the same normalisation. |
| 11 | CER unit | Not specified. | Unicode code points after normalisation, not grapheme clusters. |
| 12 | Naturalness | The plan lists UTMOSv2 and DNSMOS; the brief lists UTMOS. | UTMOS (`utmos22_strong` through SpeechMOS, as the midterm used). UTMOSv2 and DNSMOS are not built. |
| 13 | Rubric clip ids | The plan's example is `hi/own_0007`; round 0 clip ids are file names without a language. | The rubric id is `<language>/<round 0 clip id>`, with the language taken from the clip's folder. |
| 14 | Claim extractor | Left open by the brief. | Rule-based: sentences typed by keyword. Rough, and enough only while all priors are equal. |
| 15 | Rung 2 probabilities | Round 0 captions carry no confidence. | Every claim gets `extractor.probability` (0.75). |
| 16 | Timed claims in the composer | Not specified how a time is written. | A claim with a span gets " (start-end s)" appended; the rule-based extractor never sets a span, so a sentence keeps its own wording. |
| 17 | Order of composed claims | Not specified. | Ranking order, most valuable first. |
| 18 | Branch | The brief says `claim-calibrated-loop`; the request said `Tentative`. | `Tentative`, branched from `main`. |
| 19 | Reuse of the midterm scoring | The `midterm` branch is not merged into `main`. | Nothing is imported from it. Its ideas (bootstrap, model wrappers) are rewritten here to the no-defaults rule. |

## 10. Not built in Phase 0

These are in the evaluation plan but outside this task: the hash-based splits (`splits.py`),
tool-measured and transform-derived rubric items, the relevance checks, FAD and PSD, the
round-trip evaluation, and a script that scores a whole round 0 run's generated audio. The
generator measures exist as library functions; the script that applies them to a run folder is
the next step.
