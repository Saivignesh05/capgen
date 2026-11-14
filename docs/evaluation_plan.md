# Track 2 Evaluation Plan — Captioner, Generator and Full Loop

Spec for the `evaluation` package. Written to guide implementation (e.g. by Claude Code) and to produce `results.csv` plus the "Baselines & Evaluation Setup" and "Early Results" sections of the mid-term report.

Sources: Rime problem statement (Track 2 + Appendix A, F), mid-term submission guidelines, Track 2 doubt-session answers (Q1–Q31).

---

## 0. Ground rules

1. **Mirror the official measures wherever possible.** Where Rime uses private rubrics or its own listeners, build a *proxy*. In every table and in `results.csv`, label each number as **official-tool** (same model and formula Rime uses) or **proxy** (our approximation).
2. **Never use the pipeline's own models to grade themselves.** For example, intelligibility is measured with IndicConformer / Whisper, never with Indic-Transcribe-Core. The generator is judged by a *different-family* ALM, not by the Qwen captioner.
3. **Held-out real audio is for reporting only.** It must never be used to choose training data, prompts or checkpoints (Q11, Q26). Keep a separate tuning split.
4. **Every run writes a run manifest**: git commit, SHA-256 of every checkpoint, decoding settings, precision/quantisation, seeds, input-list hash. The same data feeds `round0.zip` and `freeze.txt`.
5. **Store per-item outputs as JSONL**, never only aggregates, so that confidence intervals and error breakdowns can be recomputed.
6. **Code style**: pydantic settings and IO models, **no default values** for any setting or argument, imports at the top of each file, no cluster names, usernames or personal paths in code (use placeholders). Models return raw output; cleaning and validation is the consumer's job. Everything must run offline on the HPC cluster, so every checkpoint is pre-downloaded and pinned by hash.

### 0.1 Conflicts between the problem statement and the doubt session — compute both

| Topic | Problem statement | Doubt session (Q31, later) | What to do |
|---|---|---|---|
| Points split | Rubric 10; relevance 5; naturalness 2; speaker diversity 2; intelligibility 1 | Describe 7; Simulate 7 (relevance 3, naturalness 2, intelligibility 1, speaker similarity 1); Loop gain 6 | Report the raw measures; points are derived later |
| Intelligibility ASR | AI4Bharat IndicConformer CER | Whisper CER "by skill" | Compute **both** |
| Generator speaker measure | Speaker diversity (10 samples, distinct speakers) | Speaker similarity (WavLM cosine vs target recording) | Compute **both** (§4.4, §4.5) |
| Rubric scoring | Difficulty-weighted items | Unweighted +1 / 0 / −1, divided by the number of true items, floored at 0 | Compute **both** (weights = 1 gives the Q31 version) |
| Naturalness CMOS | Against other teams | Against the round-0 output | Proxy only (§4.3) |

Ask on Discord which one is binding, and put the answer in the report's "Your choices" section.

---

## 1. Datasets and splits

All splits are chosen **deterministically by hash**, Appendix A style: sort by `sha256(f"{salt}|{split_name}|{clip_id}")` as an unsigned hex integer, then take the first *m*. Freeze the salt in config.

| Split | Contents | Size per language (target) | Use |
|---|---|---|---|
| `heldout_real` | Own recorded clips (the 30 contributed per language) + a Vaani slice | 30 own + ~70 Vaani | **Reporting only** |
| `tune_real` | Different Vaani clips, speaker-disjoint from `heldout_real` | ~100 | Prompt and threshold tuning, debugging |
| `transform_eval` | Clean clips with *known* edits applied (§2.3) | ~50 sources × transforms | Captioner sensitivity, exact labels |
| `gen_prompts` | Own 30 descriptions per language (with text) + ~50 extra (templated + LLM-written, checked by a person) | 80 | Generator evaluation |
| `roundtrip` | = `heldout_real` | — | Whole-pipeline evaluation (§5) |

- Vaani slice: sample across districts and noise conditions. Enforce speaker-disjointness using the Vaani speaker metadata, grouping by speaker before hashing.
- Keep the Vaani transcripts for ASR evaluation. Clean them with the same tag-stripping rules as the Track 1 Part 1 baseline: drop clips with `[unintelligible]`, `[inaudible]` or `<UNKNOWN_SEGMENT>`; strip other bracketed tags, curly-brace text and `--`; collapse whitespace.
- Your own contributed clips may be used for dev reporting. Do not tune on them.

---

## 2. Dev rubrics (Rime's rubrics are private, so we build our own)

### 2.1 Item schema (JSONL, one item per line)

```json
{
  "clip_id": "hi/own_0007",
  "item_id": "hi/own_0007#05",
  "axis": "sound | ecology | intent",
  "item_type": "Speaker | Source | Morphology | Pitch | Timing | Event | Channel | Space | Other sounds | Coupling | Addressee | Subtext | Change",
  "statement": "The speaker laughs at 3.2–3.9 s",
  "truth": true,
  "start_s": 3.2,
  "end_s": 3.9,
  "weight": 2,
  "provenance": "human:<annotator_code> | tool:<tool_name>@<checkpoint_sha> | transform:<transform_id>"
}
```

### 2.2 Three sources of items

1. **Human-written** (gold, small). Two team members independently write 8–15 items per clip in the F3 format, then adjudicate. Include 2–4 **false** items per clip, such as a plausible-but-wrong gender or age, an event that did not happen, the wrong channel, or the wrong pitch direction. False items are what penalise invented detail. Weights: 1 = easy (gender), 2 = medium (channel, rate), 3 = hard (subtext, exact timing). Target: as many own clips as time allows, at least 15–20 per language.
2. **Tool-measured** (cheap, all clips). These are generated from signal features with fixed tolerances:
   - F0 median and range in semitones (torchcrepe or librosa pyin)
   - speaking rate in aksharas/s (from ASR text and VAD speech time)
   - pauses over 300 ms (silero-vad)
   - SNR and C50/reverb (e.g. brouhaha)
   - bandwidth: narrowband if the spectral roll-off is below ~4 kHz
   - clipping
   - non-speech events with times (BEATs or PANNs AudioSet classes: laugh, cough, music, vehicle…)

   Turn each feature into a true statement, and into a false one by perturbing it.
3. **Transform-derived** (exact, §2.3). The edit you applied *is* the label.

Report rubric scores **separately by item provenance**. Human-written items are the headline; tool items are a sanity check.

### 2.3 Transform set (exact labels)

Start from clean read clips, then apply one transform per copy, with parameters stored:

| Transform | Parameter sweep | True item it produces |
|---|---|---|
| Telephone band-pass + codec | 300–3400 Hz, with and without codec | Channel: narrowband phone |
| Additive noise | SNR 20 / 10 / 5 / 0 dB, with a named noise type | Other sounds + noise level |
| Room impulse response | small / medium / large room, by RT60 | Space |
| Pitch shift | ±2, ±4 semitones | Pitch (relative to the original) |
| Tempo change | 0.8×, 1.25× | Timing |
| Spliced event | laugh / cough / door at a known time | Event at t ± 0.25 s |
| Clipping | soft / hard | Channel artefact |

This gives **sensitivity curves** (for example, "the captioner first mentions noise at SNR ≤ X dB"). That is strong failure analysis. Use different source clips and seeds from any transform data later used for training.

---

## 3. Captioner evaluation

**Round 0** is `Qwen/Qwen3-Omni-30B-A3B-Captioner` at revision `a2bd106c…`, run through its own audio-only interface with no text prompt (Q20). Record the precision. Round 0 must not run at a lower precision than the final system (Q23).

### 3.1 Rubric score (headline, mirrors the official measure)

1. **Truncate** the description to the first 1,000 characters (Q21), then to 300 words (problem statement).
2. **Validity checks** (score 0 if any fails): the output is not empty, it is English, and it does not address the grader or contain text not about the audio (Q21).
3. **Grader**: `sarvamai/sarvam-m`, greedy, at most 16 new tokens (Q31). It asks one question per item. Official precision is bf16. If that does not fit the cluster, run 4-bit and record it as a deviation, or run on other hardware, and keep a ≤7B proxy grader as a fallback.
4. **Frozen grader prompt** (ours; the official prompt is not published):
   ```
   You check whether an audio description supports a statement.
   Description: {description}
   Statement: {statement}
   {if timed: "For STATED, the description must place it between {start-0.25} s and {end+0.25} s."}
   Answer with exactly one word: STATED if the description clearly commits to the statement,
   CONTRADICTED if it says something incompatible, NOT_MENTIONED otherwise.
   A guess between alternatives is NOT_MENTIONED.
   ```
   Parse the answer strictly. Anything unparseable counts as NOT_MENTIONED, and the parse-failure rate is logged.
5. **Scoring per clip**:
   - True item: +w if stated, 0 if not mentioned, −w if contradicted.
   - False item: −w if stated, 0 otherwise.
   - Clip score = max(0, sum ÷ Σ w over true items).
   - With w = 1 this is the Q31 version; with difficulty weights it is the problem-statement version.
6. **Aggregate**: mean per language, then the mean of hi and ta. Add a 95% bootstrap CI (§6).
7. **Validate the grader (rigour points).** Team members hand-label ~150 (description, item) pairs as stated, not mentioned or contradicted. Report the grader's accuracy and Cohen's κ against them, and do the same for the proxy grader if one is used.

### 3.2 Errors by category (required in the mid-term report)

Break down the rubric score and its components:
- by **axis** (sound / ecology / intent) and by each of the **13 item types**
- **recall** of true items, **contradiction rate**, and **false-assertion rate** (share of false items stated)
- **timed vs global** items; for timed items, the share placed inside the ±250 ms window
- by **clip type**: plain or read clip vs eventful clip; own recording vs Vaani vs transform
- by **language**

### 3.3 Auxiliary automatic metrics (no rubric needed, run on every clip)

| Metric | How |
|---|---|
| Transform detection accuracy | For each transform copy, does the caption state the edit correctly? Plot accuracy against transform strength |
| Feature consistency ("description_score") | An LLM extracts structured claims from the caption (gender, age band, F0 Hz, rate, events with times, channel, noise). Compare with measured features: gender accuracy, \|ΔF0\| in semitones, rate error %, event F1 with a 250 ms collar |
| Hallucination on plain clips | Number of asserted events or sounds not detected on clips with nothing happening |
| Specificity | Share of captions containing numbers (Hz, semitones, ms, s); timed statements per caption; length; share cut by the 1,000-character or 300-word limit |
| Second-family reference | Caption the same clips with Audio Flamingo 3 (or UALM) and score them with the same rubric. This is a *reference point*, not a baseline |

---

## 4. Generator evaluation

Round 0 is the frozen generator submitted at mid-term (Indic Parler-TTS). For each `gen_prompts` item, generate **10 samples** with fixed, recorded seeds (the official protocol generates 10 per description). The output must be **≥ 24 kHz**; check the sample rate and log violations. Loudness normalisation is the only post-processing allowed (Q25).

### 4.1 Intelligibility (official tools)

- ASR: **AI4Bharat IndicConformer** (problem statement) **and** `openai/whisper-large-v3` with the language set (Q31). Report both.
- Text normalisation for both sides: Unicode NFC, lowercase, remove punctuation, collapse whitespace.
- CER against the native-script target text, averaged over descriptions (first sample, or the mean of all 10; state which). Value reported = 1 − CER, floored at 0. Also report WER.
- "CER by skill": break down by description category (plain, emotional, noisy channel, whisper, fast, code-mixed text, numbers in text).

### 4.2 Instruction relevance (proxy for the listener check)

1. **Decompose** each description into atomic checks once, with an LLM, then review by a person and freeze to JSONL. This is the "generation rubric": its items are the statements the description makes.
   ```json
   {"description_id": "...", "check_id": "...", "axis": "...", "item_type": "...",
    "statement": "median pitch near 185 Hz",
    "verifier": "signal | alm",
    "signal_spec": {"feature": "f0_median_hz", "op": "within_semitones", "target": 185, "tol": 2}}
   ```
2. **Verify each check.**
   - **Signal verifier** where measurable: pitch ±2 semitones, rate ±20%, pauses, loudness change in dB, narrowband, reverb, SNR ±5 dB, event at t ± 0.25 s, duration.
   - **ALM verifier** for perceptual checks such as emotion, subtext and addressee. Ask a yes/no question to a *different-family* ALM (Audio Flamingo 3 / UALM).
   - Tolerances are our design choices: freeze them and list them in the report under "Your choices".
3. **Score**: share of checks met, per description, then the mean. Break down by axis and item type.
4. **Calibrate with people.** Three team members, blind, rate a subset (≥ 20 descriptions per language × 1 sample) with the majority deciding, as the organisers do. Report how often the automatic verdict agrees with the human majority, per verifier type.
5. **Round-trip relevance** (secondary): caption the generated audio with the second-family captioner and grade that caption against the check list with the §3.1 grader.

### 4.3 Naturalness (proxy; official is listener CMOS)

- **UTMOSv2** and **DNSMOS**. These are trained mostly on non-Indian speech, so use them as relative signals between rounds, not as absolute claims.
- **FAD** between generated audio and real audio of the same language (`heldout_real`), plus the PSD metrics already planned for the package.
- **Artefact checks**: clipping, long silences, truncation (duration ÷ text length outside bounds), repeated or skipped words (ASR insertions and deletions).
- **In-team CMOS** from round 1 on: blind round_k vs round_0 on the same description, −3 to +3, random order. At round 0 the CMOS against itself is 0 by definition, so at mid-term report only the MOS proxies.

### 4.4 Speaker diversity (problem-statement measure)

- SV model: `microsoft/wavlm-base-plus-sv`, threshold **0.86** (the model card's same-speaker threshold; freeze it).
- For each description, take the samples that **pass relevance** (§4.2, using a fixed pass rule such as ≥ 50% of checks met), then count distinct speakers greedily: a sample is a new speaker if its cosine similarity to every kept representative is below the threshold. Document this rule, since the official clustering rule is not published.
- Report mean count ÷ 10, plus the unconditioned count (all 10 samples), so that relevance failures and diversity failures can be told apart.
- Also report **attribute consistency**: when a description fixes gender or age, do all 10 samples respect it? Diversity must not break the instruction.

### 4.5 Speaker similarity (doubt-session measure)

- For items generated from a real clip (the "simulate" split; §5 round trip), take the WavLM cosine similarity between the output and the target recording, averaged over items.
- The ceiling reference is the similarity between two different real clips of the same speaker, where available.

---

## 5. Whole-pipeline (round-trip) evaluation

Real clip → Indic-Transcribe-Core ASR + captioner → files → generator (text = ASR transcript, description = caption) → audio. Then compare the regenerated audio with the original:

| Stage or metric | Measures |
|---|---|
| ASR stage | CER/WER of Indic-Transcribe-Core against cleaned Vaani or own transcripts; language-ID accuracy (hi vs ta). ASR errors propagate into generator text, so report them |
| Content | CER of the regenerated audio (IndicConformer) against the *gold* transcript and against the ASR transcript; the gap isolates generator errors from ASR errors |
| Attribute transfer | Apply the **original clip's** true rubric items to the regenerated audio, using the signal and ALM verifiers from §4.2: what share still holds? |
| Prosody distance | \|ΔF0 median\| in semitones, F0 range ratio, rate ratio, pause-count difference |
| Voice | WavLM speaker similarity to the original (§4.5) |
| Pipeline health | Missing or invalid outputs per stage (a missing output scores 0), wall time and GPU-hours per stage |

This is the evaluation of "the whole pipeline". It shows where information is lost: in the caption, in the ASR, or in the generator.

---

## 6. Statistics

- **Bootstrap**: 1,000 replicates over items, deterministic hashing as in Appendix A. Draw *j* of replicate *r* selects the item at position `int(sha256(f"{seed}|bootstrap|{measure}|{r}|{j}"), 16) mod n` (items sorted by ID). The CI is the 25th and 975th sorted values (positions 24 and 974). Resample each language separately, then average.
- **Paired comparisons** (round_k vs round_0, loop vs control): identical draws for both systems.
- **Gap-closed level** (Q3) for later rounds: g = (x − b) ÷ (c − b). Levels: 0 if the CI lies fully on the worse side; 1 if the CI includes 0; 2 if clearly better with g < 0.25; 3 if 0.25 ≤ g < 0.5; 4 if g ≥ 0.5. Ceilings are 1.0 (100%) for the rubric score and relevance. Implement this now so that each round prints its level.
- Always report **n_items**. With ~100 clips per language, CIs will be wide: say so in the report rather than over-claiming.

---

## 7. Controlled comparisons (list these as "designed" in the mid-term report)

1. **Round_k vs round_(k−1)** on `heldout_real` and `gen_prompts`: same items, paired bootstrap.
2. **Loop vs control** (Q30): same architecture, code, starting checkpoints, data budget, steps, seeds, checkpoint selection and inference settings, but each model learns from the round-0 version of the other. Scores the rubric and relevance loop gain.
3. **With vs without our specific method** (e.g. confidence filter, transform-derived labels), all else fixed (Q29 recommends this).
4. **Grader sensitivity**: official grader vs proxy grader on the same captions, to show conclusions don't flip.
5. **Auditor check**: rubric score under a second-family captioner on generated audio, to show the generator isn't learning cues only our captioner hears.

---

## 8. `results.csv` (mid-term)

Columns, as in the guidelines: `track, part, language, system, measure, n_items, value`. Use the scoring-table measure names; add proxies as clearly named extra rows and explain them in the report. Example *format* (values are placeholders):

```csv
track,part,language,system,measure,n_items,value
2,captioner,hi,round_0,rubric_score,<n>,<v>
2,captioner,ta,round_0,rubric_score,<n>,<v>
2,generator,hi,round_0,instruction_relevance,<n>,<v>
2,generator,hi,round_0,intelligibility,<n>,<v>
2,generator,hi,round_0,speaker_diversity,<n>,<v>
2,generator,hi,round_0,speaker_similarity,<n>,<v>
2,generator,hi,round_0,naturalness_utmos,<n>,<v>
```

Keep CIs, breakdowns and grader validation in the report (and later in `evaluation.zip`), not in `results.csv`.

---

## 9. Package layout and build order

```
evaluation/
  settings.py            # pydantic settings, no defaults
  splits.py              # hash-based split selection, speaker-disjointness
  manifest.py            # run manifest: commit, checkpoint SHA-256, seeds, decoding
  text_norm.py           # NFC, lowercase, punctuation, whitespace
  features/              # one interface per feature, swappable implementations
    f0.py  vad.py  rate.py  snr_reverb.py  bandwidth.py  events.py  speaker.py
  rubric/
    schema.py            # item and check pydantic models
    build_tool_items.py  # features -> true/false items
    transforms.py        # §2.3 transform set with stored parameters
    grader.py            # §3.1 grader + parser
    score.py             # clip score, weighted and unweighted
  captioner_eval.py      # §3
  generator_eval/
    intelligibility.py  relevance.py  naturalness.py  diversity.py  similarity.py
  roundtrip_eval.py      # §5
  stats.py               # bootstrap, paired CI, gap-closed level
  report.py              # results.csv, breakdown tables, failure-example lists
```

**Build order:**

| Priority | Item | Acceptance check |
|---|---|---|
| P0 | Splits + run manifest | Same salt → same IDs on rerun; manifest has every checkpoint hash |
| P0 | Round-0 captions on `heldout_real` + transform set | One JSONL row per clip, no missing outputs |
| P0 | Round-0 generation, 10 samples × `gen_prompts` | All ≥ 24 kHz; seeds logged |
| P0 | Intelligibility (both ASRs), speaker diversity, UTMOS/DNSMOS | Real-recording CER sanity check is low |
| P0 | Transform detection accuracy + human rubrics on ≥ 15 clips per language + grader | Grader parse-failure rate logged |
| P0 | `results.csv` + per-axis / per-item-type tables + bootstrap CIs | Matches the guideline columns |
| P1 | Grader validation (κ), tool-measured items on all clips | — |
| P1 | Relevance checks (signal + ALM) + human calibration | Agreement reported per verifier |
| P1 | Round-trip evaluation | Per-stage loss table |
| P2 | In-team CMOS, gap-closed levels, control-run comparison | From round 1 on |

**Failure analysis for the report:** for each failure category (e.g. missing timings, wrong gender or age, invented events, ignored channel or space, emotion not rendered, pitch target missed, English words mispronounced, same voice for every sample), give the count, the share and 2–3 example IDs. The revised plan should follow from the biggest categories.
