# Post-Midterm Plan: the Claim-Calibrated Loop

Track 2, 14 Nov to 6 Dec 2026 (22 days). Built from the mid-term report, `results.csv`, `round0.zip`, the doubt-session answers and the cluster manual. Companion to `claude/Track 2 Evaluation Plan`, which stays the spec for measurement.

**Reading taken:** work must be finished by 4 Dec, leaving 5 and 6 Dec as buffer. The freeze date is not announced; if it lands earlier, cut round 2 scope, never the control run.

---

## 1. What went wrong at mid-term

| # | Finding | Evidence | Consequence |
|---|---|---|---|
| 1 | Nothing we measure mirrors the official score | Captioner score is a Gemini claim-precision on 15 clips; no rubric, no sarvam-m grader, no language column in `results.csv` | We cannot tell whether a change helps. Confidence intervals of 0.43 to 0.76 hide any round-to-round gain |
| 2 | Descriptions are written for the wrong budget | The grader reads only the first 1,000 characters (Q21). Our tool descriptions are 282 to 300 words (about 1,800 characters) and round 0 generates up to 1,024 tokens | Roughly half of every description is never scored, and the order of claims is not chosen |
| 3 | The generator cannot read its input | Indic Parler-TTS drops everything past 256 tokens; round-trip relevance 0.50, UTMOS 2.45, CER 13 to 19%, one Tamil item spoken in the wrong language | Timed events and late claims are lost before synthesis |
| 4 | Nothing is trained yet | Report section 4.1 is all future work | Rounds 1 and 2, a matched control and an ablation must fit in 18 working days |
| 5 | The weakness module has no ground truth to stand on | Report section 4, "No robust weakness module" | The loop has no way to decide what to generate next |
| 6 | Novelty so far is framing, not method | Set formulation and tool grounding are sound but are known ideas combined | Novelty (20 points) needs one method explored in depth |

Two things are in our favour. Round 0 is weak on the generator side, so Simulate measures have a lot of room. And the set-of-facts framing in the report is the right foundation for what follows.

---

## 2. How the points work, and what that implies

- Final stage: Results 20, Novelty 20 (whole system for Track 2), Rigour 10, Reproducibility 10, Presentation 10. **Judged criteria outweigh measured results three to one.**
- Results: Describe 7, Simulate 7 (relevance 3, naturalness 2, intelligibility 1, speaker similarity 1), Loop gain 6. Each measure is scored by gap closed, g = (x − b) ÷ (c − b), and g ≥ 0.5 earns the full level.
- Describe and Simulate use frozen round 0 as the baseline, so architecture changes count. Loop gain uses our matched control, so **only what one model learns from the improved version of the other counts**.
- Without the control run, Rigour is capped at 2.5 of 10. Without a with/without test of our own method, the lead-part comparison is unsafe (Q29).

Design rule that follows: every component must have (a) a switch that removes it, for the ablation, and (b) a clear statement of whether it depends on the partner model, for loop gain.

---

## 3. The proposed system

### 3.1 Central idea

Today the two models exchange raw captions and raw audio. In the Claim-Calibrated Loop they exchange **claims with a calibrated probability of being right**, and both models are optimised for the scoring rule itself.

Three observations drive it:

1. **The rubric is a payoff table.** A stated claim earns +1 if it matches a true item, −1 if it contradicts one or matches a false item, and 0 if the rubric does not cover it. So a claim with probability p of being right is worth stating only when p is above about 0.5, and its value is roughly (2p − 1) × the chance its item type appears in the rubric.
2. **The description has a hard budget of 1,000 characters.** Choosing claims is therefore a knapsack problem: maximise expected score per character.
3. **Synthetic speech is the only place where truth is known by construction.** That makes the generator the natural source of calibration data for the captioner, and the captioner the natural source of training labels and of a verifier for the generator. Both links get stronger as the partner improves, which is what loop gain measures.

### 3.2 Captioner: perceive, calibrate, compose

| Stage | What it does | Trained? |
|---|---|---|
| **Perceive** | A student audio-language model emits an over-complete list of typed claims (item type, statement, optional time span). First candidate: MiDashengLM-7B with LoRA (Apache-2.0, built for captioning, fits the 7B limit). The tool extractor adds its measured claims to the same list | LoRA, each round |
| **Calibrate** | For each claim, estimate p(right) from: tool agreement, stability of the claim across K sampled captions and across harmless edits of the clip, and a per-item-type calibration curve fitted on audio whose facts are known (transforms and generator output) | Small calibrator, each round |
| **Compose** | Greedy knapsack: rank claims by expected value per character, drop those under threshold, fill 1,000 characters, realise as plain English. Deterministic, no model call | No |

This gives an ablation ladder where each rung is a controlled comparison:

1. Round-0 caption as is.
2. Round-0 caption, re-composed into 1,000 characters (budget and ordering only).
3. Plus tool claims and calibration.
4. Plus the fine-tuned student.
5. Plus calibration data from generator round k (the loop link).

Rung 2 needs no training and can be measured in week 1. If it already beats round 0, that is our first result and a clean insight for the report.

### 3.3 Generator: description as prefix, ecology by training, verify before output

- **Base model.** Indic Speak: a Llama-3.2-3B speech language model with SNAC audio tokens at 24 kHz. Because it is a language model, the full English description can be given as a free-text prefix of any length, which removes the 256-token limit and the structured-prompt conversion together. Fine-tune with LoRA. Permissive fallback of the same design: snorTTS-Indic-v0 (Apache-2.0, Hindi and Tamil).
- **Ecology by training.** Post-processing toward a target is banned at inference (Q25) but allowed for making training data. So phone band, noise at a stated level, room reverb and spliced laughs or coughs are applied to real speech with exact labels, and the generator learns to produce them itself.
- **Voices.** Train on many pool speakers with no speaker name in the prompt, so the voice is determined by the description and sampling gives different speakers for the 10 samples. This serves speaker diversity and speaker similarity with one mechanism: similarity comes from rendering the described voice accurately.
- **Verify before output (pending a ruling).** For each of the 10 samples, draw n candidates and keep the one whose claims the current captioner and tools confirm best, with an ASR check for wrong language and missing words. This is selection, not editing, and it ties generator quality to captioner quality at inference.

### 3.4 The loop, and a weakness module that needs no ground truth

Each round k:

1. Captioner k−1 and the tools label real pool audio. Only claims with calibrated p above threshold enter the generator's training descriptions.
2. Generator k trains on (description, text) → speech, real audio plus exact-label augmentations.
3. **Weakness = miscalibration and disagreement, per item type.** Measured on unlabelled tuning audio as the gap between the captioner's confidence and its agreement with tools, with the second-family auditor and with itself. No human labels needed, so it solves finding 5.
4. Generator k synthesises speech aimed at the weakest types. A clip is kept only if tools confirm what its description asked for.
5. Captioner k trains on real verified claims plus the kept synthetic clips, and its calibrator is refitted on them.

**Control run (Q30).** Same code, checkpoints, data budget, steps and seeds, but the captioner learns from round-0 generator speech and the generator learns from round-0 captions. Run it alongside every round, not at the end.

### 3.5 What is new, and what we will claim

| Claim | Status |
|---|---|
| Budgeted expected-score captioning: composing a description as a knapsack over calibrated claims under the grader's character limit | New as far as we know; state it as our method, test it with the ladder |
| Using a generator as the source of calibration data for a captioner, per item type | New combination; this is the loop-gain mechanism |
| Weakness detection from calibration gaps instead of labelled errors | New for this setting |
| Description-as-prefix fine-tuning, training-time ecology | Meaningful adaptation of known methods (Parler-TTS style annotation, ParaSpeechCaps, CapSpeech) |

A clearly explained negative result can still score 4 on novelty. If rung 5 shows no gain over the control, report it with the calibration curves that explain why.

---

## 4. Schedule

| Dates | Phase | Exit test |
|---|---|---|
| 14 to 18 Nov | **Phase 0: measure and bring up.** Dev rubrics, sarvam-m grader, official-mirror metrics, round 0 re-scored per language. Ladder rung 2. Data pipeline on a pool subset. Smoke-test fine-tuning for both models | Round 0 has a rubric score with a CI per language; one LoRA step runs for each model on the cluster |
| 18 Nov | **Gate 1** | Student captioner within reach of round 0 on tuning audio? Generator intelligible with a prefix description? If not, switch to the fallbacks in section 6 |
| 19 to 25 Nov | **Round 1** and its control | All four models trained, scored, manifest hashed |
| 26 Nov to 1 Dec | **Round 2** and its control; ablation ladder; Pareto runs | Round 2 complete (the final system must be round 2 or later) |
| 2 to 4 Dec | Freeze, final scoring, `freeze.txt`, `code.zip`, report, `summary.md`, slides | A clean machine reproduces the outputs from the README |
| 5 to 6 Dec | Buffer | |

Cluster facts that shape this: 4 GPU nodes with 2 GPUs each, a 12-hour limit per job, no internet. So every training job must checkpoint and resume, and four jobs (loop and control, two models) want to run in parallel. Keep round 1 small: about 40 hours of pool audio per language plus about 10,000 synthetic clips. More data earns no credit, and GPU-hours of failed runs are counted.

---

## 5. Who does what

Four pairs. Names P1 to P8 are placeholders; assign as you like.

| Pair | Owns | Phase 0 | Rounds 1 and 2 | Final |
|---|---|---|---|---|
| **Measurement** P1, P2 | Evaluation package, statistics, human checks | P1: dev rubrics (40 clips per language, true and false items), grader and its validation. P2: Whisper CER by skill, WavLM similarity, diversity, bootstrap, gap-closed levels, `results.csv` with language | Score every model the day it finishes; in-team CMOS and relevance listening; grader agreement study | Final tables, CIs, error-by-category figures |
| **Data engine** P3, P4 | Pool pipeline, claim ledger, manifests | P3: pool subset, speaker-disjoint splits, feature extractor at scale. P4: transform and splice library with exact labels, training manifest (one row per example) | P3: label and filter real audio each round. P4: targeted synthesis requests, tool verification of synthetic clips, **owner of the control run** | `data_manifest.csv`, manifest hashes |
| **Captioner** P5, P6 | Perceive, calibrate, compose | P5: student model on the cluster, LoRA smoke test. P6: claim schema, composer, ladder rung 2 | P5: train student per round. P6: calibrator, weakness module, rungs 3 to 5 | Pareto of score against size and compute |
| **Generator** P7, P8 | Prefix fine-tuning, ecology, verification | P7: Indic Speak LoRA with free-text prefix, 24 kHz check. P8: Parler description compressor as a same-week fallback; language and CER guard | P7: train per round. P8: verify-before-output, diversity and similarity tuning | Sample pack for the presentation |

Shared duties: P2 is the reproducibility owner (one command per result, pinned environments, clean-machine test). P6 leads the report and `summary.md`. P4 is the Discord contact. A 15-minute stand-up daily, and a go or no-go review at each gate.

---

## 6. Risks and fallbacks

| Risk | Sign | Fallback |
|---|---|---|
| Student captioner worse than the 30B round 0 | Gate 1 rubric score clearly lower | Keep round 0 as the perceiver and apply calibrate and compose on top; rungs 2 and 3 still stand, and the loop link moves to the calibrator |
| Indic Speak does not fine-tune cleanly, or licence concerns | No intelligible speech after two days | snorTTS-Indic-v0, or stay on Indic Parler-TTS with a trained description compressor that fits the 256 tokens |
| Tools or reranking ruled out of the frozen system | Discord answer | Tools become training-only label sources and the student absorbs them; drop verify-before-output and report it as an ablation |
| Cluster queue | Jobs waiting over 6 hours | Halve round data, keep both rounds and the control |
| Loop shows no gain over control | Paired CI includes zero | Report as a negative result with per-type calibration evidence |

---

## 7. Ask on Discord now

1. Freeze, evaluation-window and end-term dates.
2. May the frozen captioner include signal-processing tools and a deterministic composer around the model?
3. May the generator pick each of its 10 samples from several candidates using our own captioner and an ASR?
4. Which is binding: speaker similarity (Q31) or speaker diversity (problem statement); Whisper or IndicConformer; weighted or unweighted rubric?
5. Is the 300-word limit still in force alongside the 1,000-character limit?
6. Does a simulate item give the generator anything besides text and description?

## 8. Not yet verified

- The GPU model and memory on the cluster (the round-0 README still has a placeholder).
- That sarvam-m fits in bf16 on our nodes.
- That Indic Speak accepts a long free-text prefix after LoRA; this is the assumption Gate 1 tests.
- Licence terms of each extra speech corpus we add beyond the pool.

## Sources

- [Indic Speak, community conversion card](https://huggingface.co/iniquitous/indic-speak-mlx-bf16)
- [snorTTS-Indic-v0](https://huggingface.co/snorbyte/snorTTS-Indic-v0/blob/main/README.md)
- [MiDashengLM-7B](https://huggingface.co/mispeech/midashenglm-7b-0804-fp32)
- [IndicF5](https://huggingface.co/ai4bharat/IndicF5)
- [ParaSpeechCaps: Scaling Rich Style-Prompted Text-to-Speech Datasets](https://arxiv.org/abs/2503.04713v1)
- [CapSpeech](https://fugumt.com/fugumt/paper_check/2506.02863v1_enmode)
