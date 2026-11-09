# How features-extractor works

One Hindi/Tamil clip in → measured acoustic features out (a **global** part about the whole
clip and a **local** part with timed items), plus a tool description per clip: evidence for
writing captions in the style of the Rime Track 2 examples. Every neural model is **frozen**
(fixed revision, checked by hash, loaded offline), so results are reproducible and it runs
on a Madhava GPU node with no internet.

## 1. Layout

```
DIR/tool_descriptions.json                  copied in at the end of a run
features-extractor/
├── README.md  EXPLANATION.md
├── environment.yml        conda env with exact library versions
├── config.yaml            the one config (models, thresholds, checks)
├── copy_descriptions.py   copies a tool_descriptions.json into any folder (run calls it)
├── models/                created by fetch-models; models.lock.json = revision + sha256 of every file
├── slurm/extract.sbatch   GPU job, sharded with a job array
└── src/                   the package; run from features-extractor/ as `python -m src` (no install)
    ├── __main__.py        commands: fetch-models | run | merge | describe | clip
    ├── pipeline.py        one clip → features dict (global / local) + checks
    ├── dsp.py             every acoustic measurement (no models)
    ├── models.py          frozen models: lock, fetch, VAD / language / sound events
    ├── describe.py        features → tool description text
    └── audio.py           load, truncation check, 16 kHz mono
features-extractor-results/<DIR name>/
└── <run>/                             one per run, never overwritten (start time, or job<ID> on SLURM)
    ├── features/<clip>.json           all features of one clip (indented)
    ├── tool_descriptions.json         [{"audio_path", "tool_description"}, ...] (indented)
    ├── tool_descriptions.rejected.json  clips that failed a rejecting check, and why
    └── run.json                       folder, config + hash, version, device, started/finished, status, counts
tests/                                 pytest; works next to or inside features-extractor/
```

`audio_path` is relative to DIR's parent: running on `/data/hi` gives `"hi/clip_001.wav"`.

## 2. Models (all frozen)

| step | model | runs on | gives |
|---|---|---|---|
| speech / pauses | **Silero VAD v5.1.2** (TorchScript file from the v5.1.2 release tag) | CPU | phrases and pauses |
| language | **openai/whisper-small**, language head only, scored over hi/ta | GPU | language + confidence |
| sound events | **MIT/ast-finetuned-audioset-10-10-0.4593** (AudioSet, 527 classes) | GPU | laughter, cough, crying, music… with times; what else is heard outside speech |
| syllables | **facebook/wav2vec2-xlsr-53-espeak-cv-ft** (multilingual phoneme recogniser, ~300M) | GPU | each recognised vowel = one syllable, with its time (no words, no transcript) |
| speakers | **microsoft/wavlm-base-plus-sv** (x-vectors, ~94M) | GPU | how many voices, and where the voice changes |
| pitch | pYIN (librosa), an algorithm | CPU | pitch contour |
| everything else | numpy / scipy | CPU | levels, SNR, bandwidth, hum, noise steadiness, reverberation, phrase splitting, phrase pitch / loudness / edges, changes, voice numbers, event pulses, coupling, arousal; syllables without the phoneme model |
| text (optional) | `script.py`: Devanagari / Tamil rules | CPU | syllables per word, fillers, word times laid over the nuclei |

**Licences** (all permissive, none gated, no agreement to accept): Silero VAD MIT, Whisper MIT,
AST BSD-3-Clause, wav2vec2-XLSR Apache-2.0, WavLM-SV MIT (Microsoft UniSpeech). The voice
measures below are the usual job of Praat / parselmouth, which is GPL-3.0, so they are
implemented here in numpy instead.

**Freezing.** `fetch-models` downloads each model into `models/<name>/`. The first time it
records the exact Hugging Face commit and the sha256 of every file in `models/models.lock.json`;
later fetches download that same commit and fail if any byte differs. At start-up the extractor
re-hashes the files against the lock (`verify_models: true`) and refuses anything else; loading
never uses the network. Keep `models.lock.json` in git: it *is* the model version. Library
versions are pinned in `environment.yml`. Each features JSON records the package version, a
hash of the config, the model revisions and the device. (GPU floating point can differ in the
last digits between GPU types.)

No speech recognition: no measured feature needs the transcript, and Indic-Transcribe runs as
its own stage of the pipeline.

**One loading trap.** PyTorch 2.1 renamed weight-norm parameters, so wav2vec2 and WavLM
checkpoints (saved before that) load their positional-convolution weights as random numbers;
transformers reports it as "newly initialized ... You should probably TRAIN this model" and the
output is quietly ruined. `models.fix_weight_norm` copies those tensors across by name after
loading, and `FX_MODELS=1 pytest` asserts that it copied some. If you ever swap in another
wav2vec2-family checkpoint, keep that call.

**What is stated: the 65 % rule.** A *claim* goes in the description only if it is estimated to
be right at least 65 % of the time on real recordings: timing, pitch, loudness, phrase edges,
recording quality, the room bucket, speaker count and change times, the sound-event labels (with
their probability) and, with a transcript, syllable counts and fillers.

**What is measured but not claimed.** Voice numbers (HNR, CPP, H1–H2, jitter, shimmer), per-phrase
arousal and per-event coupling go into the JSON, and the description prints the voice numbers once,
labelled as measurements. They are evidence for the audio-language model, which keeps every
judgement: age, gender, dialect, phonation labels (breathy, creaky, falsetto, whisper), who the
speakers are, and the whole intent axis (addressee, subtext, what changes).

These estimates are priors, not measurements: check 30–50 real clips by ear and drop any claim
type that turns out below 65 %.

## 3. Running

- `run DIR` processes every `.wav` under DIR as a **new run**: a new folder named by its start
  time (`--run-id NAME` to choose the name). Old runs are never touched. At the end the run's
  `tool_descriptions.json` is copied into DIR by `copy_descriptions.py`, so DIR holds the latest one.
- `runs DIR` lists the runs with status (`finished`, `stopped`, `running`) and counts.
- A clip that raises is recorded with `status: "failed"` and its error, and the run carries on, so
  one odd file cannot lose a long job; failed clips are listed in `tool_descriptions.rejected.json`.
- `--resume RUN` continues that run (or `latest`): clips whose features JSON exists and was made
  with the same config and version are reused, the last finished clip is redone (it may have been
  cut off), the rest are processed. Unreadable files and clips made with another config/version
  are redone.
- Files are written to a temp file and renamed, so a killed job never leaves a half-written
  file. When a run is stopped (Ctrl-C, or SLURM's SIGTERM at the time limit) its outputs are
  still written for the clips done so far, `run.json` says `stopped`, and the console prints the
  exact `--resume` command.
- `--shard I/N` processes every N-th file starting at I; all shards of a job use the same run
  (`slurm/extract.sbatch` passes `--run-id job<JOBID>`) and write `.part-I-of-N` files.
  `merge DIR --run job<JOBID>` then builds the run's `tool_descriptions.json` and `run.json` and
  copies the descriptions into DIR. `describe DIR --run RUN` does the same rebuild, e.g. after
  editing `describe.py` (no models needed).
- The console shows each clip's time, elapsed time and time left, and at the end the total,
  seconds per clip (over the clips that finished), speed vs real time and each step's share.
- All four torch models share the process's thread pool. `torch.set_num_threads` is global, so
  `vad.threads` is left unset by default; setting it to 1 (as Silero's own example does) makes
  every other model single-threaded too.

## 4. Features JSON

```jsonc
{
 "clip":   {"path", "duration_s", "sample_rate", "channels", "format", "narrowband"},
 "status": "ok" | "rejected",
 "checks": [{"check": "min_snr_db", "message": "SNR 8.1 dB", "reject": false}],
 "global": {
   "levels":    {"peak_dbfs", "rms_dbfs", "clipping_pct", "severe_clipping", "silence_pct"},
   "recording": {"snr_db", "noise_floor_dbfs", "snr_method", "bandwidth": {"low_hz", "high_hz", "band_limited"},
                 "hum": {"hz", "harmonics", "prominence_db"} | null, "noise_variation_db",
                 "space": {"decay_s", "bucket", "tails"} | null},
   "language":  {"code", "confidence", "scores": {"hi", "ta"}, "source"},
   "pitch":     {"median_hz", "p10_hz", "p90_hz", "range_st"},
   "speech":    {"speech_s", "speech_ratio", "phrases", "pauses", "syllables", "syllables_per_s",
                 "phrase_rhythm": {"period_s", "cv"} | null},
   "voice":     {"hnr_db", "cpp_db", "h1h2_db", "jitter_pct", "shimmer_db", "voiced_pct"},
   "speakers":  {"count", "per_phrase", "change_times", "threshold", "source"} | null,
   "other_sounds": [{"class", "p"}]        // AST's top AudioSet classes outside speech
 },
 "local": {
   "phrases": [{"n", "start", "end", "syllables", "syllables_per_s", "rate_first_half", "rate_second_half",
                "pitch": {"median_hz", "p10_hz", "p90_hz", "range_st", "trend_st", "peak_hz", "peak_s"},
                "pitch_confirmed", "final_pitch_st", "final_pitch_confirmed",
                "level_dbfs", "level_vs_others_db", "onset_ms", "offset_ms",
                "voice": {... as global}, "arousal": {"pitch_above_clip_st", "range_st",
                "level_above_clip_db", "rate_ratio"}, "speaker"}],
   "changes": [{"t", "phrase", "pitch_st", "from_hz", "to_hz", "check_st", "confirmed", "reason",
                "level_db", "level_confirmed", "level_reason", "transient"}],   // only what applies
   "pauses":  [{"start", "end", "ms"}],
   "events":  [{"start", "end", "label", "p", "p_max", "speech_overlap", "pulses",
                "coupling": {"level_change_db", "brightness_change_db"}}],
   "words":   {"source", "method", "syllables", "syllables_per_s", "words": [{"text", "syllables",
               "filler", "start", "end"}], "fillers": [...], "drawn_out": {...} | null} | null
 },
 "run": {"version", "config_sha", "device", "models", "model_revisions", "seconds"}
}
```

How each value is measured (rubric item types in brackets; est. = estimated share of correct claims):
- **phrases / pauses** [Timing, est. ~90 %]: Silero speech probability every 32 ms; speech starts at
  ≥ 0.50 and ends below 0.35, padded 50 ms; phrases closer than 200 ms are joined, shorter than
  150 ms dropped. Silero sometimes runs through short stops, so each phrase is then split where it
  is silent for ≥ 200 ms: 25 dB below the phrase's speech level, or within 6 dB of the clip's pause
  level (noisy recordings, where a pause is only ~15–20 dB down). Shorter dips (consonant closures,
  e.g. Tamil double consonants, ~50–180 ms) stay inside.
- **syllables** [Timing]: with `syllables: {impl: phonemes}` (default), every vowel the phoneme
  recogniser hears is one syllable, timed by its 20 ms CTC frame; rate = syllables in phrases /
  speech time. Without it (`envelope`): peaks of the 300–3000 Hz envelope ≥ 80 ms apart, which on
  two Tamil clips with transcripts gave ~5.4/s against the true ~6.7/s (about 15–20 % low). Check
  the phoneme model on clips whose transcript you know before relying on it (see README).
- **rate_first_half / rate_second_half / rate_change** [Timing]: each half's rate from the median
  spacing of its syllables (≥ 3 spacings per half). Speeding up / slowing down is claimed only when
  the phoneme-model halves move ≥ 30 % AND the independent envelope detector agrees on the
  direction by ≥ 20 % (on a real clip, the 20 ms CTC frames alone produced a false "speeds up"
  that the envelope read as flat); with a single detector, ≥ 50 %.
- **phrase_rhythm** [Timing, ~80 %]: spacing of phrase starts (≥ 3 phrases); stated as a regular
  period when its variation is ≤ 15 %.
- **pitch** [Pitch, ~80–90 %]: pYIN 70–450 Hz, 10 ms hop, speech only. `range_st` = 10th→90th
  percentile in semitones; `trend_st` = median of the last third minus the first third; `peak` =
  95th percentile and when it occurs (stated when ≥ 4 semitones above the median). The clip-level
  and each phrase's median are also computed on the second, independent contour (numpy
  autocorrelation): the text states a phrase's pitch only when the two medians agree within
  1.5 semitones (`pitch_confirmed`; `confirmed` inside the clip-level `pitch`). On a 30-clip set
  this suppressed the 5 phrase medians and 1 clip median a manual check had found wrong (pYIN
  sub-octave errors on two low, creaky voices) while keeping 103 correct ones.
- **final_pitch_st** [Pitch, ~75 %]: median of the phrase's last 300 ms minus the rest; stated at
  ≥ 2 semitones, and only when the second contour agrees on the direction
  (`final_pitch_confirmed`) — on the same set that removed the 2 wrong "ends higher/lower" claims.
- **level_vs_others_db** [Change, ~85 %]: a phrase's median frame level minus the median over all
  phrases; stated as louder / softer at ≥ 3 dB.
- **onset_ms / offset_ms** [Morphology, ~65–70 %]: from a fine level envelope (10 ms windows,
  2.5 ms hop): rise from 20 dB below the phrase peak (or 3 dB above the noise) to 6 dB below it,
  and the fall back. Stated only at the extremes (≤ 20 ms abrupt / ≥ 120 ms gradual start;
  ≤ 20 ms cut-off / ≥ 150 ms fade) and only when SNR ≥ 20 dB.
- **changes** [Pitch, Change, ~75–80 %]: sustained steps inside a phrase, never across a pause.
  The windows before and after must both lie inside the phrase and differ by at least: pitch 6
  semitones (medians of 150 ms; at most the 3 largest per phrase, so ordinary intonation does not
  flood the text), level 6 dB (80th percentile of 300 ms, which follows syllable peaks; silent frames
  are left out and both windows must be ≥ 80 % speech, so a stop is never read as "softer, then
  louder"). Steps within 150 ms are one event. Four guards then decide what reaches the text; a
  step that fails one stays in the JSON with `confirmed: false` (pitch), or `transient: true` /
  `level_confirmed: false` (level), and a `reason`:
  * 11–13 semitones: pitch trackers often jump exactly an octave by mistake;
  * **a second, independent pitch contour** (numpy autocorrelation) must agree on the jump's
    direction and reach 40 % of its size, over the same before/after windows. On a 30-clip Tamil
    set this removed ~3 % of jumps, including two where pYIN read a drop while two other methods
    saw a rise;
  * **a loudness step with an opposite one within `transient_s` (0.6 s)** is one brief burst or
    dip, not two sustained changes — 23 % of all loudness steps on that set;
  * **the median level around the step must move ≥ 3 dB in the same direction** (the step itself is
    found on the 80th percentile, which a single burst can drag) — removed the 3 remaining wrong
    loudness steps on that set, keeping 33 correct ones.
  Thresholds: `changes:` in `config.yaml`.
- **SNR** [Channel, ~85 %]: speech power (frames fully inside phrases) / noise power (frames fully
  outside). Speech frames contain noise too, so a true 5 dB SNR reads about 6.2 dB.
- **bandwidth** [Channel, ~90 %]: long-term spectrum of the speech **at the file's own sample rate**
  in 125 Hz bands; `high_hz` = highest band within 45 dB of the strongest; `band_limited` (≤ 4.5 kHz)
  catches phone / codec audio saved at a higher rate.
- **hum** [Channel, ~85–90 %]: narrow peaks at 50 or 60 Hz and harmonics 1–4 standing ≥ 10 dB above
  the spectrum around them, measured on the pauses (≥ 1.5 s of them, else the whole clip with 3
  harmonics required, because speech harmonics can mimic peaks).
- **noise_variation_db** [Channel, ~75 %]: 90th minus 10th percentile of the frame level outside
  speech; ≤ 6 dB is stated as steady. "Fluctuating" is not stated: pauses also hold breaths and other
  short sounds, which are not background noise.
- **events** [Event, Other sounds, ~55–80 % by label]: AST in two passes to save compute: one pass
  per 10.24 s chunk; only chunks where a label reaches half its threshold get 0.96 s windows (hop
  0.48 s). A label's score is the max over its AudioSet classes; windows above the label's threshold
  become an event, edges snapped to the energy, same-label events < 250 ms apart joined. Thresholds
  are set for AST (0.4–0.5; breathing 0.5, since lower values fire on breathy speech). Stated with
  the probability. `speech_overlap` = share inside phrases.
- **space** [Space, ~65 % as 3 buckets]: after each phrase end the level is followed until it is
  within 3 dB of the pause floor; the straight-line slope gives a decay time (seconds for 60 dB).
  The median over the phrase ends becomes dry / some reverberation / reverberant, and nothing
  finer — a blind estimate from speech cannot support more. `null` when no tail is long or deep
  enough, which is the usual case for close microphones in quiet rooms (all six test clips).
  Only a non-dry bucket is stated.
- **voice numbers** [Source, numbers only]: per voiced frame and per phrase. HNR = window-corrected
  autocorrelation peak (Boersma 1993); CPP = cepstral peak height above the cepstrum's regression
  line; H1–H2 = first minus second harmonic, no formant correction; jitter and shimmer = mean
  cycle-to-cycle change in period (%) and in peak amplitude (dB), from the pitch contour. Noise,
  the vowel and the microphone move all of these, and our period marks are not Praat's, so compare
  phrases of one clip, never against published norms.
- **pulses** [Event]: peaks of a sound event's own 20 ms envelope ("three laugh pulses"); `null`
  for events under 0.2 s.
- **coupling** [Coupling]: level and brightness in the second before vs the second after each event
  starts. Stated only when the level moves ≥ 5 dB; whether it is a reaction is for the ALM, and the
  event's own time is only accurate to about one window.
- **arousal** [Subtext / Change evidence]: per phrase, pitch above the clip median (semitones),
  pitch range, level above the clip median, rate ratio. Never stated; it is the ALM's evidence.
- **speakers** [Speaker]: one WavLM x-vector per phrase, average-link clustering on cosine
  similarity. One expressive speaker moves a lot in this space (at the model card's verification
  threshold 0.86, three single-speaker test clips came out as 2-3 "voices"), so the claim is
  two-sided: phrases merge at 0.5, and only a split that survives that merge is stated
  ("at least N clearly different voices"); "one voice throughout" is stated only when every pair
  is >= 0.8 similar; anything in between is left unstated (`confident: null`). Phrases under 1 s
  take the nearest speaker's label but never create one, and a phrase under `min_embed_s` (0.5 s) is
  widened into the surrounding audio before embedding, because WavLM's TDNN head needs ~0.3 s of
  frames and fails outright below that. Who the voices are stays with the ALM.
- **words** [Timing, Event; needs a transcript]: `script.py` counts syllables per word from the
  Devanagari / Tamil script itself (a consonant is one syllable unless the virama / pulli follows;
  Hindi drops a word-final inherent "a"), which is exact and needs no model. The words' syllable
  counts are then laid over the detected nuclei in order, and times are kept only when the text's
  total and the nuclei agree within 30 % (`method: syllable_proportional`), else counts only. The
  transcript's syllable rate replaces the audio-only one in the description, because it is exact.
  Fillers are words that exactly match a short per-language list of hesitation sounds.
- **other_sounds** [Other sounds, ~60–65 %]: one AST pass over the audio outside speech (joined, up
  to 10.24 s, needs ≥ 1 s); speech-like classes and silence left out; top 3 with p ≥ 0.15.

## 5. Tool description

A global paragraph, then one line per phrase, pause, change and sound event in time order:

```
Tamil, 9.7 s. Recording: 16 kHz mono file, SNR about 17 dB (noticeable background noise), noise
floor -39 dBFS, speech energy up to about 7.4 kHz, steady background noise, peak -6.1 dBFS, no
clipping. Other sounds (timed below): laughter. Outside speech the sound-event model also hears:
Fan (0.31). Voice: median pitch 124 Hz, usual range 98–160 Hz (8 semitones, 10th–90th percentile).
Speech: 7.1 s in 6 phrases separated by 5 pauses (210–410 ms); about 3.7 syllables per second overall.
0.52–1.90 s, phrase 1: about 5 syllables (3.6 per second); pitch median 130 Hz, range 4 semitones,
falls about 3 semitones start to end; ends 3 semitones lower than the rest of the phrase; 3 dB louder
than the other phrases.
1.90–2.31 s, pause of 410 ms.
3.12 s, within phrase 2: pitch jumps up 6 semitones (122 → 172 Hz); 7 dB louder.
5.60–6.40 s, laughter (outside speech, p 0.62).
```
(illustrative numbers)

A line appears only when something is notable (thresholds above), so plain clips stay short.
No transcript and nothing about voice quality, speakers, room or intent (see the 65 % rule).

## 6. Config, checks, extending

`config.yaml` is commented. Swappable parts (`impl:`): `vad` silero | energy, `language` whisper |
folder (hi/ ta/ in the path), `pitch.method` pyin | autocorr, `events` ast or delete the section.
Checks write a message into the clip's JSON; those under `checks.reject` also keep the clip out of
`tool_descriptions.json` (default: too short, < 0.5 s of speech, language not hi/ta). An unreadable
or truncated file is always rejected.

A new model: a class in `models.py` with the same method (`probabilities`, `identify` or `detect`),
one entry in the `VAD` / `LANGUAGE` / `EVENTS` dict and, if it has weights, one line in `SOURCES`
(`fetch-models` then freezes it). A new measurement: a function in `dsp.py`, one line in
`pipeline.py`, one line in `describe.py` if it belongs in the text.

## 7. Running on Madhava

Madhava has no internet, so the env and the frozen models are prepared on the laptop and copied
(README has the commands). In `slurm/extract.sbatch` set `--partition` to the GPU partition
(and `--account` if your group needs one). Each array task (`--array=0-3`) is one shard on one GPU;
all shards write into the run `job<JOBID>`; `merge --run job<JOBID>` joins them. Logs: `logs/`.

## 8. Tests

`pytest tests` (118 tests, ~3 s, no model files) uses synthetic signals with known answers: tones at
known pitch, noise at an exact SNR, syllable trains at a known rate, phrases at known times, pitch and
loudness steps, octave jumps, phone-band noise, final falls, tempo changes, abrupt vs gradual onsets,
mains hum, steady vs fluctuating noise, a convolved reverberant room, laugh-like pulse trains,
hand-counted Tamil and Hindi syllables, speaker clustering, truncated files, killed and resumed runs.
`FX_MODELS=1 pytest tests` also runs Silero, Whisper, AST, the phoneme model and WavLM from `models/`.
