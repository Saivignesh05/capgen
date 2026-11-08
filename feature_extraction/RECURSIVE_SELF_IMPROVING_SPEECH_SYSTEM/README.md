# Recursive Self-Improving Speech System - Vaani Audio Dataset

This repository contains local offline copies of 200 audio clips (100 Hindi, 100 Tamil) downloaded from the [ARTPARK-IISc/Vaani](https://huggingface.co/datasets/ARTPARK-IISc/Vaani) dataset on Hugging Face.

## Quick start

Use Python 3.10 or newer. The project is CPU-first; a CUDA installation is not
required.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the bundled examples from this project directory:

```bash
python run_single_demo.py
python verify_offline.py
```

The first pipeline run may download the Silero VAD model if it is not already
present in `models/`. YAMNet requires `models/yamnet.onnx`; if it is absent,
sound-event results are reduced or unavailable. Model files are intentionally
ignored by Git because they are large binaries. Obtain them from their
official sources and verify checksums before use.

The downloaded Vaani audio and metadata are also ignored by Git. On a fresh
clone, download the sample dataset explicitly when you have permission to use
it:

```bash
python download_vaani.py
```

This requires network access and may require a Hugging Face login/token. The
repository remains usable with your own audio files without downloading Vaani.

## Analyze your own audio

`run_single_demo.py` accepts one or more audio paths. WAV is recommended, but
formats supported by the installed `soundfile`/`libsndfile` build may also
work. Paths can be absolute or relative to this project directory:

```bash
python run_single_demo.py "C:\Users\you\Music\sample.wav"
python run_single_demo.py "C:\audio\one.wav" "C:\audio\two.wav" --no-plots
```

Results are written under `outputs/json/`, `outputs/plots/`, and
`outputs/reports/`; human-review templates are written under
`data/ground_truth/`. The pipeline converts audio to a temporary 16 kHz mono
analysis copy but does not overwrite the original file. For reliable results,
use clear recordings and avoid clipping; stereo and other sample rates are
accepted and normalized for analysis.

## Requirements and cautions

- Keep private recordings outside the repository or under the ignored
  `data/user_audio/` directory. Do not commit recordings, transcripts,
  speaker identifiers, access tokens, or API responses.
- The pipeline can use `HF_TOKEN`, `HUGGINGFACE_TOKEN`, or `GROQ_API_KEY` for
  optional cloud transcription. Do not put these values in source files or
  README examples; configure them only in the process environment.
- Automatic VAD, YAMNet events, voice-quality labels, and acoustic changes are
  measurements or candidates, not proof of emotion, identity, gender, health,
  intent, or other sensitive attributes.
- Listen to the original recording and manually verify candidate events before
  moving anything into `ground_truth_events` or using evaluation scores.
- Respect consent, licensing, privacy, and applicable data-protection rules
  for every recording you analyze. Do not use the output for high-impact
  decisions without independent validation and human oversight.

## Directory Structure

```text
data/
    vaani/
        hindi/
            audio/
                hindi_001.wav
                hindi_002.wav
                ...
                hindi_100.wav
        tamil/
            audio/
                tamil_001.wav
                tamil_002.wav
                ...
                tamil_100.wav
        metadata/
            hindi.csv
            tamil.csv
```

## Dataset Specifications

- **Dataset Source:** `ARTPARK-IISc/Vaani` (Project Vaani by IISc & ARTPARK)
- **Languages:** 
  - Hindi (`source_configuration: "Hindi"`)
  - Tamil (`source_configuration: "Tamil"`)
- **Total Files:** 200 WAV audio files (100 Hindi, 100 Tamil)
- **Audio Format:** Linear PCM WAV
- **Sampling Rate:** 16,000 Hz (16 kHz mono)
- **Audio Integrity:** Untouched original samples (no resampling, normalization, or transcoding)
- **Offline Ready:** All audio files and metadata CSVs are permanently stored locally and can be accessed without internet connectivity.

## Metadata Schema

The CSV files located in `data/vaani/metadata/hindi.csv` and `data/vaani/metadata/tamil.csv` include the following columns:

| Column | Description |
| :--- | :--- |
| `clip_id` | Original source filename / audio clip identifier from the dataset |
| `language` | Language label (`Hindi` / `Tamil`) |
| `duration` | Audio duration in seconds |
| `transcript` | Annotation/transcription text (empty if not transcribed) |
| `speakerID` | Unique speaker identifier |
| `state` | Indian state where the recording was conducted |
| `district` | Administrative district of recording |
| `source_configuration` | Hugging Face dataset configuration (`Hindi` / `Tamil`) |
| `original_sample_rate`| Original audio sampling rate (`16000`) |
| `audio_file_path` | Relative path to local audio file |

## Scripts

### 1. Download Script (`download_vaani.py`)
Downloads and saves 100 Hindi and 100 Tamil audio samples in streaming mode without loading the complete 7.21 TB repository into local cache. Supports resumability and deduplication.

```bash
python download_vaani.py
```

### 2. Offline Verification Script (`verify_offline.py`)
Performs thorough offline validation:
- Verifies that all 200 audio files exist and are non-empty.
- Confirms readability and decodability with `soundfile`.
- Validates metadata consistency between CSV records and audio files.
- Generates `VERIFICATION_REPORT.md`.

```bash
python verify_offline.py
```

### Ground-truth event representation

The analysis JSON keeps frame-level measurements for diagnostics, but acoustic
change detections are consolidated before they enter the timeline. Overlapping
detections of the same type are merged, and co-occurring pitch, loudness, and
spectral changes are represented as one `delivery_change` event with component
evidence. The output also records `raw_acoustic_candidate_count` and
`consolidated_acoustic_event_count` so the reduction is auditable.

Generated ground-truth templates are marked `unverified` and must be manually
reviewed before they are used for scoring.

Each analysis JSON also contains a `description` object. It combines the
matching Vaani metadata record (speaker label, language, state, district, and
transcript availability) with measured global properties and consolidated local
events. Dataset labels are reported as metadata; age, gender, dialect,
addressee, subtext, and other subjective Rime items are not inferred without
independent evidence.

The pipeline uses the local CPU YAMNet ONNX model when
`onnxruntime` and `models/yamnet.onnx` are available. If the model is
unavailable, heuristic spectral detection is retained only as candidate
evidence. Even YAMNet outputs remain candidates until human audio audit;
the demo creates an `unverified` audit template and does not score it
automatically.

The provisioned YAMNet ONNX file is approximately 15 MB and runs through
`onnxruntime` on CPU. It is an event detector, not a proof of an event:
YAMNet predictions are placed in `sound_event_candidates` and the audit
`review_queue`; only manually confirmed items can enter `ground_truth_events`.

The description also includes a CPU-only `voice_profile` derived from existing
acoustic measurements. It reports likelihoods for modal-like, breathy-like,
hoarse/rough-like, and strained-like voice quality. Gender is reported as
unknown unless independently supplied in metadata; pitch alone is not used as
a gender classifier.

## Current analysis pipeline

The pipeline is CPU-first and processes each audio clip independently:

```text
WAV audio
  -> preprocessing and Vaani metadata
  -> Silero VAD
  -> global acoustic analysis
  -> YAMNet sound-event candidates
  -> local acoustic-change detection
  -> unified timeline
  -> Rime-aligned description
  -> JSON export
  -> manual ground-truth audit and evaluation
```

The main orchestration is in `pipeline.py`.

### 1. Preprocessing and metadata

`src/preprocessing.py` reads the original WAV without changing it, records
duration, sample rate, channels, clipping, RMS, and silence, and creates a
16-kHz mono analysis copy when necessary. It also matches the clip against the
local Vaani CSV files and attaches language, speaker ID, state, district,
source configuration, and transcript availability.

Vaani metadata is dataset metadata. It is not inferred from the waveform.

### 2. Speech and silence detection

`src/vad_segmentation.py` uses the local `models/silero_vad.jit` model. It
produces speech segments, pauses, silence intervals, speech duration, speech
ratio, timestamps, and VAD confidence. It answers where speech is likely
present; it does not identify gender, emotion, words, or speaker identity.

### 3. Global information

`src/global_analysis.py` summarizes the complete clip using Librosa, NumPy,
SciPy, and pYIN.

Global signal measurements include:

- Peak amplitude and RMS statistics
- Zero-crossing rate
- Spectral centroid
- Spectral bandwidth
- Spectral rolloff
- Spectral flatness

Global voice measurements include:

- Median, minimum, maximum, and percentile F0
- Voiced-frame percentage
- Harmonic-to-noise ratio
- Estimated syllables per second

Global recording measurements include:

- Estimated SNR
- Background noise floor
- Clipping indicators
- Conservative reverberation evidence

These are measurements over the recording. They are not automatically
semantic claims such as emotion, anger, or gender.

### 4. Voice-quality profile

`src/voice_profile.py` uses existing HNR, voiced ratio, spectral flatness,
spectral centroid, and spectral bandwidth measurements. It reports
probabilistic acoustic likelihoods for:

- `modal_like`
- `breathy_like`
- `hoarse_or_rough_like`
- `strained_like`

These are observed acoustic likelihoods, not clinical or perceptual
diagnoses. Gender currently remains `unknown`. F0 alone is not used to infer
gender.

### 5. YAMNet local sound candidates

`src/sound_event_detection.py` uses the local approximately 15 MB
`models/yamnet.onnx` model through CPU ONNX Runtime. It evaluates overlapping
short windows and maps AudioSet classes into candidate categories such as:

- Cough, throat clearing, laughter, sigh, breathing, gasp, and crying
- Music and singing
- Animals and birds
- Knocks, doors, impacts, clicks, and clatter
- Background speech and whispering

Class-specific thresholds, minimum durations, and boundary refinement are
applied. Each result includes timestamps, probability, maximum probability,
timestamp uncertainty, detection method, and `unverified` status.

YAMNet results are candidates, not ground truth. For example, a broad
`background_speech` result may actually be the primary speaker and must be
audited before being interpreted as a second speaker.

### 6. Local acoustic changes

`src/acoustic_changes.py` detects measurable changes inside speech using:

- F0
- RMS loudness
- Spectral centroid
- Zero-crossing rate
- Spectral tilt
- Voicing state

It can produce pitch increases/decreases, loudness increases/decreases,
spectral changes, and grouped `delivery_change` events. Nearby same-type
detections and co-occurring feature changes are merged. Thresholds, merge
gaps, maximum duration, and other controls are in `src/config.py`.

The output preserves both `raw_acoustic_candidate_count` and
`consolidated_acoustic_event_count`. These events describe acoustic changes;
they do not automatically mean emotion, intention, shouting, or a semantic
event.

### 7. Local versus global output

The JSON separates recording-wide facts from timed observations:

| Output | Meaning |
|---|---|
| `global_info` | Measurements summarizing the whole recording |
| `speech_segments` | Timed speech intervals from VAD |
| `sound_event_candidates` | Timed, unverified YAMNet predictions |
| `acoustic_changes` | Timed measurable delivery transitions |
| `local_events` | Detector-promoted local sound events |
| `timeline` | Chronological fusion of speech, sound, and acoustic events |
| `ground_truth_events` | Human-confirmed events only |

`src/timeline.py` combines speech segments, local sound events, and acoustic
changes into one chronological timeline while preserving their categories,
timestamps, confidence, and verification status.

### 8. Rime-aligned descriptions

`src/description.py` combines Vaani metadata, global measurements, voice
profile results, speech segments, YAMNet candidates, and acoustic changes.
It generates:

- `description`
- `global_description`
- `local_description`
- `global_facts`
- `local_facts`
- `rubric_items`

The rubric inventory covers speaker, source, morphology, pitch, timing, event,
channel, space, other sounds, coupling, addressee, subtext, and change.

Each item carries an evidence status such as `supported`,
`observed_measurement`, `observed_candidate`, `needs_human_review`, or
`unavailable`. Unsupported properties such as age, gender, dialect,
addressee, subtext, and emotion are not invented.

### 9. JSON output

`src/export_json.py` writes one structured file per clip under
`outputs/json/`. The main fields are:

```text
clip_id
metadata
global_info
speech_segments
local_events
sound_event_candidates
acoustic_changes
timeline
processing_info
description
```

The sample [`outputs/json/hindi_002_analysis.json`](outputs/json/hindi_002_analysis.json)
contains the current format. In that sample, Silero detects one full speech
segment, YAMNet provides candidate events, global acoustic measurements are
present, and gender remains unknown.

### 10. Ground-truth and evaluation workflow

`src/evaluation.py` creates an audit template with:

```text
ground_truth_events: []
review_queue: [...]
verification_required: true
```

The correct workflow is:

1. Run the automatic analysis.
2. Listen to the original audio.
3. Review candidate labels and timestamps.
4. Correct, remove, or add events.
5. Move only confirmed events into `ground_truth_events`.
6. Mark them `human_verified`.
7. Run evaluation.

The evaluator refuses to score unverified ground truth. After verification it
calculates collar-based precision, recall, F1, onset error, offset error, and
per-class results.

## Current limitations

- Gender classification model is not installed.
- CAM++ is not a gender classifier; it is a speaker-embedding model.
- Transcript word/phrase alignment is not implemented because inspected Vaani
  records do not contain usable transcript text.
- No emotion model is installed.
- YAMNet predictions require calibration and manual review.
- Complete human auditing of all clips has not been performed.

The current system is therefore a lightweight fact-extraction and audit
pipeline, not an automatic source of unquestionable ground truth.
