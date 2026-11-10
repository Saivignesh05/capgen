# Indic-Parler-TTS Generator

A local text-to-speech generator built on [AI4Bharat Indic-Parler-TTS](https://huggingface.co/ai4bharat/indic-parler-tts). Given a text transcript and a natural-language voice description, the system generates a speech waveform and saves it as a WAV file.

---

## How it works

The generator uses two separate inputs:

**Transcript** — the text to be spoken. Indic-Parler-TTS supports multiple Indic languages and English.

**Voice description** — a natural-language sentence describing the desired speaker and recording conditions (e.g. speaker name, gender, speaking speed, pitch, background noise level). This is encoded by a FLAN-T5 text encoder and conditions the generation.

Both inputs are tokenised separately and passed to the ParlerTTS model, which produces a raw audio waveform. The waveform is then written to a 16-bit PCM WAV file.

---

## Project structure

```
generator/
├── generator/
│   ├── __init__.py
│   ├── indic_parler_generator.py   # IndicParlerGenerator class + AudioResult
│   └── audio_writer.py             # save_audio() function
│
├── models/
│   ├── indic-parler-tts/           # model snapshot (downloaded, not committed)
│   │   └── config.json             # committed – required by the generator
│   └── description-tokenizer/      # FLAN-T5 tokenizer cache (downloaded, not committed)
│
├── outputs/
│   └── generated_audio/            # WAV files are saved here (not committed)
│
├── logs/                           # empty; available for log output
│
├── download_model.py               # downloads model weights and tokenizer
├── test_generator.py               # end-to-end usage example / smoke test
├── requirements.txt
└── .gitignore
```

---

## Prerequisites

- **Python 3.10** (the venv in this project uses Python 3.10.22)
- Git
- A [Hugging Face](https://huggingface.co) account with access to the gated model
- ~4 GB free disk space for model weights
- ~8 GB RAM (see [Hardware requirements](#hardware-requirements-and-limitations))

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd generator
```

### 2. Create and activate a Python 3.10 virtual environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### 3. Install PyTorch

PyTorch must be installed separately because the correct wheel depends on your hardware.

**CPU (any platform):**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Apple Silicon (MPS):**
```bash
pip install torch torchaudio
```

**NVIDIA CUDA 12.x:**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

> **Note:** pip will print a dependency conflict warning about `descript-audiotools` requiring `protobuf<5`. This is a stale metadata constraint — `descript-audiotools` does not import protobuf at runtime and works correctly with protobuf 5.x. See [Known issues](#known-issues) for details.

---

## Hugging Face authentication

The Indic-Parler-TTS model is hosted in a **gated repository**. You must:

1. Create a Hugging Face account at <https://huggingface.co>.
2. Visit the model page and accept the access conditions:  
   <https://huggingface.co/ai4bharat/indic-parler-tts>
3. Log in from the command line:

```bash
hf auth login
```

Enter your Hugging Face access token when prompted. The token is stored locally by the `huggingface_hub` library — do not paste it into source code or environment files.

---

## Downloading the model

Run the download script from the project root with the virtual environment active:

```bash
python download_model.py
```

This script:
- Downloads the full model snapshot (~3.75 GB) into `models/indic-parler-tts/`
- Reads `config.json` to identify the description tokenizer (`google/flan-t5-large`)
- Downloads the tokenizer into `models/description-tokenizer/`
- Verifies that the required files are present

The download is resumable. Already-downloaded files are not re-fetched on subsequent runs.

Expected directory state after download:
```
models/
├── indic-parler-tts/
│   ├── config.json
│   ├── model.safetensors        (~3.75 GB)
│   ├── generation_config.json
│   ├── tokenizer.json
│   ├── tokenizer.model
│   └── tokenizer_config.json
└── description-tokenizer/
    └── models--google--flan-t5-large/
```

---

## Usage

```python
from generator.indic_parler_generator import IndicParlerGenerator
from generator.audio_writer import save_audio

# Load the model (takes ~1–2 minutes on CPU; model is loaded once)
generator = IndicParlerGenerator()

# Define the transcript and voice description
transcript = "नमस्ते, आप कैसे हैं?"

description = (
    "Divya speaks in Hindi with a calm, friendly and natural tone. "
    "She speaks at a moderate speed with clear pronunciation and a medium pitch. "
    "The recording is of very high quality with no background noise."
)

# Generate audio
audio_result = generator.generate_audio(
    transcript=transcript,
    description=description,
    seed=42,          # optional: for reproducible output
)

print("Duration:", audio_result.duration, "seconds")
print("Sample rate:", audio_result.sample_rate, "Hz")

# Save the WAV file
output_path = save_audio(
    audio=audio_result,
    output_dir="outputs/generated_audio",
    filename="output.wav",
)

print("Saved to:", output_path)
```

To run the included test script:

```bash
python test_generator.py
```

This loads the model, generates a short Hindi utterance, and saves the result to `outputs/generated_audio/hindi_test_001.wav`.

---

## Hardware requirements and limitations

| Resource | Minimum | Notes |
|---|---|---|
| RAM | 8 GB | Tight on 8 GB machines; close other applications before loading the model |
| Disk | 4 GB | Model weights alone are ~3.75 GB |
| CPU | Any x86-64 or Apple Silicon | Inference is slow on CPU (minutes per utterance) |
| GPU (optional) | NVIDIA CUDA or Apple MPS | Significantly faster; set via `device` parameter |

**8 GB RAM machines:** The model checkpoint is 3.75 GB. After loading it into memory and accounting for the OS and Python overhead, 8 GB machines may swap. Close other applications before running. MPS on Apple Silicon uses shared memory, so GPU and system RAM compete for the same pool.

The generator selects the device automatically (`cuda` → `mps` → `cpu`). To override:

```python
generator = IndicParlerGenerator(device="cpu")
```

---

## Known issues

### 1. Hugging Face gated repository / HTTP 403

**Cause:** You have not accepted the model access conditions or are not logged in.

**Solution:**
1. Accept the conditions at <https://huggingface.co/ai4bharat/indic-parler-tts>
2. Run `hf auth login` and authenticate.

---

### 2. `KeyError: parler_tts` when loading model configuration

**Cause:** `transformers.AutoConfig.from_pretrained()` does not recognise the custom `parler_tts` architecture type and raises a `KeyError`.

**Solution (implemented):** The generator reads `config.json` directly with Python's `json` module instead of using `AutoConfig`. This avoids the architecture lookup entirely.

---

### 3. Protobuf `gencode` / runtime version mismatch

**Symptom:**
```
RuntimeError: ... gencode version 7.x runtime 5.x
```

**Cause:** `sentencepiece==0.2.3` bundles protobuf generated code that is incompatible with the protobuf 5.x runtime.

**Solution (implemented):** Use `sentencepiece==0.2.1`, which is pinned in `requirements.txt`.

---

### 4. `descript-audiotools` protobuf conflict warning

**Symptom:** pip prints:
```
descript-audiotools 0.7.4 requires protobuf<5.0.0, but you have protobuf 5.29.6 which is incompatible.
```

**Cause:** `descript-audiotools` is a transitive dependency of `parler-tts` (via `descript-audio-codec`). Its package metadata declares `protobuf<5.0.0` as a constraint. However, `descript-audiotools` does not import protobuf at runtime — the constraint is a stale artefact from an older version that used gRPC, which has since been removed.

**Status:** This is a metadata-only conflict. The warning can be safely ignored. The environment works correctly with `protobuf==5.29.6`.

---

### 5. Incorrect description tokenizer loading

**Cause:** Passing `cache_dir` to `AutoTokenizer.from_pretrained()` with a directory that uses the Hugging Face hub cache format (`models--<org>--<model>/`) requires the `cache_dir` to point to the **parent** directory, not to the model snapshot itself.

**Solution (implemented):** The generator passes `models/description-tokenizer/` as `cache_dir`. The hub then finds `models--google--flan-t5-large/` inside it automatically.

---

### 6. Flash Attention 2 warning

**Symptom:**
```
Flash attention 2 is not installed
```

**Cause:** The parler-tts library checks for Flash Attention 2 at import time. It is not installed in this environment.

**Status:** This is an informational warning, not an error. The model runs correctly without Flash Attention 2. Flash Attention 2 requires a CUDA-capable GPU and is not available on Apple Silicon MPS or CPU.

---

### 7. Memory limitations on 8 GB RAM machines

**Status:** Partially resolved — the model loads and runs on 8 GB machines, but it is slow and leaves little headroom. Closing background applications before loading the model is recommended. Full generation of a short utterance on CPU may take several minutes.

---

## Dependency versions (verified working environment)

| Package | Version |
|---|---|
| Python | 3.10.22 |
| torch | 2.14.1 |
| torchaudio | 2.11.1 |
| transformers | 4.46.1 |
| parler-tts | 0.2.2 (commit `d108732`) |
| accelerate | 1.15.0 |
| numpy | 1.26.4 |
| soundfile | 0.14.0 |
| huggingface\_hub | 0.36.2 |
| safetensors | 0.8.0 |
| sentencepiece | 0.2.1 |
| protobuf | 5.29.6 |

---

## Model attribution

This project uses the **AI4Bharat Indic-Parler-TTS** model.

- Model repository: <https://huggingface.co/ai4bharat/indic-parler-tts>
- Parler-TTS library: <https://github.com/huggingface/parler-tts>

Please refer to the model repository for licensing terms and usage conditions. The model weights are subject to the terms on the Hugging Face model page and are not redistributed in this repository.

---

## License

See [LICENSE](LICENSE) for the code in this repository. The model weights have separate terms defined by AI4Bharat on the Hugging Face model page.
