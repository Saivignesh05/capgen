
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer


# --------------------------------------------------
# Audio result structure
# --------------------------------------------------

@dataclass
class AudioResult:
    waveform: np.ndarray
    sample_rate: int
    transcript: str
    description: str

    @property
    def duration(self) -> float:
        """Return audio duration in seconds."""
        return len(self.waveform) / self.sample_rate


# --------------------------------------------------
# Indic Parler-TTS Generator
# --------------------------------------------------

class IndicParlerGenerator:

    def __init__(
        self,
        model_dir: Optional[str] = None,
        description_tokenizer_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):

        # Project root directory
        self.project_dir = Path(__file__).resolve().parents[1]

        # Local model directory
        self.model_dir = (
            Path(model_dir)
            if model_dir
            else self.project_dir / "models" / "indic-parler-tts"
        )

        # Hugging Face cache directory for FLAN-T5 tokenizer
        self.description_tokenizer_dir = (
            Path(description_tokenizer_dir)
            if description_tokenizer_dir
            else self.project_dir / "models" / "description-tokenizer"
        )

        # Device selection
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)

        print("\nLoading Indic-Parler-TTS...")
        print("Device:", self.device)

        # Verify local model directory
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_dir}"
            )

        config_path = self.model_dir / "config.json"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Model configuration not found: {config_path}"
            )

        # --------------------------------------------------
        # Read model configuration
        # --------------------------------------------------

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        try:
            description_tokenizer_name = (
                self.config["text_encoder"]["_name_or_path"]
            )
        except KeyError as exc:
            raise ValueError(
                "Could not find text_encoder._name_or_path "
                "in model config.json"
            ) from exc

        # --------------------------------------------------
        # Load Indic Parler-TTS model
        # --------------------------------------------------

        print("\nLoading model weights...")

        self.model = (
            ParlerTTSForConditionalGeneration.from_pretrained(
                str(self.model_dir),
                local_files_only=True,
            )
        )

        self.model.to(self.device)
        self.model.eval()

        # --------------------------------------------------
        # Load transcript tokenizer
        # --------------------------------------------------

        print("\nLoading transcript tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir),
            local_files_only=True,
        )

        # --------------------------------------------------
        # Load description tokenizer
        # --------------------------------------------------

        print("\nLoading description tokenizer...")
        print("Tokenizer:", description_tokenizer_name)

        self.description_tokenizer = AutoTokenizer.from_pretrained(
            description_tokenizer_name,
            cache_dir=str(self.description_tokenizer_dir),
            local_files_only=True,
        )

        # Model sampling rate
        self.sample_rate = self.model.config.sampling_rate

        print("\nIndic-Parler-TTS loaded successfully.")
        print("Sample rate:", self.sample_rate)
        print("Device:", self.device)

    # --------------------------------------------------
    # Generate audio
    # --------------------------------------------------

    def generate_audio(
        self,
        transcript: str,
        description: str,
        seed: Optional[int] = None,
        **generation_kwargs,
    ) -> AudioResult:

        if not isinstance(transcript, str) or not transcript.strip():
            raise ValueError("Transcript must be a non-empty string.")

        if not isinstance(description, str) or not description.strip():
            raise ValueError("Description must be a non-empty string.")

        print("\nGenerating audio...")
        print("Transcript:", transcript)
        print("Voice description:", description)

        # Optional reproducibility seed
        if seed is not None:
            torch.manual_seed(seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # --------------------------------------------------
        # Tokenize voice description
        # --------------------------------------------------

        description_inputs = self.description_tokenizer(
            description,
            return_tensors="pt",
        )

        description_inputs = {
            key: value.to(self.device)
            for key, value in description_inputs.items()
        }

        # --------------------------------------------------
        # Tokenize transcript
        # --------------------------------------------------

        prompt_inputs = self.tokenizer(
            transcript,
            return_tensors="pt",
        )

        prompt_inputs = {
            key: value.to(self.device)
            for key, value in prompt_inputs.items()
        }

        # --------------------------------------------------
        # Generate waveform
        # --------------------------------------------------

        print("\nRunning model inference...")

        with torch.inference_mode():

            generation = self.model.generate(
                input_ids=description_inputs["input_ids"],
                attention_mask=description_inputs["attention_mask"],
                prompt_input_ids=prompt_inputs["input_ids"],
                prompt_attention_mask=prompt_inputs["attention_mask"],
                **generation_kwargs,
            )

        # --------------------------------------------------
        # Convert generated tensor to NumPy waveform
        # --------------------------------------------------

        waveform = generation.detach().cpu().numpy().squeeze()

        waveform = np.asarray(waveform, dtype=np.float32)

        if waveform.ndim != 1:
            raise RuntimeError(
                f"Expected mono audio waveform, got shape {waveform.shape}"
            )

        if waveform.size == 0:
            raise RuntimeError("Model generated an empty audio waveform.")

        if not np.isfinite(waveform).all():
            raise RuntimeError(
                "Generated waveform contains NaN or infinite values."
            )

        # --------------------------------------------------
        # Return structured audio result
        # --------------------------------------------------

        result = AudioResult(
            waveform=waveform,
            sample_rate=self.sample_rate,
            transcript=transcript,
            description=description,
        )

        print("\nAudio generation completed.")
        print("Duration: {:.2f} seconds".format(result.duration))
        print("Sample rate:", result.sample_rate)
        print("Waveform shape:", result.waveform.shape)

        return result

