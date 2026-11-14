"""AI4Bharat IndicConformer as an intelligibility ASR (problem statement). Needs torch,
transformers, onnxruntime and librosa, and the model folder downloaded by scripts/download_models.py.
The repo is gated: accept its terms on Hugging Face before downloading.
"""
from typing import Literal

import librosa
import torch
from pydantic import field_validator
from transformers import AutoModel

from evaluation.generator_eval.interfaces import SpeechRecogniserInterface
from evaluation.settings import Language, Strict, must_be_directory

SAMPLE_RATE = 16000


class IndicConformerSettings(Strict):
    model_dir: str
    decoding: Literal["ctc", "rnnt"]

    @field_validator("model_dir")
    @classmethod
    def model_dir_must_exist(cls, model_dir: str) -> str:
        return must_be_directory(model_dir)


class IndicConformerRecogniser(SpeechRecogniserInterface):
    def __init__(self, settings: IndicConformerSettings):
        self.settings = settings
        # The repo ships its own model code, which runs the network through ONNX on the CPU. CAREFUL I1
        self.model = AutoModel.from_pretrained(settings.model_dir, trust_remote_code=True, local_files_only=True)

    @torch.no_grad()
    def transcribe(self, audio_path: str, language: Language) -> str:
        wave, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        output = self.model(torch.from_numpy(wave).unsqueeze(0), language, self.settings.decoding)
        if isinstance(output, (list, tuple)):       # CAREFUL I2
            output = output[0]
        return str(output)      # raw: the measure normalises it


# ---------------------------------------------------------------------------------------------
# CAREFUL codes
#
# I1   Not yet run: the repo is gated and was not available where this was written. If the repo's
#      code looks for its ONNX files in the Hugging Face cache instead of the model folder, loading
#      from a plain folder fails; the fix is then to load by repo id from a cache folder, in this
#      file only.
# I2   The model card shows a string being returned; some versions return a list with one string
#      per batch item. Both are handled.
# ---------------------------------------------------------------------------------------------
