from abc import ABC, abstractmethod

import torch
from parler_tts import ParlerTTSForConditionalGeneration      # CAREFUL G1
from transformers import AutoTokenizer

from pipeline.audio_types import AudioDescription, AudioOut, AudioTranscript
from pipeline.settings import GeneratorSettings


class GeneratorInterface(ABC):
    """Interface for generators."""

    @abstractmethod
    def generate(self, transcript: AudioTranscript, description: AudioDescription) -> AudioOut:
        """Speak the transcript in the voice and style the description asks for."""


class Generator(GeneratorInterface):
    def __init__(self, settings: GeneratorSettings):
        self.settings = settings
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(settings.model_dir)
        self.model.to(settings.device)
        self.model.eval()
        self.transcript_tokenizer = AutoTokenizer.from_pretrained(settings.model_dir)
        self.description_tokenizer = AutoTokenizer.from_pretrained(settings.description_tokenizer_dir)
        self.sample_rate = self.model.config.sampling_rate      # read from the model's config

    @torch.no_grad()
    def generate(self, transcript: AudioTranscript, description: AudioDescription) -> AudioOut:
        # The generator consumes these, so it checks them. CAREFUL G3
        if not transcript.text:
            raise ValueError("Transcript is empty")
        if not description.text:
            raise ValueError("Description is empty")

        description_inputs = self.description_tokenizer(
            description.text,
            return_tensors="pt").to(self.settings.device)
        transcript_inputs = self.transcript_tokenizer(
            transcript.text,
            return_tensors="pt").to(self.settings.device)

        generation = self.model.generate(
            input_ids=description_inputs.input_ids,
            attention_mask=description_inputs.attention_mask,
            prompt_input_ids=transcript_inputs.input_ids,
            prompt_attention_mask=transcript_inputs.attention_mask)     # CAREFUL G4

        waveform = generation[0].cpu().float().numpy()     # [0]: one clip per call. CAREFUL G5
        return AudioOut(waveform=waveform, sample_rate=self.sample_rate)


# ---------------------------------------------------------------------------------------------
# CAREFUL codes
#
# G1   parler_tts is not on PyPI; environments/generator/requirements.txt installs it from GitHub,
#      pinned to a commit. It needs transformers 4.46.1, which is why the generator has its own
#      environment.
# G2   The description tokenizer is not in the model folder. It must be the tokenizer named in the
#      model's config, model.config.text_encoder._name_or_path (google/flan-t5-large), which is
#      downloads.description_tokenizer in settings.yaml. A different tokenizer gives no error, just
#      wrong voices.
# G3   Empty text is checked here because the generator is the one that consumes it.
# G4   Generation samples randomly: the same inputs give different audio on each run. The caller
#      sets the seed (scripts/stages/generate.py seeds every sample), not this class. Even then,
#      results only repeat on the same GPU and library versions.
# G5   The waveform is raw model output. Whoever uses it checks it: scripts/stages/generate.py
#      rejects empty or NaN waveforms; silence or cut-offs still need listening.
# ---------------------------------------------------------------------------------------------
