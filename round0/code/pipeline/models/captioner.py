from abc import ABC, abstractmethod

import librosa, soundfile
import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

from pipeline.audio_types import AudioClip, AudioDescription
from pipeline.settings import CaptionerSettings


class CaptionerInterface(ABC):
    """Interface for captioners."""

    @abstractmethod
    def describe(self, clip: AudioClip) -> AudioDescription:
        """Take one audio clip, return a description."""


class Captioner(CaptionerInterface):
    def __init__(self, settings: CaptionerSettings):
        self.settings = settings
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(settings.model_dir)
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            settings.model_dir,
            dtype=settings.dtype,       # CAREFUL C4
            device_map="auto",          # CAREFUL C5
            attn_implementation=settings.attn_implementation)
        self.model.eval()
        self.sample_rate = self.processor.feature_extractor.sampling_rate     # read from the model's config

        # The prompt is audio only, so it is the same for every clip.
        conversation = [{"role": "user", "content": [{"type": "audio", "audio": ""}]}]    # CAREFUL C6
        self.prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False)

    @torch.no_grad()
    def describe(self, clip: AudioClip) -> AudioDescription:
        duration = soundfile.info(clip.path).duration
        if duration > self.settings.max_clip_seconds:      # CAREFUL C7
            raise ValueError(f"{clip.path} is {duration:.1f}s, max_clip_seconds is {self.settings.max_clip_seconds}s")

        audio, _ = librosa.load(
            clip.path,
            sr=self.sample_rate,
            mono=True)   # resamples to the model's rate; stereo channels are averaged into one
        inputs = self.processor(
            text=self.prompt,
            audio=[audio],
            return_tensors="pt",
            padding=True)
        inputs = inputs.to(self.model.device).to(self.model.dtype)     # casts only float tensors; input_ids stay integers

        thinker_output, _audio = self.model.generate(      # returns (text, audio); audio is None here. CAREFUL C10
            **inputs,
            return_audio=False,
            thinker_return_dict_in_generate=True,
            thinker_max_new_tokens=self.settings.max_new_tokens,
            thinker_do_sample=False)        # no sampling so no randomness. CAREFUL C8, C9

        # Keep only the generated tokens, not the prompt. [0]: one clip per call.
        new_tokens = thinker_output.sequences[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]     # keep the text exactly as the model wrote it
        return AudioDescription(text=text)


# ---------------------------------------------------------------------------------------------
# CAREFUL codes
#
# C1   bfloat16 needs an Ampere-or-newer GPU (A100, H100). V100 has no bfloat16.
# C2   flash_attention_2 needs the flash-attn package installed, and only works with bfloat16/float16.
# C3   A description that hits max_new_tokens is cut off mid-sentence, with no error.
# C4   The `dtype=` keyword needs a recent transformers. Older versions only accept `torch_dtype=`.
# C5   device_map="auto" spreads the model over every GPU the job can see (CUDA_VISIBLE_DEVICES).
#      The 30B weights alone are about 60 GB in bfloat16, so request enough GPU memory.
# C6   The "audio" value is never read; the template only inserts one audio placeholder.
#      So exactly one audio array must be passed to the processor per prompt.
# C7   This is the only length guard. The processor does not truncate audio, so a long clip goes in
#      whole: slower, more GPU memory, and less detailed descriptions.
# C8   Greedy decoding is repeatable on the same GPU and library versions, but a different GPU or
#      transformers version can give slightly different text.
# C9   Short descriptions are fine: the model emits an end token and stops. Rarely, greedy decoding never
#      emits it and repeats a phrase until max_new_tokens. A description that used all max_new_tokens
#      is the sign to check for this (or for C3).
# C10  Qwen3OmniMoeForConditionalGeneration.generate always returns a tuple (thinker output, audio),
#      even with return_audio=False, when the audio is None (transformers 4.57). Unpack it before
#      reading .sequences, or every clip fails with AttributeError.
# ---------------------------------------------------------------------------------------------
