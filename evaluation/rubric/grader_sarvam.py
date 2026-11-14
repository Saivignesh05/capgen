"""The official rubric grader: sarvamai/sarvam-m, greedy, at most 16 new tokens (doubt session Q31).

Needs torch and transformers, and the model folder downloaded by scripts/download_models.py.
About 48 GB of GPU memory in bfloat16 (a 24B model). CAREFUL S1
"""
import torch
from pydantic import field_validator
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.rubric.grader import GraderInterface
from evaluation.settings import GraderOptions, must_be_directory


class SarvamGraderSettings(GraderOptions):
    """The grader's options plus where its model is."""
    model_dir: str

    @field_validator("model_dir")
    @classmethod
    def model_dir_must_exist(cls, model_dir: str) -> str:
        return must_be_directory(model_dir)


class SarvamGrader(GraderInterface):
    def __init__(self, settings: SarvamGraderSettings):
        self.settings = settings
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_dir, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.model_dir, torch_dtype=getattr(torch, settings.dtype), device_map="auto", local_files_only=True)
        self.model.eval()

    @torch.no_grad()
    def answer(self, prompt: str) -> str:
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=self.settings.enable_thinking)      # CAREFUL S2
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=self.settings.max_new_tokens, do_sample=False)     # greedy
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)      # raw: the caller parses it


# ---------------------------------------------------------------------------------------------
# CAREFUL codes
#
# S1   Not yet run on the cluster. If bfloat16 does not fit, the plan allows 4-bit as a recorded
#      deviation; that is a new implementation of GraderInterface, not an option of this one.
# S2   sarvam-m thinks by default and writes its reasoning before "</think>". With 16 new tokens
#      the answer would never be reached, so the settings file turns thinking off.
# ---------------------------------------------------------------------------------------------
