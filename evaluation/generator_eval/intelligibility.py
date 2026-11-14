"""Intelligibility of generated speech: the character error rate of an ASR's transcript against the
text the generator was asked to say (docs/evaluation_plan.md, section 4.1).

The measure is the same whichever ASR is behind the interface: Whisper large-v3 (doubt session) or
IndicConformer (problem statement). Both are reported. The value is 1 - CER, floored at 0.
"""
from pydantic import BaseModel, ConfigDict

from evaluation.generator_eval.interfaces import SpeechRecogniserInterface
from evaluation.settings import Language
from evaluation.text_norm import character_error_rate, intelligibility_value


class IntelligibilityRecord(BaseModel):
    """One generated sample scored by one ASR."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    audio_path: str
    language: Language
    target_text: str
    transcript: str     # the ASR's raw output
    cer: float
    value: float        # 1 - CER, floored at 0


def score_sample(recogniser: SpeechRecogniserInterface, audio_path: str, language: Language, target_text: str) -> IntelligibilityRecord:
    """Transcribe one generated sample and compare it with the text it should say."""
    transcript = recogniser.transcribe(audio_path, language)
    cer = character_error_rate(target_text, transcript)
    return IntelligibilityRecord(audio_path=audio_path, language=language, target_text=target_text,
                                 transcript=transcript, cer=cer, value=intelligibility_value(cer))
