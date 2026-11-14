"""The evaluation models themselves. Each test loads a real model, so it skips with a reason when
the model is not downloaded, a package is missing, or the GPU is too small. Run `pytest -rs` to see
the reasons. Needs settings/paths.yaml (copy paths.example.yaml).
"""
import glob
import os

import pytest

pytest.importorskip("torch", reason="the evaluation models need torch")
pytest.importorskip("transformers", reason="the evaluation models need transformers")
pytest.importorskip("librosa", reason="the evaluation models read audio with librosa")

from evaluation.generator_eval.diversity import cosine_similarity  # noqa: E402
from evaluation.generator_eval.utmos_scorer import UtmosScorer, UtmosSettings  # noqa: E402
from evaluation.generator_eval.wavlm_speaker import WavLMSpeakerEmbedder, WavLMSpeakerSettings  # noqa: E402
from evaluation.generator_eval.whisper_asr import WhisperRecogniser, WhisperSettings  # noqa: E402
from evaluation.rubric.grader import build_prompt, parse_verdict  # noqa: E402
from evaluation.rubric.grader_sarvam import SarvamGrader, SarvamGraderSettings  # noqa: E402
from evaluation.settings import model_folder, speechmos_folder, torch_hub_folder  # noqa: E402
from helpers import REPO_DIR, evaluation_settings, machine_paths, require_cuda, require_model_folder, rubric_item  # noqa: E402

SETTINGS = evaluation_settings()


def devset_clips(language: str) -> list[str]:
    """Two real clips of the language from the devset, or skip."""
    clips = sorted(glob.glob(os.path.join(REPO_DIR, "devset", language, "clips", "*.wav")))
    if len(clips) < 2:
        pytest.skip(f"no devset clips for {language}")
    return clips[:2]


def device() -> str:
    """The settings' device, or skip if it is cuda and this computer has no GPU."""
    if SETTINGS.device == "cuda":
        require_cuda(2)
    return SETTINGS.device


def test_wavlm_speaker_embedder():
    folder = require_model_folder(model_folder(machine_paths(), SETTINGS.downloads.speaker))
    embedder = WavLMSpeakerEmbedder(WavLMSpeakerSettings(model_dir=folder, device=device()))
    first, second = devset_clips("hi")
    embedding = embedder.embed(first)
    assert len(embedding) == 512
    assert cosine_similarity(embedding, embedder.embed(first)) == pytest.approx(1.0, abs=1e-4)
    assert cosine_similarity(embedding, embedder.embed(second)) < 0.9999


def test_whisper_recogniser():
    folder = require_model_folder(model_folder(machine_paths(), SETTINGS.downloads.whisper))
    recogniser = WhisperRecogniser(WhisperSettings(
        model_dir=folder, device=device(), dtype=SETTINGS.intelligibility.whisper_dtype,
        max_new_tokens=SETTINGS.intelligibility.whisper_max_new_tokens, max_clip_seconds=SETTINGS.intelligibility.max_clip_seconds))
    hindi = recogniser.transcribe(devset_clips("hi")[0], "hi")
    tamil = recogniser.transcribe(devset_clips("ta")[0], "ta")
    assert any("ऀ" <= character <= "ॿ" for character in hindi), f"no Devanagari in: {hindi}"
    assert any("஀" <= character <= "௿" for character in tamil), f"no Tamil script in: {tamil}"


def test_utmos_scorer():
    paths = machine_paths()
    scorer = UtmosScorer(UtmosSettings(
        speechmos_dir=require_model_folder(speechmos_folder(paths)), torch_hub_dir=require_model_folder(torch_hub_folder(paths)),
        speechmos_model=SETTINGS.naturalness.speechmos_model, device=device()))
    score = scorer.score(devset_clips("hi")[0])
    assert 1.0 <= score <= 5.0


def test_sarvam_grader_answers_in_one_parseable_word():
    folder = require_model_folder(model_folder(machine_paths(), SETTINGS.downloads.grader))
    require_cuda(45)
    grader = SarvamGrader(SarvamGraderSettings(model_dir=folder, **SETTINGS.grader.model_dump()))
    item = rubric_item("a", True, 1).model_copy(update={"statement": "The speaker is a man."})
    stated = grader.answer(build_prompt("A man in his thirties speaks calmly in a quiet room.", item, SETTINGS.grader))
    contradicted = grader.answer(build_prompt("A woman in her thirties speaks calmly in a quiet room.", item, SETTINGS.grader))
    not_mentioned = grader.answer(build_prompt("The recording is clean, with no background noise.", item, SETTINGS.grader))
    assert parse_verdict(stated) == "STATED", stated
    assert parse_verdict(contradicted) == "CONTRADICTED", contradicted
    assert parse_verdict(not_mentioned) == "NOT_MENTIONED", not_mentioned
