"""The generator measures' model-free parts: text normalisation, CER, speaker counting, and the
results.csv writer. The models themselves are tested in test_models.py."""
import pytest

from evaluation.generator_eval.diversity import cosine_similarity, count_distinct_speakers, similarity_to_target, speaker_diversity
from evaluation.generator_eval.intelligibility import score_sample
from evaluation.generator_eval.interfaces import SpeakerEmbedderInterface, SpeechRecogniserInterface
from evaluation.report import ResultRow, write_results_csv
from evaluation.text_norm import character_error_rate, edit_distance, intelligibility_value, normalise


class FixedRecogniser(SpeechRecogniserInterface):
    def __init__(self, transcript: str):
        self.transcript = transcript

    def transcribe(self, audio_path: str, language: str) -> str:
        return self.transcript


class TableEmbedder(SpeakerEmbedderInterface):
    def __init__(self, table: dict[str, list[float]]):
        self.table = table

    def embed(self, audio_path: str) -> list[float]:
        return self.table[audio_path]


# ---- text normalisation and CER ----------------------------------------------------------------

def test_normalise():
    assert normalise("  Hello,   World! ") == "hello world"
    assert normalise("भैया, मैंने सौ दिए थे।") == "भैया मैंने सौ दिए थे"       # vowel signs kept, comma and danda removed
    assert normalise("நான்-தானே சொன்னேன்?") == "நான் தானே சொன்னேன்"
    assert normalise("é") == "é"       # NFC: e + combining accent becomes one character


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("abc", "abc") == 0
    assert edit_distance("abc", "") == 3
    assert edit_distance("", "ab") == 2


def test_character_error_rate():
    assert character_error_rate("abcd", "abcd") == 0.0
    assert character_error_rate("abcd", "abxd") == 0.25
    assert character_error_rate("Abcd!", "abcd") == 0.0          # case and punctuation do not count
    assert character_error_rate("ab", "abcdef") == 2.0           # can be above 1
    with pytest.raises(ValueError):
        character_error_rate("?!", "abc")


def test_intelligibility_value_is_floored_at_zero():
    assert intelligibility_value(0.25) == 0.75
    assert intelligibility_value(2.0) == 0.0


def test_score_sample():
    record = score_sample(FixedRecogniser("abxd"), "sample.wav", "hi", "abcd")
    assert record.cer == 0.25
    assert record.value == 0.75
    assert record.transcript == "abxd"


# ---- speaker similarity and diversity -----------------------------------------------------------

def test_cosine_similarity():
    assert cosine_similarity([1.0, 0.0], [2.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 3.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 1.0], [1.0, 0.0]) == pytest.approx(0.70710678)
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 0.0], [1.0])
    with pytest.raises(ValueError):
        cosine_similarity([0.0, 0.0], [1.0, 0.0])


def test_count_distinct_speakers():
    a = [1.0, 0.0, 0.0]
    a_again = [0.99, 0.05, 0.0]         # cosine with a: 0.9987
    b = [0.0, 1.0, 0.0]
    c = [0.0, 0.0, 1.0]
    assert count_distinct_speakers([a, a_again, a], 0.86) == 1
    assert count_distinct_speakers([a, b, a_again, c, b], 0.86) == 3
    assert count_distinct_speakers([], 0.86) == 0


def test_speaker_diversity_needs_the_protocol_number_of_samples():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert speaker_diversity([a, b, a, a, a, a, a, a, a, a], 0.86, 10) == 0.2
    with pytest.raises(ValueError, match="needs 10 samples"):
        speaker_diversity([a, b], 0.86, 10)


def test_similarity_to_target():
    embedder = TableEmbedder({"generated.wav": [1.0, 1.0], "target.wav": [1.0, 0.0]})
    assert similarity_to_target(embedder, "generated.wav", "target.wav") == pytest.approx(0.70710678)


# ---- results.csv ---------------------------------------------------------------------------------

def test_results_csv_has_the_guideline_columns(tmp_path):
    path = str(tmp_path / "results.csv")
    rows = [ResultRow(track=2, part="captioner", language="hi", system="round_0", measure="rubric_score", n_items=20, value=0.4125),
            ResultRow(track=2, part="generator", language="ta", system="round_0", measure="speaker_diversity", n_items=10, value=0.2)]
    write_results_csv(rows, path)
    with open(path, encoding="utf-8") as f:
        assert f.read() == ("track,part,language,system,measure,n_items,value\n"
                            "2,captioner,hi,round_0,rubric_score,20,0.412500\n"
                            "2,generator,ta,round_0,speaker_diversity,10,0.200000\n")


def test_result_row_rejects_other_tracks_and_languages():
    with pytest.raises(ValueError):
        ResultRow(track=1, part="captioner", language="hi", system="round_0", measure="rubric_score", n_items=1, value=0.5)
    with pytest.raises(ValueError):
        ResultRow(track=2, part="captioner", language="en", system="round_0", measure="rubric_score", n_items=1, value=0.5)
