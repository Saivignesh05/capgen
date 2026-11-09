"""Hindi / Tamil text: syllable counts, fillers, word timing."""

import json

import numpy as np
import pytest

from src import script


@pytest.mark.parametrize("word, n", [
    # Tamil: counted by hand from a real transcript (consonant + pulli adds nothing)
    ("வணக்கம்", 3), ("ஹலோ", 2), ("கேட்குதா", 3), ("கேட்குதாடா", 4), ("நினைக்கிறேன்", 4),
    ("சிக்னலே", 3), ("இல்லை", 2), ("ஏய்", 1), ("மச்சான்", 2), ("கோவப்படாத", 5),
    # Hindi: a word-final bare consonant loses its inherent "a" (ghar, raam, kamal)
    ("घर", 1), ("राम", 1), ("कमल", 2), ("नमस्ते", 3), ("हिंदी", 2), ("अच्छा", 2), ("आदमी", 3),
])
def test_syllable_counts_match_hand_counts(word, n):
    assert script.syllables(word) == n


def test_punctuation_and_other_scripts():
    assert script.syllables("வணக்கம்,") == 3 and script.syllables("घर।") == 1
    assert script.syllables("hello") == 0 and script.syllables("") == 0 and script.syllables("123") == 0


def test_words_and_fillers():
    ws = script.words("ஆ ஏ மச்சான் இங்கே பார்", "ta")
    assert [w["text"] for w in ws] == ["ஆ", "ஏ", "மச்சான்", "இங்கே", "பார்"]
    assert [w["syllables"] for w in ws] == [1, 1, 2, 2, 1]
    assert [w["filler"] for w in ws] == [True, True, False, False, False]
    assert [w["filler"] for w in script.words("हम्म घर", "hi")] == [True, False]
    assert [w["filler"] for w in script.words("ஆ மச்சான்", "hi")] == [False, False]   # wrong language: no fillers


def test_align_words_lays_syllables_over_the_nuclei():
    ws = script.words("ஹலோ மச்சான் பார்", "ta")                      # 2 + 2 + 1 = 5 syllables
    nuc = np.array([0.50, 0.70, 1.00, 1.20, 1.60])
    out = script.align_words(ws, nuc, [(0.4, 2.0)])
    assert [round(w["start"], 2) for w in out] == [0.46, 0.96, 1.56]
    assert [round(w["end"], 2) for w in out] == [0.74, 1.24, 1.64]
    assert all(out[i]["end"] <= out[i + 1]["end"] for i in range(len(out) - 1))     # monotonic


def test_align_words_refuses_a_bad_match():
    ws = script.words("ஹலோ மச்சான் பார்", "ta")
    assert script.align_words(ws, np.array([0.5, 0.7]), [(0.4, 2.0)]) is None       # 5 syllables, 2 nuclei
    assert script.align_words(ws, np.array([]), [(0.4, 2.0)]) is None
    assert script.align_words(ws, np.array([5.0, 5.2, 5.4, 5.6, 5.8]), [(0.4, 2.0)]) is None   # all outside


def test_longest_word_only_when_clear():
    ws = [{"text": "a", "syllables": 1, "filler": False, "start": 0.0, "end": 0.2},
          {"text": "b", "syllables": 1, "filler": False, "start": 0.2, "end": 0.4},
          {"text": "c", "syllables": 1, "filler": False, "start": 0.4, "end": 0.6},
          {"text": "d", "syllables": 1, "filler": False, "start": 0.6, "end": 1.4}]
    out = script.longest_word(ws)
    assert out["text"] == "d" and out["ratio"] == pytest.approx(4.0)
    assert script.longest_word(ws[:3]) is None                                      # too few words
    even = [{**w, "end": w["start"] + 0.2} for w in ws]
    assert script.longest_word(even) is None


def test_transcript_sources(tmp_path):
    (tmp_path / "map.json").write_text(json.dumps({"ta/clips/x.wav": "வணக்கம்"}), encoding="utf-8")
    table = script.load_transcripts({"file": str(tmp_path / "map.json")})
    assert script.text_for({"file": "x"}, table, "/data/ta/clips/x.wav") == "வணக்கம்"
    (tmp_path / "rows.json").write_text(json.dumps([{"audio_path": "a/y.wav", "text": "घर"}]), encoding="utf-8")
    table = script.load_transcripts({"file": str(tmp_path / "rows.json")})
    assert script.text_for({"file": "x"}, table, "/data/a/y.wav") == "घर"
    (tmp_path / "z.txt").write_text("இல்லை", encoding="utf-8")
    assert script.text_for({"dir": str(tmp_path)}, {}, "/data/z.wav") == "இல்லை"
    assert script.text_for(None, {}, "/data/z.wav") is None and script.text_for({"dir": str(tmp_path)}, {}, None) is None
