"""Unit tests that need no model: settings, bootstrap, speaker counting, judge parsing, reading a
round 0 run, and building results.csv. Run from midterm/:  python -m pytest scoring/tests"""
import csv
import json
import os
import shutil

import numpy as np
import pytest

from scoring import common, llm, results
from scoring.models import count_speakers
from scoring.settings import load_settings

MIDTERM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SETTINGS = os.path.join(MIDTERM, "settings.yaml")


@pytest.fixture
def settings(tmp_path):
    """The real settings file, with model_dir pointing at a temporary folder."""
    text = open(SETTINGS, encoding="utf-8").read().replace("model_dir: /path/to/scoring-models", f"model_dir: {tmp_path.as_posix()}")
    path = os.path.join(MIDTERM, "settings.test-tmp.yaml")     # same folder, so relative paths resolve the same way
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    try:
        yield load_settings(path)
    finally:
        os.remove(path)


def test_settings_load_and_resolve_paths(settings):
    assert settings.devset == os.path.normpath(os.path.join(MIDTERM, "..", "devset"))
    assert os.path.isdir(settings.devset)
    assert settings.references[0].files[0].endswith(os.path.join("devset", "hi", "ground_truth_descriptions.jsonl"))
    assert settings.stages == ["intelligibility", "speaker_diversity", "naturalness", "caption_generated", "judge"]


def test_settings_reject_unknown_keys(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text(open(SETTINGS, encoding="utf-8").read() + "\nunknown_key: 1\n", encoding="utf-8")
    with pytest.raises(Exception, match="unknown_key"):
        load_settings(str(path))


def test_bootstrap_is_deterministic_and_brackets_the_mean():
    values = {f"c{i:02d}": float(i) for i in range(30)}
    low, high = common.bootstrap_ci(values, 0, "m", 1000)
    assert low < sum(values.values()) / 30 < high
    assert common.bootstrap_ci(values, 0, "m", 1000) == (low, high)
    assert common.bootstrap_ci(values, 1, "m", 1000) != (low, high)
    assert common.bootstrap_ci({"a": 1.0}, 0, "m", 1000) is None


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


def test_count_speakers():
    a, b = unit([1, 0, 0]), unit([0, 1, 0])
    assert count_speakers([], 0.86) == 0
    assert count_speakers([a, a, a], 0.86) == 1
    assert count_speakers([a, b, a, b], 0.86) == 2
    # a chain: x~y and y~z but not x~z -> complete linkage keeps two speakers
    x, y, z = unit([1, 0]), unit([np.cos(0.4), np.sin(0.4)]), unit([np.cos(0.8), np.sin(0.8)])
    assert count_speakers([x, y, z], 0.9) == 2


def test_parse_json_handles_think_blocks_and_fences():
    assert llm.parse_json('<think>hmm [1]</think>```json\n[{"id": 1}]\n```') == [{"id": 1}]
    assert llm.parse_json('Sure: {"items": [1, 2]} done') == {"items": [1, 2]}
    with pytest.raises(ValueError):
        llm.parse_json("no json here")


def test_clean_statements_and_labels():
    statements = llm.clean_statements({"rubric": [{"statement": "A man speaks", "start": None, "end": None},
                                                   {"statement": "laughs", "start": "2.1", "end": 2.6},
                                                   "a plain string", {"statement": ""}]})
    assert [s["statement"] for s in statements] == ["A man speaks", "laughs", "a plain string"]
    assert statements[1]["start"] == 2.1
    assert "[2.10-2.60 s]" in llm.format_items(statements)

    labels, missing = llm.collect_labels([{"id": 1, "label": "stated"}, {"id": "3", "label": "Not Stated"},
                                          {"id": 9, "label": "stated"}, {"id": 2, "label": "maybe"}], 3, llm.RUBRIC_LABELS)
    assert labels == ["stated", None, "not_stated"] and missing == 1


def test_rubric_score_matches_rime_formula():
    assert llm.rubric_score(["stated", "stated", "contradicted", "not_stated"]) == 0.25
    assert llm.rubric_score(["contradicted", "not_stated"]) == 0.0      # floored at 0
    assert llm.rubric_score(["stated", None]) == 0.5                     # missing = not stated


def test_truncate_words():
    assert llm.truncate_words("a b c", 5) == "a b c"
    assert llm.truncate_words("a b c d", 2) == "a b"


def make_run(tmp_path, devset):
    """A fake round 0 run over two real devset clips (one hi, one ta)."""
    run = tmp_path / "run-x"
    (run / "audio").mkdir(parents=True)
    hi = os.path.join(devset, "hi", "clips", "kashish1.wav")
    ta = os.path.join(devset, "ta", "clips", "divya1.wav")
    clips = [{"clip_id": "kashish1", "path": hi, "sha256": "x"}, {"clip_id": "divya1", "path": ta, "sha256": "y"}]
    (run / "clips.jsonl").write_text("".join(json.dumps(c) + "\n" for c in clips), encoding="utf-8")
    (run / "transcripts.jsonl").write_text(json.dumps({"clip_id": "kashish1", "language": "ta", "transcript": "नमस्ते"}) + "\n", encoding="utf-8")
    (run / "descriptions.jsonl").write_text(json.dumps({"clip_id": "kashish1", "description": "A woman."}) + "\n", encoding="utf-8")
    rows = []
    for sample in (1, 0):
        path = run / "audio" / "kashish1" / f"{sample}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(hi, path)
        rows.append({"clip_id": "kashish1", "sample": sample, "seed": 1, "file": f"audio/kashish1/{sample}.wav"})
    (run / "audio.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(run)


def test_load_run(tmp_path, settings, caplog):
    import logging
    logger = logging.getLogger("test")
    items = {i.clip_id: i for i in common.load_run(make_run(tmp_path, settings.devset), logger)}
    hi, ta = items["kashish1"], items["divya1"]
    assert hi.language == "hi" and hi.asr_language == "ta"     # language from the folder, not the ASR
    assert [s.sample for s in hi.samples] == [0, 1]
    assert ta.language == "ta" and ta.transcript is None and ta.samples == ()


def test_folder_language():
    assert common.folder_language("/x/devset/hi/clips/a.wav") == "hi"
    assert common.folder_language(r"C:\x\devset\ta\clips\a.wav") == "ta"
    assert common.folder_language("/x/other/a.wav") is None


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_results_csv_from_stage_outputs(tmp_path, settings):
    scores = str(tmp_path)
    write_jsonl(os.path.join(scores, "intelligibility.jsonl"), [
        {"clip_id": "a", "sample": 0, "language": "hi", "edits": 1, "n": 10},
        {"clip_id": "a", "sample": 1, "language": "hi", "edits": 3, "n": 10},     # item a: 4/20 = 0.2
        {"clip_id": "b", "sample": 0, "language": "hi", "edits": 0, "n": 5}])     # item b: 0
    write_jsonl(os.path.join(scores, "speaker_diversity.jsonl"), [
        {"clip_id": "a", "language": "ta", "n_samples": 10, "n_speakers": 3},
        {"clip_id": "b", "language": "ta", "n_samples": 10, "n_speakers": 1}])
    write_jsonl(os.path.join(scores, "rubric.jsonl"), [
        {"reference": "tool", "measure": "rubric_score_tool", "clip_id": "a", "language": "hi", "score": 0.5}])
    write_jsonl(os.path.join(scores, "relevance.jsonl"), [
        {"clip_id": "a", "sample": 0, "language": "ta", "met": 1, "n_checks": 4},
        {"clip_id": "a", "sample": 1, "language": "ta", "met": 3, "n_checks": 4}])
    write_jsonl(os.path.join(scores, "naturalness.jsonl"), [
        {"clip_id": "a", "sample": 0, "language": "hi", "utmos": 3.0, "sigmos.ovrl": 2.0},
        {"clip_id": "a", "sample": 1, "language": "hi", "utmos": 4.0, "sigmos.ovrl": 3.0}])
    common.write_json(os.path.join(scores, "naturalness_corpus.json"), {"hi": {"fad": 12.5, "n_generated": 2, "n_native": 30}})

    rows, detail = results.compute(scores, settings, "round_0")
    got = {(r["part"], r["language"], r["measure"]): (r["value"], r["n_items"]) for r in rows}
    assert got[("generator", "hi", "intelligibility")] == (0.1, 2)
    assert got[("generator", "ta", "speaker_diversity")] == (2.0, 2)
    assert got[("captioner", "hi", "rubric_score_tool")] == (0.5, 1)
    assert got[("generator", "ta", "instruction_relevance")] == (0.5, 1)
    assert got[("generator", "hi", "naturalness_utmos")] == (3.5, 1)
    assert got[("generator", "hi", "naturalness_sigmos_ovrl")] == (2.5, 1)
    assert got[("generator", "hi", "naturalness_fad")] == (12.5, 2)
    assert "rubric_score" in detail["missing"]            # ground truth is empty
    assert all(r["track"] == 2 and r["system"] == "round_0" for r in rows)

    path = os.path.join(scores, "results.csv")
    results.write_csv(path, rows)
    with open(path, encoding="utf-8") as f:
        assert next(csv.reader(f)) == ["track", "part", "language", "system", "measure", "value", "n_items"]


def test_intelligibility_scoring_uses_score_hi_ta():
    import sys
    sys.path.insert(0, os.path.dirname(MIDTERM))
    from score_hi_ta import score_pair
    scored = score_pair("मैं घर जा रहा हूँ", "मैं घर जा रहा हूं", "hi")
    assert scored["cer_ns"]["N"] > 0 and scored["cer_ns"]["edits"] == 1
