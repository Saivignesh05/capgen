"""Tests of the subset and merge script (no models). Run from midterm/:  python -m pytest scoring/tests"""
import csv
import json
import os

import pytest

from scoring import diversity

DEVSET = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "devset")


def test_speaker_of():
    assert diversity.speaker_of("kashish1") == "kashish"
    assert diversity.speaker_of("sarkar-sagar1") == "sarkar-sagar"
    assert diversity.speaker_of("skd_5_rrk_3") == "skd_5_rrk"
    assert diversity.speaker_of("x") == "x"


def test_choose_spreads_over_speakers():
    clips = ["a1", "a2", "a3", "b1", "b2", "c1"]
    assert diversity.choose(clips, 3) == ["a1", "b1", "c1"]
    assert diversity.choose(clips, 5) == ["a1", "a2", "b1", "b2", "c1"]
    assert diversity.choose(clips, 10) == sorted(clips)


def test_subset_from_the_real_devset(tmp_path):
    out = str(tmp_path / "div")
    manifest = diversity.make_subset(DEVSET, out, 10)
    for language in ("hi", "ta"):
        files = sorted(os.listdir(os.path.join(out, language, "clips")))
        assert len(files) == 10 and files == [n + ".wav" for n in manifest["clips"][language]]
        assert len({diversity.speaker_of(n) for n in manifest["clips"][language]}) >= 7
    assert json.load(open(os.path.join(out, "subset.json")))["per_language"] == 10
    with pytest.raises(SystemExit):     # never mixed with an old subset
        diversity.make_subset(DEVSET, out, 10)


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=diversity.COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def row(measure, language, value, n, system="round_0"):
    return {"track": 2, "part": "generator", "language": language, "system": system, "measure": measure, "value": value, "n_items": n}


def test_merge(tmp_path):
    main, sub, out = (str(tmp_path / n) for n in ("a.csv", "b.csv", "out.csv"))
    write(main, [row("intelligibility", "hi", 0.1, 30), row("speaker_diversity", "hi", 1.0, 30)])
    write(sub, [row("speaker_diversity", "hi", 4.5, 10), row("speaker_diversity", "ta", 3.0, 10), row("intelligibility", "hi", 0.2, 10)])
    rows = diversity.merge(main, sub, out)
    got = {(r["measure"], r["language"]): (r["value"], r["n_items"]) for r in rows}
    assert got == {("intelligibility", "hi"): ("0.1", "30"), ("speaker_diversity", "hi"): ("4.5", "10"),
                   ("speaker_diversity", "ta"): ("3.0", "10")}
    with open(out, encoding="utf-8") as f:
        assert next(csv.reader(f)) == diversity.COLUMNS


def test_merge_refuses_mismatched_inputs(tmp_path):
    main, sub, out = (str(tmp_path / n) for n in ("a.csv", "b.csv", "out.csv"))
    write(main, [row("intelligibility", "hi", 0.1, 30)])
    write(sub, [row("intelligibility", "hi", 0.2, 10)])
    with pytest.raises(SystemExit):     # no diversity rows
        diversity.merge(main, sub, out)
    write(sub, [row("speaker_diversity", "hi", 4.5, 10, system="round_1")])
    with pytest.raises(SystemExit):     # different systems
        diversity.merge(main, sub, out)
