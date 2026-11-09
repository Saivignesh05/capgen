"""End to end: one clip -> features JSON -> description; folder runs, shards, merge, CLI."""

import json

import numpy as np
import pytest
from conftest import ROOT, SR, config, harmonic, noise, sine, syllables, t_axis, utterance, wav_bytes

from src import __main__ as cli
from src.describe import describe, kept_events
from src.pipeline import Extractor


@pytest.fixture(scope="module")
def ex():
    return Extractor(config())


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "ta"
    (d / "sub").mkdir(parents=True)
    (d / "a.wav").write_bytes(wav_bytes(utterance()))
    (d / "sub" / "b.wav").write_bytes(wav_bytes(utterance(4.6)))
    (d / "broken.wav").write_bytes(b"RIFFjunk")
    (d / "quiet.wav").write_bytes(wav_bytes(np.zeros(SR, np.float32)))
    (d / "notes.txt").write_text("x")
    return d


def extract(ex, tmp_path, x=None, name="ta/a.wav"):
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(wav_bytes(utterance() if x is None else x))
    return ex.extract(p)


# ------------------------------------------------------------------ features


def test_global_features_match_the_signal(ex, tmp_path):
    f = extract(ex, tmp_path)
    g = f["global"]
    assert f["status"] == "ok" and f["clip"]["duration_s"] == 5.0
    assert g["language"]["code"] == "ta"
    assert g["speech"]["phrases"] == 2 and g["speech"]["pauses"] == 1
    assert g["speech"]["speech_s"] == pytest.approx(3.2, abs=0.2)
    assert g["speech"]["syllables_per_s"] == pytest.approx(4, abs=0.6)
    assert g["pitch"]["median_hz"] == pytest.approx(130, rel=0.03)
    assert g["pitch"]["p90_hz"] == pytest.approx(200, rel=0.03)
    assert g["recording"]["snr_db"] > 30 and "spectral" not in g
    assert g["voice"]["hnr_db"] > 10 and g["voice"]["jitter_pct"] is not None      # numbers, no label


def test_local_phrases_pauses_and_pitch_movement(ex, tmp_path):
    loc = extract(ex, tmp_path)["local"]
    p1, p2 = loc["phrases"]
    assert p1["start"] == pytest.approx(0.5, abs=0.1) and p1["end"] == pytest.approx(2.0, abs=0.1)
    assert p2["start"] == pytest.approx(2.8, abs=0.1) and p2["end"] == pytest.approx(4.5, abs=0.1)
    assert p1["pitch"]["median_hz"] == pytest.approx(130, rel=0.03) and abs(p1["pitch"]["trend_st"]) < 1
    assert p2["pitch"]["trend_st"] == pytest.approx(12 * np.log2(200 / 130), abs=1)   # 7.5 semitone rise
    assert p1["syllables_per_s"] == pytest.approx(4, abs=0.8)
    assert abs(p1["level_vs_others_db"]) < 3
    (pause,) = loc["pauses"]
    assert pause["ms"] == pytest.approx(800, abs=150)
    assert {"n", "start", "end", "syllables", "syllables_per_s", "rate_first_half", "rate_second_half",
            "rate_change", "pitch", "pitch_confirmed", "final_pitch_st", "final_pitch_confirmed",
            "level_dbfs", "level_vs_others_db", "onset_ms", "offset_ms", "voice", "arousal"} == set(p1)
    assert p1["pitch_confirmed"] and p2["pitch_confirmed"]     # clean synthetic tones: trackers agree


def test_only_the_real_step_is_a_change(ex, tmp_path):
    (ch,) = extract(ex, tmp_path)["local"]["changes"]          # 4 syl/s pulsing is not a change
    assert ch["t"] == pytest.approx(3.6, abs=0.12) and ch["phrase"] == 2
    assert ch["pitch_st"] == pytest.approx(12 * np.log2(200 / 130), abs=0.5)
    assert ch["from_hz"] == pytest.approx(130, rel=0.03) and ch["to_hz"] == pytest.approx(200, rel=0.03)
    assert "level_db" not in ch


def test_json_is_plain_and_records_how_it_was_made(ex, tmp_path):
    f = extract(ex, tmp_path)
    assert json.loads(json.dumps(f)) == f                       # no numpy left
    run = f["run"]
    assert run["config_sha"] == ex.config_sha and run["device"] == "cpu"
    assert run["models"] == {"vad": "energy", "language": "folder", "syllables": "envelope"}
    assert run["seconds"]["total"] > 0


def test_same_input_same_output(ex, tmp_path):
    a, b = extract(ex, tmp_path), extract(ex, tmp_path)
    for f in (a, b):
        f["run"].pop("seconds")
    assert a == b


def test_checks_reject_and_warn(ex, tmp_path):
    assert extract(ex, tmp_path, np.zeros(SR, np.float32))["status"] == "rejected"      # no speech
    short = extract(ex, tmp_path, utterance()[:SR // 10])
    assert short["status"] == "rejected" and short["checks"][0]["check"] == "min_duration_s"
    other = extract(ex, tmp_path, name="misc/a.wav")                                     # language und
    assert other["status"] == "rejected" and any(c["check"] == "language" for c in other["checks"])
    noisy = utterance() + np.random.default_rng(0).standard_normal(5 * SR).astype(np.float32) * 0.05
    f = extract(ex, tmp_path, noisy)
    assert f["status"] == "ok" and any(c["check"] == "min_snr_db" and not c["reject"] for c in f["checks"])


def test_broken_file_is_rejected_not_crashing(ex, tmp_path):
    p = tmp_path / "x.wav"
    p.write_bytes(b"RIFFjunk")
    f = ex.extract(p)
    assert f["status"] == "rejected" and f["checks"][0]["check"] == "decode"


# ------------------------------------------------------------------ description


def test_description_has_global_then_timed_local_lines(ex, tmp_path):
    text = describe(extract(ex, tmp_path))
    glob, *local = text.split("\n")
    assert glob.startswith("Tamil, 5.0 s. Recording: 16 kHz mono file")
    assert "Speech: 3." in glob and "2 phrases with one pause of" in glob and "median pitch 130 Hz" in glob
    assert [line.split(",")[0].split("–")[0] for line in local] == sorted(
        line.split(",")[0].split("–")[0] for line in local)
    assert len(local) == 4 and "pause of" in local[1] and "rises about 7 semitones" in local[2]
    assert ", within phrase 2: pitch jumps up 7 semitones (130 → 200 Hz)" in local[3]
    assert "band-limited" in glob                     # synthetic voice has nothing above 2.5 kHz
    assert "{" not in text and "None" not in text


def test_events_are_stated_with_probability_and_other_voices_never(ex, tmp_path):
    f = extract(ex, tmp_path)
    f["local"]["events"] = [
        {"start": 2.1, "end": 2.6, "label": "background_speech", "p": 0.6, "p_max": 0.6, "speech_overlap": 0.0},
        {"start": 3.0, "end": 3.4, "label": "laughter", "p": 0.5, "p_max": 0.6, "speech_overlap": 1.0}]
    assert [e["label"] for e in kept_events(f)] == ["laughter"]
    text = describe(f)
    assert "Other sounds (timed below): laughter." in text and "other voices" not in text
    assert "3.00–3.40 s, laughter (during speech, p 0.50)." in text
    f["local"]["events"] = []
    assert "No other sounds" not in describe(f)                  # no sound-event model ran: say nothing
    f["run"]["models"]["events"] = "ast-audioset"
    assert "No other sounds detected by the sound-event model" in describe(f)


def test_unconfirmed_pitch_jumps_are_not_stated(ex, tmp_path):
    """Three guards: an octave-sized jump, a second tracker that disagrees, and a loudness step
    that is half of a brief burst. All stay in the JSON, none reach the text."""
    from src.describe import change_bits
    octave = {"t": 1.0, "pitch_st": 12.1, "from_hz": 110, "to_hz": 221, "confirmed": False,
              "reason": "possible octave error"}
    assert change_bits(octave) == []
    assert change_bits({**octave, "level_db": 7.0}) == ["7 dB louder"]
    # the real faz4 failure: pYIN read a drop where two other methods saw a rise
    assert change_bits({"t": 2.63, "pitch_st": -6.1, "from_hz": 123, "to_hz": 86, "check_st": 17.2,
                        "confirmed": False, "reason": "second pitch tracker disagrees"}) == []
    assert change_bits({"t": 1.0, "pitch_st": -7.0, "from_hz": 150, "to_hz": 100, "check_st": -6.5,
                        "confirmed": True}) == ["pitch drops 7 semitones (150 → 100 Hz)"]
    assert change_bits({"t": 1.0, "level_db": 8.0, "transient": True}) == []
    x = harmonic(np.where(t_axis(3.0) < 1.5, 110.0, 220.0), 3.0, 0.3)          # an octave step
    ch = extract(ex, tmp_path, x)["local"]["changes"]
    assert any(not c.get("confirmed", True) for c in ch)
    assert "jumps up 12" not in describe(extract(ex, tmp_path, x))


def test_unconfirmed_phrase_pitch_is_not_stated(ex, tmp_path):
    """The real faz1/faz4 failures: pYIN's phrase median, final pitch and a level step the
    second view disagrees with all stay in the JSON but out of the text."""
    from src.describe import change_bits, phrase_bits
    f = extract(ex, tmp_path)
    ph = dict(f["local"]["phrases"][0])
    assert any(b.startswith("pitch median") for b in phrase_bits(ph, False))
    ph["pitch_confirmed"] = False
    assert not any("pitch" in b for b in phrase_bits(ph, False))               # median, range, peak all gone
    ph.update(pitch_confirmed=True, final_pitch_st=-5.0, final_pitch_confirmed=False)
    assert not any(b.startswith("ends") for b in phrase_bits(ph, False))
    ph["final_pitch_confirmed"] = True
    assert "ends 5 semitones lower than the rest of the phrase" in phrase_bits(ph, False)
    # a level step whose median view disagrees (burst-dominated) is kept but not stated
    assert change_bits({"t": 1.0, "level_db": 7.0, "level_confirmed": False,
                        "level_reason": "median level barely moves (burst-dominated step)"}) == []
    # one merged event can carry a confirmed pitch jump and an unconfirmed level step
    assert change_bits({"t": 1.0, "pitch_st": 7.0, "from_hz": 100, "to_hz": 150, "confirmed": True,
                        "level_db": 7.0, "level_confirmed": False}) == ["pitch jumps up 7 semitones (100 → 150 Hz)"]


def test_a_brief_burst_is_not_two_loudness_changes():
    """23 % of the loudness steps on a 30-clip set were opposite-sign pairs under 0.6 s apart -
    one burst reported as "louder" then "softer"."""
    from src.pipeline import _mark_transients
    ch = [{"t": 18.33, "level_db": 7.4}, {"t": 18.81, "level_db": -7.1}, {"t": 20.3, "level_db": -9.0}]
    _mark_transients(ch, 0.6)
    assert ch[0]["transient"] and ch[1]["transient"] and "transient" not in ch[2]
    far = [{"t": 1.0, "level_db": 7.0}, {"t": 2.0, "level_db": -7.0}]           # 1 s apart: both real
    _mark_transients(far, 0.6)
    assert not any("transient" in c for c in far)
    same = [{"t": 1.0, "level_db": 7.0}, {"t": 1.3, "level_db": 8.0}]           # same direction: real
    _mark_transients(same, 0.6)
    assert not any("transient" in c for c in same)


def test_second_opinion_compares_the_same_before_after_window():
    from src.pipeline import _second_opinion
    t = np.arange(0, 3, 0.01)
    f0 = np.where(t < 1.5, 120.0, 180.0)
    assert _second_opinion(t, f0, 1.5, 0.15) == pytest.approx(12 * np.log2(1.5))
    assert _second_opinion(t, np.full_like(t, np.nan), 1.5, 0.15) is None       # unvoiced: no opinion
    assert _second_opinion(t, f0, 0.005, 0.15) is None                          # too few frames before


# ------------------------------------------------------------------ folder runs / CLI


def quiet(*_):
    pass


def load(p):
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture
def base(tmp_path):
    return tmp_path / "features-extractor-results" / "ta"


def test_run_writes_its_own_folder_and_copies_descriptions_into_dir(folder, base):
    s = cli.run(config(), folder, base, log=quiet)
    res = base / s["run"]
    assert res.parent == base and s["results"] == str(res)
    rows = load(res / "tool_descriptions.json")
    assert [r["audio_path"] for r in rows] == ["ta/a.wav", "ta/sub/b.wav"] and s["written"] == 2
    assert rows[0]["tool_description"].startswith("Tamil, 5.0 s.")
    copied = folder / "tool_descriptions.json"
    assert copied.read_text() == (res / "tool_descriptions.json").read_text() and s["copied_to"] == str(copied)
    assert copied.read_text().startswith('[\n  {\n    "audio_path"')                    # indented
    assert {r["audio_path"] for r in load(res / "tool_descriptions.rejected.json")} == {"ta/broken.wav", "ta/quiet.wav"}
    f = load(res / "features" / "sub" / "b.json")
    assert f["clip"]["path"] == "ta/sub/b.wav" and {"global", "local", "run"} <= set(f)
    meta = load(res / "run.json")
    assert meta["status"] == "finished" and meta["descriptions"] == 2 and meta["folder"] == str(folder)
    assert not list(base.rglob("*.jsonl")) and not list(base.rglob("*.tmp")) and not (folder / "features").exists()


def test_every_run_is_kept(folder, base):
    a = cli.run(config(), folder, base, log=quiet)["run"]
    b = cli.run(config(), folder, base, log=quiet)["run"]
    assert a != b and (base / a / "tool_descriptions.json").exists() and (base / b / "tool_descriptions.json").exists()
    assert [r["run"] for r in cli.list_runs(base)] == [a, b]


def test_default_results_base_is_next_to_the_project(folder):
    assert cli.results_dir(folder) == ROOT.parent / "features-extractor-results" / "ta"


def test_resume_named_run_redoes_only_the_last_clip(folder, base, monkeypatch):
    first = cli.run(config(), folder, base, log=quiet)["run"]
    cli.run(config(), folder, base, log=quiet)                                       # a newer run
    seen, real = [], Extractor.extract
    monkeypatch.setattr(Extractor, "extract", lambda self, p: seen.append(p.name) or real(self, p))
    logs = []
    s = cli.run(config(), folder, base, resume=first, log=logs.append)
    assert s["run"] == first and seen == ["b.wav"] and (s["computed"], s["reused"]) == (1, 3)
    assert any("redoing b.wav" in m for m in logs) and len(cli.list_runs(base)) == 2      # no new run
    assert cli.run(config(), folder, base, resume="latest", log=quiet)["run"] != first
    with pytest.raises(SystemExit, match="not found"):
        cli.run(config(), folder, base, resume="nope", log=quiet)


def test_interrupted_run_keeps_its_work_and_resumes(folder, base, monkeypatch):
    real, calls = Extractor.extract, []

    def flaky(self, p):
        calls.append(p.name)
        if len(calls) == 3:
            raise KeyboardInterrupt("killed")                                       # e.g. SLURM time limit
        return real(self, p)
    monkeypatch.setattr(Extractor, "extract", flaky)
    logs = []
    with pytest.raises(KeyboardInterrupt):
        cli.run(config(), folder, base, log=logs.append)
    (run_name,) = [r["run"] for r in cli.list_runs(base)]
    assert load(base / run_name / "run.json")["status"] == "stopped"
    assert any(f"--resume {run_name}" in m for m in logs)
    assert [r["audio_path"] for r in load(folder / "tool_descriptions.json")] == ["ta/a.wav"]
    monkeypatch.setattr(Extractor, "extract", real)
    s = cli.run(config(), folder, base, resume=run_name, log=quiet)
    assert (s["reused"], s["computed"]) == (1, 3) and len(load(folder / "tool_descriptions.json")) == 2
    assert load(base / run_name / "run.json")["status"] == "finished"


def test_resume_redoes_corrupt_or_other_config_features(folder, base):
    name = cli.run(config(), folder, base, log=quiet)["run"]
    (base / name / "features" / "a.json").write_text("{not json")
    other = config(pitch={"method": "autocorr", "fmin_hz": 60, "fmax_hz": 450})
    s = cli.run(other, folder, base, resume=name, log=quiet)
    assert (s["computed"], s["reused"]) == (4, 0)


def test_shards_share_a_run_then_merge_equals_a_single_run(folder, base):
    cli.run(config(), folder, base, log=quiet)
    single = (folder / "tool_descriptions.json").read_text()
    (folder / "tool_descriptions.json").unlink()
    for i in range(3):
        assert cli.run(config(), folder, base, shard=(i, 3), run_id="job7", log=quiet)["copied_to"] is None
    assert len(list((base / "job7").glob("*.part-*"))) == 9          # descriptions, rejected, run.json per shard
    assert not (folder / "tool_descriptions.json").exists()
    cli.rebuild(folder, "job7", base)
    assert (folder / "tool_descriptions.json").read_text() == single and not list((base / "job7").glob("*.part-*"))
    assert load(base / "job7" / "run.json")["status"] == "finished"


def test_describe_rebuilds_a_run_from_saved_features(folder, base):
    name = cli.run(config(), folder, base, log=quiet)["run"]
    before = (folder / "tool_descriptions.json").read_text()
    (folder / "tool_descriptions.json").unlink()
    assert cli.rebuild(folder, name, base) == 2 and (folder / "tool_descriptions.json").read_text() == before
    (folder / "a.wav").unlink()                                                     # deleted clips drop out
    assert cli.rebuild(folder, "latest", base) == 1


def test_copy_script_by_hand(folder, base, tmp_path):
    import subprocess
    import sys
    res = base / cli.run(config(), folder, base, log=quiet)["run"]
    out = subprocess.run([sys.executable, str(ROOT / "copy_descriptions.py"), str(res), str(tmp_path / "elsewhere")],
                         capture_output=True, text=True, check=True)
    assert (tmp_path / "elsewhere" / "tool_descriptions.json").read_text() == (res / "tool_descriptions.json").read_text()
    assert out.stdout.strip().endswith("tool_descriptions.json")
    (res / "tool_descriptions.json").write_text('{"not": "a list"}')
    bad = subprocess.run([sys.executable, str(ROOT / "copy_descriptions.py"), str(res), str(tmp_path / "x")],
                         capture_output=True, text=True)
    assert bad.returncode != 0 and not (tmp_path / "x" / "tool_descriptions.json").exists()


def test_cli_and_shipped_config(folder, tmp_path, capsys):
    import yaml
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(config()))
    base = tmp_path / "r"
    assert cli.main(["run", str(folder), "--config", str(cfg), "--results", str(base), "--run-id", "first"]) == 0
    assert '"run": "first"' in capsys.readouterr().out and (folder / "tool_descriptions.json").exists()
    assert cli.main(["merge", str(folder), "--run", "first", "--results", str(base)]) == 0
    assert cli.main(["runs", str(folder), "--results", str(base)]) == 0
    assert capsys.readouterr().out.splitlines()[-1].startswith("first")
    with pytest.raises(SystemExit):
        cli.main(["run", str(folder), "--shard", "3/2"])
    with pytest.raises(SystemExit):
        cli.main(["merge", str(folder)])                                            # --run is required
    shipped = yaml.safe_load((ROOT / "config.yaml").read_text())
    assert shipped["vad"]["impl"] == "silero" and shipped["language"]["impl"] == "whisper"
    assert shipped["events"]["impl"] == "ast" and shipped["pitch"]["method"] == "pyin"


# ------------------------------------------------------------------ added rubric items


def test_added_fields_and_no_false_tempo_claims(ex, tmp_path):
    f = extract(ex, tmp_path)
    g, ph = f["global"], f["local"]["phrases"][1]
    assert g["recording"]["hum"] is None and g["recording"]["noise_variation_db"] < 3
    assert ph["rate_first_half"] == pytest.approx(ph["rate_second_half"], rel=0.1)    # steady 4 syl/s
    text = describe(f)
    assert "speeds up" not in text and "slows down" not in text and "mains hum" not in text
    for gone in ("phonation", "breathy", "pressed", "creak", "whisper", "brighter", "darker",
                 "breath", "click"):
        assert gone not in text, gone                               # labels stay with the ALM
    assert "Voice measurements (numbers only, not a judgement of voice quality):" in text


def test_hum_end_to_end(ex, tmp_path):
    x = utterance() + (0.01 * sine(50, 5.0, 1.0) + 0.005 * sine(100, 5.0, 1.0)
                       + 0.003 * sine(150, 5.0, 1.0)).astype(np.float32)
    f = extract(ex, tmp_path, x)
    assert f["global"]["recording"]["hum"]["hz"] == 50 and "mains hum at 50 Hz" in describe(f)


def test_description_lines_for_shape_and_timing(ex, tmp_path):
    f = extract(ex, tmp_path)
    ph = f["local"]["phrases"][0]
    ph.update(rate_first_half=4.0, rate_second_half=6.0, rate_change="speeds_up", final_pitch_st=-3.2,
              onset_ms=10.0, offset_ms=240.0)
    f["global"]["speech"]["phrase_rhythm"] = {"period_s": 0.72, "cv": 0.05}
    text = describe(f)
    line = text.split("\n")[1]
    for bit in ("speeds up from about 4.0 to 6.0 syllables per second",
                "ends 3 semitones lower than the rest of the phrase", "abrupt start (10 ms rise)", "fades out over 240 ms"):
        assert bit in line, bit
    assert "nearly regular 0.72 s period" in text
    f["global"]["recording"]["snr_db"] = 12                                       # noisy: edges not stated
    assert "abrupt start" not in describe(f) and "fades out" not in describe(f)


# ------------------------------------------------------------------ fixes from the real-clip check


def test_a_stop_inside_speech_is_a_pause_not_a_loudness_change(tmp_path):
    from src.pipeline import Extractor as E
    ex = E(config(vad={"impl": "energy", "threshold": 0.5, "neg_threshold": 0.05, "min_speech_s": 0.15,
                      "min_silence_s": 0.5, "pad_s": 0.05}))          # VAD that runs through 300 ms stops
    x = noise(4.0, -70)
    x[int(0.5 * SR):int(1.5 * SR)] += syllables(4, 1.0, 140, 0.4)
    x[int(1.8 * SR):int(3.2 * SR)] += syllables(4, 1.4, 140, 0.4)              # 300 ms stop at 1.5-1.8
    f = extract(ex, tmp_path, x)
    ph, loc = f["local"]["phrases"], f["local"]
    assert len(ph) == 2 and loc["pauses"][0]["start"] == pytest.approx(1.5, abs=0.1)
    assert not any("level_db" in c for c in loc["changes"])
    assert "softer" not in describe(f) and "louder" not in describe(f)


def test_48k_file_bandwidth_is_not_capped_at_8k(ex, tmp_path):
    from scipy.signal import resample_poly
    x48 = resample_poly(utterance(), 3, 1).astype(np.float32) + noise(5.0, -40, sr=48000)
    p = tmp_path / "ta" / "x48.wav"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(wav_bytes(x48, 48000))
    f = ex.extract(p)
    assert f["global"]["recording"]["bandwidth"]["high_hz"] > 10000


def test_no_filler_claims(ex, tmp_path):
    text = describe(extract(ex, tmp_path))
    assert "same level as the other phrases" not in text and "fluctuating" not in text


# ------------------------------------------------------------------ added: speakers, space, words, pulses


def test_speakers_and_space_in_the_description(ex, tmp_path):
    f = extract(ex, tmp_path)
    f["global"]["speakers"] = {"count": 1, "confident": "single", "per_phrase": [0, 0], "change_times": []}
    assert "One voice throughout." in describe(f)
    f["global"]["speakers"] = {"count": 2, "confident": "multi", "per_phrase": [0, 1], "change_times": [2.73]}
    assert "At least 2 clearly different voices; the voice changes at 2.73 s." in describe(f)
    f["global"]["speakers"] = {"count": 1, "confident": None, "per_phrase": [0, 0], "change_times": []}
    assert "One voice" not in describe(f) and "different voices" not in describe(f)   # in between: silent
    f["global"]["recording"]["space"] = {"decay_s": 0.75, "bucket": "reverberant", "tails": 3}
    assert "the room adds reverberant (level falls at about 80 dB/s after 3 phrase ends)" in describe(f)
    f["global"]["recording"]["space"] = {"decay_s": 0.2, "bucket": "dry", "tails": 3}
    assert "the room adds" not in describe(f)                       # dry is not stated


def test_event_pulses_and_coupling_in_the_event_line(ex, tmp_path):
    f = extract(ex, tmp_path)
    f["local"]["events"] = [{"start": 3.0, "end": 3.9, "label": "laughter", "p": 0.62, "p_max": 0.7,
                             "speech_overlap": 0.0, "pulses": 3,
                             "coupling": {"level_change_db": 8.2, "brightness_change_db": 1.0}}]
    line = [x for x in describe(f).split("\n") if "laughter" in x][-1]
    assert line == "3.00–3.90 s, laughter (outside speech, p 0.62, 3 pulses); the voice is 8 dB louder in the second after it starts."
    f["local"]["events"][0].update(pulses=1, coupling={"level_change_db": 2.0})
    line = [x for x in describe(f).split("\n") if "laughter" in x][-1]
    assert line == "3.00–3.90 s, laughter (outside speech, p 0.62)."   # 1 pulse and a small change: not stated


def test_transcript_gives_syllable_counts_fillers_and_drawn_out_words(tmp_path, monkeypatch):
    from src import script
    from src.pipeline import Extractor as E
    text = "ஆ மச்சான் இங்கே பார் கோவப்படாத"                            # 1 + 2 + 2 + 1 + 5 = 11 syllables
    (tmp_path / "tr.json").write_text(json.dumps({"ta/a.wav": text}), encoding="utf-8")
    cfg = config(transcript={"file": str(tmp_path / "tr.json")})
    ex = E(cfg)
    monkeypatch.setattr(ex.syllables, "nuclei", lambda clip: np.linspace(0.6, 4.4, 11))
    f = extract(ex, tmp_path)
    w = f["local"]["words"]
    assert w["source"] == "transcript" and w["method"] == "syllable_proportional" and w["syllables"] == 11
    assert [x["text"] for x in w["words"]] == text.split()
    assert w["syllables_per_s"] == pytest.approx(11 / f["global"]["speech"]["speech_s"], rel=1e-3)   # JSON rounds
    assert [x["text"] for x in w["fillers"]] == ["ஆ"]
    out = describe(f)
    assert "11 syllables in the transcript" in out and 'hesitation sound "ஆ"' in out


def test_no_transcript_configured_means_no_words(ex, tmp_path):
    f = extract(ex, tmp_path)
    assert "words" not in f["local"] and "syllables in the transcript" not in describe(f)


def test_tempo_needs_both_detectors_to_agree():
    from src.pipeline import _tempo
    assert _tempo((7.1, 10.0), (8.4, 8.5)) is None                      # the real Sr2 false claim
    assert _tempo((10.0, 6.2), (7.3, 5.4)) == "slows_down"              # the real Sr2 true claim
    assert _tempo((4.8, 8.3), (5.0, 7.0)) == "speeds_up"
    assert _tempo((4.8, 8.3), (7.0, 5.0)) is None                       # directions disagree
    assert _tempo((4.0, 6.5), None) == "speeds_up" and _tempo((4.0, 5.0), None) is None
    assert _tempo(None, (4.0, 8.0)) is None


def test_one_bad_clip_does_not_lose_the_run(folder, base, monkeypatch):
    """A 15 000-clip job must not die on one odd file: the clip is recorded as failed and the
    rest continue (this came from a real crash in WavLM's speaker head)."""
    real = Extractor.extract

    def boom(self, p):
        if p.name == "a.wav":
            raise RuntimeError("Kernel size can't be greater than actual input size")
        return real(self, p)
    monkeypatch.setattr(Extractor, "extract", boom)
    logs = []
    s = cli.run(config(), folder, base, log=logs.append)
    res = base / s["run"]
    assert s["written"] == 1 and any("a.wav FAILED RuntimeError" in m for m in logs)
    rejected = {r["audio_path"]: r for r in load(res / "tool_descriptions.rejected.json")}
    assert "ta/a.wav" in rejected and "Kernel size" in rejected["ta/a.wav"]["checks"][0]["message"]
    assert load(res / "features" / "a.json")["status"] == "failed"
    assert [r["audio_path"] for r in load(res / "tool_descriptions.json")] == ["ta/sub/b.wav"]
