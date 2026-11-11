"""The runner's planning: which clips and samples are still to do. No models needed."""
import pytest

pytest.importorskip("soundfile")        # scripts.records imports pipeline.audio_types

from pipeline.seeds import sample_seed
from scripts.records import AudioRecord, ClipRecord, DescriptionRecord, TranscriptRecord
from scripts.run import find_clips, generation_tasks, missing_clips


def clip(clip_id: str) -> ClipRecord:
    return ClipRecord(clip_id=clip_id, path=f"/clips/{clip_id}.wav", sha256="0" * 64)


def test_find_clips_searches_subfolders_and_records_hashes(tmp_path):
    (tmp_path / "hi").mkdir()
    (tmp_path / "hi" / "a.wav").write_bytes(b"a")
    (tmp_path / "b.wav").write_bytes(b"b")
    clips = find_clips(str(tmp_path))
    assert sorted(c.clip_id for c in clips) == ["a", "b"]
    assert all(len(c.sha256) == 64 for c in clips)


def test_find_clips_rejects_two_clips_with_the_same_name(tmp_path):
    for folder in ["hi", "ta"]:
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "same.wav").write_bytes(b"x")
    with pytest.raises(ValueError, match="share the id"):
        find_clips(str(tmp_path))


def test_find_clips_rejects_an_empty_folder(tmp_path):
    with pytest.raises(ValueError, match="no .wav"):
        find_clips(str(tmp_path))


def test_missing_clips_are_those_without_output():
    clips = [clip("a"), clip("b"), clip("c")]
    done = [DescriptionRecord(clip_id="b", description="x")]
    assert [c.clip_id for c in missing_clips(clips, done)] == ["a", "c"]


def test_generation_needs_both_a_transcript_and_a_description():
    clips = [clip("a"), clip("b")]
    transcripts = [TranscriptRecord(clip_id="a", language="hi", transcript="नमस्ते"), TranscriptRecord(clip_id="b", language="ta", transcript="வணக்கம்")]
    descriptions = [DescriptionRecord(clip_id="a", description="A calm voice.")]
    tasks = generation_tasks(clips, transcripts, descriptions, [], samples_per_item=3, base_seed=0)
    assert [(t.clip_id, t.sample) for t in tasks] == [("a", 0), ("a", 1), ("a", 2)]


def test_clips_with_no_speech_are_not_spoken():
    clips = [clip("a")]
    transcripts = [TranscriptRecord(clip_id="a", language="hi", transcript="")]
    descriptions = [DescriptionRecord(clip_id="a", description="Silence.")]
    assert generation_tasks(clips, transcripts, descriptions, [], samples_per_item=3, base_seed=0) == []


def test_samples_already_written_are_not_made_again_and_keep_their_seed():
    clips = [clip("a")]
    transcripts = [TranscriptRecord(clip_id="a", language="hi", transcript="नमस्ते")]
    descriptions = [DescriptionRecord(clip_id="a", description="A calm voice.")]
    done = [AudioRecord(clip_id="a", sample=0, seed=sample_seed(7, "a", 0), file="audio/a/0.wav")]
    tasks = generation_tasks(clips, transcripts, descriptions, done, samples_per_item=2, base_seed=7)
    assert [(t.sample, t.seed) for t in tasks] == [(1, sample_seed(7, "a", 1))]
