"""The training manifest and the run manifest."""
import hashlib
import json

import pytest

from evaluation.manifest import RunManifest, file_sha256, folder_sha256, write_manifest
from training_data.manifest import TrainingExample, training_manifest_sha256, write_training_manifest

SHA_A = "a" * 64
SHA_B = "b" * 64


def example(source_id: str, round_number: int, removal_reason: str | None) -> TrainingExample:
    return TrainingExample(source_id=source_id, audio_sha256=SHA_A, round=round_number, label_producer="captioner_round_0",
                           label_producer_checkpoint_sha256=SHA_B, settings={"seed": "0", "decoding": "greedy"},
                           removal_reason=removal_reason)


def test_manifest_bytes_are_exactly_as_documented(tmp_path):
    path = str(tmp_path / "manifest.csv")
    write_training_manifest([example("pool/clip_2", 1, "language check failed"), example("pool/clip_1", 1, None)], path)
    expected = ("source_id,audio_sha256,round,label_producer,label_producer_checkpoint_sha256,settings,removal_reason\n"
                f'pool/clip_1,{SHA_A},1,captioner_round_0,{SHA_B},"{{""decoding"": ""greedy"", ""seed"": ""0""}}",\n'
                f'pool/clip_2,{SHA_A},1,captioner_round_0,{SHA_B},"{{""decoding"": ""greedy"", ""seed"": ""0""}}",language check failed\n')
    with open(path, encoding="utf-8", newline="") as f:
        assert f.read() == expected
    assert training_manifest_sha256(path) == hashlib.sha256(expected.encode("utf-8")).hexdigest()


def test_manifest_hash_does_not_depend_on_row_order(tmp_path):
    examples = [example("pool/clip_3", 2, None), example("pool/clip_1", 1, None), example("pool/clip_2", 1, "too noisy")]
    first = str(tmp_path / "first.csv")
    second = str(tmp_path / "second.csv")
    write_training_manifest(examples, first)
    write_training_manifest(list(reversed(examples)), second)
    assert training_manifest_sha256(first) == training_manifest_sha256(second)


def test_manifest_hash_changes_with_the_content(tmp_path):
    first = str(tmp_path / "first.csv")
    second = str(tmp_path / "second.csv")
    write_training_manifest([example("pool/clip_1", 1, None)], first)
    write_training_manifest([example("pool/clip_1", 1, "removed")], second)
    assert training_manifest_sha256(first) != training_manifest_sha256(second)


def test_the_same_example_twice_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="appears twice"):
        write_training_manifest([example("pool/clip_1", 1, None), example("pool/clip_1", 1, "removed")], str(tmp_path / "m.csv"))


def test_training_example_checks_its_fields():
    with pytest.raises(ValueError):
        TrainingExample(source_id="x", audio_sha256="not-a-hash", round=1, label_producer="p",
                        label_producer_checkpoint_sha256=SHA_B, settings={}, removal_reason=None)
    with pytest.raises(ValueError):
        example("x", 1, "  ")
    with pytest.raises(ValueError):
        example("x", -1, None)


def test_file_and_folder_hashes(tmp_path):
    model = tmp_path / "model"
    (model / "sub").mkdir(parents=True)
    (model / "config.json").write_bytes(b"{}")
    (model / "sub" / "weights.bin").write_bytes(b"\x00\x01")
    assert file_sha256(str(model / "config.json")) == hashlib.sha256(b"{}").hexdigest()
    lines = f"config.json\n{hashlib.sha256(b'{}').hexdigest()}\nsub/weights.bin\n{hashlib.sha256(bytes([0, 1])).hexdigest()}\n"
    assert folder_sha256(str(model)) == hashlib.sha256(lines.encode("utf-8")).hexdigest()
    # caches are not part of the model
    (model / "__pycache__").mkdir()
    (model / "__pycache__" / "x.pyc").write_bytes(b"cache")
    assert folder_sha256(str(model)) == hashlib.sha256(lines.encode("utf-8")).hexdigest()


def test_run_manifest_is_written_as_json(tmp_path):
    path = str(tmp_path / "manifest.json")
    manifest = RunManifest(script="scripts/ladder_rung2.py", started_utc="2026-11-14T00:00:00Z", code_commit="abc",
                           settings_sha256=SHA_A, input_sha256={"rubric": SHA_B}, checkpoint_sha256={"grader": SHA_A},
                           notes={"extractor": "rules"})
    write_manifest(manifest, path)
    with open(path, encoding="utf-8") as f:
        assert RunManifest.model_validate(json.load(f)) == manifest
