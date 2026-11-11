"""Checkpoint hashing in scripts/hash_checkpoints.py, on small fake model folders. No models needed."""
import hashlib
import json

import pytest

pytest.importorskip("huggingface_hub")      # scripts.hash_checkpoints imports scripts.download

from scripts import hash_checkpoints
from scripts.download import MARKER_NAME


def make_checkpoint(folder) -> None:
    (folder / "sub").mkdir(parents=True)
    (folder / "config.json").write_text('{"a": 1}')
    (folder / "sub" / "weights.bin").write_bytes(b"\x00\x01\x02")
    (folder / MARKER_NAME).write_text("not part of the model")
    (folder / ".cache").mkdir()
    (folder / ".cache" / "bookkeeping").write_text("not part of the model")


def test_marker_and_cache_are_not_part_of_the_checkpoint(tmp_path):
    make_checkpoint(tmp_path)
    assert hash_checkpoints.checkpoint_files(str(tmp_path)) == ["config.json", "sub/weights.bin"]


def test_file_hashes_match_sha256(tmp_path):
    make_checkpoint(tmp_path)
    result = hash_checkpoints.hash_checkpoint(str(tmp_path))
    assert result["files"]["sub/weights.bin"] == hashlib.sha256(b"\x00\x01\x02").hexdigest()


def test_checkpoint_hash_is_the_hash_of_the_sha256sum_listing(tmp_path):
    make_checkpoint(tmp_path)
    result = hash_checkpoints.hash_checkpoint(str(tmp_path))
    listing = "".join(f"{result['files'][name]}  {name}\n" for name in ["config.json", "sub/weights.bin"])
    assert result["sha256"] == hashlib.sha256(listing.encode()).hexdigest()


def test_any_changed_byte_changes_the_checkpoint_hash(tmp_path):
    make_checkpoint(tmp_path)
    before = hash_checkpoints.hash_checkpoint(str(tmp_path))["sha256"]
    (tmp_path / "sub" / "weights.bin").write_bytes(b"\x00\x01\x03")
    assert hash_checkpoints.hash_checkpoint(str(tmp_path))["sha256"] != before


def test_hashes_are_plain_json(tmp_path):
    make_checkpoint(tmp_path)
    json.dumps(hash_checkpoints.hash_checkpoint(str(tmp_path)))
