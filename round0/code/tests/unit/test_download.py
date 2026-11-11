"""Skip and resume logic of scripts/download.py, with the real download replaced by a fake. No internet needed."""
import os

import pytest

pytest.importorskip("huggingface_hub")

from pipeline.settings import DownloadSettings
from scripts import download

VALID_HASH = "a" * 40


@pytest.fixture
def fake_snapshot_download(monkeypatch):
    """Replace the real download with one that only creates the folder, and record each call."""
    calls = []

    def fake(repo_id, revision, local_dir, allow_patterns):
        os.makedirs(local_dir, exist_ok=True)
        calls.append(repo_id)

    monkeypatch.setattr(download, "snapshot_download", fake)
    return calls


def test_first_run_downloads_and_writes_a_marker(tmp_path, fake_snapshot_download):
    repo = DownloadSettings(repo_id="owner/name", revision=VALID_HASH, allow_patterns=None)
    download.download(repo, str(tmp_path / "owner" / "name"))
    assert fake_snapshot_download == ["owner/name"]
    assert os.path.isfile(tmp_path / "owner" / "name" / download.MARKER_NAME)


def test_second_run_skips_a_finished_download(tmp_path, fake_snapshot_download):
    repo = DownloadSettings(repo_id="owner/name", revision=VALID_HASH, allow_patterns=None)
    download.download(repo, str(tmp_path / "owner" / "name"))
    download.download(repo, str(tmp_path / "owner" / "name"))
    assert fake_snapshot_download == ["owner/name"]


def test_a_changed_revision_raises_instead_of_mixing_versions(tmp_path, fake_snapshot_download):
    download.download(DownloadSettings(repo_id="owner/name", revision=VALID_HASH, allow_patterns=None), str(tmp_path / "owner" / "name"))
    with pytest.raises(ValueError, match="already holds"):
        download.download(DownloadSettings(repo_id="owner/name", revision="b" * 40, allow_patterns=None), str(tmp_path / "owner" / "name"))


def test_an_interrupted_download_leaves_no_marker_and_runs_again(tmp_path, monkeypatch):
    def crash(repo_id, revision, local_dir, allow_patterns):
        os.makedirs(local_dir, exist_ok=True)
        raise ConnectionError("network dropped")

    repo = DownloadSettings(repo_id="owner/name", revision=VALID_HASH, allow_patterns=None)
    monkeypatch.setattr(download, "snapshot_download", crash)
    with pytest.raises(ConnectionError):
        download.download(repo, str(tmp_path / "owner" / "name"))
    assert not os.path.isfile(tmp_path / "owner" / "name" / download.MARKER_NAME)

    calls = []
    monkeypatch.setattr(download, "snapshot_download", lambda **kwargs: calls.append(kwargs["repo_id"]))
    download.download(repo, str(tmp_path / "owner" / "name"))
    assert calls == ["owner/name"]
