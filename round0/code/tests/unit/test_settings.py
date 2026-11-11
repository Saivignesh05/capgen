"""settings.yaml and the classes that check it. No models needed."""
import os

import pytest
import yaml
from pydantic import ValidationError

from helpers import EVAL_SETTINGS_PATH, SETTINGS_PATH
from pipeline.settings import (DownloadSettings, Settings, asr_settings, captioner_settings,
                               generator_settings, load_settings, model_folder)

VALID_HASH = "a" * 40


def write_settings(tmp_path, source: str, changes: dict) -> str:
    """Copy one of the repository's settings files with model_dir set to tmp_path, apply changes, return the path."""
    with open(source, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["model_dir"] = str(tmp_path)
    for key, value in changes.items():
        data[key] = value
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


def test_repository_settings_load(tmp_path):
    settings = load_settings(write_settings(tmp_path, SETTINGS_PATH, {}))
    assert isinstance(settings, Settings)


def test_evaluation_settings_load(tmp_path):
    settings = load_settings(write_settings(tmp_path, EVAL_SETTINGS_PATH, {}))
    assert settings.generation.samples_per_item == 10


def test_devset_and_evaluation_settings_differ_only_in_samples_per_item():
    with open(SETTINGS_PATH, encoding="utf-8") as f:
        devset = yaml.safe_load(f)
    with open(EVAL_SETTINGS_PATH, encoding="utf-8") as f:
        evaluation = yaml.safe_load(f)
    del devset["generation"]["samples_per_item"]
    del evaluation["generation"]["samples_per_item"]
    assert devset == evaluation


def test_repository_settings_have_one_folder_per_repo(tmp_path):
    settings = load_settings(write_settings(tmp_path, SETTINGS_PATH, {}))
    folders = [model_folder(settings, repo) for _name, repo in settings.downloads.named()]
    assert len(folders) == len(set(folders))


def test_model_dir_must_exist(tmp_path):
    path = write_settings(tmp_path, SETTINGS_PATH, {})
    with open(path) as f:
        data = yaml.safe_load(f)
    data["model_dir"] = str(tmp_path / "missing")
    with open(path, "w") as f:
        yaml.safe_dump(data, f)
    with pytest.raises(ValidationError, match="No such directory"):
        load_settings(path)


def test_unknown_key_is_rejected(tmp_path):
    with pytest.raises(ValidationError):
        load_settings(write_settings(tmp_path, SETTINGS_PATH, {"captioner_typo": {}}))


def test_misspelled_field_is_rejected(tmp_path):
    with pytest.raises(ValidationError):
        load_settings(write_settings(tmp_path, SETTINGS_PATH, {"asr": {"max_clip_second": 30.0}}))


def test_missing_section_is_rejected(tmp_path):
    path = write_settings(tmp_path, SETTINGS_PATH, {})
    with open(path) as f:
        data = yaml.safe_load(f)
    del data["generation"]
    with open(path, "w") as f:
        yaml.safe_dump(data, f)
    with pytest.raises(ValidationError):
        load_settings(path)


def test_unknown_dtype_is_rejected(tmp_path):
    captioner = {"dtype": "int8", "attn_implementation": "sdpa", "max_new_tokens": 1024, "max_clip_seconds": 30.0}
    with pytest.raises(ValidationError):
        load_settings(write_settings(tmp_path, SETTINGS_PATH, {"captioner": captioner}))


def test_model_settings_need_downloaded_folders(tmp_path):
    settings = load_settings(write_settings(tmp_path, SETTINGS_PATH, {}))
    for build in [captioner_settings, asr_settings, generator_settings]:
        with pytest.raises(ValidationError, match="No such directory"):
            build(settings)


def test_model_settings_combine_options_and_folder(tmp_path):
    settings = load_settings(write_settings(tmp_path, SETTINGS_PATH, {}))
    for _name, repo in settings.downloads.named():
        os.makedirs(model_folder(settings, repo))
    captioner = captioner_settings(settings)
    assert captioner.model_dir == os.path.join(str(tmp_path), settings.downloads.captioner.repo_id)
    assert captioner.dtype == settings.captioner.dtype
    generator = generator_settings(settings)
    assert generator.description_tokenizer_dir == os.path.join(str(tmp_path), settings.downloads.description_tokenizer.repo_id)


def test_settings_cannot_be_changed(tmp_path):
    settings = load_settings(write_settings(tmp_path, SETTINGS_PATH, {}))
    with pytest.raises(ValidationError):
        settings.generation.seed = 1


def test_download_accepts_valid_settings():
    assert DownloadSettings(repo_id="owner/name", revision=VALID_HASH, allow_patterns=None).repo_id == "owner/name"


@pytest.mark.parametrize("revision", ["main", "a2bd106", "A" * 40])
def test_download_rejects_anything_but_a_full_commit_hash(revision):
    with pytest.raises(ValidationError, match="commit hash"):
        DownloadSettings(repo_id="owner/name", revision=revision, allow_patterns=None)


@pytest.mark.parametrize("repo_id", ["name", "owner/", "a/b/c", ""])
def test_download_rejects_a_bad_repo_id(repo_id):
    with pytest.raises(ValidationError, match="owner/name"):
        DownloadSettings(repo_id=repo_id, revision=VALID_HASH, allow_patterns=None)


def test_download_rejects_an_empty_file_list():
    with pytest.raises(ValidationError, match="download nothing"):
        DownloadSettings(repo_id="owner/name", revision=VALID_HASH, allow_patterns=[])
