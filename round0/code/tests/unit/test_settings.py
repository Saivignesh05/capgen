"""settings.yaml, paths.yaml and the classes that check them. No models needed."""
import os

import pytest
import yaml
from pydantic import ValidationError

from helpers import EVAL_SETTINGS_PATH, EXAMPLE_PATHS_PATH, SETTINGS_PATH
from pipeline.settings import (DownloadSettings, Paths, Settings, asr_settings, captioner_settings,
                               generator_settings, load_paths, load_settings, model_folder)

VALID_HASH = "a" * 40


def write_yaml(tmp_path, name: str, data: dict) -> str:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data))
    return str(path)


def read_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def tmp_paths(tmp_path) -> Paths:
    return Paths(models=str(tmp_path / "models"), environments=str(tmp_path / "envs"), runs=str(tmp_path / "runs"))


# ---- settings.yaml ------------------------------------------------------------------------

def test_repository_settings_load():
    assert isinstance(load_settings(SETTINGS_PATH), Settings)
    assert load_settings(SETTINGS_PATH).generation.samples_per_item == 1


def test_evaluation_settings_load():
    assert load_settings(EVAL_SETTINGS_PATH).generation.samples_per_item == 10


def test_devset_and_evaluation_settings_differ_only_in_samples_per_item():
    devset, evaluation = read_yaml(SETTINGS_PATH), read_yaml(EVAL_SETTINGS_PATH)
    del devset["generation"]["samples_per_item"]
    del evaluation["generation"]["samples_per_item"]
    assert devset == evaluation


def test_settings_hold_no_machine_paths():
    assert "model_dir" not in read_yaml(SETTINGS_PATH)


def test_unknown_key_is_rejected(tmp_path):
    data = read_yaml(SETTINGS_PATH)
    data["captioner_typo"] = {}
    with pytest.raises(ValidationError):
        load_settings(write_yaml(tmp_path, "settings.yaml", data))


def test_misspelled_field_is_rejected(tmp_path):
    data = read_yaml(SETTINGS_PATH)
    data["asr"] = {"max_clip_second": 30.0}
    with pytest.raises(ValidationError):
        load_settings(write_yaml(tmp_path, "settings.yaml", data))


def test_missing_section_is_rejected(tmp_path):
    data = read_yaml(SETTINGS_PATH)
    del data["generation"]
    with pytest.raises(ValidationError):
        load_settings(write_yaml(tmp_path, "settings.yaml", data))


def test_unknown_dtype_is_rejected(tmp_path):
    data = read_yaml(SETTINGS_PATH)
    data["captioner"]["dtype"] = "int8"
    with pytest.raises(ValidationError):
        load_settings(write_yaml(tmp_path, "settings.yaml", data))


def test_settings_cannot_be_changed():
    settings = load_settings(SETTINGS_PATH)
    with pytest.raises(ValidationError):
        settings.generation.seed = 1


# ---- paths.yaml ---------------------------------------------------------------------------

def test_example_paths_file_has_the_three_folders_and_is_rejected_as_is():
    assert set(read_yaml(EXAMPLE_PATHS_PATH)) == {"models", "environments", "runs"}
    with pytest.raises(ValidationError, match="example value"):
        load_paths(EXAMPLE_PATHS_PATH)


def test_paths_load(tmp_path):
    data = {"models": "/data/models", "environments": "/data/envs", "runs": "/data/runs"}
    assert load_paths(write_yaml(tmp_path, "paths.yaml", data)).models == "/data/models"


def test_paths_must_be_absolute(tmp_path):
    data = {"models": "../models", "environments": "/data/envs", "runs": "/data/runs"}
    with pytest.raises(ValidationError, match="absolute path"):
        load_paths(write_yaml(tmp_path, "paths.yaml", data))


def test_example_paths_are_rejected(tmp_path):
    data = {"models": "/absolute/path/to/models", "environments": "/data/envs", "runs": "/data/runs"}
    with pytest.raises(ValidationError, match="example value"):
        load_paths(write_yaml(tmp_path, "paths.yaml", data))


# ---- model settings: settings + paths -----------------------------------------------------

def test_one_folder_per_repo(tmp_path):
    settings, paths = load_settings(SETTINGS_PATH), tmp_paths(tmp_path)
    folders = [model_folder(paths, repo) for _name, repo in settings.downloads.named()]
    assert len(folders) == len(set(folders))


def test_model_settings_need_downloaded_folders(tmp_path):
    settings, paths = load_settings(SETTINGS_PATH), tmp_paths(tmp_path)
    for build in [captioner_settings, asr_settings, generator_settings]:
        with pytest.raises(ValidationError, match="No such directory"):
            build(settings, paths)


def test_model_settings_combine_options_and_folder(tmp_path):
    settings, paths = load_settings(SETTINGS_PATH), tmp_paths(tmp_path)
    for _name, repo in settings.downloads.named():
        os.makedirs(model_folder(paths, repo))
    captioner = captioner_settings(settings, paths)
    assert captioner.model_dir == os.path.join(paths.models, settings.downloads.captioner.repo_id)
    assert captioner.dtype == settings.captioner.dtype
    generator = generator_settings(settings, paths)
    assert generator.description_tokenizer_dir == os.path.join(paths.models, settings.downloads.description_tokenizer.repo_id)


# ---- downloads ----------------------------------------------------------------------------

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
