"""Helpers shared by the tests (fixtures are in conftest.py).

Every test that needs something this computer may not have (a package, a downloaded model, a GPU,
a speech clip) skips with a reason instead of failing. Run `pytest -rs` to see the reasons.
"""
import os

import numpy as np
import pytest

from pipeline.settings import Paths, Settings, load_paths, load_settings

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(CODE_DIR, "tests", "data")
SETTINGS_PATH = os.path.join(os.path.dirname(CODE_DIR), "settings.yaml")      # round0/settings.yaml (devset runs)
EVAL_SETTINGS_PATH = os.path.join(os.path.dirname(CODE_DIR), "settings_eval.yaml")      # round0/settings_eval.yaml
PATHS_PATH = os.path.join(os.path.dirname(CODE_DIR), "paths.yaml")      # round0/paths.yaml (your machine's folders)
EXAMPLE_PATHS_PATH = os.path.join(os.path.dirname(CODE_DIR), "paths.example.yaml")


def repo_settings_and_paths() -> tuple[Settings, Paths]:
    """The repository's settings.yaml and paths.yaml, or skip the whole test file if they cannot be loaded."""
    try:
        return load_settings(SETTINGS_PATH), load_paths(PATHS_PATH)
    except Exception as error:
        pytest.skip(f"cannot load settings.yaml or paths.yaml (copy paths.example.yaml to paths.yaml and set its folders): {error}", allow_module_level=True)


def require_cuda(min_total_gb: float) -> None:
    """Skip unless CUDA GPUs with at least `min_total_gb` of memory in total are visible."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU on this computer")
    total_gb = sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())) / 1e9
    if total_gb < min_total_gb:
        pytest.skip(f"needs {min_total_gb:.0f} GB of GPU memory, this computer has {total_gb:.0f} GB")


def write_tone(path: str, seconds: float, sample_rate: int) -> str:
    """Write a quiet 440 Hz tone to `path` as a .wav file and return the path."""
    soundfile = pytest.importorskip("soundfile")
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    soundfile.write(path, (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sample_rate)
    return path


def speech_path(language: str) -> str:
    """Path of a real speech clip in tests/data/<language>.wav, or skip if it is not there."""
    path = os.path.join(DATA_DIR, f"{language}.wav")
    if not os.path.isfile(path):
        pytest.skip(f"no speech clip at {path} (optional, see EXPLANATION.md: Tests)")
    return path
