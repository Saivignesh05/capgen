"""Helpers shared by the tests.

Every test that needs something this computer may not have (a package, a downloaded model, a GPU)
skips with a reason instead of failing. Run `pytest -rs` to see the reasons.
"""
import os

import pytest

from claims.settings import ClaimsSettings
from claims.settings import load_settings as load_claims_settings
from evaluation.rubric.schema import RubricItem
from evaluation.settings import EvaluationSettings, Paths, load_paths, load_settings

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_PATH = os.path.join(REPO_DIR, "settings", "phase0.yaml")
PATHS_PATH = os.path.join(REPO_DIR, "settings", "paths.yaml")     # your machine's folders (not committed)


def evaluation_settings() -> EvaluationSettings:
    return load_settings(SETTINGS_PATH)


def claims_settings() -> ClaimsSettings:
    return load_claims_settings(SETTINGS_PATH)


def machine_paths() -> Paths:
    """settings/paths.yaml, or skip the test if it is not there."""
    try:
        return load_paths(PATHS_PATH)
    except Exception as error:
        pytest.skip(f"cannot load settings/paths.yaml (copy paths.example.yaml to paths.yaml and set its folders): {error}")


def require_model_folder(folder: str) -> str:
    """Skip unless the model folder exists (downloaded by scripts/download_models.py)."""
    if not os.path.isdir(folder):
        pytest.skip(f"model not downloaded: {folder}")
    return folder


def require_cuda(min_total_gb: float) -> None:
    """Skip unless CUDA GPUs with at least `min_total_gb` of memory in total are visible."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU on this computer")
    total_gb = sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())) / 1e9
    if total_gb < min_total_gb:
        pytest.skip(f"needs {min_total_gb:.0f} GB of GPU memory, this computer has {total_gb:.0f} GB")


def rubric_item(item_id: str, truth: bool, weight: int) -> RubricItem:
    """A global rubric item of clip hi/clip_a with the given id, truth and weight."""
    return RubricItem(clip_id="hi/clip_a", item_id=item_id, axis="sound", item_type="Speaker",
                      statement=f"statement {item_id}", truth=truth, start_s=None, end_s=None,
                      weight=weight, provenance="human:test")
