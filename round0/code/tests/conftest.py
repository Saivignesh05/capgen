"""Fixtures shared by the tests. Helper functions are in helpers.py."""
import pytest

from helpers import write_tone


@pytest.fixture
def tone_path(tmp_path) -> str:
    """A 3-second tone as a .wav file."""
    return write_tone(str(tmp_path / "tone.wav"), seconds=3.0, sample_rate=16000)
