"""The lock writer: portable lines, never an empty file. No models needed."""
import subprocess

from scripts import write_lock


def test_local_file_paths_become_plain_versions(monkeypatch):
    frozen = "numpy==1.26.4\npackaging @ file:///home/conda/feedstock_root/build/work\nparler_tts @ git+https://github.com/huggingface/parler-tts.git@d108732\n"
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: frozen)
    monkeypatch.setattr(write_lock.importlib.metadata, "version", lambda name: "26.3")
    assert write_lock.lock_lines() == [
        "numpy==1.26.4",
        "packaging==26.3",
        "parler_tts @ git+https://github.com/huggingface/parler-tts.git@d108732",
    ]


def test_committed_locks_have_no_machine_specific_lines():
    import glob
    for path in glob.glob(f"{write_lock.__file__.rsplit('/scripts/', 1)[0]}/environments/*/requirements.lock.txt"):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        assert text.strip(), f"{path} is empty"
        assert "file://" not in text, f"{path} has a local file path"
