"""The submission zip: what goes in, and what stops it. No models needed."""
import os

from scripts import package


def make_round0(tmp_path, lock_text: str) -> str:
    root = tmp_path / "round0"
    for relative, text in {
        "README.md": "Read me.",
        "paths.yaml": "models: /somewhere/personal",
        "checkpoint_hashes.json": "{}",
        "code/scripts/run.py": "print('hi')",
        "code/scripts/__pycache__/run.cpython-310.pyc": "x",
        "code/tests/data/hi.wav": "voice",
        "code/environments/asr-captioner/requirements.lock.txt": lock_text,
        "code/environments/generator/requirements.lock.txt": lock_text,
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return str(root)


def test_personal_and_generated_files_are_left_out(tmp_path, monkeypatch):
    monkeypatch.setattr(package, "ROUND0_FOLDER", make_round0(tmp_path, "numpy==1.26.4\n"))
    files = package.files_to_package()
    assert "README.md" in files and "checkpoint_hashes.json" in files
    assert "paths.yaml" not in files
    assert not any("__pycache__" in f or f.startswith(os.path.join("code", "tests", "data")) for f in files)


def test_a_good_folder_has_no_problems(tmp_path, monkeypatch):
    monkeypatch.setattr(package, "ROUND0_FOLDER", make_round0(tmp_path, "numpy==1.26.4\n"))
    assert package.problems(package.files_to_package()) == []


def test_empty_locks_and_missing_hashes_are_problems(tmp_path, monkeypatch):
    root = make_round0(tmp_path, "")
    os.remove(os.path.join(root, "checkpoint_hashes.json"))
    monkeypatch.setattr(package, "ROUND0_FOLDER", root)
    found = " ".join(package.problems(package.files_to_package()))
    assert "checkpoint_hashes.json is missing" in found and "missing or empty" in found


def test_a_file_mentioning_the_home_folder_is_a_problem(tmp_path, monkeypatch):
    root = make_round0(tmp_path, "numpy==1.26.4\n")
    with open(os.path.join(root, "README.md"), "w") as f:
        f.write(f"models are in {os.path.expanduser('~')}/models")
    monkeypatch.setattr(package, "ROUND0_FOLDER", root)
    assert any("home folder" in problem for problem in package.problems(package.files_to_package()))
