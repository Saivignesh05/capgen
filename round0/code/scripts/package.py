"""Make the submission zip of round 0: everything needed to reproduce it, nothing personal.

Started by `./round0 package <zip file>`. It refuses to write the zip if something a reviewer needs
is missing, or if any file mentions your home folder. See CAREFUL codes at the bottom.
"""
import os
import sys
import zipfile

ROUND0_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SKIPPED_FILES = {"paths.yaml"}      # personal to each machine. CAREFUL P1
SKIPPED_FOLDERS = {"__pycache__", ".pytest_cache", ".git"}
SKIPPED_SUFFIXES = (".pyc", ".partial", ".tar.gz", ".zip")
SKIPPED_PREFIXES = (os.path.join("code", "tests", "data") + os.sep,)       # voice recordings. CAREFUL P2
TEXT_SUFFIXES = (".py", ".sh", ".md", ".yaml", ".yml", ".txt", ".json", ".ini", "")


def files_to_package() -> list[str]:
    """Every file to include, as paths relative to the round0 folder, sorted."""
    files = []
    for root, folders, names in os.walk(ROUND0_FOLDER):
        folders[:] = sorted(folder for folder in folders if folder not in SKIPPED_FOLDERS)
        for name in sorted(names):
            relative = os.path.relpath(os.path.join(root, name), ROUND0_FOLDER)
            if name in SKIPPED_FILES or name.endswith(SKIPPED_SUFFIXES) or relative.startswith(SKIPPED_PREFIXES):
                continue
            files.append(relative)
    return files


def problems(files: list[str]) -> list[str]:
    """Everything that must be fixed before the zip can be submitted."""
    found = []
    if "checkpoint_hashes.json" not in files:
        found.append("checkpoint_hashes.json is missing: run ./round0 freeze")
    for name in ["asr-captioner", "generator"]:
        lock = os.path.join("code", "environments", name, "requirements.lock.txt")
        path = os.path.join(ROUND0_FOLDER, lock)
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            found.append(f"{lock} is missing or empty: run ./round0 setup")
        elif "file://" in open(path, encoding="utf-8").read():
            found.append(f"{lock} has a local file:// path: delete it and run ./round0 setup")

    home = os.path.expanduser("~")
    personal = {home, os.path.realpath(home)}       # CAREFUL P3
    for relative in files:
        if not relative.endswith(TEXT_SUFFIXES):
            continue
        try:
            text = open(os.path.join(ROUND0_FOLDER, relative), encoding="utf-8").read()
        except UnicodeDecodeError:
            continue
        for path in personal:
            if path in text:
                found.append(f"{relative} mentions your home folder ({path}): remove it")
    return found


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m scripts.package <zip file to write>")
    zip_path = os.path.abspath(sys.argv[1])
    if zip_path.startswith(ROUND0_FOLDER + os.sep):
        raise SystemExit("Write the zip outside the round0 folder.")

    files = files_to_package()
    found = problems(files)
    if found:
        print("Not packaging, fix these first:", file=sys.stderr)
        for problem in found:
            print(f"  - {problem}", file=sys.stderr)
        sys.exit(1)

    with zipfile.ZipFile(zip_path + ".partial", "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in files:
            archive.write(os.path.join(ROUND0_FOLDER, relative), os.path.join("round0", relative))
    os.replace(zip_path + ".partial", zip_path)
    print(f"Wrote {zip_path}: {len(files)} files, {os.path.getsize(zip_path) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------------------------
# CAREFUL codes
#
# P1   paths.yaml holds your machine's folders, so it is never packaged; reviewers make their own
#      from paths.example.yaml.
# P2   tests/data holds optional speech clips for the ASR tests. Recordings of people's voices are
#      not ours to redistribute, so they are never packaged.
# P3   Any text file that contains your home folder's path (for example in an example or a log)
#      would show your username, so packaging stops and names the file.
# ---------------------------------------------------------------------------------------------
