import argparse
from pathlib import Path


def rename_folder(folder):
    audio_files = sorted(
        (f for f in folder.iterdir()
         if f.is_file() and f.suffix.lower() == ".wav"),
        key=lambda f: f.name
    )

    if not audio_files:
        return

    expected_names = [f"clip_{i:02d}.wav" for i in range(1, len(audio_files) + 1)]

    if [f.name for f in audio_files] == expected_names:
        print(f"{folder}: Already in required format.")
        return

    temp_files = []

    for i, file in enumerate(audio_files, start=1):
        temp_file = file.with_name(f"__temp_{i:02d}.wav")
        file.rename(temp_file)
        temp_files.append(temp_file)

    for i, file in enumerate(temp_files, start=1):
        file.rename(folder / expected_names[i - 1])

    print(f"{folder}: Renamed {len(audio_files)} WAV files.")


def main(root):
    if not root.is_dir():
        raise ValueError(f"Invalid directory: {root}")

    folders = [root] + [p for p in root.rglob("*") if p.is_dir()]

    for folder in folders:
        rename_folder(folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args()

    main(args.dir.expanduser())