from dataclasses import dataclass
from pathlib import Path
from typing import List
import argparse
import soundfile as sf


MAX_DURATION_SECONDS = 30.0
AUDIO_EXTENSIONS = {".wav"}


@dataclass
class Violation:
    condition: str
    audio_path: str
    error: str


def _check_valid_wav(audio_paths: List[Path]) -> List[Violation]:
    violations = []

    for path in audio_paths:
        try:
            info = sf.info(str(path))

            if info.format != "WAV":
                raise ValueError(f"Expected WAV format, found {info.format}.")

            # Decode the entire file to catch damaged/truncated audio data,
            # not just an invalid or unreadable WAV header.
            decoded_frames = 0
            with sf.SoundFile(str(path)) as audio_file:
                while True:
                    chunk = audio_file.read(
                        frames=8192,
                        dtype="float32",
                        always_2d=True,
                    )
                    if len(chunk) == 0:
                        break
                    decoded_frames += len(chunk)

            if decoded_frames != info.frames:
                raise ValueError(
                    f"Incomplete audio data: expected {info.frames} frames, "
                    f"decoded {decoded_frames}."
                )

        except Exception as exc:
            violations.append(
                Violation(
                    condition="Valid WAV file",
                    audio_path=str(path),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    return violations


def _check_duration(audio_paths: List[Path]) -> List[Violation]:
    violations = []

    for path in audio_paths:
        try:
            info = sf.info(str(path))
            duration = info.frames / info.samplerate

            if duration >= MAX_DURATION_SECONDS:
                violations.append(
                    Violation(
                        condition="Duration < 30 seconds",
                        audio_path=str(path),
                        error=(
                            f"Duration is {duration:.3f} seconds; "
                            f"expected strictly less than {MAX_DURATION_SECONDS:g} seconds."
                        ),
                    )
                )

        except Exception as exc:
            violations.append(
                Violation(
                    condition="Duration < 30 seconds",
                    audio_path=str(path),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    return violations


def _collect_wav_paths(directory: Path) -> List[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
    )


def _print_condition_report(
    condition: str,
    total_files: int,
    violations: List[Violation],
) -> None:
    passed = total_files - len(violations)

    if violations:
        print(f"\n✗ {condition}: {passed}/{total_files} passed")
        for violation in violations:
            print(
                f"  {violation.audio_path} | "
                f"{violation.error}"
            )
    else:
        print(f"\n✓ {condition}: {passed}/{total_files} passed")
        print("  No violation objects.")


def main() -> List[Violation]:
    parser = argparse.ArgumentParser(description="Validate WAV audio files.")
    parser.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="Directory containing audio files",
    )
    args = parser.parse_args()

    directory = args.dir.expanduser()

    if not directory.is_dir():
        print(f"Error: '{directory}' is not a valid directory.")
        return []

    audio_paths = _collect_wav_paths(directory)

    if not audio_paths:
        print("No WAV files found.")
        return []

    print(f"\nFound {len(audio_paths)} WAV file(s).")

    wav_violations = _check_valid_wav(audio_paths)
    duration_violations = _check_duration(audio_paths)

    _print_condition_report("Valid WAV file", len(audio_paths), wav_violations)
    _print_condition_report(
        "Duration strictly less than 30 seconds",
        len(audio_paths),
        duration_violations,
    )

    all_violations = wav_violations + duration_violations
    print(f"\nTotal violation objects: {len(all_violations)}")

    return all_violations


if __name__ == "__main__":
    main()
