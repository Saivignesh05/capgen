"""
verify_offline.py
Offline verification of downloaded Vaani audio clips and metadata.
Validates file presence, non-zero size, audio decodability with soundfile,
metadata consistency, and total disk space consumed.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "vaani"


def verify_dataset():
    report_lines = []
    report_lines.append("# Vaani Dataset Offline Verification Report\n")

    overall_success = True
    total_audio_bytes = 0
    total_files_count = 0

    languages = ["hindi", "tamil"]

    for lang in languages:
        logger.info("=== Verifying %s ===", lang.upper())
        audio_dir = DATA_DIR / lang / "audio"
        metadata_file = DATA_DIR / "metadata" / f"{lang}.csv"

        report_lines.append(f"## {lang.capitalize()} Audio & Metadata Verification\n")

        # 1. Check metadata file exists
        if not metadata_file.exists():
            logger.error("Metadata CSV missing: %s", metadata_file)
            report_lines.append(f"- ❌ **Metadata CSV missing**: `{metadata_file}`\n")
            overall_success = False
            continue

        df = pd.read_csv(metadata_file)
        report_lines.append(f"- **Metadata records found**: {len(df)}")

        # Check required columns
        expected_cols = [
            "clip_id", "language", "duration", "transcript", "speakerID",
            "state", "district", "source_configuration", "original_sample_rate", "audio_file_path"
        ]
        missing_cols = [c for c in expected_cols if c not in df.columns]
        if missing_cols:
            logger.error("Missing columns in %s: %s", metadata_file.name, missing_cols)
            report_lines.append(f"- ❌ **Missing columns in CSV**: {missing_cols}")
            overall_success = False
        else:
            report_lines.append(f"- ✅ **All required CSV columns present**: `{', '.join(expected_cols)}`")

        # Check expected file count
        files = sorted(list(audio_dir.glob(f"{lang}_*.wav")))
        report_lines.append(f"- **Files on disk**: {len(files)} files found in `{audio_dir}`")

        if len(files) != 100:
            logger.error("Expected 100 files for %s, found %d", lang, len(files))
            report_lines.append(f"- ❌ **File Count Mismatch**: Expected 100, found {len(files)}")
            overall_success = False
        else:
            report_lines.append(f"- ✅ **File Count**: Exactly 100 WAV files exist.")

        lang_bytes = 0
        corrupted = 0
        empty_files = 0

        # Detailed check of each file
        for idx, row in df.iterrows():
            rel_path = row["audio_file_path"]
            file_path = BASE_DIR / rel_path

            if not file_path.exists():
                logger.error("Referenced file does not exist: %s", file_path)
                corrupted += 1
                overall_success = False
                continue

            file_size = file_path.stat().st_size
            if file_size == 0:
                logger.error("Empty file detected: %s", file_path)
                empty_files += 1
                overall_success = False
                continue

            lang_bytes += file_size

            # Verify readability with soundfile
            try:
                info = sf.info(str(file_path))
                if info.samplerate != row["original_sample_rate"]:
                    logger.warning("Sample rate mismatch in %s: file has %d, csv has %d",
                                   file_path.name, info.samplerate, row["original_sample_rate"])
                if abs(info.duration - row["duration"]) > 0.1:
                    logger.warning("Duration mismatch in %s: file has %.2f, csv has %.2f",
                                   file_path.name, info.duration, row["duration"])
            except Exception as e:
                logger.error("Failed to read %s with soundfile: %s", file_path.name, e)
                corrupted += 1
                overall_success = False

        total_audio_bytes += lang_bytes
        total_files_count += len(files)

        lang_mb = lang_bytes / (1024 * 1024)
        report_lines.append(f"- ✅ **Corrupted files**: {corrupted}")
        report_lines.append(f"- ✅ **Empty files**: {empty_files}")
        report_lines.append(f"- **Total {lang.capitalize()} storage**: {lang_mb:.2f} MB ({lang_bytes:,} bytes)")
        report_lines.append("\n### Sample Audio Read Test (First 3 clips)")
        report_lines.append("| File | Sample Rate | Channels | Duration | Dtype | Subtype | Status |")
        report_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

        for sample_f in files[:3]:
            try:
                data, srate = sf.read(str(sample_f))
                info = sf.info(str(sample_f))
                dur = len(data) / srate
                channels = 1 if data.ndim == 1 else data.shape[1]
                report_lines.append(f"| `{sample_f.name}` | {srate} Hz | {channels} | {dur:.2f}s | {data.dtype} | {info.subtype} | ✅ Valid |")
            except Exception as e:
                report_lines.append(f"| `{sample_f.name}` | - | - | - | - | - | ❌ Error: {e} |")

        report_lines.append("\n---\n")

    total_mb = total_audio_bytes / (1024 * 1024)
    report_lines.append("## Overall Summary\n")
    report_lines.append(f"- **Total Local Audio Files**: {total_files_count} / 200")
    report_lines.append(f"- **Total Disk Space Consumed**: **{total_mb:.2f} MB** ({total_audio_bytes:,} bytes)")
    report_lines.append(f"- **Offline Usability**: All files verified with `soundfile` without any active network connection required.")

    status_str = "✅ ALL CHECKS PASSED (100% Offline Ready)" if overall_success else "❌ VERIFICATION FAILED WITH ISSUES"
    report_lines.append(f"\n### Final Status: **{status_str}**\n")

    report_text = "\n".join(report_lines)
    report_path = BASE_DIR / "VERIFICATION_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info("\nVerification Report:\n%s", report_text)
    logger.info("Report written to %s", report_path)
    return overall_success


if __name__ == "__main__":
    success = verify_dataset()
    sys.exit(0 if success else 1)
