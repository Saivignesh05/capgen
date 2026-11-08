"""
download_vaani.py
Downloads a local copy of 100 Hindi and 100 Tamil audio samples from ARTPARK-IISc/Vaani.
Preserves original audio content, sample rate (16kHz), and creates metadata CSVs.
"""

import io
import os
import sys
import csv
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Hugging Face compatibility shims for environment consistency
import httpx
import huggingface_hub
import huggingface_hub.errors
import huggingface_hub.hf_file_system
import huggingface_hub.utils

for attr in ["BucketNotFoundError"]:
    if not hasattr(huggingface_hub.errors, attr):
        setattr(huggingface_hub.errors, attr, getattr(huggingface_hub.errors, "EntryNotFoundError", Exception))

for attr in ["HfFileSystemResolvedBucketPath", "HfFileSystemResolvedRepositoryPath"]:
    if not hasattr(huggingface_hub.hf_file_system, attr):
        setattr(huggingface_hub.hf_file_system, attr, getattr(huggingface_hub.hf_file_system, "HfFileSystemResolvedPath", Exception))

if not hasattr(huggingface_hub.utils, "httpx"):
    huggingface_hub.utils.httpx = httpx

import datasets
from datasets import load_dataset
import soundfile as sf
import pandas as pd

DATASET_NAME = "ARTPARK-IISc/Vaani"
TARGET_CONFIGS = {
    "hindi": {"config_name": "Hindi", "count": 100, "prefix": "hindi"},
    "tamil": {"config_name": "Tamil", "count": 100, "prefix": "tamil"},
}

CSV_COLUMNS = [
    "clip_id",
    "language",
    "duration",
    "transcript",
    "speakerID",
    "state",
    "district",
    "source_configuration",
    "original_sample_rate",
    "audio_file_path",
]

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "vaani"


def ensure_directories():
    """Create the required folder structure."""
    for lang in ["hindi", "tamil"]:
        audio_dir = DATA_DIR / lang / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = DATA_DIR / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Directory structure confirmed at: %s", DATA_DIR)


def load_existing_metadata(csv_path: Path):
    """Load existing metadata for resumable downloads."""
    if not csv_path.exists():
        return [], set()
    try:
        df = pd.read_csv(csv_path)
        existing_records = df.to_dict("records")
        existing_clip_ids = set(df["clip_id"].dropna().astype(str))
        logger.info("Found %d existing records in %s", len(existing_records), csv_path.name)
        return existing_records, existing_clip_ids
    except Exception as e:
        logger.warning("Could not read existing metadata from %s: %s", csv_path, e)
        return [], set()


def download_language_samples(lang_key: str, config_info: dict, token: str):
    """Download exactly target_count valid audio samples for a language configuration."""
    config_name = config_info["config_name"]
    target_count = config_info["count"]
    prefix = config_info["prefix"]

    audio_dir = DATA_DIR / lang_key / "audio"
    metadata_path = DATA_DIR / "metadata" / f"{lang_key}.csv"

    existing_records, existing_clip_ids = load_existing_metadata(metadata_path)
    current_count = len(existing_records)

    if current_count >= target_count:
        logger.info("[%s] Already has %d/%d samples. Skipping download.", lang_key.upper(), current_count, target_count)
        return existing_records[:target_count]

    logger.info("[%s] Starting streaming from config '%s'. Current count: %d, Target: %d",
                lang_key.upper(), config_name, current_count, target_count)

    # Load in streaming mode to avoid full dataset download
    ds = load_dataset(DATASET_NAME, config_name, split="train", streaming=True, token=token)
    # Cast audio column with decode=False to get raw bytes directly without decoding bottlenecks
    ds = ds.cast_column("audio", datasets.Audio(decode=False))

    records = list(existing_records)
    seen_clip_ids = set(existing_clip_ids)

    stream_iter = iter(ds)
    consecutive_errors = 0

    while len(records) < target_count:
        try:
            sample = next(stream_iter)
            consecutive_errors = 0
        except StopIteration:
            logger.warning("[%s] End of stream reached with %d samples.", lang_key.upper(), len(records))
            break
        except Exception as e:
            consecutive_errors += 1
            logger.error("[%s] Stream read error: %s (consecutive errors: %d)", lang_key.upper(), e, consecutive_errors)
            if consecutive_errors > 20:
                logger.critical("[%s] Too many consecutive streaming errors. Aborting.", lang_key.upper())
                break
            continue

        audio_info = sample.get("audio")
        if not audio_info:
            continue

        raw_bytes = audio_info.get("bytes")
        raw_path = audio_info.get("path") or f"unknown_{len(records)+1}.wav"
        clip_id = Path(raw_path).name

        # Deduplicate
        if clip_id in seen_clip_ids:
            continue

        if not raw_bytes or len(raw_bytes) == 0:
            logger.warning("[%s] Skipping empty audio bytes for clip_id: %s", lang_key.upper(), clip_id)
            continue

        # Validate with soundfile to verify integrity and obtain audio properties
        try:
            with io.BytesIO(raw_bytes) as bio:
                audio_data, sample_rate = sf.read(bio)
                duration = len(audio_data) / sample_rate
        except Exception as sf_err:
            logger.error("[%s] Corrupted audio file for clip_id %s: %s", lang_key.upper(), clip_id, sf_err)
            continue

        # Determine target file name: e.g. hindi_001.wav
        next_idx = len(records) + 1
        target_filename = f"{prefix}_{next_idx:03d}.wav"
        target_path = audio_dir / target_filename

        # Write original audio bytes to disk preserving pristine audio content and rate
        with open(target_path, "wb") as f:
            f.write(raw_bytes)

        # Relative path for portable metadata
        rel_audio_path = f"data/vaani/{lang_key}/audio/{target_filename}"

        # Extract transcript if present
        raw_transcript = sample.get("transcript") or sample.get("transcription") or ""
        if str(raw_transcript).strip().lower() in ["none", "nan", ""]:
            raw_transcript = ""

        # Extract other metadata
        record = {
            "clip_id": clip_id,
            "language": sample.get("language") or config_name,
            "duration": round(duration, 3),
            "transcript": raw_transcript,
            "speakerID": sample.get("speakerID") or "NA",
            "state": sample.get("state") or "NA",
            "district": sample.get("district") or "NA",
            "source_configuration": config_name,
            "original_sample_rate": int(sample_rate),
            "audio_file_path": rel_audio_path,
        }

        records.append(record)
        seen_clip_ids.add(clip_id)

        # Save progress metadata incrementally
        df_progress = pd.DataFrame(records, columns=CSV_COLUMNS)
        df_progress.to_csv(metadata_path, index=False)

        if len(records) % 10 == 0 or len(records) == target_count:
            logger.info("[%s] Progress: %d/%d downloaded (Latest: %s, District: %s, Duration: %.2fs)",
                        lang_key.upper(), len(records), target_count, target_filename, record["district"], duration)

    logger.info("[%s] Completed! Final count: %d/%d", lang_key.upper(), len(records), target_count)
    return records


def main():
    ensure_directories()

    token = huggingface_hub.get_token()
    if not token:
        logger.error("No Hugging Face token found in environment/cache. Please run 'hf auth login' or 'huggingface-cli login'.")
        sys.exit(1)

    logger.info("Authenticated with Hugging Face successfully.")

    for lang_key, config_info in TARGET_CONFIGS.items():
        download_language_samples(lang_key, config_info, token)

    logger.info("All downloads finished. Proceeding to offline verification...")


if __name__ == "__main__":
    main()
