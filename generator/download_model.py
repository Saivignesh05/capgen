
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_REPO = "ai4bharat/indic-parler-tts"

MODEL_DIR = BASE_DIR / "models" / "indic-parler-tts"

DESCRIPTION_TOKENIZER_DIR = (
    BASE_DIR / "models" / "description-tokenizer"
)


# --------------------------------------------------
# Download main model
# --------------------------------------------------

def download_main_model():

    print("\n" + "=" * 60)
    print("Downloading Indic-Parler-TTS")
    print("=" * 60 + "\n")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=str(MODEL_DIR),
        token=True
    )

    print("\nMain model downloaded successfully.")


# --------------------------------------------------
# Read model configuration
# --------------------------------------------------

def get_description_tokenizer_name():

    config_path = MODEL_DIR / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Model config not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    try:
        tokenizer_name = config["text_encoder"]["_name_or_path"]
    except KeyError as exc:
        raise KeyError(
            "Could not find text_encoder._name_or_path "
            "in the model config.json"
        ) from exc

    print("\nDescription tokenizer:", tokenizer_name)

    return tokenizer_name


# --------------------------------------------------
# Download description tokenizer
# --------------------------------------------------

def download_description_tokenizer():

    print("\n" + "=" * 60)
    print("Downloading description tokenizer")
    print("=" * 60 + "\n")

    tokenizer_name = get_description_tokenizer_name()

    DESCRIPTION_TOKENIZER_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=str(DESCRIPTION_TOKENIZER_DIR),
    )

    print("\nDescription tokenizer downloaded successfully.")


# --------------------------------------------------
# Verify downloaded files
# --------------------------------------------------

def verify_downloads():

    print("\n" + "=" * 60)
    print("Verifying downloaded files")
    print("=" * 60)

    required_files = [
        MODEL_DIR / "config.json",
        MODEL_DIR / "model.safetensors",
        MODEL_DIR / "generation_config.json",
    ]

    all_present = True

    for file_path in required_files:

        if file_path.exists():
            print(f"[OK] {file_path.name}")
        else:
            print(f"[MISSING] {file_path.name}")
            all_present = False

    if not all_present:
        raise RuntimeError(
            "Some required model files are missing."
        )

    print("\nAll required model files are present.")


# --------------------------------------------------
# Main
# --------------------------------------------------

def download_model():

    download_main_model()

    download_description_tokenizer()

    verify_downloads()

    print("\n" + "=" * 60)
    print("DOWNLOAD PROCESS COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    download_model()

