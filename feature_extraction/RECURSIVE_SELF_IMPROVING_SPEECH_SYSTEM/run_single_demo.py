"""Run the analysis pipeline on bundled or user-supplied audio files."""

import argparse
from pathlib import Path
import logging

from pipeline import AudioAnalysisPipeline
from src.config import PipelineConfig
from src.evaluation import EventEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_CLIPS = [
    "data/vaani/hindi/audio/hindi_001.wav",
    "data/vaani/hindi/audio/hindi_002.wav",
    "data/vaani/hindi/audio/hindi_003.wav",
]


def main():
    parser = argparse.ArgumentParser(
        description="Extract acoustic, speech, and sound-event features from WAV/audio files."
    )
    parser.add_argument(
        "audio_files",
        nargs="*",
        help="Audio files to process. If omitted, the first bundled Vaani clips are used.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip visualization generation.",
    )
    args = parser.parse_args()

    config = PipelineConfig(device="cpu")  # CPU by default as requested
    pipeline = AudioAnalysisPipeline(config)

    clips = args.audio_files or DEFAULT_CLIPS

    results = []
    gt_dir = Path("data/ground_truth")
    gt_dir.mkdir(parents=True, exist_ok=True)

    for clip in clips:
        clip_path = Path(clip)
        if not clip_path.exists():
            logger.warning(f"Clip {clip} not found, skipping.")
            continue

        logger.info(f"\n=======================================================")
        logger.info(f"PROCESSING CLIP: {clip_path.name}")
        logger.info(f"=======================================================")
        
        res = pipeline.process_file(clip_path, generate_plots=not args.no_plots)
        results.append(res)

        # Print Word Alignment & Pause summary
        words = res.get("word_alignment", [])
        pauses = res.get("inter_word_pauses", [])
        sig_pauses = [p for p in pauses if p.get("is_significant_pause")]
        units = res.get("caption_units", [])

        logger.info(f"==> Word-Level Alignment for {res['clip_id']}:")
        logger.info(f"    - Total aligned words: {len(words)}")
        logger.info(f"    - Inter-word pauses > 150ms: {len(sig_pauses)}")
        logger.info(f"    - Caption units formed (Rime Stage 1): {len(units)}")
        if words:
            sample_words = words[:6]
            logger.info(f"    - Sample words with timestamps:")
            for sw in sample_words:
                logger.info(f"        {sw['word_id']} '{sw['word']}': {sw['start_time']}s -> {sw['end_time']}s (dur: {sw['duration']}s, conf: {sw['confidence']})")
        logger.info(f"    - Full Word Alignment Markdown Report: {res.get('alignment_report_path')}")

        # Generate Stage 9 Human Ground-Truth verification template
        gt_template_path = gt_dir / f"{res['clip_id']}_ground_truth.json"
        EventEvaluator.create_verification_template(res["json_path"], gt_template_path)
        logger.info(f"Ground truth verification template generated: {gt_template_path}")

        logger.info(
            "Scoring skipped: the generated ground-truth template is unverified "
            "until a researcher audits it."
        )

    logger.info("\nAll target demo clips processed successfully!")


if __name__ == "__main__":
    main()
