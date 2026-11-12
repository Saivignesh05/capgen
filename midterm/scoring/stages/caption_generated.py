"""Caption generated samples (for instruction relevance): the round 0 captioner, Qwen3-Omni, describes
the first `relevance.samples_per_item` samples of each item. The judge stage then checks the
generator's instruction against these captions (the captioner as an automatic judge, as Rime suggests).

Uses the round 0 captioner code and the round 0 run's own settings.yaml, so it is the same model,
revision and options as the system being scored. Needs ~70 GB of GPU memory.

Output caption_generated.jsonl, one line per sample: clip_id, sample, language, caption
"""
import os

from scoring.common import append_jsonl, read_jsonl, record_error
from scoring.stage import main

NAME = "caption_generated"


def run(settings, items, scores_dir, run_folder, logger):
    from pipeline.audio_types import AudioClip     # round0/code, on PYTHONPATH (scoring.run)
    from pipeline.models.captioner import Captioner
    from pipeline.settings import captioner_settings, load_settings as load_round0_settings

    output = os.path.join(scores_dir, f"{NAME}.jsonl")
    done_keys = {(r["clip_id"], r["sample"]) for r in read_jsonl(output)}
    per_item = settings.relevance.samples_per_item
    todo = [(i, s) for i in items if i.description for s in i.samples[:per_item]
            if (i.clip_id, s.sample) not in done_keys]
    logger.info("%d samples to caption (%d already done, %d per item)", len(todo), len(done_keys), per_item)
    if not todo:
        return 0, 0

    round0_settings = load_round0_settings(os.path.join(run_folder, "settings.yaml"))
    captioner = Captioner(captioner_settings(round0_settings))
    logger.info("captioner loaded: %s@%s", round0_settings.downloads.captioner.repo_id, round0_settings.downloads.captioner.revision)

    done = failed = 0
    for number, (item, sample) in enumerate(todo, 1):
        key = f"{item.clip_id}#{sample.sample}"
        try:
            caption = captioner.describe(AudioClip(path=sample.path)).text
            if not caption:
                raise ValueError("empty caption")
            append_jsonl(output, {"clip_id": item.clip_id, "sample": sample.sample, "language": item.language, "caption": caption})
            done += 1
            logger.info("[%d/%d] %s: %d words", number, len(todo), key, len(caption.split()))
        except Exception as error:
            failed += 1
            record_error(scores_dir, NAME, key, error, logger)
    return done, failed


if __name__ == "__main__":
    main(NAME, run)
